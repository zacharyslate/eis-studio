"""Series analysis: several measurements exported together (or uploaded together).

Typical use: impedance spectra of one pellet at several temperatures, followed by DC
polarisation (chronoamperometry) steps at several potentials, all in one export.

Reading
-------
A file is scanned for *every* numeric table (not only the largest), each with its own header.
Inside a table, rows are classified:

* impedance rows: a frequency column with f > 0 and a real part (BioLogic writes f = 0 on rows
  of non-impedance techniques in linked experiments);
* time-series rows: time and current present.

Contiguous runs are then split into individual measurements wherever a technique/sequence
index changes (BioLogic ``Ns``, ``cycle number``), the frequency jumps back up by more than a
decade (a new sweep), the time resets, or (when no index column exists) the potential steps.
Every decision is recorded in ``SeriesReport.notes``.

Analysis
--------
* Ionic resistance per spectrum: equivalent-circuit fit (sum of chosen resistances), Z' at the
  minimum of -Z'' (arc/spike junction), or the blocking spike extrapolated to -Z'' = 0.
* sigma = L/(R A) per spectrum; Arrhenius fit of log10(sigma T) (or log10 sigma) vs 1000/T.
* DC polarisation with ion-blocking electrodes: steady-state current of each potential step
  (mean over the last part of the step), linear fit I = E/R_e (+ I_0), sigma_e = L/(R_e A).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import conductivity as cond
from . import ecm as ecmm
from .io import _classify, _decode, _excel_to_text, _is_negated, _split, _to_float

K_B_EV = 8.617333262e-5          # Boltzmann constant, eV/K
EV_TO_KJ_MOL = 96.485332


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Measurement:
    kind: str                     # "eis" | "ca" (potentiostatic step) | "other" (time series, E not constant)
    name: str
    source: str
    index: int = 0
    # impedance
    freq: np.ndarray | None = None
    Z: np.ndarray | None = None
    # time series (SI units: s, A, V)
    t: np.ndarray | None = None
    I: np.ndarray | None = None
    E: np.ndarray | None = None
    notes: list = field(default_factory=list)

    @property
    def n(self):
        return int((self.freq if self.kind == "eis" else self.t).size)

    def summary(self):
        if self.kind == "eis":
            return f"{self.n} pts, {self.freq.min():.3g}–{self.freq.max():.3g} Hz"
        dur = float(self.t[-1] - self.t[0]) if self.n else 0.0
        e = f", E ≈ {np.nanmean(self.E):.4g} V" if self.E is not None else ""
        return f"{self.n} pts, {dur:.4g} s{e}"


@dataclass
class SeriesReport:
    notes: list = field(default_factory=list)
    tables: list = field(default_factory=list)   # one dict per table found: file, header, roles


# --------------------------------------------------------------------------- #
# Column recognition for time series
# --------------------------------------------------------------------------- #
_TIME = re.compile(r"^(corrected\s+|elapsed\s+)?time\b|^t\s*([/(\[]|$)", re.I)
_CURR = re.compile(r"<\s*i\s*>|^i\s*([/(\[]|$)|current|^im$|^idc\b|\.current|^i_?(meas|we)", re.I)
_POT_PREF = [re.compile(p, re.I) for p in (r"<?\s*ewe\s*>?", r"ecell", r"^vf$", r"\bwe\(\d\)\.potential",
                                            r"^e\s*([/(\[]|$)|^ewe", r"potential(?!.*applied)", r"voltage",
                                            r"^v\s*([/(\[]|$)", r"control\s*/\s*m?v", r"applied")]
_STEP = re.compile(r"^ns$|^step$|step\s*(no|number|index)|^sequence|^cycle\s*(number|no)?$|^loop", re.I)

_CURRENT_UNITS = {"pa": 1e-12, "na": 1e-9, "ua": 1e-6, "µa": 1e-6, "μa": 1e-6, "ma": 1e-3, "a": 1.0}
_POT_UNITS = {"mv": 1e-3, "v": 1.0}
_TIME_UNITS = {"ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0}


def _unit(header: str, table: dict, default):
    """Unit factor from '/mA', '(mA)', '[mA]' or a trailing unit token."""
    h = header.strip().lower().replace(" ", "")
    m = re.search(r"[/(\[]([a-zµμ]+)[)\]]?$", h)
    if m and m.group(1) in table:
        return table[m.group(1)]
    return default


def _classify_ts(headers):
    roles = {}
    for i, h in enumerate(headers):
        hs = str(h).strip()
        if not hs:
            continue
        if "time" not in roles and _TIME.search(hs):
            roles["time"] = i
        elif "current" not in roles and _CURR.search(hs) and "control" not in hs.lower():
            roles["current"] = i
        elif _STEP.search(hs):
            roles.setdefault("steps", []).append(i)
    # potential: best-ranked pattern wins (measured working-electrode potential before set-points)
    for pat in _POT_PREF:
        for i, h in enumerate(headers):
            if i in roles.values() or not str(h).strip():
                continue
            if pat.search(str(h).strip()) and "current" not in str(h).lower():
                roles["potential"] = i
                break
        if "potential" in roles:
            break
    return roles


# --------------------------------------------------------------------------- #
# Table discovery (all blocks, each with its own header)
# --------------------------------------------------------------------------- #
def _blocks(text):
    """All numeric blocks for the best (delimiter, decimal) reading.

    Returns (blocks, delim, decimal_comma) with blocks = [(start_line, rows)]."""
    lines = text.splitlines()
    best = None
    for delim in ("\t", ";", ",", None):
        for dc in (False, True):
            if dc and delim == ",":
                continue
            blocks, start, run = [], None, []
            for idx, ln in enumerate(lines):
                toks = _split(ln, delim) if ln.strip() else []
                nonempty = [t for t in toks if t != ""]
                nnum = sum(_to_float(t, dc) is not None for t in nonempty)
                if nnum >= 3 and nnum >= 0.6 * len(nonempty):
                    if start is None:
                        start = idx
                    run.append(toks)
                else:
                    if run:
                        blocks.append((start, run))
                    start, run = None, []
            if run:
                blocks.append((start, run))
            blocks = [b for b in blocks if len(b[1]) >= 5]
            score = sum(len(r) * min(int(np.median([len(x) for x in r])), 12) for _, r in blocks)
            if best is None or score > best[0]:
                best = (score, blocks, delim, dc)
    if best is None or not best[1]:
        return [], None, False
    return best[1], best[2], best[3]


def _header_for(lines, start, delim, dc, ncol):
    """Pick the text row above a block that names the most columns (skips unit rows)."""
    best, best_n = None, 0
    for back in range(1, 7):
        j = start - back
        if j < 0:
            break
        toks = _split(lines[j], delim)
        if len(toks) < 2 or sum(_to_float(t, dc) is not None for t in toks if t) > 0:
            if best is not None:
                break
            continue
        n = len(_classify(toks)) + len([k for k in _classify_ts(toks) if k != "steps"])
        if n > best_n:
            best, best_n = toks, n
    return best


def _to_array(rows, ncol, dc):
    out = np.full((len(rows), ncol), np.nan)
    for i, r in enumerate(rows):
        for j in range(min(ncol, len(r))):
            v = _to_float(r[j], dc)
            if v is not None:
                out[i, j] = v
    return out


# --------------------------------------------------------------------------- #
# Splitting a table into measurements
# --------------------------------------------------------------------------- #
def _runs(mask):
    """(start, stop) index pairs of True runs."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(idx[::2], idx[1::2]))


def _cut_points(n, *conditions):
    cuts = set()
    for c in conditions:
        if c is not None:
            cuts.update((np.flatnonzero(c) + 1).tolist())
    return sorted(c for c in cuts if 0 < c < n)


def _segments(a, b, cuts):
    edges = [a] + [c for c in cuts if a < c < b] + [b]
    return list(zip(edges[:-1], edges[1:]))


def _parse_table(arr, header, fname, report, start_idx):
    if not header and arr.shape[1] >= 3:
        f0 = arr[:, 0]
        ok = np.isfinite(f0) & (f0 > 0)
        if ok.mean() > 0.9 and np.log10(np.nanmax(f0[ok]) / np.nanmin(f0[ok])) > 1:
            header = ["Frequency", "Z'", "Z''"] + [f"col {i}" for i in range(3, arr.shape[1])]
            report.notes.append(f"{fname}: no header; columns read as f, Z′, Z″ (sign of Z″ detected from the data).")
    roles = _classify(header) if header else {}
    ts = _classify_ts(header) if header else {}
    report.tables.append({"file": fname, "rows": int(arr.shape[0]),
                          "columns": {k: (header[v] if isinstance(v, int) else [header[i] for i in v])
                                      for k, v in {**roles, **ts}.items()} if header else {}})
    out = []
    n = arr.shape[0]
    col = lambda i: arr[:, i] if i is not None and i < arr.shape[1] else None  # noqa: E731

    # --- impedance rows
    f = col(roles.get("freq"))
    Z = None
    if f is not None and ("re" in roles and "im" in roles):
        re_, im_ = col(roles["re"]), col(roles["im"])
        sign = -1.0 if _is_negated(str(header[roles["im"]])) else 1.0
        Z = re_ + 1j * sign * im_
    elif f is not None and "mod" in roles and "phase" in roles:
        mod, ph = col(roles["mod"]), col(roles["phase"])
        ph = np.deg2rad(ph) if np.nanmax(np.abs(ph)) > 2 * np.pi + 0.1 else ph
        if _is_negated(str(header[roles["phase"]])):
            ph = -ph
        Z = mod * np.exp(1j * ph)
    if Z is not None:
        is_eis = np.isfinite(f) & (f > 0) & np.isfinite(Z.real) & np.isfinite(Z.imag) & (np.abs(Z) > 0)
    else:
        is_eis = np.zeros(n, bool)

    # --- time-series rows
    t, I_ = col(ts.get("time")), col(ts.get("current"))
    E = col(ts.get("potential"))
    if t is not None and I_ is not None:
        t = t * _unit(header[ts["time"]], _TIME_UNITS, 1.0)
        I_ = I_ * _unit(header[ts["current"]], _CURRENT_UNITS, 1.0)
        if E is not None:
            E = E * _unit(header[ts["potential"]], _POT_UNITS, 1.0)
        is_ts = ~is_eis & np.isfinite(t) & np.isfinite(I_)
    else:
        is_ts = np.zeros(n, bool)

    if not header:
        report.notes.append(f"{fname}: a table without a recognisable header was skipped.")
        return out

    # change of sequence / cycle index
    step_change = None
    for si in ts.get("steps", []):
        s = arr[:, si]
        ch = np.r_[np.diff(np.nan_to_num(s, nan=-1)) != 0]
        step_change = ch if step_change is None else (step_change | ch)

    k = start_idx
    for a, b in _runs(is_eis):
        lf = np.log10(f[a:b])
        jump = np.diff(lf)
        direction = np.sign(np.median(jump)) if jump.size else -1.0
        cuts = _cut_points(b - a, jump * (direction or -1.0) < -1.0,
                           step_change[a:b - 1] if step_change is not None else None)
        for s0, s1 in _segments(0, b - a, cuts):
            if s1 - s0 < 5:
                continue
            fs, Zs = f[a + s0:a + s1], Z[a + s0:a + s1]
            m = Measurement("eis", "", fname, k)
            if sign_flip_needed(header, roles, Zs):
                Zs = Zs.real - 1j * Zs.imag
                m.notes.append("Most imaginary values were positive, so the column was read as −Z″.")
            order = np.argsort(fs)[::-1]
            fs, Zs = fs[order], Zs[order]
            _, uniq = np.unique(fs, return_index=True)
            keep = np.sort(uniq)
            fs, Zs = fs[keep], Zs[keep]
            order = np.argsort(fs)[::-1]
            m.freq, m.Z = fs[order], Zs[order]
            out.append(m)
            k += 1

    for a, b in _runs(is_ts):
        tt = t[a:b]
        reset = np.diff(tt) < 0
        e_step = None
        if step_change is None and E is not None:
            dE = np.abs(np.diff(E[a:b]))
            noise = np.nanmedian(dE) if dE.size else 0.0
            e_step = dE > max(2e-3, 20 * noise)
        cuts = _cut_points(b - a, reset, step_change[a:b - 1] if step_change is not None else None, e_step)
        for s0, s1 in _segments(0, b - a, cuts):
            if s1 - s0 < 5:
                continue
            sl = slice(a + s0, a + s1)
            m = Measurement("ca", "", fname, k)
            m.t = t[sl] - t[sl][0]
            m.I = I_[sl]
            m.E = E[sl] if E is not None else None
            if m.E is not None:
                spread = np.nanpercentile(m.E, 95) - np.nanpercentile(m.E, 5)
                if spread > max(5e-3, 0.02 * abs(np.nanmedian(m.E))):
                    m.kind = "other"
                    m.notes.append(f"Potential not constant (spread {spread * 1e3:.3g} mV): not a "
                                   "potentiostatic step, excluded from the DC analysis by default.")
            if np.nanmax(np.abs(m.I)) == 0:
                m.kind = "other"
                m.notes.append("Current is zero throughout (open circuit?).")
            out.append(m)
            k += 1
    return out


def sign_flip_needed(header, roles, Z):
    if "im" in roles and _is_negated(str(header[roles["im"]])):
        return False                       # already converted from the header
    if "phase" in roles and "im" not in roles:
        return False
    return bool(np.median(Z.imag) > 0)


def load_series(files):
    """files: list of (raw_bytes, filename). Returns (measurements, SeriesReport)."""
    report = SeriesReport()
    out = []
    for raw, fname in files:
        ext = os.path.splitext(fname)[1].lower()
        try:
            text = _excel_to_text(raw) if ext in (".xlsx", ".xls") else _decode(raw)
        except Exception as exc:  # noqa: BLE001
            report.notes.append(f"{fname}: could not be read ({exc}).")
            continue
        lines = text.splitlines()
        blocks, delim, dc = _blocks(text)
        if not blocks:
            report.notes.append(f"{fname}: no numeric table found.")
            continue
        if dc:
            report.notes.append(f"{fname}: decimal commas detected and converted.")
        for start, rows in blocks:
            ncol = int(np.median([len(r) for r in rows]))
            header = _header_for(lines, start, delim, dc, ncol)
            arr = _to_array(rows, ncol, dc)
            got = _parse_table(arr, header, fname, report, len(out))
            out.extend(got)
    # names
    ne = nc = 0
    for m in out:
        if m.kind == "eis":
            ne += 1
            m.name = f"EIS {ne}"
        else:
            nc += 1
            m.name = f"DC {nc}" if m.kind == "ca" else f"Other {nc}"
        m.index = out.index(m)
    n_eis = sum(m.kind == "eis" for m in out)
    n_ca = sum(m.kind == "ca" for m in out)
    report.notes.insert(0, f"Found {n_eis} impedance spectra and {n_ca} potentiostatic steps"
                           + (f" (+{len(out) - n_eis - n_ca} other time series)" if len(out) > n_eis + n_ca else "")
                           + f" in {len(files)} file(s).")
    return out, report


# --------------------------------------------------------------------------- #
# Ionic resistance from one spectrum
# --------------------------------------------------------------------------- #
R_METHODS = {
    "Equivalent-circuit fit": "ecm",
    "Z′ at the minimum of −Z″ (arc/spike junction)": "min",
    "Blocking spike extrapolated to −Z″ = 0": "spike",
}


def _running_median(y, w=5):
    h = w // 2
    yp = np.r_[np.full(h, y[0]), y, np.full(h, y[-1])]
    return np.array([np.median(yp[i:i + w]) for i in range(y.size)])


def tail_start(freq, Z):
    """Index (in high->low f order) of the junction between the arcs and a capacitive low-frequency
    tail, or None when the spectrum has no such tail.

    -Z'' is smoothed with a 5-point running median, so one noisy point (mains pickup at 50/60 Hz,
    for example) does not create a false minimum; the junction is the smoothed minimum, moved to the
    lowest measured point within two points of it. A tail exists when -Z'' then rises to at least
    twice that minimum and the last point has -phase above 30 deg."""
    mz = -Z.imag
    n = mz.size
    if n < 6:
        return None
    sm = _running_median(mz, 5)
    lo = int(np.argmin(sm[:n - 2]))
    # the raw minimum next to the smoothed one (keeps the junction on a measured point)
    a, b = max(lo - 2, 0), min(lo + 3, n - 2)
    i = a + int(np.argmin(mz[a:b]))
    phase_end = -np.degrees(np.angle(Z[-1]))
    if i >= n - 3 or phase_end < 30 or sm[-1] < 2 * max(sm[i], 1e-12 * np.abs(Z).max()):
        return None
    return i


def r_minimum(freq, Z):
    """Z' at the minimum of -Z'' between the arcs and the tail: quadratic fit of the smoothed -Z''
    against Z' over the five points around the junction."""
    i = tail_start(freq, Z)
    if i is None:
        raise ValueError("No low-frequency spike found (−Z″ does not rise at low frequency).")
    sm = _running_median(-Z.imag, 5)
    a, b = max(i - 2, 0), min(i + 3, Z.size)
    x, y = Z.real[a:b], sm[a:b]
    if x.size >= 3 and np.ptp(x) > 0:
        c = np.polyfit(x - x.mean(), y, 2)
        if c[0] > 0:
            xv = -c[1] / (2 * c[0]) + x.mean()
            if x.min() <= xv <= x.max():
                return float(xv), float(np.ptp(x) / 4)
    return float(Z.real[i]), float(abs(Z.real[min(i + 1, Z.size - 1)] - Z.real[max(i - 1, 0)]) / 2)


def r_spike(freq, Z, n_min=4):
    """Real-axis intercept of a straight-line fit to the low-frequency spike (-Z'' vs Z')."""
    i = tail_start(freq, Z)
    if i is None:
        raise ValueError("No low-frequency spike found (−Z″ does not rise at low frequency).")
    x, y = Z.real[i:], -Z.imag[i:]
    if x.size < n_min:
        raise ValueError(f"The spike has only {x.size} points; need at least {n_min}.")
    # use the straighter low-frequency part (drop the curved first third near the junction)
    k = max(0, int(0.33 * x.size)) if x.size >= 3 * n_min else 0
    x, y = x[k:], y[k:]
    A = np.c_[x, np.ones_like(x)]
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    m, c = coef
    if m <= 0:
        raise ValueError("The spike does not rise with Z′, so it cannot be extrapolated.")
    x0 = -c / m
    # standard error of the intercept from the regression
    dof = max(x.size - 2, 1)
    s2 = float(np.sum((y - A @ coef) ** 2) / dof)
    cov = s2 * np.linalg.inv(A.T @ A)
    dx0 = float(np.sqrt(cov[1, 1] / m ** 2 + cov[0, 0] * c ** 2 / m ** 4 - 2 * cov[0, 1] * c / m ** 3))
    angle = float(np.degrees(np.arctan(m)))
    return float(x0), dx0, angle


@dataclass
class SpectrumResult:
    name: str
    T_C: float | None
    method: str
    R: float | None = None
    dR: float = 0.0
    fit: object = None                # ECMResult for the ECM method
    drt: object = None                # DRTResult (optional)
    sigma: dict | None = None
    notes: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def analyse_spectrum(m: Measurement, T_C, method, geom=None, circuit=None, r_names=None, n_starts=4,
                     time_budget_s=20.0, f_window=None):
    freq, Z = m.freq, m.Z
    if f_window:
        keep = (freq >= f_window[0]) & (freq <= f_window[1])
        freq, Z = freq[keep], Z[keep]
    res = SpectrumResult(m.name, T_C, method)
    try:
        if method == "ecm":
            names, _ = ecmm.parse_circuit(circuit)
            g = ecmm.default_guess(names, freq, Z, circuit)
            fit = ecmm.fit_ecm(freq, Z, circuit, dict(zip(names, g)), {}, ecmm.default_bounds(circuit),
                               "modulus", False, n_starts, 0, time_budget_s)
            res.fit = fit
            res.notes += fit.warnings
            all_r = [n for n, *_ in fit.all_params() if ecmm.element_type(n) == "R"]
            use = [n for n in (r_names or all_r) if n in all_r]
            if not use:
                raise ValueError(f"None of the chosen resistances ({', '.join(r_names or [])}) is in {circuit}.")
            res.R = float(sum(fit.param(n)[0] for n in use))
            res.dR = float(np.sqrt(sum(fit.param(n)[1] ** 2 for n in use)))
            res.extra = {"resistances": "+".join(use), "fit_max_res_pct": fit.metrics.get("max_abs_rel_res_pct")}
        elif method == "min":
            res.R, res.dR = r_minimum(freq, Z)
        elif method == "spike":
            res.R, res.dR, ang = r_spike(freq, Z)
            res.extra = {"spike_angle_deg": ang}
            if ang < 60:
                res.notes.append(f"The spike is inclined at {ang:.0f}° to the real axis (far from 90°); its "
                                 "extrapolation carries a large model error.")
    except Exception as exc:  # noqa: BLE001 - report per spectrum, keep the series going
        res.notes.append(f"{type(exc).__name__}: {exc}")
        res.R = None
    if res.R is not None and geom is not None:
        res.sigma = cond.conductivity(res.R, geom, res.dR)
    return res


# --------------------------------------------------------------------------- #
# Arrhenius
# --------------------------------------------------------------------------- #
def arrhenius(T_C, sigma, times_T=True):
    """Fit log10(sigma*T) (or log10 sigma) against 1000/T.

    Returns dict with Ea (eV, kJ/mol) and its standard error, prefactor, R^2, and the
    fitted line. sigma in S/cm, T in deg C."""
    T_C = np.asarray(T_C, float)
    s = np.asarray(sigma, float)
    ok = np.isfinite(T_C) & np.isfinite(s) & (s > 0)
    T_C, s = T_C[ok], s[ok]
    if T_C.size < 2:
        raise ValueError("Need at least two spectra with a temperature and a conductivity.")
    if np.unique(T_C).size < 2:
        raise ValueError("All temperatures are equal.")
    T = T_C + 273.15
    x = 1000.0 / T
    y = np.log10(s * T) if times_T else np.log10(s)
    A = np.c_[x, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    m, c = coef
    yhat = A @ coef
    n = x.size
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    if n > 2:
        s2 = ss_res / (n - 2)
        cov = s2 * np.linalg.inv(A.T @ A)
        dm = float(np.sqrt(cov[0, 0]))
    else:
        dm = float("nan")
    k = 1000.0 * K_B_EV * np.log(10)
    Ea = -m * k
    dEa = dm * k
    out = {"Ea_eV": float(Ea), "dEa_eV": float(dEa), "Ea_kJ_mol": float(Ea * EV_TO_KJ_MOL),
           "dEa_kJ_mol": float(dEa * EV_TO_KJ_MOL), "slope": float(m), "intercept": float(c),
           "log10_prefactor": float(c), "r2": float(r2), "n": int(n), "times_T": bool(times_T),
           "x": x, "y": y, "T_C": T_C}
    # conductivity at 25 deg C from the fit (flag extrapolation)
    T25 = 298.15
    y25 = m * 1000 / T25 + c
    out["sigma25_S_cm"] = float(10 ** y25 / T25 if times_T else 10 ** y25)
    out["sigma25_extrapolated"] = bool(not (T_C.min() - 1e-9 <= 25.0 <= T_C.max() + 1e-9))
    return out


# --------------------------------------------------------------------------- #
# DC polarisation (electronic conductivity)
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    name: str
    E: float
    I_ss: float
    dI: float
    drift_pct: float
    t_window: tuple
    notes: list = field(default_factory=list)


def steady_state(m: Measurement, tail_frac=0.1, E_override=None, min_pts=5):
    """Mean current over the last ``tail_frac`` of the step; drift = change over that window
    (from a linear fit) relative to the mean, in %."""
    t, I_ = m.t, m.I
    n = t.size
    t0 = t[-1] - tail_frac * (t[-1] - t[0])
    sel = t >= t0
    if sel.sum() < min_pts:
        sel = np.zeros(n, bool)
        sel[-min(min_pts, n):] = True
    ts, Is = t[sel], I_[sel]
    I_ss = float(np.mean(Is))
    dI = float(np.std(Is, ddof=1) / np.sqrt(Is.size)) if Is.size > 1 else 0.0
    slope = np.polyfit(ts - ts[0], Is, 1)[0] if Is.size > 2 and np.ptp(ts) > 0 else 0.0
    drift = float(abs(slope * np.ptp(ts)) / abs(I_ss) * 100) if I_ss else float("inf")
    if E_override is not None and np.isfinite(E_override):
        E = float(E_override)
    elif m.E is not None:
        E = float(np.nanmean(m.E[sel]))
    else:
        E = float("nan")
    r = StepResult(m.name, E, I_ss, dI, drift, (float(ts[0]), float(ts[-1])))
    if drift > 5:
        r.notes.append(f"Current still changes by {drift:.1f} % over the averaging window: not at steady state; "
                       "the electronic conductivity will be overestimated.")
    return r


def dc_fit(E, I_ss, through_origin=False):
    """Linear fit I = G E (+ I0). Returns dict with R_e = 1/G and its standard error."""
    E = np.asarray(E, float)
    I_ = np.asarray(I_ss, float)
    ok = np.isfinite(E) & np.isfinite(I_)
    E, I_ = E[ok], I_[ok]
    need = 1 if through_origin else 2
    if E.size < need or (not through_origin and np.unique(E).size < 2):
        raise ValueError("Need at least two steps at different potentials (or one, forced through the origin).")
    A = E[:, None] if through_origin else np.c_[E, np.ones_like(E)]
    coef, *_ = np.linalg.lstsq(A, I_, rcond=None)
    yhat = A @ coef
    dof = E.size - A.shape[1]
    G = float(coef[0])
    I0 = 0.0 if through_origin else float(coef[1])
    if dof > 0:
        s2 = float(np.sum((I_ - yhat) ** 2) / dof)
        dG = float(np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0]))
    else:
        dG = float("nan")
    ss_tot = float(np.sum((I_ - I_.mean()) ** 2))
    r2 = 1 - float(np.sum((I_ - yhat) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    if G <= 0:
        raise ValueError("The steady-state current does not increase with potential; no electronic resistance "
                         "can be extracted.")
    R_e = 1.0 / G
    dR = dG / G ** 2 if np.isfinite(dG) else 0.0
    return {"G_S": G, "dG_S": dG, "I0_A": I0, "R_e_ohm": R_e, "dR_e_ohm": float(dR), "r2": float(r2),
            "n": int(E.size), "through_origin": bool(through_origin), "E": E, "I": I_}


# --------------------------------------------------------------------------- #
# Per-sample geometry, relative density and plot normalisation
# --------------------------------------------------------------------------- #
NORM_MODES = {
    "None (Ω)": "none",
    "Geometry: Z·A/L (Ω cm)": "geom",
    "Geometry and density: Z·(A/L)·ρ_rel^m (Ω cm, dense-equivalent)": "dense",
    "Area: Z·A (Ω cm²)": "area",
    "Shape: Z/R (each spectrum by its own resistance)": "shape",
}


def sample_geometry(base: cond.Geometry | None, thickness_mm=None, diameter_mm=None):
    """Per-sample disc geometry when thickness/diameter are given, else the shared one."""
    t = None if thickness_mm is None or not np.isfinite(thickness_mm) or thickness_mm <= 0 else float(thickness_mm)
    d = None if diameter_mm is None or not np.isfinite(diameter_mm) or diameter_mm <= 0 else float(diameter_mm)
    if base is None and (t is None or d is None):
        return None
    if t is None and d is None:
        return base
    L_cm = t / 10 if t is not None else base.thickness_cm
    A_cm2 = np.pi * (d / 10) ** 2 / 4 if d is not None else base.area_cm2
    return cond.Geometry(L_cm, A_cm2, base.d_thickness_cm if base and t is None else 0.0,
                         base.d_area_cm2 if base and d is None else 0.0,
                         f"L = {L_cm * 10:g} mm, A = {A_cm2:.4g} cm²")


def relative_density(geom: cond.Geometry | None, mass_mg, rho_theory_g_cm3):
    """(geometric density g/cm3, relative density) or (None, None)."""
    if geom is None or mass_mg is None or not np.isfinite(mass_mg) or mass_mg <= 0:
        return None, None
    rho = mass_mg / 1000 / (geom.thickness_cm * geom.area_cm2)
    if rho_theory_g_cm3 and rho_theory_g_cm3 > 0:
        return float(rho), float(rho / rho_theory_g_cm3)
    return float(rho), None


def norm_factor(mode, geom=None, rho_rel=None, m_exp=1.5, R=None):
    """(factor, unit) so that the plotted impedance is factor * Z. None if not computable."""
    if mode == "none":
        return 1.0, "Ω"
    if mode == "geom":
        return (geom.area_cm2 / geom.thickness_cm, "Ω cm") if geom else None
    if mode == "dense":
        if geom is None or rho_rel is None:
            return None
        return geom.area_cm2 / geom.thickness_cm * rho_rel ** m_exp, "Ω cm"
    if mode == "area":
        return (geom.area_cm2, "Ω cm²") if geom else None
    if mode == "shape":
        return (1.0 / R, "R") if R else None
    raise ValueError(mode)


_T_IN_NAME = re.compile(r"(?<![A-Za-z0-9.])(-?\d+(?:\.\d+)?)\s*(°\s*C|ºC|degC|C|K)(?![A-Za-z])")


def temperature_from_name(name):
    """Temperature in deg C read from a file name such as 'pellet_25C.csv', 'x_80 °C', 'T_353K'.
    None unless exactly one temperature-like token is present."""
    stem = os.path.splitext(os.path.basename(name))[0]
    hits = {(float(v), u.replace(" ", "")) for v, u in _T_IN_NAME.findall(stem)}
    if len(hits) != 1:
        return None
    v, u = hits.pop()
    T = round(v - 273.15, 2) if u == "K" else v
    return T if -273.15 < T < 2000 else None

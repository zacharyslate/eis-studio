"""Equivalent-circuit fitting on top of impedance.py.

We call ``impedance.models.circuits.fitting.circuit_fit`` directly (the same
routine ``CustomCircuit.fit`` uses) so that fixed parameters, bounds and the
weighting scheme are fully controlled by the user.

Circuit syntax is impedance.py's: ``-`` = series, ``p(a,b)`` = parallel,
elements numbered, e.g. ``R0-p(R1,CPE1)-p(R2,CPE2)-CPE3``.
Docs: https://impedancepy.readthedocs.io/en/latest/circuit-elements.html
"""

from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from impedance.models.circuits import fitting
from scipy.optimize import curve_fit
from impedance.models.circuits.elements import circuit_elements

WEIGHTINGS = {
    "Modulus (1/|Z|)": "modulus",
    "Proportional (1/|Z'|, 1/|Z''|)": "proportional",
    "Unit (unweighted)": "unit",
}

# sensible magnitudes for each parameter slot; refined from data in default_guess()
_SLOT_DEFAULTS = {
    "R": [None], "C": [1e-6], "L": [1e-7], "W": [100.0],
    "CPE": [1e-6, 0.85], "La": [1e-7, 0.9], "Wo": [None, 1.0], "Ws": [None, 1.0],
    "G": [None, 1.0], "Gs": [None, 1.0, 0.5], "TLMQ": [None, 1e-6, 0.85],
    "K": [None, 1.0], "Zarc": [None, 1e-3, 0.85], "T": [1.0, 1.0, 1.0, 1.0],
}


def element_type(name: str) -> str:
    """'CPE1_0' -> 'CPE', 'R12' -> 'R'."""
    base = name.split("_")[0]
    return re.match(r"[A-Za-z]+", base).group(0)


def parse_circuit(circuit: str):
    """Validate a circuit string and return (param_names, units).

    Raises ValueError with a readable message if the string is malformed.
    """
    circuit = circuit.replace(" ", "")
    if not circuit:
        raise ValueError("Circuit string is empty.")
    if circuit.count("(") != circuit.count(")"):
        raise ValueError("Unbalanced parentheses in circuit string.")
    elems = fitting.extract_circuit_elements(circuit)
    names, units = [], []
    for e in elems:
        et = re.match(r"[A-Za-z]+", e)
        if et is None or et.group(0) not in circuit_elements or et.group(0) in ("p", "s"):
            raise ValueError(f"Unknown element '{e}'. Available: "
                             f"{', '.join(k for k in circuit_elements if k not in ('p', 's'))}")
        fn = circuit_elements[et.group(0)]
        if fn.num_params == 1:
            names.append(e)
            units.append(fn.units[0])
        else:
            for j in range(fn.num_params):
                names.append(f"{e}_{j}")
                units.append(fn.units[j])
    if len(set(elems)) != len(elems):
        raise ValueError("Element names must be unique (e.g. R0, R1, R2).")
    return names, units


def _arc_frequencies(freq, Z, n, lf_hi, lf_lo):
    """Starting characteristic frequencies for n parallel R-C/CPE pairs, high to low.

    Uses the apexes of -Z'' (local maxima, strongest first) inside [lf_lo, lf_hi]. An arc whose
    apex lies above the window (-Z'' falling from the first point) is placed just above f_max.
    Pairs not located this way are spread evenly over the remaining window, as before."""
    if n == 0:
        return []
    lf = np.log10(freq)
    inside = (lf <= lf_hi + 1e-9) & (lf >= lf_lo - 1e-9)
    y = -Z.imag.copy()
    y[~inside] = -np.inf
    found = []
    if y.size >= 5:
        ys = np.convolve(np.where(np.isfinite(y), y, 0.0), np.ones(3) / 3, mode="same")
        ys[~inside] = -np.inf
        for i in range(1, y.size - 1):
            if inside[i] and ys[i] > ys[i - 1] and ys[i] >= ys[i + 1] and ys[i] > 0:
                found.append((ys[i], lf[i]))
        first = int(np.argmax(inside))
        if inside[first] and first + 2 < y.size and ys[first] > ys[first + 1] > ys[first + 2] and ys[first] > 0:
            found.append((ys[first], lf_hi + 0.3))          # apex above the measured window
    found = sorted(found, reverse=True)[:n]
    fcs = sorted((x for _, x in found), reverse=True)
    while len(fcs) < n:                                     # fill the largest gaps evenly
        edges = [lf_hi + 0.3] + fcs + [lf_lo]
        gaps = [(edges[i] - edges[i + 1], i) for i in range(len(edges) - 1)]
        g, i = max(gaps)
        fcs.insert(i, edges[i] - g / 2)
        fcs = sorted(fcs, reverse=True)
    return [10 ** x for x in fcs]


def default_guess(names, freq, Z, circuit=None):
    """Data-informed starting values so first fits usually converge.

    Parallel R-C/CPE pairs are given characteristic frequencies spread evenly (in log f)
    across the measured window, first pair at high frequency. Identical starting values
    for two arcs would let the optimiser swap their labels.
    """
    zr = Z.real
    r_hf = max(float(np.min(zr)), 1e-6)
    pairs = _PAIR.findall(circuit.replace(" ", "")) if circuit else []
    in_pairs = {x for pr in pairs for x in pr}
    tail = any(element_type(n.split("_")[0]) in ("CPE", "C", "W", "Wo", "Ws")
               and n.split("_")[0] not in in_pairs for n in names)
    lf_hi, lf_lo = np.log10(freq.max()), np.log10(freq.min())
    if tail:
        # series capacitive/diffusive tail (e.g. blocking electrodes): the arcs end where -phase
        # is smallest, before the tail rises; the tail's own Z' must not set the arc size.
        start = max(1, int(0.05 * freq.size))
        i_min = start + int(np.argmin(np.abs(np.angle(Z[start:]))))
        span = max(float(zr[i_min] - r_hf), 1e-3)
        lf_lo = np.log10(freq[i_min])
    else:
        span = max(float(np.max(zr) - np.min(zr)), 1e-3)
    paired = {}
    for k, (a, b, fc) in enumerate(zip([p[0] for p in pairs], [p[1] for p in pairs],
                                      _arc_frequencies(freq, Z, len(pairs), lf_hi, lf_lo))):
        paired[a] = paired[b] = (fc, len(pairs))
    n_r = sum(1 for n in names if element_type(n) == "R")
    first_r = next((x for x in names if element_type(x) == "R"), None)
    r_share = span / max(len(pairs), 1)
    guesses = []
    for n in names:
        t = element_type(n)
        el = n.split("_")[0]
        idx = int(n.split("_")[1]) if "_" in n else 0
        slot = _SLOT_DEFAULTS.get(t, [1.0])
        val = slot[min(idx, len(slot) - 1)]
        if t == "R":
            if el in paired:
                val = r_share
            elif n == first_r:
                val = r_hf
            else:
                val = span / max(n_r - 1, 1)
        elif el in paired and t in ("CPE", "C"):
            fc, _ = paired[el]
            alpha = 0.85 if t == "CPE" else 1.0
            q = 1.0 / (r_share * (2 * np.pi * fc) ** alpha)
            val = q if idx == 0 else alpha
        elif t == "CPE" and idx == 0:
            # unpaired CPE (e.g. blocking electrode): match |Z| at the lowest frequency
            val = 1.0 / (max(np.abs(Z[-1]), 1e-9) * (2 * np.pi * freq.min()) ** 0.85)
        elif val is None:
            val = span
        guesses.append(float(val))
    return guesses


def default_bounds(circuit, fixed=None):
    """impedance.py's own defaults: [0, inf) for everything, CPE/La exponent in [0, 1].
    Returned as {free_param_name: (lo, hi)}."""
    fixed = fixed or {}
    names, _ = parse_circuit(circuit)
    free = [n for n in names if n not in fixed]
    lo, hi = fitting.set_default_bounds(circuit.replace(" ", ""), constants=fixed)
    return {n: (float(l), float(h)) for n, l, h in zip(free, lo, hi)}


@dataclass
class ECMResult:
    circuit: str
    names: list
    units: list
    values: np.ndarray
    stderr: np.ndarray
    fixed: dict
    freq: np.ndarray
    Z: np.ndarray
    Z_fit: np.ndarray
    f_smooth: np.ndarray
    Z_smooth: np.ndarray
    weighting: str
    metrics: dict = field(default_factory=dict)
    derived: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def param(self, name):
        if name in self.fixed:
            return self.fixed[name], 0.0
        i = self.names.index(name)
        return float(self.values[i]), float(self.stderr[i])

    def all_params(self):
        """Ordered list of (name, unit, value, stderr, fixed) over the whole circuit."""
        full_names, full_units = parse_circuit(self.circuit)
        rows = []
        for n, u in zip(full_names, full_units):
            v, e = self.param(n)
            rows.append((n, u, v, e, n in self.fixed))
        return rows


def _model(circuit, constants):
    return fitting.wrapCircuit(circuit, constants)


def predict(circuit, constants, params, freq):
    out = _model(circuit, constants)(np.asarray(freq, float), *params)
    n = len(freq)
    return np.asarray(out[:n]) + 1j * np.asarray(out[n:])


def _objective(Z, Zf, weighting):
    if weighting == "modulus":
        m = np.abs(Z)
        return float(np.sum(((Z.real - Zf.real) / m) ** 2 + ((Z.imag - Zf.imag) / m) ** 2))
    if weighting == "proportional":
        return float(np.sum(((Z.real - Zf.real) / np.maximum(abs(Z.real), 1e-12)) ** 2
                            + ((Z.imag - Zf.imag) / np.maximum(abs(Z.imag), 1e-12)) ** 2))
    return float(np.sum(np.abs(Z - Zf) ** 2))


def _top_level_pairs(circuit):
    """Simple parallel pairs p(A,B) that sit directly in the top-level series chain."""
    out = []
    for m in _PAIR.finditer(circuit):
        depth = circuit[:m.start()].count("(") - circuit[:m.start()].count(")")
        if depth == 0:
            out.append((m.group(1), m.group(2)))
    return out


def _pair_tau(pair, get):
    a, b = pair
    ta, tb = element_type(a), element_type(b)
    d = {ta: a, tb: b}
    if "R" in d and "CPE" in d:
        R, Q, al = get(d["R"]), get(d["CPE"] + "_0"), get(d["CPE"] + "_1")
        if not (R > 0 and Q > 0 and al > 0):
            return np.inf
        with np.errstate(over="ignore"):
            return (R * Q) ** (1 / al)
    if "R" in d and "C" in d:
        return get(d["R"]) * get(d["C"])
    return None


def _pair_param_names(pair):
    names = []
    for el in pair:
        n = circuit_elements[element_type(el)].num_params
        names += [el] if n == 1 else [f"{el}_{j}" for j in range(n)]
    return names


def canonicalize(circuit, free, values, errs, fixed):
    """Order interchangeable series pairs by time constant (first in the circuit = fastest).

    R0-p(R1,CPE1)-p(R2,CPE2) has the same impedance with arcs 1 and 2 swapped; without this,
    multi-start runs report label permutations as different solutions, and R1 means
    different things in different files.
    """
    values, errs = np.array(values, float), np.array(errs, float) if errs is not None else None
    idx = {n: i for i, n in enumerate(free)}
    get = lambda n: fixed[n] if n in fixed else values[idx[n]]  # noqa: E731
    groups = {}
    for pr in _top_level_pairs(circuit):
        sig = tuple(element_type(x) for x in pr)
        groups.setdefault(sig, []).append(pr)
    new_v = values.copy()
    new_e = errs.copy() if errs is not None else None
    for sig, prs in groups.items():
        if len(prs) < 2 or any(n in fixed for pr in prs for n in _pair_param_names(pr)):
            continue
        taus = [_pair_tau(pr, get) for pr in prs]
        if any(t is None for t in taus):
            continue
        order = np.argsort(taus)
        for slot, src in zip(prs, [prs[k] for k in order]):
            for n_dst, n_src in zip(_pair_param_names(slot), _pair_param_names(src)):
                new_v[idx[n_dst]] = values[idx[n_src]]
                if new_e is not None:
                    new_e[idx[n_dst]] = errs[idx[n_src]]
    return new_v, new_e


def _starts(p0, lo, hi, n, seed):
    rng = np.random.default_rng(seed)
    out = [list(p0)]
    for _ in range(max(n - 1, 0)):
        p = []
        for v, l, h in zip(p0, lo, hi):
            if np.isfinite(h) and h <= 1.0:          # exponent-like parameter
                p.append(float(rng.uniform(max(l, 0.5), h)))
            else:
                q = v * 10 ** rng.uniform(-1.5, 1.5)
                p.append(float(np.clip(q, l, h)))
        out.append(p)
    return out


class _Budget(Exception):
    pass


def _timed_curve_fit(freq, Z, circuit, p0, fixed, lo, hi, weighting, deadline):
    """The non-global path of impedance.py's ``circuit_fit`` (same model, bounds, weighting,
    maxfev=1e5 and ftol=1e-13), with a wall-clock deadline and one change: ``x_scale="jac"``.
    With the default unit scaling the trust-region solver takes steps of the same absolute size
    for every parameter, so a CPE Q of 1e-10 (pF-nF pellet capacitances) next to an R of 1e3
    barely moves from its starting value.

    Every evaluated parameter vector is scored; if the deadline passes, the best one seen is
    returned with ``converged=False`` instead of blocking the interface for minutes.
    """
    f = np.asarray(freq, float)
    y = np.hstack([Z.real, Z.imag])
    model = fitting.wrapCircuit(circuit, fixed)
    if weighting == "modulus":
        sig = np.hstack([np.abs(Z), np.abs(Z)])
    elif weighting == "proportional":
        sig = np.hstack([np.maximum(np.abs(Z.real), 1e-12), np.maximum(np.abs(Z.imag), 1e-12)])
    else:
        sig = None
    w = 1.0 / sig if sig is not None else 1.0
    best = {"obj": np.inf, "p": np.asarray(p0, float)}

    def timed(ff, *p):
        out = model(ff, *p)
        obj = float(np.sum(((out - y) * w) ** 2))
        if np.isfinite(obj) and obj < best["obj"]:
            best["obj"], best["p"] = obj, np.asarray(p, float)
        if time.monotonic() > deadline:
            raise _Budget()
        return out

    try:
        popt, pcov = curve_fit(timed, f, y, p0=p0, bounds=(lo, hi), sigma=sig, maxfev=int(1e5), ftol=1e-13,
                               x_scale="jac")
        return popt, np.sqrt(np.diag(pcov)), True
    except _Budget:
        return best["p"], None, False


def fit_ecm(freq, Z, circuit, initial: dict, fixed: dict | None = None, bounds: dict | None = None,
            weighting="modulus", global_opt=False, n_starts=1, seed=0, time_budget_s=45.0):
    """Fit ``circuit`` to (freq, Z).

    initial : {param_name: guess} for every *free* parameter
    fixed   : {param_name: value} held constant
    bounds  : {param_name: (lo, hi)}; defaults: [0, inf), CPE exponents [0, 1]

    With ``n_starts > 1`` the fit is repeated from log-perturbed starting points (factor
    10^±1.5; exponents in [0.5, 1]) and the lowest weighted objective is kept; equally good
    minima with different parameters are reported as non-identifiability. All starts together
    get ``time_budget_s`` seconds; a start that runs out of time contributes the best point it
    reached and the result is flagged as not converged.
    """
    circuit = circuit.replace(" ", "")
    fixed = dict(fixed or {})
    names_all, units_all = parse_circuit(circuit)
    free = [n for n in names_all if n not in fixed]
    if not free:
        raise ValueError("All parameters are fixed; nothing to fit.")
    units = [units_all[names_all.index(n)] for n in free]
    p0 = [float(initial[n]) for n in free]
    bd = default_bounds(circuit, fixed)
    bd.update(bounds or {})
    lo = [float(bd[n][0]) for n in free]
    hi = [float(bd[n][1]) for n in free]
    if any(l >= h for l, h in zip(lo, hi)):
        raise ValueError("Each lower bound must be smaller than its upper bound.")
    p0 = [float(np.clip(p, l, h)) for p, l, h in zip(p0, lo, hi)]

    trials = []
    caught = set()
    t_end = time.monotonic() + float(time_budget_s)
    timed_out = 0
    starts = _starts(p0, lo, hi, 1 if global_opt else int(n_starts), seed)
    for k, start in enumerate(starts):
        if k > 0 and time.monotonic() > t_end:
            caught.add(f"Time budget used up: {k} of {len(starts)} starts were run.")
            break
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                if global_opt:
                    popt, perr = fitting.circuit_fit(freq, Z, circuit, start, constants=fixed, bounds=(lo, hi),
                                                     global_opt=True)
                else:
                    # the first start always gets some time even if a previous call was slow
                    deadline = max(t_end, time.monotonic() + 5.0) if k == 0 else t_end
                    popt, perr, ok = _timed_curve_fit(freq, Z, circuit, start, fixed, lo, hi, weighting, deadline)
                    timed_out += 0 if ok else 1
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                caught.add(f"A start failed: {exc}")
                continue
            caught.update(str(x.message) for x in w)
        obj = _objective(Z, predict(circuit, fixed, popt, freq), weighting)
        popt, perr = canonicalize(circuit, free, popt, perr, fixed)
        trials.append((obj, np.asarray(popt, float), perr))
    if not trials:
        raise RuntimeError("The fit did not converge from any starting point. Check the circuit and initial values.")
    trials.sort(key=lambda t: t[0])
    _, popt, perr = trials[0]
    perr = np.full(len(popt), np.nan) if perr is None else np.asarray(perr, float)
    multistart = _multistart_summary(trials, free)
    if timed_out:
        multistart["warnings"].insert(0, (
            f"{timed_out} start(s) stopped at the {time_budget_s:.0f} s time limit without converging. The parameters "
            "shown are the best point reached. Slow convergence usually means the circuit has more parameters than "
            "this spectrum can determine: remove elements whose process is not visible in the data, fix "
            "parameters you know, or tighten bounds."))
    multistart["metrics"]["n_starts_timed_out"] = timed_out

    Z_fit = predict(circuit, fixed, popt, freq)
    fs = np.logspace(np.log10(freq.max()), np.log10(freq.min()), 400)
    Z_sm = predict(circuit, fixed, popt, fs)

    res = ECMResult(circuit, free, units, np.asarray(popt, float), perr, fixed, freq, Z, Z_fit, fs, Z_sm,
                    weighting)
    res.metrics = fit_metrics(Z, Z_fit, n_params=len(free))
    res.derived = derived_quantities(res)
    res.metrics.update(multistart["metrics"])
    res.warnings = sorted(caught) + multistart["warnings"] + _sanity_warnings(res, lo, hi)
    return res


def _multistart_summary(trials, names):
    best = trials[0][0]
    same = [t for t in trials if t[0] <= best * 1.01 + 1e-15]
    out = {"metrics": {"n_starts_converged": len(trials), "n_starts_at_best_minimum": len(same)}, "warnings": []}
    if len(trials) > 1 and len(same) == 1:
        out["warnings"].append(f"Only 1 of {len(trials)} starts reached the best minimum; the error surface is rugged. "
                               "Consider a simpler circuit or tighter bounds.")
    if len(same) > 1:
        P = np.array([t[1] for t in same])
        spread = np.ptp(P, axis=0) / np.maximum(np.abs(P[0]), 1e-300)
        bad = [n for n, sp in zip(names, spread) if sp > 0.10]
        if bad:
            out["warnings"].append("Equally good fits (within 1% of the best objective) give different values for "
                                   f"{', '.join(bad)}: these parameters are not identifiable from this spectrum.")
    return out


def fit_metrics(Z, Z_fit, n_params):
    mod = np.abs(Z)
    dre = (Z.real - Z_fit.real) / mod
    dim = (Z.imag - Z_fit.imag) / mod
    n = Z.size
    chi2 = float(np.sum(dre ** 2 + dim ** 2))  # Boukamp pseudo-chi-square, modulus weighted
    dof = max(2 * n - n_params, 1)
    rmse = float(np.sqrt(np.mean(np.abs(Z - Z_fit) ** 2)))
    return {
        "pseudo_chi2": chi2,
        "pseudo_chi2_per_dof": chi2 / dof,
        "rmse_ohm": rmse,
        "mean_abs_rel_res_re_pct": float(np.mean(np.abs(dre)) * 100),
        "mean_abs_rel_res_im_pct": float(np.mean(np.abs(dim)) * 100),
        "max_abs_rel_res_pct": float(np.max(np.abs(np.concatenate([dre, dim]))) * 100),
        "n_points": int(n),
        "n_free_params": int(n_params),
    }


def relative_residuals(Z, Z_fit):
    mod = np.abs(Z)
    return (Z.real - Z_fit.real) / mod * 100, (Z.imag - Z_fit.imag) / mod * 100


_PAIR = re.compile(r"p\(\s*([A-Za-z]+\d+)\s*,\s*([A-Za-z]+\d+)\s*\)")


def derived_quantities(res: ECMResult):
    """Effective capacitance and characteristic frequency for simple parallel pairs.

    p(R, CPE): C_eff = Q^(1/a) * R^((1-a)/a)   (Hsu & Mansfeld, Corrosion 57 (2001) 747)
               f_c   = 1 / (2 pi (R Q)^(1/a))
    p(R, C)  : f_c = 1 / (2 pi R C)
    Capacitance magnitudes help assign arcs (bulk ~pF, grain boundary ~nF, electrode ~uF;
    Irvine, Sinclair & West, Adv. Mater. 2 (1990) 132).
    """
    out = []
    for a, b in _PAIR.findall(res.circuit):
        ta, tb = element_type(a), element_type(b)
        pair = {ta: a, tb: b}
        if "R" in pair and "CPE" in pair:
            R, _ = res.param(pair["R"])
            Q, _ = res.param(pair["CPE"] + "_0")
            al, _ = res.param(pair["CPE"] + "_1")
            if R > 0 and Q > 0 and al > 0:
                # in log space: with a small exponent the powers overflow a float
                ln_c = (np.log(Q) + (1 - al) * np.log(R)) / al
                ln_tau = np.log(R * Q) / al
                ceff = float(np.exp(ln_c)) if abs(ln_c) < 700 else float("nan")
                tau = float(np.exp(ln_tau)) if abs(ln_tau) < 700 else float("nan")
                fc = 1 / (2 * np.pi * tau) if np.isfinite(tau) and tau > 0 else float("nan")
                out.append({"pair": f"p({a},{b})", "R_ohm": R, "C_eff_F": ceff, "alpha": al,
                            "f_c_Hz": fc, "tau_s": tau})
        elif "R" in pair and "C" in pair:
            R, _ = res.param(pair["R"])
            C, _ = res.param(pair["C"])
            if R > 0 and C > 0:
                out.append({"pair": f"p({a},{b})", "R_ohm": R, "C_eff_F": C, "alpha": 1.0,
                            "f_c_Hz": 1 / (2 * np.pi * R * C), "tau_s": R * C})
    return out


def _sanity_warnings(res: ECMResult, lo, hi):
    w = []
    for n, v, e, l, h in zip(res.names, res.values, res.stderr, lo, hi):
        if np.isfinite(e) and v != 0 and abs(e / v) > 1.0:
            w.append(f"{n}: standard error exceeds 100% of the value; the parameter is not determined "
                     "by the data (over-parameterised circuit or arc outside the frequency window).")
        span = (h - l) if np.isfinite(h) else None
        if (l > 0 and abs(v - l) <= 1e-9 * max(abs(l), 1)) or (span and abs(h - v) <= 1e-6 * span) or (l == 0 and v < 1e-12 and element_type(n) == "R"):
            w.append(f"{n} = {v:.3g} sits at a bound; the fit is constrained, not converged freely.")
    for d in res.derived:
        if d["f_c_Hz"] > res.freq.max() or d["f_c_Hz"] < res.freq.min():
            w.append(f"{d['pair']}: characteristic frequency {d['f_c_Hz']:.3g} Hz lies outside the measured "
                     f"range ({res.freq.min():.3g}-{res.freq.max():.3g} Hz); its R and C are extrapolated.")
    return w

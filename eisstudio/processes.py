"""Per-process analysis of a fitted temperature series.

Each R-C or R-CPE pair of the equivalent circuit is one relaxation process. Across a series the
same process is identified by its time constant (fastest first), not by the element's name: a
fit can call the fast arc R1 at one temperature and R2 at the next. Each process is then
classified by its effective capacitance scaled to unit cell constant (Irvine, Sinclair & West,
Adv. Mater. 2 (1990) 132), and gets its own Arrhenius fits:

* conductivity, for a bulk process: sigma = (L/A) / R;
* specific grain-boundary conductivity (brick-layer model), for a grain-boundary process:
  sigma_gb = (L/A) (C_bulk / C_gb) / R_gb, which needs a bulk process in the same spectrum;
* 1/R, for any process (an interface has a resistance, not a conductivity);
* the characteristic frequency f_c = 1/(2 pi tau);
* the position of the matching DRT peak, when DRTs were computed.

A point whose characteristic frequency lies outside the measured window has an extrapolated time
constant; its resistance is often still fixed by where the arc meets the next arc or the spike.
The caller decides which fits keep such points. A fit needs at least three temperatures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import ecm as ecmm
from .series import arrhenius

# Irvine, Sinclair & West (1990), capacitances for a cell constant L/A = 1 cm^-1
CLASSES = [
    (2e-11, "bulk"),
    (1e-8, "grain boundary"),
    (1e-7, "surface layer / interphase"),
    (np.inf, "electrode interface"),
]
CLASS_NAMES = [c for _, c in CLASSES] + ["series resistance"]
_PAIR = re.compile(r"p\(\s*([A-Za-z]+\d+)\s*,\s*([A-Za-z]+\d+)\s*\)")


def classify(C_eff_F, cell_constant_per_cm=None):
    """Class of a process from its effective capacitance. With the cell constant L/A (1/cm) the
    capacitance is scaled to unit cell constant first, as in Irvine et al.; without it the raw
    value is used (correct only for L/A near 1 cm^-1)."""
    if C_eff_F is None or not np.isfinite(C_eff_F) or C_eff_F <= 0:
        return "unknown"
    c = C_eff_F * (cell_constant_per_cm or 1.0)
    for lim, name in CLASSES:
        if c <= lim:
            return name
    return CLASSES[-1][1]


@dataclass
class Point:
    key: object
    T_C: float
    element: str            # e.g. "p(R1,CPE1)" or "R0"
    R: float
    dR: float
    C_eff: float | None
    alpha: float | None
    f_c: float | None
    outside: bool           # characteristic frequency outside the measured window


@dataclass
class Process:
    index: int              # 1 = fastest
    kind: str               # "pair" or "series"
    points: list = field(default_factory=list)
    cls: str = "unknown"
    notes: list = field(default_factory=list)
    drt_tau: dict = field(default_factory=dict)     # key -> tau of the matched DRT peak

    @property
    def label(self):
        names = sorted({p.element for p in self.points})
        if self.kind == "series":
            return f"{names[0]} (series)"
        return f"P{self.index}: " + " / ".join(names)

    @property
    def r_name(self):
        """Name of the resistance (most common over the series)."""
        names = [_r_of(p.element) for p in self.points]
        return max(set(names), key=names.count) if names else None


def _r_of(element):
    m = _PAIR.match(element)
    if not m:
        return element
    a, b = m.groups()
    return a if ecmm.element_type(a) == "R" else b


def track(fits, cell_constant_per_cm=None):
    """fits: list of (key, T_C, ECMResult) in temperature order. Returns (processes, notes).

    Pairs are ranked by characteristic frequency in each spectrum (fastest first) and the k-th
    pair of every spectrum forms process k. Series resistances (not in a pair) are processes of
    their own, matched by name."""
    notes = []
    fits = [f for f in fits if f[2] is not None]
    if not fits:
        return [], notes
    n_pairs = min(len(f[2].derived) for f in fits)
    if any(len(f[2].derived) != n_pairs for f in fits):
        notes.append("Not every spectrum has the same number of R–C/CPE pairs; only the first "
                     f"{n_pairs} (fastest) are tracked.")
    procs = [Process(k + 1, "pair") for k in range(n_pairs)]
    in_pair = set()
    for key, T, res in fits:
        fmin, fmax = float(np.min(res.freq)), float(np.max(res.freq))
        pairs = sorted(res.derived, key=lambda d: -(d["f_c_Hz"] if np.isfinite(d["f_c_Hz"]) else 0.0))
        for k, d in enumerate(pairs[:n_pairs]):
            rn = _r_of(d["pair"])
            in_pair.update(_PAIR.match(d["pair"]).groups())
            R, dR = res.param(rn)
            fc = d["f_c_Hz"] if np.isfinite(d["f_c_Hz"]) else None
            procs[k].points.append(Point(key, T, d["pair"], R, dR, d["C_eff_F"], d["alpha"], fc,
                                         bool(fc is None or not (fmin <= fc <= fmax))))
    # series resistances
    first = fits[0][2]
    for n, *_ in first.all_params():
        if ecmm.element_type(n) == "R" and n not in in_pair:
            pr = Process(len(procs) + 1, "series", cls="series resistance")
            for key, T, res in fits:
                try:
                    R, dR = res.param(n)
                except ValueError:
                    continue
                pr.points.append(Point(key, T, n, R, dR, None, None, None, False))
            procs.append(pr)
    for pr in procs:
        if pr.kind != "pair":
            continue
        names = [p.element for p in pr.points]
        common = max(set(names), key=names.count)
        swapped = [p for p in pr.points if p.element != common]
        if swapped:
            pr.notes.append("Matched by time constant: at " + ", ".join(f"{p.T_C:g} °C" for p in swapped)
                            + f" this process was fitted as {swapped[0].element} rather than {common}.")
        cls = [classify(p.C_eff, cell_constant_per_cm) for p in pr.points if not p.outside]
        cls = cls or [classify(p.C_eff, cell_constant_per_cm) for p in pr.points]
        pr.cls = max(set(cls), key=cls.count)
        if len(set(cls)) > 1:
            pr.notes.append("Its capacitance falls in different classes at different temperatures ("
                            + ", ".join(sorted(set(cls))) + ").")
        c = np.array([p.C_eff for p in pr.points if p.C_eff], float)
        if c.size > 1 and np.max(c) / np.min(c) > 5:
            pr.notes.append(f"Its effective capacitance changes {np.max(c) / np.min(c):.0f}-fold over the "
                            "series. A real bulk or grain-boundary capacitance changes little with "
                            "temperature, so the fit probably gives this element a different meaning at "
                            "some temperatures (often an arc leaving the window).")
        out = [p for p in pr.points if p.outside]
        if out:
            pr.notes.append("Characteristic frequency outside the measured window at "
                            + ", ".join(f"{p.T_C:g} °C" for p in out)
                            + ": those values are extrapolated.")
    return procs, notes


QUANTITIES = {
    "sigma": "σ = (L/A)/R",
    "sigma_gb": "specific σ_gb = (L/A)(C_bulk/C_gb)/R",
    "inv_R": "1/R",
    "f_c": "f_c = 1/(2πτ)",
    "drt": "DRT peak 1/(2πτ)",
}


def values(proc, quantity, cell_constant_per_cm=None, bulk=None, include_outside=False):
    """(T_C, y, dy, outside_mask) for one process and quantity; None when not applicable."""
    pts = proc.points
    T = np.array([p.T_C for p in pts], float)
    out = np.array([p.outside for p in pts], bool)
    if quantity == "inv_R":
        y = np.array([1 / p.R if p.R > 0 else np.nan for p in pts])
        dy = np.array([p.dR / p.R ** 2 if p.R > 0 else np.nan for p in pts])
    elif quantity == "sigma":
        if not cell_constant_per_cm:
            return None
        y = np.array([cell_constant_per_cm / p.R if p.R > 0 else np.nan for p in pts])
        dy = y * np.array([p.dR / p.R if p.R > 0 else np.nan for p in pts])
    elif quantity == "sigma_gb":
        if not cell_constant_per_cm or bulk is None:
            return None
        cb = {p.key: p.C_eff for p in bulk.points}
        y = np.array([cell_constant_per_cm * (cb.get(p.key) or np.nan) / p.C_eff / p.R
                      if p.R > 0 and p.C_eff else np.nan for p in pts])
        out = out | np.array([any(b.outside for b in bulk.points if b.key == p.key) for p in pts])
        dy = y * np.array([p.dR / p.R if p.R > 0 else np.nan for p in pts])
    elif quantity == "f_c":
        if proc.kind != "pair":
            return None
        y = np.array([p.f_c if p.f_c else np.nan for p in pts])
        dy = np.zeros_like(y)
    elif quantity == "drt":
        if not proc.drt_tau:
            return None
        y = np.array([1 / (2 * np.pi * proc.drt_tau[p.key]) if p.key in proc.drt_tau else np.nan for p in pts])
        dy = np.zeros_like(y)
        out = np.zeros_like(out)
    else:
        raise ValueError(quantity)
    return T, y, dy, out


MIN_POINTS = 3


def fit(proc, quantity, cell_constant_per_cm=None, bulk=None, include_outside=False, times_T=True):
    """Arrhenius fit for one process. times_T applies to conductivities only. Returns the
    arrhenius() dict plus 'used' (bool mask) and 'quantity', or None."""
    v = values(proc, quantity, cell_constant_per_cm, bulk, include_outside)
    if v is None:
        return None
    T, y, dy, out = v
    use = np.isfinite(y) & (y > 0) & (include_outside | ~out)
    if use.sum() < MIN_POINTS or np.unique(T[use]).size < MIN_POINTS:
        return None
    arr = arrhenius(T[use], y[use], times_T=times_T and quantity in ("sigma", "sigma_gb"))
    arr.update({"used": use, "quantity": quantity, "T_all": T, "y_all": y, "dy_all": dy, "outside": out})
    return arr


def match_drt(procs, drts, max_decades=0.5):
    """Attach to each pair process, per spectrum, the DRT peak nearest (in log tau) to its ECM time
    constant, within ``max_decades``. Each peak goes to one process at most (closest pairs first);
    peaks outside the window, or within 0.15 decade of its edges (where a regularised DRT piles up
    whatever lies beyond), are skipped. drts: {key: DRTResult}."""
    pair_procs = [pr for pr in procs if pr.kind == "pair"]
    for pr in pair_procs:
        pr.drt_tau = {}
    keys = {p.key for pr in pair_procs for p in pr.points}
    for key in keys:
        d = drts.get(key)
        if d is None:
            continue
        # skip peaks outside the window or pinned within 0.15 decade of its edges
        t_lo, t_hi = 1 / (2 * np.pi * float(np.max(d.freq))), 1 / (2 * np.pi * float(np.min(d.freq)))
        peaks = [pk for pk in d.peaks if not pk.get("outside")
                 and np.log10(pk["tau_s"] / t_lo) > 0.15 and np.log10(t_hi / pk["tau_s"]) > 0.15]
        cand = []
        for pr in pair_procs:
            p = next((q for q in pr.points if q.key == key), None)
            if p is None or not p.f_c:
                continue
            tau_c = 1 / (2 * np.pi * p.f_c)
            for j, pk in enumerate(peaks):
                dist = abs(np.log10(pk["tau_s"] / tau_c))
                if dist <= max_decades:
                    cand.append((dist, id(pr), j, pr, pk))
        used_p, used_k = set(), set()
        for dist, pid, j, pr, pk in sorted(cand, key=lambda c: c[0]):
            if pid in used_p or j in used_k:
                continue
            pr.drt_tau[key] = pk["tau_s"]
            used_p.add(pid)
            used_k.add(j)


def suggested_ionic(procs):
    """Resistance names that belong in the ionic resistance: bulk and grain-boundary processes."""
    return [pr.r_name for pr in procs if pr.kind == "pair" and pr.cls in ("bulk", "grain boundary")]

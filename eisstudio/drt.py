"""Distribution of relaxation times via pyDRTtools (Ciucci lab).

Model (as implemented by pyDRTtools):
    Z(f) = R_inf + j 2 pi f L + integral gamma(ln tau) / (1 + j 2 pi f tau) d(ln tau)

so gamma is in Ohm per unit ln(tau), and the area under a peak (integrated over
ln tau) is that process's polarisation resistance. The characteristic frequency
of a peak at tau_p is f_p = 1 / (2 pi tau_p).

Refs: Wan et al., Electrochim. Acta 184 (2015) 483 (RBF discretisation);
      Saccoccio et al., Electrochim. Acta 147 (2014) 470 (ridge regularisation);
      Maradesa et al., J. Electrochem. Soc. 170 (2023) 030502 (lambda selection);
      Ciucci & Chen, Electrochim. Acta 167 (2015) 439 (Bayesian credible bands).

Notes on pyDRTtools 0.2 (PyPI) that this wrapper works around:
  * ``import pyDRTtools`` imports the PyQt5 GUI; we register empty stand-ins for the
    GUI modules so it imports on a headless server.
  * ``induct_used=2`` ("discard inductive data") is handled identically to 0 in
    ``runs.simple_run``; we implement discarding ourselves by removing points with Z''>0.
  * The 'Re Data' branch prints ``entry.lambda_lambda`` (typo), which raises
    AttributeError; we pre-set that attribute so the run completes.
  * In 'Im Data' mode R_inf is not part of the regression; we estimate it afterwards
    as mean(Z' - Z'_DRT) and flag it as estimated.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks


def _import_pydrttools():
    for name in ("pyDRTtools.cli", "pyDRTtools.GUI", "pyDRTtools.layout"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    with contextlib.redirect_stdout(io.StringIO()):
        from pyDRTtools import runs  # noqa: WPS433
    from cvxopt import solvers
    solvers.options["show_progress"] = False
    return runs


RBF_TYPES = ["Gaussian", "C0 Matern", "C2 Matern", "C4 Matern", "C6 Matern",
             "Inverse Quadratic", "Inverse Quadric", "Cauchy", "PWL"]
DATA_USED = {"Combined Re-Im": "Combined Re-Im Data", "Imaginary only": "Im Data", "Real only": "Re Data"}
INDUCTANCE = {
    "Fit inductance L": "fit",
    "Ignore inductance (keep all points)": "ignore",
    "Discard inductive points (Z'' > 0)": "discard",
}
DERIVATIVE = ["1st order", "2nd order"]
LAMBDA_METHODS = {
    "GCV (generalised cross-validation)": "GCV",
    "mGCV (modified GCV)": "mGCV",
    "rGCV (robust GCV)": "rGCV",
    "L-curve": "LC",
    "Re-Im discrepancy": "re-im",
    "k-fold cross-validation": "kf",
    "Manual (enter lambda)": "custom",
}
SHAPE_CONTROL = ["FWHM Coefficient", "Shape Factor"]


@dataclass
class DRTSettings:
    rbf_type: str = "Gaussian"
    data_used: str = "Combined Re-Im Data"
    inductance: str = "fit"
    der_used: str = "1st order"
    cv_type: str = "GCV"
    reg_param: float = 1e-3
    shape_control: str = "FWHM Coefficient"
    coeff: float = 0.5
    bayesian: bool = False
    n_mc: int = 2000
    peak_prominence: float = 0.02  # fraction of max(gamma)


@dataclass
class DRTResult:
    settings: DRTSettings
    freq: np.ndarray
    Z: np.ndarray
    Z_fit: np.ndarray
    tau: np.ndarray
    gamma: np.ndarray
    R_inf: float
    L: float
    lam: float
    R_inf_estimated: bool = False
    gamma_lo: np.ndarray | None = None
    gamma_hi: np.ndarray | None = None
    gamma_mean: np.ndarray | None = None
    peaks: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def R_pol(self) -> float:
        """Total polarisation resistance = integral of gamma over ln(tau)."""
        return float(np.trapezoid(self.gamma, np.log(self.tau)))


def run_drt(freq, Z, s: DRTSettings) -> DRTResult:
    runs = _import_pydrttools()
    warn = []
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    if s.inductance == "discard":
        keep = Z.imag <= 0
        if (~keep).sum():
            warn.append(f"Discarded {(~keep).sum()} inductive points (Z'' > 0) before the DRT.")
        freq, Z = freq[keep], Z[keep]
        induct_used = 0
    else:
        induct_used = 1 if s.inductance == "fit" else 0
    if freq.size < 8:
        raise ValueError("Need at least 8 points for a DRT.")

    order = np.argsort(freq)[::-1]
    freq, Z = freq[order], Z[order]
    dlog = np.diff(np.log10(freq))
    if np.std(dlog) / abs(np.mean(dlog)) > 0.01:
        warn.append("Frequencies are not uniformly log-spaced, so pyDRTtools uses full numerical "
                    "quadrature (slower). Results are unaffected.")

    entry = runs.EIS_object(freq, Z.real.copy(), Z.imag.copy())
    entry.lambda_lambda = "n/a"  # works around the 'Re Data' typo in pyDRTtools 0.2
    kw = dict(rbf_type=s.rbf_type, data_used=s.data_used, induct_used=induct_used, der_used=s.der_used,
              cv_type=s.cv_type, reg_param=float(s.reg_param), shape_control=s.shape_control,
              coeff=float(s.coeff))
    with contextlib.redirect_stdout(io.StringIO()):
        if s.bayesian:
            runs.Bayesian_run(entry, NMC_sample=int(s.n_mc), **kw)
        else:
            runs.simple_run(entry, **kw)

    lam = float(np.ravel(entry.lambda_value)[0])
    tau = np.asarray(entry.out_tau_vec, float).ravel()
    gamma = np.asarray(entry.gamma, float).ravel()
    Zfit = np.asarray(entry.mu_Z_re) + 1j * np.asarray(entry.mu_Z_im)
    R_inf, L = float(entry.R), float(entry.L)
    est = False
    if s.data_used == "Im Data":
        R_inf = float(np.mean(Z.real - Zfit.real))
        Zfit = Zfit + R_inf
        est = True
        warn.append("Imaginary-only DRT: R_inf is not part of the regression and was estimated as "
                    "mean(Z' - Z'_DRT).")
    if s.data_used == "Re Data":
        warn.append("Real-only DRT: the inductance cannot be identified and the imaginary part is predicted, "
                    "not fitted.")

    res = DRTResult(s, freq, Z, Zfit, tau, gamma, R_inf, L, lam, est)
    if s.bayesian:
        res.gamma_lo = np.asarray(entry.lower_bound, float).ravel()
        res.gamma_hi = np.asarray(entry.upper_bound, float).ravel()
        res.gamma_mean = np.asarray(entry.mean, float).ravel()

    bounds = (s.cv_type != "custom") and (abs(np.log10(lam) - (-7)) < 0.05 or abs(np.log10(lam)) < 0.05)
    if bounds:
        warn.append(f"lambda = {lam:.2e} is at the edge of pyDRTtools' search interval [1e-7, 1]; the selection "
                    "criterion did not find an interior optimum. Inspect the DRT with a few manual lambda values.")
    if s.cv_type != "custom" and abs(np.log(lam) - np.log(s.reg_param)) < 1e-6:
        warn.append(f"The {s.cv_type} optimiser returned the initial guess λ = {lam:.2e} unchanged, i.e. it did not "
                    "find a minimum (flat or ill-conditioned criterion). Try another λ-selection method or a "
                    "different initial λ, and compare.")
    res.peaks = find_drt_peaks(res, s.peak_prominence)
    res.metrics = _metrics(Z, Zfit)
    res.warnings = warn + _window_warnings(res)
    return res


def find_drt_peaks(res: DRTResult, prominence=0.02, minor_frac=0.03):
    """Locate peaks inside the measured tau window and integrate each between adjacent minima.

    Splitting at minima (rather than fitting Gaussians) is transparent but assigns
    overlapping tails to the nearer peak; treat R of strongly overlapping peaks as approximate.

    Peaks holding < ``minor_frac`` of the total area are flagged ``minor``. Small peaks flanking
    a large one are a known artefact of Tikhonov-regularised DRT applied to depressed (CPE-like)
    arcs; never assign a physical process to a minor peak without independent evidence
    (temperature/pO2/bias dependence, or a repeat measurement).
    """
    tau, g = res.tau, res.gamma
    if g.max() <= 0:
        return []
    tmin, tmax = 1 / (2 * np.pi * res.freq.max()), 1 / (2 * np.pi * res.freq.min())
    idx, _ = find_peaks(g, prominence=prominence * g.max())
    idx = [i for i in idx if tmin <= tau[i] <= tmax]
    if not idx:
        return []
    lnt = np.log(tau)
    minima, _ = find_peaks(-g)
    edges = [0]
    for a, b in zip(idx[:-1], idx[1:]):
        between = [m for m in minima if a < m < b]
        edges.append(between[int(np.argmin(g[between]))] if between else (a + b) // 2)
    edges.append(len(g) - 1)
    total = max(float(np.trapezoid(np.clip(g, 0, None), lnt)), 1e-300)
    out = []
    for k, i in enumerate(idx):
        lo, hi = edges[k], edges[k + 1]
        R = float(np.trapezoid(g[lo:hi + 1], lnt[lo:hi + 1]))
        tp = float(tau[i])
        out.append({"peak": k + 1, "tau_s": tp, "f_Hz": 1 / (2 * np.pi * tp), "gamma_max_ohm": float(g[i]),
                    "R_ohm": R, "C_eq_F": tp / R if R > 0 else np.nan,
                    "tau_lo_s": float(tau[lo]), "tau_hi_s": float(tau[hi]),
                    "share_pct": 100 * R / total, "minor": bool(R < minor_frac * total)})
    return out


def integrate_window(res: DRTResult, tau_lo, tau_hi):
    m = (res.tau >= tau_lo) & (res.tau <= tau_hi)
    if m.sum() < 2:
        return 0.0
    return float(np.trapezoid(res.gamma[m], np.log(res.tau[m])))


def _metrics(Z, Zfit):
    mod = np.abs(Z)
    dre = (Z.real - Zfit.real) / mod
    dim = (Z.imag - Zfit.imag) / mod
    return {
        "rmse_ohm": float(np.sqrt(np.mean(np.abs(Z - Zfit) ** 2))),
        "mean_abs_rel_res_re_pct": float(np.mean(np.abs(dre)) * 100),
        "mean_abs_rel_res_im_pct": float(np.mean(np.abs(dim)) * 100),
        "max_abs_rel_res_pct": float(np.max(np.abs(np.concatenate([dre, dim]))) * 100),
        "n_points": int(Z.size),
    }


def _window_warnings(res: DRTResult):
    w = []
    tmin, tmax = 1 / (2 * np.pi * res.freq.max()), 1 / (2 * np.pi * res.freq.min())
    lnt = np.log(res.tau)
    total = res.R_pol
    if total > 0:
        out_lo = float(np.trapezoid(np.where(res.tau < tmin, res.gamma, 0), lnt))
        out_hi = float(np.trapezoid(np.where(res.tau > tmax, res.gamma, 0), lnt))
        if out_lo > 0.05 * total:
            w.append(f"{out_lo / total:.0%} of the DRT area lies at tau shorter than the highest measured frequency. "
                     "A process is partly outside the window; R_inf and the first peak are not separately determined.")
        if out_hi > 0.05 * total:
            w.append(f"{out_hi / total:.0%} of the DRT area lies at tau longer than the lowest measured frequency. "
                     "If the low-frequency end is a blocking-electrode capacitive tail, crop it out: the DRT model "
                     "assumes a finite DC resistance and a capacitive tail has none.")
    # crude blocking-electrode detector: phase approaching -90 deg at low f with Z' still increasing
    ph = np.degrees(np.angle(res.Z[-5:]))
    if res.Z.size > 10 and np.mean(ph) < -60:
        w.append(f"The lowest-frequency points have phase ~{np.mean(ph):.0f} deg (capacitive tail). DRT of blocking "
                 "electrodes is ill-posed; set f_min above the onset of the tail.")
    return w

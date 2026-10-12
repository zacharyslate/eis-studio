"""Validation of the analysis core against synthetic spectra with known parameters.
Run:  python -m pytest -q tests/   (or python tests/test_core.py)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "eisstudio" / "sample_data"

from eisstudio import conductivity, drt, ecm, export, io, plotting, validation  # noqa: E402

TRUTH = {"R0": 50, "R1": 200, "CPE1_0": 1e-7, "CPE1_1": 0.9, "R2": 500, "CPE2_0": 1e-4, "CPE2_1": 0.8}


def _load(name, **kw):
    raw = (SAMPLES / name).read_bytes()
    sweeps, rep = io.load_spectrum(raw, name, **kw)
    return sweeps, rep


def test_parsers_agree():
    ref, _ = _load("synthetic_2arc.csv")
    ref = ref[0]
    for name in ("synthetic_2arc_biologic.mpt", "synthetic_2arc_gamry.DTA", "synthetic_2arc_modphase.txt"):
        sw, rep = _load(name)
        d = sw[0]
        assert d.n == ref.n, (name, d.n, rep.notes)
        err = np.max(np.abs(d.Z - ref.Z) / np.abs(ref.Z))
        assert err < 1e-4, (name, err, rep.notes, rep.columns)
        assert np.allclose(d.freq, ref.freq, rtol=1e-5)
    sw, rep = _load("synthetic_2arc_biologic.mpt")
    assert len(sw) == 2, rep.notes  # two cycles split into two sweeps


def test_blocking_headerless_sign():
    sw, rep = _load("synthetic_blocking_pellet.txt")
    d = sw[0]
    assert np.median(d.Z.imag) < 0, rep.notes  # -Z'' column auto-flipped
    sw2, _ = _load("synthetic_blocking_pellet.txt", imag_sign="Column is Z'' (signed)")
    assert np.median(sw2[0].Z.imag) > 0       # override respected


def _fit_truth():
    d = _load("synthetic_2arc.csv")[0][0]
    circ = "R0-p(R1,CPE1)-p(R2,CPE2)-L0"
    names, _ = ecm.parse_circuit(circ)
    guess = dict(zip(names, ecm.default_guess(names, d.freq, d.Z, circ)))
    return d, circ, guess, ecm.fit_ecm(d.freq, d.Z, circ, guess, weighting="modulus")


def test_ecm_recovers_truth():
    d, circ, guess, res = _fit_truth()
    for k, v in TRUTH.items():
        got, e = res.param(k)
        assert abs(got - v) / v < 0.05, (k, got, v)
        assert abs(got - v) < 4 * e + 1e-12 * v, (k, got, v, e)  # truth within ~4 standard errors
    assert res.metrics["mean_abs_rel_res_re_pct"] < 0.5
    # arcs are interchangeable: multi-start must not report a label swap as non-identifiability,
    # and R1 must be the fast (high-frequency) arc
    res_ms = ecm.fit_ecm(d.freq, d.Z, circ, guess, weighting="modulus", n_starts=10, seed=3)
    assert not any("not identifiable" in w for w in res_ms.warnings), res_ms.warnings
    assert abs(res_ms.param("R1")[0] - 200) / 200 < 0.05, res_ms.all_params()
    # fixed parameter path
    res2 = ecm.fit_ecm(d.freq, d.Z, circ, {k: v for k, v in guess.items() if k != "R0"}, fixed={"R0": 50.0})
    assert res2.param("R0") == (50.0, 0.0)


def test_ecm_derived_capacitance():
    res = _fit_truth()[3]
    d1 = [x for x in res.derived if x["pair"] == "p(R1,CPE1)"][0]
    R, Q, a = 200, 1e-7, 0.9
    assert abs(d1["C_eff_F"] - Q ** (1 / a) * R ** ((1 - a) / a)) / d1["C_eff_F"] < 0.1


def test_drt_recovers_resistances():
    d = _load("synthetic_2arc.csv")[0][0]
    res = drt.run_drt(d.freq, d.Z, drt.DRTSettings(inductance="fit"))
    assert abs(res.R_inf - 50) < 3, res.R_inf
    assert abs(res.R_pol - 700) / 700 < 0.03, res.R_pol
    assert len(res.peaks) >= 2, res.peaks
    big = sorted(res.peaks, key=lambda p: -p["R_ohm"])[:2]
    Rs = sorted(p["R_ohm"] for p in big)
    assert abs(Rs[0] - 200) / 200 < 0.15 and abs(Rs[1] - 500) / 500 < 0.1, Rs
    # tau of the ZARC: (R Q)^(1/a)
    t1 = (200 * 1e-7) ** (1 / 0.9)
    tp = min(p["tau_s"] for p in big)
    assert abs(np.log10(tp / t1)) < 0.15, (tp, t1)
    assert res.metrics["mean_abs_rel_res_re_pct"] < 1.0


def test_drt_options_run():
    d = _load("synthetic_2arc.csv")[0][0]
    for data_used in drt.DATA_USED.values():
        for ind in drt.INDUCTANCE.values():
            r = drt.run_drt(d.freq, d.Z, drt.DRTSettings(data_used=data_used, inductance=ind))
            assert np.all(np.isfinite(r.gamma))
    r = drt.run_drt(d.freq, d.Z, drt.DRTSettings(cv_type="custom", reg_param=1e-4, rbf_type="C2 Matern",
                                                 der_used="2nd order"))
    assert abs(r.lam - 1e-4) < 1e-12


def test_out_of_window_arc_is_flagged():
    d = _load("synthetic_blocking_pellet.txt")[0][0]
    circ = "R0-p(R1,CPE1)-CPE2"
    names, _ = ecm.parse_circuit(circ)
    res = ecm.fit_ecm(d.freq, d.Z, circ, dict(zip(names, ecm.default_guess(names, d.freq, d.Z, circ))), n_starts=12)
    assert any("outside the measured" in w or "not determined" in w for w in res.warnings), res.warnings
    # multi-start must beat the single-start local minimum (chi2/dof 1.5e-4 from the default guess)
    assert res.metrics["pseudo_chi2_per_dof"] < 5e-5, res.metrics
    # the blocking-electrode CPE is well determined; the R0/R1 split is NOT (arc above window)
    for k, v in {"CPE2_0": 2e-6, "CPE2_1": 0.8}.items():
        assert abs(res.param(k)[0] - v) / v < 0.05, (k, res.param(k))
    # R0 + R1 (total ionic resistance) is still well determined even if the split is not
    tot = res.param("R0")[0] + res.param("R1")[0]
    assert abs(tot - 190) / 190 < 0.05, tot
    dr = drt.run_drt(d.freq, d.Z, drt.DRTSettings())
    assert any("capacitive tail" in w or "blocking" in w for w in dr.warnings), dr.warnings


def test_circuit_tree_roundtrip_and_parallel_branches():
    from eisstudio import circuit_tree as ct
    for c in ["R0-p(R1,CPE1)-CPE2", "L0-R0-p(R1,CPE1)-p(R2,CPE2)", "R0-p(R1-Wo1,CPE1)", "p(R1-p(R2,C2),CPE1)-W1"]:
        assert ct.to_string(ct.parse(c)) == c
    assert ct.to_string(ct.parse("R0-p(p(R1,C1),R2)")) == "R0-p(R1,C1,R2)"
    # impedance.py evaluates 3-branch parallels correctly
    f = np.logspace(5, -1, 30)
    w = 2 * np.pi * f
    Z = ecm.predict("p(R1,C1,R2)", {}, [100.0, 1e-6, 300.0], f)
    exact = 1 / (1 / 100.0 + 1j * w * 1e-6 + 1 / 300.0)
    assert np.allclose(Z, exact, rtol=1e-10)
    # new elements get free names
    t = {"t": "s", "c": [{"t": "e", "k": "R", "n": "R0"}, {"t": "e", "k": "R", "n": None}, {"t": "e", "k": "R", "n": "R0"}]}
    assert ct.to_string(t) == "R0-R1-R2"


def test_fit_time_budget():
    import time as _t
    d = _load("synthetic_2arc.csv")[0][0]
    circ = "R0-p(R1,CPE1)-p(R2-Wo1,CPE2)"     # Warburg not supported by this spectrum: slow to converge
    names, _ = ecm.parse_circuit(circ)
    t0 = _t.monotonic()
    res = ecm.fit_ecm(d.freq, d.Z, circ, dict(zip(names, ecm.default_guess(names, d.freq, d.Z, circ))),
                      n_starts=4, time_budget_s=8)
    assert _t.monotonic() - t0 < 25                     # stopped near the 8 s budget, not after minutes
    stopped = res.metrics["n_starts_timed_out"] >= 1
    assert stopped == any("time limit" in w for w in res.warnings)   # a stop is always reported


def test_conductivity_arithmetic():
    g = conductivity.Geometry.disc(thickness=1.0, diameter=10.0, unit="mm", d_thickness=0.01, d_diameter=0.05)
    c = conductivity.conductivity(50.0, g, dR_ohm=0.5)
    L, A = 0.1, np.pi * 1.0 ** 2 / 4
    assert abs(c["sigma_S_cm"] - L / (50 * A)) < 1e-12
    rel = np.sqrt(0.01 ** 2 + 0.01 ** 2 + (2 * 0.005) ** 2)
    assert abs(c["dsigma_S_cm"] / c["sigma_S_cm"] - rel) < 1e-12


def test_linkk_flags_good_data():
    d = _load("synthetic_2arc.csv")[0][0]
    kk = validation.lin_kk(d.freq, d.Z)
    assert kk["mean_abs_res_pct"] < 1.0, kk["mean_abs_res_pct"]
    # residuals are normalised by |Z|: noise was 0.3 %
    assert 0.05 < kk["mean_abs_res_pct"], kk


def test_figures_and_exports(tmp_path=Path("/tmp")):
    d = _load("synthetic_2arc.csv")[0][0]
    res = _fit_truth()[3]
    rr, ri = ecm.relative_residuals(res.Z, res.Z_fit)
    fig = plotting.fig_ecm_composite(7.0, res.freq, res.Z, res.f_smooth, res.Z_smooth, res.Z_fit, rr, ri)
    for fmt in ("png", "pdf", "svg"):
        assert len(plotting.to_bytes(fig, fmt, dpi=150)) > 1000
    dr = drt.run_drt(d.freq, d.Z, drt.DRTSettings())
    dr_re, dr_im = ecm.relative_residuals(dr.Z, dr.Z_fit)
    fig2 = plotting.fig_drt_composite(7.0, dr.freq, dr.Z, dr.Z_fit, dr_re, dr_im, dr.tau, dr.gamma, peaks=dr.peaks)
    assert len(plotting.to_bytes(fig2, "png", dpi=100)) > 1000
    assert "Zre_fit_ohm" in export.ecm_spectrum_df(res).columns
    assert "gamma_ohm" in export.drt_gamma_df(dr).columns
    assert b"impedance" in export.settings_json("ecm", {"circuit": res.circuit})


if __name__ == "__main__":
    import inspect
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            fn()
            print("PASS", name)

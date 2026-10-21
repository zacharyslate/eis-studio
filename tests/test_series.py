"""Series analysis: reading mixed files, ionic/electronic conductivity, Arrhenius, normalisation.

Truth for the synthetic series files is documented in tests/make_samples.py."""
from pathlib import Path

import numpy as np
import pytest

from eisstudio import conductivity as cond
from eisstudio import series as S

D = Path(__file__).resolve().parents[1] / "eisstudio" / "sample_data" / "series"
TEMPS = [25, 40, 55, 70, 85]
KB = 8.617333262e-5
GEOM = cond.Geometry.disc(1.0, 10.0, "mm")


def true_R(Tc):
    A0 = 2e-5 * 298.15 * np.exp(0.45 / (KB * 298.15))
    T = Tc + 273.15
    return GEOM.thickness_cm / (A0 / T * np.exp(-0.45 / (KB * T)) * GEOM.area_cm2)


def load(names):
    return S.load_series([((D / n).read_bytes(), n) for n in names])


def test_linked_file_is_split():
    ms, rep = load(["LPSC_pellet_T-series_and_DC_linked.mpt"])
    kinds = [m.kind for m in ms]
    assert kinds == ["eis"] * 5 + ["ca"] * 5
    assert all(m.n == 69 for m in ms[:5]) and all(m.n == 901 for m in ms[5:])
    assert np.allclose([np.mean(m.E) for m in ms[5:]], [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-3)
    assert all(np.median(m.Z.imag) < 0 for m in ms[:5])        # -Im(Z) header converted to signed Z''


def test_separate_files_and_potential_steps():
    ms, _ = load([f"pellet_{t}C.csv" for t in TEMPS] + ["pellet_DC_polarisation.txt"])
    assert [m.kind for m in ms] == ["eis"] * 5 + ["ca"] * 5   # DC table split on potential jumps
    a, _ = load(["LPSC_pellet_T-series_and_DC_linked.mpt"])
    for x, y in zip(ms[:5], a[:5]):                            # same physics, independent noise
        assert abs(np.abs(x.Z).max() / np.abs(y.Z).max() - 1) < 0.05


def test_gamry_style_chronoamperometry_and_multiblock(tmp_path):
    t = np.arange(0, 600, 2.0)
    lines = ["EXPLAIN", "TAG\tCHRONOA", "CURVE\tTABLE", "\tPt\tT\tVf\tIm\tVu\tSig\tAch\tIERange\tOver",
             "\t#\ts\tV vs. Ref.\tA\tV\tV\tV\t#\tbits"]
    for i, tt in enumerate(t):
        lines.append(f"\t{i}\t{tt:.3f}\t2.000000E-01\t{3e-9 + 1e-9 * np.exp(-tt / 50):.6E}\t0\t0\t0\t7\t0")
    p = tmp_path / "ca.DTA"
    p.write_text("\n".join(lines) + "\n", encoding="latin-1")
    ms, _ = S.load_series([(p.read_bytes(), p.name)])
    assert len(ms) == 1 and ms[0].kind == "ca" and abs(np.mean(ms[0].E) - 0.2) < 1e-9
    st = S.steady_state(ms[0], 0.1)
    assert abs(st.I_ss - 3e-9) / 3e-9 < 0.01
    # two tables with their own headers in one CSV
    f = np.logspace(5, 0, 30)
    Z = 100 + 1 / (1j * 2 * np.pi * f * 1e-6)
    txt = "Frequency (Hz),Re (Ohm),Im (Ohm)\n" + "\n".join(f"{a},{b.real},{b.imag}" for a, b in zip(f, Z))
    txt += "\n\nTime (s),Potential (V),Current (uA)\n" + "\n".join(f"{tt},0.3,{0.01 + np.exp(-tt / 30)}" for tt in t)
    ms, _ = S.load_series([(txt.encode(), "two_tables.csv")])
    assert [m.kind for m in ms] == ["eis", "ca"]
    assert abs(S.steady_state(ms[1], 0.1).I_ss - 1e-8) / 1e-8 < 0.01   # µA unit read from the header


@pytest.mark.parametrize("method,tol", [("min", 0.02), ("spike", 0.04), ("ecm", 0.01)])
def test_ionic_resistance_and_activation_energy(method, tol):
    ms, _ = load(["LPSC_pellet_T-series_and_DC_linked.mpt"])
    res = [S.analyse_spectrum(m, T, method, GEOM, "p(R1,CPE1)-CPE2") for m, T in zip(ms[:5], TEMPS)]
    for r, T in zip(res, TEMPS):
        assert abs(r.R / true_R(T) - 1) < tol, (method, T, r.R, true_R(T), r.notes)
    arr = S.arrhenius(TEMPS, [r.sigma["sigma_S_cm"] for r in res])
    assert abs(arr["Ea_eV"] - 0.45) < 0.01
    assert abs(arr["sigma25_S_cm"] / 2e-5 - 1) < 0.03 and not arr["sigma25_extrapolated"]


def test_electronic_conductivity():
    ms, _ = load(["LPSC_pellet_T-series_and_DC_linked.mpt"])
    st = [S.steady_state(m, 0.1) for m in ms[5:]]
    assert all(s.drift_pct < 5 for s in st)
    fit = S.dc_fit([s.E for s in st], [s.I_ss for s in st])
    se = cond.conductivity(fit["R_e_ohm"], GEOM, fit["dR_e_ohm"])
    assert abs(se["sigma_S_cm"] / 2e-9 - 1) < 0.01
    # a step that has not settled is flagged
    m = ms[5]
    short = S.Measurement("ca", "x", "x", t=m.t[:40], I=m.I[:40], E=m.E[:40])
    assert S.steady_state(short, 0.2).notes


def test_density_normalisation_collapses_pellets():
    names = sorted(p.name for p in D.glob("density_*.csv"))
    ms, _ = load(names)
    R = []
    for m in ms:
        L, mass = [float(x) for x in m.source.split("_L")[1].replace(".csv", "").replace("mm_m", " ").replace("mg", "").split()]
        g = S.sample_geometry(GEOM, L, 10.0)
        _, rr = S.relative_density(g, mass, 1.86)
        k, unit = S.norm_factor("dense", g, rr, 1.5)
        r = S.analyse_spectrum(m, None, "min")
        R.append(r.R * k)
        assert unit == "Ω cm"
    assert np.ptp(R) / np.mean(R) < 0.03      # sigma = sigma_dense * rho_rel^1.5 by construction


def test_temperature_from_file_name():
    assert S.temperature_from_name("MOF_Bi_EIS_25C_150Kg_100mV.csv") == 25
    assert S.temperature_from_name("x_-20C.txt") == -20
    assert S.temperature_from_name("s_353.15K.csv") == 80
    assert S.temperature_from_name("a_25C_b_40C.csv") is None          # ambiguous
    assert S.temperature_from_name("LPSC_pellet_T-series_and_DC_linked.mpt") is None


def test_junction_ignores_a_single_noisy_point():
    ms, _ = load(["pellet_25C.csv"])
    m = ms[0]
    j0 = S.tail_start(m.freq, m.Z)
    Z = m.Z.copy()
    k = j0 + 4                                    # mains pickup on the spike side of the junction
    Z[k] = Z[k].real - 1j * 0.2 * abs(Z[k].imag)  # -Z'' drops to a fifth: a false local minimum
    assert abs(S.tail_start(m.freq, Z) - j0) <= 1
    assert abs(S.r_minimum(m.freq, Z)[0] / S.r_minimum(m.freq, m.Z)[0] - 1) < 0.01


def test_every_figure_function_is_locked_and_thread_safe():
    import threading

    from eisstudio import circuit_draw, plotting, series_plots
    funcs = [getattr(mod, n) for mod in (plotting, series_plots) for n in dir(mod)
             if n.startswith("fig_") and callable(getattr(mod, n))]
    funcs += [plotting.to_bytes, circuit_draw.circuit_figure]
    unlocked = [f.__name__ for f in funcs if not getattr(f, "_eis_locked", False)]
    assert not unlocked, unlocked
    ms, _ = load(["LPSC_pellet_T-series_and_DC_linked.mpt"])
    items = [{"label": f"{t} °C", "color": c, "f": m.freq, "Z": m.Z}
             for m, t, c in zip(ms[:5], TEMPS, series_plots.series_colors(TEMPS)[0])]
    errors = []

    def work():
        try:
            for _ in range(3):
                plotting.to_bytes(series_plots.fig_series_eis(7.0, items), "png", 60)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[:2]


def _squidstat_like(with_index_columns=True, noise_I=2e-9, G=2e-8, seed=3):
    """Admiral-Squidstat-style export: Segment, ActionId; EIS (21), open circuit (41, with the meaningless
    constant current such exports report), potentiostatic steps (3) with noisy E and I."""
    rng = np.random.default_rng(seed)
    rows, t, seg = [], 0.0, 0
    f = np.logspace(5, -1, 31)
    Z = 2000 / (1 + 1j * 2 * np.pi * f * 2000 * 1e-9) + 1 / (2e-6 * (1j * 2 * np.pi * f) ** 0.6)
    for fi, zi in zip(f, Z):
        t += 1
        rows.append((seg, t, 0.13, 1.0e-6, fi, abs(zi), zi.real, zi.imag, np.degrees(np.angle(zi)), 21))
    seg += 1
    for k in range(600):                                  # 10 min open circuit, drifting potential
        t += 1
        rows.append((seg, t, 0.06 - 3e-5 * k, 1.35e-6, 0, 0, 0, 0, 0, 41))
    for E in (0.1, 0.2, 0.3, 0.4, 0.5):
        seg += 1
        for k in range(1200):
            t += 1
            I = G * E + 5e-8 * E * np.exp(-k / 100) + noise_I * rng.standard_normal()
            Ek = E + 4e-4 * rng.standard_normal() + (3e-3 if rng.random() < 0.02 else 0.0)   # spikes > 2 mV
            rows.append((seg, t, Ek, I, 0, 0, 0, 0, 0, 3))
        seg += 1
        for k in range(60):                               # short rests between steps
            t += 1
            rows.append((seg, t, E - 0.02 * (1 - np.exp(-k / 10)), 1.4e-6, 0, 0, 0, 0, 0, 41))
    head = ["Segment", "Elapsed Time (s)", "Potential (V)", "Current (A)", "Frequency (Hz)", "|Z| (ohms)",
            "Zre (ohms)", "Zim (ohms)", "Phase of Z (deg)", "ActionId"]
    keep = range(len(head)) if with_index_columns else [i for i, h in enumerate(head) if h not in ("Segment", "ActionId")]
    lines = [",".join(head[i] for i in keep)] + [",".join(repr(float(r[i])) for i in keep) for r in rows]
    return ("\n".join(lines) + "\n").encode()


@pytest.mark.parametrize("index_columns", [True, False])
def test_dc_steps_are_not_fragmented_and_open_circuit_is_excluded(index_columns):
    ms, _ = S.load_series([(_squidstat_like(index_columns), "squid.csv")])
    kinds = [m.kind for m in ms]
    assert kinds.count("eis") == 1
    ca = [m for m in ms if m.kind == "ca"]
    assert len(ca) == 5, [(m.kind, m.summary()) for m in ms]
    assert all(m.n >= 1100 for m in ca)
    assert np.allclose([np.mean(m.E) for m in ca], [0.1, 0.2, 0.3, 0.4, 0.5], atol=2e-3)
    assert all(np.mean(m.I) < 1e-7 for m in ca)            # no open-circuit pseudo-current leaked in


def test_dc_fit_reports_resolution_and_upper_bound():
    ms, _ = S.load_series([(_squidstat_like(noise_I=2e-10), "a.csv")])
    st = [S.steady_state(m, 0.25) for m in ms if m.kind == "ca"]
    assert all(s.resolved for s in st)
    fit = S.dc_fit([s.E for s in st], [s.I_ss for s in st], False, [s.dI for s in st])
    assert fit["significant"] and abs(fit["G_S"] / 2e-8 - 1) < 0.1
    # current buried in noise: unresolved steps, slope not significant, upper bound above the truth
    ms, _ = S.load_series([(_squidstat_like(noise_I=2e-7, G=2e-9), "b.csv")])
    st = [S.steady_state(m, 0.1) for m in ms if m.kind == "ca"]
    assert not any(s.resolved for s in st)
    fit = S.dc_fit([s.E for s in st], [s.I_ss for s in st], False, [s.dI for s in st])
    assert not fit["significant"] and fit["R_e_ohm"] is None and fit["G_upper_S"] > 2e-9


def _blocking_with_small_process(n=70):
    """Bulk arc (100 kOhm at ~100 kHz) + a small resistive process near the real axis (5 %, 100 Hz)
    + a blocking CPE tail, sorted high -> low frequency."""
    f = np.logspace(6, -1, n)
    w = 2j * np.pi * f
    bulk = 1 / (1 / 1e5 + 1e-11 * w ** 0.95)
    small = 1 / (1 / 5e3 + (1 / (2 * np.pi * 100 * 5e3)) * w)
    tail = 1 / (2e-6 * w ** 0.7)
    return f, bulk + small + tail


def test_drt_cut_rules_keep_the_small_process_with_the_phase_rule():
    f, Z = _blocking_with_small_process()
    n_j, fc_j = S.drt_cut(f, Z, "junction")
    n_p, fc_p = S.drt_cut(f, Z, "phase", 15)
    n_a, fc_a = S.drt_cut(f, Z, "none")
    assert n_j < n_p < n_a == f.size and fc_a is None
    assert fc_p < 100 < fc_j          # the 100 Hz process is kept by the phase rule only
    ph = -np.degrees(np.angle(Z))
    assert ph[n_p - 1] <= 15 < ph[n_p]
    n_m, fc_m = S.drt_cut(f, Z, "phase", 15, f_min=10.0)
    assert fc_m >= 10 and f[n_m] < 10
    # no tail: everything kept whatever the rule
    Zb = 1 / (1 / 1e5 + 1e-11 * (2j * np.pi * f) ** 0.95)
    assert S.drt_cut(f, Zb, "junction") == (f.size, None)


def test_series_driver_fixed_parameters_starts_and_individual_settings():
    from eisstudio import drt as drtm
    ms, _ = load(["LPSC_pellet_T-series_and_DC_linked.mpt"])
    items = [(k, m, T) for k, (m, T) in enumerate(zip(ms[:5], TEMPS))]
    circ = "p(R1,CPE1)-CPE2"
    base = S.analyse_series(items, "ecm", circ, ["R1"], 2, 20, None, "data", kk=True,
                            drt_settings=drtm.DRTSettings(cv_type="custom", reg_param=1e-3, inductance="discard"))
    fixed = S.analyse_series(items, "ecm", circ, ["R1"], 2, 20, None, "chain", initial=None,
                             fixed={"CPE2_1": 0.9}, bounds={"CPE1_1": (0.5, 1.0)}, kk=False,
                             overrides={2: {"circuit": circ, "initial": {}, "fixed": {"CPE1_1": 0.95}, "bounds": {}},
                                        3: {"circuit": "R0-CPE1", "initial": {}, "fixed": {}, "bounds": {}}})
    for k, T in enumerate(TEMPS):
        b, x = base[k], fixed[k]
        assert abs(b.R / true_R(T) - 1) < 0.02, (T, b.R, b.notes)
        assert abs(x.R / b.R - 1) < 0.05, (T, x.R, b.R, x.notes)
        assert b.kk is not None and b.drt is not None and x.kk is None and x.drt is None
        assert b.drt_f_cut is not None and b.drt.freq.min() >= b.drt_f_cut   # tail cut off the DRT
        if k == 2:
            assert x.individual and x.fit.param("CPE1_1")[0] == 0.95 and "CPE2_1" not in x.fit.fixed
        else:
            assert not x.individual and x.fit.param("CPE2_1")[0] == 0.9
            assert 0.5 <= x.fit.param("CPE1_1")[0] <= 1.0


def test_effective_capacitance_survives_a_vanishing_exponent():
    from types import SimpleNamespace

    from eisstudio import ecm as ecmm
    vals = {"R1": 3e3, "CPE1_0": 1e-7, "CPE1_1": 0.004}
    res = SimpleNamespace(circuit="p(R1,CPE1)-CPE2", param=lambda n: (vals[n], 0.0))
    d = ecmm.derived_quantities(res)
    assert len(d) == 1 and not np.isfinite(d[0]["C_eff_F"])
    vals["CPE1_1"] = 0.9
    d = ecmm.derived_quantities(res)
    assert np.isclose(d[0]["C_eff_F"], 1e-7 ** (1 / 0.9) * 3e3 ** (0.1 / 0.9))


def _three_process_series(temps=(25, 40, 55, 70, 85), seed=7):
    """Bulk (Ea 0.45 eV, 5 pF), grain boundary (0.60 eV, 1 nF) and a blocking electrode (CPE) on a
    cell with L/A = 1 cm^-1, 1 MHz - 0.1 Hz, 0.3 % noise."""
    rng = np.random.default_rng(seed)
    f = np.logspace(6, -1, 71)
    w = 2j * np.pi * f
    out = []
    for T in temps:
        k = 1 / KB * (1 / 298.15 - 1 / (T + 273.15))
        Rb, Rgb = 1e6 * np.exp(-0.45 * k), 2e6 * np.exp(-0.60 * k)
        Z = (1 / (1 / Rb + 5e-12 * w ** 0.97) + 1 / (1 / Rgb + 1e-9 * w ** 0.9)
             + 1 / (1e-6 * w ** 0.8))
        Z = Z * (1 + 0.003 * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)))
        out.append(S.Measurement("eis", f"{T} C", "synthetic", freq=f, Z=Z))
    return out


def test_processes_are_tracked_classified_and_give_their_own_activation_energies():
    from eisstudio import drt as drtm
    from eisstudio import processes as P
    temps = (25, 40, 55, 70, 85)
    ms = _three_process_series(temps)
    items = [(k, m, T) for k, (m, T) in enumerate(zip(ms, temps))]
    res = S.analyse_series(items, "ecm", "p(R1,CPE1)-p(R2,CPE2)-CPE3", None, 4, 30, None, "data", kk=False,
                           drt_settings=drtm.DRTSettings(cv_type="custom", reg_param=1e-4, inductance="discard"))
    procs, notes = P.track([(k, T, res[k].fit) for k, _, T in items], cell_constant_per_cm=1.0)
    pairs = [p for p in procs if p.kind == "pair"]
    assert [p.cls for p in pairs] == ["bulk", "grain boundary"], [(p.label, p.cls) for p in pairs]
    assert P.suggested_ionic(procs) == [pairs[0].r_name, pairs[1].r_name]
    bulk, gb = pairs
    a = P.fit(bulk, "sigma", 1.0, times_T=False)
    assert abs(a["Ea_eV"] - 0.45) < 0.01, a["Ea_eV"]
    a = P.fit(gb, "inv_R", 1.0)
    assert abs(a["Ea_eV"] - 0.60) < 0.015, a["Ea_eV"]
    # specific gb conductivity uses the effective capacitances; for a CPE, C_eff = Q^(1/a) R^((1-a)/a)
    # follows R, which adds Ea_gb (1-a_gb)/a_gb - Ea_b (1-a_b)/a_b to the activation energy
    a = P.fit(gb, "sigma_gb", 1.0, bulk=bulk, times_T=False)
    expect = 0.60 + 0.60 * 0.1 / 0.9 - 0.45 * 0.03 / 0.97
    assert abs(a["Ea_eV"] - expect) < 0.02, (a["Ea_eV"], expect)
    # for a ZARC (R || CPE with constant Q) tau = (RQ)^(1/a), so f_c and the DRT peak give Ea_R / a
    assert abs(P.fit(bulk, "f_c", 1.0)["Ea_eV"] - 0.45 / 0.97) < 0.02
    assert abs(P.fit(gb, "f_c", 1.0)["Ea_eV"] - 0.60 / 0.9) < 0.02
    P.match_drt(procs, {k: res[k].drt for k, _, _ in items})
    assert len(gb.drt_tau) >= 4
    a = P.fit(gb, "drt", 1.0)
    assert abs(a["Ea_eV"] - 0.60 / 0.9) < 0.03, a["Ea_eV"]
    # the electrode element would be classed as such
    assert P.classify(1e-6, 1.0) == "electrode interface" and P.classify(5e-12, 1.0) == "bulk"

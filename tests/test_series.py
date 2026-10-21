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

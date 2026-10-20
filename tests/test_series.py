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

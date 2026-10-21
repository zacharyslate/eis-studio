"""Synthetic spectra with KNOWN parameters, written in several real export formats.

Truth (all files): R0=50 ohm, p(R1=200, CPE1: Q=1e-7, a=0.9), p(R2=500, CPE2: Q=1e-4, a=0.8)
(arc characteristic frequencies ~26 kHz and ~6.7 Hz, both inside the window), L=1e-7 H
noise: 0.3 % of |Z|, Gaussian, seed 1. 10 points/decade, 1 MHz - 10 mHz.
blocking_pellet: R0=40 + p(R1=150,CPE Q=1e-9 a=0.85) + blocking CPE (Q=2e-6, a=0.8), 7 MHz - 1 Hz.
(the R1 arc peaks at ~17 MHz, ABOVE the window: the realistic room-temperature sulfide-pellet case)
"""
import numpy as np
from pathlib import Path

out = Path(__file__).resolve().parents[1] / "eisstudio" / "sample_data"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(1)

def zcpe(Q, a, w):
    return 1 / (Q * (1j * w) ** a)

def par(a, b):
    return a * b / (a + b)

f = np.logspace(6, -2, 81)
w = 2 * np.pi * f
Z = 50 + par(200, zcpe(1e-7, 0.9, w)) + par(500, zcpe(1e-4, 0.8, w)) + 1j * w * 1e-7
Z = Z + 0.003 * np.abs(Z) * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))

# 1) plain CSV with header, signed Z''
np.savetxt(out / "synthetic_2arc.csv", np.c_[f, Z.real, Z.imag], delimiter=",",
           header="Frequency (Hz),Z' (Ohm),Z'' (Ohm)", comments="", fmt="%.8e")

# 2) BioLogic-style .mpt: latin-1, tab separated, DECIMAL COMMA, -Im(Z), header block
hdr = ["EC-Lab ASCII FILE", "Nb header lines : 6", "", "Synthetic file mimicking EC-Lab export",
       "Acquisition started on : 10/12/2026", ""]
cols = "freq/Hz\tRe(Z)/Ohm\t-Im(Z)/Ohm\t|Z|/Ohm\tPhase(Z)/deg\ttime/s\tcycle number"
lines = hdr[:5] + [cols]
for k in range(2):  # two cycles concatenated, as EC-Lab does
    for fi, zi in zip(f, Z * (1 + 0.02 * k)):
        vals = [fi, zi.real, -zi.imag, abs(zi), np.degrees(np.angle(zi)), 0.0, k + 1]
        lines.append("\t".join(f"{v:.6E}".replace(".", ",") for v in vals))
(out / "synthetic_2arc_biologic.mpt").write_text("\n".join(lines) + "\n", encoding="latin-1")

# 3) Gamry-style DTA (ZCURVE table)
dta = ["EXPLAIN", "TAG\tEISPOT", "TITLE\tLABEL\tPotentiostatic EIS\tTest &Identifier", "ZCURVE\tTABLE",
       "\tPt\tTime\tFreq\tZreal\tZimag\tZsig\tZmod\tZphz\tIdc\tVdc\tIERange",
       "\t#\ts\tHz\tohm\tohm\tV\tohm\t°\tA\tV\t#"]
for i, (fi, zi) in enumerate(zip(f, Z)):
    dta.append(f"\t{i}\t{i}\t{fi:.6E}\t{zi.real:.6E}\t{zi.imag:.6E}\t1\t{abs(zi):.6E}\t{np.degrees(np.angle(zi)):.4f}\t0\t0\t5")
(out / "synthetic_2arc_gamry.DTA").write_text("\n".join(dta) + "\n", encoding="latin-1")

# 4) |Z| + phase table, semicolon separated (Excel-EU style)
lines = ["Frequency [Hz];|Z| [Ohm];Phase [deg]"]
for fi, zi in zip(f, Z):
    lines.append(f"{fi:.6e};{abs(zi):.6e};{np.degrees(np.angle(zi)):.5f}".replace(".", ","))
(out / "synthetic_2arc_modphase.txt").write_text("\n".join(lines) + "\n")

# 5) blocking-electrode pellet (solid electrolyte style): headerless, -Z'' column
fb = np.logspace(np.log10(7e6), 0, 69)
wb = 2 * np.pi * fb
Zb = 40 + par(150, zcpe(1e-9, 0.85, wb)) + zcpe(2e-6, 0.8, wb)
Zb = Zb + 0.003 * np.abs(Zb) * (rng.standard_normal(fb.size) + 1j * rng.standard_normal(fb.size))
np.savetxt(out / "synthetic_blocking_pellet.txt", np.c_[fb, Zb.real, -Zb.imag], fmt="%.8e")
print("wrote", sorted(p.name for p in out.iterdir()))

# ---------------------------------------------------------------------------------------------
# Series samples (eisstudio/sample_data/series/). Truth:
#   pellet L = 1.00 mm, d = 10.0 mm (A = 0.7854 cm^2), mass 125.7 mg (rho = 1.60 g/cm^3)
#   sigma*T = A0 exp(-Ea/kT), Ea = 0.45 eV, sigma(25 C) = 2.0e-5 S/cm
#   bulk: p(R_b, CPE: alpha 0.95, C_eff = eps0*25*A/L), blocking CPE Q = 2e-6, alpha 0.85
#   electronic sigma_e = 2.0e-9 S/cm (R_e = 6.366e7 ohm); DC steps 0.1 ... 0.5 V, 1800 s each
#   different-density set: same powder, rho_rel 0.70 ... 0.90, sigma ~ sigma_dense * rho_rel^1.5
# ---------------------------------------------------------------------------------------------
sdir = out / "series"
sdir.mkdir(exist_ok=True)
kB, EPS0 = 8.617333262e-5, 8.8541878128e-14
L_cm, d_cm = 0.100, 1.00
A_cm2 = np.pi * d_cm ** 2 / 4
Ea, s25 = 0.45, 2.0e-5
A0 = s25 * 298.15 * np.exp(Ea / (kB * 298.15))


def sigma_T(Tc):
    T = Tc + 273.15
    return A0 / T * np.exp(-Ea / (kB * T))


def pellet_Z(fq, sig, L=L_cm, A=A_cm2, noise=0.003):
    w = 2 * np.pi * fq
    Rb = L / (sig * A)
    Ceff, a = EPS0 * 25 * A / L, 0.95
    Q = Ceff ** a * Rb ** (a - 1)
    Zs = par(Rb, zcpe(Q, a, w)) + zcpe(2e-6 * A / A_cm2, 0.85, w)
    return Zs + noise * np.abs(Zs) * (rng.standard_normal(fq.size) + 1j * rng.standard_normal(fq.size))


fs_ = np.logspace(np.log10(7e6), 0, 69)
temps = [25, 40, 55, 70, 85]
Re_ohm = L_cm / (2.0e-9 * A_cm2)
pots = [0.1, 0.2, 0.3, 0.4, 0.5]
t_step = np.arange(0, 1800.01, 2.0)


def ca_current(E, t):
    R25 = L_cm / (s25 * A_cm2)
    I = E / Re_ohm + E / R25 * 5e-4 * (0.6 * np.exp(-t / 20) + 0.4 * np.exp(-t / 200))
    return I * (1 + 0.01 * rng.standard_normal(t.size)) + 5e-12 * rng.standard_normal(t.size)


# 1) EC-Lab linked experiment (PEIS loop over temperature, then CA with 5 steps), one .mpt
cols = ["mode", "ox/red", "error", "control changes", "Ns changes", "counter inc.", "Ns", "freq/Hz",
        "Re(Z)/Ohm", "-Im(Z)/Ohm", "|Z|/Ohm", "Phase(Z)/deg", "time/s", "<Ewe>/V", "<I>/mA", "cycle number"]
rows, tnow = [], 0.0
for c, Tc in enumerate(temps):
    Zt = pellet_Z(fs_, sigma_T(Tc))
    for j, (fi, zi) in enumerate(zip(fs_, Zt)):
        tnow += 1.0
        rows.append([1, 0, 0, 0, int(j == 0), 0, 0, fi, zi.real, -zi.imag, abs(zi), np.degrees(np.angle(zi)),
                     tnow, 0.0, 0.0, c + 1])
    tnow += 1800.0                                   # thermal equilibration between temperatures
for k, E in enumerate(pots):
    I = ca_current(E, t_step)
    for j, (tt, ii) in enumerate(zip(t_step, I)):
        rows.append([2, 1, 0, int(j == 0), int(j == 0), 0, k, 0, 0, 0, 0, 0, tnow + tt,
                     E + 1e-4 * rng.standard_normal(), ii * 1e3, 1])
    tnow += t_step[-1] + 2
hdr = ["EC-Lab ASCII FILE", f"Nb header lines : 8", "", "Linked experiments: PEIS loop (25-85 C), CA 0.1-0.5 V",
       "Synthetic file mimicking an EC-Lab export", "Electrode material : Au | pellet | Au", "",
       "\t".join(cols)]
fmt = lambda v: f"{v:.6E}" if isinstance(v, float) else str(v)  # noqa: E731
(sdir / "LPSC_pellet_T-series_and_DC_linked.mpt").write_text(
    "\n".join(hdr + ["\t".join(fmt(v) for v in r) for r in rows]) + "\n", encoding="latin-1")

# 2) the same temperature study as separate files (one CSV per temperature) + DC steps as one table
for Tc in temps:
    Zt = pellet_Z(fs_, sigma_T(Tc))
    np.savetxt(sdir / f"pellet_{Tc}C.csv", np.c_[fs_, Zt.real, -Zt.imag], delimiter=",",
               header="Frequency (Hz),Z' (Ohm),-Z'' (Ohm)", comments="", fmt="%.8e")
dc = []
t0 = 0.0
for E in pots:
    I = ca_current(E, t_step)
    for tt, ii in zip(t_step, I):
        dc.append((t0 + tt, E + 1e-4 * rng.standard_normal(), ii))
    t0 += t_step[-1] + 2
np.savetxt(sdir / "pellet_DC_polarisation.txt", np.array(dc), delimiter="\t",
           header="Time (s)\tPotential (V)\tCurrent (A)", comments="", fmt="%.8e")

# 3) five pellets of the same powder pressed to different densities (25 C), separate files
rho_th = 1.86
for k, (rr, Lmm) in enumerate(zip([0.70, 0.75, 0.80, 0.85, 0.90], [1.30, 1.20, 1.12, 1.06, 1.00])):
    sig = 4.0e-5 * rr ** 1.5
    Zd = pellet_Z(fs_, sig, L=Lmm / 10)
    mass_mg = rr * rho_th * (Lmm / 10) * A_cm2 * 1000
    np.savetxt(sdir / f"density_{k + 1}_L{Lmm:.2f}mm_m{mass_mg:.1f}mg.csv", np.c_[fs_, Zd.real, Zd.imag],
               delimiter=",", header="Frequency (Hz),Z' (Ohm),Z'' (Ohm)", comments="", fmt="%.8e")
print("wrote series samples:", sorted(p.name for p in sdir.iterdir()))

# 4) polycrystalline pellet with a bulk and a grain-boundary arc, 25-85 C, separate files.
#    bulk: sigma_b(25 C) = 1.0e-5 S/cm, Ea = 0.40 eV, C_eff = eps0*30*A/L, alpha 0.95
#    grain boundary: R_gb = 3 R_b at 25 C, Ea = 0.55 eV, C_eff = 100 x bulk, alpha 0.90
#    blocking CPE Q = 2e-6, alpha 0.85; 10 MHz - 0.1 Hz, 0.3 % noise
fp = np.logspace(7, -1, 81)
wq = 2 * np.pi * fp
for T in temps:
    k = (1 / (kB * 298.15) - 1 / (kB * (T + 273.15)))
    Rb = L_cm / (1.0e-5 * A_cm2) * np.exp(-0.40 * k)
    Rgb = 3 * L_cm / (1.0e-5 * A_cm2) * np.exp(-0.55 * k)
    Cb = EPS0 * 30 * A_cm2 / L_cm
    Qb = Cb ** 0.95 * Rb ** (0.95 - 1)
    Qgb = (100 * Cb) ** 0.90 * Rgb ** (0.90 - 1)
    Zp = par(Rb, zcpe(Qb, 0.95, wq)) + par(Rgb, zcpe(Qgb, 0.90, wq)) + zcpe(2e-6, 0.85, wq)
    Zp = Zp + 0.003 * np.abs(Zp) * (rng.standard_normal(fp.size) + 1j * rng.standard_normal(fp.size))
    np.savetxt(sdir / f"polycrystal_{T}C.csv", np.c_[fp, Zp.real, Zp.imag], delimiter=",",
               header="Frequency (Hz),Z' (Ohm),Z'' (Ohm)", comments="", fmt="%.8e")
print("wrote polycrystal series")

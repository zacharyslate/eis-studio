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

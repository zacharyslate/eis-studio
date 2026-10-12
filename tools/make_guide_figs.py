"""Figures for the measurement guide (docs/measure.html), computed from circuit models.

    python tools/make_guide_figs.py && python tools/build_site.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_theory_figs import EPS0, OUT, LogLog, Nyq, pow10, sub, swatch, svg, txt  # noqa: E402

F_FARADAY = 96485.33212
R_GAS = 8.314462618


# --------------------------------------------------------------------------- #
# 1. Lead inductance on a pellet with a partly visible grain-boundary arc
# --------------------------------------------------------------------------- #
RB, RGB, QGB, AGB = 30.0, 40.0, 1.5e-8, 0.9
QE, AE = 2e-6, 0.85


def pellet_Z(f, L):
    w = 2 * np.pi * f
    return RB + RGB / (1 + RGB * QGB * (1j * w) ** AGB) + 1 / (QE * (1j * w) ** AE) + 1j * w * L


def fig_inductance():
    f = np.logspace(np.log10(7e6), 0, 900)
    F = Nyq(640, 20, 140, 46, padT=20, ymin=-40)
    F.H = int(F.padT + 60 + (F.ymax - F.ymin) * F.s)
    b = F.axes((20, 40, 60, 100, 120, 140), (-40, -20, 0, 20, 40))
    for L, cls, lab in ((0.0, "s1", "no lead inductance"), (0.2e-6, "s2", "0.2 µH"), (1e-6, "s3", "1 µH")):
        Z = pellet_Z(f, L)
        keep = (-Z.imag <= 46) & (-Z.imag >= -40) & (Z.real <= 140)
        b += F.path(Z[keep], "model " + cls)
    X, Y = F.X, F.Y
    b += f'<circle class="key" cx="{X(RB):.1f}" cy="{Y(0):.1f}" r="5"/>' + txt(X(RB) - 9, Y(0) - 9, "R" + sub("b"), "lab", "end")
    b += (f'<circle class="key" cx="{X(RB + RGB):.1f}" cy="{Y(0):.1f}" r="5"/>'
          + txt(X(RB + RGB) + 9, Y(0) + 22, "R" + sub("b") + " + R" + sub("gb"), "lab", "start"))
    z7 = pellet_Z(np.array([7e6]), 1e-6)[0]
    b += txt(X(z7.real) + 10, Y(-z7.imag) + 4, "7 MHz", "lab muted", "start")
    lx, ly = X(92), Y(-12)
    for i, (cls, lab) in enumerate((("model s1", "no lead inductance"), ("model s2", "L = 0.2 µH"), ("model s3", "L = 1 µH"))):
        b += swatch(lx, ly + 22 * i, cls, lab)
    return svg(F.W, F.H, b, "Effect of lead inductance on a pellet spectrum",
               "Three spectra of the same pellet: bulk 30 ohm, a grain-boundary arc of 40 ohm near 1.3 MHz, and blocking "
               "electrodes. With 0.2 and 1 microhenry of lead inductance the high-frequency end bends below the axis and "
               "the grain-boundary arc is distorted.", "g-ind")


# --------------------------------------------------------------------------- #
# 2. Measurement window: inductance and stray-capacitance limits vs real samples
# --------------------------------------------------------------------------- #
def fig_window():
    P = LogLog(640, 420, (1e-1, 1e8), (1e-2, 1e10), padL=82)
    f = np.logspace(-1, 8, 300)
    w = 2 * np.pi * f
    L = 1e-6
    Cs = 5e-12
    # zones
    zL = " L".join(f"{P.X(a):.1f},{P.Y(max(v, 1e-2)):.1f}" for a, v in zip(f, w * L))
    b = (f'<path class="zone" d="M{P.X(1e-1):.1f},{P.Y(1e-2):.1f} L{zL} L{P.X(1e8):.1f},{P.Y(1e-2):.1f} Z"/>')
    zC = " L".join(f"{P.X(a):.1f},{P.Y(min(v, 1e10)):.1f}" for a, v in zip(f, 1 / (w * Cs)))
    b += f'<path class="zone" d="M{P.X(1e-1):.1f},{P.Y(1e10):.1f} L{zC} L{P.X(1e8):.1f},{P.Y(1e10):.1f} Z"/>'
    b += f'<rect class="zone2" x="{P.X(7e6):.1f}" y="{P.padT}" width="{P.X(1e8) - P.X(7e6):.1f}" height="{P.H - P.padB - P.padT}"/>'
    b += P.frame([1e-1, 1e1, 1e3, 1e5, 1e7], [1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10], "Frequency / Hz", "|Z| / Ω",
                 xfmt=pow10, yfmt=pow10)
    def inside(xs, ys):
        m = (ys >= P.yr[0]) & (ys <= P.yr[1])
        return xs[m], ys[m]
    b += P.line(*inside(f, w * L), "guide") + P.line(*inside(f, 1 / (w * Cs)), "guide")
    # samples: disc 1 mm thick, 10 mm diameter, eps_r 30, blocking CPE
    K = 0.1 / (np.pi * 0.25)                       # cm^-1
    A, Lm = np.pi * 0.005 ** 2, 1e-3
    Cb = EPS0 * 30 * A / Lm
    for sig, cls, fl in ((1e-2, "s1", 3e5), (1e-4, "s2", 3e3), (1e-6, "s3", 1.5e2), (1e-8, "s4", 1.5e-1)):
        R = K / sig
        Z = R / (1 + 1j * w * R * Cb) + 1 / (QE * (1j * w) ** AE)
        b += P.line(*inside(f, np.abs(Z)), "model " + cls)
        b += txt(P.X(fl), P.Y(R) - 9, f"σ = {pow10(sig)} S cm⁻¹", "lab", "start")
    b += txt(P.X(2e-1), P.Y(4e-2) - 6, f"lead inductance dominates below |Z| = ωL (L = 1 µH)", "lab muted")
    b += txt(P.X(2e-1), P.Y(1e10) + 22, "stray capacitance dominates above |Z| = 1/(ωC) (C = 5 pF)", "lab muted")
    b += txt(P.X(8e6), P.Y(1e10) + 44, "above", "lab muted") + txt(P.X(8e6), P.Y(1e10) + 62, "7 MHz", "lab muted")
    return svg(P.W, P.H, b, "Where a measurement can be trusted",
               "Log-log plot of impedance magnitude against frequency. Shaded regions mark where 1 microhenry of lead "
               "inductance (bottom right) and 5 picofarad of stray capacitance (top right) dominate, and the region above "
               "7 MHz. Four pellets with conductivities from 1e-2 to 1e-8 S/cm are drawn; the most conductive one meets the "
               "inductance region at high frequency, the least conductive one meets the stray-capacitance region.", "g-win")


# --------------------------------------------------------------------------- #
# 3. Linearity: harmonic distortion of a Butler-Volmer interface vs amplitude
# --------------------------------------------------------------------------- #
def thd(amp_mV_rms, alpha, T=298.15):
    fF = F_FARADAY / (R_GAS * T)
    t = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    eta = amp_mV_rms * 1e-3 * np.sqrt(2) * np.sin(t)
    i = np.exp(alpha * fF * eta) - np.exp(-(1 - alpha) * fF * eta)
    sp = np.abs(np.fft.rfft(i))
    return np.sqrt(np.sum(sp[2:20] ** 2)) / sp[1]


def fig_linearity():
    amps = np.logspace(0, 2, 60)
    P = LogLog(640, 360, (1, 100), (1e-3, 100), padL=78)
    b = P.frame([1, 2, 5, 10, 20, 50, 100], [1e-3, 1e-2, 1e-1, 1, 10, 100], "Excitation amplitude / mV rms",
                "Harmonic distortion of the current / %", yfmt=lambda v: f"{v:g}".replace("e-0", "e−"), minor=False)
    for lvl in (1, 5):
        b += f'<line class="guide" x1="{P.padL}" y1="{P.Y(lvl):.1f}" x2="{P.W - P.padR}" y2="{P.Y(lvl):.1f}"/>'
        b += txt(P.W - P.padR - 6, P.Y(lvl) - 6, f"{lvl} %", "lab muted", "end")
    for a, cls in ((0.5, "s2"), (0.3, "s1")):
        y = np.array([100 * thd(x, a) for x in amps])
        m = y > 1e-3
        b += P.line(amps[m], y[m], "model " + cls)
    b += f'<line class="model s3" x1="{P.padL}" y1="{P.Y(1e-3) - 3:.1f}" x2="{P.W - P.padR}" y2="{P.Y(1e-3) - 3:.1f}"/>'
    lx, ly = P.X(1.1), P.Y(50)
    b += swatch(lx, ly, "model s1", "charge transfer, α = 0.3")
    b += swatch(lx, ly + 22, "model s2", "charge transfer, α = 0.5")
    b += swatch(lx, ly + 44, "model s3", "ohmic transport (no distortion)")
    return svg(P.W, P.H, b, "Non-linearity of an electrode reaction versus excitation amplitude",
               "Total harmonic distortion of the current through a Butler-Volmer interface at 25 C. With symmetric "
               "kinetics (alpha 0.5) it reaches 1 percent near 18 mV rms; with alpha 0.3 it is 5.5 percent already at "
               "10 mV rms. An ohmic resistor does not distort at any amplitude.", "g-lin")


# --------------------------------------------------------------------------- #
# 4. Wiring of a two-electrode pellet cell
# --------------------------------------------------------------------------- #
def fig_wiring():
    W, H = 640, 340
    cx, top, bot, half, e = 200, 140, 200, 120, 8
    b = (f'<rect class="cage" x="30" y="22" width="372" height="300" rx="6"/>'
         + txt(42, 310, "grounded Faraday cage, for high-impedance samples", "lab muted"))
    b += (f'<rect class="electrode" x="{cx - half + 14}" y="{top - e}" width="{2 * half - 28}" height="{e}"/>'
          f'<rect class="pellet" x="{cx - half}" y="{top}" width="{2 * half}" height="{bot - top}" rx="3"/>'
          f'<rect class="electrode" x="{cx - half + 14}" y="{bot}" width="{2 * half - 28}" height="{e}"/>'
          + txt(cx, (top + bot) / 2 + 6, "pellet", "lab", "middle"))
    inst = 486
    yWE, yWS, yRE, yCE = 66, 90, 250, 274
    b += (f'<rect class="box" x="{inst}" y="40" width="134" height="260" rx="6"/>'
          + txt(inst + 67, 150, "potentiostat /", "lab", "middle") + txt(inst + 67, 170, "impedance", "lab", "middle")
          + txt(inst + 67, 190, "analyser", "lab", "middle"))
    for name, y in (("WE", yWE), ("WS", yWS), ("RE", yRE), ("CE", yCE)):
        b += f'<circle class="node" cx="{inst}" cy="{y}" r="4"/>' + txt(inst + 12, y + 5, name, "lab")

    def twisted(x0, x1, ya, yb):
        mid, amp = (ya + yb) / 2, (yb - ya) / 2
        xs = np.linspace(x0, x1, 140)
        ph = (xs - x0) / 16 * np.pi
        d1 = "M" + " L".join(f"{x:.1f},{mid - amp * np.cos(p):.1f}" for x, p in zip(xs, ph))
        d2 = "M" + " L".join(f"{x:.1f},{mid + amp * np.cos(p):.1f}" for x, p in zip(xs, ph))
        return f'<path class="twist" d="{d1}"/><path class="twist b" d="{d2}"/>'
    xt = 300
    # current lead (ink) and sense lead (blue) from each electrode, then twisted to the instrument
    b += f'<path class="wire" d="M{cx - 30},{top - e} V{yWE} H{xt}"/>'
    b += f'<path class="wire sense" d="M{cx + 30},{top - e} V{yWS} H{xt}"/>'
    b += twisted(xt, inst, yWE, yWS)
    b += f'<path class="wire" d="M{cx - 30},{bot + e} V{yCE} H{xt}"/>'
    b += f'<path class="wire sense" d="M{cx + 30},{bot + e} V{yRE} H{xt}"/>'
    b += twisted(xt, inst, yRE, yCE)
    b += txt(cx - 36, (yWE + top) / 2 + 4, "current", "lab muted", "end") + txt(cx + 36, (yWS + top) / 2 + 8, "sense", "lab muted", "start")
    b += txt(444, 178, "short,", "lab muted", "middle") + txt(444, 196, "twisted", "lab muted", "middle")
    return svg(W, H, b, "Wiring a two-electrode pellet cell",
               "The working and working-sense leads both connect to the top electrode, and the counter and reference "
               "leads both connect to the bottom electrode. Each pair is twisted and kept short. The cell sits in a "
               "grounded Faraday cage.", "g-wire", "eqplot geom")


if __name__ == "__main__":
    figs = {"ind": fig_inductance(), "window": fig_window(), "lin": fig_linearity(), "wiring": fig_wiring()}
    for k, v in figs.items():
        (OUT / f"_gfig_{k}.html").write_text(v, encoding="utf-8")
    print("wrote", sorted(figs))
    for a in (0.5, 0.3):
        print("alpha", a, {m: round(100 * thd(m, a), 2) for m in (5, 10, 25, 50)})

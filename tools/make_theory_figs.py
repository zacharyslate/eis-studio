"""Generate the SVG figures for the theory page from exact models (and, for the DRT,
from EIS Studio's own DRT engine).

    python tools/make_theory_figs.py && python tools/build_site.py

Each figure is written to site_src/_fig_<name>.html and included in theory.html with
{{INCLUDE:_fig_<name>.html}}. Styling lives in docs/style.css (.eqplot ...).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site_src"
EPS0 = 8.8541878128e-12      # F/m, CODATA 2018
KB = 8.617333262e-5          # eV/K, CODATA 2018 (exact)

SUB = "<tspan baseline-shift='-0.22em' font-size='72%'>{}</tspan>"
SUP = "<tspan baseline-shift='0.45em' font-size='72%'>{}</tspan>"


def sub(t):
    return SUB.format(t)


def sup(t):
    return SUP.format(t)


def svg(W, H, body, title, desc, ident, cls="eqplot"):
    return (f'<svg class="{cls}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="{ident}-t {ident}-d">'
            f'<title id="{ident}-t">{title}</title><desc id="{ident}-d">{desc}</desc>{body}</svg>')


def txt(x, y, s, cls="lab", anchor="start", extra=""):
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{extra}>{s}</text>'


# --------------------------------------------------------------------------- #
# Circuit drawing (IEC style: resistor = box)
# --------------------------------------------------------------------------- #
def _element(kind, label, x, y):
    """Draw a two-terminal element centred vertically on y, starting at x. Returns (svg, width)."""
    if kind == "R":
        w = 56
        g = (f'<line class="wire" x1="{x}" y1="{y}" x2="{x + 8}" y2="{y}"/>'
             f'<rect class="part" x="{x + 8}" y="{y - 9}" width="40" height="18"/>'
             f'<line class="wire" x1="{x + 48}" y1="{y}" x2="{x + w}" y2="{y}"/>')
    elif kind == "C":
        w = 44
        g = (f'<line class="wire" x1="{x}" y1="{y}" x2="{x + 18}" y2="{y}"/>'
             f'<line class="plate" x1="{x + 18}" y1="{y - 14}" x2="{x + 18}" y2="{y + 14}"/>'
             f'<line class="plate" x1="{x + 26}" y1="{y - 14}" x2="{x + 26}" y2="{y + 14}"/>'
             f'<line class="wire" x1="{x + 26}" y1="{y}" x2="{x + w}" y2="{y}"/>')
    elif kind == "Q":  # constant phase element: chevron plates
        w = 46
        g = (f'<line class="wire" x1="{x}" y1="{y}" x2="{x + 19}" y2="{y}"/>'
             f'<path class="plate" d="M{x + 13},{y - 14} L{x + 19},{y} L{x + 13},{y + 14}"/>'
             f'<path class="plate" d="M{x + 23},{y - 14} L{x + 29},{y} L{x + 23},{y + 14}"/>'
             f'<line class="wire" x1="{x + 29}" y1="{y}" x2="{x + w}" y2="{y}"/>')
    else:
        raise ValueError(kind)
    g += txt(x + w / 2, y - 18, label, "clab", "middle")
    return g, w


def circuit(spec, x, y):
    """spec: list of ("R"|"C"|"Q", label) or ("par", [(kind,label),(kind,label)]). Returns (svg, width)."""
    out, cx = f'<circle class="node" cx="{x}" cy="{y}" r="3.5"/>', x
    for item in spec:
        out += f'<line class="wire" x1="{cx}" y1="{y}" x2="{cx + 14}" y2="{y}"/>'
        cx += 14
        if item[0] == "par":
            (k1, l1), (k2, l2) = item[1]
            dy = 26
            g1, w1 = _element(k1, l1, cx + 14, y - dy)
            g2, w2 = _element(k2, l2, cx + 14, y + dy)
            wb = max(w1, w2) + 28
            out += (f'<line class="wire" x1="{cx}" y1="{y - dy}" x2="{cx}" y2="{y + dy}"/>'
                    f'<line class="wire" x1="{cx + wb}" y1="{y - dy}" x2="{cx + wb}" y2="{y + dy}"/>'
                    f'<line class="wire" x1="{cx}" y1="{y - dy}" x2="{cx + 14}" y2="{y - dy}"/>'
                    f'<line class="wire" x1="{cx}" y1="{y + dy}" x2="{cx + 14}" y2="{y + dy}"/>'
                    f'<line class="wire" x1="{cx + 14 + w1}" y1="{y - dy}" x2="{cx + wb}" y2="{y - dy}"/>'
                    f'<line class="wire" x1="{cx + 14 + w2}" y1="{y + dy}" x2="{cx + wb}" y2="{y + dy}"/>')
            out += g1 + g2.replace('class="clab"', 'class="clab below"').replace(
                f'y="{y + dy - 18:.1f}"', f'y="{y + dy + 30:.1f}"')
            cx += wb
        else:
            g, w = _element(item[0], item[1], cx, y)
            out += g
            cx += w
    out += f'<line class="wire" x1="{cx}" y1="{y}" x2="{cx + 14}" y2="{y}"/><circle class="node" cx="{cx + 14}" cy="{y}" r="3.5"/>'
    return out, cx + 14 - x


# --------------------------------------------------------------------------- #
# Plot frames
# --------------------------------------------------------------------------- #
class Nyq:
    """Equal-scale Nyquist frame: one ohm is the same length on both axes."""

    def __init__(self, W, x0, x1, ymax, padL=64, padR=20, padT=22, padB=60, ymin=0.0):
        self.W, self.x0, self.padL, self.padT, self.padR = W, x0, padL, padT, padR
        self.s = (W - padL - padR) / (x1 - x0)
        self.ymax, self.ymin = ymax, ymin
        self.H = int(padT + padB + (ymax - ymin) * self.s)

    def X(self, v):
        return self.padL + (v - self.x0) * self.s

    def Y(self, v):
        return self.padT + (self.ymax - v) * self.s

    def path(self, Z, cls="model"):
        d = "M" + " L".join(f"{self.X(z.real):.1f},{self.Y(-z.imag):.1f}" for z in Z)
        return f'<path class="{cls}" d="{d}"/>'

    def axes(self, xticks, yticks, xlab="Z′ / Ω", ylab="−Z″ / Ω"):
        X, Y = self.X, self.Y
        y0 = Y(max(self.ymin, 0) if self.ymin > 0 else 0)
        out = (f'<g class="axes"><line x1="{self.padL}" y1="{Y(0):.1f}" x2="{self.W - self.padR}" y2="{Y(0):.1f}"/>'
               f'<line x1="{self.padL}" y1="{Y(self.ymin):.1f}" x2="{self.padL}" y2="{self.padT - 6}"/>')
        for v in xticks:
            out += (f'<line x1="{X(v):.1f}" y1="{Y(0):.1f}" x2="{X(v):.1f}" y2="{Y(0) + 6:.1f}"/>'
                    + txt(X(v), Y(0) + 23, v, "tick", "middle"))
        for v in yticks:
            out += (f'<line x1="{self.padL - 6}" y1="{Y(v):.1f}" x2="{self.padL}" y2="{Y(v):.1f}"/>'
                    + txt(self.padL - 10, Y(v) + 5, v, "tick", "end"))
        mid = (self.padT + Y(self.ymin)) / 2
        out += txt((self.padL + self.W - self.padR) / 2, Y(0) + 50, xlab, "axlab", "middle")
        out += txt(18, mid, ylab, "axlab", "middle", f' transform="rotate(-90 18 {mid:.0f})"')
        return out + "</g>"


class LogLog:
    def __init__(self, W, H, xr, yr, padL=78, padR=24, padT=24, padB=60, xlog=True, ylog=True):
        self.W, self.H, self.padL, self.padR, self.padT, self.padB = W, H, padL, padR, padT, padB
        self.xr, self.yr, self.xlog, self.ylog = xr, yr, xlog, ylog

    def _t(self, v, r, log):
        a, b = (np.log10(r[0]), np.log10(r[1])) if log else r
        v = np.log10(v) if log else v
        return (v - a) / (b - a)

    def X(self, v):
        return self.padL + self._t(v, self.xr, self.xlog) * (self.W - self.padL - self.padR)

    def Y(self, v):
        return self.H - self.padB - self._t(v, self.yr, self.ylog) * (self.H - self.padT - self.padB)

    def line(self, xs, ys, cls="model"):
        d = "M" + " L".join(f"{self.X(a):.1f},{self.Y(b):.1f}" for a, b in zip(xs, ys))
        return f'<path class="{cls}" d="{d}"/>'

    def frame(self, xticks, yticks, xlab, ylab, xfmt=None, yfmt=None, minor=True):
        L, R, T, B = self.padL, self.W - self.padR, self.padT, self.H - self.padB
        out = f'<g class="axes"><rect class="frame" x="{L}" y="{T}" width="{R - L}" height="{B - T}"/>'
        for v in xticks:
            out += f'<line x1="{self.X(v):.1f}" y1="{B}" x2="{self.X(v):.1f}" y2="{B + 6}"/>'
            out += txt(self.X(v), B + 24, xfmt(v) if xfmt else v, "tick", "middle")
        for v in yticks:
            out += f'<line x1="{L - 6}" y1="{self.Y(v):.1f}" x2="{L}" y2="{self.Y(v):.1f}"/>'
            out += txt(L - 10, self.Y(v) + 5, yfmt(v) if yfmt else v, "tick", "end")
        if minor and self.xlog:
            for dec in range(int(np.floor(np.log10(self.xr[0]))), int(np.ceil(np.log10(self.xr[1])))):
                for m in range(2, 10):
                    v = m * 10.0 ** dec
                    if self.xr[0] < v < self.xr[1]:
                        out += f'<line class="minor" x1="{self.X(v):.1f}" y1="{B}" x2="{self.X(v):.1f}" y2="{B + 3}"/>'
        out += txt((L + R) / 2, self.H - 12, xlab, "axlab", "middle")
        mid = (T + B) / 2
        out += txt(18, mid, ylab, "axlab", "middle", f' transform="rotate(-90 18 {mid:.0f})"')
        return out + "</g>"


def sci(v, sig=2):
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    ms = f"{m:.{sig}g}"
    if ms == "10":
        ms, e = "1", e + 1
    es = str(e).replace("-", "−")
    return f"{ms} × 10{sup(es)}"


def swatch(x, y, cls, label):
    return (f'<line class="{cls}" x1="{x}" y1="{y}" x2="{x + 34}" y2="{y}"/>'
            + txt(x + 42, y + 5, label, "lab"))


def pow10(v):
    e = int(round(np.log10(v)))
    return f"10{sup(e if e >= 0 else '−' + str(-e))}"


# --------------------------------------------------------------------------- #
# 1. AC excitation and response in time
# --------------------------------------------------------------------------- #
def fig_ac():
    W, H, L, R, mid, amp = 640, 280, 56, 620, 150, 70
    phi = np.deg2rad(35)
    t = np.linspace(0, 2.2 * 2 * np.pi, 600)
    X = lambda a: L + (a / t[-1]) * (R - L)  # noqa: E731
    v = " L".join(f"{X(a):.1f},{mid - amp * np.sin(a):.1f}" for a in t)
    i = " L".join(f"{X(a):.1f},{mid - 0.62 * amp * np.sin(a + phi):.1f}" for a in t)
    b = (f'<g class="axes"><line x1="{L}" y1="{mid}" x2="{R}" y2="{mid}"/><line x1="{L}" y1="{mid - amp - 16}" x2="{L}" y2="{mid + amp + 16}"/></g>'
         + txt(R, mid + 22, "time", "axlab", "end"))
    b += f'<path class="model" d="M{v}"/><path class="emph" d="M{i}"/>'
    # peak markers for phase shift: first peaks
    tv = np.pi / 2
    ti = np.pi / 2 - phi
    yv, yi = mid - amp, mid - 0.62 * amp
    b += (f'<line class="guide" x1="{X(tv):.1f}" y1="{yv:.1f}" x2="{X(tv):.1f}" y2="{mid + amp + 10:.1f}"/>'
          f'<line class="guide" x1="{X(ti):.1f}" y1="{yi:.1f}" x2="{X(ti):.1f}" y2="{mid + amp + 10:.1f}"/>'
          f'<g class="dim"><line x1="{X(ti):.1f}" y1="{mid + amp + 4:.1f}" x2="{X(tv):.1f}" y2="{mid + amp + 4:.1f}"/></g>'
          + txt((X(ti) + X(tv)) / 2, mid + amp + 26, "φ / ω", "lab", "middle"))
    b += swatch(R - 250, 24, "model", "voltage  V(t) = V" + sub("0") + " sin ωt")
    b += swatch(R - 250, 50, "emph", "current  I(t) = I" + sub("0") + " sin(ωt + φ)")
    return svg(W, H, b, "Voltage excitation and current response",
               "A sine-wave voltage and the resulting current, smaller and shifted earlier in time by the phase angle phi "
               "divided by the angular frequency.", "f-ac")


# --------------------------------------------------------------------------- #
# 2. R0 + (R1 || C1): Nyquist with circuit, and Bode phase
# --------------------------------------------------------------------------- #
R0, R1, C1 = 20.0, 200.0, 1e-9


def fig_rc():
    w = 2 * np.pi * np.logspace(9, 2, 500)
    Z = R0 + R1 / (1 + 1j * w * R1 * C1)
    F = Nyq(640, 0, 260, 150)
    X, Y = F.X, F.Y
    apex = R0 + R1 / 2
    fc = 1 / (2 * np.pi * R1 * C1)
    b = F.axes(range(0, 261, 40), (0, 40, 80, 120))
    b += F.path(Z)
    for v, lab, dx, anc in ((R0, "R" + sub("0"), -10, "end"), (R0 + R1, "R" + sub("0") + " + R" + sub("1"), 10, "start")):
        b += f'<circle class="key" cx="{X(v):.1f}" cy="{Y(0):.1f}" r="5"/>' + txt(X(v) + dx, Y(0) - 10, lab, "lab", anc)
    b += f'<line class="guide" x1="{X(apex):.1f}" y1="{Y(0):.1f}" x2="{X(apex):.1f}" y2="{Y(R1 / 2):.1f}"/>'
    b += f'<circle class="key emphfill" cx="{X(apex):.1f}" cy="{Y(R1 / 2):.1f}" r="6"/>'
    b += txt(X(apex), Y(R1 / 2) - 16, f"ωτ = 1  (f = {fc / 1e3:.0f} kHz)", "lab emphtext", "middle")
    b += txt(X(apex) + 8, Y(R1 / 4) + 5, "R" + sub("1") + "/2", "lab muted")
    b += txt(X(46), Y(14), "← high f", "lab muted", "start") + txt(X(214), Y(14), "low f →", "lab muted", "end")
    c, cw = circuit([("R", "R" + sub("0")), ("par", [("R", "R" + sub("1")), ("C", "C" + sub("1"))])], 0, 0)
    b += f'<g transform="translate({F.W - cw - 30},{F.padT + 40})">{c}</g>'
    return svg(F.W, F.H, b, "Nyquist plot of R0 in series with a parallel R1-C1 element",
               f"A semicircle from R0 = {R0:.0f} ohm to R0 + R1 = {R0 + R1:.0f} ohm. Its top, at omega tau = 1 "
               f"({fc / 1e3:.0f} kHz), has height R1/2. The circuit diagram is drawn in the corner.", "f-rc")


def fig_rc_bode():
    f = np.logspace(2, 9, 400)
    w = 2 * np.pi * f
    Z = R0 + R1 / (1 + 1j * w * R1 * C1)
    ph = -np.degrees(np.angle(Z))
    fc = 1 / (2 * np.pi * R1 * C1)
    P = LogLog(640, 260, (1e2, 1e9), (0, 70), ylog=False)
    b = P.frame([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9], [0, 20, 40, 60], "Frequency / Hz", "−Phase / °", xfmt=pow10)
    b += P.line(f, ph)
    i = int(np.argmax(ph))
    fpk = f[i]
    b += f'<line class="guide" x1="{P.X(fc):.1f}" y1="{P.Y(0):.1f}" x2="{P.X(fc):.1f}" y2="{P.Y(68):.1f}"/>'
    b += txt(P.X(fc) - 8, P.Y(62), f"f = 1/(2πR" + sub("1") + "C" + sub("1") + f") = {fc / 1e3:.0f} kHz", "lab", "end")
    b += f'<circle class="key emphfill" cx="{P.X(fpk):.1f}" cy="{P.Y(ph[i]):.1f}" r="6"/>'
    b += txt(P.X(fpk) + 12, P.Y(ph[i]) - 6, f"phase peak at {fpk / 1e6:.1f} MHz", "lab emphtext")
    return svg(P.W, P.H, b, "Bode phase plot of the same circuit",
               f"Minus the phase angle versus frequency. It peaks at {fpk / 1e3:.0f} kHz, above the characteristic "
               f"frequency of the RC element ({fc / 1e3:.0f} kHz) because the series resistance R0 shifts the phase peak.",
               "f-rcb"), fpk


# --------------------------------------------------------------------------- #
# 3. CPE: depressed arcs
# --------------------------------------------------------------------------- #
def fig_cpe():
    R, tau = 100.0, 1e-4
    F = Nyq(640, 0, 110, 58, padT=26)
    b = F.axes(range(0, 101, 20), (0, 20, 40))
    shades = ["s1", "s2", "s3", "s4"]
    for a, cls in zip((1.0, 0.9, 0.8, 0.7), shades):
        Q = tau ** a / R
        w = 2 * np.pi * np.logspace(9, -3, 700)
        Z = R / (1 + R * Q * (1j * w) ** a)
        b += F.path(Z, "model " + cls)
        k = int(np.argmax(-Z.imag))
        b += txt(F.X(Z.real[k]), F.Y(-Z.imag[k]) + 21, f"α = {a:g}", "lab", "middle")
    return svg(F.W, F.H, b, "Impedance arcs of R in parallel with a constant phase element",
               "Four arcs with the same resistance R and time constant. As the exponent alpha decreases from 1 to 0.7 "
               "the arc is depressed below the real axis centre and becomes flatter.", "f-cpe")


# --------------------------------------------------------------------------- #
# 4. Geometry
# --------------------------------------------------------------------------- #
def fig_geometry():
    W, H = 640, 300
    cx, top, bot = 250, 96, 172
    half = 170
    e = 9
    b = (f'<rect class="electrode" x="{cx - half + 16}" y="{top - e}" width="{2 * half - 32}" height="{e}"/>'
         f'<rect class="pellet" x="{cx - half}" y="{top}" width="{2 * half}" height="{bot - top}" rx="3"/>'
         f'<rect class="electrode" x="{cx - half + 16}" y="{bot}" width="{2 * half - 32}" height="{e}"/>')
    bx = cx + half + 70
    b += (f'<path class="wire" d="M{cx},{top - e} V{top - 40} H{bx}"/>'
          f'<path class="wire" d="M{cx},{bot + e} V{bot + 40} H{bx}"/>'
          f'<rect class="box" x="{bx}" y="{top - 58}" width="122" height="{bot - top + 116}" rx="4"/>'
          + txt(bx + 61, (top + bot) / 2 - 4, "impedance", "lab", "middle")
          + txt(bx + 61, (top + bot) / 2 + 18, "analyser", "lab", "middle"))
    xL = cx - half - 30
    b += (f'<g class="dim"><line x1="{xL}" y1="{top}" x2="{xL}" y2="{bot}"/>'
          f'<line x1="{xL - 6}" y1="{top}" x2="{xL + 6}" y2="{top}"/><line x1="{xL - 6}" y1="{bot}" x2="{xL + 6}" y2="{bot}"/></g>'
          + txt(xL - 12, (top + bot) / 2 + 8, "L", "lab big", "end"))
    yd = bot + e + 62
    x1, x2 = cx - half + 16, cx + half - 16
    b += (f'<g class="dim"><line x1="{x1}" y1="{yd}" x2="{x2}" y2="{yd}"/>'
          f'<line x1="{x1}" y1="{yd - 6}" x2="{x1}" y2="{yd + 6}"/><line x1="{x2}" y1="{yd - 6}" x2="{x2}" y2="{yd + 6}"/></g>'
          f'<line class="guide" x1="{x1}" y1="{bot + e}" x2="{x1}" y2="{yd}"/><line class="guide" x1="{x2}" y1="{bot + e}" x2="{x2}" y2="{yd}"/>'
          + txt(cx, yd + 26, "electrode diameter d,  A = πd²/4", "lab", "middle"))
    b += txt(cx, (top + bot) / 2 + 6, "solid electrolyte pellet", "lab", "middle")
    b += txt(cx - half + 18, top - e - 12, "ion-blocking electrode", "lab muted")
    return svg(W, H, b, "Measurement geometry",
               "Side view of a pellet of thickness L between two ion-blocking electrodes of diameter d, wired to an "
               "impedance analyser.", "f-geo", "eqplot geom")


# --------------------------------------------------------------------------- #
# 5. Polycrystalline pellet spectrum with circuit
# --------------------------------------------------------------------------- #
def fig_pellet():
    Rb, Cb = 100.0, 2e-11
    Rgb, Cgb = 150.0, 1e-9
    Q, a = 2e-6, 0.85
    f = np.logspace(10, -0.3, 1200)
    w = 2 * np.pi * f
    Z = Rb / (1 + 1j * w * Rb * Cb) + Rgb / (1 + 1j * w * Rgb * Cgb) + 1 / (Q * (1j * w) ** a)
    F = Nyq(640, 0, 400, 230, padT=26)
    keep = -Z.imag <= 222
    Zk, fk = Z[keep], f[keep]
    b = F.axes(range(0, 401, 100), (0, 100, 200))
    hi = fk >= 7e6
    k = int(np.argmax(~hi))
    b += F.path(Zk[: k + 1], "model unmeasured") + F.path(Zk[k:], "model")
    X, Y = F.X, F.Y
    fb, fg = 1 / (2 * np.pi * Rb * Cb), 1 / (2 * np.pi * Rgb * Cgb)
    b += txt(X(50), Y(86), "bulk", "lab", "middle") + txt(X(50), Y(86) + 19, f"20 pF, {fb / 1e6:.0f} MHz", "lab muted", "middle")
    b += txt(X(175), Y(114), "grain boundary", "lab", "middle") + txt(X(175), Y(114) + 19, f"1 nF, {fg / 1e6:.1f} MHz", "lab muted", "middle")
    b += txt(X(316), Y(150), "electrodes", "lab", "start") + txt(X(316), Y(150) + 19, "CPE, ~µF", "lab muted", "start")
    for v, lab, dx, anc in ((Rb, "R" + sub("b"), 0, "middle"), (Rb + Rgb, "R" + sub("b") + " + R" + sub("gb"), -12, "end")):
        b += f'<circle class="key" cx="{X(v):.1f}" cy="{Y(0):.1f}" r="5"/>' + txt(X(v) + dx, Y(0) - 13, lab, "lab", anc)
    b += (f'<line class="model unmeasured" x1="{X(288):.1f}" y1="{Y(52):.1f}" x2="{X(310):.1f}" y2="{Y(52):.1f}"/>'
          + txt(X(315), Y(52) + 5, "f > 7 MHz", "lab muted"))
    c, cw = circuit([("par", [("R", "R" + sub("b")), ("C", "C" + sub("b"))]),
                     ("par", [("R", "R" + sub("gb")), ("C", "C" + sub("gb"))]), ("Q", "CPE")], 0, 0)
    b += f'<g transform="translate({X(8):.0f},{Y(182):.0f})">{c}</g>'
    return svg(F.W, F.H, b, "Model spectrum of a polycrystalline pellet between ion-blocking electrodes",
               f"A small bulk arc (100 ohm, 20 pF, top at {fb / 1e6:.0f} MHz, dashed because it lies above 7 MHz), a "
               f"grain-boundary arc (150 ohm, 1 nF, {fg / 1e6:.1f} MHz), then a steep electrode spike. The circuit "
               "is two parallel R-C elements and a constant phase element in series.", "f-pel")


# --------------------------------------------------------------------------- #
# 6. Bulk relaxation frequency versus conductivity
# --------------------------------------------------------------------------- #
def fig_bulkfreq():
    sig = np.logspace(-9, -1, 50)                   # S/cm
    P = LogLog(640, 380, (1e-9, 1e-1), (1e-1, 1e10), padL=82)
    xt = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
    yt = [1e-1, 1e1, 1e3, 1e5, 1e7, 1e9]
    b = f'<rect class="zone" x="{P.padL}" y="{P.Y(1e10):.1f}" width="{P.W - P.padL - P.padR}" height="{P.Y(7e6) - P.Y(1e10):.1f}"/>'
    b += P.frame(xt, yt, "Conductivity σ / S cm⁻¹", "Bulk arc frequency f / Hz", xfmt=pow10, yfmt=pow10)
    b += txt(P.padL + 10, P.Y(1e10) + 22, "above 7 MHz: bulk arc not measured by many analysers", "lab muted")
    b += f'<circle class="key emphfill" cx="{P.padL + 16:.1f}" cy="{P.Y(1e10) + 41:.1f}" r="6"/>'
    b += txt(P.padL + 28, P.Y(1e10) + 46, f"1 mS cm⁻¹ with ε{sub('r')} = 30 → 60 MHz", "lab")
    b += f'<line class="guide" x1="{P.padL}" y1="{P.Y(7e6):.1f}" x2="{P.W - P.padR}" y2="{P.Y(7e6):.1f}"/>'
    lx, ly = P.X(3e-5), P.Y(3e1)
    for i, (er, cls) in enumerate(((10, "s2"), (30, "s1"), (100, "s3"))):
        fb = sig * 100 / (2 * np.pi * EPS0 * er)
        b += P.line(sig, fb, "model " + cls)
        b += swatch(lx, ly + 24 * i, "model " + cls, f"ε{sub('r')} = {er}")
    s1 = 1e-3
    f1 = s1 * 100 / (2 * np.pi * EPS0 * 30)
    b += f'<circle class="key emphfill" cx="{P.X(s1):.1f}" cy="{P.Y(f1):.1f}" r="6"/>'

    return svg(P.W, P.H, b, "Bulk relaxation frequency versus conductivity",
               "Log-log lines f = sigma / (2 pi epsilon0 epsilon_r) for relative permittivities 10, 30 and 100. "
               f"At 1 mS/cm and epsilon_r = 30 the bulk arc peaks at {f1 / 1e6:.0f} MHz, inside the shaded region above "
               "7 MHz.", "f-bf")


# --------------------------------------------------------------------------- #
# 7. Arrhenius plot
# --------------------------------------------------------------------------- #
ARR_T_C = np.array([25.0, 40.0, 55.0, 70.0, 85.0, 100.0])
_ARR_EA, _ARR_A = 0.30, 2.0e3     # eV, S K / cm (generating values for the example data)
_rng = np.random.default_rng(7)
ARR_SIG = _ARR_A / (ARR_T_C + 273.15) * np.exp(-_ARR_EA / (KB * (ARR_T_C + 273.15))) * \
    np.exp(_rng.normal(0, 0.03, ARR_T_C.size))
ARR_SIG = np.array([float(f"{s:.3g}") for s in ARR_SIG])          # rounded as a user would type them


def arrhenius_fit(T_C, sig):
    T = T_C + 273.15
    x, y = 1000 / T, np.log(sig * T)
    n = x.size
    sx = np.sum((x - x.mean()) ** 2)
    m = np.sum((x - x.mean()) * (y - y.mean())) / sx
    c = y.mean() - m * x.mean()
    res = y - (m * x + c)
    se = np.sqrt(np.sum(res ** 2) / (n - 2) / sx)
    r2 = 1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)
    Ea = -m * 1000 * KB
    return Ea, se * 1000 * KB, np.exp(c), r2, m, c


def fig_arrhenius():
    Ea, dEa, A, r2, m, c = arrhenius_fit(ARR_T_C, ARR_SIG)
    T = ARR_T_C + 273.15
    x = 1000 / T
    y = np.log10(ARR_SIG * T)
    P = LogLog(640, 330, (2.6, 3.4), (np.floor(y.min() * 2) / 2 - 0.1, np.ceil(y.max() * 2) / 2 + 0.1),
               xlog=False, ylog=False, padT=44)
    yt = np.arange(np.ceil(P.yr[0] * 2) / 2, P.yr[1], 0.5)
    b = P.frame([2.6, 2.8, 3.0, 3.2, 3.4], [round(v, 1) for v in yt], "1000 / T  (K⁻¹)", "log₁₀(σT / S K cm⁻¹)",
                yfmt=lambda v: f"{v:.1f}".replace("-", "−"), minor=False)
    xx = np.linspace(2.62, 3.38, 50)
    b += P.line(xx, (m * xx + c) / np.log(10), "emph")
    for xi, yi in zip(x, y):
        b += f'<circle class="pt" cx="{P.X(xi):.1f}" cy="{P.Y(yi):.1f}" r="5"/>'
    for tc in (100, 75, 50, 25):
        xv = 1000 / (tc + 273.15)
        b += f'<line x1="{P.X(xv):.1f}" y1="{P.padT}" x2="{P.X(xv):.1f}" y2="{P.padT - 6}" class="axisline"/>'
        b += txt(P.X(xv), P.padT - 11, f"{tc} °C", "tick", "middle")
    yb = P.yr[0] + 0.32 * (P.yr[1] - P.yr[0])
    b += txt(P.X(2.64), P.Y(yb), f"slope = −E{sub('a')} / (1000 k{sub('B')} ln 10)", "lab")
    b += txt(P.X(2.64), P.Y(yb) + 24, f"E{sub('a')} = {Ea:.3f} ± {dEa:.3f} eV", "lab emphtext")
    return svg(P.W, P.H, b, "Arrhenius plot",
               f"Six conductivity points between 25 and 100 degrees C fall on a straight line of log(sigma T) versus "
               f"1000/T. The fitted activation energy is {Ea:.3f} plus or minus {dEa:.3f} eV.", "f-arr"), (Ea, dEa, A, r2)


# --------------------------------------------------------------------------- #
# 8. DRT of the two-arc sample, computed by EIS Studio
# --------------------------------------------------------------------------- #
def fig_drt():
    sys.path.insert(0, str(ROOT))
    from eisstudio import drt, io
    raw = (ROOT / "eisstudio" / "sample_data" / "synthetic_2arc.csv").read_bytes()
    d = io.load_spectrum(raw, "synthetic_2arc.csv")[0][0]
    r = drt.run_drt(d.freq, d.Z, drt.DRTSettings())
    tau, g = r.tau, r.gamma
    big = sorted([p for p in r.peaks if not p["minor"]], key=lambda p: p["tau_s"])
    P = LogLog(640, 330, (tau.min(), tau.max()), (0, g.max() * 1.42), ylog=False, padT=24)
    decs = [10.0 ** k for k in range(int(np.ceil(np.log10(tau.min()))), int(np.floor(np.log10(tau.max()))) + 1, 2)]
    b = ""
    for p in big:
        m = (tau >= p["tau_lo_s"]) & (tau <= p["tau_hi_s"])
        pts = " L".join(f"{P.X(a):.1f},{P.Y(v):.1f}" for a, v in zip(tau[m], g[m]))
        b += (f'<path class="area" d="M{P.X(tau[m][0]):.1f},{P.Y(0):.1f} L{pts} '
              f'L{P.X(tau[m][-1]):.1f},{P.Y(0):.1f} Z"/>')
    b += P.frame(decs, [0, 100, 200, 300], "Relaxation time τ / s", "γ(ln τ) / Ω", xfmt=pow10, minor=False)
    b += P.line(tau, g)
    for p in big:
        b += txt(P.X(p["tau_s"]), P.Y(p["gamma_max_ohm"]) - 30, f"τ = {sci(p['tau_s'])} s", "lab", "middle")
        b += txt(P.X(p["tau_s"]), P.Y(p["gamma_max_ohm"]) - 10, f"area = {p['R_ohm']:.0f} Ω", "lab emphtext", "middle")
    return svg(P.W, P.H, b, "Distribution of relaxation times of the two-arc sample",
               f"Gamma versus tau computed by EIS Studio. Two main peaks at tau = {big[0]['tau_s']:.2g} s and "
               f"{big[1]['tau_s']:.2g} s; their shaded areas, {big[0]['R_ohm']:.0f} and {big[1]['R_ohm']:.0f} ohm, are the "
               "resistances of the two processes (true values 200 and 500 ohm).", "f-drt"), big


if __name__ == "__main__":
    figs = {"ac": fig_ac(), "rc": fig_rc(), "cpe": fig_cpe(), "geometry": fig_geometry(),
            "pellet": fig_pellet(), "bulkfreq": fig_bulkfreq()}
    bode, fpk = fig_rc_bode()
    arr, arrstats = fig_arrhenius()
    drtfig, peaks = fig_drt()
    figs.update({"rcbode": bode, "arrhenius": arr, "drt": drtfig})
    for k, v in figs.items():
        (OUT / f"_fig_{k}.html").write_text(v, encoding="utf-8")
    # default data for the activation-energy calculator = the points in the Arrhenius figure
    (OUT / "_arr_data.txt").write_text("\n".join(f"{t:g}, {s:.3g}" for t, s in zip(ARR_T_C, ARR_SIG)), encoding="utf-8")
    print("wrote", sorted(figs))
    print("RC phase peak Hz", fpk)
    print("Arrhenius data T_C", ARR_T_C.tolist(), "sigma", ARR_SIG.tolist())
    print("Arrhenius fit Ea, dEa, A, R2", arrstats)
    print("DRT peaks", [(round(p["tau_s"], 8), round(p["R_ohm"], 1)) for p in peaks])

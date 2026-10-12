"""Publication drawing of an equivalent circuit with matplotlib (PNG, PDF, SVG, TIFF),
using the same layout rules as the in-app editor: series left to right, parallel branches
stacked between two rails.
"""

from __future__ import annotations

import re

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Rectangle

from . import circuit_tree as ct
from .plotting import _locked, pub_style

# layout units (1 unit ~ 1 pt at scale 1)
EW, EH, LAB = 64.0, 34.0, 18.0
GAP, PAD, BGAP, RAIL = 14.0, 6.0, 14.0, 14.0
INK = "#15212B"


def _measure(n):
    if n["t"] == "e":
        n["w"], n["h"] = EW, EH + 2 * LAB
        return
    for c in n["c"]:
        _measure(c)
    if n["t"] == "s":
        n["w"] = sum(c["w"] for c in n["c"]) + GAP * (len(n["c"]) - 1) + 2 * PAD
        n["h"] = max(c["h"] for c in n["c"]) + 2 * PAD
    else:
        n["w"] = max(c["w"] for c in n["c"]) + 2 * PAD + 2 * RAIL
        n["h"] = sum(c["h"] for c in n["c"]) + BGAP * (len(n["c"]) - 1) + 2 * PAD


def _place(n, x, y):
    """x, y = top-left in a y-down frame; n['cy'] = y of the signal line."""
    n["x"], n["y"], n["cy"] = x, y, y + n["h"] / 2
    if n["t"] == "e":
        return
    if n["t"] == "s":
        cx = x + PAD
        for c in n["c"]:
            _place(c, cx, n["cy"] - c["h"] / 2)
            cx += c["w"] + GAP
    else:
        cy = y + PAD
        for c in n["c"]:
            _place(c, x + PAD + RAIL + (n["w"] - 2 * PAD - 2 * RAIL - c["w"]) / 2, cy)
            cy += c["h"] + BGAP


def _label(name):
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", name or "")
    if not m:
        return name or ""
    kind, idx = m.groups()
    body = kind if len(kind) == 1 else r"\mathrm{" + kind + "}"
    return rf"${body}_{{{idx}}}$"


class _Pen:
    def __init__(self, ax, H, lw):
        self.ax, self.H, self.lw = ax, H, lw

    def Y(self, y):  # y-down layout -> y-up axes
        return self.H - y

    def line(self, x1, y1, x2, y2, lw=None):
        self.ax.plot([x1, x2], [self.Y(y1), self.Y(y2)], color=INK, lw=lw or self.lw, solid_capstyle="round",
                     zorder=2)

    def poly(self, xs, ys, lw=None):
        self.ax.plot(xs, [self.Y(v) for v in ys], color=INK, lw=lw or self.lw, solid_joinstyle="miter",
                     solid_capstyle="round", zorder=2)

    def box(self, x, y, w, h, text=None, fs=7):
        self.ax.add_patch(Rectangle((x, self.Y(y + h)), w, h, facecolor="white", edgecolor=INK, lw=self.lw, zorder=3))
        if text:
            self.ax.text(x + w / 2, self.Y(y + h / 2), text, ha="center", va="center", fontsize=fs, zorder=4)


def _symbol(p, kind, x, y, fs):
    L, R = x - EW / 2, x + EW / 2
    plate = p.lw * 1.5
    if kind == "C":
        p.line(L, y, x - 5, y); p.line(x + 5, y, R, y)
        p.line(x - 5, y - 13, x - 5, y + 13, plate); p.line(x + 5, y - 13, x + 5, y + 13, plate)
    elif kind == "CPE":
        p.line(L, y, x - 2, y); p.line(x + 7, y, R, y)
        p.poly([x - 8, x - 2, x - 8], [y - 13, y, y + 13], plate)
        p.poly([x + 1, x + 7, x + 1], [y - 13, y, y + 13], plate)
    elif kind == "L":
        p.line(L, y, x - 18, y); p.line(x + 18, y, R, y)
        for i in range(4):
            p.ax.add_patch(Arc((x - 13.5 + 9 * i, p.Y(y)), 9, 12, theta1=0, theta2=180, color=INK, lw=p.lw, zorder=2))
    else:
        bw = 34 if kind == "R" else 42
        p.line(L, y, x - bw / 2, y); p.line(x + bw / 2, y, R, y)
        p.box(x - bw / 2, y - 9, bw, 18, None if kind == "R" else kind, fs * 0.8)


def _draw(p, n, fs):
    if n["t"] == "e":
        cx = n["x"] + n["w"] / 2
        _symbol(p, n["k"], cx, n["cy"], fs)
        p.ax.text(cx, p.Y(n["cy"] - EH / 2 - 4), _label(n.get("n")), ha="center", va="bottom", fontsize=fs)
        return
    if n["t"] == "s":
        kids = n["c"]
        p.line(n["x"], n["cy"], kids[0]["x"], n["cy"])
        for a, b in zip(kids[:-1], kids[1:]):
            p.line(a["x"] + a["w"], n["cy"], b["x"], n["cy"])
        p.line(kids[-1]["x"] + kids[-1]["w"], n["cy"], n["x"] + n["w"], n["cy"])
    else:
        lx, rx = n["x"] + PAD + RAIL / 2, n["x"] + n["w"] - PAD - RAIL / 2
        top, bot = n["c"][0]["cy"], n["c"][-1]["cy"]
        p.line(n["x"], n["cy"], lx, n["cy"]); p.line(rx, n["cy"], n["x"] + n["w"], n["cy"])
        p.line(lx, top, lx, bot); p.line(rx, top, rx, bot)
        for c in n["c"]:
            p.line(lx, c["cy"], c["x"], c["cy"]); p.line(c["x"] + c["w"], c["cy"], rx, c["cy"])
    for c in n["c"]:
        _draw(p, c, fs)


@_locked
def circuit_figure(circuit: str, width_in: float | None = None, font_size: float = 8):
    """Matplotlib Figure of ``circuit`` (impedance.py string). Width defaults to the natural size."""
    tree = ct.assign_names(ct.parse(circuit))
    if tree is None:
        raise ValueError("The circuit is empty.")
    _measure(tree)
    M, lead = 6.0, 14.0
    _place(tree, M + lead, M)
    W, H = tree["w"] + 2 * (M + lead), tree["h"] + 2 * M
    natural_w = W / 72.0 * 0.9                     # ~0.9 pt per layout unit
    width_in = width_in or natural_w
    width_in = min(width_in, natural_w * 1.6)      # don't blow tiny circuits up to poster size
    with pub_style(font_size):
        fig = Figure(figsize=(width_in, width_in * H / W))
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")
        scale = width_in * 72.0 / W                # points per layout unit
        p = _Pen(ax, H, lw=min(1.2, max(0.8, 1.6 * scale)))   # same weight range as the plot axes and curves
        fs = font_size * min(1.0, max(0.75, scale))
        p.line(M, tree["cy"], M + lead, tree["cy"])
        p.line(M + lead + tree["w"], tree["cy"], W - M, tree["cy"])
        for xx in (M, W - M):
            ax.plot([xx], [p.Y(tree["cy"])], "o", color=INK, ms=3.0, zorder=4)
        _draw(p, tree, fs)
    return fig

"""Drawing for the EIS explainer video: one Scene per system (cell schematic, zoom animation,
Nyquist, Bode phase, equivalent circuit, DRT, captions) plus full-screen cards.

Rendering uses Agg blitting: every timeline segment draws a static background once; each frame
restores it and draws only the moving artists.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter  # noqa: E402

import models as M  # noqa: E402
from eisstudio import circuit_draw as cd  # noqa: E402
from eisstudio import circuit_tree as ct  # noqa: E402

for p in (Path(__file__).parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(p))

W, H, DPI = 1920, 1080, 100
PAPER, INK, MUTED, RULE = "#F3F6F5", M.INK, "#5B6B77", "#C9D3D9"
NU = 0.9          # on-screen oscillation frequency of the zoom animations (Hz of video time)

matplotlib.rcParams.update({
    "font.family": ["Source Sans 3", "Source Sans 3 Greek", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 16,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.linewidth": 1.4,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.major.size": 6, "ytick.major.size": 6,
    "axes.labelsize": 16, "text.color": INK,
})


def tint(c, a):
    """Colour c mixed with white (a = share of c)."""
    r = np.array(to_rgb(c))
    return tuple(1 - a * (1 - r))


def rect_px(fig, x0, y0, x1, y1, **kw):
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H], **kw)


def wrap(text, width=116):
    return "\n".join(textwrap.wrap(text, width))


def fmt_f(f):
    for div, u in ((1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz"), (1e-3, "mHz")):
        if f >= div * 0.9995 or u == "mHz":
            v = f / div
            s = f"{v:.3g}" if v < 100 else f"{v:.0f}"
            return f"{s} {u}"


def log_axis(ax, which="x"):
    loc = LogLocator(base=10, numticks=12)
    getattr(ax, f"{which}axis").set_major_locator(loc)
    getattr(ax, f"{which}axis").set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    getattr(ax, f"{which}axis").set_minor_formatter(NullFormatter())


def drive(t, theta):
    """d: charge displacement, v: current, e: field/voltage (all in [-1, 1]) at video time t."""
    w = 2 * np.pi * NU * t
    return -np.cos(w + theta), np.sin(w + theta), np.sin(w)


# ======================================================================================= #
# Zoom animations
# ======================================================================================= #
CATION, ANION, HOST = "#22313C", "#8D9AA5", "#D9DEE2"


class Anim:
    """Artists in the zoom axes (x 0..100, y 0..52.8). static: drawn in the background."""

    def __init__(self, ax, color, opts):
        self.ax, self.c, self.o = ax, color, opts
        self.static, self.dyn = [], []
        self.rng = np.random.default_rng(7)
        self.build()

    def S(self, a):
        self.static.append(a)
        return a

    def D(self, a):
        self.dyn.append(a)
        return a

    def text(self, x, y, s, dyn=False, **kw):
        kw.setdefault("fontsize", 15)
        kw.setdefault("color", MUTED)
        a = self.ax.text(x, y, s, **kw)
        return self.D(a) if dyn else self.S(a)

    def rect(self, x, y, w, h, **kw):
        return self.S(self.ax.add_patch(Rectangle((x, y), w, h, **kw)))

    def dots(self, color=CATION, ms=11, mfc=None, dyn=True, **kw):
        (a,) = self.ax.plot([], [], "o", color=color, ms=ms, mfc=mfc or color, mew=1.6, ls="none", **kw)
        return self.D(a) if dyn else self.S(a)

    def field_arrow(self, x, y, L=24, label="E"):
        (self._sh,) = self.ax.plot([], [], color=self.c, lw=4, solid_capstyle="butt")
        (self._hd,) = self.ax.plot([], [], color=self.c, ms=16, ls="none")
        self.D(self._sh), self.D(self._hd)
        self._fa = (x, y, L)
        self.text(x - L / 2 - 2, y, label, ha="right", va="center", fontsize=17, color=self.c, style="italic")

    def set_arrow(self, e):
        x, y, L = self._fa
        x2 = x + e * L / 2
        self._sh.set_data([x - e * L / 2, x2], [y, y])
        self._hd.set_data([x2], [y])
        self._hd.set_marker(">" if e >= 0 else "<")
        self._hd.set_markersize(6 + 12 * abs(e))

    def legend(self, items, x=1.5, y=1.6):
        for s, kind in items:
            if kind == "cat":
                self.S(self.ax.plot([x + 1], [y + 0.4], "o", color=CATION, ms=9)[0])
            elif kind == "an":
                self.S(self.ax.plot([x + 1], [y + 0.4], "o", color=ANION, mfc="white", ms=10, mew=1.6)[0])
            elif kind == "proc":
                self.S(self.ax.plot([x + 1], [y + 0.4], "o", color=self.c, ms=9)[0])
            self.text(x + 2.6, y + 0.4, s, va="center", fontsize=14, color=INK)
            x += 3.6 + 1.25 * len(s)

    def spread(self, n, x0, x1, y0, y1, dmin=5.0):
        """n well-separated random points (simple dart throwing)."""
        pts, tries = [], 0
        while len(pts) < n and tries < 5000:
            tries += 1
            p = self.rng.uniform([x0, y0], [x1, y1])
            if all(np.hypot(*(p - q)) > dmin for q in pts):
                pts.append(p)
        a = np.array(pts)
        return a[:, 0], a[:, 1]

    def remove(self):
        for a in self.static + self.dyn:
            a.remove()

    def build(self):
        pass

    def update(self, t, f, theta):
        pass


class Ohmic(Anim):
    def build(self):
        self.rect(0, 0, 100, 52.8, color="#EAF1F6", lw=0)
        n = 18
        x, y = self.spread(2 * n, 6, 94, 8, 44, 7.0)
        k = len(x) // 2
        self.cx, self.cy, self.ax_, self.ay = x[:k], y[:k], x[k:2 * k], y[k:2 * k]
        n = k
        self.ph = self.rng.uniform(0, 6.3, (4, n))
        self.cat = self.dots(CATION, 11)
        self.an = self.dots(ANION, 13, "white")
        self.field_arrow(72, 2.6)
        self.text(3, 50, self.o.get("title", "liquid electrolyte"), va="center", fontsize=16, color=INK)
        self.legend([("cation", "cat"), ("anion", "an")], y=1.2)

    def update(self, t, f, theta):
        d, v, e = drive(t, theta)
        j = 0.7 * np.sin(5 * t + self.ph)
        self.cat.set_data(self.cx + 5 * d + j[0], self.cy + j[1])
        self.an.set_data(self.ax_ - 3 * d + j[2], self.ay + j[3])
        self.set_arrow(e)


class Hop(Anim):
    """Ions hopping between sites of a crystal (style 'crystal') or along water channels ('membrane')."""

    def build(self):
        style = self.o.get("style", "crystal")
        self.a = 10.0
        if style == "crystal":
            self.rect(0, 0, 100, 52.8, color="#F4F1EA", lw=0)
            gx, gy = np.meshgrid(np.arange(5, 100, 10), np.arange(5, 50, 10))
            self.S(self.ax.plot(gx.ravel(), gy.ravel(), "o", color="#C9C1AE", ms=30, ls="none", zorder=1)[0])
            sx, sy = np.meshgrid(np.arange(10, 90, 10), np.arange(10, 50, 10))
            self.S(self.ax.plot(sx.ravel(), sy.ravel(), "o", color="#BDB49C", mfc="none", ms=9, mew=1.2,
                                ls="none", zorder=1)[0])
            pts = np.c_[sx.ravel(), sy.ravel()]
            keep = self.rng.random(len(pts)) < 0.55
            self.base = pts[keep]
            self.text(3, 50.5, self.o.get("title", "crystal lattice: anions (large) and Li sites (rings)"),
                      va="center", fontsize=15, color=INK)
            ion = self.o.get("ion", "Li⁺")
        else:
            self.rect(0, 0, 100, 52.8, color="#ECE7D6", lw=0)
            for yc in (12, 26, 40):
                xs = np.linspace(0, 100, 200)
                self.S(self.ax.fill_between(xs, yc - 3.4 + 0.6 * np.sin(xs / 6), yc + 3.4 + 0.6 * np.sin(xs / 7 + 1),
                                            color="#CFE3F0", lw=0))
                self.S(self.ax.plot(np.arange(4, 100, 8), np.full(12, yc + 4.6), "v", color="#9A8F70", ms=7,
                                    ls="none")[0])
            self.text(3, 50.5, self.o.get("title", "membrane: water channels lined with SO₃⁻ groups (▼)"),
                      va="center", fontsize=15, color=INK)
            xs, ys = np.meshgrid(np.arange(10, 90, 10), [12, 26, 40])
            self.base = np.c_[xs.ravel(), ys.ravel()][self.rng.random(xs.size) < 0.6]
            ion = self.o.get("ion", "H⁺")
        n = len(self.base)
        self.hopper = self.rng.random(n) < 0.45
        self.ph = self.rng.uniform(-0.5, 0.5, n)
        self.jit = self.rng.uniform(0, 6.3, (2, n))
        self.ions = self.dots(CATION, 11, zorder=4)
        self.hops = self.dots(self.c, 12, zorder=5)
        self.field_arrow(72, 2.4)
        self.legend([(ion, "cat"), (f"{ion} that hops", "proc")], x=1.5, y=1.2)

    def update(self, t, f, theta):
        d, v, e = drive(t, theta)
        s = 0.5 * (1 + np.tanh(5 * (d + self.ph)))
        x = self.base[:, 0] + np.where(self.hopper, self.a * (s - 0.5), 0.6 * np.sin(9 * t + self.jit[0]))
        y = self.base[:, 1] + 0.5 * np.sin(8 * t + self.jit[1])
        self.ions.set_data(x[~self.hopper], y[~self.hopper])
        self.hops.set_data(x[self.hopper], y[self.hopper])
        self.set_arrow(e)


class GrainBoundary(Anim):
    def build(self):
        self.rect(0, 0, 100, 52.8, color="#F4F1EA", lw=0)
        self.xb = lambda y: 47 + 0.12 * y
        bw = 5.0
        ys = np.array([0, 52.8])
        self.S(self.ax.add_patch(Polygon([(self.xb(0), 0), (self.xb(0) + bw, 0), (self.xb(52.8) + bw, 52.8),
                                          (self.xb(52.8), 52.8)], color=tint(self.c, 0.22), lw=0)))
        for off in (0, bw):
            self.S(self.ax.plot(self.xb(ys) + off, ys, color=self.c, lw=1.5, ls=(0, (4, 3)))[0])
        self.bw = bw
        for ang, x0, x1 in ((12, 0, 47), (-24, 53, 100)):
            g = np.arange(-60, 160, 6.0)
            gx, gy = np.meshgrid(g, g)
            a = np.radians(ang)
            X, Y = gx * np.cos(a) - gy * np.sin(a), gx * np.sin(a) + gy * np.cos(a)
            left = (X < self.xb(Y) - 1) if x0 == 0 else (X > self.xb(Y) + bw + 1)
            m = left & (X > 0) & (X < 100) & (Y > 0) & (Y < 50)
            self.S(self.ax.plot(X[m], Y[m], "o", color="#CDBF9F", ms=6, ls="none")[0])
        self.text(20, 50.5, "grain 1", ha="center", va="center", fontsize=16, color=INK)
        self.text(78, 50.5, "grain 2", ha="center", va="center", fontsize=16, color=INK)
        self.text(self.xb(26) + bw / 2, 50.5, "boundary", ha="center", va="center", fontsize=15, color=self.c)
        xl, yl = self.spread(13, 4, 40, 6, 46, 6.0)
        xr, yr = self.spread(13, 60, 96, 6, 46, 6.0)
        self.x0, self.y = np.r_[xl, xr], np.r_[yl, yr]
        n = len(self.x0)
        self.side = np.r_[np.zeros(len(xl)), np.ones(len(xr))].astype(bool)
        self.ph = self.rng.uniform(0, 6.3, n)
        self.ions = self.dots(self.c, 12, zorder=5)
        self.plus = [self.text(0, 0, "+", dyn=True, fontsize=22, fontweight="bold", color=self.c, ha="center",
                               va="center") for _ in range(4)]
        self.minus = [self.text(0, 0, "−", dyn=True, fontsize=22, fontweight="bold", color=MUTED, ha="center",
                                va="center") for _ in range(4)]
        self.field_arrow(76, 2.6)
        self.legend([("Li⁺", "proc")], y=1.2)

    def update(self, t, f, theta):
        d, v, e = drive(t, theta)
        x = self.x0 + 11 * d + 0.5 * np.sin(7 * t + self.ph)
        xb = self.xb(self.y)
        x = np.where(self.side, np.maximum(x, xb + self.bw + 1.4), np.minimum(x, xb - 1.4))
        self.ions.set_data(x, self.y + 0.4 * np.sin(6 * t + self.ph))
        for i, (p, m) in enumerate(zip(self.plus, self.minus)):
            yy = 9 + 11 * i
            side = -1 if d > 0 else 1
            p.set_position((self.xb(yy) + self.bw / 2 + side * 5.5, yy))
            m.set_position((self.xb(yy) + self.bw / 2 - side * 5.5, yy))
            p.set_alpha(min(1, abs(d) * 1.3))
            m.set_alpha(min(1, abs(d) * 1.3))
        self.set_arrow(e)


class ElectrodeBlock(Anim):
    def build(self):
        self.xe = 70.0
        self.rect(0, 0, self.xe, 52.8, color="#F4F1EA", lw=0)
        self.rect(self.xe, 0, 100 - self.xe, 52.8, color="#D7C58F", lw=0)
        self.S(self.ax.plot([self.xe, self.xe], [0, 52.8], color=INK, lw=2)[0])
        self.text(3, 50.5, self.o.get("left", "solid electrolyte"), va="center", fontsize=16, color=INK)
        self.text(self.xe + 3, 50.5, self.o.get("right", "metal electrode"), va="center", fontsize=16, color=INK)
        self.text(self.xe + 3, 46.5, "conducts e⁻, blocks Li⁺", va="center", fontsize=14)
        self.cx, self.cy = self.spread(22, 4, 64, 7, 44, 6.0)
        self.ax0, self.ay = self.spread(22, 4, 66, 7, 44, 6.0)
        n = max(len(self.cx), len(self.ax0))
        self.ph = self.rng.uniform(0, 6.3, (2, n))
        self.cat = self.dots(self.c, 12, zorder=5)
        self.an = self.dots(ANION, 12, "white", zorder=4)
        self.el = [self.text(self.xe + 2.2, 8 + 6 * i, "−", dyn=True, fontsize=22, fontweight="bold",
                             color=INK, ha="center", va="center") for i in range(7)]
        self.field_arrow(35, 2.6)
        self.legend([("Li⁺", "proc"), ("fixed anion sites (immobile)", "an")], x=50, y=-0.6)
        self.text(self.xe - 1, 2.6, "double layer", ha="right", va="center", fontsize=14, color=self.c)

    def update(self, t, f, theta):
        d, v, e = drive(t, theta)
        x = self.cx + 9 * d + 0.5 * np.sin(6 * t + self.ph[0][:len(self.cx)])
        x = np.clip(x, 1.5, self.xe - 1.5 - 1.6 * (self.cy % 3))
        self.cat.set_data(x, self.cy)
        self.an.set_data(self.ax0 + 0.4 * np.sin(8 * t + self.ph[1][:len(self.ax0)]), self.ay)
        for i, m in enumerate(self.el):
            m.set_alpha(float(np.clip(d * 1.4, 0, 1)))
            m.set_text("−")
        self.set_arrow(e)


class ChargeTransfer(Anim):
    """Ions crossing an interface and changing state (reaction / insertion)."""

    def X(self, x):
        return 100 - x if self.o.get("flip") else x

    def build(self):
        xi = 52.0
        self.xi = xi
        L, R = (0, xi) if not self.o.get("flip") else (100 - xi, 100)
        self.rect(min(self.X(0), self.X(xi)), 0, xi, 52.8, color="#EAF1F6" if self.o.get("liquid") else "#F4F1EA",
                  lw=0)
        self.rect(min(self.X(xi), self.X(100)), 0, 100 - xi, 52.8, color=HOST, lw=0)
        hx, hy = np.meshgrid(np.arange(xi + 5, 100, 7), np.arange(5, 48, 7))
        self.S(self.ax.plot(self.X(hx.ravel()), hy.ravel(), "s", color="#BCC3CA", ms=7, ls="none")[0])
        self.S(self.ax.plot([self.X(xi)] * 2, [0, 52.8], color=INK, lw=2)[0])
        ha_l, ha_r = ("left", "right") if not self.o.get("flip") else ("right", "left")
        self.text(self.X(2), 50.5, self.o.get("left", "electrolyte"), ha=ha_l, va="center", fontsize=16, color=INK)
        self.text(self.X(98), 50.5, self.o.get("right", "electrode"), ha=ha_r, va="center", fontsize=16, color=INK)
        self.text(50, 45.6, self.o.get("reaction", ""), ha="center", va="center", fontsize=18, color=self.c,
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=self.c, lw=1.2))
        n = 9
        self.y = np.linspace(6, 40, n) + self.rng.uniform(-1, 1, n)
        self.x0 = xi - self.rng.uniform(5, 20, n)
        self.span = self.rng.uniform(14, 26, n)
        self.ph = self.rng.uniform(-0.7, 0.7, n)
        self.jit = self.rng.uniform(0, 6.3, n)
        self.bx, self.by = self.spread(12, 4, xi - 22, 6, 40, 7.0)
        self.bg = self.dots(CATION, 10)
        self.ions = self.dots(CATION, 12, zorder=5)
        self.done = self.dots(self.c, 12, zorder=5)
        self.dl = [self.text(self.X(xi - 2.2), 8 + 6 * i, "+", dyn=True, fontsize=20, fontweight="bold",
                             color=CATION, ha="center", va="center") for i in range(6)]
        self.dr = [self.text(self.X(xi + 2.2), 8 + 6 * i, "−", dyn=True, fontsize=20, fontweight="bold",
                             color=INK, ha="center", va="center") for i in range(6)]
        self.legend([(self.o.get("ion", "Li⁺"), "cat"), (self.o.get("product", "after the reaction"), "proc")],
                    x=1.5, y=1.2)

    def update(self, t, f, theta):
        d, v, e = drive(t, theta)
        s = 0.5 * (1 + np.tanh(3.5 * (d + self.ph)))
        x = self.x0 + self.span * s
        y = self.y + 0.5 * np.sin(5 * t + self.jit)
        crossed = x > self.xi
        self.ions.set_data(self.X(x[~crossed]), y[~crossed])
        self.done.set_data(self.X(x[crossed]), y[crossed])
        self.bg.set_data(self.X(self.bx + 0.6 * np.sin(7 * t + self.by)), self.by)
        a = float(np.clip(abs(v) * 1.2, 0, 1))
        for p, m in zip(self.dl, self.dr):
            p.set_alpha(a)
            m.set_alpha(a)
            p.set_text("+" if v > 0 else "−")
            m.set_text("−" if v > 0 else "+")


class Film(Anim):
    def build(self):
        self.rect(0, 0, 40, 52.8, color="#EAF1F6", lw=0)
        self.rect(40, 0, 14, 52.8, color="#E3D6BC", lw=0)
        bl = self.rng.uniform([40.5, 1], [53.5, 51], (60, 2))
        self.S(self.ax.plot(bl[:, 0], bl[:, 1], "o", color="#C5B08A", ms=7, ls="none", alpha=0.8)[0])
        self.rect(54, 0, 46, 52.8, color="#D5D9DD", lw=0)
        for yy in np.arange(3, 50, 4.2):
            self.S(self.ax.plot([55, 100], [yy, yy], color="#7E8891", lw=2.2)[0])
        self.text(2, 50.5, "liquid electrolyte", va="center", fontsize=16, color=INK)
        self.text(47, 50.5, "SEI", ha="center", va="center", fontsize=16, color=self.c, fontweight="bold")
        self.text(98, 50.5, "graphite layers", ha="right", va="center", fontsize=16, color=INK)
        n = 8
        self.y = 5.1 + 4.2 * self.rng.choice(np.arange(11), n, replace=False)
        self.x0 = self.rng.uniform(22, 36, n)
        self.span = self.rng.uniform(26, 40, n)
        self.ph = self.rng.uniform(-0.6, 0.6, n)
        self.ions = self.dots(CATION, 12, zorder=5)
        self.inside = self.dots(self.c, 12, zorder=5)
        self.legend([("Li⁺ (solvated)", "cat"), ("Li⁺ in SEI / graphite", "proc")], y=1.2)

    def update(self, t, f, theta):
        d, v, e = drive(t, theta)
        s = 0.5 * (1 + np.tanh(2.5 * (d + self.ph)))
        x = self.x0 + self.span * s
        out = x < 40
        self.ions.set_data(x[out], self.y[out])
        self.inside.set_data(x[~out], self.y[~out])


class Diffusion(Anim):
    def build(self):
        self.tau = self.o["tau"]
        cx, cy, R = 26.0, 26.0, 22.0
        self.R = R
        n = 140
        xs = np.linspace(cx - R, cx + R, n)
        X, Y = np.meshgrid(xs, np.linspace(cy - R, cy + R, n))
        r = np.hypot(X - cx, Y - cy)
        self.depth = np.clip((R - r) / R, 0, 1)
        self.mask = r > R
        cmap = LinearSegmentedColormap.from_list("li", ["#7FB2D9", "#F7F7F2", self.c])
        cmap.set_bad((0, 0, 0, 0))
        self.im = self.D(self.ax.imshow(np.zeros_like(X), extent=[cx - R, cx + R, cy - R, cy + R], origin="lower",
                                        cmap=cmap, vmin=-1, vmax=1, interpolation="bilinear", zorder=2))
        self.S(self.ax.add_patch(Circle((cx, cy), R, fill=False, ec=INK, lw=2, zorder=3)))
        self.text(cx, 50.8, self.o.get("title", "one active particle"), ha="center", va="center", fontsize=16,
                  color=INK)
        # depth profile
        x0, x1, y0, y1 = 58, 93, 10, 40
        self.pf = (x0, x1, y0, y1)
        self.S(self.ax.plot([x0, x1], [(y0 + y1) / 2] * 2, color=RULE, lw=1.2)[0])
        self.S(self.ax.plot([x0, x0], [y0, y1], color=INK, lw=1.5)[0])
        self.text(x0, y0 - 3, "surface", ha="center", va="center", fontsize=14)
        self.text(x1, y0 - 3, "centre", ha="center", va="center", fontsize=14)
        self.text((x0 + x1) / 2, y1 + 5.5, "extra Li vs depth", ha="center", va="center", fontsize=15, color=INK)
        self.x = np.linspace(0, 1, 160)
        (self.prof,) = self.ax.plot([], [], color=self.c, lw=3)
        (self.env,) = self.ax.plot([], [], color=self.c, lw=1.2, ls=(0, (3, 3)))
        self.D(self.prof), self.D(self.env)
        (self.dl,) = self.ax.plot([], [], color=INK, lw=1.4, ls=(0, (2, 2)))
        self.D(self.dl)
        self.dlt = self.text(0, 0, "", dyn=True, fontsize=15, color=INK, ha="left", va="bottom")
        self.legend([("more Li", "proc")], x=58, y=1.2)

    def profile(self, f):
        a = np.sqrt(1j * 2 * np.pi * f * self.tau)
        x = self.x
        return (np.exp(-a * x) + np.exp(-a * (2 - x))) / (1 + np.exp(-2 * a))

    def update(self, t, f, theta):
        w = 2 * np.pi * NU * t
        c = self.profile(f)
        cr = (c * np.exp(1j * w)).real
        img = np.interp(self.depth, self.x, cr)
        img = np.ma.array(img, mask=self.mask)
        self.im.set_data(img)
        x0, x1, y0, y1 = self.pf
        X = x0 + (x1 - x0) * self.x
        hh = (y1 - y0) / 2
        self.prof.set_data(X, (y0 + y1) / 2 + hh * cr)
        self.env.set_data(X, (y0 + y1) / 2 + hh * np.abs(c))
        dd = min(1.0, np.sqrt(2 / (2 * np.pi * f * self.tau)))
        xd = x0 + (x1 - x0) * dd
        self.dl.set_data([xd, xd], [y0, y1])
        self.dlt.set_position((min(xd + 0.8, 82), y1 - 1))
        self.dlt.set_text("depth ≈ √(D/ω)" if dd < 1 else "whole particle")


class Bubbles(Anim):
    def build(self):
        self.rect(0, 0, 58, 52.8, color="#EEF3F7", lw=0)
        self.rect(58, 0, 6, 52.8, color="#3D4852", lw=0)
        self.rect(64, 0, 36, 52.8, color="#EFE8CC", lw=0)
        pts = []
        while len(pts) < 22:
            p = self.rng.uniform([3, 3], [54, 50])
            if all(np.hypot(*(p - q)) > 9.5 for q in pts):
                pts.append(p)
        for p in pts:
            self.S(self.ax.add_patch(Circle(p, 3.2, color="#AEB6BD", lw=0, zorder=2)))
        self.fib = np.array(pts)
        self.text(2, 50.5, "porous Ti layer", va="center", fontsize=16, color=INK, zorder=6,
                  bbox=dict(fc="#EEF3F7", ec="none", pad=1))
        self.text(61, 50.5, "IrO₂", ha="center", va="center", fontsize=14, color="white", zorder=6)
        self.text(98, 50.5, "membrane", ha="right", va="center", fontsize=16, color=INK)
        n = 12
        self.off = self.rng.uniform(0, 1, n)
        self.y0 = self.rng.uniform(5, 46, n)
        self.b = self.D(self.ax.scatter([], [], s=[], facecolors="white", edgecolors=self.c, linewidths=2, zorder=4))
        self.h2o = self.dots("#3E8DC4", 7, zorder=3)
        self.wy = self.rng.uniform(4, 48, 16)
        self.woff = self.rng.uniform(0, 1, 16)
        self.text(30, 2.4, "← O₂ bubbles out     H₂O in →", ha="center", va="center", fontsize=15, color=INK,
                  zorder=6, bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
        self.st = self.D(self.ax.scatter([], [], s=[], facecolors="white", edgecolors=self.c, linewidths=2, zorder=4))
        self.phase = 0.0
        self.tprev = None

    def update(self, t, f, theta):
        d, v, e = drive(t, theta)
        dt = 0 if self.tprev is None else t - self.tprev
        self.tprev = t
        self.phase += dt * 0.12 * (1 + 0.8 * v)
        p = (self.off + self.phase) % 1
        x = 56 - 56 * p
        y = self.y0 + 2.5 * np.sin(6 * p + self.off * 6)
        r = 1.0 + 2.2 * p
        self.b.set_offsets(np.c_[x, y])
        self.b.set_sizes((10.37 * 2 * r) ** 2 / 4)
        ys = np.array([8, 20, 32, 44])
        rs = 1.6 + 1.3 * (1 + d)
        self.st.set_offsets(np.c_[np.full(4, 57.2 - rs * 0.6), ys])
        self.st.set_sizes(np.full(4, (10.37 * 2 * rs) ** 2 / 4))
        wp = (self.woff + self.phase * 1.5) % 1
        self.h2o.set_data(2 + 54 * wp, self.wy)


class Inductance(Anim):
    def build(self):
        self.rect(0, 0, 100, 52.8, color="#F6F7F8", lw=0)
        th = np.linspace(0, 10 * np.pi, 600)
        xc = 36 + 28 * th / th[-1] + 2.4 * np.cos(th)
        yc = 24 + 8 * np.sin(th)
        self.S(self.ax.plot(np.r_[2, 34.6, xc, 66, 98], np.r_[24, 24, yc, 24, 24], color="#B87333", lw=4,
                            solid_capstyle="round", zorder=3)[0])
        self.loops = [self.D(self.ax.add_patch(Ellipse((50, 25), 40 + 13 * k, 18 + 8 * k, fill=False, ec=self.c,
                                                       lw=2.4, ls=(0, (5, 3)), zorder=2))) for k in range(3)]
        (self.cur,) = self.ax.plot([], [], color=INK, ms=20, ls="none", zorder=4)
        self.D(self.cur)
        self.text(3, 50.5, "cable + wound electrodes = a coil", va="center", fontsize=16, color=INK)
        self.text(3, 3.0, "magnetic field (dashed) builds up with the current, so the voltage leads the current",
                  va="center", fontsize=14, color=MUTED)
        self.text(14, 28.2, "I", fontsize=18, style="italic", color=INK, ha="center")

    def update(self, t, f, theta):
        d, v, e = drive(t, theta)
        for k, L in enumerate(self.loops):
            L.set_alpha(float(np.clip(abs(v) * 1.1 - 0.1 * k, 0, 1)))
            L.width = 40 + 13 * k + 5 * abs(v)
            L.height = 18 + 8 * k + 3 * abs(v)
        self.cur.set_data([14 + 4 * np.sign(v)], [24])
        self.cur.set_marker(">" if v >= 0 else "<")
        self.cur.set_markersize(8 + 14 * abs(v))


ANIMS = {"ohmic": Ohmic, "hop": Hop, "proton": Hop, "gb": GrainBoundary, "electrode_block": ElectrodeBlock,
         "charge_transfer": ChargeTransfer, "film": Film, "diffusion": Diffusion, "bubbles": Bubbles,
         "inductance": Inductance}

ZOOM_OPTS = {
    ("intro", "ohm"): dict(title="liquid electrolyte between the electrodes"),
    ("intro", "ct"): dict(left="liquid electrolyte", right="metal electrode", reaction="Ox + e⁻ ⇌ Red", ion="Ox",
                          product="Red", liquid=True),
    ("pellet", "bulk"): dict(title="inside a grain: anions (large) and Li sites (rings)"),
    ("pellet", "el"): dict(left="solid electrolyte", right="Au electrode"),
    ("ssb", "se"): dict(title="solid electrolyte separator: Li⁺ hop between sites"),
    ("ssb", "an"): dict(left="solid electrolyte", right="Li metal", reaction="Li⁺ + e⁻ ⇌ Li", product="Li",
                        flip=True),
    ("ssb", "ca"): dict(left="solid electrolyte", right="NMC particle", reaction="Li⁺ + e⁻ ⇌ Li in NMC",
                        product="Li in NMC"),
    ("ssb", "dif"): dict(title="one NMC particle"),
    ("pem", "mem"): dict(style="membrane", ion="H⁺"),
    ("pem", "her"): dict(left="membrane", right="Pt catalyst", reaction="2H⁺ + 2e⁻ ⇌ H₂", ion="H⁺", product="H"),
    ("pem", "oer"): dict(left="membrane", right="IrO₂ catalyst", reaction="2H₂O ⇌ O₂ + 4H⁺ + 4e⁻", ion="H⁺",
                         product="O / OH on IrO₂", flip=True),
    ("lib", "ohm"): dict(title="electrolyte in the separator pores"),
    ("lib", "ct"): dict(left="liquid electrolyte", right="NMC particle", reaction="Li⁺ + e⁻ ⇌ Li in NMC",
                        product="Li in NMC", liquid=True),
    ("lib", "dif"): dict(title="one NMC particle"),
}


# ======================================================================================= #
# Cell schematics (axes x 0..100, y 0..50)
# ======================================================================================= #
def _lab(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 15)
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("color", INK)
    ax.text(x, y, s, **kw)


def _wires(ax, xl, xr, ytop=44):
    ax.plot([xl, xl, 44], [36, ytop, ytop], color=INK, lw=2, solid_capstyle="round")
    ax.plot([xr, xr, 56], [36, ytop, ytop], color=INK, lw=2, solid_capstyle="round")
    ax.add_patch(Circle((50, ytop), 5, fc="white", ec=INK, lw=2, zorder=3))
    t = np.linspace(-3, 3, 50)
    ax.plot(50 + t, ytop + 1.6 * np.sin(t * np.pi / 3), color=INK, lw=2, zorder=4)
    _lab(ax, 57, ytop + 3.5, "potentiostat", ha="left", fontsize=14, color=MUTED)


def _grains(ax, x0, x1, y0, y1, n, fc, ec, seed=3, lw=1.6, clip=None):
    from scipy.spatial import Voronoi
    rng = np.random.default_rng(seed)
    pts = rng.uniform([x0 - 10, y0 - 10], [x1 + 10, y1 + 10], (n, 2))
    vor = Voronoi(pts)
    segs = [vor.vertices[r] for r in vor.ridge_vertices if -1 not in r]
    r = Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec="none")
    ax.add_patch(r)
    lc = LineCollection(segs, colors=ec, lw=lw)
    ax.add_collection(lc)
    lc.set_clip_path(clip or r)
    return vor


def _particles(rng, x0, x1, y0, y1, n, rmin, rmax, gap=0.6):
    out = []
    tries = 0
    while len(out) < n and tries < 20000:
        tries += 1
        r = rng.uniform(rmin, rmax)
        p = rng.uniform([x0 + r, y0 + r], [x1 - r, y1 - r])
        if all(np.hypot(*(p - q)) > r + s + gap for q, s in out):
            out.append((p, r))
    return out


def schematic(ax, key):
    """Draw the cell; return {region: (x, y, w, h)} in axes data units."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    if key == "intro":
        ax.add_patch(Rectangle((8, 6), 8, 30, fc="#B9C0C6", ec=INK, lw=1.5))
        ax.add_patch(Rectangle((84, 6), 8, 30, fc="#B9C0C6", ec=INK, lw=1.5))
        ax.add_patch(Rectangle((16, 6), 68, 30, fc="#E3EEF6", ec=INK, lw=1.5))
        rng = np.random.default_rng(2)
        ax.plot(rng.uniform(19, 81, 30), rng.uniform(9, 33, 30), "o", color=CATION, ms=5)
        ax.plot(rng.uniform(19, 81, 30), rng.uniform(9, 33, 30), "o", color=ANION, mfc="white", ms=6)
        _lab(ax, 12, 2.5, "electrode")
        _lab(ax, 50, 2.5, "liquid electrolyte")
        _lab(ax, 88, 2.5, "electrode")
        _wires(ax, 12, 88)
        return {"electrolyte": (38, 12, 24, 18), "interface": (79, 8, 9, 26)}
    if key == "pellet":
        ax.add_patch(Rectangle((8, 6), 7, 30, fc="#D7C58F", ec=INK, lw=1.5))
        ax.add_patch(Rectangle((85, 6), 7, 30, fc="#D7C58F", ec=INK, lw=1.5))
        _grains(ax, 15, 85, 6, 36, 70, "#F1ECDF", "#A89A78", seed=4)
        ax.add_patch(Rectangle((15, 6), 70, 30, fc="none", ec=INK, lw=1.5))
        _lab(ax, 11.5, 2.5, "Au")
        _lab(ax, 50, 2.5, "pressed pellet: grains and grain boundaries")
        _lab(ax, 88.5, 2.5, "Au")
        _wires(ax, 11.5, 88.5)
        return {"bulk": (30, 15, 9, 9), "gb": (52, 16, 9, 9), "electrode": (81, 8, 8, 26)}
    if key == "ssb":
        ax.add_patch(Rectangle((6, 6), 10, 30, fc="#C7CDD2", ec=INK, lw=1.5))
        _grains(ax, 16, 40, 6, 36, 30, "#F1ECDF", "#B5A887", seed=5, lw=1.2)
        _grains(ax, 40, 86, 6, 36, 60, "#F1ECDF", "#B5A887", seed=6, lw=1.2)
        rng = np.random.default_rng(11)
        parts = _particles(rng, 41, 85, 7, 35, 14, 3.0, 4.6)
        for p, r in parts:
            ax.add_patch(Circle(p, r, fc="#6E7B88", ec=INK, lw=1.2, zorder=3))
        ax.add_patch(Rectangle((86, 6), 6, 30, fc="#C7CDD2", ec=INK, lw=1.5))
        ax.add_patch(Rectangle((16, 6), 70, 30, fc="none", ec=INK, lw=1.5, zorder=4))
        _lab(ax, 11, 2.5, "Li metal")
        _lab(ax, 28, 2.5, "solid electrolyte")
        _lab(ax, 63, 2.5, "composite cathode (NMC + SE)")
        _lab(ax, 89, 2.5, "Al")
        _wires(ax, 11, 89)
        (pa, ra) = max(parts, key=lambda q: q[1] - 0.05 * abs(q[0][1] - 21))
        (pb, rb) = sorted(parts, key=lambda q: q[0][0])[0]
        return {"separator": (22, 15, 12, 12), "anode_if": (12.5, 8, 7, 26),
                "cathode_if": (pb[0] - rb - 2.5, pb[1] - 3.5, 7, 7),
                "particle": (pa[0] - ra - 1.2, pa[1] - ra - 1.2, 2 * ra + 2.4, 2 * ra + 2.4)}
    if key == "pem":
        ax.add_patch(Rectangle((4, 6), 6, 30, fc="#9EA7AF", ec=INK, lw=1.5))
        ax.add_patch(Rectangle((10, 6), 20, 30, fc="#E8EDF1", ec=INK, lw=1.2))
        rng = np.random.default_rng(3)
        for p, r in _particles(rng, 10.5, 29.5, 6.5, 35.5, 26, 1.4, 2.0, 0.8):
            ax.add_patch(Circle(p, r, fc="#AEB6BD", ec="none"))
        ax.add_patch(Rectangle((30, 6), 4, 30, fc="#3D4852", ec=INK, lw=1.2))
        ax.add_patch(Rectangle((34, 6), 18, 30, fc="#EFE8CC", ec=INK, lw=1.2))
        ax.add_patch(Rectangle((52, 6), 4, 30, fc="#22272B", ec=INK, lw=1.2))
        ax.add_patch(Rectangle((56, 6), 22, 30, fc="#D9DCDF", ec=INK, lw=1.2))
        for i in range(9):
            yy = 8 + 3.2 * i
            ax.plot([57, 77], [yy + rng.uniform(-0.6, 0.6), yy + rng.uniform(-0.6, 0.6)], color="#8D949A", lw=1.4)
        ax.add_patch(Rectangle((78, 6), 6, 30, fc="#9EA7AF", ec=INK, lw=1.5))
        for p, r in _particles(np.random.default_rng(9), 11, 29, 8, 34, 7, 0.9, 1.5, 3):
            ax.add_patch(Circle(p, r, fc="white", ec="#5B6B77", lw=1, zorder=3))
        _lab(ax, 7, 2.5, "plate", fontsize=13)
        _lab(ax, 20, 2.5, "Ti porous layer", fontsize=14)
        _lab(ax, 43, 2.5, "membrane", fontsize=14)
        _lab(ax, 67, 2.5, "carbon paper", fontsize=14)
        _lab(ax, 81, 2.5, "plate", fontsize=13)
        _lab(ax, 32, 39, "anode\nIrO₂", fontsize=13)
        _lab(ax, 54, 39, "cathode\nPt/C", fontsize=13)
        _lab(ax, 93, 30, "H₂ →", fontsize=15)
        _lab(ax, 93, 14, "", fontsize=15)
        _lab(ax, -1.5, 30, "O₂ ←", fontsize=15, ha="left")
        _lab(ax, -1.5, 14, "H₂O →", fontsize=15, ha="left")
        ax.plot([7, 7, 44], [36, 46, 46], color=INK, lw=2)
        ax.plot([81, 81, 56], [36, 46, 46], color=INK, lw=2)
        ax.add_patch(Rectangle((44, 43), 12, 6, fc="white", ec=INK, lw=2, zorder=3))
        _lab(ax, 50, 46, "DC + AC", fontsize=13, zorder=4)
        return {"membrane": (37, 13, 12, 12), "cathode_cl": (50.5, 8, 7, 26), "anode_cl": (28.5, 8, 7, 26),
                "ptl": (14, 13, 12, 12)}
    if key == "lib":
        ax.add_patch(Rectangle((6, 6), 3, 28, fc="#C98B5E", ec=INK, lw=1.2))
        ax.add_patch(Rectangle((9, 6), 30, 28, fc="#E3EEF6", ec=INK, lw=1.2))
        rng = np.random.default_rng(5)
        flakes = []
        for _ in range(400):
            p = rng.uniform([12, 8.5], [36, 31.5])
            if all(abs(p[0] - q[0]) > 7.4 or abs(p[1] - q[1]) > 3.4 for q in flakes):
                flakes.append(p)
        for p in flakes:
            ax.add_patch(Ellipse(p, 6.4, 2.6, angle=rng.uniform(-12, 12), fc="#4F5963", ec=INK, lw=0.8))
        ax.add_patch(Rectangle((39, 6), 10, 28, fc="#F7F7F2", ec=INK, lw=1.2, hatch="..", zorder=1))
        ax.add_patch(Rectangle((49, 6), 36, 28, fc="#E3EEF6", ec=INK, lw=1.2))
        parts = _particles(np.random.default_rng(8), 49.5, 84.5, 6.5, 33.5, 16, 2.6, 4.0, 0.5)
        for p, r in parts:
            ax.add_patch(Circle(p, r, fc="#7C7391", ec=INK, lw=1, zorder=3))
        ax.add_patch(Rectangle((85, 6), 3, 28, fc="#C7CDD2", ec=INK, lw=1.2))
        _lab(ax, 24, 2.5, "graphite anode")
        _lab(ax, 44, 2.5, "separator", fontsize=14)
        _lab(ax, 67, 2.5, "NMC cathode")
        # leads with a coil
        th = np.linspace(0, 6 * np.pi, 200)
        ax.plot([7.5, 7.5, 30], [34, 44, 44], color="#B87333", lw=2.4)
        ax.plot(30 + 10 * th / th[-1] + 1.2 * np.cos(th), 44 + 2.4 * np.sin(th), color="#B87333", lw=2.4)
        ax.plot([41.2, 44], [44, 44], color="#B87333", lw=2.4)
        ax.plot([86.5, 86.5, 56], [34, 44, 44], color="#B87333", lw=2.4)
        ax.add_patch(Circle((50, 44), 5, fc="white", ec=INK, lw=2, zorder=3))
        t = np.linspace(-3, 3, 50)
        ax.plot(50 + t, 44 + 1.6 * np.sin(t * np.pi / 3), color=INK, lw=2, zorder=4)
        _lab(ax, 57, 47.5, "cables", ha="left", fontsize=14, color=MUTED)
        fl = min(flakes, key=lambda q: np.hypot(q[0] - 32, q[1] - 20))
        pa = min(parts, key=lambda q: np.hypot(q[0][0] - 66, q[0][1] - 20))
        pb = min(parts, key=lambda q: np.hypot(q[0][0] - 76, q[0][1] - 13))
        (p, r) = pb
        return {"leads": (27, 40, 17, 8), "separator": (40, 12, 8, 14), "sei": (fl[0] + 1, fl[1] - 2.2, 4.5, 4.4),
                "particle_surf": (p[0] - r - 1.8, p[1] - 2.5, 5, 5),
                "particle": (pa[0][0] - pa[1] - 1, pa[0][1] - pa[1] - 1, 2 * pa[1] + 2, 2 * pa[1] + 2)}
    raise KeyError(key)


# ======================================================================================= #
# Coloured circuit
# ======================================================================================= #
class _CPen(cd._Pen):
    color = INK

    def line(self, x1, y1, x2, y2, lw=None):
        self.ax.plot([x1, x2], [self.Y(y1), self.Y(y2)], color=self.color, lw=lw or self.lw,
                     solid_capstyle="round", zorder=2)

    def poly(self, xs, ys, lw=None):
        self.ax.plot(xs, [self.Y(v) for v in ys], color=self.color, lw=lw or self.lw, solid_joinstyle="miter",
                     solid_capstyle="round", zorder=2)

    def box(self, x, y, w, h, text=None, fs=7):
        self.ax.add_patch(Rectangle((x, self.Y(y + h)), w, h, facecolor="white", edgecolor=self.color, lw=self.lw,
                                    zorder=3))
        if text:
            self.ax.text(x + w / 2, self.Y(y + h / 2), text, ha="center", va="center", fontsize=fs, zorder=4,
                         color=self.color)


def _csymbol(p, kind, x, y, fs):
    if kind == "L":
        EW = cd.EW
        p.line(x - EW / 2, y, x - 18, y)
        p.line(x + 18, y, x + EW / 2, y)
        for i in range(4):
            p.ax.add_patch(Arc((x - 13.5 + 9 * i, p.Y(y)), 9, 12, theta1=0, theta2=180, color=p.color, lw=p.lw,
                               zorder=2))
    else:
        cd._symbol(p, kind, x, y, fs)


def _cdraw(p, n, fs, colors):
    if n["t"] == "e":
        cx = n["x"] + n["w"] / 2
        p.color = colors.get(n["n"], INK)
        _csymbol(p, n["k"], cx, n["cy"], fs)
        p.ax.text(cx, p.Y(n["cy"] - cd.EH / 2 - 4), cd._label(n.get("n")), ha="center", va="bottom", fontsize=fs,
                  color=p.color)
        p.color = INK
        return
    p.color = INK
    if n["t"] == "s":
        kids = n["c"]
        p.line(n["x"], n["cy"], kids[0]["x"], n["cy"])
        for a, b in zip(kids[:-1], kids[1:]):
            p.line(a["x"] + a["w"], n["cy"], b["x"], n["cy"])
        p.line(kids[-1]["x"] + kids[-1]["w"], n["cy"], n["x"] + n["w"], n["cy"])
    else:
        lx, rx = n["x"] + cd.PAD + cd.RAIL / 2, n["x"] + n["w"] - cd.PAD - cd.RAIL / 2
        top, bot = n["c"][0]["cy"], n["c"][-1]["cy"]
        p.line(n["x"], n["cy"], lx, n["cy"])
        p.line(rx, n["cy"], n["x"] + n["w"], n["cy"])
        p.line(lx, top, lx, bot)
        p.line(rx, top, rx, bot)
        for c in n["c"]:
            p.line(lx, c["cy"], c["x"], c["cy"])
            p.line(c["x"] + c["w"], c["cy"], rx, c["cy"])
    for c in n["c"]:
        _cdraw(p, c, fs, colors)


def circuit(ax, system, wpx, hpx):
    tree = ct.assign_names(ct.parse(system.circuit))
    cd._measure(tree)
    Mg, lead = 6.0, 14.0
    cd._place(tree, Mg + lead, Mg + 6)
    Wd, Hd = tree["w"] + 2 * (Mg + lead), tree["h"] + 2 * Mg + 6
    scale = min(wpx / Wd, hpx / Hd)                       # px per unit
    cx, cy = Wd / 2, Hd / 2
    ax.set_xlim(cx - wpx / scale / 2, cx + wpx / scale / 2)
    ax.set_ylim(cy - hpx / scale / 2, cy + hpx / scale / 2)
    ax.axis("off")
    lw = float(np.clip(1.9 * scale / 1.389, 2.0, 3.6))
    fs = float(np.clip(12 * scale / 1.389 * 1.389 / 1.0, 15, 22))
    p = _CPen(ax, Hd, lw)
    colors = {e: b.color for b in system.blocks for e in b.elements}
    p.line(Mg, tree["cy"], Mg + lead, tree["cy"])
    p.line(Mg + lead + tree["w"], tree["cy"], Wd - Mg, tree["cy"])
    for xx in (Mg, Wd - Mg):
        ax.plot([xx], [p.Y(tree["cy"])], "o", color=INK, ms=7, zorder=4)
    _cdraw(p, tree, fs, colors)
    kids = tree["c"] if tree["t"] == "s" else [tree]
    assert len(kids) == len(system.blocks), (system.key, len(kids))
    return {b.key: (n["x"] - 4, Hd - n["y"] - n["h"] - 2, n["w"] + 8, n["h"] + 4) for b, n in zip(system.blocks, kids)}


# ======================================================================================= #
# Scene
# ======================================================================================= #
class Scene:
    def __init__(self, system: M.System):
        self.s = s = system
        self.fig = fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=PAPER)
        self.canvas = FigureCanvasAgg(fig)
        # ---------- data ----------
        dec = np.log10(s.fmax / s.fmin)
        self.fl = np.logspace(np.log10(s.fmax), np.log10(s.fmin), int(60 * dec) + 1)
        self.Zl = s.Z(self.fl) * s.scale
        self.al = s.active(self.fl)
        k0, k1 = np.ceil(8 * np.log10(s.fmin)), np.floor(8 * np.log10(s.fmax))
        self.fd = 10 ** (np.arange(k1, k0 - 1, -1) / 8)
        self.Zd = s.Z(self.fd) * s.scale
        self.ad = s.active(self.fd)
        self.bands = []
        i0 = 0
        for i in range(1, len(self.fl) + 1):
            if i == len(self.fl) or self.al[i] != self.al[i0]:
                self.bands.append([self.al[i0], self.fl[i0], self.fl[min(i, len(self.fl) - 1)]])
                i0 = i
        self.drt = M.compute_drt(s)
        self.dtau, self.dgam = self.drt.tau, self.drt.gamma * s.scale
        self.df = 1 / (2 * np.pi * self.dtau)
        taus = {b.key: b.tau for b in s.blocks if b.tau}
        self.dblock = np.array([min(taus, key=lambda k: abs(np.log10(taus[k] / t))) for t in self.dtau])
        # ---------- layout ----------
        fig.text(40 / W, 1040 / H, s.title, fontsize=30, fontweight="bold", va="center")
        fig.text(42 / W, 1003 / H, s.subtitle, fontsize=18, color=MUTED, va="center")
        self.ftxt = fig.text(1405 / W, 1046 / H, "", fontsize=30, fontweight="bold", ha="right", va="center")
        self.ptxt = fig.text(1405 / W, 1006 / H, "", fontsize=17, color=MUTED, ha="right", va="center")
        self.axw = rect_px(fig, 1470, 996, 1880, 1072)
        self.axw.set_xlim(0, 2)
        self.axw.set_ylim(-1.25, 1.25)
        self.axw.axis("off")
        self.axw.plot([0, 2], [0, 0], color=RULE, lw=1)
        fig.text(1452 / W, 1052 / H, "V", fontsize=17, ha="right", va="center", style="italic")
        self.itxt = fig.text(1452 / W, 1016 / H, "I", fontsize=17, ha="right", va="center", style="italic")
        (self.vline,) = self.axw.plot([], [], color=INK, lw=2.4)
        (self.iline,) = self.axw.plot([], [], color=INK, lw=2.4, ls=(0, (1, 0)))
        fig.add_artist(Line2D([30 / W, 1890 / W], [980 / H] * 2, color=RULE, lw=1.2))

        self.axs = rect_px(fig, 40, 612, 760, 972)
        self.regions = schematic(self.axs, s.key)
        self.zt = fig.text(40 / W, 578 / H, "", fontsize=19, fontweight="bold", va="center", zorder=5,
                            bbox=dict(fc=PAPER, ec="none", pad=2))
        self.zt2 = fig.text(760 / W, 578 / H, "motion slowed down for display", fontsize=14, color=MUTED,
                             ha="right", va="center", zorder=5, bbox=dict(fc=PAPER, ec="none", pad=2))
        self.axz = rect_px(fig, 40, 182, 760, 562)
        self.axz.set_xlim(0, 100)
        self.axz.set_ylim(0, 52.8)
        self.axz.set_xticks([])
        self.axz.set_yticks([])

        # plots
        for x, y, t in ((880, 972, "Nyquist plot"), (1440, 972, "Bode phase"), (820, 548, "Equivalent circuit"),
                        (1440, 548, "Distribution of relaxation times (DRT)")):
            fig.text(x / W, y / H, t, fontsize=19, fontweight="bold", va="center")
        self.axn = rect_px(fig, 900, 664, 1340, 952)
        self.axb = rect_px(fig, 1440, 664, 1880, 952)
        self.axc = rect_px(fig, 810, 212, 1350, 528)
        self.axd = rect_px(fig, 1440, 262, 1880, 524)
        self._setup_plots()
        cb = circuit(self.axc, s, 540, 316)
        self.cboxes = cb

        # caption
        fig.add_artist(FancyBboxPatch((40 / W, 20 / H), 1840 / W, 140 / H, boxstyle="round,pad=0,rounding_size=0.006",
                                      fc="white", ec=RULE, lw=1.2, transform=fig.transFigure))
        self.capbar = fig.add_artist(Rectangle((40 / W, 20 / H), 10 / W, 140 / H, fc=INK, ec="none",
                                               transform=fig.transFigure))
        self.cap = fig.text(72 / W, 90 / H, "", fontsize=23, va="center", linespacing=1.18)

        # dynamic artists (current band)
        mk = dict(ls="none", marker="o", ms=8, mec="white", mew=1.0)
        (self.dn_line,) = self.axn.plot([], [], lw=2.6, zorder=4)
        (self.dn_dots,) = self.axn.plot([], [], zorder=5, **mk)
        (self.db_line,) = self.axb.plot([], [], lw=2.6, zorder=4)
        (self.db_dots,) = self.axb.plot([], [], zorder=5, **mk)
        (self.cur_n,) = self.axn.plot([], [], "o", ms=17, mfc="none", mew=3, zorder=6)
        (self.cur_b,) = self.axb.plot([], [], "o", ms=17, mfc="none", mew=3, zorder=6)
        self.cur_bv = self.axb.axvline(1, color=INK, lw=1.2, ls=(0, (3, 3)), zorder=3)
        self.cur_dv = self.axd.axvline(1, color=INK, lw=1.2, ls=(0, (3, 3)), zorder=3)
        self.dd_fill = self.axd.add_patch(Polygon(np.zeros((3, 2)), closed=True, lw=0, zorder=2))
        (self.dd_line,) = self.axd.plot([], [], lw=2.6, zorder=4)
        self.base_dyn = [self.ftxt, self.ptxt, self.vline, self.iline, self.dn_line, self.dn_dots, self.db_line,
                         self.db_dots, self.dd_fill, self.dd_line, self.cur_bv, self.cur_dv, self.cur_n, self.cur_b]
        self.state_artists = []
        self.anim = None
        self.state = None
        self.bg = None

    # --------------------------------------------------------------------------------- #
    def _setup_plots(self):
        s = self.s
        ax = self.axn
        x0, x1, y0, _ = s.nyq_view
        y1 = y0 + (x1 - x0) * 288 / 440
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_facecolor("white")
        ax.axhline(0, color=RULE, lw=1.2, zorder=1)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(f"Z′ ({s.unit})", labelpad=2)
        ax.set_ylabel(f"−Z″ ({s.unit})", labelpad=4)
        Z = self.Zl
        inside = (Z.real >= x0) & (Z.real <= x1) & (-Z.imag >= y0) & (-Z.imag <= y1)
        if not inside.all():
            print(f"  [{s.key}] {np.sum(~inside)} of {len(Z)} Nyquist points outside the view "
                  f"(f < {self.fl[~inside].max():.3g} Hz)" if (~inside).any() else "")
        ph = -np.degrees(np.angle(self.Zl))
        lo = min(0, np.floor(ph.min() / 15) * 15)
        hi = max(45, np.ceil((ph.max() + 5) / 15) * 15)
        for a in (self.axb, self.axd):
            a.set_xscale("log")
            a.set_xlim(s.fmin / 10 ** 0.15, s.fmax * 10 ** 0.15)
            log_axis(a)
            a.set_facecolor("white")
            a.set_xlabel("frequency (Hz)", labelpad=2)
        self.axb.set_ylim(lo, hi)
        self.axb.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 1.5, 3, 5, 10]))
        self.axb.axhline(0, color=RULE, lw=1.2, zorder=1)
        self.axb.set_ylabel("−phase (°)")
        self.axd.set_ylim(0, self.dgam.max() * 1.42)
        self.axd.yaxis.set_major_locator(MaxNLocator(4))
        self.axd.set_ylabel(f"γ ({s.unit})")
        self.axd.text(0.985, 0.955, "peak at f = 1/(2πτ),  area = R", transform=self.axd.transAxes,
                      ha="right", va="top", fontsize=14, color=MUTED, zorder=8,
                      bbox=dict(fc="white", ec="none", pad=1.5))
        if s.drt_fmin > s.fmin:
            self.axd.axvspan(s.fmin / 10, s.drt_fmin, color="#E6EAED", lw=0, zorder=0)
        if s.drt_fmin > s.fmin * 10:
            self.axd.text(np.sqrt(s.fmin * s.drt_fmin), self.dgam.max() * 0.55,
                          "not used\nfor the DRT", ha="center", va="center", fontsize=13, color=MUTED)
        for a in (self.axn, self.axb, self.axd):
            a.grid(True, color="#E4E9EC", lw=0.8)
            a.set_axisbelow(True)

    def _seg(self, key, fa, fb, dense=True):
        """Data of band `key` between fa > fb (dense line or dots)."""
        f, Z = (self.fl, self.Zl) if dense else (self.fd, self.Zd)
        m = (f <= fa * (1 + 1e-9)) & (f >= fb * (1 - 1e-9))
        if dense:
            # include the neighbours so consecutive bands join up
            idx = np.where(m)[0]
            if len(idx):
                lo, hi = max(idx[0] - 1, 0), min(idx[-1] + 1, len(f) - 1)
                m[lo:hi + 1] = True
        return f[m], Z[m]

    def _ph(self, Z):
        return -np.degrees(np.angle(Z))

    # --------------------------------------------------------------------------------- #
    def set_state(self, band_idx, caption, summary=False, hold=False):
        """band_idx: index of the band being swept (all earlier bands drawn complete)."""
        for a in self.state_artists:
            a.remove()
        self.state_artists = []
        if self.anim:
            self.anim.remove()
            self.anim = None
        s = self.s
        SA = self.state_artists.append
        n_done = len(self.bands) if summary else band_idx
        for key, fa, fb in self.bands[:n_done]:
            c = s.block(key).color
            f, Z = self._seg(key, fa, fb)
            SA(self.axn.plot(Z.real, -Z.imag, color=c, lw=2.6, zorder=4)[0])
            SA(self.axb.plot(f, self._ph(Z), color=c, lw=2.6, zorder=4)[0])
            f, Z = self._seg(key, fa, fb, dense=False)
            SA(self.axn.plot(Z.real, -Z.imag, "o", color=c, ms=8, mec="white", mew=1, zorder=5)[0])
            SA(self.axb.plot(f, self._ph(Z), "o", color=c, ms=8, mec="white", mew=1, zorder=5)[0])
            self._drt_part(key, fa, fb, fa, final=True)
        active = None if summary else self.bands[band_idx][0]
        blk = s.block(active) if active else None
        col = blk.color if blk else INK
        # caption
        self.cap.set_text(wrap(caption))
        self.capbar.set_facecolor(col)
        # highlight in schematic + connector + zoom
        if blk:
            x, y, w, h = self.regions[blk.region]
            SA(self.axs.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", fc=tint(col, 0.18),
                                                 ec=col, lw=3, zorder=6, alpha=0.9)))
            tr = self.axs.transData
            for (px, py), (qx, qy) in (((x, y), (40, 562)), ((x + w, y), (760, 562))):
                X, Y = tr.transform((px, py))
                SA(self.fig.add_artist(Line2D([X / W, qx / W], [Y / H, qy / H], color=col, lw=1.6,
                                              ls=(0, (4, 3)))))
            for sp in self.axz.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(3.5)
            self.zt.set_text(f"Zoom: {blk.label}")
            self.zt.set_color(col)
            self.zt2.set_visible(True)
            self.axz.set_facecolor("white")
            opts = dict(ZOOM_OPTS.get((s.key, blk.key), {}))
            if blk.zoom == "diffusion":
                opts["tau"] = blk.params[1]
            if blk.zoom == "proton":
                opts.setdefault("style", "membrane")
            self.anim = ANIMS[blk.zoom](self.axz, col, opts)
            bx, by, bw, bh = self.cboxes[blk.key]
            SA(self.axc.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=2,rounding_size=6",
                                                 fc=tint(col, 0.14), ec=col, lw=2.5, zorder=0)))
        else:
            for sp in self.axz.spines.values():
                sp.set_edgecolor(RULE)
                sp.set_linewidth(1.2)
            self.zt.set_text("The processes in this cell")
            self.zt.set_color(INK)
            self.zt2.set_visible(False)
            self._process_table(SA)
        if summary:
            self._annotate(SA)
        # V/I colours
        self.iline.set_color(col)
        self.itxt.set_color(col)
        for a in (self.dn_line, self.dn_dots, self.db_line, self.db_dots, self.dd_line, self.cur_n, self.cur_b):
            a.set_color(col)
        self.dn_dots.set_markeredgecolor("white")
        self.db_dots.set_markeredgecolor("white")
        self.dd_fill.set_facecolor(tint(col, 0.35))
        self.state = dict(band=band_idx, summary=summary, hold=hold, active=active)
        dyn = self.base_dyn + (self.anim.dyn if self.anim else [])
        for a in dyn:
            a.set_visible(False)
        self.canvas.draw()
        self.bg = self.canvas.copy_from_bbox(self.fig.bbox)
        for a in dyn:
            a.set_visible(True)
        self.dyn = dyn

    def _drt_part(self, key, fa, fb, fcur, final=False):
        """DRT revealed for frequencies between fa and max(fb, fcur), coloured by block assignment."""
        lo = max(fb, fcur) if not final else fb
        m = (self.df <= fa * 1.0001) & (self.df >= lo) & (self.df >= self.s.drt_fmin * 0.999)
        # extend by one point so neighbouring parts join
        idx = np.where(m)[0]
        if len(idx) == 0:
            return None
        lo_i, hi_i = max(idx[0] - 1, 0), min(idx[-1] + 1, len(m) - 1)
        x, y = self.df[lo_i:hi_i + 1], self.dgam[lo_i:hi_i + 1]
        col = self.s.block(key).color
        if final:
            poly = np.r_[np.c_[x, y], [[x[-1], 0], [x[0], 0]]]
            self.state_artists.append(self.axd.add_patch(Polygon(poly, closed=True, fc=tint(col, 0.3), lw=0,
                                                                 zorder=2)))
            self.state_artists.append(self.axd.plot(x, y, color=col, lw=2.6, zorder=4)[0])
        return x, y

    def _process_table(self, SA):
        s = self.s
        ax = self.axz
        SA(ax.add_patch(Rectangle((0, 0), 100, 52.8, fc="white", lw=0, zorder=0)))
        y = 46
        SA(ax.text(3, y, "process", fontsize=16, color=MUTED, va="center"))
        SA(ax.text(52, y, "circuit", fontsize=16, color=MUTED, va="center"))
        SA(ax.text(70, y, "value", fontsize=16, color=MUTED, va="center"))
        SA(ax.text(88, y, "1/(2πτ)", fontsize=16, color=MUTED, va="center"))
        y -= 8.5 if len(s.blocks) <= 4 else 7.6
        step = 8.5 if len(s.blocks) <= 4 else 7.6
        for b in s.blocks:
            SA(ax.add_patch(Rectangle((3, y - 1.6), 3.2, 3.2, fc=b.color, lw=0)))
            SA(ax.text(8, y, b.label, fontsize=18, va="center", color=INK))
            SA(ax.text(52, y, ", ".join(cd._label(e) for e in b.elements), fontsize=18, va="center", color=b.color))
            if b.fragment.startswith("L"):
                val = f"{b.params[0] * 1e9:.0f} nH"
            else:
                val = f"{b.params[0] * s.scale:.3g} {s.unit}" if not b.fragment.startswith("CPE") else "—"
            SA(ax.text(70, y, val, fontsize=18, va="center", color=INK))
            fp = fmt_f(1 / (2 * np.pi * b.tau)) if b.tau else "—"
            SA(ax.text(88, y, fp, fontsize=18, va="center", color=INK))
            y -= step
        SA(ax.text(3, 3, "values are illustrative", fontsize=14, color=MUTED, va="center"))

    def _annotate(self, SA):
        s = self.s
        x0, x1 = self.axn.get_xlim()
        y0, y1 = self.axn.get_ylim()
        for b in s.blocks:
            if b.tau and not b.fragment.startswith("Wo"):
                fc = 1 / (2 * np.pi * b.tau)
                Z = s.Z(np.array([fc]))[0] * s.scale
                SA(self.axn.annotate(b.label, (Z.real, -Z.imag), xytext=(0, 26), textcoords="offset points",
                                     ha="center", fontsize=15, color=b.color, fontweight="bold",
                                     bbox=dict(fc="white", ec="none", pad=1, alpha=0.85), zorder=7))
        for b in s.blocks:
            if b.fragment.startswith(("CPE", "Wo")):
                Z = s.Z(np.array([s.fmin * 1.3]))[0] * s.scale
                SA(self.axn.annotate(b.label, (Z.real, -Z.imag), xytext=(-14, 0), textcoords="offset points",
                                     ha="right", va="center", fontsize=15, color=b.color, fontweight="bold",
                                     bbox=dict(fc="white", ec="none", pad=1, alpha=0.85), zorder=7))
        # DRT peak labels: the largest peak of each block
        best = {}
        for p in self.drt.peaks:
            k = p.get("block")
            if k and p["f_Hz"] >= s.drt_fmin and (k not in best or p["R_ohm"] > best[k]["R_ohm"]):
                best[k] = p
        for p in best.values():
            b = s.block(p["block"])
            g = np.interp(np.log(p["tau_s"]), np.log(self.dtau), self.dgam)
            SA(self.axd.annotate(cd._label(b.elements[0]), (p["f_Hz"], g), xytext=(0, 8), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=17, color=b.color, zorder=7))

    # --------------------------------------------------------------------------------- #
    def frame(self, t, f):
        """Render one frame at video time t and measurement frequency f. Returns an RGB array."""
        s, st = self.s, self.state
        self.canvas.restore_region(self.bg)
        Zf = s.Z(np.array([f]))[0] * s.scale
        ang = np.angle(Zf)
        self.ftxt.set_text(f"f = {fmt_f(f)}")
        deg = int(round(np.degrees(ang)))
        self.ptxt.set_text(f"|Z| = {abs(Zf):.3g} {s.unit}    phase = {deg:+d}°".replace("+0°", "0°").replace("-", "−"))
        x = np.linspace(0, 2, 200)
        w = 2 * np.pi * NU * t
        self.vline.set_data(x, np.sin(2 * np.pi * x + w))
        self.iline.set_data(x, 0.8 * np.sin(2 * np.pi * x + w - ang))
        empty = ([], [])
        if not st["summary"] and not st["hold"]:
            key, fa, fb = self.bands[st["band"]]
            fl, Zl = self._seg(key, fa, max(fb, f))
            mm = fl >= f
            fl, Zl = fl[mm], Zl[mm]
            fl, Zl = np.r_[fl, f], np.r_[Zl, Zf]
            self.dn_line.set_data(Zl.real, -Zl.imag)
            self.db_line.set_data(fl, self._ph(Zl))
            fd, Zd = self._seg(key, fa, max(fb, f), dense=False)
            self.dn_dots.set_data(Zd.real, -Zd.imag)
            self.db_dots.set_data(fd, self._ph(Zd))
            r = self._drt_part(key, fa, fb, f)
            if r is not None:
                xx, yy = r
                xx, yy = xx[xx >= f], yy[xx >= f]
            if r is not None and len(xx) > 1:
                self.dd_line.set_data(xx, yy)
                self.dd_fill.set_xy(np.r_[np.c_[xx, yy], [[xx[-1], 0], [xx[0], 0]]])
                self.dd_fill.set_visible(True)
            else:
                self.dd_line.set_data(*empty)
                self.dd_fill.set_visible(False)
        else:
            for a in (self.dn_line, self.dn_dots, self.db_line, self.db_dots, self.dd_line):
                a.set_data(*empty)
            self.dd_fill.set_visible(False)
        show_cur = not st["summary"]
        self.cur_n.set_data([Zf.real], [-Zf.imag]) if show_cur else self.cur_n.set_data(*empty)
        self.cur_b.set_data([f], [np.degrees(-ang)]) if show_cur else self.cur_b.set_data(*empty)
        self.cur_bv.set_xdata([f, f])
        self.cur_dv.set_xdata([f, f])
        self.cur_bv.set_visible(show_cur)
        self.cur_dv.set_visible(show_cur)
        if self.anim:
            self.anim.update(t, f, -ang)
        for a in self.dyn:
            if a.get_visible():
                self.fig.draw_artist(a)
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3]


# ======================================================================================= #
# Cards
# ======================================================================================= #
class Card:
    def __init__(self):
        self.fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=PAPER)
        self.canvas = FigureCanvasAgg(self.fig)
        self.key = None
        self.img = None

    def render(self, key, build):
        if key != self.key:
            self.fig.clear()
            build(self.fig)
            self.canvas.draw()
            self.img = np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()
            self.key = key
        return self.img

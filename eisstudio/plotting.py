"""Publication-quality figures.

Design choices (deliberate, so reviewers don't bounce the figure):
* Nyquist plots use an equal aspect ratio. A stretched Nyquist plot misrepresents
  depressed vs ideal arcs; this is a common reviewer complaint.
* Data = open black circles, fit = solid line (Okabe-Ito vermillion); residual
  Re/Im use blue circles vs vermillion squares, so colour is never the only cue.
* Text is kept as text in PDF/SVG (fonttype 42 / 'none') so it stays editable in
  Illustrator/Inkscape. PNG export defaults to 600 dpi.
* Widths follow common journal columns: single 3.5 in (8.9 cm), 1.5-col 5.5 in,
  double 7.0 in (17.8 cm).
"""

from __future__ import annotations

import functools
import io
import threading
from contextlib import contextmanager

import matplotlib
from pyparsing import ParserElement

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (rc_context only; figures are not pyplot-managed)
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# Matplotlib is not thread-safe (its mathtext parser and rcParams are shared). Streamlit runs every
# user session, and every deferred download, in its own thread, so all figure building and
# rendering goes through one process-wide lock.
_MPL_LOCK = threading.RLock()


def _locked(fn):
    """Run ``fn`` under the process-wide matplotlib lock.

    Mathtext uses pyparsing's global packrat cache and clears it only after a *successful* parse.
    A parse interrupted by an exception (historically: a concurrent parse emptying the parser's
    state) leaves the failure cached, and every later render of the same label fails with
    "Expected end of text". So the cache is cleared before each call, and once more before a
    single retry if a mathtext parse error escapes.
    """
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with _MPL_LOCK:
            ParserElement.reset_cache()
            try:
                return fn(*a, **k)
            except ValueError as exc:
                if "ParseException" not in str(exc) and "ParseFatalException" not in str(exc):
                    raise
                ParserElement.reset_cache()
                return fn(*a, **k)
    wrapper._eis_locked = True
    return wrapper


# Save-time settings are global (they are read at savefig, possibly on a download thread):
# keep text as text in vector output so it stays editable.
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

DATA = "#000000"
FIT = "#D55E00"   # vermillion
RE_C = "#0072B2"  # blue
IM_C = "#D55E00"  # vermillion
BAND = "#0072B2"

WIDTHS = {"Single column (3.5 in)": 3.5, "1.5 column (5.5 in)": 5.5, "Double column (7.0 in)": 7.0}
UNIT_MODES = ("Ω", "Ω·cm² (area-specific)", "Ω·cm (resistivity)")


def unit_scale(mode, area_cm2=None, thickness_cm=None):
    """Return (factor, label). Impedance values are multiplied by factor before plotting."""
    if mode.startswith("Ω·cm²") and area_cm2:
        return area_cm2, r"Ω cm$^2$"
    if mode.startswith("Ω·cm (") and area_cm2 and thickness_cm:
        return area_cm2 / thickness_cm, r"Ω cm"
    return 1.0, "Ω"


@contextmanager
def pub_style(font_size=8):
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "mathtext.default": "regular",
        "axes.formatter.use_mathtext": True,
    }
    with plt.rc_context(rc):
        yield


def _log_minor(ax, axis="x"):
    loc = LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(loc)
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_formatter(NullFormatter())


def _panel_label(ax, label):
    if label:
        ax.text(-0.02, 1.02, f"({label})", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")


# --------------------------------------------------------------------------- #
# Individual panels
# --------------------------------------------------------------------------- #
def ax_nyquist(ax, Z, Z_line=None, Z_pts=None, scale=1.0, unit="Ω", freq=None, decades=True,
               fit_label="Fit", label=None, zoom=None):
    """zoom: optional max Z' (in plotted units) to show, e.g. to hide a blocking-electrode tail."""
    ax.plot(Z.real * scale, -Z.imag * scale, "o", mfc="none", mec=DATA, mew=0.8, label="Data", zorder=3)
    if Z_line is not None:
        ax.plot(Z_line.real * scale, -Z_line.imag * scale, "-", color=FIT, label=fit_label, zorder=4)
    elif Z_pts is not None:
        ax.plot(Z_pts.real * scale, -Z_pts.imag * scale, "-", color=FIT, label=fit_label, zorder=4)
    if decades and freq is not None:
        lf = np.log10(freq)
        xs, ys = Z.real * scale, -Z.imag * scale
        diag = max(np.hypot(np.ptp(xs), np.ptp(ys)), 1e-30)
        placed = []
        for d in range(int(np.ceil(lf.min())), int(np.floor(lf.max())) + 1):
            i = int(np.argmin(np.abs(lf - d)))
            if abs(lf[i] - d) > 0.08:
                continue
            ax.plot(xs[i], ys[i], "o", ms=4, color=DATA, zorder=5)
            # label only if not crowding an already-labelled decade (>4 % of the plot diagonal away)
            if all(np.hypot(xs[i] - px, ys[i] - py) > 0.04 * diag for px, py in placed):
                ax.annotate(f"$10^{{{d}}}$ Hz" if not placed else f"$10^{{{d}}}$", (xs[i], ys[i]), xytext=(3, 4),
                            textcoords="offset points", fontsize=matplotlib.rcParams["font.size"] - 2,
                            color="#444444")
                placed.append((xs[i], ys[i]))
    ax.set_xlabel(f"Z′ / {unit}")
    ax.set_ylabel(f"−Z″ / {unit}")
    ax._nyq_xy = (Z.real * scale, -Z.imag * scale)  # used by equalize_nyquist() after layout
    ax._nyq_zoom = zoom
    ax.axhline(0, color="#999999", lw=0.5, zorder=1)
    ax.legend(loc="best")
    _panel_label(ax, label)


def ax_bode_mod(ax, f, Z, f_line=None, Z_line=None, scale=1.0, unit="Ω", label=None):
    ax.loglog(f, np.abs(Z) * scale, "o", mfc="none", mec=DATA, mew=0.8, label="Data")
    if Z_line is not None:
        ax.loglog(f_line, np.abs(Z_line) * scale, "-", color=FIT, label="Fit")
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel(f"|Z| / {unit}")
    _log_minor(ax, "x")
    ax.legend(loc="best")
    _panel_label(ax, label)


def ax_bode_phase(ax, f, Z, f_line=None, Z_line=None, label=None):
    ax.semilogx(f, -np.degrees(np.angle(Z)), "o", mfc="none", mec=DATA, mew=0.8, label="Data")
    if Z_line is not None:
        ax.semilogx(f_line, -np.degrees(np.angle(Z_line)), "-", color=FIT, label="Fit")
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("−Phase / °")
    _log_minor(ax, "x")
    ax.legend(loc="best")
    _panel_label(ax, label)


def ax_residuals(ax, f, r_re, r_im, label=None, title=None):
    ax.semilogx(f, r_re, "o-", color=RE_C, mfc="none", ms=3.5, lw=0.7, label=r"$\Delta_{re}$")
    ax.semilogx(f, r_im, "s-", color=IM_C, mfc="none", ms=3.5, lw=0.7, label=r"$\Delta_{im}$")
    ax.axhline(0, color="#666666", lw=0.6)
    lim = max(1.0, float(np.nanmax(np.abs(np.concatenate([r_re, r_im])))) * 1.15)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel(r"Residual / % of |Z|")
    _log_minor(ax, "x")
    ax.legend(loc="best", ncol=2)
    if title:
        ax.set_title(title, loc="left")
    _panel_label(ax, label)


def ax_drt(ax, tau, gamma, lo=None, hi=None, mean=None, peaks=None, f_window=None, label=None,
           gamma_mode="gamma"):
    """gamma_mode: 'gamma' -> gamma(ln tau) in Ohm; 'g' -> g(tau) = gamma/tau in Ohm/s."""
    y = gamma if gamma_mode == "gamma" else gamma / tau
    ylabel = r"$\gamma(\ln\tau)$ / Ω" if gamma_mode == "gamma" else r"$g(\tau) = \gamma/\tau$ / Ω s$^{-1}$"
    if f_window is not None:
        tmin, tmax = 1 / (2 * np.pi * f_window[1]), 1 / (2 * np.pi * f_window[0])
        ax.axvspan(tau.min(), tmin, color="#000000", alpha=0.06, lw=0)
        ax.axvspan(tmax, tau.max(), color="#000000", alpha=0.06, lw=0)
    if lo is not None and hi is not None:
        ylo, yhi = (lo, hi) if gamma_mode == "gamma" else (lo / tau, hi / tau)
        ax.fill_between(tau, ylo, yhi, color=BAND, alpha=0.2, lw=0, label="99% credible band")
    ax.plot(tau, y, "-", color=DATA, lw=1.3, label="DRT" if lo is None else "MAP")
    if mean is not None:
        ax.plot(tau, mean if gamma_mode == "gamma" else mean / tau, "--", color=BAND, lw=1.0, label="Posterior mean")
    if peaks:
        for p in peaks:
            yp = p["gamma_max_ohm"] if gamma_mode == "gamma" else p["gamma_max_ohm"] / p["tau_s"]
            ax.annotate(f"P{p['peak']}" + ("*" if p.get("minor") else ""), (p["tau_s"], yp), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=matplotlib.rcParams["font.size"] - 1,
                        color="#888888" if p.get("minor") else "#000000")
    ax.set_xscale("log")
    ax.set_xlim(tau.min(), tau.max())
    if gamma_mode == "g":
        ax.set_yscale("log")
    else:
        top = float(np.nanmax(hi if (hi is not None) else y))
        ax.set_ylim(min(0, float(np.nanmin(y))), top * 1.15 if top > 0 else 1)  # headroom for peak labels
    _log_minor(ax, "x")
    ax.set_xlabel(r"$\tau$ / s")
    ax.set_ylabel(ylabel)
    sec = ax.secondary_xaxis("top", functions=(lambda t: 1 / (2 * np.pi * np.maximum(t, 1e-300)),
                                                lambda f: 1 / (2 * np.pi * np.maximum(f, 1e-300))))
    sec.set_xlabel(r"$f = 1/(2\pi\tau)$ / Hz")
    sec.tick_params(direction="in")
    ax.tick_params(top=False)
    if (lo is not None) or (mean is not None):
        ax.legend(loc="best")
    _panel_label(ax, label)


def equalize_nyquist(fig):
    """Equal data scaling on every Nyquist axis without wasting the panel.

    Called after tight_layout: we read each axis' physical size and choose limits so one Ohm
    is the same length on both axes, keeping -Z'' >= small negative margin instead of
    centring (which would leave the lower half of the panel empty).
    """
    fig.canvas.draw()
    W, H = fig.get_size_inches()
    for ax in fig.axes:
        if not hasattr(ax, "_nyq_xy"):
            continue
        x, y = ax._nyq_xy
        lines = [ln for ln in ax.get_lines() if np.asarray(ln.get_xdata()).size > 2]
        xs = np.concatenate([np.asarray(ln.get_xdata(), float) for ln in lines] + [x])
        ys = np.concatenate([np.asarray(ln.get_ydata(), float) for ln in lines] + [y])
        zoom = getattr(ax, "_nyq_zoom", None)
        bb = ax.get_position()
        ratio = (bb.height * H) / (bb.width * W)  # physical height / width
        if zoom and np.sum(xs <= zoom) >= 2:
            # fixed Z' window; -Z'' follows from equal scaling, the tail is clipped
            x0 = np.nanmin(xs[xs <= zoom])
            x0, x1 = x0 - 0.04 * (zoom - x0), zoom
            y0 = min(np.nanmin(ys[xs <= zoom]), 0.0) - 0.04 * (x1 - x0)
            y1 = y0 + (x1 - x0) * ratio
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            continue
        x0, x1 = np.nanmin(xs), np.nanmax(xs)
        y0, y1 = min(np.nanmin(ys), 0.0), np.nanmax(ys)
        pad = 0.06 * max(x1 - x0, y1 - y0)
        x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
        if (y1 - y0) / (x1 - x0) > ratio:   # tall data: widen x to the right of the data
            x1 = x0 + (y1 - y0) / ratio
        else:                                # wide data: extend y upwards only
            y1 = y0 + (x1 - x0) * ratio
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig(width, aspect):
    # Plain Figure objects (not pyplot) are garbage-collected when the app drops them;
    # pyplot would keep every figure from every rerun alive.
    fig = Figure(figsize=(width, width * aspect))
    FigureCanvasAgg(fig)
    return fig


@_locked
def fig_single(kind, width, **kw):
    """kind in nyquist | bode | residuals | drt. Returns a Figure."""
    with pub_style(kw.pop("font_size", 8)):
        if kind == "bode":
            fig = _fig(width, 0.45 if width > 4 else 0.85)
            if width > 4:
                a1, a2 = fig.subplots(1, 2)
            else:
                a1, a2 = fig.subplots(2, 1, sharex=True)
            ax_bode_mod(a1, kw["f"], kw["Z"], kw.get("f_line"), kw.get("Z_line"), kw.get("scale", 1), kw.get("unit", "Ω"), "a")
            ax_bode_phase(a2, kw["f"], kw["Z"], kw.get("f_line"), kw.get("Z_line"), "b")
            if width <= 4:
                a1.set_xlabel("")
        else:
            fig = _fig(width, {"nyquist": 0.8, "residuals": 0.6, "drt": 0.7}[kind])
            ax = fig.add_subplot(111)
            if kind == "nyquist":
                ax_nyquist(ax, kw["Z"], kw.get("Z_line"), kw.get("Z_pts"), kw.get("scale", 1), kw.get("unit", "Ω"),
                           kw.get("f"), kw.get("decades", True), kw.get("fit_label", "Fit"), zoom=kw.get("zoom"))
            elif kind == "residuals":
                ax_residuals(ax, kw["f"], kw["r_re"], kw["r_im"])
            elif kind == "drt":
                ax_drt(ax, kw["tau"], kw["gamma"], kw.get("lo"), kw.get("hi"), kw.get("mean"), kw.get("peaks"),
                       kw.get("f_window"), gamma_mode=kw.get("gamma_mode", "gamma"))
        fig.tight_layout()
        equalize_nyquist(fig)
    return fig


@_locked
def fig_ecm_composite(width, f, Z, f_line, Z_line, Z_fit_pts, r_re, r_im, scale=1.0, unit="Ω", decades=True,
                      font_size=8, zoom=None):
    with pub_style(font_size):
        fig = _fig(width, 0.78)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        ax_nyquist(fig.add_subplot(gs[0, 0]), Z, Z_line, None, scale, unit, f, decades, label="a", zoom=zoom)
        ax_bode_mod(fig.add_subplot(gs[0, 1]), f, Z, f_line, Z_line, scale, unit, label="b")
        ax_bode_phase(fig.add_subplot(gs[1, 0]), f, Z, f_line, Z_line, label="c")
        ax_residuals(fig.add_subplot(gs[1, 1]), f, r_re, r_im, label="d")
        fig.tight_layout()
        equalize_nyquist(fig)
    return fig


@_locked
def fig_drt_composite(width, f, Z, Z_fit, r_re, r_im, tau, gamma, lo=None, hi=None, mean=None, peaks=None,
                      scale=1.0, unit="Ω", decades=True, gamma_mode="gamma", font_size=8, zoom=None):
    with pub_style(font_size):
        fig = _fig(width, 0.95)
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.05])
        ax_nyquist(fig.add_subplot(gs[0, 0]), Z, None, Z_fit, scale, unit, f, decades, fit_label="DRT fit", label="a",
                   zoom=zoom)
        ax_drt(fig.add_subplot(gs[0, 1]), tau, gamma, lo, hi, mean, peaks, (f.min(), f.max()), label="b",
               gamma_mode=gamma_mode)
        ax_bode_mod(fig.add_subplot(gs[1, 0]), f, Z, f, Z_fit, scale, unit, label="c")
        ax_bode_phase(fig.add_subplot(gs[1, 1]), f, Z, f, Z_fit, label="d")
        ax_residuals(fig.add_subplot(gs[2, :]), f, r_re, r_im, label="e")
        fig.tight_layout()
        equalize_nyquist(fig)
    return fig


@_locked
def fig_kk(width, f, r_re, r_im, M, font_size=8):
    with pub_style(font_size):
        fig = _fig(width, 0.55)
        ax = fig.add_subplot(111)
        ax_residuals(ax, f, r_re, r_im, title=f"Lin-KK residuals (M = {M} RC elements)")
        fig.tight_layout()
        equalize_nyquist(fig)
    return fig


@_locked
def to_bytes(fig, fmt="png", dpi=600):
    """Thread-safe export: no rcParams changes here (styling was fixed at figure creation)."""
    buf = io.BytesIO()
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt == "tiff" else {}
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.03, **kw)
    return buf.getvalue()


MIME = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml", "tiff": "image/tiff"}

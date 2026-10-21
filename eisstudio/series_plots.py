"""Figures for series analysis: overlaid spectra, Arrhenius plot, DC polarisation.

Same conventions as plotting.py: equal-scaled Nyquist axes, text kept as text in vector output,
journal column widths. Spectra are coloured by temperature (sequential, cold = dark blue, hot =
orange) when every spectrum has a temperature, otherwise with the Okabe-Ito qualitative palette;
every curve is also named in the legend, so colour is never the only cue.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib import colormaps, gridspec

from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from .plotting import _fig, _locked, _log_minor, _panel_label, equalize_nyquist, pub_style
from .series import tail_start

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#882255",
             "#44AA99", "#999933"]


def series_colors(temps):
    """List of colours; sequential by temperature if all are given and not all equal."""
    t = np.array([np.nan if v is None else float(v) for v in temps], float)
    if t.size and np.all(np.isfinite(t)) and np.ptp(t) > 0:
        cmap = colormaps["plasma"]
        u = (t - t.min()) / np.ptp(t)
        return [matplotlib.colors.to_hex(cmap(0.05 + 0.75 * x)) for x in u], True
    return [OKABE_ITO[i % len(OKABE_ITO)] for i in range(t.size)], False


def current_unit(values):
    v = np.nanmedian(np.abs(np.asarray(values, float))) if len(values) else 1.0
    for f, u in ((1e-12, "pA"), (1e-9, "nA"), (1e-6, "µA"), (1e-3, "mA")):
        if v < f * 1e3:
            return f, u
    return 1.0, "A"


def _legend(ax, n):
    ax.legend(loc="best", ncol=1 if n <= 6 else 2, handlelength=1.4, borderaxespad=0.3,
              fontsize=matplotlib.rcParams["legend.fontsize"] - (1 if n > 6 else 0))


def auto_zoom(items):
    """Max Z' that shows every arc but hides blocking spikes: 1.15 x the largest Z' at which a
    spike starts. None when no spectrum has a spike."""
    ends = []
    for it in items:
        j = tail_start(it["f"], it["Z"])
        if j is not None:
            ends.append(float(it["Z"].real[j]))
        else:
            ends.append(float(np.max(it["Z"].real)))
    if not any(tail_start(it["f"], it["Z"]) is not None for it in items):
        return None
    lo = min(float(np.min(it["Z"].real)) for it in items)
    hi = max(ends)
    return hi + 0.15 * (hi - min(lo, 0.0))


@_locked
def fig_series_eis(width, items, unit="Ω", font_size=8, zoom="auto", drt_x="tau"):
    """items: dicts with label, color, f, Z (already normalised), optional f_line/Z_line (fit) and
    tau/gamma (DRT, normalised with the same factor). zoom: 'auto' hides blocking spikes, None or
    0 shows everything, a number is the largest Z' shown."""
    with pub_style(font_size):
        fig = _fig(width, 0.80)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        axn, axm = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        axp, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        xs, ys = [], []
        for it in items:
            c, Z, f = it["color"], it["Z"], it["f"]
            kw = dict(mfc="none", mec=c, mew=0.7, ms=3.2, ls="none")
            axn.plot(Z.real, -Z.imag, "o", label=it["label"], **kw)
            axm.loglog(f, np.abs(Z), "o", **kw)
            axp.semilogx(f, -np.degrees(np.angle(Z)), "o", **kw)
            if it.get("Z_line") is not None:
                Zl = it["Z_line"]
                axn.plot(Zl.real, -Zl.imag, "-", color=c, lw=0.9)
                axm.loglog(it["f_line"], np.abs(Zl), "-", color=c, lw=0.9)
                axp.semilogx(it["f_line"], -np.degrees(np.angle(Zl)), "-", color=c, lw=0.9)
            xs.append(Z.real)
            ys.append(-Z.imag)
            if it.get("tau") is not None:
                x = it["tau"] if drt_x == "tau" else 1 / (2 * np.pi * it["tau"])
                axd.plot(x, it["gamma"], "-", color=c, lw=1.1, label=it["label"])
        axn.set_xlabel(f"Z′ / {unit}")
        axn.set_ylabel(f"−Z″ / {unit}")
        axn.axhline(0, color="#999999", lw=0.5, zorder=1)
        axn._nyq_xy = (np.concatenate(xs), np.concatenate(ys)) if xs else (np.zeros(1), np.zeros(1))
        axn._nyq_zoom = auto_zoom(items) if zoom == "auto" else (zoom or None)
        _legend(axn, len(items))
        _panel_label(axn, "a")
        axm.set_xlabel("Frequency / Hz")
        axm.set_ylabel(f"|Z| / {unit}")
        _log_minor(axm, "x")
        _panel_label(axm, "b")
        axp.set_xlabel("Frequency / Hz")
        axp.set_ylabel("−Phase / °")
        _log_minor(axp, "x")
        _panel_label(axp, "c")
        if any(it.get("tau") is not None for it in items):
            axd.set_xscale("log")
            _log_minor(axd, "x")
            if drt_x == "tau":
                axd.set_xlabel(r"$\tau$ / s")
                sec = axd.secondary_xaxis("top", functions=(lambda t: 1 / (2 * np.pi * np.maximum(t, 1e-300)),
                                                             lambda f: 1 / (2 * np.pi * np.maximum(f, 1e-300))))
                sec.set_xlabel(r"$f = 1/(2\pi\tau)$ / Hz")
                sec.tick_params(direction="in")
                axd.tick_params(top=False)
            else:
                axd.set_xlabel(r"$f = 1/(2\pi\tau)$ / Hz")
            axd.set_ylabel(rf"$\gamma(\ln\tau)$ / {unit}")
            lo, hi = axd.get_ylim()
            axd.set_ylim(min(0, lo), hi)
        else:
            axd.text(0.5, 0.5, "DRT not computed", transform=axd.transAxes, ha="center", va="center", color="#777777")
            axd.set_xticks([])
            axd.set_yticks([])
        _panel_label(axd, "d")
        fig.tight_layout()
        equalize_nyquist(fig)
    return fig


@_locked
def fig_arrhenius(width, arr, T_C, sigma, dsigma=None, labels=None, colors=None, font_size=8):
    """log-scale sigma*T (or sigma) vs 1000/T with the fitted line and a temperature axis on top."""
    with pub_style(font_size):
        fig = _fig(min(width, 5.0), 0.78)
        ax = fig.add_subplot(111)
        T_C = np.asarray(T_C, float)
        s = np.asarray(sigma, float)
        T = T_C + 273.15
        x = 1000 / T
        y = s * T if arr["times_T"] else s
        dy = (np.asarray(dsigma, float) * (T if arr["times_T"] else 1)) if dsigma is not None else None
        for i in range(x.size):
            ax.errorbar(x[i], y[i], yerr=None if dy is None else dy[i], fmt="o", ms=5, mfc="white",
                        mec=(colors[i] if colors else "#000000"), ecolor=(colors[i] if colors else "#000000"),
                        mew=1.2, capsize=2, zorder=3)
        xf = np.linspace(x.min() - 0.03 * np.ptp(x), x.max() + 0.03 * np.ptp(x), 50)
        ax.plot(xf, 10 ** (arr["slope"] * xf + arr["intercept"]), "-", color="#D55E00", lw=1.1, zorder=2,
                label="Arrhenius fit")
        ax.set_yscale("log")
        ax.set_xlabel(r"1000 / T / K$^{-1}$")
        ax.set_ylabel(r"σT / S cm$^{-1}$ K" if arr["times_T"] else r"σ / S cm$^{-1}$")
        sec = ax.secondary_xaxis("top", functions=(lambda v: 1000 / np.maximum(v, 1e-9) - 273.15,
                                                   lambda v: 1000 / np.maximum(v + 273.15, 1e-9)))
        sec.set_xlabel("T / °C")
        sec.tick_params(direction="in")
        ax.tick_params(top=False)
        dEa = arr["dEa_eV"]
        txt = (f"$E_a$ = {arr['Ea_eV']:.3f}" + (f" ± {dEa:.1g}" if np.isfinite(dEa) else "") + " eV\n"
               f"({arr['Ea_kJ_mol']:.1f} kJ mol$^{{-1}}$), $R^2$ = {arr['r2']:.4f}")
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top")
        fig.tight_layout()
    return fig


@_locked
def fig_dc(width, steps, fit, sigma_text=None, font_size=8, log_current=True):
    """steps: dicts with label, color, t, I, window (t0, t1), I_ss, dI, E. fit: series.dc_fit result."""
    with pub_style(font_size):
        fig = _fig(width, 0.42 if width > 4 else 0.85)
        a1, a2 = (fig.subplots(1, 2) if width > 4 else fig.subplots(2, 1))
        k, u = current_unit([s["I_ss"] for s in steps])
        for s in steps:
            I = np.abs(s["I"]) if log_current else s["I"]
            a1.plot(s["t"], I / k, "-", color=s["color"], lw=0.9, label=s["label"])
            w = (s["t"] >= s["window"][0]) & (s["t"] <= s["window"][1])
            a1.plot(s["t"][w], (np.abs(s["I"][w]) if log_current else s["I"][w]) / k, "-", color=s["color"], lw=2.6,
                    alpha=0.45, solid_capstyle="butt")
        if log_current:
            a1.set_yscale("log")
            a1.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            a1.yaxis.set_major_formatter(ScalarFormatter())
            a1.yaxis.set_minor_formatter(NullFormatter())
            a1.set_ylabel(f"|I| / {u}")
        else:
            a1.set_ylabel(f"I / {u}")
        a1.set_xlabel("Time in step / s")
        _legend(a1, len(steps))
        _panel_label(a1, "a")
        E = np.array([s["E"] for s in steps])
        Iss = np.array([s["I_ss"] for s in steps]) / k
        dI = np.array([s["dI"] for s in steps]) / k
        for e, i, d, s in zip(E, Iss, dI, steps):
            a2.errorbar(e, i, yerr=d, fmt="o", ms=5, mfc="white", mec=s["color"], ecolor=s["color"], mew=1.2,
                        capsize=2, zorder=3)
        if fit is not None:
            ee = np.linspace(min(0.0, np.nanmin(E)), np.nanmax(E) * 1.05, 50)
            a2.plot(ee, (fit["G_S"] * ee + fit["I0_A"]) / k, "-", color="#D55E00", lw=1.1, zorder=2)
            txt = (rf"$R_e$ = {fit['R_e_ohm']:.3g} Ω" + (f" ± {fit['dR_e_ohm']:.1g}" if np.isfinite(fit['dR_e_ohm'])
                                                          and fit['dR_e_ohm'] > 0 else "")
                   + (f"\n{sigma_text}" if sigma_text else "") + f"\n$R^2$ = {fit['r2']:.4f}")
            a2.text(0.04, 0.96, txt, transform=a2.transAxes, ha="left", va="top")
        a2.axhline(0, color="#999999", lw=0.5)
        a2.axvline(0, color="#999999", lw=0.5)
        a2.set_xlabel("E / V")
        a2.set_ylabel(f"$I_{{ss}}$ / {u}")
        _panel_label(a2, "b")
        fig.tight_layout()
    return fig

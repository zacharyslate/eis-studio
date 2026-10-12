"""EIS Studio: Streamlit interface for ECM (impedance.py) and DRT (pyDRTtools) analysis.

Launched by the ``eis-studio`` command (see eisstudio/cli.py) or, for hosted deployments,
through the repository-root ``app.py`` shim.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from eisstudio import __version__
from eisstudio import conductivity as cond
from eisstudio import drt as drtm
from eisstudio import ecm as ecmm
from eisstudio import export as ex
from eisstudio import io as eio
from eisstudio import plotting as pl
from eisstudio import validation as val

SAMPLES = Path(__file__).parent / "sample_data"

st.set_page_config(page_title="EIS Studio · ECM & DRT", page_icon="⚡", layout="wide")

CIRCUIT_PRESETS = {
    "Custom": None,
    "Pellet, blocking electrodes, bulk arc visible: R0-p(R1,CPE1)-CPE2": "R0-p(R1,CPE1)-CPE2",
    "Pellet, bulk + grain boundary + blocking: R0-p(R1,CPE1)-p(R2,CPE2)-CPE3": "R0-p(R1,CPE1)-p(R2,CPE2)-CPE3",
    "Pellet, arcs above window (RT sulfides): R0-CPE1": "R0-CPE1",
    "One arc: R0-p(R1,CPE1)": "R0-p(R1,CPE1)",
    "Two arcs: R0-p(R1,CPE1)-p(R2,CPE2)": "R0-p(R1,CPE1)-p(R2,CPE2)",
    "Two arcs + inductance: L0-R0-p(R1,CPE1)-p(R2,CPE2)": "L0-R0-p(R1,CPE1)-p(R2,CPE2)",
    "Randles with finite diffusion: R0-p(R1-Wo1,CPE1)": "R0-p(R1-Wo1,CPE1)",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fmt(v, digits=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{digits}g}"


def _parse_float(text, name):
    t = str(text).strip().lower().replace(",", ".")
    if t in ("inf", "+inf", "∞"):
        return np.inf
    if t in ("-inf", "-∞"):
        return -np.inf
    try:
        return float(t)
    except ValueError as exc:
        raise ValueError(f"'{text}' for {name} is not a number.") from exc


def _hash(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode() if not isinstance(p, (bytes, np.ndarray)) else
                 (p.tobytes() if isinstance(p, np.ndarray) else p))
    return h.hexdigest()[:12]


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_drt(freq, Zre, Zim, settings_tuple):
    s = drtm.DRTSettings(*settings_tuple)
    return drtm.run_drt(freq, Zre + 1j * Zim, s)


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_kk(freq, Zre, Zim):
    return val.lin_kk(freq, Zre + 1j * Zim)


def show(fig, stretch=False):
    """On-screen preview at modest DPI (downloads use the export DPI)."""
    st.image(pl.to_bytes(fig, "png", 110), width="stretch" if stretch else "content")


def _deferred(fig, fmt, dpi):
    """Render a figure only when its download button is clicked."""
    return lambda: pl.to_bytes(fig, fmt, dpi)


def _zip_builder(static_files: dict, figures: dict, dpi):
    """Callable producing a ZIP with every figure in PNG, PDF and SVG plus the tables (built on click)."""
    def build():
        files = dict(static_files)
        for stem, fig in figures.items():
            for fmt in ("png", "pdf", "svg"):
                files[f"{stem}.{fmt}"] = pl.to_bytes(fig, fmt, dpi)
        return ex.zip_bundle(files)
    return build


def _download_row(label_prefix, fig_bytes_fn, csv_items, zip_name, zip_files, key):
    fmt = st.session_state.get("fig_fmt", "png")
    dpi = st.session_state.get("fig_dpi", 600)
    cols = st.columns(len(csv_items) + 2)
    cols[0].download_button(f"⬇ {label_prefix} figure (.{fmt})", lambda: fig_bytes_fn(fmt, dpi),
                            file_name=f"{zip_name}_figure.{fmt}", mime=pl.MIME[fmt], key=f"{key}_fig",
                            width="stretch")
    for c, (lab, fname, data) in zip(cols[1:-1], csv_items):
        c.download_button(f"⬇ {lab}", data, file_name=fname, mime="text/csv", key=f"{key}_{fname}",
                          width="stretch")
    cols[-1].download_button("⬇ Everything (.zip)", zip_files, file_name=f"{zip_name}.zip",
                             mime="application/zip", key=f"{key}_zip", width="stretch", type="primary")


# --------------------------------------------------------------------------- #
# Sidebar: data, geometry, figure settings
# --------------------------------------------------------------------------- #
st.sidebar.title("EIS Studio")
st.sidebar.caption(f"v{__version__} · ECM via impedance.py · DRT via pyDRTtools")

with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("Upload a spectrum",
                          type=["csv", "txt", "tsv", "dat", "mpt", "dta", "z", "xlsx", "xls", "par"],
                          help="Gamry .DTA, ZView .z, BioLogic .mpt, VersaStudio/Parstat/Autolab/CHI text "
                               "exports, or any table with frequency + Re/Im (or |Z| + phase) columns.")
    sample = None
    if up is None:
        samples = sorted(p.name for p in SAMPLES.glob("*") if p.is_file())
        sample = st.selectbox("…or try a synthetic sample", ["–"] + samples)
    instrument = st.selectbox("File format", list(eio.INSTRUMENTS), index=0)
    imag_sign = st.selectbox("Imaginary column sign", eio.IMAG_SIGN_OPTIONS, index=0,
                             help="'Auto' converts -Z'' columns when headers say so or when most values are "
                                  "positive. Override if the auto-detection is wrong.")

raw, fname = None, None
if up is not None:
    raw, fname = up.getvalue(), up.name
elif sample and sample != "–":
    raw, fname = (SAMPLES / sample).read_bytes(), sample

if raw is None:
    st.title("EIS Studio")
    st.markdown(
        "Upload an impedance spectrum in the sidebar (or pick a synthetic sample) to fit an "
        "**equivalent circuit** (impedance.py) or compute the **distribution of relaxation times** "
        "(pyDRTtools), get the ionic **conductivity** from your sample geometry, and download "
        "**CSV data** and **publication figures** (PNG/PDF/SVG).")
    st.info("Sign convention used throughout: Z = Z′ + jZ″ with Z″ < 0 for capacitive behaviour; plots show −Z″.")
    st.stop()

with st.sidebar:
    colmap = None
    with st.expander("Manual column mapping"):
        headers = eio.table_preview(raw, fname)
        use_manual = st.checkbox("Map columns manually", value=False, disabled=not headers)
        if use_manual and headers:
            opts = ["–"] + headers
            def _pick(lbl, k):
                v = st.selectbox(lbl, opts, key=k)
                return None if v == "–" else headers.index(v)
            colmap = {"freq": _pick("Frequency", "cm_f"), "re": _pick("Z′ (real)", "cm_re"),
                      "im": _pick("Z″ or −Z″ (imag)", "cm_im"), "mod": _pick("|Z| (if no Re/Im)", "cm_mod"),
                      "phase": _pick("Phase (if no Re/Im)", "cm_ph")}
            if colmap["freq"] is None:
                colmap = None

try:
    sweeps, report = eio.load_spectrum(raw, fname, instrument, imag_sign, colmap)
except Exception as exc:  # noqa: BLE001 - show any parser failure to the user
    st.error(f"Could not read **{fname}**: {exc}")
    st.stop()

with st.sidebar:
    sweep_idx = 0
    if len(sweeps) > 1:
        sweep_idx = st.selectbox("Sweep", range(len(sweeps)), format_func=lambda i: f"Sweep {i + 1} "
                                 f"({sweeps[i].n} pts)")
    data_full = sweeps[sweep_idx]
    fmin_d, fmax_d = float(data_full.freq.min()), float(data_full.freq.max())
    st.subheader("Frequency window")
    c1, c2 = st.columns(2)
    f_min = c1.number_input("f min (Hz)", value=fmin_d, min_value=0.0, format="%.4g")
    f_max = c2.number_input("f max (Hz)", value=fmax_d, min_value=0.0, format="%.4g")
    drop_ind = st.checkbox("Drop inductive points (Z″ > 0)", value=False,
                           help="Applies to ECM and the KK check. The DRT has its own inductance options.")

    st.header("2 · Sample geometry")
    shape = st.radio("Geometry", ["Disc (thickness + diameter)", "Thickness + area"], horizontal=False)
    lu = st.selectbox("Length unit", list(cond.LENGTH_TO_CM), index=0)
    c1, c2 = st.columns(2)
    thick = c1.number_input(f"Thickness ({lu})", value=1.0, min_value=0.0, format="%.4g")
    dthick = c2.number_input(f"± thickness ({lu})", value=0.0, min_value=0.0, format="%.3g")
    if shape.startswith("Disc"):
        diam = c1.number_input(f"Diameter ({lu})", value=10.0, min_value=0.0, format="%.4g")
        ddiam = c2.number_input(f"± diameter ({lu})", value=0.0, min_value=0.0, format="%.3g")
        geom = cond.Geometry.disc(thick, diam, lu, dthick, ddiam) if thick > 0 and diam > 0 else None
    else:
        au = st.selectbox("Area unit", list(cond.AREA_TO_CM2), index=0)
        area = c1.number_input(f"Area ({au})", value=0.785, min_value=0.0, format="%.4g")
        darea = c2.number_input(f"± area ({au})", value=0.0, min_value=0.0, format="%.3g")
        geom = cond.Geometry.slab(thick, area, lu, au, dthick, darea) if thick > 0 and area > 0 else None
    temp_label = st.text_input("Temperature / sample label (goes into exports)", value="")
    if geom:
        st.caption(f"L/A = {geom.cell_constant_per_cm:.4g} cm⁻¹ · A = {geom.area_cm2:.4g} cm²")

    st.header("3 · Figures")
    st.selectbox("Width", list(pl.WIDTHS), index=2, key="fig_width")
    st.selectbox("Impedance units", pl.UNIT_MODES, index=0, key="unit_mode",
                 help="Ω·cm² multiplies by electrode area; Ω·cm multiplies by A/L (resistivity).")
    c1, c2 = st.columns(2)
    c1.selectbox("Format", ["png", "pdf", "svg", "tiff"], key="fig_fmt")
    c2.selectbox("DPI (raster)", [300, 600, 1200], index=1, key="fig_dpi")
    st.slider("Font size (pt)", 6, 12, 8, key="font_size")
    st.checkbox("Mark frequency decades on Nyquist", value=True, key="decades")
    st.number_input("Nyquist zoom: max Z′ shown (0 = all)", value=0.0, min_value=0.0, format="%.4g", key="zoom",
                    help="In the plotted units. Use it to hide a blocking-electrode tail.")

data = data_full.cropped(f_min, f_max, drop_inductive=drop_ind)
if data.n < 8:
    st.error("Fewer than 8 points remain inside the frequency window.")
    st.stop()

width = pl.WIDTHS[st.session_state.fig_width]
scale, unit = pl.unit_scale(st.session_state.unit_mode, geom.area_cm2 if geom else None,
                            geom.thickness_cm if geom else None)
if st.session_state.unit_mode != pl.UNIT_MODES[0] and geom is None:
    st.sidebar.warning("Area-normalised units need the sample geometry; plotting in Ω.")
zoom = st.session_state.zoom or None
fs = st.session_state.font_size

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.title(f"EIS Studio · {data.name}")
st.caption(f"{data.n} points · {data.freq.min():.3g}–{data.freq.max():.3g} Hz · reader: {report.reader}"
           + (f" · columns: {report.columns}" if report.columns else ""))
for n in report.notes:
    st.caption(f"ℹ {n}")

tab_data, tab_ecm, tab_drt, tab_help = st.tabs(["Data & KK check", "Equivalent circuit (ECM)",
                                                "DRT", "Methods & references"])

# ---------------------------- DATA TAB ------------------------------------- #
with tab_data:
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = pl.fig_single("nyquist", 4.5, Z=data.Z, f=data.freq, scale=scale, unit=unit,
                            decades=st.session_state.decades, zoom=zoom, font_size=fs)
        show(fig, stretch=True)
    with c2:
        fig = pl.fig_single("bode", 4.5, f=data.freq, Z=data.Z, scale=scale, unit=unit, font_size=fs)
        show(fig, stretch=True)

    st.subheader("Kramers–Kronig consistency (Lin-KK)")
    st.markdown("Run this first. A spectrum that is not KK-consistent (drift during the sweep, non-linear "
                "response, unstable cell) cannot be described by *any* linear, causal circuit or DRT, and fitted "
                "parameters from it are not physically meaningful.")
    try:
        kk = _cached_kk(data.freq, data.Z.real, data.Z.imag)
        verdict = ("✅ residuals within ±1 %: consistent" if kk["max_abs_res_pct"] < 1 else
                   "⚠️ some residuals exceed 1 %: inspect where" if kk["max_abs_res_pct"] < 3 else
                   "❌ residuals exceed 3 %: the spectrum (or part of it) is not KK-consistent")
        st.markdown(f"M = {kk['M']} RC elements · μ = {kk['mu']:.3f} · mean |res| = "
                    f"{kk['mean_abs_res_pct']:.2f} % · max |res| = {kk['max_abs_res_pct']:.2f} % · {verdict}")
        st.caption("Rule of thumb, not a statistical test: compare the residual magnitude with your instrument's "
                   "noise level, and look for systematic trends (drift shows up at low frequency).")
        kfig = pl.fig_kk(width, data.freq, kk["res_re_pct"], kk["res_im_pct"], kk["M"], font_size=fs)
        show(kfig)
        kk_df = pd.DataFrame({"freq_Hz": data.freq, "res_re_pct": kk["res_re_pct"], "res_im_pct": kk["res_im_pct"]})
        c1, c2 = st.columns(2)
        f_ = st.session_state.fig_fmt
        c1.download_button(f"⬇ KK figure (.{f_})", _deferred(kfig, f_, st.session_state.fig_dpi),
                           file_name=f"{data.name}_linKK.{f_}", mime=pl.MIME[f_])
        c2.download_button("⬇ KK residuals (.csv)", ex.csv_bytes(kk_df), file_name=f"{data.name}_linKK.csv",
                           mime="text/csv")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Lin-KK test failed: {exc}")

    with st.expander("Parsed data table"):
        st.dataframe(pd.DataFrame({"f / Hz": data.freq, "Z′ / Ω": data.Z.real, "Z″ / Ω": data.Z.imag,
                                   "|Z| / Ω": np.abs(data.Z), "phase / °": np.degrees(np.angle(data.Z))}),
                     width="stretch", height=300)

# ---------------------------- ECM TAB -------------------------------------- #
with tab_ecm:
    st.markdown("Syntax: `-` series, `p(a,b)` parallel, numbered elements. Elements: `R`, `C`, `L`, `CPE` (Q, α), "
                "`W`, `Wo`, `Ws`, `La`, `G`, `Gs`, `K`, `Zarc`, `TLMQ`, `T`. "
                "[Element definitions](https://impedancepy.readthedocs.io/en/latest/circuit-elements.html)")
    c1, c2 = st.columns([2, 3])
    preset = c1.selectbox("Preset", list(CIRCUIT_PRESETS), index=2)
    default_circ = CIRCUIT_PRESETS[preset] or st.session_state.get("circuit_text", "R0-p(R1,CPE1)-CPE2")
    circuit = c2.text_input("Circuit", value=default_circ, key=f"circ_{preset}").replace(" ", "")
    st.session_state["circuit_text"] = circuit

    try:
        names, units = ecmm.parse_circuit(circuit)
        circ_ok = True
    except Exception as exc:  # noqa: BLE001
        st.error(f"Circuit error: {exc}")
        circ_ok = False

    if circ_ok:
        tkey = f"ptable_{circuit}_{_hash(data.freq, data.Z)}"
        if tkey not in st.session_state or st.session_state.get("reset_" + tkey):
            g = ecmm.default_guess(names, data.freq, data.Z, circuit)
            bd = ecmm.default_bounds(circuit)
            st.session_state[tkey] = pd.DataFrame({
                "parameter": names, "unit": units,
                "initial": [f"{v:.4g}" for v in g],
                "fixed": [False] * len(names),
                "lower": [f"{bd[n][0]:g}" for n in names],
                "upper": ["inf" if not np.isfinite(bd[n][1]) else f"{bd[n][1]:g}" for n in names],
            })
            st.session_state["reset_" + tkey] = False
        st.caption("Initial values are data-informed guesses. Edit them, tick **fixed** to hold a parameter "
                   "constant at its initial value, or tighten bounds (type `inf` for no upper bound).")
        table = st.data_editor(
            st.session_state[tkey], key=f"ed_{tkey}", hide_index=True, width="stretch",
            disabled=["parameter", "unit"],
            column_config={"fixed": st.column_config.CheckboxColumn("fixed"),
                           "initial": st.column_config.TextColumn("initial"),
                           "lower": st.column_config.TextColumn("lower"),
                           "upper": st.column_config.TextColumn("upper")})
        c1, c2, c3, c4 = st.columns([2, 1.3, 1.2, 1])
        weighting = c1.selectbox("Weighting", list(ecmm.WEIGHTINGS), index=0,
                                 help="Modulus weighting is the usual choice when measurement variances are "
                                      "unknown; unit weighting over-weights the largest |Z| points.")
        n_starts = c2.number_input("Multi-start fits", 1, 50, 8,
                                   help="Refit from N log-perturbed starting points and keep the best. Also "
                                        "detects non-identifiable parameters.")
        global_opt = c3.checkbox("Basin-hopping", value=False,
                                 help="impedance.py's global optimiser (slow; no weighting, RMSE objective).")
        if c4.button("Reset table"):
            st.session_state["reset_" + tkey] = True
            st.rerun()

        run_ecm = st.button("Fit circuit", type="primary")
        fit_key = _hash(circuit, table.to_json(), weighting, n_starts, global_opt, data.freq, data.Z)
        if run_ecm:
            try:
                initial, fixed, bounds = {}, {}, {}
                for _, r in table.iterrows():
                    v = _parse_float(r["initial"], r["parameter"])
                    if bool(r["fixed"]):
                        fixed[r["parameter"]] = v
                    else:
                        initial[r["parameter"]] = v
                        bounds[r["parameter"]] = (_parse_float(r["lower"], r["parameter"]),
                                                  _parse_float(r["upper"], r["parameter"]))
                with st.spinner("Fitting…"):
                    res = ecmm.fit_ecm(data.freq, data.Z, circuit, initial, fixed, bounds,
                                       ecmm.WEIGHTINGS[weighting], global_opt, int(n_starts))
                st.session_state["ecm_res"] = res
                st.session_state["ecm_key"] = fit_key
                st.session_state["ecm_settings"] = {
                    "circuit": circuit, "initial": initial, "fixed": fixed,
                    "bounds": {k: [float(a), float(b)] for k, (a, b) in bounds.items()},
                    "weighting": weighting, "n_starts": int(n_starts), "basin_hopping": global_opt,
                    "f_window_Hz": [float(data.freq.min()), float(data.freq.max())],
                    "drop_inductive": drop_ind, "file": fname, "label": temp_label}
            except Exception as exc:  # noqa: BLE001
                st.error(f"Fit failed: {exc}")

        res = st.session_state.get("ecm_res")
        if res is not None and res.circuit != circuit:
            res = None
        if res is not None:
            if st.session_state.get("ecm_key") != fit_key:
                st.info("Inputs changed since this fit. Press **Fit circuit** to update.")
            for w in res.warnings:
                st.warning(w)
            m = res.metrics
            st.markdown(
                f"**Fit quality** · pseudo-χ² = {m['pseudo_chi2']:.3e} (per d.o.f. {m['pseudo_chi2_per_dof']:.2e}) · "
                f"mean |Δ| Re/Im = {m['mean_abs_rel_res_re_pct']:.2f} / {m['mean_abs_rel_res_im_pct']:.2f} % · "
                f"max |Δ| = {m['max_abs_rel_res_pct']:.2f} % · best minimum reached by "
                f"{m.get('n_starts_at_best_minimum', 1)}/{m.get('n_starts_converged', 1)} starts")

            c1, c2 = st.columns([3, 2])
            with c1:
                rows = [{"parameter": n, "value": _fmt(v), "± std. error": ("fixed" if fx else _fmt(e, 2)),
                         "rel. error": ("–" if fx or v == 0 or not np.isfinite(e) else f"{abs(e / v) * 100:.1f} %"),
                         "unit": u} for n, u, v, e, fx in res.all_params()]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption("Std. errors are from the curve_fit covariance (impedance.py); they assume the model is "
                           "correct and ignore parameter correlation. Treat them as lower bounds.")
            with c2:
                if res.derived:
                    st.dataframe(pd.DataFrame([{
                        "element": d["pair"], "C_eff / F": f"{d['C_eff_F']:.3g}", "α": f"{d['alpha']:.3f}",
                        "f_c / Hz": f"{d['f_c_Hz']:.3g}"} for d in res.derived]), hide_index=True,
                        width="stretch")
                    st.caption("Typical C_eff (Irvine, Sinclair & West 1990): bulk ~10⁻¹² F, grain boundary "
                               "10⁻¹¹–10⁻⁸ F, electrode/interface 10⁻⁷–10⁻⁵ F. These are magnitudes per cm "
                               "geometry; scale by your L/A before assigning.")

            st.subheader("Conductivity")
            r_names = [n for n, u, v, e, fx in res.all_params() if ecmm.element_type(n) == "R"]
            sel = st.multiselect("Resistances that make up the ionic resistance", r_names, default=r_names,
                                 help="E.g. bulk only, or bulk + grain boundary for total conductivity. "
                                      "Exclude contact/lead resistances if you modelled them separately.")
            cnd = None
            if sel and geom:
                Rsum = sum(res.param(n)[0] for n in sel)
                dR = float(np.sqrt(sum(res.param(n)[1] ** 2 for n in sel)))
                cnd = cond.conductivity(Rsum, geom, dR)
                st.metric("σ", cond.fmt_sigma(cnd), help="σ = L / (R·A)")
                st.caption(f"R = {Rsum:.4g} Ω ({' + '.join(sel)}) · {geom.description} · relative error budget: "
                           + ", ".join(f"{k} {v * 100:.2f} %" for k, v in cnd["rel_err_budget"].items())
                           + (" · ⚠️ R errors are added in quadrature without covariance; if the summed "
                              "resistances are strongly anti-correlated (an arc at the window edge) the true "
                              "uncertainty of their sum is smaller than shown." if len(sel) > 1 else ""))
            elif not geom:
                st.info("Enter sample thickness and diameter/area in the sidebar to compute σ.")

            st.subheader("Figures")
            rr, ri = ecmm.relative_residuals(res.Z, res.Z_fit)
            comp = pl.fig_ecm_composite(width, res.freq, res.Z, res.f_smooth, res.Z_smooth, res.Z_fit, rr, ri,
                                        scale, unit, st.session_state.decades, fs, zoom)
            t1, t2, t3, t4 = st.tabs(["Composite", "Nyquist", "Bode", "Residuals"])
            singles = {
                "nyquist": pl.fig_single("nyquist", width if width < 4 else 3.5, Z=res.Z, Z_line=res.Z_smooth,
                                         f=res.freq, scale=scale, unit=unit, decades=st.session_state.decades,
                                         zoom=zoom, font_size=fs),
                "bode": pl.fig_single("bode", width, f=res.freq, Z=res.Z, f_line=res.f_smooth, Z_line=res.Z_smooth,
                                      scale=scale, unit=unit, font_size=fs),
                "residuals": pl.fig_single("residuals", width, f=res.freq, r_re=rr, r_im=ri, font_size=fs),
            }
            with t1:
                show(comp)
            for tab, k in zip((t2, t3, t4), ("nyquist", "bode", "residuals")):
                with tab:
                    show(singles[k])
                    f_ = st.session_state.fig_fmt
                    st.download_button(f"⬇ {k} (.{f_})", _deferred(singles[k], f_, st.session_state.fig_dpi),
                                       file_name=f"{data.name}_ECM_{k}.{f_}", mime=pl.MIME[f_], key=f"ecm_dl_{k}")

            spec_csv = ex.csv_bytes(ex.ecm_spectrum_df(res))
            par_csv = ex.csv_bytes(ex.ecm_params_df(res, cnd))
            settings = dict(st.session_state.get("ecm_settings", {}))
            settings["geometry"] = geom.__dict__ if geom else None
            settings["conductivity_resistances"] = sel
            f_, dpi = st.session_state.fig_fmt, st.session_state.fig_dpi
            zipf = _zip_builder({f"{data.name}_ECM_spectrum.csv": spec_csv,
                                 f"{data.name}_ECM_parameters.csv": par_csv,
                                 f"{data.name}_ECM_settings.json": ex.settings_json("ECM", settings)},
                                {f"{data.name}_ECM_composite": comp,
                                 **{f"{data.name}_ECM_{k}": fg for k, fg in singles.items()}}, dpi)
            st.markdown("**Downloads**")
            _download_row("Composite", lambda fm, d: pl.to_bytes(comp, fm, d),
                          [("Spectrum + fit (.csv)", f"{data.name}_ECM_spectrum.csv", spec_csv),
                           ("Parameters (.csv)", f"{data.name}_ECM_parameters.csv", par_csv)],
                          f"{data.name}_ECM", zipf, "ecm")

# ---------------------------- DRT TAB -------------------------------------- #
with tab_drt:
    st.markdown("Ridge-regularised DRT (Tikhonov) with RBF discretisation, as implemented in pyDRTtools. "
                "γ is per unit ln τ, so the area under a peak is that process's resistance.")
    with st.form("drt_form"):
        c1, c2, c3 = st.columns(3)
        rbf = c1.selectbox("Discretisation (RBF)", drtm.RBF_TYPES, index=0)
        shape_ctl = c1.selectbox("RBF shape control", drtm.SHAPE_CONTROL, index=0)
        coeff = c1.number_input("FWHM coefficient / shape factor", value=0.5, min_value=1e-3, format="%.3g")
        data_used_lbl = c2.selectbox("Data used", list(drtm.DATA_USED), index=0)
        ind_lbl = c2.selectbox("Inductance", list(drtm.INDUCTANCE), index=0)
        der = c2.selectbox("Regularisation derivative", drtm.DERIVATIVE, index=0)
        lam_lbl = c3.selectbox("λ selection", list(drtm.LAMBDA_METHODS), index=0,
                               help="Maradesa et al. (2023) compare these criteria; GCV is the pyDRTtools default.")
        lam = c3.number_input("λ (manual, or initial guess)", value=1e-3, min_value=1e-10, format="%.2e")
        bayes = c3.checkbox("Bayesian credible band (HMC)", value=False,
                            help="Samples the truncated-Gaussian posterior (99 % band). Slower.")
        c4, c5, c6 = st.columns(3)
        n_mc = c4.number_input("HMC samples", 1000, 20000, 2000, step=500)
        prom = c5.number_input("Peak prominence (fraction of max γ)", value=0.02, min_value=0.0, max_value=1.0)
        gmode = c6.selectbox("Plot", ["γ(ln τ) vs τ", "g(τ) = γ/τ vs τ"], index=0)
        go = st.form_submit_button("Compute DRT", type="primary")

    s = drtm.DRTSettings(rbf_type=rbf, data_used=drtm.DATA_USED[data_used_lbl], inductance=drtm.INDUCTANCE[ind_lbl],
                         der_used=der, cv_type=drtm.LAMBDA_METHODS[lam_lbl], reg_param=float(lam),
                         shape_control=shape_ctl, coeff=float(coeff), bayesian=bayes, n_mc=int(n_mc),
                         peak_prominence=float(prom))
    s_tuple = tuple(s.__dict__.values())
    if go:
        try:
            with st.spinner("Computing DRT…" + (" (HMC sampling)" if bayes else "")):
                st.session_state["drt_res"] = _cached_drt(data_full.cropped(f_min, f_max).freq,
                                                          data_full.cropped(f_min, f_max).Z.real,
                                                          data_full.cropped(f_min, f_max).Z.imag, s_tuple)
                st.session_state["drt_key"] = _hash(s_tuple, f_min, f_max, data_full.freq, data_full.Z)
        except Exception as exc:  # noqa: BLE001
            st.error(f"DRT failed: {exc}")

    dres = st.session_state.get("drt_res")
    if dres is not None:
        if st.session_state.get("drt_key") != _hash(s_tuple, f_min, f_max, data_full.freq, data_full.Z):
            st.info("Settings or data changed since this DRT. Press **Compute DRT** to update.")
        for w in dres.warnings:
            st.warning(w)
        m = dres.metrics
        st.markdown(f"**R∞** = {dres.R_inf:.4g} Ω{' (estimated)' if dres.R_inf_estimated else ''} · "
                    f"**L** = {dres.L:.3g} H · **λ** = {dres.lam:.3g} · **R_pol** = ∫γ d ln τ = {dres.R_pol:.4g} Ω · "
                    f"mean |Δ| Re/Im = {m['mean_abs_rel_res_re_pct']:.2f} / {m['mean_abs_rel_res_im_pct']:.2f} %")
        if dres.peaks:
            pk = pd.DataFrame([{"peak": f"P{p['peak']}" + (" *" if p["minor"] else ""), "τ / s": f"{p['tau_s']:.3g}",
                                "f = 1/(2πτ) / Hz": f"{p['f_Hz']:.3g}", "R / Ω": f"{p['R_ohm']:.4g}",
                                "share": f"{p['share_pct']:.1f} %", "C = τ/R / F": f"{p['C_eq_F']:.3g}"}
                               for p in dres.peaks])
            st.dataframe(pk, hide_index=True, width="stretch")
            if any(p["minor"] for p in dres.peaks):
                st.caption("\\* minor peak (< 3 % of the total area). Small peaks flanking a large one are a known "
                           "artefact of regularised DRT on depressed (CPE-like) arcs; don't assign a process to them "
                           "without independent evidence (T, pO₂, bias dependence, or repeat measurements).")
            st.caption("Peak R is integrated between adjacent minima of γ; for overlapping peaks this split is "
                       "approximate.")

        st.subheader("Conductivity")
        basis_opts = ["R∞ only (bulk when its arc is above the window)"] + \
                     [f"R∞ + P1…P{k}" for k in range(1, len(dres.peaks) + 1)]
        basis = st.selectbox("Ionic resistance =", basis_opts, index=0)
        dcnd = None
        if geom:
            k = basis_opts.index(basis)
            Rb = dres.R_inf + sum(p["R_ohm"] for p in dres.peaks[:k])
            dcnd = cond.conductivity(Rb, geom, 0.0)
            st.metric("σ", cond.fmt_sigma(dcnd))
            st.caption(f"R = {Rb:.4g} Ω · {geom.description} · uncertainty includes geometry only (the regularised "
                       "DRT has no simple parameter covariance; use the Bayesian band or ECM for R uncertainty).")
        else:
            st.info("Enter sample geometry in the sidebar to compute σ.")

        st.subheader("Figures")
        gm = "gamma" if gmode.startswith("γ(") else "g"
        a_re, a_im = ecmm.relative_residuals(dres.Z, dres.Z_fit)
        comp = pl.fig_drt_composite(width, dres.freq, dres.Z, dres.Z_fit, a_re, a_im, dres.tau, dres.gamma,
                                    dres.gamma_lo, dres.gamma_hi, dres.gamma_mean, dres.peaks, scale, unit,
                                    st.session_state.decades, gm, fs, zoom)
        singles = {
            "nyquist": pl.fig_single("nyquist", width if width < 4 else 3.5, Z=dres.Z, Z_pts=dres.Z_fit, f=dres.freq,
                                     scale=scale, unit=unit, decades=st.session_state.decades, zoom=zoom,
                                     fit_label="DRT fit", font_size=fs),
            "bode": pl.fig_single("bode", width, f=dres.freq, Z=dres.Z, f_line=dres.freq, Z_line=dres.Z_fit,
                                  scale=scale, unit=unit, font_size=fs),
            "residuals": pl.fig_single("residuals", width, f=dres.freq, r_re=a_re, r_im=a_im, font_size=fs),
            "drt": pl.fig_single("drt", width, tau=dres.tau, gamma=dres.gamma, lo=dres.gamma_lo, hi=dres.gamma_hi,
                                 mean=dres.gamma_mean, peaks=dres.peaks, f_window=(dres.freq.min(), dres.freq.max()),
                                 gamma_mode=gm, font_size=fs),
        }
        tabs = st.tabs(["Composite", "γ vs τ", "Nyquist", "Bode", "Residuals"])
        with tabs[0]:
            show(comp)
        for tab, k in zip(tabs[1:], ("drt", "nyquist", "bode", "residuals")):
            with tab:
                show(singles[k])
                f_ = st.session_state.fig_fmt
                st.download_button(f"⬇ {k} (.{f_})", _deferred(singles[k], f_, st.session_state.fig_dpi),
                                   file_name=f"{data.name}_DRT_{k}.{f_}", mime=pl.MIME[f_], key=f"drt_dl_{k}")

        spec_csv = ex.csv_bytes(ex.drt_spectrum_df(dres))
        gam_csv = ex.csv_bytes(ex.drt_gamma_df(dres))
        sum_csv = ex.csv_bytes(ex.drt_summary_df(dres, dcnd))
        settings = {**s.__dict__, "f_window_Hz": [float(dres.freq.min()), float(dres.freq.max())], "file": fname,
                    "label": temp_label, "geometry": geom.__dict__ if geom else None, "conductivity_basis": basis}
        dpi = st.session_state.fig_dpi
        zipf = _zip_builder({f"{data.name}_DRT_spectrum.csv": spec_csv, f"{data.name}_DRT_gamma.csv": gam_csv,
                             f"{data.name}_DRT_summary.csv": sum_csv,
                             f"{data.name}_DRT_settings.json": ex.settings_json("DRT", settings)},
                            {f"{data.name}_DRT_composite": comp,
                             **{f"{data.name}_DRT_{k}": fg for k, fg in singles.items()}}, dpi)
        st.markdown("**Downloads**")
        _download_row("Composite", lambda fm, d: pl.to_bytes(comp, fm, d),
                      [("Spectrum + DRT fit (.csv)", f"{data.name}_DRT_spectrum.csv", spec_csv),
                       ("γ(τ) (.csv)", f"{data.name}_DRT_gamma.csv", gam_csv),
                       ("Summary & peaks (.csv)", f"{data.name}_DRT_summary.csv", sum_csv)],
                      f"{data.name}_DRT", zipf, "drt")

# ---------------------------- HELP TAB ------------------------------------- #
with tab_help:
    st.markdown((Path(__file__).parent / "METHODS.md").read_text(encoding="utf-8"))

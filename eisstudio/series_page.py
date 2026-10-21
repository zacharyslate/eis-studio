"""Series mode of the Streamlit app: several spectra and DC polarisation steps analysed together.

Rendered from app.py when the sidebar mode is "Series"; everything heavy lives in series.py and
series_plots.py so it can be tested without Streamlit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from eisstudio import conductivity as cond
from eisstudio import drt as drtm
from eisstudio import ecm as ecmm
from eisstudio import export as ex
from eisstudio import plotting as pl
from eisstudio import series as S
from eisstudio import series_plots as sp

SERIES_DIR = Path(__file__).parent / "sample_data" / "series"
SAMPLE_SETS = {
    "Temperature series + DC steps (one EC-Lab file)": ["LPSC_pellet_T-series_and_DC_linked.mpt"],
    "Temperature series as separate files + DC table": [f"pellet_{t}C.csv" for t in (25, 40, 55, 70, 85)]
    + ["pellet_DC_polarisation.txt"],
    "Five pellets pressed to different densities": sorted(p.name for p in SERIES_DIR.glob("density_*.csv")),
}
SAMPLE_HINTS = {
    "Temperature series + DC steps (one EC-Lab file)": "Simulated at 25, 40, 55, 70, 85 °C; pellet 1.00 mm × "
    "10.0 mm, 125.7 mg. Type the temperatures below.",
    "Temperature series as separate files + DC table": "Same pellet as the EC-Lab sample: 25, 40, 55, 70, 85 °C.",
    "Five pellets pressed to different densities": "All 10.0 mm diameter, 25 °C; thickness and mass are in the "
    "file names (theoretical density 1.86 g/cm³). Enter them per pellet in the table.",
}
PRESETS = {
    "Arc + blocking spike: p(R1,CPE1)-CPE2": "p(R1,CPE1)-CPE2",
    "Intercept + spike: R0-CPE1": "R0-CPE1",
    "Intercept, arc + spike: R0-p(R1,CPE1)-CPE2": "R0-p(R1,CPE1)-CPE2",
    "Two arcs + spike: p(R1,CPE1)-p(R2,CPE2)-CPE3": "p(R1,CPE1)-p(R2,CPE2)-CPE3",
    "Custom": None,
}


def _h(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else repr(p).encode())
    return h.hexdigest()[:12]


def _dl(target, *a, **k):
    k.setdefault("on_click", "ignore")
    return target.download_button(*a, **k)


def _show(fig):
    st.image(pl.to_bytes(fig, "png", 110), width="content")


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


@st.cache_data(show_spinner=False, max_entries=8)
def _load(files):
    return S.load_series(list(files))


def _fig_downloads(fig, stem, key):
    fmt, dpi = st.session_state.s_fmt, st.session_state.s_dpi
    _dl(st, f"⬇ {stem} (.{fmt})", lambda: pl.to_bytes(fig, fmt, dpi), file_name=f"{stem}.{fmt}", mime=pl.MIME[fmt],
        key=key)


def render():
    ss = st.session_state
    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("1 · Data")
        ups = st.file_uploader("Upload one or more files", accept_multiple_files=True,
                               type=["csv", "txt", "tsv", "dat", "mpt", "dta", "z", "xlsx", "xls"],
                               help="One export holding several measurements (e.g. an EC-Lab linked experiment), "
                                    "several files, or both. Impedance spectra and potentiostatic (DC) steps are "
                                    "found and separated automatically.", key="s_upload")
        sample = "–"
        if not ups:
            sample = st.selectbox("…or try a synthetic sample set", ["–"] + list(SAMPLE_SETS), key="s_sample")
        st.header("2 · Sample")
        st.caption("Shared values; per-sample thickness, diameter and mass can be entered in the table.")
        c1, c2 = st.columns(2)
        thick = c1.number_input("Thickness (mm)", value=1.0, min_value=0.0, format="%.4g", key="s_thick")
        dthick = c2.number_input("± (mm)", value=0.0, min_value=0.0, format="%.3g", key="s_dthick")
        diam = c1.number_input("Diameter (mm)", value=10.0, min_value=0.0, format="%.4g", key="s_diam")
        ddiam = c2.number_input("± (mm) ", value=0.0, min_value=0.0, format="%.3g", key="s_ddiam")
        rho_th = st.number_input("Theoretical density (g/cm³, 0 = not used)", value=0.0, min_value=0.0,
                                 format="%.4g", key="s_rho_th",
                                 help="With a pellet mass in the table this gives the relative density "
                                      "ρ_rel = m / (A·L) / ρ_theoretical, used for the density normalisation.")
        st.header("3 · Figures")
        st.selectbox("Width", list(pl.WIDTHS), index=2, key="s_width")
        c1, c2 = st.columns(2)
        c1.selectbox("Format", ["png", "pdf", "svg", "tiff"], key="s_fmt")
        c2.selectbox("DPI (raster)", [300, 600, 1200], index=1, key="s_dpi")
        st.slider("Font size (pt)", 6, 12, 8, key="s_font")

    files = []
    if ups:
        files = [(u.getvalue(), u.name) for u in ups]
    elif sample != "–":
        files = [((SERIES_DIR / f).read_bytes(), f) for f in SAMPLE_SETS[sample]]
    st.title("EIS Studio · Series analysis")
    if not files:
        st.markdown(
            "Analyse several measurements together: impedance spectra at several temperatures (or of several "
            "pellets) and DC polarisation steps at several potentials, from one export or several files.\n\n"
            "* **Ionic conductivity** of every spectrum, and the **activation energy** from an Arrhenius fit.\n"
            "* **Electronic conductivity** from the steady-state currents of the DC steps.\n"
            "* Overlaid **Nyquist, Bode |Z|, −phase and DRT** plots, optionally normalised by geometry and "
            "relative density, plus the DC curves and the I–V line.\n\n"
            "Upload files in the sidebar, or try a synthetic sample set.")
        st.stop()

    geom0 = cond.Geometry.disc(thick, diam, "mm", dthick, ddiam) if thick > 0 and diam > 0 else None
    fkey = _h(*[f[0] for f in files], *[f[1] for f in files])
    try:
        meas, report = _load(tuple(files))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the files: {exc}")
        st.stop()
    for n in report.notes:
        st.caption(f"ℹ {n}")
    if sample != "–" and not ups:
        st.info(SAMPLE_HINTS[sample])
    if not meas:
        st.error("No impedance spectra or time series were found.")
        with st.expander("Tables seen"):
            st.json(report.tables)
        st.stop()
    with st.expander("How the files were read"):
        st.json(report.tables)
        for m in meas:
            for n in m.notes:
                st.caption(f"{m.name}: {n}")

    # ------------------------------------------------------------------ measurement table
    st.subheader("Measurements")
    tkey = f"s_table_{fkey}"
    if tkey not in ss:
        per_file = {}
        for m in meas:
            per_file[m.source] = per_file.get(m.source, 0) + (m.kind == "eis")
        # a temperature in the file name applies only when that file holds a single spectrum
        T_names = [S.temperature_from_name(m.source) if m.kind == "eis" and per_file[m.source] == 1 else None
                   for m in meas]
        ss[tkey + "_Tnames"] = sum(t is not None for t in T_names)
        ss[tkey] = pd.DataFrame({
            "use": [m.kind != "other" for m in meas],
            "name": [m.name for m in meas],
            "type": [{"eis": "EIS", "ca": "DC step", "other": "other"}[m.kind] for m in meas],
            "file": [m.source for m in meas],
            "content": [m.summary() for m in meas],
            "T (°C)": T_names,
            "thickness (mm)": [None] * len(meas),
            "diameter (mm)": [None] * len(meas),
            "mass (mg)": [None] * len(meas),
            "E (V)": [round(float(np.nanmean(m.E)), 4) if m.E is not None else None for m in meas],
        })
        ss[tkey + "_v"] = 0
    if ss.get(tkey + "_Tnames"):
        st.caption(f"ℹ {ss[tkey + '_Tnames']} temperature(s) were read from the file names. Check them in the table.")
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    tlist = c1.text_input("Quick entry: temperatures of the spectra in order, °C (comma-separated)",
                          placeholder="25, 40, 55, 70, 85", key=f"s_tlist_{fkey}")
    if c2.button("Fill", width="stretch"):
        try:
            vals = [float(v) for v in tlist.replace(";", ",").split(",") if v.strip()]
            df = ss[tkey].copy()
            idx = [i for i, t in enumerate(df["type"]) if t == "EIS"]
            for i, v in zip(idx, vals):
                df.at[i, "T (°C)"] = v
            ss[tkey] = df
            ss[tkey + "_v"] += 1
            st.rerun()
        except ValueError:
            st.error("Temperatures must be numbers separated by commas.")
    table = st.data_editor(
        ss[tkey], key=f"s_ed_{fkey}_{ss[tkey + '_v']}", hide_index=True, width="stretch",
        disabled=["name", "type", "file", "content"],
        column_config={
            "use": st.column_config.CheckboxColumn("use", help="Include in the analysis and figures"),
            "T (°C)": st.column_config.NumberColumn("T (°C)", format="%.4g",
                                                    help="Needed for the Arrhenius fit; leave empty otherwise"),
            "thickness (mm)": st.column_config.NumberColumn("thickness (mm)", format="%.4g",
                                                            help="Per-sample value (empty = sidebar value)"),
            "diameter (mm)": st.column_config.NumberColumn("diameter (mm)", format="%.4g",
                                                           help="Per-sample value (empty = sidebar value)"),
            "mass (mg)": st.column_config.NumberColumn("mass (mg)", format="%.4g",
                                                       help="Pellet mass, for the relative density"),
            "E (V)": st.column_config.NumberColumn("E (V)", format="%.4g",
                                                   help="DC steps: potential (read from the file; edit to override)"),
        })
    eis_rows = [(i, meas[i]) for i in range(len(meas)) if table.at[i, "use"] and meas[i].kind == "eis"]
    # spectra in order of temperature (those without one last, in table order)
    eis_rows.sort(key=lambda r: (_num(table.at[r[0], "T (°C)"]) is None, _num(table.at[r[0], "T (°C)"]) or 0.0))
    dc_rows = [(i, meas[i]) for i in range(len(meas)) if table.at[i, "use"] and meas[i].kind in ("ca", "other")]

    def geom_of(i):
        return S.sample_geometry(geom0, _num(table.at[i, "thickness (mm)"]), _num(table.at[i, "diameter (mm)"]))

    def rho_of(i):
        return S.relative_density(geom_of(i), _num(table.at[i, "mass (mg)"]), rho_th)

    def label_of(i):
        T = _num(table.at[i, "T (°C)"])
        return f"{T:g} °C" if T is not None else meas[i].name

    tab_eis, tab_dc, tab_dl = st.tabs(["Impedance: σ and Eₐ", "DC polarisation: σₑ", "Downloads"])
    width = pl.WIDTHS[ss.s_width]
    fs = ss.s_font
    outputs = {"figures": {}, "csv": {}, "settings": {}}

    # ------------------------------------------------------------------ impedance
    with tab_eis:
        if not eis_rows:
            st.info("No impedance spectra selected.")
        else:
            with st.form("s_eis_form"):
                c1, c2 = st.columns([1.3, 1])
                method_lbl = c1.selectbox("Ionic resistance from", list(S.R_METHODS), index=0)
                preset = c2.selectbox("Circuit preset", list(PRESETS), index=0)
                circuit = st.text_input("Circuit (impedance.py syntax), used when fitting",
                                        value=PRESETS[preset] or ss.get("s_circuit_custom", "p(R1,CPE1)-CPE2"))
                c1, c2, c3 = st.columns(3)
                rsel = c1.text_input("Resistances that make up the ionic resistance",
                                     value="all", help="Comma-separated element names (e.g. R1 for bulk, "
                                     "R1,R2 for total), or 'all'.")
                n_starts = c2.number_input("Multi-start fits per spectrum", 1, 20, 4)
                tsel = c3.selectbox("Arrhenius quantity", ["σT (S cm⁻¹ K)", "σ (S cm⁻¹)"], index=0,
                                    help="ln(σT) vs 1/T follows from the Nernst–Einstein relation and is the usual "
                                         "choice for ionic conductors; Eₐ from σ differs by about kT.")
                c1, c2, c3, c4 = st.columns(4)
                do_drt = c1.checkbox("Compute DRT", value=True)
                lam = c2.number_input("DRT λ (manual)", value=1e-3, min_value=1e-8, format="%.1e")
                trim = c3.checkbox("Leave the blocking spike out of the DRT", value=True,
                                   help="Points after the minimum of −Z″ (the capacitive tail) have no finite "
                                        "resistance and distort a DRT.")
                fwin = c4.text_input("Frequency window (Hz), e.g. 1e6-10", value="",
                                     help="Applied to every spectrum. Empty = all points.")
                go = st.form_submit_button("Analyse spectra", type="primary")
            ss["s_circuit_custom"] = circuit
            method = S.R_METHODS[method_lbl]
            f_window = None
            if fwin.strip():
                try:
                    a, b = [float(x) for x in fwin.replace(" ", "").split("-", 1)] if fwin.count("-") == 1 else \
                        [float(x) for x in fwin.replace(" ", "").split(",")]
                    f_window = (min(a, b), max(a, b))
                except ValueError:
                    st.error("Frequency window: write it as 'high-low', e.g. 1e6-10.")
            r_names = None if rsel.strip().lower() in ("", "all") else [x.strip() for x in rsel.split(",")]
            akey = _h(fkey, method, circuit, r_names, int(n_starts), do_drt, lam, trim, f_window,
                      [(i, _num(table.at[i, "T (°C)"])) for i, _ in eis_rows])
            if go:
                if method == "ecm":
                    try:
                        ecmm.parse_circuit(circuit)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Circuit error: {exc}")
                        st.stop()
                results = {}
                bar = st.progress(0.0, text="Analysing…")
                for k, (i, m) in enumerate(eis_rows):
                    bar.progress(k / len(eis_rows), text=f"{m.name}: resistance…")
                    r = S.analyse_spectrum(m, _num(table.at[i, "T (°C)"]), method, None, circuit, r_names,
                                           int(n_starts), 20.0, f_window)
                    if do_drt:
                        bar.progress((k + 0.5) / len(eis_rows), text=f"{m.name}: DRT…")
                        f_, Z_ = m.freq, m.Z
                        if f_window:
                            keep = (f_ >= f_window[0]) & (f_ <= f_window[1])
                            f_, Z_ = f_[keep], Z_[keep]
                        if trim:
                            j = S.tail_start(f_, Z_)
                            if j is not None:
                                f_, Z_ = f_[:j + 1], Z_[:j + 1]
                        try:
                            r.drt = drtm.run_drt(f_, Z_, drtm.DRTSettings(cv_type="custom", reg_param=float(lam),
                                                                          inductance="discard"))
                            r.notes += [f"DRT: {w}" for w in r.drt.warnings]
                        except Exception as exc:  # noqa: BLE001
                            r.notes.append(f"DRT failed: {exc}")
                    results[i] = r
                bar.progress(1.0, text="Done")
                bar.empty()
                ss["s_results"], ss["s_results_key"] = results, akey
            results = ss.get("s_results")
            if results is not None and ss.get("s_results_key") != akey:
                st.info("Settings or temperatures changed since the last analysis. Press **Analyse spectra** to "
                        "update.")
            if results is not None and not all(i in results for i, _ in eis_rows):
                results = None
                st.info("The selection changed. Press **Analyse spectra**.")

            # per-spectrum table (conductivity uses each sample's own geometry)
            rows, sig_ok, arr = [], [], None
            if results:
                for i, m in eis_rows:
                    r = results[i]
                    g = geom_of(i)
                    c = cond.conductivity(r.R, g, r.dR) if (r.R and g) else None
                    rho, rrel = rho_of(i)
                    T = _num(table.at[i, "T (°C)"])
                    rows.append({"spectrum": m.name, "file": m.source, "T (°C)": T,
                                 "R (Ω)": r.R, "± R (Ω)": r.dR, "σ (S/cm)": c["sigma_S_cm"] if c else None,
                                 "± σ (S/cm)": c["dsigma_S_cm"] if c else None,
                                 "L/A (1/cm)": g.cell_constant_per_cm if g else None,
                                 "ρ (g/cm³)": rho, "ρ_rel": rrel,
                                 "fit: max |res| (%)": (r.fit.metrics.get("max_abs_rel_res_pct") if r.fit else None),
                                 "notes": " · ".join(r.notes)})
                    if c and T is not None:
                        sig_ok.append((i, T, c["sigma_S_cm"], c["dsigma_S_cm"]))
                df = pd.DataFrame(rows)
                st.dataframe(df.drop(columns=["notes"]), hide_index=True, width="stretch",
                             column_config={k: st.column_config.NumberColumn(format="%.4g") for k in df.columns
                                            if k not in ("spectrum", "file", "notes")})
                bad = [(r["spectrum"], r["notes"]) for r in rows if r["notes"]]
                if bad:
                    with st.expander(f"Notes and warnings ({len(bad)} spectra)"):
                        for n, t in bad:
                            st.markdown(f"**{n}**: {t}")
                if geom0 is None:
                    st.info("Enter thickness and diameter (sidebar or table) to get conductivities.")
                outputs["csv"]["series_spectra_results.csv"] = ex.csv_bytes(df)

                if len(sig_ok) >= 2 and len({t for _, t, _, _ in sig_ok}) >= 2:
                    try:
                        arr = S.arrhenius([t for _, t, _, _ in sig_ok], [s for _, _, s, _ in sig_ok],
                                          times_T=tsel.startswith("σT"))
                    except ValueError as exc:
                        st.warning(str(exc))
                if arr:
                    resid = arr["y"] - (arr["slope"] * arr["x"] + arr["intercept"])
                    bad_T = [(sig_ok[k], r) for k, r in enumerate(resid) if abs(r) > 0.15]
                    if bad_T:
                        st.warning("These spectra lie far from the Arrhenius line ("
                                   + ", ".join(f"{meas[i].name} at {t:g} °C: {10 ** r:.2g}× the fitted σT"
                                               for (i, t, _, _), r in bad_T)
                                   + "). Check their fits and notes before trusting Eₐ: a fit that found a wrong "
                                     "minimum, or a resistance that means something different at that temperature, "
                                     "is the usual cause. Untick them in the table to exclude them.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Activation energy",
                              f"{arr['Ea_eV']:.3f} ± {arr['dEa_eV']:.1g} eV" if np.isfinite(arr["dEa_eV"])
                              else f"{arr['Ea_eV']:.3f} eV", f"{arr['Ea_kJ_mol']:.1f} kJ/mol", delta_color="off")
                    c2.metric("σ at 25 °C (from the fit)", f"{arr['sigma25_S_cm']:.3e} S/cm",
                              "extrapolated" if arr["sigma25_extrapolated"] else "interpolated", delta_color="off")
                    c3.metric("R² of the Arrhenius fit", f"{arr['r2']:.4f}", f"{arr['n']} temperatures",
                              delta_color="off")
                    st.caption("Uncertainty of Eₐ is the standard error of the slope (scatter about the line). It "
                               "does not include systematic errors such as a thermocouple offset or a resistance "
                               "assignment that changes with temperature (e.g. bulk → total as an arc leaves the "
                               "window).")
                    outputs["csv"]["series_arrhenius.csv"] = ex.csv_bytes(pd.DataFrame([{
                        k: v for k, v in arr.items() if not isinstance(v, np.ndarray)}]))
                elif len(eis_rows) > 1:
                    st.caption("Enter at least two different temperatures to fit an activation energy.")

            # ---------------- figures
            st.subheader("Figures")
            c1, c2, c3, c4 = st.columns([2.2, 0.8, 1, 1])
            norm_lbl = c1.selectbox("Normalise impedance", list(S.NORM_MODES), index=0, key="s_norm",
                                    help="Geometry removes thickness/area differences between pellets. Density also "
                                         "scales by ρ_rel^m (σ ≈ σ_dense·ρ_rel^m; m = 1 linear, 1.5 Bruggeman for "
                                         "spherical pores). Shape scales every spectrum by its own resistance.")
            m_exp = c2.number_input("m", value=1.5, min_value=0.0, max_value=4.0, step=0.5, key="s_mexp",
                                    disabled=S.NORM_MODES[norm_lbl] != "dense")
            drt_x = c3.selectbox("DRT x-axis", ["τ (s)", "f = 1/(2πτ) (Hz)"], key="s_drtx")
            show_fit = c4.checkbox("Show fits", value=True, key="s_showfit")
            c1, c2 = st.columns([1, 2])
            hide = c1.checkbox("Hide blocking spikes in the Nyquist plot", value=True, key="s_hide")
            zoom = c2.number_input("…or show Z′ up to (plotted units; 0 = automatic)", value=0.0, min_value=0.0,
                                   format="%.4g", key="s_zoom")
            mode = S.NORM_MODES[norm_lbl]
            colors, by_T = sp.series_colors([_num(table.at[i, "T (°C)"]) for i, _ in eis_rows])
            items, unit, missing = [], None, []
            for (i, m), col in zip(eis_rows, colors):
                r = results.get(i) if results else None
                nf = S.norm_factor(mode, geom_of(i), rho_of(i)[1], m_exp, r.R if r else None)
                if nf is None:
                    missing.append(m.name)
                    nf = (1.0, "Ω")
                k, unit_i = nf
                unit = unit or unit_i
                lab = label_of(i)
                rr = rho_of(i)[1]
                if mode == "dense" and rr:
                    lab += f" (ρ_rel {rr:.2f})"
                it = {"label": lab, "color": col, "f": m.freq, "Z": m.Z * k}
                if r is not None and show_fit and r.fit is not None:
                    it["f_line"], it["Z_line"] = r.fit.f_smooth, r.fit.Z_smooth * k
                if r is not None and r.drt is not None:
                    it["tau"], it["gamma"] = r.drt.tau, r.drt.gamma * k
                items.append(it)
            if missing:
                st.warning(f"Normalisation not possible for {', '.join(missing)} (needs "
                           + {"geom": "geometry", "dense": "geometry, mass and theoretical density",
                              "area": "geometry", "shape": "an analysed resistance"}.get(mode, "data")
                           + "); those spectra are plotted in Ω and the units are mixed.")
            unit_lbl = {"R": "R"}.get(unit, unit) if unit else "Ω"
            if mode == "shape":
                unit_lbl = "R"
            fig_eis = sp.fig_series_eis(width, items, unit_lbl, fs, zoom if zoom else ("auto" if hide else None),
                                        "tau" if drt_x.startswith("τ") else "f")
            _show(fig_eis)
            _fig_downloads(fig_eis, "series_impedance", "s_dl_eis")
            if by_T:
                st.caption("Colours run from dark blue (coldest) to orange (hottest).")
            outputs["figures"]["series_impedance"] = fig_eis
            if results and sig_ok and arr:
                cols_T = sp.series_colors([t for _, t, _, _ in sig_ok])[0]
                fig_a = sp.fig_arrhenius(width, arr, [t for _, t, _, _ in sig_ok], [s for _, _, s, _ in sig_ok],
                                         [d for _, _, _, d in sig_ok], colors=cols_T, font_size=fs)
                _show(fig_a)
                _fig_downloads(fig_a, "series_arrhenius", "s_dl_arr")
                outputs["figures"]["series_arrhenius"] = fig_a
            # long-format data
            long = []
            for (i, m), it in zip(eis_rows, items):
                long.append(pd.DataFrame({"spectrum": m.name, "file": m.source, "T_C": _num(table.at[i, "T (°C)"]),
                                          "freq_Hz": m.freq, "Zre_ohm": m.Z.real, "Zim_ohm": m.Z.imag,
                                          f"Zre_plotted_{unit_lbl}": it["Z"].real,
                                          f"minus_Zim_plotted_{unit_lbl}": -it["Z"].imag}))
            outputs["csv"]["series_spectra_data.csv"] = ex.csv_bytes(pd.concat(long))
            drts = [pd.DataFrame({"spectrum": m.name, "T_C": _num(table.at[i, "T (°C)"]), "tau_s": it["tau"],
                                  "f_Hz": 1 / (2 * np.pi * it["tau"]), f"gamma_plotted_{unit_lbl}": it["gamma"]})
                    for (i, m), it in zip(eis_rows, items) if it.get("tau") is not None]
            if drts:
                outputs["csv"]["series_drt.csv"] = ex.csv_bytes(pd.concat(drts))
            outputs["settings"]["impedance"] = {"method": method_lbl, "circuit": circuit, "resistances": rsel,
                                                "n_starts": int(n_starts), "drt": do_drt, "drt_lambda": lam,
                                                "drt_trim_spike": trim, "f_window_Hz": f_window,
                                                "arrhenius": tsel, "normalisation": norm_lbl, "m": m_exp}

    # ------------------------------------------------------------------ DC polarisation
    with tab_dc:
        st.markdown("Steady-state currents of potentiostatic steps on a cell with **ion-blocking electrodes** "
                    "(e.g. Au, stainless steel): once the ionic current has decayed, the remaining current is "
                    "electronic, and the slope of I<sub>ss</sub> against E gives R<sub>e</sub> and "
                    "σ<sub>e</sub> = L/(R<sub>e</sub>·A).", unsafe_allow_html=True)
        if not dc_rows:
            st.info("No DC steps selected.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            tail = c1.number_input("Average over the last (% of each step)", 1.0, 100.0, 10.0, step=5.0, key="s_tail")
            origin = c2.checkbox("Force the line through the origin", value=False, key="s_origin")
            logI = c3.checkbox("Log current axis", value=True, key="s_logI")
            res_now = ss.get("s_results") or {}
            cmp_opts = ["–"] + [meas[i].name for i, _ in eis_rows if i in res_now and res_now[i].R]
            cmp = c4.selectbox("Compare with spectrum (σ_ion)", cmp_opts, key="s_cmp",
                               help="Same pellet, same temperature as the DC steps: gives the electronic "
                                    "transference number t_e = σ_e / (σ_e + σ_ion).")
            steps, sres = [], []
            for i, m in dc_rows:
                r = S.steady_state(m, tail / 100, _num(table.at[i, "E (V)"]))
                sres.append((i, m, r))
            E_cols = sp.series_colors([r.E for _, _, r in sres])[0]
            for (i, m, r), col in zip(sres, E_cols):
                steps.append({"label": f"{r.E:.3g} V" if np.isfinite(r.E) else m.name, "color": col, "t": m.t,
                              "I": m.I, "window": r.t_window, "I_ss": r.I_ss, "dI": r.dI, "E": r.E})
            dfs = pd.DataFrame([{"step": m.name, "file": m.source, "E (V)": r.E, "I_ss (A)": r.I_ss, "± I_ss (A)": r.dI,
                                 "drift over window (%)": r.drift_pct, "window (s)": f"{r.t_window[0]:.4g}–{r.t_window[1]:.4g}",
                                 "notes": " · ".join(m.notes + r.notes)} for i, m, r in sres])
            st.dataframe(dfs, hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.4g")
                                        for k in ("E (V)", "I_ss (A)", "± I_ss (A)", "drift over window (%)")})
            outputs["csv"]["series_dc_steps.csv"] = ex.csv_bytes(dfs)
            fit, sig_txt = None, None
            try:
                fit = S.dc_fit([r.E for _, _, r in sres], [r.I_ss for _, _, r in sres], origin)
            except ValueError as exc:
                st.warning(str(exc))
            if fit:
                g = geom_of(dc_rows[0][0])
                ce = cond.conductivity(fit["R_e_ohm"], g, fit["dR_e_ohm"]) if g else None
                c1, c2, c3 = st.columns(3)
                c1.metric("R_e", f"{fit['R_e_ohm']:.4g} Ω", f"± {fit['dR_e_ohm']:.2g} Ω" if fit["dR_e_ohm"] else None,
                          delta_color="off")
                if ce:
                    c2.metric("σ_e", f"{ce['sigma_S_cm']:.3e} S/cm",
                              f"± {ce['dsigma_S_cm']:.1e} S/cm" if ce["dsigma_S_cm"] else None, delta_color="off")
                    sig_txt = rf"$\sigma_e$ = {ce['sigma_S_cm']:.2e} S cm$^{{-1}}$"
                else:
                    c2.info("Enter the geometry to get σ_e.")
                if ce and cmp != "–":
                    ii = next(i for i, m in eis_rows if m.name == cmp)
                    ci = cond.conductivity(res_now[ii].R, geom_of(ii), res_now[ii].dR) if geom_of(ii) else None
                    if ci:
                        te = ce["sigma_S_cm"] / (ce["sigma_S_cm"] + ci["sigma_S_cm"])
                        c3.metric("t_e = σ_e/(σ_e+σ_ion)", f"{te:.2e}", f"σ_ion = {ci['sigma_S_cm']:.3e} S/cm",
                                  delta_color="off")
                st.caption(f"Linear fit of I_ss vs E over {fit['n']} steps"
                           + (" through the origin" if origin else f", intercept {fit['I0_A']:.3g} A")
                           + f" · R² = {fit['r2']:.4f}. Check that every step reached steady state (drift column) "
                           "and that the potentials stay inside the electrolyte's stability window; a curving "
                           "I–V line means the electronic conductivity depends on the applied potential.")
                outputs["csv"]["series_dc_fit.csv"] = ex.csv_bytes(pd.DataFrame([{
                    "R_e_ohm": fit["R_e_ohm"], "dR_e_ohm": fit["dR_e_ohm"], "I0_A": fit["I0_A"], "r2": fit["r2"],
                    "n_steps": fit["n"], "through_origin": fit["through_origin"],
                    "sigma_e_S_cm": ce["sigma_S_cm"] if ce else None,
                    "dsigma_e_S_cm": ce["dsigma_S_cm"] if ce else None}]))
            fig_dc = sp.fig_dc(width, steps, fit, sig_txt, fs, logI)
            _show(fig_dc)
            _fig_downloads(fig_dc, "series_dc_polarisation", "s_dl_dc")
            outputs["figures"]["series_dc_polarisation"] = fig_dc
            outputs["csv"]["series_dc_curves.csv"] = ex.csv_bytes(pd.concat([
                pd.DataFrame({"step": m.name, "E_V": r.E, "t_s": m.t, "I_A": m.I}) for i, m, r in sres]))
            outputs["settings"]["dc"] = {"average_last_pct": tail, "through_origin": origin}

    # ------------------------------------------------------------------ downloads
    with tab_dl:
        outputs["settings"]["files"] = [f[1] for f in files]
        outputs["settings"]["geometry_shared"] = geom0.__dict__ if geom0 else None
        outputs["settings"]["theoretical_density_g_cm3"] = rho_th
        outputs["settings"]["table"] = table.to_dict(orient="records")
        st.markdown("Everything from the other tabs: result tables, data in long format, figures in PNG, PDF and "
                    "SVG, and a settings file with the software versions.")
        for name, data in outputs["csv"].items():
            _dl(st, f"⬇ {name}", data, file_name=name, mime="text/csv", key=f"s_dl_{name}")
        dpi = ss.s_dpi

        def build():
            files_out = dict(outputs["csv"])
            files_out["series_settings.json"] = ex.settings_json("series", outputs["settings"])
            for stem, fig in outputs["figures"].items():
                for fmt in ("png", "pdf", "svg"):
                    files_out[f"{stem}.{fmt}"] = pl.to_bytes(fig, fmt, dpi)
            return ex.zip_bundle(files_out)
        _dl(st, "⬇ Everything (.zip)", build, file_name="eis_series.zip", mime="application/zip", key="s_dl_zip",
            type="primary")

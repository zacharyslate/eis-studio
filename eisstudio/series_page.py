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
from eisstudio import ecm as ecmm
from eisstudio import export as ex
from eisstudio import plotting as pl
from eisstudio import series as S
from eisstudio import processes as PR
from eisstudio import series_plots as sp
from eisstudio.circuit_editor import PRESETS as CIRCUIT_PRESETS
from eisstudio.ui_shared import (GAMMA_MODES, circuit_block, drt_inputs, edit_param_table, fit_options,
                                 param_table_df, read_param_table)

SERIES_DIR = Path(__file__).parent / "sample_data" / "series"
SAMPLE_SETS = {
    "Temperature series + DC steps (one EC-Lab file)": ["LPSC_pellet_T-series_and_DC_linked.mpt"],
    "Temperature series as separate files + DC table": [f"pellet_{t}C.csv" for t in (25, 40, 55, 70, 85)]
    + ["pellet_DC_polarisation.txt"],
    "Five pellets pressed to different densities": sorted(p.name for p in SERIES_DIR.glob("density_*.csv")),
    "Bulk + grain boundary, 25–85 °C (separate files)": [f"polycrystal_{t}C.csv" for t in (25, 40, 55, 70, 85)],
}
SAMPLE_HINTS = {
    "Temperature series + DC steps (one EC-Lab file)": "Simulated at 25, 40, 55, 70, 85 °C; pellet 1.00 mm × "
    "10.0 mm, 125.7 mg. Type the temperatures below.",
    "Temperature series as separate files + DC table": "Same pellet as the EC-Lab sample: 25, 40, 55, 70, 85 °C.",
    "Bulk + grain boundary, 25–85 °C (separate files)": "Pellet 1.00 mm × 10.0 mm. Bulk: σ(25 °C) = 1.0 × 10⁻⁵ "
    "S/cm, Eₐ = 0.40 eV; grain boundary: R_gb = 3 R_bulk at 25 °C, Eₐ = 0.55 eV. Use the preset with two arcs from "
    "the origin + blocking electrodes, p(R1,CPE1)-p(R2,CPE2)-CPE3, and see Arrhenius by process.",
    "Five pellets pressed to different densities": "All 10.0 mm diameter, 25 °C; thickness and mass are in the "
    "file names (theoretical density 1.86 g/cm³). Enter them per pellet in the table.",
}
EDITOR_PRESETS = [
    ("Pellet, arc from the origin + blocking electrodes", "p(R1,CPE1)-CPE2"),
    ("Pellet, two arcs from the origin + blocking electrodes", "p(R1,CPE1)-p(R2,CPE2)-CPE3"),
] + [p for p in CIRCUIT_PRESETS if p[1] not in ("p(R1,CPE1)-CPE2", "p(R1,CPE1)-p(R2,CPE2)-CPE3")]


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


def _fig_downloads(fig, stem, key, target=st):
    fmt, dpi = st.session_state.s_fmt, st.session_state.s_dpi
    _dl(target, f"⬇ {stem} (.{fmt})", lambda: pl.to_bytes(fig, fmt, dpi), file_name=f"{stem}.{fmt}", mime=pl.MIME[fmt],
        key=key)


def _fmtv(v, digits=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{digits}g}"


def _peak_flag(p, d, cut_applied=True):
    """Short warning for a DRT peak: tail onset near/beyond the low-frequency end of the data, outside the window
    at high frequency, or minor."""
    tau_end = 1 / (2 * np.pi * float(d.freq.min()))
    if cut_applied and p["tau_s"] > tau_end:
        return "beyond the low-f cut: tail onset?"
    if cut_applied and p["tau_s"] > tau_end / 3:
        return "near the low-f cut: tail onset?"
    if p.get("outside"):
        return "outside the window"
    return "minor" if p["minor"] else ""


def _process_section(procs, proc_notes, geom, times_T, width, fs, outputs, akey):
    """Arrhenius plot and activation energy of every process of the circuit, tracked by time constant."""
    st.markdown("#### Arrhenius by process")
    st.caption("Each R–C/CPE pair of the circuit is one process, followed across the temperatures by its time "
               "constant (fastest first) rather than by its name. The class comes from the effective capacitance "
               "scaled to unit cell constant (Irvine, Sinclair & West 1990): bulk below 2 × 10⁻¹¹ F, grain "
               "boundary to 10⁻⁸ F, surface layer or interphase to 10⁻⁷ F, electrode interface above. Correct it "
               "in the table if you know better.")
    cc = geom.cell_constant_per_cm if geom else None
    if cc is None:
        st.info("Enter the sample geometry to scale the capacitances and get conductivities; classes below use "
                "the raw capacitance.")
    for n in proc_notes:
        st.caption(f"ℹ {n}")
    rng = lambda v: f"{min(v):.2g}–{max(v):.2g}" if v else "–"  # noqa: E731
    base = pd.DataFrame([{
        "process": pr.label, "class": pr.cls,
        "C_eff (F)": rng([p.C_eff for p in pr.points if p.C_eff]),
        "C_eff·L/A (F cm⁻¹)": rng([p.C_eff * cc for p in pr.points if p.C_eff]) if cc else "–",
        "f_c (Hz)": rng([p.f_c for p in pr.points if p.f_c]),
        "outside window": ", ".join(f"{p.T_C:g} °C" for p in pr.points if p.outside) or "–"} for pr in procs])
    ed = st.data_editor(base, key=f"s_proc_ed_{akey}", hide_index=True, width="stretch",
                        disabled=[c for c in base.columns if c != "class"],
                        column_config={"class": st.column_config.SelectboxColumn("class", options=PR.CLASS_NAMES,
                                                                                 required=True)})
    for pr, c in zip(procs, ed["class"]):
        pr.cls = c
    c1, c2 = st.columns(2)
    incl_r = not c1.checkbox("Resistances: leave out arcs whose apex lies outside the window", value=False,
                             key="s_proc_excl_r",
                             help="An arc's resistance is often still fixed by where it meets the next arc or the "
                                  "spike even when its apex is above the highest frequency (check against Z′ at the "
                                  "−Z″ minimum). Tick this when the split between two resistances is uncertain.")
    incl_f = c2.checkbox("Time constants: include arcs whose apex lies outside the window", value=False,
                         key="s_proc_incl_f",
                         help="Off by default: the characteristic frequency of such an arc is an extrapolation of "
                              "the fitted CPE.")
    bulk = next((pr for pr in procs if pr.cls == "bulk" and pr.kind == "pair"), None)
    palette = sp.OKABE_ITO
    rows, cond_series, freq_series = [], [], []
    for k, pr in enumerate(procs):
        col = palette[k % len(palette)]
        q = {"bulk": "sigma", "grain boundary": "sigma_gb"}.get(pr.cls)
        fc = PR.fit(pr, q, cc, bulk, incl_r, times_T) if q else None
        fr = PR.fit(pr, "inv_R", cc, bulk, incl_r)
        ff = PR.fit(pr, "f_c", cc, bulk, incl_f)
        fd = PR.fit(pr, "drt", cc, bulk, False)
        e = lambda a: f"{a['Ea_eV']:.3f} ± {a['dEa_eV']:.3f}" if a and np.isfinite(a["dEa_eV"]) else (  # noqa: E731
            f"{a['Ea_eV']:.3f}" if a else "–")
        alphas = [p.alpha for p in pr.points if p.alpha]
        sname = ("σT" if times_T else "σ") if q != "sigma_gb" else ("specific σ_gb·T" if times_T else "specific σ_gb")
        rows.append({"process": pr.label, "class": pr.cls,
                     f"Eₐ from {sname} (eV)": e(fc),
                     "Eₐ from 1/R (eV)": e(fr), "Eₐ from f_c (eV)": e(ff), "Eₐ from DRT peak (eV)": e(fd),
                     "mean CPE α": f"{np.mean(alphas):.2f}" if alphas else "–",
                     "temperatures": len(pr.points),
                     "apex outside window": sum(p.outside for p in pr.points)})
        if fc:
            cond_series.append({"label": pr.label + (" (specific σ_gb)" if q == "sigma_gb" else ""), "color": col,
                                "T_C": fc["T_all"], "y": fc["y_all"], "used": fc["used"], "fit": fc})
        if ff:
            freq_series.append({"label": pr.label + ", ECM", "color": col, "T_C": ff["T_all"], "y": ff["y_all"],
                                "used": ff["used"], "fit": ff})
        if fd:
            freq_series.append({"label": pr.label + ", DRT", "color": col, "T_C": fd["T_all"], "y": fd["y_all"],
                                "used": fd["used"], "fit": fd, "marker": "x"})
    tab = pd.DataFrame(rows).fillna("–")
    st.dataframe(tab, hide_index=True, width="stretch")
    st.caption(
        "A fit needs at least three temperatures (\"–\" otherwise). Hollow symbols in the figure are left out of "
        "that fit. Read the columns together. **1/R** is the activation energy of the process's resistance. For a bulk "
        "process **σ** gives the same value; σT, if chosen above, adds about k_BT. For a grain boundary, the "
        "**specific σ_gb** of the brick-layer model, (L/A)(C_bulk/C_gb)/R_gb, uses effective capacitances; for "
        "CPE elements these follow R as R^((1−α)/α), which adds roughly Eₐ(1−α)/α. For an R‖CPE arc with a "
        "temperature-independent Q, the characteristic frequency and the DRT peak follow (RQ)^(−1/α) and give "
        "about Eₐ/α. Agreement within those factors supports the assignment; a large mismatch means the fit "
        "gives the element different meanings at different temperatures. An electrode or interphase process has "
        "a resistance, not a conductivity, so only 1/R and f_c are given for it.")
    for pr in procs:
        if pr.notes:
            st.caption(f"⚠ **{pr.label}**: " + " ".join(pr.notes))
    if cond_series or freq_series:
        fig = sp.fig_processes(width, cond_series, freq_series, times_T, fs)
        _show(fig)
        _fig_downloads(fig, "series_arrhenius_by_process", "s_dl_proc")
        outputs["figures"]["series_arrhenius_by_process"] = fig
    outputs["csv"]["series_processes.csv"] = ex.csv_bytes(pd.DataFrame([{
        "process": pr.label, "class": pr.cls, "T_C": p.T_C, "element": p.element, "R_ohm": p.R, "dR_ohm": p.dR,
        "C_eff_F": p.C_eff, "alpha": p.alpha, "f_c_Hz": p.f_c, "outside_window": p.outside,
        "drt_peak_tau_s": pr.drt_tau.get(p.key)} for pr in procs for p in pr.points]))
    outputs["csv"]["series_process_activation_energies.csv"] = ex.csv_bytes(tab)


def _inspector(eis_rows, results, table, label_of, method, circuit, r_names, names, units, n_starts, time_budget,
               f_window, wkey, global_opt, drt_s, cut, theta, gm, fkey, width, fs, zoom, akey_of, ecm_ok):
    """One spectrum in detail: its fit, DRT and KK check, with individual settings if needed."""
    ss = st.session_state
    st.markdown("#### Inspect one spectrum")
    k = st.selectbox("Spectrum", range(len(eis_rows)), key=f"s_pick_{fkey}",
                     format_func=lambda j: f"{label_of(eis_rows[j][0])} · {eis_rows[j][1].name}")
    i, m = eis_rows[k]
    r = results[i]
    T = _num(table.at[i, "T (°C)"])
    okey = f"{fkey}:{i}"
    ovs = ss.setdefault("s_overrides", {})
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in m.name)

    def _store(new_ov):
        if new_ov:
            ovs[okey] = new_ov
        else:
            ovs.pop(okey, None)
        ov_now = {int(kk.split(":", 1)[1]): v for kk, v in ovs.items() if kk.split(":", 1)[0] == fkey}
        ss["s_results_key"] = akey_of(ov_now)

    for n in r.notes:
        st.caption(f"ℹ {n}")
    t_ecm, t_drt, t_kk = st.tabs(["Equivalent circuit", "DRT", "Kramers–Kronig"])
    zoom_i = sp.auto_zoom([{"f": m.freq, "Z": m.Z}]) if zoom == "auto" else zoom

    # ---------------- ECM
    with t_ecm:
        res = r.fit
        if res is None:
            st.info("No circuit fit for this spectrum." if method == "ecm" else
                    "The resistance comes from the spectrum directly (no circuit fit). Choose *Equivalent-circuit "
                    "fit* above to fit a circuit to every spectrum.")
        else:
            mt = res.metrics
            st.markdown(
                f"**Fit quality** · pseudo-χ² = {mt['pseudo_chi2']:.3e} · mean |Δ| Re/Im = "
                f"{mt['mean_abs_rel_res_re_pct']:.2f} / {mt['mean_abs_rel_res_im_pct']:.2f} % · max |Δ| = "
                f"{mt['max_abs_rel_res_pct']:.2f} % · best minimum reached by "
                f"{mt.get('n_starts_at_best_minimum', 1)}/{mt.get('n_starts_converged', 1)} starts"
                + (" · **fitted with its own settings**" if r.individual else ""))
            c1, c2 = st.columns([3, 2])
            prow = [{"parameter": n, "value": _fmtv(v), "± std. error": ("fixed" if fx else _fmtv(e, 2)),
                     "rel. error": ("–" if fx or v == 0 or not np.isfinite(e) else f"{abs(e / v) * 100:.1f} %"),
                     "unit": u} for n, u, v, e, fx in res.all_params()]
            c1.dataframe(pd.DataFrame(prow), hide_index=True, width="stretch")
            if res.derived:
                c2.dataframe(pd.DataFrame([{
                    "element": d["pair"], "C_eff / F": _fmtv(d["C_eff_F"], 3), "α": f"{d['alpha']:.3f}",
                    "f_c / Hz": _fmtv(d["f_c_Hz"], 3)} for d in res.derived]), hide_index=True, width="stretch")
                c2.caption("Typical C_eff (Irvine, Sinclair & West 1990): bulk ~10⁻¹² F, grain boundary "
                           "10⁻¹¹–10⁻⁸ F, electrode/interface 10⁻⁷–10⁻⁵ F (per cm geometry: scale by L/A).")
            rr, ri = ecmm.relative_residuals(res.Z, res.Z_fit)
            comp = pl.fig_ecm_composite(width, res.freq, res.Z, res.f_smooth, res.Z_smooth, res.Z_fit, rr, ri, 1.0,
                                        "Ω", True, fs, zoom_i)
            _show(comp)
            c1, c2, c3 = st.columns(3)
            _fig_downloads(comp, f"{stem}_ECM", "s_i_dl_ecm", c1)
            _dl(c2, "⬇ Parameters (.csv)", ex.csv_bytes(ex.ecm_params_df(res, None)),
                file_name=f"{stem}_ECM_parameters.csv", mime="text/csv", key="s_i_dl_par")
            _dl(c3, "⬇ Spectrum + fit (.csv)", ex.csv_bytes(ex.ecm_spectrum_df(res)),
                file_name=f"{stem}_ECM_spectrum.csv", mime="text/csv", key="s_i_dl_spec")

        if method == "ecm" and ecm_ok and circuit:
            with st.expander("Refit this spectrum with its own settings", expanded=r.individual):
                st.caption("For a spectrum the series settings don't suit (an arc leaving the window, a wrong "
                           "minimum). Its starting values are this spectrum's current result; edit them, fix or "
                           "bound parameters, and refit. The spectrum keeps these settings when the series is "
                           "analysed again with the same circuit. Mind that the ionic resistance must still mean "
                           "the same thing as for the other spectra, or Eₐ is not meaningful.")
                if res is not None:
                    vals = {n: v for n, _, v, _, _ in res.all_params()}
                    g0 = [vals.get(n, np.nan) for n in names]
                else:
                    g0 = ecmm.default_guess(names, m.freq, m.Z, circuit)
                itkey = f"s_itab_{okey}_{circuit}"
                if itkey not in ss or ss.get("reset_" + itkey):
                    df0 = param_table_df(names, units, g0, circuit)
                    if res is not None:
                        df0["fixed"] = [bool(fx) for *_, fx in res.all_params()]
                    ss[itkey] = df0
                    ss["reset_" + itkey] = False
                itab = edit_param_table(ss[itkey], f"s_ied_{itkey}")
                c1, c2, c3 = st.columns(3)
                if c1.button("Refit this spectrum", type="primary", key="s_i_refit"):
                    try:
                        ini, fx, bd = read_param_table(itab)
                        with st.spinner("Fitting…"):
                            new = S.analyse_spectrum(m, T, method, None, circuit, r_names, n_starts, time_budget,
                                                     f_window, ini, fx, bd, wkey, global_opt)
                        new.individual = True
                        new.notes.insert(0, "Fitted with its own settings.")
                        new.kk, new.drt, new.drt_f_cut = r.kk, r.drt, r.drt_f_cut
                        new.notes += [n for n in r.notes if n.startswith(("DRT", "Lin-KK"))]
                        results[i] = new
                        ov = dict(ovs.get(okey, {}))
                        ov.update({"circuit": circuit, "initial": ini, "fixed": fx,
                                   "bounds": {kk: (float(a), float(b)) for kk, (a, b) in bd.items()}})
                        _store(ov)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                if c2.button("Use the series settings again", disabled="fixed" not in ovs.get(okey, {}),
                             key="s_i_unset"):
                    ov = {kk: v for kk, v in ovs.get(okey, {}).items()
                          if kk not in ("circuit", "initial", "fixed", "bounds")}
                    _store(ov)
                    ss["reset_" + itkey] = True
                    st.info("This spectrum will use the series settings at the next **Analyse spectra**.")
                if c3.button("Reset this table", key="s_i_reset"):
                    ss["reset_" + itkey] = True
                    st.rerun()

    # ---------------- DRT
    with t_drt:
        d = r.drt
        if d is None:
            st.info("No DRT for this spectrum (tick *Compute DRT* above, or see its notes).")
        else:
            mt = d.metrics
            st.markdown(f"**R∞** = {d.R_inf:.4g} Ω · **L** = {d.L:.3g} H · **λ** = {d.lam:.3g} · **R_pol** = "
                        f"∫γ d ln τ = {d.R_pol:.4g} Ω · mean |Δ| Re/Im = {mt['mean_abs_rel_res_re_pct']:.2f} / "
                        f"{mt['mean_abs_rel_res_im_pct']:.2f} % · data down to {d.freq.min():.3g} Hz"
                        + (" (low-frequency tail cut)" if r.drt_f_cut else ""))
            if d.peaks:
                st.dataframe(pd.DataFrame([{
                    "peak": f"P{p['peak']}" + (" *" if p["minor"] else "") + (" †" if p.get("outside") else ""),
                    "τ / s": f"{p['tau_s']:.3g}",
                    "f = 1/(2πτ) / Hz": f"{p['f_Hz']:.3g}", "R / Ω": f"{p['R_ohm']:.4g}",
                    "share": f"{p['share_pct']:.1f} %", "C = τ/R / F": f"{p['C_eq_F']:.3g}",
                    "flag": _peak_flag(p, d, r.drt_f_cut is not None)} for p in d.peaks]),
                    hide_index=True, width="stretch")
                st.caption("\\* minor peak (< 3 % of the area) · † outside the measured window: position and area "
                           "extrapolated (at the long-τ end usually the onset of the blocking tail).")
            a_re, a_im = ecmm.relative_residuals(d.Z, d.Z_fit)
            comp = pl.fig_drt_composite(width, d.freq, d.Z, d.Z_fit, a_re, a_im, d.tau, d.gamma, d.gamma_lo,
                                        d.gamma_hi, d.gamma_mean, d.peaks, 1.0, "Ω", True, gm, fs, zoom_i)
            _show(comp)
            c1, c2, c3 = st.columns(3)
            _fig_downloads(comp, f"{stem}_DRT", "s_i_dl_drt", c1)
            _dl(c2, "⬇ γ(τ) (.csv)", ex.csv_bytes(ex.drt_gamma_df(d)), file_name=f"{stem}_DRT_gamma.csv",
                mime="text/csv", key="s_i_dl_gamma")
            _dl(c3, "⬇ Summary & peaks (.csv)", ex.csv_bytes(ex.drt_summary_df(d, None)),
                file_name=f"{stem}_DRT_summary.csv", mime="text/csv", key="s_i_dl_drtsum")
        st.markdown("**Low-frequency limit for this spectrum's DRT**")
        cur = ovs.get(okey, {}).get("drt_f_min")
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        fmin = c1.number_input("Lowest frequency given to the DRT (Hz; 0 = the series rule)",
                               value=float(cur or 0.0), min_value=0.0, format="%.4g", key=f"s_i_fmin_{okey}",
                               help="Move it and compare: a real process stays where it is, the peak made by the "
                                    "onset of a blocking tail follows the cut.")
        if c2.button("Recompute DRT", key="s_i_drt", disabled=drt_s is None):
            try:
                with st.spinner("Computing DRT…"):
                    new_d, fc = S.analyse_drt(m, drt_s, f_window, cut, theta, fmin or None)
                r.drt, r.drt_f_cut = new_d, fc
                r.notes = [n for n in r.notes if not n.startswith("DRT")] + [f"DRT: {w}" for w in new_d.warnings]
                ov = {kk: v for kk, v in ovs.get(okey, {}).items() if kk != "drt_f_min"}
                if fmin:
                    ov["drt_f_min"] = float(fmin)
                _store(ov)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"DRT failed: {exc}")
        if cur:
            c3.caption(f"Set for this spectrum: {cur:g} Hz")

    # ---------------- KK
    with t_kk:
        kk = r.kk
        if kk is None:
            st.info("No Kramers–Kronig check for this spectrum.")
        else:
            verdict = ("✅ residuals within ±1 %: consistent" if kk["max_abs_res_pct"] < 1 else
                       "⚠️ some residuals exceed 1 %: inspect where" if kk["max_abs_res_pct"] < 3 else
                       "❌ residuals exceed 3 %: the spectrum (or part of it) is not KK-consistent")
            st.markdown(f"M = {kk['M']} RC elements · μ = {kk['mu']:.3f} · mean |res| = "
                        f"{kk['mean_abs_res_pct']:.2f} % · max |res| = {kk['max_abs_res_pct']:.2f} % · {verdict}")
            f_kk = S._window(m.freq, m.Z, f_window)[0]
            kfig = pl.fig_kk(width, f_kk, kk["res_re_pct"], kk["res_im_pct"], kk["M"], font_size=fs)
            _show(kfig)
            c1, c2 = st.columns(2)
            _fig_downloads(kfig, f"{stem}_linKK", "s_i_dl_kk", c1)
            _dl(c2, "⬇ KK residuals (.csv)", ex.csv_bytes(pd.DataFrame({
                "freq_Hz": f_kk, "res_re_pct": kk["res_re_pct"], "res_im_pct": kk["res_im_pct"]})),
                file_name=f"{stem}_linKK.csv", mime="text/csv", key="s_i_dl_kkcsv")
            st.caption("Lin-KK (Schönleber et al. 2014, as implemented in impedance.py). Isolated spikes in the "
                       "residuals at round frequencies (e.g. 50 Hz, or a decade such as 1 kHz or 10 kHz where the "
                       "instrument changes current range) point to the measurement rather than the sample.")


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
            ref_i, ref_m = eis_rows[0]
            st.markdown("#### Settings")
            c1, c2 = st.columns([1.4, 1])
            method_lbl = c1.selectbox("Ionic resistance from", list(S.R_METHODS), index=0, key="s_method")
            fwin = c2.text_input("Frequency window (Hz), e.g. 1e6-10", value="", key="s_fwin",
                                 help="Applied to every spectrum (fit, DRT and KK check). Empty = all points.")
            method = S.R_METHODS[method_lbl]
            f_window = None
            if fwin.strip():
                try:
                    a, b = [float(x) for x in fwin.replace(" ", "").split("-", 1)] if fwin.count("-") == 1 else \
                        [float(x) for x in fwin.replace(" ", "").split(",")]
                    f_window = (min(a, b), max(a, b))
                except ValueError:
                    st.error("Frequency window: write it as 'high-low', e.g. 1e6-10.")

            circuit, r_names, names, units = "", None, [], []
            initial = fixed = bounds = None
            start, wkey, wlbl, n_starts, global_opt, ptable = "data", "modulus", "", 4, False, None
            ecm_ok = True
            if method == "ecm":
                with st.container(border=True):
                    circuit = circuit_block("s_", "p(R1,CPE1)-CPE2", presets=EDITOR_PRESETS, file_stem="series",
                                            width=width, font_size=fs, dpi=ss.s_dpi)
                    try:
                        names, units = ecmm.parse_circuit(circuit) if circuit else ([], [])
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Circuit error: {exc}")
                    ecm_ok = bool(names)
                    if ecm_ok:
                        r_all = [n for n in names if ecmm.element_type(n) == "R"]
                        pend = ss.pop("s_rsel_pending", None)
                        if pend and pend[0] == circuit:
                            ss[f"s_rsel_{circuit}"] = [n for n in pend[1] if n in r_all]
                        r_names = st.multiselect(
                            "Resistances that make up the ionic resistance", r_all, default=r_all,
                            key=f"s_rsel_{circuit}",
                            help="E.g. the bulk arc only, or bulk + grain boundary for the total conductivity. "
                                 "Leave out a lead/contact resistance (an R0 at the high-frequency intercept) and "
                                 "any element of the electrode interface (µF capacitance, a characteristic "
                                 "frequency below the window). Changing this needs no new analysis.")
                        if not r_names:
                            st.error("Choose at least one resistance.")
                            ecm_ok = False
                        start_lbl = st.radio(
                            "Starting values", list(S.START_MODES), key="s_start", horizontal=True,
                            help="Data-informed guesses are made for each spectrum from its own arcs and are the "
                                 "safest default. The table's values suit spectra that are all alike. Starting "
                                 "from the previous spectrum's result follows a temperature series smoothly, but "
                                 "one wrong minimum is carried on to the next spectra.")
                        start = S.START_MODES[start_lbl]
                        tkey = f"s_ptable_{circuit}_{fkey}_{ref_i}"
                        if tkey not in ss or ss.get("reset_" + tkey):
                            ss[tkey] = param_table_df(names, units, ecmm.default_guess(names, ref_m.freq, ref_m.Z,
                                                                                       circuit), circuit)
                            ss["reset_" + tkey] = False
                        st.caption(
                            f"**fixed** values and **bounds** apply to every spectrum. The **initial** values are "
                            f"the data-informed guesses for {ref_m.name} (the first spectrum); they are used for "
                            "every spectrum only with the second starting option, and for the first spectrum with "
                            "the third. Fixing a CPE exponent across a series is common; fixing a resistance or "
                            "a Q is rarely physical, since both change with temperature.")
                        ptable = edit_param_table(ss[tkey], f"s_ed_{tkey}")
                        wkey, n_starts, global_opt, wlbl = fit_options("s_", 4)
                        if st.button("Reset table", key="s_reset_table"):
                            ss["reset_" + tkey] = True
                            st.rerun()
                        try:
                            initial, fixed, bounds = read_param_table(ptable)
                        except ValueError as exc:
                            st.error(str(exc))
                            ecm_ok = False

            with st.form("s_drt_form"):
                st.markdown("**DRT**")
                c1, c2, c3 = st.columns([0.8, 2.2, 0.8])
                do_drt = c1.checkbox("Compute DRT", value=True, key="s_do_drt")
                cut_lbl = c2.selectbox(
                    "Blocking-electrode tail", list(S.DRT_CUTS), index=0, key="s_drt_cut",
                    help="A capacitive tail has no finite resistance; the DRT turns whatever part of it is "
                         "included into a large peak at the long-τ end of the data. Cutting at the −Z″ minimum "
                         "is the most conservative choice but also drops anything between the arcs and the "
                         "steep tail. Cutting where −phase first exceeds θ keeps that near-real-axis stretch, but "
                         "note that a depressed tail (CPE exponent well below 1) has a real part of its own: Z′ "
                         "keeps rising past the junction with no further process, so those points mostly describe "
                         "the tail. "
                         "Each spectrum's cut is shown in the results table, and its DRT is drawn dotted beyond "
                         "it. A different limit can be set per spectrum in the inspector below.")
                theta = c3.number_input("θ (°)", 1.0, 80.0, 15.0, step=5.0, key="s_drt_theta",
                                        help="Used by the phase rule only.")
                drt_s, gmode = drt_inputs(
                    "s_", lam_default="custom",
                    lam_help="Manual by default in series mode: the same λ for every spectrum keeps the DRTs "
                             "comparable. Automatic criteria pick very different λ from spectrum to spectrum, and "
                             "with part of a blocking tail included they often run to the edge of the search "
                             "range (under-regularised).")
                go = st.form_submit_button("Analyse spectra", type="primary")
            cut = S.DRT_CUTS[cut_lbl]

            ovs = ss.setdefault("s_overrides", {})
            ov_now = {int(k.split(":", 1)[1]): v for k, v in ovs.items() if k.split(":", 1)[0] == fkey}
            temps = [(i, _num(table.at[i, "T (°C)"])) for i, _ in eis_rows]
            time_budget = 60.0 if global_opt else 20.0

            def akey_of(ov):
                return _h(fkey, method, circuit, ptable.to_json() if ptable is not None else None, start,
                          wkey, n_starts, global_opt, do_drt, sorted(drt_s.__dict__.items()), cut, theta, f_window,
                          temps, sorted((k, sorted(v.items())) for k, v in ov.items()))
            akey = akey_of(ov_now)
            if go:
                if method == "ecm" and not ecm_ok:
                    st.error("Fix the circuit or the parameter table first.")
                else:
                    if drt_s.bayesian and do_drt:
                        st.info(f"HMC sampling for {len(eis_rows)} spectra: this takes a while.")
                    bar = st.progress(0.0, text="Analysing…")
                    results = S.analyse_series(
                        [(i, m, _num(table.at[i, "T (°C)"])) for i, m in eis_rows], method, circuit, r_names,
                        n_starts, time_budget, f_window, start, initial, fixed, bounds, wkey, global_opt,
                        drt_s if do_drt else None, cut, theta, True, ov_now,
                        progress=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
                    bar.empty()
                    ss["s_results"], ss["s_results_key"] = results, akey
            results = ss.get("s_results")
            if results is not None and ss.get("s_results_key") != akey:
                st.info("Settings or temperatures changed since the last analysis. Press **Analyse spectra** to "
                        "update.")
            if results is not None and not all(i in results for i, _ in eis_rows):
                results = None
                st.info("The selection changed. Press **Analyse spectra**.")
            # the ionic resistance follows the current choice of resistances without refitting
            procs, proc_notes = [], []
            if results and method == "ecm":
                for i, _ in eis_rows:
                    r = results[i]
                    if r.fit is not None:
                        try:
                            r.R, r.dR, use = S.ionic_from_fit(r.fit, r_names)
                            r.extra["resistances"] = "+".join(use)
                        except ValueError as exc:
                            r.R = None
                            r.notes.append(str(exc))
                g_cc = next((geom_of(i) for i, _ in eis_rows if geom_of(i)), None)
                fits_T = [(i, _num(table.at[i, "T (°C)"]), results[i].fit) for i, _ in eis_rows
                          if results[i].fit is not None and _num(table.at[i, "T (°C)"]) is not None]
                fits_T.sort(key=lambda x: x[1])
                if fits_T:
                    procs, proc_notes = PR.track(fits_T, g_cc.cell_constant_per_cm if g_cc else None)
                    drts_now = {i: results[i].drt for i, _, _ in fits_T if results[i].drt is not None}
                    if drts_now:
                        PR.match_drt(procs, drts_now)

            # per-spectrum table (conductivity uses each sample's own geometry)
            rows, sig_ok, arr = [], [], None
            if results:
                st.markdown("#### Results")
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
                                 "KK: max |res| (%)": r.kk["max_abs_res_pct"] if r.kk else None,
                                 "DRT down to (Hz)": (r.drt_f_cut if r.drt_f_cut else
                                                      (float(r.drt.freq.min()) if r.drt is not None else None)),
                                 "own settings": "yes" if r.individual else "",
                                 "notes": " · ".join(r.notes)})
                    if c and T is not None:
                        sig_ok.append((i, T, c["sigma_S_cm"], c["dsigma_S_cm"]))
                df = pd.DataFrame(rows)
                st.dataframe(df.drop(columns=["notes"]), hide_index=True, width="stretch",
                             column_config={k: st.column_config.NumberColumn(format="%.3e" if "σ" in k else "%.4g")
                                            for k in df.columns if k not in ("spectrum", "file", "notes",
                                                                             "own settings")})
                in_sum = set(r_names or [n for n in names if ecmm.element_type(n) == "R"])
                wrong = [pr for pr in procs if pr.kind == "pair" and pr.r_name in in_sum
                         and pr.cls in ("electrode interface", "surface layer / interphase")]
                if wrong:
                    sugg = PR.suggested_ionic(procs) + [pr.points[0].element for pr in procs
                                                        if pr.kind == "series" and pr.points[0].element in in_sum]
                    c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
                    c1.warning(
                        "The ionic resistance includes " + ", ".join(
                            f"**{pr.r_name}** ({pr.cls}: C_eff "
                            f"{min(p.C_eff for p in pr.points if p.C_eff):.2g}–{max(p.C_eff for p in pr.points if p.C_eff):.2g} F)"
                            for pr in wrong)
                        + ". An electrode or interphase element is not part of the ionic conduction through the "
                          "sample; including it lowers σ and changes Eₐ.")
                    if sugg and c2.button("Use " + " + ".join(sugg), key="s_use_sugg", width="stretch"):
                        ss["s_rsel_pending"] = (circuit, sugg)
                        st.rerun()
                bad_kk = [r["spectrum"] for r in rows if r["KK: max |res| (%)"] and r["KK: max |res| (%)"] > 3]
                if bad_kk:
                    st.warning(f"Lin-KK residuals exceed 3 % for {', '.join(bad_kk)}. Part of those spectra is not "
                               "consistent with a linear, stable system (drift, noise, a current-range switch); "
                               "see the KK tab in the inspector below for where.")
                bad = [(r["spectrum"], r["notes"]) for r in rows if r["notes"]]
                if bad:
                    with st.expander(f"Notes and warnings ({len(bad)} spectra)"):
                        for n, t in bad:
                            st.markdown(f"**{n}**: {t}")
                if geom0 is None:
                    st.info("Enter thickness and diameter (sidebar or table) to get conductivities.")
                outputs["csv"]["series_spectra_results.csv"] = ex.csv_bytes(df)

                tsel = st.selectbox("Arrhenius quantity", ["σT (S cm⁻¹ K)", "σ (S cm⁻¹)"], index=0, key="s_tsel",
                                    help="ln(σT) vs 1/T follows from the Nernst–Einstein relation and is the usual "
                                         "choice for ionic conductors; Eₐ from σ differs by about kT.")
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

                # ---------------- Arrhenius by process
                if len({p.T_C for pr in procs for p in pr.points}) >= 2:
                    _process_section(procs, proc_notes, g_cc, tsel.startswith("σT"), width, fs, outputs, akey)

                # ---------------- fitted parameters across the series
                fits = [(i, m, results[i]) for i, m in eis_rows if results[i].fit is not None]
                if fits:
                    st.markdown("#### Fitted parameters")
                    prow, qcols = [], []
                    for i, m, r in fits:
                        row = {"spectrum": m.name, "T (°C)": _num(table.at[i, "T (°C)"])}
                        for n, u, v, e, fx in r.fit.all_params():
                            row[n] = v
                            row[f"± {n}"] = None if fx else e
                            if n not in qcols:
                                qcols.append(n)
                        for d in r.fit.derived:
                            for lab, kk in ((f"C_eff {d['pair']} (F)", "C_eff_F"), (f"f_c {d['pair']} (Hz)", "f_c_Hz")):
                                row[lab] = d[kk]
                                if lab not in qcols:
                                    qcols.append(lab)
                        row["fixed"] = ", ".join(n for n, *_, fx in r.fit.all_params() if fx)
                        row["starts at best minimum"] = (f"{r.fit.metrics.get('n_starts_at_best_minimum', 1)}/"
                                                         f"{r.fit.metrics.get('n_starts_converged', 1)}")
                        prow.append(row)
                    dfp = pd.DataFrame(prow)
                    st.dataframe(dfp, hide_index=True, width="stretch",
                                 column_config={k: st.column_config.NumberColumn(format="%.4g") for k in dfp.columns
                                                if k not in ("spectrum", "fixed", "starts at best minimum")})
                    outputs["csv"]["series_fit_parameters.csv"] = ex.csv_bytes(dfp)
                    c1, c2 = st.columns([1, 1.6])
                    q = c1.selectbox("Plot a parameter", qcols, key=f"s_trend_{circuit}",
                                     help="A bulk arc's effective capacitance should change little with "
                                          "temperature; a jump in C_eff or α between neighbouring spectra usually "
                                          "means the fit assigned the arc to a different process there.")
                    Ts = [_num(table.at[i, "T (°C)"]) for i, _, _ in fits]
                    use_T = all(t is not None for t in Ts)
                    xs = Ts if use_T else list(range(1, len(fits) + 1))
                    ys = [row.get(q, np.nan) for row in prow]
                    dys = [row.get(f"± {q}", None) for row in prow]
                    dys = [np.nan if v is None else v for v in dys]
                    unit_q = units[names.index(q)] if q in names else ""
                    fig_t = sp.fig_param_trend(width, xs, ys, dys, "T / °C" if use_T else "spectrum",
                                               f"{q} / {unit_q}" if unit_q and "(" not in q else q,
                                               sp.series_colors(Ts)[0] if use_T else None, None,
                                               logy=not q.endswith("_1"), font_size=fs)
                    with c2:
                        _show(fig_t)
                        _fig_downloads(fig_t, f"series_trend_{q}".replace(" ", "_").replace("(", "").replace(")", ""),
                                       "s_dl_trend")
                    outputs["figures"]["series_parameter_trend"] = fig_t

                # ---------------- DRT peaks across the series
                drts_r = [(i, m, results[i]) for i, m in eis_rows if results[i].drt is not None]
                if drts_r:
                    st.markdown("#### DRT peaks")
                    pk_rows = []
                    for i, m, r in drts_r:
                        d = r.drt
                        for p in d.peaks:
                            pk_rows.append({
                                "spectrum": m.name, "T (°C)": _num(table.at[i, "T (°C)"]), "peak": f"P{p['peak']}",
                                "τ (s)": p["tau_s"], "f = 1/(2πτ) (Hz)": p["f_Hz"], "R (Ω)": p["R_ohm"],
                                "share (%)": p["share_pct"], "C = τ/R (F)": p["C_eq_F"],
                                "λ": d.lam, "R∞ (Ω)": d.R_inf,
                                "flag": _peak_flag(p, d, r.drt_f_cut is not None)})
                    if pk_rows:
                        dpk = pd.DataFrame(pk_rows)
                        st.dataframe(dpk, hide_index=True, width="stretch",
                                     column_config={k: st.column_config.NumberColumn(format="%.3g") for k in dpk.columns
                                                    if k not in ("spectrum", "peak", "flag")})
                        st.caption("A peak at the long-τ end of a spectrum's data (flag *near the low-f cut*) is "
                                   "usually the onset of the blocking tail rather than a process: check it by "
                                   "moving that spectrum's low-frequency limit in the inspector; a real process "
                                   "stays put, the tail artefact follows the cut. Peak R is integrated between "
                                   "adjacent minima of γ, so for overlapping peaks the split is approximate.")
                        outputs["csv"]["series_drt_peaks.csv"] = ex.csv_bytes(dpk)
                    lams = [r.drt.lam for _, _, r in drts_r]
                    if max(lams) / max(min(lams), 1e-300) > 10:
                        st.warning(f"λ differs by a factor {max(lams) / min(lams):.0f} between spectra (automatic "
                                   "selection). DRTs regularised that differently are not directly comparable: "
                                   "peak heights and widths change with λ. Consider one manual λ for the series.")

            # ---------------- figures
            st.markdown("#### Figures")
            c1, c2, c3, c4 = st.columns([2.2, 0.8, 1, 1])
            norm_lbl = c1.selectbox("Normalise impedance", list(S.NORM_MODES), index=0, key="s_norm",
                                    help="Geometry removes thickness/area differences between pellets. Density also "
                                         "scales by ρ_rel^m (σ ≈ σ_dense·ρ_rel^m; m = 1 linear, 1.5 Bruggeman for "
                                         "spherical pores). Shape scales every spectrum by its own resistance.")
            m_exp = c2.number_input("m", value=1.5, min_value=0.0, max_value=4.0, step=0.5, key="s_mexp",
                                    disabled=S.NORM_MODES[norm_lbl] != "dense")
            drt_x = c3.selectbox("DRT x-axis", ["τ (s)", "f = 1/(2πτ) (Hz)"], key="s_drtx")
            show_fit = c4.checkbox("Show fits", value=True, key="s_showfit")
            c1, c3, c2 = st.columns([1, 1, 1.4])
            hide = c1.checkbox("Hide blocking spikes in the Nyquist plot", value=True, key="s_hide")
            fit_arcs = c3.checkbox("Scale the DRT axis to the arcs", value=True, key="s_drt_arcs",
                                   help="Peaks made by the onset of the blocking tail (near or beyond each "
                                        "spectrum's low-frequency cut) are usually far taller than the arcs. With "
                                        "this on they run off the top of the panel instead of flattening the arcs.")
            zoom = c2.number_input("…or show Z′ up to (plotted units; 0 = automatic)", value=0.0, min_value=0.0,
                                   format="%.4g", key="s_zoom")
            mode = S.NORM_MODES[norm_lbl]
            gm = "g" if ss.get("s_drt_gmode", GAMMA_MODES[0]) != GAMMA_MODES[0] else "gamma"
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
                    it["tau_cut"] = 1 / (2 * np.pi * r.drt_f_cut) if r.drt_f_cut else None
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
                                        "tau" if drt_x.startswith("τ") else "f", gm, fit_arcs)
            _show(fig_eis)
            _fig_downloads(fig_eis, "series_impedance", "s_dl_eis")
            cap = []
            if by_T:
                cap.append("Colours run from dark blue (coldest) to orange (hottest).")
            if any(it.get("tau_cut") for it in items):
                cap.append("Each DRT is drawn dotted beyond the lowest frequency it was given.")
            if cap:
                st.caption(" ".join(cap))
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
                                  "f_Hz": 1 / (2 * np.pi * it["tau"]), f"gamma_plotted_{unit_lbl}": it["gamma"],
                                  "beyond_low_f_cut": (it["tau"] > it["tau_cut"]) if it.get("tau_cut") else False})
                    for (i, m), it in zip(eis_rows, items) if it.get("tau") is not None]
            if drts:
                outputs["csv"]["series_drt.csv"] = ex.csv_bytes(pd.concat(drts))

            # ---------------- one spectrum in detail
            if results:
                _inspector(eis_rows, results, table, label_of, method, circuit, r_names, names, units, n_starts,
                           time_budget, f_window, wkey, global_opt, drt_s, cut, theta, gm, fkey, width, fs,
                           zoom if zoom else ("auto" if hide else None), akey_of, ecm_ok)

            outputs["settings"]["impedance"] = {
                "method": method_lbl, "circuit": circuit, "resistances": r_names, "starting_values": start,
                "parameter_table": ptable.to_dict(orient="records") if ptable is not None else None,
                "weighting": wlbl, "n_starts": int(n_starts), "basin_hopping": global_opt,
                "drt": do_drt, "drt_settings": drt_s.__dict__, "drt_tail_cut": cut_lbl, "drt_theta_deg": theta,
                "f_window_Hz": f_window, "arrhenius": ss.get("s_tsel"), "normalisation": norm_lbl, "m": m_exp,
                "individual_settings": {meas[i].name: v for i, v in ov_now.items()}}

    # ------------------------------------------------------------------ DC polarisation
    with tab_dc:
        st.markdown("Steady-state currents of potentiostatic steps on a cell with **ion-blocking electrodes** "
                    "(e.g. Au, stainless steel): once the ionic current has decayed, the remaining current is "
                    "electronic, and the slope of I<sub>ss</sub> against E gives R<sub>e</sub> and "
                    "σ<sub>e</sub> = L/(R<sub>e</sub>·A).", unsafe_allow_html=True)
        if not dc_rows:
            st.info("No DC steps selected.")
        else:
            steps0 = [(i, m, S.steady_state(m, 0.2, _num(table.at[i, "E (V)"]))) for i, m in dc_rows]
            all_pos = all(r.resolved and r.I_ss > 0 for _, _, r in steps0)
            c1, c2, c3, c4 = st.columns(4)
            tail = c1.number_input("Average over the last (% of each step)", 1.0, 100.0, 20.0, step=5.0, key="s_tail",
                                   help="Longer windows average more noise away, but only if the current has "
                                        "stopped changing over them: check the drift and resolved columns.")
            origin = c2.checkbox("Force the line through the origin", value=False, key="s_origin")
            logI = c3.checkbox("Log current axis", value=all_pos, key=f"s_logI_{fkey}",
                               help="Only sensible when every current is clearly positive; noisy currents that "
                                    "cross zero are shown on a linear axis by default.")
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
                              "I": m.I, "window": r.t_window, "I_ss": r.I_ss, "dI": r.dI, "E": r.E,
                              "resolved": r.resolved})
            dfs = pd.DataFrame([{"step": m.name, "file": m.source, "E (V)": r.E, "I_ss (A)": r.I_ss, "± I_ss (A)": r.dI,
                                 "noise (A)": r.noise_A, "resolved": "yes" if r.resolved else "no",
                                 "steady": "no" if r.drifting else "yes",
                                 "drift over window (%)": r.drift_pct,
                                 "window (s)": f"{r.t_window[0]:.4g}–{r.t_window[1]:.4g}",
                                 "notes": " · ".join(m.notes + r.notes)} for i, m, r in sres])
            st.dataframe(dfs, hide_index=True, width="stretch",
                         column_config={**{k: st.column_config.NumberColumn(format="%.3e")
                                           for k in ("I_ss (A)", "± I_ss (A)", "noise (A)")},
                                        **{k: st.column_config.NumberColumn(format="%.3g")
                                           for k in ("E (V)", "drift over window (%)")}})
            n_unres = sum(not r.resolved for _, _, r in sres)
            n_drift = sum(r.drifting for _, _, r in sres)
            if n_unres:
                st.warning(f"{n_unres} of {len(sres)} steady-state currents are not resolved from the noise "
                           "(|I_ss| < 3 × its uncertainty). The fit weights each step by its uncertainty, so σ_e rests "
                           "mainly on the resolved steps; when the slope is not significant only an upper bound is "
                           "given. A lower current range, a shielded cell or longer steps would resolve them.")
            if n_drift:
                st.warning(f"{n_drift} step(s) were still changing over the averaging window. Their current is "
                           "not yet the steady-state electronic current and overestimates σ_e.")
            outputs["csv"]["series_dc_steps.csv"] = ex.csv_bytes(dfs)
            fit, sig_txt = None, None
            try:
                fit = S.dc_fit([r.E for _, _, r in sres], [r.I_ss for _, _, r in sres], origin,
                               [r.dI for _, _, r in sres])
            except ValueError as exc:
                st.warning(str(exc))
            if fit:
                g = geom_of(dc_rows[0][0])
                c1, c2, c3 = st.columns(3)
                ce = None
                if fit["significant"]:
                    ce = cond.conductivity(fit["R_e_ohm"], g, fit["dR_e_ohm"]) if g else None
                    c1.metric("R_e", f"{fit['R_e_ohm']:.4g} Ω", f"± {fit['dR_e_ohm']:.2g} Ω", delta_color="off")
                    if ce:
                        c2.metric("σ_e", f"{ce['sigma_S_cm']:.3e} S/cm", f"± {ce['dsigma_S_cm']:.1e} S/cm",
                                  delta_color="off")
                        sig_txt = rf"$\sigma_e$ = {ce['sigma_S_cm']:.2e} S cm$^{{-1}}$"
                else:
                    Gu = fit["G_upper_S"]
                    c1.metric("R_e", f"> {1 / Gu:.3g} Ω" if Gu > 0 else "not determined",
                              "slope not significant", delta_color="off")
                    if g and Gu > 0:
                        ce_up = Gu * g.thickness_cm / g.area_cm2
                        c2.metric("σ_e (upper bound, ~95 %)", f"< {ce_up:.2e} S/cm", "slope + 2 s.e.", delta_color="off")
                        sig_txt = rf"$\sigma_e$ < {ce_up:.1e} S cm$^{{-1}}$ (slope not significant)"
                        ce = {"sigma_S_cm": ce_up, "upper_bound": True}
                if not g:
                    c2.info("Enter the geometry to get σ_e.")
                if ce and cmp != "–":
                    ii = next(i for i, m in eis_rows if m.name == cmp)
                    ci = cond.conductivity(res_now[ii].R, geom_of(ii), res_now[ii].dR) if geom_of(ii) else None
                    if ci:
                        te = ce["sigma_S_cm"] / (ce["sigma_S_cm"] + ci["sigma_S_cm"])
                        c3.metric("t_e = σ_e/(σ_e+σ_ion)", ("< " if ce.get("upper_bound") else "") + f"{te:.2e}",
                                  f"σ_ion = {ci['sigma_S_cm']:.3e} S/cm", delta_color="off")
                if fit["curved"]:
                    st.warning(f"The I–V points curve (quadratic term at {abs(fit['curvature_t']):.1f} standard errors). "
                               "Ohmic electronic leakage gives a straight line through the origin; a current that sets "
                               "in or grows faster at higher potential points to an electrode reaction or decomposition, "
                               "and a single σ_e from the slope is not meaningful. Fit only the low-potential steps "
                               "(untick the others in the table) or report the I–V curve.")
                st.caption(f"Weighted linear fit of I_ss vs E over {fit['n']} steps"
                           + (" through the origin" if origin else f", intercept {fit['I0_A']:.3g} A")
                           + f" · slope {fit['G_S']:.3g} ± {fit['dG_S']:.2g} S · reduced χ² {fit['chi2_red']:.2g}. "
                           "Each step is weighted by 1/uncertainty². A curving I–V line (e.g. no current at low "
                           "potential, then a rise) means the current is not ohmic electronic leakage: an electrode "
                           "reaction or decomposition contributes, and the linear σ_e is not meaningful.")
                outputs["csv"]["series_dc_fit.csv"] = ex.csv_bytes(pd.DataFrame([{
                    "slope_G_S": fit["G_S"], "dG_S": fit["dG_S"], "significant": fit["significant"],
                    "G_upper_S": fit["G_upper_S"], "R_e_ohm": fit["R_e_ohm"], "dR_e_ohm": fit["dR_e_ohm"],
                    "I0_A": fit["I0_A"], "chi2_red": fit["chi2_red"], "n_steps": fit["n"],
                    "through_origin": fit["through_origin"],
                    "sigma_e_S_cm": ce["sigma_S_cm"] if ce else None,
                    "sigma_e_is_upper_bound": bool(ce and ce.get("upper_bound")),
                    "dsigma_e_S_cm": ce.get("dsigma_S_cm") if ce else None}]))
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

"""Widgets shared by the single-spectrum and series pages: circuit editor block, parameter table,
DRT settings. Each takes a key prefix so both pages keep independent state."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import circuit_tree as ct
from . import drt as drtm
from . import ecm as ecmm
from . import plotting as pl
from .circuit_draw import circuit_figure
from .circuit_editor import circuit_editor


def parse_float(text, name):
    t = str(text).strip().lower().replace(",", ".")
    if t in ("inf", "+inf", "∞"):
        return np.inf
    if t in ("-inf", "-∞"):
        return -np.inf
    try:
        return float(t)
    except ValueError as exc:
        raise ValueError(f"'{text}' for {name} is not a number.") from exc


# --------------------------------------------------------------------------- #
# Circuit: visual editor + text box + undo + diagram downloads
# --------------------------------------------------------------------------- #
def circuit_block(prefix="", default="R0-p(R1,CPE1)-CPE2", presets=None, file_stem="circuit", width=3.5,
                  font_size=8, dpi=600):
    """Render the circuit editor and return the current circuit string ('' when empty).

    One source of truth: session_state[prefix + 'circuit_text']. The visual editor and the text box
    both write to it; changes are applied at the top of the next run (widgets can't be changed
    after they are drawn), and every change is kept in a history for Undo."""
    ss = st.session_state
    k_text, k_hist, k_pend, k_in, k_err = (prefix + s for s in ("circuit_text", "circuit_hist", "circuit_pending",
                                                                  "circuit_input", "circuit_typed_error"))
    ss.setdefault(k_text, default)
    ss.setdefault(k_hist, [])
    if k_pend in ss:
        new_text = ss.pop(k_pend)
        if new_text != ss[k_text]:
            ss[k_hist].append(ss[k_text])
            del ss[k_hist][:-50]
            ss[k_text] = new_text
        ss[k_in] = ss[k_text]
    ss.setdefault(k_in, ss[k_text])

    def _typed():
        try:
            parsed = ct.parse(ss[k_in])
        except ct.CircuitSyntaxError as exc:
            ss[k_err] = str(exc)
            return
        ss.pop(k_err, None)
        ss[k_pend] = ct.to_string(parsed) if parsed else ""

    st.markdown("**Equivalent circuit**: pick a preset, or drag elements onto the canvas. Drop beside an "
                "element to connect it in series, above or below to connect it in parallel.")
    edited = circuit_editor(ss[k_text], key=prefix + "circuit_editor", presets=presets)
    if edited is not None:
        ss[k_pend] = ct.to_string(edited.get("tree"))
        st.rerun()
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    c1.text_input("Circuit string (impedance.py syntax; edit directly if you prefer typing)", key=k_in,
                  on_change=_typed, help="`-` joins elements in series, `p(a,b,…)` puts them in parallel, and "
                  "elements are numbered: R0-p(R1,CPE1)-CPE2. Element definitions: "
                  "https://impedancepy.readthedocs.io/en/latest/circuit-elements.html")
    if c2.button("Undo", disabled=not ss[k_hist], width="stretch", key=prefix + "circuit_undo"):
        ss[k_text] = ss[k_hist].pop()
        ss[k_pend] = ss[k_text]
        st.rerun()
    if ss.get(k_err):
        st.error(f"Circuit string not applied: {ss[k_err]}")
    circuit = ss[k_text]
    if circuit:
        try:
            cfig = circuit_figure(circuit, width_in=min(width, 7.0), font_size=font_size)
            cols = st.columns([2.2, 1, 1, 1, 1])
            cols[0].markdown("**Download the circuit diagram**", help="Drawn like the other figures (fonts, line "
                             "weights, width from the sidebar). Vector formats keep the text editable.")
            for col, fmt in zip(cols[1:], ("png", "pdf", "svg", "tiff")):
                col.download_button(fmt.upper(), (lambda f=fmt: pl.to_bytes(cfig, f, dpi)),
                                    file_name=f"{file_stem}_circuit.{fmt}", mime=pl.MIME[fmt],
                                    key=f"{prefix}circ_dl_{fmt}", width="stretch", on_click="ignore")
        except Exception as exc:  # noqa: BLE001 - drawing must never block fitting
            st.caption(f"Circuit diagram unavailable: {exc}")
    else:
        st.info("The circuit is empty. Choose a preset or drag an element onto the canvas.")
    return circuit


# --------------------------------------------------------------------------- #
# Parameter table (initial / fixed / bounds)
# --------------------------------------------------------------------------- #
def param_table_df(names, units, guess, circuit):
    bd = ecmm.default_bounds(circuit)
    return pd.DataFrame({
        "parameter": names, "unit": units,
        "initial": [f"{v:.4g}" for v in guess],
        "fixed": [False] * len(names),
        "lower": [f"{bd[n][0]:g}" for n in names],
        "upper": ["inf" if not np.isfinite(bd[n][1]) else f"{bd[n][1]:g}" for n in names],
    })


def edit_param_table(df, key):
    return st.data_editor(
        df, key=key, hide_index=True, width="stretch", disabled=["parameter", "unit"],
        column_config={"fixed": st.column_config.CheckboxColumn("fixed"),
                       "initial": st.column_config.TextColumn("initial"),
                       "lower": st.column_config.TextColumn("lower"),
                       "upper": st.column_config.TextColumn("upper")})


def read_param_table(table):
    """(initial, fixed, bounds) dicts from an edited parameter table. Raises ValueError."""
    initial, fixed, bounds = {}, {}, {}
    for _, r in table.iterrows():
        v = parse_float(r["initial"], r["parameter"])
        if bool(r["fixed"]):
            fixed[r["parameter"]] = v
        else:
            initial[r["parameter"]] = v
            bounds[r["parameter"]] = (parse_float(r["lower"], r["parameter"]),
                                      parse_float(r["upper"], r["parameter"]))
    return initial, fixed, bounds


def fit_options(prefix="", n_default=8):
    """Weighting, multi-start count and basin-hopping. Returns (weighting key, n_starts, global_opt)."""
    c1, c2, c3 = st.columns([2, 1.3, 1.2])
    weighting = c1.selectbox("Weighting", list(ecmm.WEIGHTINGS), index=0, key=prefix + "weighting",
                             help="Modulus weighting is the usual choice when measurement variances are "
                                  "unknown; unit weighting over-weights the largest |Z| points.")
    n_starts = c2.number_input("Multi-start fits", 1, 50, n_default, key=prefix + "n_starts",
                               help="Refit from N log-perturbed starting points and keep the best. Also "
                                    "detects non-identifiable parameters.")
    global_opt = c3.checkbox("Basin-hopping", value=False, key=prefix + "basin",
                             help="impedance.py's global optimiser (slow; no weighting, RMSE objective).")
    return ecmm.WEIGHTINGS[weighting], int(n_starts), global_opt, weighting


# --------------------------------------------------------------------------- #
# DRT settings
# --------------------------------------------------------------------------- #
GAMMA_MODES = ["γ(ln τ) vs τ", "g(τ) = γ/τ vs τ"]


def drt_inputs(prefix="", lam_default="GCV", lam_help=None):
    """The DRT settings widgets (place inside a form). Returns (DRTSettings, gamma_mode_label)."""
    lam_keys = list(drtm.LAMBDA_METHODS.values())
    c1, c2, c3 = st.columns(3)
    rbf = c1.selectbox("Discretisation (RBF)", drtm.RBF_TYPES, index=0, key=prefix + "drt_rbf")
    shape_ctl = c1.selectbox("RBF shape control", drtm.SHAPE_CONTROL, index=0, key=prefix + "drt_shape")
    coeff = c1.number_input("FWHM coefficient / shape factor", value=0.5, min_value=1e-3, format="%.3g",
                            key=prefix + "drt_coeff")
    data_used_lbl = c2.selectbox("Data used", list(drtm.DATA_USED), index=0, key=prefix + "drt_data")
    ind_lbl = c2.selectbox("Inductance", list(drtm.INDUCTANCE), index=0, key=prefix + "drt_ind")
    der = c2.selectbox("Regularisation derivative", drtm.DERIVATIVE, index=0, key=prefix + "drt_der")
    lam_lbl = c3.selectbox("λ selection", list(drtm.LAMBDA_METHODS), index=lam_keys.index(lam_default),
                           key=prefix + "drt_lam_method",
                           help=lam_help or "Maradesa et al. (2023) compare these criteria; GCV is the pyDRTtools "
                                            "default.")
    lam = c3.number_input("λ (manual, or initial guess)", value=1e-3, min_value=1e-10, format="%.2e",
                          key=prefix + "drt_lam")
    bayes = c3.checkbox("Bayesian credible band (HMC)", value=False, key=prefix + "drt_bayes",
                        help="Samples the truncated-Gaussian posterior (99 % band). Slower.")
    c4, c5, c6 = st.columns(3)
    n_mc = c4.number_input("HMC samples", 1000, 20000, 2000, step=500, key=prefix + "drt_nmc")
    prom = c5.number_input("Peak prominence (fraction of max γ)", value=0.02, min_value=0.0, max_value=1.0,
                           key=prefix + "drt_prom")
    gmode = c6.selectbox("Plot", GAMMA_MODES, index=0, key=prefix + "drt_gmode")
    s = drtm.DRTSettings(rbf_type=rbf, data_used=drtm.DATA_USED[data_used_lbl], inductance=drtm.INDUCTANCE[ind_lbl],
                         der_used=der, cv_type=drtm.LAMBDA_METHODS[lam_lbl], reg_param=float(lam),
                         shape_control=shape_ctl, coeff=float(coeff), bayesian=bayes, n_mc=int(n_mc),
                         peak_prominence=float(prom))
    return s, gmode

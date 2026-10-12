"""CSV/JSON/ZIP exports. Every bundle carries the settings and library versions used,
so a figure in a paper can be regenerated exactly."""

from __future__ import annotations

import datetime as _dt
import io
import json
import zipfile
from importlib import metadata

import numpy as np
import pandas as pd

from .ecm import relative_residuals


def versions():
    out = {}
    for pkg in ("impedance", "pyDRTtools", "numpy", "scipy", "cvxopt", "matplotlib"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _spectrum_cols(f, Z):
    return {
        "freq_Hz": f,
        "Zre_ohm": Z.real,
        "Zim_ohm": Z.imag,
        "minus_Zim_ohm": -Z.imag,
        "Zmod_ohm": np.abs(Z),
        "phase_deg": np.degrees(np.angle(Z)),
    }


def ecm_spectrum_df(res):
    r_re, r_im = relative_residuals(res.Z, res.Z_fit)
    d = _spectrum_cols(res.freq, res.Z)
    d.update({
        "Zre_fit_ohm": res.Z_fit.real,
        "Zim_fit_ohm": res.Z_fit.imag,
        "Zmod_fit_ohm": np.abs(res.Z_fit),
        "phase_fit_deg": np.degrees(np.angle(res.Z_fit)),
        "res_re_pct_of_Zmod": r_re,
        "res_im_pct_of_Zmod": r_im,
    })
    return pd.DataFrame(d)


def ecm_params_df(res, cond=None):
    rows = []
    for n, u, v, e, fixed in res.all_params():
        rows.append({"parameter": n, "value": v, "std_error": (np.nan if fixed else e),
                     "rel_error_pct": (np.nan if fixed or v == 0 else abs(e / v) * 100),
                     "unit": u, "fixed": fixed})
    for d in res.derived:
        rows.append({"parameter": f"C_eff {d['pair']}", "value": d["C_eff_F"], "unit": "F"})
        rows.append({"parameter": f"f_c {d['pair']}", "value": d["f_c_Hz"], "unit": "Hz"})
    for k, v in res.metrics.items():
        rows.append({"parameter": f"fit: {k}", "value": v})
    if cond:
        rows += _cond_rows(cond)
    return pd.DataFrame(rows)


def _cond_rows(cond):
    return [
        {"parameter": "conductivity: R used", "value": cond["R_ohm"], "std_error": cond["dR_ohm"], "unit": "Ohm"},
        {"parameter": "conductivity: cell constant L/A", "value": cond["cell_constant_per_cm"], "unit": "cm^-1"},
        {"parameter": "conductivity: sigma", "value": cond["sigma_S_cm"], "std_error": cond["dsigma_S_cm"],
         "unit": "S/cm"},
    ]


def drt_spectrum_df(res):
    r_re, r_im = relative_residuals(res.Z, res.Z_fit)
    d = _spectrum_cols(res.freq, res.Z)
    d.update({
        "Zre_DRT_ohm": res.Z_fit.real,
        "Zim_DRT_ohm": res.Z_fit.imag,
        "Zmod_DRT_ohm": np.abs(res.Z_fit),
        "phase_DRT_deg": np.degrees(np.angle(res.Z_fit)),
        "res_re_pct_of_Zmod": r_re,
        "res_im_pct_of_Zmod": r_im,
    })
    return pd.DataFrame(d)


def drt_gamma_df(res):
    d = {"tau_s": res.tau, "f_equiv_Hz": 1 / (2 * np.pi * res.tau), "gamma_ohm": res.gamma,
         "g_tau_ohm_per_s": res.gamma / res.tau}
    if res.gamma_lo is not None:
        d.update({"gamma_lo99_ohm": res.gamma_lo, "gamma_hi99_ohm": res.gamma_hi, "gamma_mean_ohm": res.gamma_mean})
    return pd.DataFrame(d)


def drt_summary_df(res, cond=None):
    rows = [
        {"quantity": "R_inf", "value": res.R_inf, "unit": "Ohm",
         "note": "estimated post hoc" if res.R_inf_estimated else ""},
        {"quantity": "L", "value": res.L, "unit": "H"},
        {"quantity": "lambda", "value": res.lam, "unit": ""},
        {"quantity": "R_pol (integral of gamma d ln tau)", "value": res.R_pol, "unit": "Ohm"},
    ]
    for p in res.peaks:
        for k in ("tau_s", "f_Hz", "R_ohm", "C_eq_F", "gamma_max_ohm"):
            rows.append({"quantity": f"P{p['peak']} {k}", "value": p[k]})
    for k, v in res.metrics.items():
        rows.append({"quantity": f"fit: {k}", "value": v})
    if cond:
        for r in _cond_rows(cond):
            rows.append({"quantity": r["parameter"], "value": r["value"], "unit": r.get("unit", ""),
                         "note": f"± {r['std_error']:.3g}" if r.get("std_error") else ""})
    return pd.DataFrame(rows)


def csv_bytes(df):
    return df.to_csv(index=False, float_format="%.8g").encode()


def settings_json(kind, settings: dict):
    payload = {"analysis": kind, "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               "software": versions(), "settings": settings}
    return json.dumps(payload, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)).encode()


def zip_bundle(files: dict):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()

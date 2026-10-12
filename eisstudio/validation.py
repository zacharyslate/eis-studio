"""Kramers-Kronig consistency check (Lin-KK, Schönleber et al., Electrochim. Acta 131 (2014) 20),
as implemented in impedance.py. Run this before trusting any ECM or DRT result: a spectrum
that is not KK-consistent (drift, non-linearity, non-stationarity) gives fitted parameters
that do not mean anything physical.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
from impedance import validation as _imp_val
from impedance.models.circuits.elements import circuit_elements as _elements


def _eval_linKK_numpy2(elements, ts, f):
    """Drop-in fix for impedance.py 1.7.1 ``eval_linKK``.

    The original builds a Python expression from ``repr`` of NumPy scalars and ``eval``s it.
    Under NumPy >= 2 those reprs read ``np.float64(...)``, which is undefined in the eval
    namespace (NameError). Casting to plain floats restores the intended behaviour.
    """
    el = [float(x) for x in np.ravel(elements)]
    fl = [float(x) for x in np.ravel(f)]
    s = "s([R({},{}),".format([el[0]], fl)
    for Rk, tk in zip(el[1:], np.ravel(ts)):
        s += f"K({[Rk, float(tk)]},{fl}),"
    s += "L({},{}),".format([el[-1]], fl)
    if len(el) == (np.size(ts) + 3):
        s += "C({},{}),".format([1 / el[-2]], fl)
    s = s.strip(",") + "])"
    return eval(s, dict(_elements))  # noqa: S307 - same construction as upstream, numeric only


_imp_val.eval_linKK = _eval_linKK_numpy2
linKK = _imp_val.linKK


def lin_kk(freq, Z, c=0.85, max_M=50, add_cap=None):
    """Run Lin-KK. ``add_cap`` defaults to True when the low-f end is capacitive."""
    if add_cap is None:
        add_cap = bool(np.degrees(np.angle(Z[-3:])).mean() < -45)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        M, mu, Z_kk, res_re, res_im = linKK(freq, Z, c=c, max_M=max_M, fit_type="complex", add_cap=add_cap)
    res_re = np.asarray(res_re) * 100
    res_im = np.asarray(res_im) * 100
    worst = float(np.max(np.abs(np.concatenate([res_re, res_im]))))
    return {
        "M": int(M), "mu": float(mu), "Z_kk": np.asarray(Z_kk), "res_re_pct": res_re, "res_im_pct": res_im,
        "max_abs_res_pct": worst, "mean_abs_res_pct": float(np.mean(np.abs(np.concatenate([res_re, res_im])))),
        "add_cap": add_cap,
    }

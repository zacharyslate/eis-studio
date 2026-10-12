"""Ionic conductivity from a resistance and sample geometry.

sigma = L / (R * A)

with L the thickness between electrodes and A the electrode contact area.
Uncertainties propagate in quadrature (independent errors):

(d sigma / sigma)^2 = (dL/L)^2 + (dR/R)^2 + (dA/A)^2,    dA/A = 2 dd/d for a disc.

Activation energy from an Arrhenius series is out of scope here; export sigma per
temperature and fit ln(sigma T) vs 1/T separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LENGTH_TO_CM = {"mm": 0.1, "cm": 1.0, "µm": 1e-4}
AREA_TO_CM2 = {"cm²": 1.0, "mm²": 0.01}


@dataclass
class Geometry:
    thickness_cm: float
    area_cm2: float
    d_thickness_cm: float = 0.0
    d_area_cm2: float = 0.0
    description: str = ""

    @classmethod
    def disc(cls, thickness, diameter, unit="mm", d_thickness=0.0, d_diameter=0.0):
        k = LENGTH_TO_CM[unit]
        L, d = thickness * k, diameter * k
        A = np.pi * d ** 2 / 4
        dA = A * 2 * (d_diameter * k) / d if d > 0 else 0.0
        return cls(L, A, d_thickness * k, dA,
                   f"disc, L = {thickness:g} {unit}, d = {diameter:g} {unit}")

    @classmethod
    def slab(cls, thickness, area, length_unit="mm", area_unit="cm²", d_thickness=0.0, d_area=0.0):
        k, ka = LENGTH_TO_CM[length_unit], AREA_TO_CM2[area_unit]
        return cls(thickness * k, area * ka, d_thickness * k, d_area * ka,
                   f"L = {thickness:g} {length_unit}, A = {area:g} {area_unit}")

    @property
    def cell_constant_per_cm(self):
        return self.thickness_cm / self.area_cm2


def conductivity(R_ohm, geom: Geometry, dR_ohm=0.0):
    """Return dict with sigma (S/cm), its 1-sigma uncertainty, and the error budget."""
    if R_ohm is None or not np.isfinite(R_ohm) or R_ohm <= 0:
        return None
    sig = geom.thickness_cm / (R_ohm * geom.area_cm2)
    rel = {
        "R": (dR_ohm / R_ohm) if dR_ohm and np.isfinite(dR_ohm) else 0.0,
        "thickness": geom.d_thickness_cm / geom.thickness_cm if geom.thickness_cm else 0.0,
        "area": geom.d_area_cm2 / geom.area_cm2 if geom.area_cm2 else 0.0,
    }
    rel_tot = float(np.sqrt(sum(v ** 2 for v in rel.values())))
    return {
        "R_ohm": float(R_ohm),
        "dR_ohm": float(dR_ohm or 0.0),
        "sigma_S_cm": float(sig),
        "dsigma_S_cm": float(sig * rel_tot),
        "sigma_mS_cm": float(sig * 1e3),
        "dsigma_mS_cm": float(sig * rel_tot * 1e3),
        "rel_err_budget": rel,
        "cell_constant_per_cm": float(geom.cell_constant_per_cm),
    }


def fmt_sigma(c):
    if c is None:
        return "n/a"
    s, ds = c["sigma_S_cm"], c["dsigma_S_cm"]
    if ds > 0:
        return f"{s:.3e} ± {ds:.1e} S cm⁻¹"
    return f"{s:.3e} S cm⁻¹"

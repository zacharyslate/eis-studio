"""Physical models behind the EIS explainer video.

Every system is a series chain of blocks; each block is one physical process with an
impedance.py circuit fragment, parameters, a colour, a zoom animation type and captions.
Spectra are evaluated with eisstudio.ecm.predict (the same code as the app) and the DRT with
eisstudio.drt (pyDRTtools). Parameter values are illustrative, chosen to be typical in
magnitude and to keep the processes visibly separated.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eisstudio import drt as drtm  # noqa: E402
from eisstudio import ecm  # noqa: E402

BLUE, VERM, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#AA4499"
INK, GREY = "#15212B", "#6B7A86"


@dataclass
class Block:
    key: str
    label: str            # short name shown on plots
    fragment: str         # impedance.py fragment, e.g. "p(R1,CPE1)"
    params: list
    color: str
    zoom: str             # zoom animation type
    caption: str          # shown while this process dominates the response
    region: str           # name of the highlight region in the schematic
    elements: list = field(default_factory=list)

    def Z(self, f):
        return ecm.predict(self.fragment, {}, self.params, f)

    @property
    def tau(self):
        """Characteristic time of an RC/CPE or Warburg block (None for R, L)."""
        p = self.params
        if self.fragment.startswith("p(R") and "CPE" in self.fragment:
            return (p[0] * p[1]) ** (1 / p[2])
        if self.fragment.startswith("p(R") and ",C" in self.fragment:
            return p[0] * p[1]
        if self.fragment.startswith("Wo") or self.fragment.startswith("Ws"):
            return p[1]
        return None


@dataclass
class System:
    key: str
    title: str
    subtitle: str
    blocks: list
    fmax: float
    fmin: float
    unit: str                 # "Ω", "mΩ", "Ω cm²"
    scale: float              # multiply Ω by this for display
    drt_fmin: float           # DRT is computed above this frequency (below: blocking/capacitive tail)
    intro: list               # captions shown before the sweep: [(seconds, text), ...]
    summary: str
    ohmic_key: str | None = None
    nyq_view: tuple | None = None   # (xmin, xmax, ymin, ymax) in display units
    drt_lam: float = 1e-3

    @property
    def circuit(self):
        return "-".join(b.fragment for b in self.blocks)

    @property
    def params(self):
        return [p for b in self.blocks for p in b.params]

    def Z(self, f):
        return sum(b.Z(f) for b in self.blocks)

    def active(self, f):
        """Block whose response dominates at frequency f.

        Activity = |Im Z_block| (an RC block peaks at its characteristic frequency, a capacitive
        or diffusion block grows toward low frequency, an inductor toward high frequency). When
        no block has an imaginary part above 6 % of |Z|, the ohmic (purely resistive) block is
        what the signal sees.
        """
        f = np.atleast_1d(np.asarray(f, float))
        fw = np.logspace(np.log10(self.fmax), np.log10(self.fmin), 400)
        # each block's |Im Z| normalised by its own maximum in the window, so a small fast arc
        # counts as "active" at its own frequency even beside a much larger slow one
        acts = np.array([np.abs(b.Z(f).imag) / max(np.abs(b.Z(fw).imag).max(), 1e-30)
                         for b in self.blocks])
        k = np.argmax(acts, axis=0)
        Zt = self.Z(f)
        fcs = [1 / (2 * np.pi * b.tau) for b in self.blocks if b.tau]
        f_fast = max(fcs) if fcs else np.inf
        keys = [b.key for b in self.blocks]
        ind = next((b.key for b in self.blocks if b.fragment.startswith("L")), None)
        out = []
        for i, kk in enumerate(k):
            flat = abs(Zt[i].imag) < 0.05 * abs(Zt[i])
            if ind and Zt[i].imag > 0.05 * abs(Zt[i]):
                out.append(ind)                 # above the real axis: the leads' inductance
            elif self.ohmic_key and flat and f[i] > f_fast:
                out.append(self.ohmic_key)      # high-frequency intercept / axis crossing
            else:
                out.append(keys[kk])
        # a sweep from high to low frequency moves through the blocks in order; never step back
        if len(f) > 1 and f[0] > f[-1]:
            idx = np.maximum.accumulate([keys.index(o) for o in out])
            out = [keys[j] for j in idx]
        return out

    def block(self, key):
        return next(b for b in self.blocks if b.key == key)


def compute_drt(system: System, lam=None):
    lam = lam or system.drt_lam
    f = np.logspace(np.log10(system.fmax), np.log10(max(system.drt_fmin, system.fmin)), 81)
    Z = system.Z(f)
    rng = np.random.default_rng(1)
    Zn = Z + 0.001 * np.abs(Z) * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    has_L = any(b.fragment.startswith("L") for b in system.blocks)
    s = drtm.DRTSettings(cv_type="custom", reg_param=lam, inductance="fit" if has_L else "ignore")
    r = drtm.run_drt(f, Zn, s)
    # assign each sizeable peak to the block with the nearest characteristic time
    taus = {b.key: b.tau for b in system.blocks if b.tau}
    for p in r.peaks:
        p["block"] = min(taus, key=lambda k: abs(np.log10(taus[k] / p["tau_s"]))) if taus else None
    return r


# --------------------------------------------------------------------------- #
# Systems
# --------------------------------------------------------------------------- #
def _q(R, fc, a):
    """CPE Q giving characteristic frequency fc for resistance R and exponent a."""
    tau = 1 / (2 * np.pi * fc)
    return tau ** a / R


INTRO = System(
    key="intro", title="How an impedance spectrum is built", subtitle="one electrode reaction in a liquid electrolyte",
    blocks=[
        Block("ohm", "electrolyte", "R0", [20.0], GREY, "ohmic",
              "At high frequency the interface charges and discharges so fast that its capacitance passes the "
              "current: only the electrolyte resistance R₀ is felt. The point sits on the real axis.",
              "electrolyte", ["R0"]),
        Block("ct", "interface", "p(R1,C1)", [100.0, 1e-5], VERM, "charge_transfer",
              "Slower now: part of the current charges the double layer (C₁), part drives the reaction across "
              "the interface (R₁). The current leads the voltage and the point climbs the arc.",
              "interface", ["R1", "C1"]),
    ],
    fmax=1e5, fmin=0.1, unit="Ω", scale=1.0, drt_fmin=0.1, ohmic_key="ohm",
    intro=[], summary="", nyq_view=(10, 130, -6, None),
)

PELLET = System(
    key="pellet", title="1  A solid electrolyte pellet", subtitle="ion-blocking electrodes on both faces",
    blocks=[
        Block("bulk", "bulk (grains)", "p(R1,C1)", [8000.0, 20e-12], BLUE, "hop",
              "Highest frequencies: Li⁺ hop back and forth between sites inside the grains. The bulk resistance "
              "and the tiny geometric capacitance (pF) give the first, fastest arc.",
              "bulk", ["R1", "C1"]),
        Block("gb", "grain boundaries", "p(R2,CPE2)", [5000.0, _q(5000.0, 2e4, 0.9), 0.9], VERM, "gb",
              "Lower frequencies: the grain boundaries respond. They hinder the ions (R₂) but are very thin, "
              "so their capacitance is larger (nF) and their arc appears at lower frequency.",
              "gb", ["R2", "CPE2"]),
        Block("el", "blocking electrodes", "CPE3", [2e-6, 0.85], GREEN, "electrode_block",
              "Lowest frequencies: the ions reach the electrodes but cannot cross them. They pile up in a "
              "double layer (µF), which behaves like a capacitor: a steep spike, no closing arc.",
              "electrode", ["CPE3"]),
    ],
    fmax=1e7, fmin=20.0, unit="kΩ", scale=1e-3, drt_fmin=1.5e3,
    intro=[(5.0, "A pellet pressed from a solid electrolyte powder sits between two metal electrodes that "
                 "conduct electrons but block ions. We sweep from 10 MHz down to 20 Hz."),
           (5.0, "Here the bulk arc is inside the window: a slower conductor, or a cooled sample. For fast "
                 "sulfides at room temperature it lies above 10 MHz and only its intercept is seen.")],
    summary="Arc widths give the bulk (R₁) and grain-boundary (R₂) resistances; σ = L/(R·A) with R₁ (bulk) or "
            "R₁ + R₂ (total). The spike shows the carriers are ions; the DRT leaves it out.",
    nyq_view=(-0.3, 16.5, -0.6, None),
)

SSB = System(
    key="ssb", title="2  An all-solid-state battery", subtitle="Li metal | solid electrolyte | composite cathode",
    blocks=[
        Block("se", "solid electrolyte", "R0", [25.0], GREY, "hop",
              "Highest frequencies: only ion transport through the solid-electrolyte separator responds. Its "
              "bulk and grain-boundary arcs lie above 1 MHz, so it appears as the intercept R₀.",
              "separator", ["R0"]),
        Block("an", "Li | SE interface", "p(R1,CPE1)", [15.0, _q(15.0, 2e4, 0.85), 0.85], BLUE, "charge_transfer",
              "Tens of kHz: Li⁺ cross the interface between the lithium metal and the solid electrolyte. "
              "Poor contact or a reaction layer makes this arc larger.",
              "anode_if", ["R1", "CPE1"]),
        Block("ca", "cathode interface", "p(R2,CPE2)", [45.0, _q(45.0, 100.0, 0.8), 0.8], VERM, "charge_transfer",
              "Around 100 Hz: charge transfer between the solid electrolyte and the cathode particles, in "
              "parallel with their double layer. In sulfide cells it is often the largest arc.",
              "cathode_if", ["R2", "CPE2"]),
        Block("dif", "diffusion in particles", "Wo3", [40.0, 20.0], GREEN, "diffusion",
              "Below about 1 Hz: Li diffuses into and out of the cathode particles. The slower the signal, "
              "the deeper the concentration wave reaches: the 45° Warburg line, turning vertical once the whole "
              "particle follows the signal.",
              "particle", ["Wo3"]),
    ],
    fmax=1e6, fmin=5e-3, unit="Ω", scale=1.0, drt_fmin=0.02, ohmic_key="se",
    intro=[(5.0, "A full cell: lithium metal anode, a solid-electrolyte separator and a composite cathode of "
                 "active particles mixed with solid electrolyte. We sweep from 1 MHz down to 5 mHz."),
           (4.0, "Each layer and interface responds at its own speed, so the spectrum separates them.")],
    summary="Read off: separator resistance R₀, interface resistances R₁ and R₂, and diffusion. The spectrum "
            "alone does not say which interface is which: symmetric cells, temperature and SOC do.",
    nyq_view=(20, 110, -4, None),
)

PEM = System(
    key="pem", title="3  A PEM water electrolyzer", subtitle="anode (O₂) | proton-exchange membrane | cathode (H₂)",
    blocks=[
        Block("mem", "membrane + contacts", "R0", [0.15], GREY, "proton",
              "Highest frequencies: protons move through the membrane (and electrons through contacts and "
              "plates). Purely resistive: the high-frequency intercept, the ohmic loss.",
              "membrane", ["R0"]),
        Block("her", "H₂ evolution", "p(R1,CPE1)", [0.02, _q(0.02, 5e3, 0.9), 0.9], BLUE, "charge_transfer",
              "A few kHz: the hydrogen evolution reaction on the platinum cathode. It is fast, so its arc is small "
              "and at high frequency.",
              "cathode_cl", ["R1", "CPE1"]),
        Block("oer", "O₂ evolution", "p(R2,CPE2)", [0.25, _q(0.25, 50.0, 0.85), 0.85], VERM, "charge_transfer",
              "Tens of Hz: the oxygen evolution reaction on the iridium-oxide anode. It is much slower, and its "
              "arc dominates the kinetic losses.",
              "anode_cl", ["R2", "CPE2"]),
        Block("mt", "mass transport", "p(R3,CPE3)", [0.06, _q(0.06, 1.0, 0.9), 0.9], PURPLE, "bubbles",
              "Around 1 Hz: water supply and gas removal through the porous layers. Bubbles block part of the "
              "surface; this arc grows at high current.",
              "ptl", ["R3", "CPE3"]),
    ],
    fmax=1e5, fmin=0.1, unit="Ω cm²", scale=1.0, drt_fmin=0.1, ohmic_key="mem", drt_lam=1e-2,
    intro=[(5.0, "Water is split while a DC current flows; a small AC signal rides on top of it. Impedance "
                 "here is measured at the operating point, per square centimetre of cell."),
           (4.0, "We sweep from 100 kHz down to 0.1 Hz.")],
    summary="Read off: the ohmic loss, the kinetic loss of each electrode and the mass-transport loss. Their "
            "sum equals the slope of the polarization curve at this current. Low-frequency assignments are "
            "debated; spectra at several currents help.",
    nyq_view=(0.12, 0.50, -0.02, None),
)

LIB = System(
    key="lib", title="4  A lithium-ion battery", subtitle="graphite | liquid electrolyte + separator | NMC",
    blocks=[
        Block("ind", "cables + windings", "L0", [1.5e-7], INK, "inductance",
              "Highest frequencies: the cable and the cell's wound electrodes act as an inductor. The points "
              "fall below the axis. This is the wiring, not chemistry.",
              "leads", ["L0"]),
        Block("ohm", "electrolyte + contacts", "R0", [0.030], GREY, "ohmic",
              "Where the curve crosses the axis: the ohmic resistance of the liquid electrolyte in the "
              "separator and pores, plus current collectors and contacts.",
              "separator", ["R0"]),
        Block("sei", "SEI film", "p(R1,CPE1)", [0.008, _q(0.008, 1e3, 0.85), 0.85], BLUE, "film",
              "Around 1 kHz: Li⁺ crossing the solid-electrolyte interphase, a thin film on the graphite. It "
              "grows with ageing, so this arc is watched in battery diagnostics.",
              "sei", ["R1", "CPE1"]),
        Block("ct", "charge transfer", "p(R2,CPE2)", [0.015, _q(0.015, 20.0, 0.85), 0.85], VERM, "charge_transfer",
              "Tens of Hz: Li⁺ cross the particle surfaces (charge transfer) while the double layer charges. "
              "It depends strongly on temperature and state of charge.",
              "particle_surf", ["R2", "CPE2"]),
        Block("dif", "solid diffusion", "Wo3", [0.020, 100.0], GREEN, "diffusion",
              "Below about 1 Hz: Li diffuses inside the particles: the 45° line. Far below this sweep the "
              "particles would fill uniformly and the line would turn vertical, like a capacitor.",
              "particle", ["Wo3"]),
    ],
    fmax=1e4, fmin=5e-3, unit="mΩ", scale=1e3, drt_fmin=5e-3, ohmic_key="ohm",
    intro=[(5.0, "A commercial-style cell, measured at rest. The impedance is only tens of milliohms, so "
                 "cables and wiring matter. We sweep from 10 kHz down to 5 mHz."),
           (4.0, "Graphite anode, porous separator soaked in liquid electrolyte, NMC cathode.")],
    summary="Read off: the ohmic resistance, the SEI and charge-transfer arcs (both grow as the cell ages), "
            "and diffusion. Tracking these over a cell's life is the basis of impedance-based diagnostics.",
    nyq_view=(28, 62, -9.6, None),
)

SYSTEMS = [PELLET, SSB, PEM, LIB]


if __name__ == "__main__":
    for s in [INTRO] + SYSTEMS:
        f = np.logspace(np.log10(s.fmax), np.log10(s.fmin), 200)
        Z = s.Z(f)
        Zc = ecm.predict(s.circuit, {}, s.params, f)
        assert np.allclose(Z, Zc), s.key
        act = s.active(f)
        changes = [(f[0], act[0])] + [(f[i], act[i]) for i in range(1, len(f)) if act[i] != act[i - 1]]
        print(f"{s.key:7s} {s.circuit:40s} Re {Z.real.min()*s.scale:.3g}..{Z.real.max()*s.scale:.3g} {s.unit}  "
              f"-Im max {(-Z.imag).max()*s.scale:.3g}")
        print("        active:", ", ".join(f"{k}@{fr:.3g}Hz" for fr, k in changes))
        for b in s.blocks:
            if b.tau:
                print(f"        {b.key}: f_c = {1/(2*np.pi*b.tau):.3g} Hz")

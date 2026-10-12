"""Generate the home-page Nyquist SVG (site_src/_hero_svg.html) from a real fit of the sample spectrum.

    python tools/make_hero.py && python tools/build_site.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eisstudio import ecm, io  # noqa: E402

raw = (ROOT / "eisstudio" / "sample_data" / "synthetic_2arc.csv").read_bytes()
d = io.load_spectrum(raw, "synthetic_2arc.csv")[0][0]
circ = "R0-p(R1,CPE1)-p(R2,CPE2)"
names, _ = ecm.parse_circuit(circ)
r = ecm.fit_ecm(d.freq, d.Z, circ, dict(zip(names, ecm.default_guess(names, d.freq, d.Z, circ))), n_starts=6)

W, padL, padR, padT, padB = 1180, 78, 24, 26, 62
x0, x1 = 20, 790
s = (W - padL - padR) / (x1 - x0)          # px per ohm, identical on both axes
ymax = 215
H = int(padT + padB + ymax * s)
X = lambda z: padL + (z - x0) * s          # noqa: E731
Y = lambda z: padT + (ymax - z) * s        # noqa: E731

pts = "".join(f'<circle cx="{X(z.real):.1f}" cy="{Y(-z.imag):.1f}" r="4.4"/>' for z in d.Z)
fit = "M" + " L".join(f"{X(z.real):.1f},{Y(-z.imag):.1f}" for z in r.Z_smooth)
xt = "".join(f'<line x1="{X(v):.1f}" y1="{Y(0):.1f}" x2="{X(v):.1f}" y2="{Y(0) + 7:.1f}"/>'
             f'<text x="{X(v):.1f}" y="{Y(0) + 26:.1f}" text-anchor="middle">{v}</text>' for v in range(100, 800, 100))
yt = "".join(f'<line x1="{padL - 7}" y1="{Y(v):.1f}" x2="{padL}" y2="{Y(v):.1f}"/>'
             f'<text x="{padL - 13}" y="{Y(v) + 5:.1f}" text-anchor="end">{v}</text>' for v in (0, 100, 200))
labels = {6: ("1 MHz", 13, -4, "start"), 4: ("10 kHz", -10, 24, "end"), 2: ("100 Hz", 12, 4, "start"),
          0: ("1 Hz", 12, -6, "start"), -2: ("10 mHz", 20, -10, "start")}
marks = ""
for k, (lab, dx, dy, anchor) in labels.items():
    i = int(np.argmin(abs(np.log10(d.freq) - k)))
    z = d.Z[i]
    marks += (f'<circle class="dec" cx="{X(z.real):.1f}" cy="{Y(-z.imag):.1f}" r="5"/>'
              f'<text class="declab" x="{X(z.real) + dx:.1f}" y="{Y(-z.imag) + dy:.1f}" text-anchor="{anchor}">{lab}</text>')
taus = sorted(dd["tau_s"] for dd in r.derived)
R1, R2 = r.param("R1")[0], r.param("R2")[0]
arcs = (f'<text class="arc" x="{X(150):.1f}" y="{Y(118):.1f}" text-anchor="middle">R₁ {R1:.0f} Ω, τ {taus[0] * 1e6:.0f} µs</text>'
        f'<text class="arc" x="{X(500):.1f}" y="{Y(196):.1f}" text-anchor="middle">R₂ {R2:.0f} Ω, τ {taus[1] * 1e3:.0f} ms</text>')
svg = f'''<svg class="nyquist" viewBox="0 0 {W} {H}" role="img" aria-labelledby="nyq-title nyq-desc">
  <title id="nyq-title">Nyquist plot of a two-arc impedance spectrum with its equivalent-circuit fit</title>
  <desc id="nyq-desc">Open circles are measured points from 1 MHz to 10 mHz; the line is the fitted circuit. A small arc (R1 about {R1:.0f} ohm, time constant {taus[0] * 1e6:.0f} microseconds) is followed by a larger arc (R2 about {R2:.0f} ohm, {taus[1] * 1e3:.0f} milliseconds).</desc>
  <g class="axes"><line x1="{padL}" y1="{Y(0):.1f}" x2="{W - padR}" y2="{Y(0):.1f}"/><line x1="{padL}" y1="{Y(0):.1f}" x2="{padL}" y2="{padT - 8}"/>{xt}{yt}
  <text class="axlab" x="{(padL + W - padR) / 2:.0f}" y="{Y(0) + 54:.1f}" text-anchor="middle">Z′ / Ω</text>
  <text class="axlab" x="22" y="{(padT + Y(0)) / 2:.0f}" text-anchor="middle" transform="rotate(-90 22 {(padT + Y(0)) / 2:.0f})">−Z″ / Ω</text></g>
  <g class="pts">{pts}</g>
  <path class="fit" d="{fit}" pathLength="1"/>
  <g class="marks">{marks}</g>{arcs}
</svg>'''
(ROOT / "site_src" / "_hero_svg.html").write_text(svg, encoding="utf-8")
print(f"hero SVG {W}x{H}, R1={R1:.1f}, R2={R2:.1f}, taus={taus}")

"""Timeline and parallel renderer for the EIS explainer video.

    python tools/video/render.py OUT.mp4 [--workers 2] [--only intro|pellet|...] [--preview N]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import models as M  # noqa: E402
import scene as S  # noqa: E402

FPS = 24
FADE = 0.45


def read_time(text, extra=1.2):
    return max(5.0, extra + len(text.split()) / 2.7)


INTRO_CAPS = [
    "Impedance spectroscopy applies a small sine-wave voltage V and measures the current I (top right). "
    "Z = V/I has a size |Z| and a phase: how far the current is shifted in time against the voltage.",
    "The measurement is repeated from high to low frequency, one point per frequency. −Z″ against Z′ gives the "
    "Nyquist plot; the phase against frequency gives the Bode plot.",
    "Each process has a time constant τ = RC and responds most at f = 1/(2πτ): its point tops the arc and the "
    "DRT peaks there, with area R. The phase peak sits nearby, shifted by the series resistance.",
]
INTRO_SUMMARY = ("One process, one arc: R₀ is where the arc starts, R₁ is its width, and its top lies at "
                 "f = 1/(2πR₁C₁) ≈ 160 Hz. Real cells have several processes in series, each with its own τ. "
                 "Here are four.")


# ----------------------------------------------------------------------------------- #
# Cards
# ----------------------------------------------------------------------------------- #
def card_title(fig):
    fig.text(0.5, 0.62, "Impedance spectroscopy", fontsize=66, fontweight="bold", ha="center", va="center")
    fig.text(0.5, 0.52, "What each part of a cell does at each frequency", fontsize=32, color=S.MUTED,
             ha="center", va="center")
    fig.text(0.5, 0.36, "a solid-electrolyte pellet  ·  an all-solid-state battery  ·  a PEM water electrolyzer  ·  "
             "a Li-ion battery", fontsize=22, ha="center", va="center", color=S.INK)
    fig.text(0.5, 0.1, "EIS Studio  ·  zacharyslate.github.io/eis-studio", fontsize=18, ha="center", color=S.MUTED)


def card_system(sysm):
    def build(fig):
        num, title = sysm.title.split("  ", 1)
        fig.text(0.5, 0.64, num, fontsize=90, fontweight="bold", ha="center", va="center", color=S.MUTED)
        fig.text(0.5, 0.5, title, fontsize=58, fontweight="bold", ha="center", va="center")
        fig.text(0.5, 0.41, sysm.subtitle, fontsize=28, ha="center", va="center", color=S.MUTED)
        fig.text(0.5, 0.32, f"sweep: {S.fmt_f(sysm.fmax)} → {S.fmt_f(sysm.fmin)}", fontsize=22, ha="center",
                 va="center", color=S.MUTED)
    return build


TAKEAWAYS = [
    "Every process answers in its own frequency window, set by its time constant τ = RC.",
    "One process appears four ways: an arc in the Nyquist plot, a phase peak, a DRT peak, and an RC element "
    "of the circuit.",
    "Bulk, grain boundaries, interfaces, reactions, transport and diffusion separate because their τ differ; "
    "when τ are within about a decade, arcs merge and the DRT helps.",
    "The spectrum does not say which process is which. Assignments need supporting experiments: temperature, "
    "thickness, symmetric cells, state of charge or operating current.",
    "Cables and fixtures add inductance at high frequency; check before reading the high-frequency intercept.",
]

REFS = [
    "A. Lasia, Electrochemical Impedance Spectroscopy and its Applications, Springer, New York, 2014.",
    "M. E. Orazem, B. Tribollet, Electrochemical Impedance Spectroscopy, 2nd ed., Wiley, Hoboken, 2017.",
    "J. T. S. Irvine, D. C. Sinclair, A. R. West, Electroceramics: characterization by impedance spectroscopy, "
    "Adv. Mater. 2 (1990) 132–138.",
    "A. Ch. Lazanas, M. I. Prodromidis, Electrochemical impedance spectroscopy — a tutorial, ACS Meas. Sci. Au 3 "
    "(2023) 162–193.",
    "T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of "
    "relaxation times deconvolution, Electrochim. Acta 184 (2015) 483–499.",
]


def card_takeaways(fig):
    fig.text(0.08, 0.86, "What to take away", fontsize=48, fontweight="bold", va="center")
    y = 0.73
    for t in TAKEAWAYS:
        fig.add_artist(Line2D([0.08], [y], marker="o", ms=10, color=M.VERM, transform=fig.transFigure))
        fig.text(0.1, y + 0.017, S.wrap(t, 92), fontsize=27, va="top", linespacing=1.25)
        y -= 0.07 + 0.05 * (len(S.wrap(t, 92).split("\n")) - 1) + 0.04


def card_refs(fig):
    fig.text(0.08, 0.86, "Further reading", fontsize=48, fontweight="bold", va="center")
    y = 0.75
    for t in REFS:
        fig.text(0.08, y, S.wrap(t, 120), fontsize=21, va="top", linespacing=1.25)
        y -= 0.065 + 0.04 * (len(S.wrap(t, 120).split("\n")) - 1) + 0.02
    fig.text(0.08, 0.24, S.wrap("Spectra in this video are simulated from equivalent circuits with illustrative "
                               "values. They were evaluated with impedance.py, and the DRTs computed with pyDRTtools, "
                               "through EIS Studio. Real spectra are rarely this clean.", 120),
             fontsize=21, va="top", color=S.MUTED, linespacing=1.25)
    fig.text(0.08, 0.07, "EIS Studio: free, open-source ECM and DRT analysis  ·  zacharyslate.github.io/eis-studio",
             fontsize=22, va="center", fontweight="bold")


# ----------------------------------------------------------------------------------- #
# Timeline
# ----------------------------------------------------------------------------------- #
def build_timeline(only=None):
    """List of segments. Each: dict(kind, dur, shot, ...). kind: card | hold | sweep | summary."""
    segs = []
    shot = 0

    def add(**k):
        segs.append(dict(shot=shot, **k))

    def system_segments(sysm, intro_caps, summary, title_card=True):
        nonlocal shot
        if title_card:
            add(kind="card", key=f"card_{sysm.key}", build=card_system(sysm), dur=3.2)
            shot += 1
        bands = band_list(sysm)
        for cap in intro_caps:
            add(kind="hold", sys=sysm.key, band=0, caption=cap, dur=read_time(cap), f=sysm.fmax)
        for i, (key, fa, fb) in enumerate(bands):
            cap = sysm.block(key).caption
            dec = np.log10(fa / fb)
            add(kind="sweep", sys=sysm.key, band=i, caption=cap, fa=fa, fb=fb, dur=max(read_time(cap), 2.2 * dec))
        add(kind="summary", sys=sysm.key, band=0, caption=summary, dur=read_time(summary, 4.0), f=sysm.fmin)
        shot += 1

    if only in (None, "intro"):
        add(kind="card", key="title", build=card_title, dur=5.0)
        shot += 1
        system_segments(M.INTRO, INTRO_CAPS, INTRO_SUMMARY, title_card=False)
    for sysm in M.SYSTEMS:
        if only in (None, sysm.key):
            system_segments(sysm, [c for _, c in sysm.intro], sysm.summary)
    if only in (None, "outro"):
        add(kind="card", key="takeaways", build=card_takeaways, dur=20.0)
        shot += 1
        add(kind="card", key="refs", build=card_refs, dur=12.0)
        shot += 1
    t = 0.0
    for s in segs:
        s["t0"] = t
        s["n0"] = int(round(t * FPS))
        t += s["dur"]
        s["n1"] = int(round(t * FPS))
    # shot extents for fades
    for s in segs:
        same = [x for x in segs if x["shot"] == s["shot"]]
        s["shot_n0"], s["shot_n1"] = same[0]["n0"], same[-1]["n1"]
    return segs


_BANDS = {}


def band_list(sysm):
    if sysm.key not in _BANDS:
        dec = np.log10(sysm.fmax / sysm.fmin)
        fl = np.logspace(np.log10(sysm.fmax), np.log10(sysm.fmin), int(60 * dec) + 1)
        al = sysm.active(fl)
        bands, i0 = [], 0
        for i in range(1, len(fl) + 1):
            if i == len(fl) or al[i] != al[i0]:
                bands.append((al[i0], fl[i0], fl[min(i, len(fl) - 1)]))
                i0 = i
        _BANDS[sysm.key] = bands
    return _BANDS[sysm.key]


SYS = {s.key: s for s in [M.INTRO] + M.SYSTEMS}


# ----------------------------------------------------------------------------------- #
# Rendering
# ----------------------------------------------------------------------------------- #
def render_range(args):
    only, n_from, n_to, out, wid = args
    segs = build_timeline(only)
    scenes, card = {}, S.Card()
    paper = np.array([int(S.PAPER[i:i + 2], 16) for i in (1, 3, 5)], float)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S.W}x{S.H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-tune", "animation",
           "-pix_fmt", "yuv420p", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cur = None
    t_start = time.time()
    for n in range(n_from, n_to):
        seg = next(s for s in segs if s["n0"] <= n < s["n1"])
        t = n / FPS
        if seg["kind"] == "card":
            img = card.render(seg["key"], seg["build"])
        else:
            sysm = SYS[seg["sys"]]
            if sysm.key not in scenes:
                scenes[sysm.key] = S.Scene(sysm)
            sc = scenes[sysm.key]
            if cur is not seg:
                sc.set_state(seg["band"], seg["caption"], summary=seg["kind"] == "summary", hold=seg["kind"] == "hold")
                cur = seg
            if seg["kind"] == "sweep":
                u = (n - seg["n0"]) / max(seg["n1"] - seg["n0"] - 1, 1)
                f = seg["fa"] * (seg["fb"] / seg["fa"]) ** u
            else:
                f = seg["f"]
            img = sc.frame(t, f)
        a = min(1.0, (n - seg["shot_n0"] + 0.5) / (FADE * FPS), (seg["shot_n1"] - n - 0.5) / (FADE * FPS))
        if a < 1.0:
            img = (img.astype(np.float32) * a + paper * (1 - a)).astype(np.uint8)
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
        if (n - n_from) % 240 == 0:
            el = time.time() - t_start
            print(f"[w{wid}] frame {n - n_from}/{n_to - n_from}  {el:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--range", nargs=2, type=float, help="render only this time window (s)")
    a = ap.parse_args()
    segs = build_timeline(a.only)
    N = segs[-1]["n1"]
    print(f"{len(segs)} segments, {N} frames, {N / FPS:.1f} s")
    for s in segs:
        print(f"  {s['t0']:7.1f}s {s['dur']:5.1f}s {s['kind']:8s} {s.get('sys', s.get('key', ''))}"
              f" {s.get('caption', '')[:70]}")
    n_from, n_to = 0, N
    if a.range:
        n_from, n_to = int(a.range[0] * FPS), min(N, int(a.range[1] * FPS))
    out = Path(a.out)
    parts = []
    w = max(1, a.workers)
    bounds = np.linspace(n_from, n_to, w + 1).astype(int)
    jobs = [(a.only, int(bounds[i]), int(bounds[i + 1]), str(out.with_suffix(f".part{i}.mp4")), i) for i in range(w)]
    t0 = time.time()
    with mp.get_context("fork").Pool(w) as pool:
        parts = pool.map(render_range, jobs)
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(out)], check=True)
    for p in parts:
        Path(p).unlink()
    lst.unlink()
    if not a.no_music:
        import music
        wav = out.with_suffix(".music.wav")
        music.write_wav(wav, music.compose((n_to - n_from) / FPS))
        silent = out.with_suffix(".silent.mp4")
        out.rename(silent)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-map", "0:v",
                        "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-20:TP=-2:LRA=7", "-ar", "48000",
                        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)], check=True)
        silent.unlink()
        wav.unlink()
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

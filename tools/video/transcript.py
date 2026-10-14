"""Write the website fragment for the explainer video (player, chapters, transcript) from the same
timeline the renderer uses, so chapter times and caption text always match the video.

    python tools/video/transcript.py            -> site_src/_video_explainer.html
"""
from __future__ import annotations

import html
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import render as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site_src" / "_video_explainer.html"


def mmss(t):
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


def main():
    segs = R.build_timeline()
    total = segs[-1]["t0"] + segs[-1]["dur"]
    names = {"intro": "How a spectrum is built"}
    names.update({s.key: s.title.split("  ", 1)[1] for s in R.M.SYSTEMS})
    chapters, groups, seen = [], {}, set()
    for s in segs:
        key = s.get("sys") or ("intro" if s.get("key") == "title" else s.get("key", "")[5:] if
                                str(s.get("key", "")).startswith("card_") else s.get("key"))
        if key == "takeaways" and key not in seen:
            chapters.append((s["t0"], "What to take away"))
        if key in names and key not in seen:
            chapters.append((s["t0"], names[key]))
        seen.add(key)
        if s.get("caption"):
            groups.setdefault(key, []).append((s["t0"], s["caption"]))
    out = []
    out.append('<figure class="video-block explainer">')
    out.append('  <video id="explainer" controls preload="none" poster="media/eis-explained-poster.jpg" playsinline>')
    out.append('    <source src="media/eis-explained.mp4" type="video/mp4">')
    out.append('    Your browser cannot play this video. <a href="media/eis-explained.mp4">Download it (MP4)</a>.')
    out.append("  </video>")
    out.append(f'  <figcaption>{mmss(total)} min, captions in the picture, quiet background music. '
               'Spectra are simulated from equivalent circuits with illustrative values; the DRTs were computed '
               'with EIS Studio.</figcaption>')
    out.append("</figure>")
    out.append('<ol class="chapters" aria-label="Chapters">')
    for t, name in chapters:
        out.append(f'  <li><button type="button" data-t="{t:.1f}"><span class="ts">{mmss(t)}</span> '
                   f'{html.escape(name)}</button></li>')
    out.append("</ol>")
    out.append('<details class="transcript">')
    out.append("  <summary>Transcript of the captions</summary>")
    for key, items in groups.items():
        out.append(f"  <h3>{html.escape(names.get(key, key))}</h3>")
        for t, cap in items:
            out.append(f'  <p><button type="button" class="ts" data-t="{t:.1f}">{mmss(t)}</button> '
                       f"{html.escape(cap)}</p>")
    out.append("</details>")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(chapters)} chapters, total {mmss(total)})")


if __name__ == "__main__":
    main()

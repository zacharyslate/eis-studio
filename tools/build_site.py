"""Build the static website in docs/ from site_src/ fragments.

    python tools/build_site.py

docs/ is served by GitHub Pages (Settings > Pages > Deploy from branch: main, folder /docs).
The methods page is generated from eisstudio/METHODS.md, the same text the app shows.
"""
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "site_src", ROOT / "docs"
VERSION = re.search(r'__version__ = "([^"]+)"', (ROOT / "eisstudio" / "__init__.py").read_text()).group(1)

NAV = [("install.html", "Install"), ("tutorial.html", "Tutorial"), ("methods.html", "Methods"),
       ("https://github.com/zacharyslate/eis-studio", "GitHub")]
MARK = ('<svg viewBox="0 0 34 22" aria-hidden="true"><path d="M2 20 A7 7 0 0 1 14 20 A10 10 0 0 1 32 20" '
        'fill="none" stroke="#D55E00" stroke-width="2.4" stroke-linecap="round"/>'
        '<line x1="1" y1="20.5" x2="33" y2="20.5" stroke="#15212B" stroke-width="1.4"/></svg>')


def layout(page, title, description, body):
    nav = "".join(f'<a href="{href}"{" aria-current=\"page\"" if href == page else ""}>{label}</a>' for href, label in NAV)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:image" content="https://zacharyslate.github.io/eis-studio/img/out-ecm.png">
<link rel="icon" href="data:image/svg+xml,{MARK.replace('"', "'").replace('#', '%23').replace('aria-hidden=\'true\'', "xmlns='http://www.w3.org/2000/svg'")}">
<link rel="preload" href="fonts/spectral-latin-600-normal.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="fonts/atkinson-hyperlegible-latin-400-normal.woff2" as="font" type="font/woff2" crossorigin>
<link rel="stylesheet" href="style.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site-header"><div class="inner">
  <a class="wordmark" href="index.html">{MARK}EIS Studio</a>
  <nav aria-label="Main">{nav}</nav>
</div></header>
<main id="main">
{body}
</main>
<footer class="site-footer"><div class="inner">
  <span>EIS Studio {VERSION}. Free and open source under the MIT licence.</span>
  <span><a href="https://github.com/zacharyslate/eis-studio">Source code</a> &nbsp; <a href="https://github.com/zacharyslate/eis-studio/issues">Report a problem</a></span>
</div></footer>
<script src="site.js" defer></script>
</body>
</html>
"""


def meta(text, key):
    m = re.search(rf"<!--\s*{key}:\s*(.*?)\s*-->", text)
    return m.group(1) if m else ""


def main():
    hero = (SRC / "_hero_svg.html").read_text(encoding="utf-8")
    for src in sorted(p for p in SRC.glob("*.html") if not p.name.startswith("_")):
        text = src.read_text(encoding="utf-8")
        body = re.sub(r"<!--\s*(title|description):.*?-->\n?", "", text)
        body = body.replace("{{HERO_SVG}}", hero).replace("{{VERSION}}", VERSION)
        (OUT / src.name).write_text(layout(src.name, meta(text, "title"), meta(text, "description"), body), encoding="utf-8")
        print("built", src.name)
    md = (ROOT / "eisstudio" / "METHODS.md").read_text(encoding="utf-8")
    html = markdown.markdown(md, extensions=["tables"])
    body = ('<div class="page-head"><h1>Methods and references</h1><p>The equations, conventions and '
            'sources behind every number EIS Studio reports. The app shows the same text in its '
            '“Methods &amp; references” tab.</p></div>\n'
            f'<section class="section methods prose" style="padding-top:1rem">{html}</section>')
    (OUT / "methods.html").write_text(layout("methods.html", "Methods and references: EIS Studio",
                                             "Equations, conventions and references used by EIS Studio for "
                                             "equivalent-circuit fitting, DRT, Kramers-Kronig validation and conductivity.",
                                             body), encoding="utf-8")
    print("built methods.html")


if __name__ == "__main__":
    main()

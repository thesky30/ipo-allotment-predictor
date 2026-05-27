"""Convert a Markdown report into a Reveal.js HTML slide deck (offline).

Slide structure:
  - The text before the first H2 (`## `) becomes the title / cover slide.
  - Each H2 section becomes a horizontal slide.
  - Each H3 (`### `) inside an H2 section becomes a vertical sub-slide.
  - Standalone `---` horizontal rules in the source are treated as section
    dividers and dropped (Reveal uses its own slide boundaries).

Assets are loaded from the local ./reveal folder (downloaded once), so the
resulting .html works fully offline. Navigate with arrow keys; press 'S' for
speaker notes, 'F' for fullscreen, 'O' for slide overview, 'E' then print for PDF.

Usage:
    python scripts/md_to_revealjs.py --in 项目汇报_新股网下中签率预测.md
    python scripts/md_to_revealjs.py --in <file.md> --out <deck.html> \
        --title "标题" --theme white --reveal reveal
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import markdown

MD_EXT = ["tables", "fenced_code", "sane_lists"]

CUSTOM_CSS = """
.reveal { font-family:"PingFang SC","Microsoft YaHei","Hiragino Sans GB",
          -apple-system,"Segoe UI",sans-serif; font-size:30px; }
.reveal .slides { text-align:left; }
.reveal h1 { font-size:1.7em; }
.reveal h2 { font-size:1.25em; color:#1b3a6b; border-bottom:3px solid #2f6df6;
             padding-bottom:.2em; margin-bottom:.5em; text-align:left; }
.reveal h3 { font-size:1.05em; color:#2f6df6; text-align:left; }
.reveal h4 { font-size:.95em; color:#555; text-align:left; }
.reveal p, .reveal li { font-size:.66em; line-height:1.5; }
.reveal ul, .reveal ol { display:block; margin-left:1em; }
.reveal li { margin:.18em 0; }
.reveal strong { color:#0b1f3a; }
.reveal blockquote { width:100%; font-size:.62em; background:#eaf1ff;
                     border-left:4px solid #2f6df6; box-shadow:none;
                     padding:.5em .9em; }
.reveal blockquote p { font-size:1em; }
.reveal table { font-size:.5em; margin:.4em 0; }
.reveal table th, .reveal table td { padding:6px 10px; border:1px solid #d0d4dc; }
.reveal table th { background:#f2f5fb; }
.reveal pre { width:100%; box-shadow:none; font-size:.46em; }
.reveal pre code { padding:14px; border-radius:6px; max-height:none; }
.reveal code { font-size:.95em; }
.reveal section { overflow-y:auto; max-height:96vh; }
.reveal .cover { text-align:center; }
.reveal .cover h1 { color:#1b3a6b; border:none; margin-bottom:.4em; }
.reveal .cover p { font-size:.7em; text-align:center; color:#444; }
.reveal .slide-number { font-size:14px; }
"""

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, minimum-scale=1, user-scalable=no">
<title>{title}</title>
<link rel="stylesheet" href="{reveal}/dist/reveal.css">
<link rel="stylesheet" href="{reveal}/dist/theme/{theme}.css" id="theme">
<link rel="stylesheet" href="{reveal}/plugin/highlight/monokai.css">
<style>{css}</style>
</head>
<body>
<div class="reveal">
  <div class="slides">
{slides}
  </div>
</div>
<script src="{reveal}/dist/reveal.js"></script>
<script src="{reveal}/plugin/highlight/highlight.js"></script>
<script>
  Reveal.initialize({{
    hash: true,
    center: false,
    transition: 'slide',
    width: 1280,
    height: 720,
    margin: 0.04,
    slideNumber: 'c/t',
    plugins: [ RevealHighlight ]
  }});
</script>
</body>
</html>
"""


def _split_sections(body: str) -> list[str]:
    """Split markdown on top-level H2 headings, keeping the heading."""
    parts = re.split(r"(?m)^(?=##\s)", body)
    return [p for p in parts if p.strip()]


def _split_subsections(section: str) -> list[str]:
    """Split an H2 block on H3 headings (content before first H3 stays first)."""
    parts = re.split(r"(?m)^(?=###\s)", section)
    return [p for p in parts if p.strip()]


def _strip_hr(text: str) -> str:
    return re.sub(r"(?m)^\s*---\s*$", "", text)


def _to_html(md_text: str) -> str:
    return markdown.markdown(_strip_hr(md_text), extensions=MD_EXT, output_format="html5")


def build_slides(md_text: str) -> str:
    # Cover = everything before the first H2.
    m = re.search(r"(?m)^##\s", md_text)
    cover_md = md_text[: m.start()] if m else md_text
    body_md = md_text[m.start():] if m else ""

    out: list[str] = []
    cover_html = _to_html(cover_md)
    out.append(f'    <section class="cover">\n{cover_html}\n    </section>')

    for section in _split_sections(body_md):
        subs = _split_subsections(section)
        if len(subs) <= 1:
            out.append(f"    <section>\n{_to_html(section)}\n    </section>")
        else:
            inner = "\n".join(
                f"      <section>\n{_to_html(s)}\n      </section>" for s in subs
            )
            out.append(f"    <section>\n{inner}\n    </section>")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Markdown → Reveal.js slide deck (offline)")
    p.add_argument("--in", dest="inp", required=True, help="input .md file")
    p.add_argument("--out", default=None, help="output .html (default: <stem>_slides.html)")
    p.add_argument("--title", default=None, help="page <title>")
    p.add_argument("--theme", default="white", help="reveal theme name (default: white)")
    p.add_argument("--reveal", default="reveal", help="path to local reveal asset folder")
    args = p.parse_args()

    src = Path(args.inp)
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    reveal_dir = Path(args.reveal)
    if not (reveal_dir / "dist" / "reveal.js").exists():
        print(f"WARNING: {reveal_dir}/dist/reveal.js not found — the deck will "
              f"not render until reveal.js assets are present.", file=sys.stderr)

    text = src.read_text(encoding="utf-8")
    slides = build_slides(text)
    title = args.title or src.stem
    html = PAGE.format(title=title, css=CUSTOM_CSS, slides=slides,
                       theme=args.theme, reveal=args.reveal.replace("\\", "/"))

    out = Path(args.out) if args.out else src.with_name(src.stem + "_slides.html")
    out.write_text(html, encoding="utf-8")
    n_h = slides.count("<section")
    print(f"Saved → {out}  ({n_h} <section> blocks)")


if __name__ == "__main__":
    main()

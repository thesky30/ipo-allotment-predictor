"""Convert a Markdown file to a self-contained, styled HTML page.

Produces an offline HTML (inline CSS, no external resources) that renders well
in any browser — suitable for sharing / presenting the project reports.

Usage:
    python scripts/md_to_html.py --in 项目汇报_新股网下中签率预测.md
    python scripts/md_to_html.py --in <file.md> --out <file.html> --title "标题"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import markdown

CSS = """
:root { --fg:#1f2329; --muted:#646a73; --line:#e5e6eb; --accent:#2f6df6;
        --accent-soft:#eaf1ff; --code-bg:#f6f8fa; --th-bg:#f2f5fb; }
* { box-sizing: border-box; }
body { margin:0; background:#f5f6f8; color:var(--fg);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
       "Microsoft YaHei","Hiragino Sans GB",sans-serif; line-height:1.75;
       font-size:16px; }
.wrap { max-width:900px; margin:32px auto; background:#fff; padding:56px 64px;
        border-radius:12px; box-shadow:0 1px 4px rgba(0,0,0,.06); }
h1,h2,h3,h4 { line-height:1.35; font-weight:700; margin:1.6em 0 .6em; }
h1 { font-size:30px; margin-top:0; padding-bottom:.4em; border-bottom:2px solid var(--line); }
h2 { font-size:23px; padding-bottom:.3em; border-bottom:1px solid var(--line); }
h3 { font-size:19px; } h4 { font-size:17px; color:var(--muted); }
p { margin:.7em 0; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
hr { border:none; border-top:1px solid var(--line); margin:2em 0; }
ul,ol { padding-left:1.5em; } li { margin:.3em 0; }
blockquote { margin:1em 0; padding:.6em 1.1em; color:var(--muted);
             background:var(--accent-soft); border-left:4px solid var(--accent);
             border-radius:0 6px 6px 0; }
blockquote p { margin:.3em 0; }
code { background:var(--code-bg); padding:.15em .4em; border-radius:4px;
       font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;
       font-size:.88em; color:#c7254e; }
pre { background:var(--code-bg); padding:16px 18px; border-radius:8px;
      overflow:auto; border:1px solid var(--line); }
pre code { background:none; padding:0; color:#24292e; }
table { border-collapse:collapse; width:100%; margin:1.2em 0; font-size:14.5px;
        display:block; overflow-x:auto; }
th,td { border:1px solid var(--line); padding:9px 13px; text-align:left;
        vertical-align:top; }
th { background:var(--th-bg); font-weight:600; white-space:nowrap; }
tr:nth-child(even) td { background:#fafbfc; }
strong { color:#0b1f3a; }
.meta { color:var(--muted); font-size:13px; margin-top:40px; padding-top:16px;
        border-top:1px solid var(--line); }
@media (max-width:680px){ .wrap{ padding:28px 20px; margin:0; border-radius:0; } }
@media print { body{ background:#fff; } .wrap{ box-shadow:none; max-width:none;
        margin:0; padding:0; } }
"""

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
{body}
<div class="meta">由 {src} 自动生成 · Markdown → HTML</div>
</div>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Markdown → self-contained styled HTML")
    p.add_argument("--in", dest="inp", required=True, help="input .md file")
    p.add_argument("--out", default=None, help="output .html (default: same name)")
    p.add_argument("--title", default=None, help="page <title> (default: file stem)")
    args = p.parse_args()

    src = Path(args.inp)
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    text = src.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "toc", "nl2br"],
        output_format="html5",
    )
    title = args.title or src.stem
    html = PAGE.format(title=title, css=CSS, body=body, src=src.name)

    out = Path(args.out) if args.out else src.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

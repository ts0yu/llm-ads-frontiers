"""Inline a walkthrough page's local ES modules into one self-contained HTML file.

Used to publish the Atlas Playground and Atlas Explorer as single-file pages.
Local `import ... from './x.mjs'` lines are dropped after the module bodies are
inlined in dependency order, and `export` keywords are stripped.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

IMPORT = re.compile(r"^import\s+[^;]*?from\s+'\./([\w.-]+\.mjs)';\s*$", re.M)


def module_order(web: Path, source: str, seen: list[str]) -> list[str]:
    for name in IMPORT.findall(source):
        if name not in seen:
            module_order(web, (web / name).read_text(), seen)
            seen.append(name)
    return seen


def inline(page: Path, extra_head: str = "") -> str:
    web = page.parent
    html = page.read_text()
    names = module_order(web, html, [])
    bodies = []
    for name in names:
        text = IMPORT.sub("", (web / name).read_text())
        bodies.append(f"// ---- {name} ----\n" + re.sub(r"^export\s+", "", text, flags=re.M))
    html = IMPORT.sub("", html)
    html = html.replace('<script type="module">', '<script type="module">\n' + "\n".join(bodies), 1)
    # The artifact skeleton supplies doctype/html/head/body; keep title, style and content.
    title = re.search(r"<title>.*?</title>", html, re.S).group(0)
    styles = "\n".join(re.findall(r"<style>.*?</style>", html, re.S))
    body = re.search(r"<body[^>]*>(.*)</body>", html, re.S).group(1)
    meta = '<meta name="claude:viewport-fit-sides" content="cover">'
    return f"{title}\n{meta}\n{extra_head}{styles}\n{body}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(inline(args.page))
    print(args.output, args.output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

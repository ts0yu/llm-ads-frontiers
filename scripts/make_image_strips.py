"""Render a lambda film strip for each image bid atlas (docs figures).

    python scripts/make_image_strips.py OUT_DIR ATLAS.atlas.json [...]

Each strip shows the leaf that owns lambda = 0.02, 0.2, 0.4, 0.6, 0.8, 0.98, with the
leaf id and the CLIP zero-shot top label (a noisy heuristic, not ground truth).
Images are the stored leaf images; nothing is regenerated or retouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LAMBDAS = (0.02, 0.2, 0.4, 0.6, 0.8, 0.98)


def strip(atlas_path: Path, out: Path, side: int = 256) -> None:
    atlas = json.loads(atlas_path.read_text())
    canvas = Image.new("RGB", (side * len(LAMBDAS), side + 44), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for k, lam in enumerate(LAMBDAS):
        leaf = next(leaf for leaf in atlas["leaves"] if leaf["lo"] < lam <= leaf["hi"])
        image = Image.open(atlas_path.parent / leaf["image"]).convert("RGB").resize((side, side))
        canvas.paste(image, (k * side, 0))
        label = leaf.get("outcomes", {}).get("clip_argmax", "")
        caption = f"λ={lam:.2f}  leaf {leaf['id']}"
        draw.text((k * side + 6, side + 4), caption, fill="black", font=font)
        draw.text((k * side + 6, side + 23), f"CLIP: {label}", fill="#555", font=font)
    canvas.save(out, quality=82)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    out_dir = Path(argv[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw in argv[1:]:
        path = Path(raw)
        out = out_dir / (path.name.replace(".atlas.json", "") + "-strip.jpg")
        strip(path, out)
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

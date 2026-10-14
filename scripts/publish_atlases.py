"""Publish bid-atlas run outputs into the Atlas Explorer.

    python scripts/publish_atlases.py RUN_DIR [RUN_DIR ...] [--max-image-side 512]

Each RUN_DIR is an --output directory of ``python -m token_auction_lab.atlas``,
``token_auction_lab.simplex_atlas`` or ``token_auction_lab.diffusion_atlas`` (or a
single atlas JSON file). Every ``bid-atlas-v1`` / ``simplex-atlas-v1`` JSON found
under it is copied to ``walkthrough/atlas_data/<run-name>/<same relative path>``
together with the images it references, and ``walkthrough/atlas_data/index.json``
(schema ``atlas-index-v1``) is rewritten. Entries of other runs, including the
synthetic fixtures, are kept; republishing a run name replaces only that run.

What is changed, and recorded in each published file's ``published`` block:

* display fields the explorer reads, only when the generator did not record them
  (``title``, ``manifest.user_prompt``, split token text);
* ``simplex-atlas-v1`` ``cells`` is renamed ``leaves`` (MOONSHOTS.md section 4) and
  split ``pieces`` are filled from ``children``/``winners``/``tokens``;
* images may be downscaled (``--max-image-side``); the original size is recorded.

What is never changed: ``mode`` (MOONSHOTS.md section 5 labels are copied verbatim;
an unknown mode is refused), bidders, seeds, leaf intervals or polygons, token IDs,
text, per-step records, outcomes, verification records and the run manifest.
Nothing here runs a model or touches real credits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `python scripts/publish_atlases.py` from any directory
    sys.path.insert(0, str(ROOT))
DEST = ROOT / "walkthrough" / "atlas_data"
INDEX_SCHEMA = "atlas-index-v1"
SCHEMAS = {"bid-atlas-v1", "simplex-atlas-v1"}
MODES = {"live_model", "synthetic_model_test", "synthetic_fixture"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MARKER = ".published-run.json"  # dotfiles are never served by the walkthrough
NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
REL_RE = re.compile(r"^[\w][\w.\-/]*$")


class PublishError(ValueError):
    pass


class IncompleteAtlas(PublishError):
    """A time-budget-truncated image atlas: skipped unless --allow-incomplete."""


# --------------------------------------------------------------------------- discovery
def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def find_atlases(run: Path) -> list[tuple[Path, dict]]:
    """Atlas JSON files under a run directory (or the file itself), sorted by path."""
    candidates = [run] if run.is_file() else sorted(run.rglob("*.json"))
    found = []
    for path in candidates:
        rel_parts = path.relative_to(run).parts if run.is_dir() else (path.name,)
        if path.name == "index.json" or any(p.startswith(".") for p in rel_parts):
            continue
        data = load_json(path)
        if isinstance(data, dict) and data.get("schema") in SCHEMAS:
            found.append((path, data))
    return found


def run_name(run: Path, prefix: str = "") -> str:
    base = run.stem if run.is_file() else run.resolve().name
    name = NAME_RE.sub("-", f"{prefix}{base}").strip("-.")
    if not name or name.startswith("fixture-"):
        raise PublishError(f"Unusable run name {name!r}; pass --name.")
    return name


def safe_relative(value: str, what: str) -> str:
    if not isinstance(value, str) or not REL_RE.match(value) or ".." in Path(value).parts:
        raise PublishError(f"Unsafe {what} path {value!r}.")
    return value


# ---------------------------------------------------------------------- display fields
def scenario_display(kind: str, schema: str, manifest: dict) -> tuple[str | None, str | None]:
    """(title, user prompt) from the repo's scenario files, for runs that predate them."""
    name = manifest.get("scenario")
    try:
        if schema == "simplex-atlas-v1":
            from token_auction_lab.prompts_n import scenarios_n

            scenario = manifest.get("scenario_definition") or scenarios_n().get(name) or {}
            return scenario.get("title"), scenario.get("task", scenario.get("prompt"))
        if kind == "image":
            from token_auction_lab.diffusion_atlas import load_scenes

            prompts = manifest.get("prompts") or {}
            scene = load_scenes()["scenes"].get(prompts.get("scene"), {})
            return scene.get("title", prompts.get("scene")), None
        if name == "live-defaults":
            inputs = manifest.get("live_inputs_without_bids") or {}
            return "Live Lab defaults", inputs.get("prompt")
        from token_auction_lab.prompts import scenarios

        scenario = scenarios().get(name) or {}
        return scenario.get("title"), scenario.get("task")
    except Exception:  # noqa: BLE001 - display fallback only
        return None, None


class TokenizerSource:
    """Loads the run's tokenizer only if a split lacks token text (never downloads)."""

    def __init__(self, explicit: str | None, auto: bool):
        self.explicit, self.auto, self.cache = explicit, auto, {}

    def get(self, manifest: dict):
        name = self.explicit or (manifest.get("tokenizer") if self.auto else None)
        if not name:
            return None
        revision = None if self.explicit else manifest.get("tokenizer_revision")
        key = (name, revision)
        if key not in self.cache:
            try:
                from transformers import AutoTokenizer

                self.cache[key] = AutoTokenizer.from_pretrained(
                    name, revision=revision, local_files_only=True)
            except Exception:  # noqa: BLE001 - token text is optional display data
                self.cache[key] = None
        return self.cache[key]


def piece(tokenizer, prefix: list[int], token_id: int) -> str:
    from token_auction_lab.token_text import token_piece

    return token_piece(tokenizer, prefix, token_id)


# ------------------------------------------------------------------------ validation
def check_pair(atlas: dict, allow_incomplete: bool) -> None:
    leaves = atlas["leaves"]
    if atlas.get("complete") is False and not allow_incomplete:
        raise IncompleteAtlas("Atlas is incomplete (time budget); pass --allow-incomplete "
                              "to publish it with the explorer's partition warning.")
    if not leaves:
        raise PublishError("Atlas has no leaves.")
    if not allow_incomplete:
        if leaves[0]["lo"] != 0.0 or leaves[-1]["hi"] != 1.0:
            raise PublishError("Leaves do not cover (0, 1].")
        for left, right in zip(leaves, leaves[1:]):
            if left["hi"] != right["lo"]:
                raise PublishError("Leaves are not contiguous.")
    if len(atlas.get("bidders", [])) != 2:
        raise PublishError("bid-atlas-v1 needs exactly two bidders.")


def check_simplex(atlas: dict) -> None:
    if len(atlas.get("bidders", [])) != 3:
        raise PublishError("The explorer shows simplex atlases for exactly three bidders.")
    area = 0.0
    for leaf in atlas["leaves"]:
        poly = leaf["polygon"]
        if any(len(p) != 3 for p in poly) or len(poly) < 3:
            raise PublishError(f"Cell {leaf['id']} is not a barycentric polygon.")
        area += abs(math.fsum(p[0] * q[1] - q[0] * p[1]
                              for p, q in zip(poly, poly[1:] + poly[:1]))) / 2
    if abs(area - 0.5) > 1e-9:
        raise PublishError(f"Cell areas sum to {area}, not 0.5.")


# ------------------------------------------------------------------------- normalize
def normalize(atlas: dict, tokenizers: TokenizerSource, allow_incomplete: bool) -> list[str]:
    """Add explorer display fields in place. Returns notes on every change made."""
    notes: list[str] = []
    schema, mode = atlas.get("schema"), atlas.get("mode")
    if schema not in SCHEMAS:
        raise PublishError(f"Unsupported schema {schema!r}.")
    if mode not in MODES:
        raise PublishError(f"Unknown mode {mode!r}; refusing to guess a label.")
    manifest = atlas.setdefault("manifest", {})
    kind = atlas.get("kind", "text")
    if schema == "simplex-atlas-v1":
        if "leaves" not in atlas and "cells" in atlas:
            atlas["leaves"] = atlas.pop("cells")
            notes.append("cells renamed to leaves (MOONSHOTS.md section 4)")
        check_simplex(atlas)
    else:
        atlas["leaves"] = sorted(atlas["leaves"], key=lambda leaf: leaf["lo"])
        check_pair(atlas, allow_incomplete)

    title, user_prompt = None, None
    if not atlas.get("title") or (kind == "text" and not manifest.get("user_prompt")
                                  and isinstance(manifest.get("prompts"), list)):
        title, user_prompt = scenario_display(kind, schema, manifest)
    if not atlas.get("title"):
        atlas["title"] = title or f"{kind} atlas"
        notes.append("title derived from the scenario files" if title else "title defaulted")
    if kind == "text" and isinstance(manifest.get("prompts"), list) \
            and not manifest.get("user_prompt") and user_prompt:
        manifest["user_prompt"] = user_prompt
        notes.append("manifest.user_prompt derived from the scenario files")

    if kind == "text" and schema == "bid-atlas-v1":
        by_id = {leaf["id"]: leaf for leaf in atlas["leaves"]}
        missing = [s for s in atlas["splits"]
                   if "a_token_text" not in s or "b_token_text" not in s]
        tokenizer = tokenizers.get(manifest) if missing else None
        for split in missing:
            if tokenizer is None:
                break
            prefix = by_id[split["parent"]]["token_ids"][: split["step"]]
            split["a_token_text"] = piece(tokenizer, prefix, split["a_token"])
            split["b_token_text"] = piece(tokenizer, prefix, split["b_token"])
        if missing:
            notes.append("split token text decoded with the run's tokenizer" if tokenizer
                         else "split token text unavailable (no tokenizer); explorer shows IDs")
    if schema == "simplex-atlas-v1":
        by_id = {leaf["id"]: leaf for leaf in atlas["leaves"]}
        filled = False
        for split in atlas["splits"]:
            if "pieces" in split:
                continue
            filled = True
            tokenizer = tokenizers.get(manifest)
            prefix = by_id[split["parent"]]["token_ids"][: split["step"]]
            split["pieces"] = [
                {"cell": c, "winner": w, "token": t,
                 "token_text": piece(tokenizer, prefix, t) if tokenizer else None}
                for c, w, t in zip(split["children"], split["winners"], split["tokens"])
            ]
            if "E" not in split:
                step = by_id[split["parent"]]["steps"][split["step"]]
                split["E"] = step.get("E")
        if filled:
            notes.append("split pieces filled from children/winners/tokens")
    return notes


# ---------------------------------------------------------------------------- images
def copy_image(src: Path, dst: Path, max_side: int | None, quality: int) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not max_side:
        shutil.copyfile(src, dst)
        return {}
    from PIL import Image

    with Image.open(src) as image:
        original = list(image.size)
        if max(image.size) <= max_side:
            shutil.copyfile(src, dst)
            return {}
        image = image.convert("RGB") if dst.suffix.lower() in {".jpg", ".jpeg"} else image
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        options = {"quality": quality} if dst.suffix.lower() in {".jpg", ".jpeg", ".webp"} \
            else {"optimize": True}
        image.save(dst, **options)
        return {"original_size": original, "published_size": list(image.size)}


# --------------------------------------------------------------------------- publish
def entry_for(atlas: dict, path: str, run: str, source: str) -> dict:
    manifest = atlas.get("manifest", {})
    seed = atlas.get("seed")
    title = atlas["title"] if seed is None else f"{atlas['title']} · seed {seed}"
    return {
        "id": path.removesuffix(".json"), "title": title, "schema": atlas["schema"],
        "kind": atlas.get("kind", "text"), "mode": atlas["mode"], "path": path,
        "bidders": atlas["bidders"], "leaves": len(atlas["leaves"]), "seed": seed,
        "model": manifest.get("model_id") or manifest.get("model"),
        "run": run, "source": source,
    }


def publish_run(run: Path, *, dest: Path, name: str, tokenizers: TokenizerSource,
                max_image_side: int | None = None, jpeg_quality: int = 85,
                allow_incomplete: bool = False, now: str | None = None) -> list[dict]:
    atlases = find_atlases(run)
    if not atlases:
        raise PublishError(f"No bid-atlas-v1 or simplex-atlas-v1 JSON under {run}.")
    root = run.parent if run.is_file() else run
    target = dest / name
    staged = dest / f".staging-{name}"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    now = now or datetime.now(timezone.utc).isoformat()
    entries = []
    try:
        for path, atlas in atlases:
            rel = path.relative_to(root).as_posix()
            source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                notes = normalize(atlas, tokenizers, allow_incomplete)
            except IncompleteAtlas as error:
                print(f"skip {rel}: {error}", file=sys.stderr)
                continue
            images = {}
            for leaf in atlas["leaves"]:
                if not leaf.get("image"):
                    continue
                image_rel = safe_relative(leaf["image"], "image")
                src = (path.parent / image_rel).resolve()
                if root.resolve() not in src.parents or not src.is_file():
                    raise PublishError(f"Image {image_rel!r} of {rel} is missing or outside "
                                       "the run directory.")
                if src.suffix.lower() not in IMAGE_SUFFIXES:
                    raise PublishError(f"Unsupported image type {src.suffix!r}.")
                info = copy_image(src, staged / Path(rel).parent / image_rel,
                                  max_image_side, jpeg_quality)
                if info:
                    images[image_rel] = info
            atlas["published"] = {
                "by": "scripts/publish_atlases.py", "at": now, "run": name,
                "source": f"{run.resolve().name}/{rel}" if run.is_dir() else rel,
                "source_sha256": source_sha, "changes": notes,
                "image_max_side": max_image_side, "resized_images": images,
                "note": "Only the listed changes were made; mode, intervals/polygons, token "
                        "IDs, text, steps, outcomes, verification and manifest provenance "
                        "are copied unchanged.",
            }
            out = staged / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(atlas, ensure_ascii=False, allow_nan=False,
                                      separators=(",", ":")) + "\n")
            published = safe_relative(f"{name}/{rel}", "atlas")
            entries.append(entry_for(atlas, published, name, atlas["published"]["source"]))
        if not entries:
            return []
        (staged / MARKER).write_text(json.dumps({"run": name, "at": now,
                                                 "atlases": [e["path"] for e in entries]}))
        if target.exists():
            if not (target / MARKER).is_file():
                raise PublishError(f"{target} exists and was not written by this script; "
                                   "refusing to overwrite it.")
            shutil.rmtree(target)
        staged.rename(target)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return entries


def update_index(dest: Path, runs: dict[str, list[dict]]) -> dict:
    path = dest / "index.json"
    index = load_json(path) or {}
    if index and index.get("schema") != INDEX_SCHEMA:
        raise PublishError(f"{path} has schema {index.get('schema')!r}, not {INDEX_SCHEMA}.")
    kept = [e for e in index.get("atlases", [])
            if not any(e.get("path", "").startswith(f"{run}/") for run in runs)]
    index = {
        "schema": INDEX_SCHEMA,
        "note": index.get("note", "Paths are relative to this file. Every entry carries its "
                                  "mode label (MOONSHOTS.md section 5)."),
        **{k: v for k, v in index.items() if k not in ("schema", "note", "atlases")},
        "atlases": kept + [e for entries in runs.values() for e in entries],
    }
    path.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    return index


def tree_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run output directories or atlas JSON")
    parser.add_argument("--dest", type=Path, default=DEST, help="explorer data directory")
    parser.add_argument("--name", action="append", default=[],
                        help="run name per RUN_DIR, in order (default: directory name)")
    parser.add_argument("--prefix", default="", help="prefix for derived run names")
    parser.add_argument("--max-image-side", type=int, default=None,
                        help="downscale images so the longer side is at most this (Pillow)")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer id or path for runs that lack split token text")
    parser.add_argument("--no-auto-tokenizer", action="store_true",
                        help="do not try the run manifest's cached tokenizer")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="publish time-budget-truncated image atlases (explorer warns); "
                             "by default they are skipped")
    parser.add_argument("--skip-missing", action="store_true",
                        help="skip run directories that do not exist or hold no atlas")
    args = parser.parse_args(argv)
    if args.name and len(args.name) != len(args.runs):
        parser.error("--name must be given once per RUN_DIR.")
    tokenizers = TokenizerSource(args.tokenizer, auto=not args.no_auto_tokenizer)
    args.dest.mkdir(parents=True, exist_ok=True)
    published: dict[str, list[dict]] = {}
    for k, run in enumerate(args.runs):
        if args.skip_missing and (not run.exists() or not find_atlases(run)):
            print(f"skip {run}: no atlas", file=sys.stderr)
            continue
        name = args.name[k] if args.name else run_name(run, args.prefix)
        name = NAME_RE.sub("-", name).strip("-.")
        if name in published:
            raise PublishError(f"Two runs map to the name {name!r}; pass --name.")
        entries = publish_run(run, dest=args.dest, name=name, tokenizers=tokenizers,
                              max_image_side=args.max_image_side,
                              jpeg_quality=args.jpeg_quality,
                              allow_incomplete=args.allow_incomplete)
        if not entries:
            print(f"skip {run}: no complete atlas", file=sys.stderr)
            continue
        published[name] = entries
        for e in entries:
            print(f"published {e['path']} [{e['mode']}] {e['kind']} {e['schema']}: "
                  f"{e['leaves']} leaves")
        print(f"  {name}: {tree_size(args.dest / name)} bytes")
    if published:
        index = update_index(args.dest, published)
        print(f"index.json: {len(index['atlases'])} atlases")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublishError as error:
        raise SystemExit(f"publish_atlases: {error}") from None

"""Measure what each answer region of a text bid atlas says.

    python scripts/analyze_live_atlases.py ATLAS.json|DIR [...] [--output results/x.json]

For every text atlas (two-bidder ``bid-atlas-v1`` with ``kind == "text"``, or
three-bidder ``simplex-atlas-v1``) this reports shares of the bid space: of
lambda = a/(a+b) in (0, 1] (a leaf weighs hi - lo), or of the bid simplex (a cell
weighs its area over the simplex area):

* ``exact:A``, ``exact:B``, ... / ``multiple`` / ``neither``: the answer contains the
  exact brand name of that bidder, of two or more, or of none (case-sensitive);
* ``unsupported_number``: the answer contains a number (``4.8``, ``2023``, ``15``)
  that appears nowhere in the user prompt, the instructions or either fact
  sheet as recorded in the run manifest (``live_inputs_without_bids``);
* ``unsupported_rating``: a number written as a star rating (``4.6-star``,
  ``4.8 stars``, ``rating of 3.9``) that no fact sheet states;
* ``variant_name``: a capitalised name that shares a word with a sponsor's name
  but is neither sponsor's exact name (``Bayview Lodge``, ``Bay Area Hotel``);
* ``unresolved``: regions a time or cell budget cut off (only a prefix is known).
  They count in no other share, so shares are of the whole bid space.

These are string heuristics, not judgements of quality or truth: a number can be
harmless (a list index) and a paraphrased brand is counted as "neither". Every
leaf flagged ``unsupported_number`` is listed with its numbers so a reader can
check. Nothing here runs a model or changes an atlas; ``mode`` labels are copied.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
RATING_RE = re.compile(
    r"(\d(?:\.\d)?)\s*(?:-|\s)?\s*stars?\b|rat(?:ing|ed)\s+(?:of\s+|at\s+)?(\d\.\d)", re.IGNORECASE)
NAME_RE = re.compile(r"[A-Z][\w’'&-]*(?:\s+[A-Z][\w’'&-]*){1,3}")


def ratings(text: str) -> set[str]:
    return {a or b for a, b in RATING_RE.findall(text)}


def variant_names(text: str, brands) -> set[str]:
    words = {w.lower() for brand in brands for w in brand.split()}
    found = set()
    for match in NAME_RE.finditer(text):
        name = match.group(0)
        tokens = [re.sub(r"[’']s$", "", t).lower() for t in name.split()]
        if any(name in brand or brand in name for brand in brands):
            continue
        if words & set(tokens):
            found.add(name)
    return found


def numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", ".") for m in NUMBER_RE.finditer(text)}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def sources(atlas: dict) -> str:
    """Everything the model was shown: user prompt, instructions, fact sheets, prompt text."""
    manifest = atlas.get("manifest", {})
    parts = list(_strings(manifest.get("live_inputs_without_bids") or {}))
    parts += list(_strings(manifest.get("scenario_definition") or {}))
    parts.append(str(manifest.get("user_prompt", "")))
    parts += [str(p.get("text", "")) for p in manifest.get("prompts") or [] if isinstance(p, dict)]
    return "\n".join(parts)


def regions(atlas: dict):
    """(id, weight, text, resolved) per region; weights are shares of the bid space.

    A region cut off by a time or cell budget (``stop_reason == "truncated"``) holds
    only a prefix of its answers, so it is reported as unresolved, not analysed.
    """
    if "cells" in atlas:  # simplex-atlas-v1: weight = polygon area / simplex area
        total = float(atlas.get("stats", {}).get("simplex_area", 0.5))
        items = [(c, float(c["area"]) / total) for c in atlas["cells"]]
    else:  # bid-atlas-v1: weight = interval length in lambda
        items = [(leaf, float(leaf["hi"]) - float(leaf["lo"]))
                 for leaf in sorted(atlas["leaves"], key=lambda leaf: leaf["lo"])]
    for region, weight in items:
        yield (region["id"], weight, region.get("text", ""),
               region.get("stop_reason") != "truncated")


def analyze(atlas: dict) -> dict:
    brands = list(atlas["bidders"])
    keys = [f"exact:{chr(65 + i)}" for i in range(len(brands))]
    allowed = numbers(sources(atlas))
    stated = ratings(sources(atlas))
    share = dict.fromkeys(keys, 0.0)
    share.update({"multiple": 0.0, "neither": 0.0, "unsupported_number": 0.0,
                  "unsupported_rating": 0.0, "variant_name": 0.0, "unresolved": 0.0})
    flagged, bad_ratings, variants = [], {}, {}
    rows = list(regions(atlas))
    for rid, weight, text, resolved in rows:
        if not resolved:
            share["unresolved"] += weight
            continue
        named = [brand in text for brand in brands]
        for key, hit in zip(keys, named):
            share[key] += weight * hit
        share["multiple"] += weight * (sum(named) >= 2)
        share["neither"] += weight * (not any(named))
        extra = sorted(numbers(text) - allowed)
        if extra:
            share["unsupported_number"] += weight
            flagged.append({"region": rid, "weight": weight, "numbers": extra})
        rated = ratings(text) - stated
        if rated:
            share["unsupported_rating"] += weight
            for r in rated:
                bad_ratings[r] = bad_ratings.get(r, 0.0) + weight
        names = variant_names(text, brands)
        if names:
            share["variant_name"] += weight
            for n in names:
                variants[n] = variants.get(n, 0.0) + weight
    report = {
        "schema_in": atlas.get("schema"),
        "mode": atlas.get("mode"),
        "scenario": atlas.get("manifest", {}).get("scenario"),
        "seed": atlas.get("seed"),
        "bidders": brands,
        "regions": len(rows),
        "distinct_texts": len({text for _, _, text, resolved in rows if resolved}),
        "verification_status": atlas.get("verification_status"),
        "share": {k: round(v, 6) for k, v in share.items()},
        "unsupported_number_regions": flagged,
        "unsupported_ratings": {k: round(v, 6) for k, v in sorted(bad_ratings.items())},
        "variant_names": {k: round(v, 6)
                          for k, v in sorted(variants.items(), key=lambda kv: -kv[1])},
    }
    if "leaves" in atlas and rows:
        report["endpoints"] = {"lambda_to_0": rows[0][2], "lambda_to_1": rows[-1][2]}
    return report


def iter_atlases(paths):
    for raw in paths:
        path = Path(raw)
        files = [path] if path.is_file() else sorted(path.rglob("*.json"))
        for file in files:
            try:
                data = json.loads(file.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if (data.get("schema") == "bid-atlas-v1" and data.get("kind") == "text") or (
                data.get("schema") == "simplex-atlas-v1"
            ):
                yield file, data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", help="write the JSON report here")
    args = parser.parse_args(argv)
    rows = [dict(file=str(f.name), **analyze(d)) for f, d in iter_atlases(args.paths)]
    if not rows:
        print("no text bid atlases found", file=sys.stderr)
        return 1
    columns = ("neither", "unsupported_number", "unsupported_rating", "variant_name", "unresolved")
    print(f"{'atlas':30} {'mode':11} regions  exact(A,B,..)       "
          "neither number rating variant unresolved")
    for r in rows:
        s = r["share"]
        exact = ",".join(f"{v:.2f}" for k, v in s.items() if k.startswith("exact:"))
        rest = "   ".join(f"{s[c]:.3f}" for c in columns)
        print(f"{r['file'][:30]:30} {str(r['mode'])[:11]:11} {r['regions']:7}  {exact:18}  {rest}")
    if args.output:
        report = {"schema": "atlas-text-analysis-v1",
                  "note": __doc__.split("\n\n")[2].strip(), "atlases": rows}
        Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

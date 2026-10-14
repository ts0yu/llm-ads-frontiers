"""Pair reference and span-hold text atlases by (scenario, seed) and compare their shares.

    python scripts/compare_span_hold.py results/live_atlas_analysis.json [--output OUT.json]

Reads the report of scripts/analyze_live_atlases.py (which must include both samplers)
and prints, per scenario, the mean share of lambda with a variant brand name, with no
exact brand name, and with a star rating no fact sheet states, under each sampler,
for the seeds present under both. String heuristics; see analyze_live_atlases.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEYS = ("variant_name", "neither", "unsupported_rating", "unsupported_number")


def compare(rows: list[dict]) -> dict:
    by = {}
    for row in rows:
        if row.get("schema_in") != "bid-atlas-v1":
            continue
        by.setdefault((row["scenario"], row["seed"]), {})[row["sampler"]] = row
    scenarios: dict[str, dict] = {}
    for (scenario, seed), pair in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if "reference" not in pair or len(pair) < 2:
            continue
        other = next(k for k in pair if k != "reference")
        entry = scenarios.setdefault(scenario, {"sampler": other, "seeds": [], "pairs": []})
        entry["seeds"].append(seed)
        entry["pairs"].append({
            "seed": seed,
            **{f"{k}:reference": pair["reference"]["share"][k] for k in KEYS},
            **{f"{k}:{other}": pair[other]["share"][k] for k in KEYS},
            "variant_names:reference": pair["reference"]["variant_names"],
            f"variant_names:{other}": pair[other]["variant_names"],
        })
    for entry in scenarios.values():
        n = len(entry["pairs"])
        entry["mean"] = {
            f"{k}:{s}": sum(p[f"{k}:{s}"] for p in entry["pairs"]) / n
            for k in KEYS for s in ("reference", entry["sampler"])
        }
        other = entry["sampler"]
        entry["seeds_lower_variant"] = sum(
            p[f"variant_name:{other}"] < p["variant_name:reference"] for p in entry["pairs"]
        )
    return scenarios


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = compare(json.loads(Path(args.report).read_text())["atlases"])
    if not result:
        print("no scenario has both samplers for the same seed", file=sys.stderr)
        return 1
    for scenario, entry in result.items():
        m, s = entry["mean"], entry["sampler"]
        print(f"{scenario} ({len(entry['seeds'])} paired seeds; {s} lower variant share in "
              f"{entry['seeds_lower_variant']}):")
        for k in KEYS:
            print(f"  {k:20} reference {m[f'{k}:reference']:.3f}   {s} {m[f'{k}:{s}']:.3f}")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

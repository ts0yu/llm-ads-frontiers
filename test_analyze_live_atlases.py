"""String heuristics of scripts/analyze_live_atlases.py on a hand-written atlas."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "analyze_live_atlases.py"
spec = importlib.util.spec_from_file_location("analyze_live_atlases", SCRIPT)
analyze_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze_mod)


def atlas(leaves):
    return {
        "schema": "bid-atlas-v1",
        "kind": "text",
        "mode": "synthetic_fixture",
        "seed": 1,
        "bidders": ["Bayview Hotel", "Lantern House"],
        "manifest": {
            "scenario": "unit",
            "live_inputs_without_bids": {
                "prompt": "I land on Oct. 11th.",
                "instructions": "Be concise.",
                "advertisers": [
                    {"brand": "Bayview Hotel", "facts": "It has 4.9 stars on Google."},
                    {"brand": "Lantern House", "facts": "It has a quiet courtyard."},
                ],
            },
        },
        "leaves": [dict(id=i, lo=lo, hi=hi, text=text) for i, (lo, hi, text) in enumerate(leaves)],
    }


class AnalyzeLiveAtlasesTest(unittest.TestCase):
    def setUp(self):
        self.atlas = atlas([
            (0.0, 0.25, "Stay at Lantern House, arriving Oct. 11th."),
            (0.25, 0.5, "Try the Bay Area’s Lantern Hotel, rated 4.6 stars."),
            (0.5, 0.75, "Bayview Hotel has a 4.8-star rating; a cab takes 25 minutes."),
            (0.75, 1.0, "Bayview Hotel has 4.9 stars."),
        ])

    def test_shares_are_lambda_measures(self):
        report = analyze_mod.analyze(self.atlas)
        share = report["share"]
        self.assertAlmostEqual(share["exact:A"], 0.5)
        self.assertAlmostEqual(share["exact:B"], 0.25)
        self.assertAlmostEqual(share["neither"], 0.25)
        self.assertAlmostEqual(share["multiple"], 0.0)
        # 4.6 and (4.8, 25) are not in the prompt or the fact sheets; 11 and 4.9 are.
        self.assertAlmostEqual(share["unsupported_number"], 0.5)
        self.assertAlmostEqual(share["unsupported_rating"], 0.5)
        self.assertEqual(report["unsupported_ratings"], {"4.6": 0.25, "4.8": 0.25})
        self.assertAlmostEqual(share["variant_name"], 0.25)
        self.assertEqual(set(report["variant_names"]), {"Bay Area’s Lantern Hotel"})
        self.assertEqual(report["mode"], "synthetic_fixture")

    def test_simplex_cells_weigh_by_area(self):
        simplex = {
            "schema": "simplex-atlas-v1",
            "mode": "synthetic_fixture",
            "bidders": ["Lantern House", "Tideglass Hotel", "Saltmarsh Inn"],
            "manifest": {"prompts": [{"text": "Lantern House has 2 courtyards."}]},
            "stats": {"simplex_area": 0.5},
            "cells": [
                {"id": 0, "area": 0.20, "text": "Lantern House has 2 courtyards."},
                {"id": 1, "area": 0.15, "text": "Lantern House or Tideglass Hotel."},
                {"id": 2, "area": 0.10, "text": "The Saltmarsh Lodge has 3 rooms."},
                {"id": 3, "area": 0.05, "text": "Stay at Salt", "stop_reason": "truncated"},
            ],
        }
        report = analyze_mod.analyze(simplex)
        share = report["share"]
        self.assertAlmostEqual(share["exact:A"], 0.7)
        self.assertAlmostEqual(share["exact:B"], 0.3)
        self.assertAlmostEqual(share["exact:C"], 0.0)
        self.assertAlmostEqual(share["multiple"], 0.3)
        # The truncated prefix "Stay at Salt" is unresolved, not "neither".
        self.assertAlmostEqual(share["neither"], 0.2)
        self.assertAlmostEqual(share["unresolved"], 0.1)
        self.assertAlmostEqual(share["unsupported_number"], 0.2)
        self.assertEqual(set(report["variant_names"]), {"The Saltmarsh Lodge"})
        self.assertNotIn("endpoints", report)
        # The published form (scripts/publish_atlases.py) renames "cells" to "leaves".
        published = dict(simplex, leaves=simplex["cells"])
        del published["cells"]
        self.assertEqual(analyze_mod.analyze(published)["share"], share)

    def test_exact_names_are_not_variants(self):
        brands = ("Bayview Hotel", "Lantern House")
        names = analyze_mod.variant_names("Bayview Hotel and Lantern House near Fort Mason", brands)
        self.assertEqual(names, set())
        self.assertEqual(analyze_mod.variant_names("Lantern Bayview", brands), {"Lantern Bayview"})

    def test_cli_writes_report_and_skips_other_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.json").write_text(json.dumps(self.atlas))
            Path(tmp, "index.json").write_text(json.dumps({"schema": "atlas-index-v1"}))
            out = Path(tmp, "report.json")
            with redirect_stdout(StringIO()):
                code = analyze_mod.main([tmp, "--output", str(out)])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text())
            self.assertEqual(report["schema"], "atlas-text-analysis-v1")
            self.assertEqual([a["file"] for a in report["atlases"]],
                             [f"{Path(tmp).name}/a.json"])


if __name__ == "__main__":
    unittest.main()

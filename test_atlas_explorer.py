"""Atlas Explorer: fixture integrity, safe static route, and browser math vs Python.

The atlases under walkthrough/atlas_data are SYNTHETIC FIXTURES (toy models). These
tests check internal consistency and the explorer's arithmetic, not model quality.
"""

import json
import math
import shutil
import subprocess
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

from auction import RandomTape, TwoBidderAuction
from token_auction_lab.walkthrough import ATLAS_DATA, ROOT, Handler, atlas_path

INDEX = json.loads((ATLAS_DATA / "index.json").read_text())
ATLASES = {e["path"]: json.loads((ATLAS_DATA / e["path"]).read_text()) for e in INDEX["atlases"]}
PAIRS = {p: a for p, a in ATLASES.items() if a["schema"] == "bid-atlas-v1"}
SIMPLEX = {p: a for p, a in ATLASES.items() if a["schema"] == "simplex-atlas-v1"}


def python_charges(leaf, lam, total):
    """Oracle: resolve each recorded contest with auction.py at bids (S*lam, S*(1-lam)).

    A disjoint pair of point masses makes every draw contested, so the recorded
    contest uniform alone decides the branch and the reference price.
    """
    auction = TwoBidderAuction([1.0, 0.0], [0.0, 1.0])
    a = b = 0.0
    for step in leaf["steps"]:
        if step["branch"] == "shared":
            continue
        result = auction.resolve(total * lam, total * (1 - lam), RandomTape(.5, .5, .5, step["u"]))
        assert result.branch == step["branch"], (leaf["id"], step, lam)
        a, b = a + result.payment_a, b + result.payment_b
    return a, b


def race_charges(leaf, lam, total):
    charges = [0.0] * len(lam)
    for step in leaf["steps"]:
        if step["winner"] is None:
            continue
        i, E = step["winner"], step["E"]
        ratios = [lam[j] / E[j] for j in range(len(lam))]
        assert max(range(len(lam)), key=ratios.__getitem__) == i, (leaf["id"], step["step"])
        charges[i] += E[i] * min(total * lam[j] / E[j] for j in range(len(lam)) if j != i)
    return charges


def centroid(polygon):
    return [sum(p[k] for p in polygon) / len(polygon) for k in range(3)]


class AtlasFixtureTests(unittest.TestCase):
    def test_index_lists_labeled_fixtures_of_each_kind(self):
        self.assertEqual(INDEX["schema"], "atlas-index-v1")
        kinds = {(a["schema"], a["kind"]) for a in ATLASES.values()}
        self.assertEqual(kinds, {("bid-atlas-v1", "text"), ("bid-atlas-v1", "image"),
                                 ("simplex-atlas-v1", "text")})
        for entry in INDEX["atlases"]:
            atlas = ATLASES[entry["path"]]
            self.assertEqual(atlas["mode"], "synthetic_fixture")
            self.assertEqual(entry["mode"], atlas["mode"])
            self.assertIn("SYNTHETIC FIXTURE", atlas["manifest"]["label"])
            self.assertTrue(atlas["manifest"]["simulated_credits"])
            self.assertEqual(entry["leaves"], len(atlas["leaves"]))

    def test_two_bidder_leaves_partition_and_respect_contest_uniforms(self):
        for path, atlas in PAIRS.items():
            leaves = atlas["leaves"]
            self.assertEqual(leaves[0]["lo"], 0.0, path)
            self.assertEqual(leaves[-1]["hi"], 1.0, path)
            for prev, leaf in zip(leaves, leaves[1:]):
                self.assertEqual(prev["hi"], leaf["lo"], path)
            longest = max(len(leaf["steps"]) for leaf in leaves)
            self.assertLessEqual(len(leaves), 1 + longest)  # MOONSHOTS.md section 1 bound
            self.assertEqual(len(atlas["splits"]), len(leaves) - 1)
            boundaries = sorted(s["u"] for s in atlas["splits"])
            self.assertEqual(boundaries, [leaf["hi"] for leaf in leaves[:-1]])
            for leaf in leaves:
                self.assertLess(leaf["lo"], leaf["hi"])
                for step in leaf["steps"]:
                    if step["branch"] == "A":
                        self.assertLessEqual(step["u"], leaf["lo"])
                    elif step["branch"] == "B":
                        self.assertGreaterEqual(step["u"], leaf["hi"])
                if atlas["kind"] == "image":
                    self.assertIsNotNone(atlas_path(leaf["image"]), leaf["image"])
            self.assertTrue(all(v["matches_direct_run"] for v in atlas["verification"]))

    def test_simplex_cells_tile_the_triangle_and_match_recorded_winners(self):
        for atlas in SIMPLEX.values():
            area = 0.0
            for leaf in atlas["leaves"]:
                poly = leaf["polygon"]
                for p in poly:
                    self.assertAlmostEqual(sum(p), 1.0, places=12)
                    self.assertTrue(all(x >= -1e-12 for x in p))
                area += abs(sum(p[0] * q[1] - q[0] * p[1]
                                for p, q in zip(poly, poly[1:] + poly[:1]))) / 2
                race_charges(leaf, centroid(poly), 1.0)  # asserts winners
            self.assertAlmostEqual(area, 0.5, places=10)
            self.assertTrue(all(v["matches_direct_run"] for v in atlas["verification"]))


class AtlasRouteTests(unittest.TestCase):
    def test_atlas_path_allows_only_data_files_inside_atlas_data(self):
        self.assertEqual(atlas_path("index.json"), ATLAS_DATA / "index.json")
        self.assertIsNotNone(atlas_path("fixture-image-cafe/leaf-00.jpg"))
        for name in ("../atlas.mjs", "../../AGENTS.md", "../../.git/config", "/etc/passwd",
                     "../../token_auction_lab/walkthrough.py", "fixture-image-cafe/../../app.mjs",
                     ".hidden.json", "missing.json", "index.json\x00.png", "../index.html"):
            self.assertIsNone(atlas_path(name), name)

    def test_server_serves_explorer_and_rejects_traversal(self):
        class Quiet(Handler):
            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for path, kind in [("/atlas.html", "text/html"), ("/atlas.mjs", "text/javascript"),
                               ("/atlas.css", "text/css"),
                               ("/atlas_data/index.json", "application/json"),
                               ("/atlas_data/fixture-image-cafe/leaf-00.jpg", "image/jpeg")]:
                with urlopen(base + path, timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.headers["Content-Type"].startswith(kind), path)
                    self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            for path in ("/atlas_data/%2e%2e/%2e%2e/AGENTS.md", "/atlas_data/..%2f..%2fREADME.md",
                         "/atlas_data/%2e%2e/app.mjs", "/atlas_data/missing.json"):
                with self.assertRaises(HTTPError) as caught:
                    urlopen(base + path, timeout=3)
                self.assertEqual(caught.exception.code, 404, path)
        finally:
            server.shutdown()
            server.server_close()


@unittest.skipUnless(shutil.which("node"), "Node is only needed for the browser-math cross-check")
class AtlasBrowserMathTests(unittest.TestCase):
    def test_browser_charges_and_lookup_match_python_reference(self):
        rng = np.random.default_rng(2026)
        cases, lookups, profiles = [], [], []
        for path, atlas in PAIRS.items():
            leaves = atlas["leaves"]
            for i, leaf in enumerate(leaves):
                lams = list(rng.uniform(leaf["lo"], leaf["hi"], 6)) + [leaf["hi"]]
                lams.append(math.nextafter(leaf["lo"], 1.0))
                for lam in lams:
                    if not leaf["lo"] < lam <= leaf["hi"]:
                        continue
                    total = float(rng.choice([1.0, 2.0, 100.0, 37.5]))
                    a, b = python_charges(leaf, float(lam), total)
                    cases.append({"atlas": path, "leaf": i, "lambda": float(lam), "S": total,
                                  "a": a, "b": b})
                # Interval (lo, hi]: hi belongs to this leaf, lo to the previous one.
                lookups += [{"atlas": path, "lambda": leaf["hi"], "index": i},
                            {"atlas": path, "lambda": math.nextafter(leaf["lo"], 1.0), "index": i}]
                if leaf["lo"] > 0:
                    lookups.append({"atlas": path, "lambda": leaf["lo"], "index": i - 1})
            lookups += [{"atlas": path, "lambda": 0.0, "index": -1},
                        {"atlas": path, "lambda": math.nextafter(1.0, 2.0), "index": -1}]
            for key in leaves[0]["outcomes"]:
                values = [leaf["outcomes"][key] for leaf in leaves]
                if all(isinstance(v, bool) for v in values):
                    switches = sum(x != y for x, y in zip(values, values[1:]))
                    measure = sum(leaf["hi"] - leaf["lo"] for leaf, v in zip(leaves, values) if v)
                    profiles.append({"atlas": path, "key": key, "switches": switches,
                                     "measure": measure})
        race = []
        for path, atlas in SIMPLEX.items():
            for i, leaf in enumerate(atlas["leaves"]):
                point = centroid(leaf["polygon"])
                race.append({"atlas": path, "leaf": i, "point": point, "S": 10.0,
                             "charges": race_charges(leaf, point, 10.0)})
        self.assertGreater(len(cases), 100)
        self.assertTrue(any(p["switches"] > 1 for p in profiles), "no non-monotone outcome")
        script = """
import fs from 'fs';
import assert from 'assert';
import {chargesAt, leafIndexAt, outcomeProfile, raceCharges, cellIndexAt, wordDiff,
        partitionProblems} from './walkthrough/atlas.mjs';
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
const atlas = p => JSON.parse(fs.readFileSync('walkthrough/atlas_data/' + p, 'utf8'));
const cache = {}; const get = p => (cache[p] ||= atlas(p));
const close = (x, y) => Math.abs(x - y) <= 1e-12 * Math.max(1, Math.abs(y));
for (const c of data.cases) {
  const leaf = get(c.atlas).leaves[c.leaf];
  const r = chargesAt(leaf, c.lambda, c.S);
  assert.ok(close(r.a, c.a), `A ${c.atlas} ${c.leaf} ${c.lambda}: ${r.a} vs ${c.a}`);
  assert.ok(close(r.b, c.b), `B ${c.atlas} ${c.leaf} ${c.lambda}: ${r.b} vs ${c.b}`);
}
for (const l of data.lookups) {
  assert.strictEqual(leafIndexAt(get(l.atlas).leaves, l.lambda), l.index, JSON.stringify(l));
}
for (const p of data.profiles) {
  const r = outcomeProfile(get(p.atlas).leaves, p.key);
  assert.strictEqual(r.switches, p.switches); assert.strictEqual(r.monotone, p.switches <= 1);
  assert.ok(Math.abs(r.measure - p.measure) < 1e-12);
}
for (const p of Object.keys(data.pairs)) {
  assert.deepStrictEqual(partitionProblems(get(p).leaves), []);
}
for (const r of data.race) {
  const leaves = get(r.atlas).leaves;
  assert.strictEqual(cellIndexAt(leaves, r.point), r.leaf);
  const got = raceCharges(leaves[r.leaf], r.point, r.S).charges;
  got.forEach((v, i) => assert.ok(close(v, r.charges[i]), `race ${r.leaf} ${i}`));
}
const leaf = get(Object.keys(data.pairs)[0]).leaves[0];
assert.throws(() => chargesAt(leaf, leaf.hi + 1e-9, 1));
assert.throws(() => chargesAt(leaf, leaf.lo, 1));
const diff = wordDiff('Stay at Bayview Hotel today', 'Stay at Lantern House today');
assert.deepStrictEqual(diff.filter(d => d.changed).map(d => d.text), ['Bayview', 'Hotel']);
"""
        payload = {"cases": cases, "lookups": lookups, "profiles": profiles, "race": race,
                   "pairs": {p: True for p in PAIRS}}
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script],
                                input=json.dumps(payload), text=True, cwd=ROOT,
                                capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

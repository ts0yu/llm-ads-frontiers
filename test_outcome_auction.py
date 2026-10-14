"""Checks for token_auction_lab/outcome_auction.py (outcome auctions on stored atlases).

1. Agreement with walkthrough/mechanism_lab.mjs on the same inputs: the committed
   model-test atlases and multi-seed synthetic city-walk / melody atlases from the
   browser playground, exported in bid-atlas-v1 form.
2. Incentives by brute force over a grid of misreports: VCG (ex ante and per leaf)
   and repaired proportional Myerson have no profitable deviation beyond 1e-9.
3. Payments are nonnegative and at most reported value x probability (IR).
4. Exactness: P_i(lambda) equals the share of seeds whose leaf at lambda satisfies
   the predicate, and b K_A(lambda), a K_B(lambda) equal direct expected charges.

All atlases here are synthetic (labeled synthetic_fixture or synthetic_model_test);
nothing is a live model run. Credits are simulated.
"""

import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from token_auction_lab import outcome_auction as oa
from token_auction_lab.walkthrough import ROOT

NODE = shutil.which("node") or (
    "/opt/node22/bin/node" if Path("/opt/node22/bin/node").exists() else None
)
DATA = ROOT / "walkthrough" / "atlas_data"
TEXT_ATLAS = DATA / "model-test-text" / "transit-hotel-seed3.json"
IMAGE_ATLAS = DATA / "model-test-image" / "cafe_seed2.atlas.json"
VALUE_PAIRS = [(40.0, 40.0), (40.0, 20.0), (10.0, 60.0), (100.0, 40.0), (3.0, 0.5)]


def random_atlas(rng, seed, steps=14, names=("Alpha", "Beta")):
    """A SYNTHETIC bid-atlas-v1 document with a consistent leaf/step structure.

    Contest uniforms are global per step; leaf boundaries are some of them. On a
    leaf (lo, hi], a contested step is A-won iff u <= lo and B-won iff u >= hi.
    Outcomes are random and deliberately non-monotone in lambda.
    """
    us = rng.uniform(0.02, 0.98, steps)
    contested = rng.random(steps) < 0.6
    cuts = sorted({float(us[k]) for k in range(steps) if contested[k] and rng.random() < 0.6})
    bounds = [0.0, *cuts, 1.0]
    leaves = []
    for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        recs = []
        for k in range(steps):
            u = float(us[k])
            branch = "shared" if not contested[k] or lo < u < hi else ("A" if u <= lo else "B")
            recs.append({"step": k, "branch": branch, "u": u})
        mid = (lo + hi) / 2
        outcomes = {
            f"mentions:{names[0]}": bool(rng.random() < 0.15 + 0.7 * mid),
            f"mentions:{names[1]}": bool(rng.random() < 0.85 - 0.7 * mid),
        }
        leaves.append({"id": i, "lo": lo, "hi": hi, "steps": recs, "outcomes": outcomes})
    return {
        "schema": oa.SCHEMA, "kind": "text", "mode": "synthetic_fixture",
        "bidders": list(names), "seed": seed, "max_steps": steps,
        "manifest": {"generator": "test_outcome_auction.random_atlas"}, "leaves": leaves,
    }


def random_lab(seed, n_seeds=25):
    rng = np.random.default_rng(seed)
    atlases = [random_atlas(rng, s) for s in range(n_seeds)]
    atlas_set = oa.AtlasSet.from_docs(atlases)
    return atlas_set, oa.build_lab(atlas_set)


def direct_leaf(atlas, lam):
    for leaf in atlas["leaves"]:
        if leaf["lo"] < lam <= leaf["hi"]:
            return leaf
    raise AssertionError(lam)


def misreports(value, lab, rival, side):
    """Grid of reports: log-spaced, near the truth, and at every piece boundary."""
    grid = set(np.geomspace(1e-3, 1e3, 160).tolist()) | {0.0, value}
    grid |= {value * (1 + d) for d in (-1e-6, 1e-6, -0.1, 0.1, -0.5, 1.0)}
    for lo, hi in zip(lab.lo.tolist(), lab.hi.tolist()):
        for lam in (lo, hi):
            if 0 < lam < 1:
                z = rival * lam / (1 - lam) if side == "A" else rival * (1 - lam) / lam
                grid |= {z, z * (1 - 1e-9), z * (1 + 1e-9)}
    return sorted(z for z in grid if z >= 0)


class OutcomeAuctionTests(unittest.TestCase):
    def assertClose(self, got, want, tol=1e-12, msg=None):
        self.assertLessEqual(abs(got - want), tol * (1 + abs(want)), msg)

    # ---------------------------------------------------------------- exactness

    def test_curves_equal_direct_leaf_shares_and_charges(self):
        for seed in (1, 2, 3):
            atlas_set, lab = random_lab(seed)
            pa, pb = oa.mention("Alpha"), oa.mention("Beta")
            rng = np.random.default_rng(100 + seed)
            lams = [*rng.uniform(0, 1, 200).tolist(), 1.0, *lab.hi.tolist(),
                    *((lab.lo + lab.hi) / 2).tolist()]
            for lam in lams:
                leaves = [direct_leaf(a, lam) for a in atlas_set.atlases]
                n = len(leaves)
                got = oa.curve_at(lab, lam)
                # Exact equality: shares of seeds, accumulated as integers.
                self.assertEqual(got["PA"], sum(pa(x) for x in leaves) / n)
                self.assertEqual(got["PB"], sum(pb(x) for x in leaves) / n)
                # Direct charges at bids (a, b) with a / (a + b) = lam, u < lam decides.
                a, b = lam * 3.0, (1 - lam) * 3.0
                pay_a = pay_b = 0.0
                for leaf in leaves:
                    for s in leaf["steps"]:
                        if s["branch"] == "shared":
                            continue
                        u = s["u"]
                        self.assertEqual(u < lam, s["branch"] == "A")
                        if u < lam:
                            pay_a += b * (u / (1 - u))
                        else:
                            pay_b += a * ((1 - u) / u)
                self.assertClose(b * got["KA"], pay_a / n, 1e-11)
                self.assertClose(a * got["KB"], pay_b / n, 1e-11)

    def test_committed_model_test_atlas_curves_are_its_leaves(self):
        atlas_set = oa.load_atlases([TEXT_ATLAS])
        self.assertEqual(atlas_set.mode, "synthetic_model_test")
        lab = oa.build_lab(atlas_set)
        atlas = atlas_set.atlases[0]
        self.assertEqual(lab.m, len(atlas["leaves"]))  # one seed: pieces are its leaves
        for j, leaf in enumerate(atlas["leaves"]):
            self.assertEqual(lab.PA[j], float(leaf["outcomes"]["mentions:HarborHop"]))
            self.assertEqual(lab.PB[j], float(leaf["outcomes"]["mentions:Lantern House"]))
        # The stored verification records hold direct run_round payments at chosen bids.
        for record in atlas["verification"]:
            got = oa.curve_at(lab, record["lambda"])
            self.assertClose(record["bid_b"] * got["KA"], record["payment_a_direct"], 1e-9)
            self.assertClose(record["bid_a"] * got["KB"], record["payment_b_direct"], 1e-9)

    # ---------------------------------------------------------------- incentives

    def _utility(self, mech, lab, side, true_value, report, rival):
        va, vb = (report, rival) if side == "A" else (rival, report)
        r = mech(lab, va, vb)
        return true_value * r["allocation"][side] - r["payments"][side]

    def _check_truthful(self, mech, lab, pairs, positive=False):
        worst = 0.0
        for va, vb in pairs:
            for side, value, rival in (("A", va, vb), ("B", vb, va)):
                truth = self._utility(mech, lab, side, value, value, rival)
                for z in misreports(value, lab, rival, side):
                    if positive and z == 0:
                        continue
                    gain = self._utility(mech, lab, side, value, z, rival) - truth
                    worst = max(worst, gain)
        return worst

    def test_vcg_mechanisms_are_truthful_by_brute_force(self):
        for seed in (4, 5):
            _, lab = random_lab(seed, n_seeds=20)
            for name, mech in (("vcg", oa.vcg_ex_ante), ("leaf-vcg", oa.leaf_vcg)):
                with self.subTest(seed=seed, mechanism=name):
                    self.assertLessEqual(self._check_truthful(mech, lab, VALUE_PAIRS), 1e-9)

    def test_repaired_proportional_is_truthful_and_raw_regret_is_exact_bound(self):
        for seed in (6, 7):
            _, lab = random_lab(seed, n_seeds=20)
            mono = oa.monotonicity(lab)
            self.assertFalse(mono["PA_nondecreasing"] and mono["PB_nonincreasing"],
                             "fixture should need the repair")
            with self.subTest(seed=seed, mechanism="proportional"):
                worst = self._check_truthful(oa.proportional_myerson, lab, VALUE_PAIRS, True)
                self.assertLessEqual(worst, 1e-9)
                for va, vb in VALUE_PAIRS:
                    r = oa.proportional_myerson(lab, va, vb)
                    self.assertLessEqual(max(r["regret"].values()), 1e-9)
            # Without the repair, the brute-force gain never exceeds the exact regret.
            for va, vb in VALUE_PAIRS:
                raw = oa.proportional_myerson(lab, va, vb, monotone=False)
                for side, value, rival in (("A", va, vb), ("B", vb, va)):
                    truth = raw["utility"][side]
                    best = max(
                        self._utility(lambda lb, x, y: oa.proportional_myerson(
                            lb, x, y, monotone=False), lab, side, value, z, rival)
                        for z in misreports(value, lab, rival, side) if z > 0
                    )
                    self.assertLessEqual(best - truth, raw["regret"][side] + 1e-9)

    def test_payments_nonnegative_and_individually_rational(self):
        for seed in (8, 9):
            _, lab = random_lab(seed)
            for va, vb in VALUE_PAIRS:
                for name in ("vcg", "proportional", "leaf-vcg"):
                    r = oa.run_mechanism(lab, name, va, vb)
                    with self.subTest(seed=seed, values=(va, vb), mechanism=name):
                        for side, v in (("A", va), ("B", vb)):
                            pay, x = r["payments"][side], r["allocation"][side]
                            self.assertGreaterEqual(pay, -1e-12)
                            self.assertLessEqual(pay, v * x + 1e-9)
                            self.assertGreaterEqual(r["utility"][side], -1e-9)
                        self.assertClose(r["revenue"], sum(r["payments"].values()))
                        if name == "leaf-vcg":
                            for s in r["per_seed"]:
                                for side, v in (("A", va), ("B", vb)):
                                    self.assertGreaterEqual(s["payments"][side], -1e-12)
                                    self.assertLessEqual(s["payments"][side], v * s[side] + 1e-9)
                # Raw proportional Myerson keeps utility >= 0 even without monotonicity.
                raw = oa.proportional_myerson(lab, va, vb, monotone=False)
                self.assertGreaterEqual(min(raw["utility"].values()), -1e-9)

    def test_vcg_maximizes_reported_welfare(self):
        _, lab = random_lab(10)
        for va, vb in VALUE_PAIRS:
            r = oa.vcg_ex_ante(lab, va, vb)
            self.assertClose(r["welfare"], float(np.max(va * lab.PA + vb * lab.PB)), 1e-12)
            self.assertEqual(oa.piece_index(lab, r["lambda"]), r["piece"])
            post = oa.leaf_vcg(lab, va, vb)
            self.assertGreaterEqual(post["welfare"], r["welfare"] - 1e-12)

    # ---------------------------------------------------------------- loading and CLI

    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = oa.main([str(a) for a in argv])
        return result, out.getvalue()

    def test_cli_on_committed_text_atlas_keeps_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            result, text = self.run_cli(DATA / "model-test-text", "--value", "HarborHop=40",
                                        "--value", "Lantern House=40", "--json", path)
            saved = json.loads(path.read_text())
        self.assertIn("MODE: synthetic_model_test", text)
        self.assertNotIn("live", text.lower())
        self.assertEqual(saved["mode"], "synthetic_model_test")
        self.assertEqual(saved["credit_unit"], "simulated")
        self.assertEqual([r["mechanism"] for r in saved["results"]],
                         ["vcg", "proportional", "leaf-vcg"])
        self.assertEqual(saved["sources"][0]["seed"], 3)
        self.assertEqual(len(saved["sources"][0]["sha256"]), 64)
        self.assertEqual(result["results"][0]["allocation"], {"A": 1.0, "B": 0.0})

    def test_cli_rejects_mixed_prompts_and_missing_outcomes(self):
        quiet = contextlib.redirect_stdout(io.StringIO())
        with quiet, self.assertRaises(SystemExit) as ctx:
            oa.main([str(DATA), "--value", "HarborHop=1", "--value", "Lantern House=1"])
        self.assertIn("one prompt", str(ctx.exception))
        with quiet, self.assertRaises(SystemExit) as ctx:
            oa.main([str(IMAGE_ATLAS), "--value", "Carmello Roastworks=1",
                     "--value", "Tidewick Tea Bar=1"])
        self.assertIn("no stored outcome", str(ctx.exception))
        with quiet, self.assertRaises(SystemExit):
            oa.main([str(TEXT_ATLAS), "--value", "HarborHop=1"])  # one value missing

    def test_numeric_outcome_threshold_on_image_fixture(self):
        result, text = self.run_cli(
            DATA / "fixture-image-cafe.json", "--value", "Tidewater Coffee=40",
            "--value", "Ember Roasters=30", "--mechanism", "vcg",
            "--outcome", "Tidewater Coffee=clip:Tidewater Coffee>=0.2",
            "--outcome", "Ember Roasters=clip:Ember Roasters>0.25")
        self.assertIn("MODE: synthetic_fixture", text)
        doc = json.loads((DATA / "fixture-image-cafe.json").read_text())
        leaf = direct_leaf(doc, result["results"][0]["lambda"])
        want_a = leaf["outcomes"]["clip:Tidewater Coffee"] >= 0.2
        self.assertEqual(result["results"][0]["allocation"]["A"], float(want_a))

    def test_predicates_and_labels(self):
        leaf = {"id": 0, "outcomes": {"mentions:Bay": True, "score": 0.4}}
        self.assertTrue(oa.mention("Bay")(leaf))
        self.assertFalse(oa.outcome_predicate("score>0.5")(leaf))
        self.assertTrue(oa.outcome_predicate("score <= 0.4")(leaf))
        with self.assertRaises(KeyError):
            oa.mention("bay")(leaf)  # case-sensitive: a different key
        with self.assertRaises(TypeError):
            oa.outcome_predicate("score")(leaf)
        self.assertIn("LIVE MODEL", oa.mode_label("live_model"))
        self.assertIn("'replay'", oa.mode_label("replay"))
        custom = oa.prepare_lab([json.loads(TEXT_ATLAS.read_text())],
                                lambda leaf: leaf["id"] % 2 == 0, lambda leaf: True)
        self.assertTrue(np.all(custom.PB == 1.0))

    def test_duplicate_seeds_and_bad_partitions_rejected(self):
        rng = np.random.default_rng(0)
        a = random_atlas(rng, 1)
        with self.assertRaises(ValueError):
            oa.AtlasSet.from_docs([a, random_atlas(rng, 1)])
        broken = json.loads(json.dumps(a))
        broken["leaves"][-1]["hi"] = 0.99
        with self.assertRaises(ValueError):
            oa.AtlasSet.from_docs([broken])


JS_CHECK = """
import fs from 'node:fs';
import { buildAtlas } from './walkthrough/atlas_core.mjs';
import { cityWalk, melody, blueShare } from './walkthrough/playground_worlds.mjs';
import * as L from './walkthrough/mechanism_lab.mjs';
const PAIRS = %s;
const arr = x => Array.from(x);
function exportAtlas(atlas, kind, names, tA, tB) {
  return {
    schema: 'bid-atlas-v1', kind, mode: 'synthetic_fixture', bidders: names, seed: atlas.seed,
    max_steps: atlas.maxSteps, manifest: { generator: 'walkthrough/playground_worlds.mjs ' + kind },
    leaves: atlas.leaves.map(l => ({
      id: l.id, lo: l.lo, hi: l.hi,
      steps: l.steps.map(s => ({ step: s.step, branch: s.branch, u: s.u ?? null })),
      outcomes: { ['mentions:' + names[0]]: !!tA(l), ['mentions:' + names[1]]: !!tB(l) },
    })),
  };
}
function run(atlases, tA, tB) {
  const lab = L.prepareLab(atlases, tA, tB);
  const P = lab.pieces;
  const mechanisms = PAIRS.map(([VA, VB]) => {
    const ref = VA / (VA + VB);
    const post = L.vcgExPost(lab, VA, VB, { ref, compareLambda: ref });
    return {
      vcg: L.vcgExAnte(lab, VA, VB, { ref }),
      post: { ...post, chosen: post.chosen.map(c => c.leaf) },
      raw: L.proportionalMyerson(lab, VA, VB),
      mono: L.proportionalMyerson(lab, VA, VB, { monotone: true }),
    };
  });
  return { pieces: { lo: P.lo, hi: P.hi, PA: arr(P.PA), PB: arr(P.PB), KA: arr(P.KA), KB: arr(P.KB) },
           mechanisms };
}
const out = {};
const worlds = {
  city: [cityWalk(), 80, ['Brindlewick Coffee', 'Quillmere Books'], l => l.state.visited[0], l => l.state.visited[1], 40],
  melody: [melody(), 32, ['Low Blue', 'High Blue'], l => blueShare(l.state.notes) < 0.15, l => blueShare(l.state.notes) > 0.3, 30],
};
for (const [name, [gen, T, names, tA, tB, n]] of Object.entries(worlds)) {
  const docs = [];
  for (let s = 1; s <= n; s += 1) docs.push(exportAtlas(buildAtlas(gen, s, T), name, names, tA, tB));
  const key = k => l => l.outcomes['mentions:' + names[k]];
  out[name] = { docs, ...run(docs, key(0), key(1)) };
}
const text = JSON.parse(fs.readFileSync('walkthrough/atlas_data/model-test-text/transit-hotel-seed3.json'));
out.text = run([text], l => l.outcomes['mentions:HarborHop'], l => l.outcomes['mentions:Lantern House']);
const image = JSON.parse(fs.readFileSync('walkthrough/atlas_data/model-test-image/cafe_seed2.atlas.json'));
out.image = run([image], l => l.id %% 2 === 0, l => l.id %% 3 === 0);
console.log(JSON.stringify(out));
"""  # noqa: E501


@unittest.skipUnless(NODE, "Node.js is required for the mechanism_lab.mjs cross-check")
class MechanismLabAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = JS_CHECK % json.dumps([list(p) for p in VALUE_PAIRS])
        cls.js = json.loads(subprocess.run(
            [NODE, "--input-type=module"], input=source, cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout)

    def compare(self, lab, js):
        tol = 1e-12
        for key in ("lo", "hi", "PA", "PB", "KA", "KB"):
            got, want = getattr(lab, key), np.array(js["pieces"][key], dtype=float)
            self.assertEqual(got.shape, want.shape, key)
            np.testing.assert_allclose(got, want, rtol=tol, atol=tol, err_msg=key)
        for (va, vb), want in zip(VALUE_PAIRS, js["mechanisms"]):
            with self.subTest(values=(va, vb)):
                vcg = oa.vcg_ex_ante(lab, va, vb)
                w = want["vcg"]
                self.assertEqual(vcg["piece"], w["piece"])
                for got, exp in ((vcg["lambda"], w["lambda"]), (vcg["payments"]["A"], w["payA"]),
                                 (vcg["payments"]["B"], w["payB"]),
                                 (vcg["welfare"], w["welfare"]), (vcg["utility"]["A"], w["uA"])):
                    self.assertAlmostEqual(got, exp, delta=tol * (1 + abs(exp)))
                post = oa.leaf_vcg(lab, va, vb)
                w = want["post"]
                self.assertEqual([s["leaf"] for s in post["per_seed"]], w["chosen"])
                for got, exp in ((post["allocation"]["A"], w["PA"]),
                                 (post["allocation"]["B"], w["PB"]),
                                 (post["payments"]["A"], w["payA"]),
                                 (post["payments"]["B"], w["payB"]),
                                 (post["output_changed_vs_ref"], w["outputChangedVsCompare"])):
                    self.assertAlmostEqual(got, exp, delta=tol * (1 + abs(exp)))
                for monotone, w in ((False, want["raw"]), (True, want["mono"])):
                    r = oa.proportional_myerson(lab, va, vb, monotone=monotone)
                    self.assertEqual(r["piece"], w["piece"])
                    for got, exp in ((r["implemented_lambda"], w["implementedLambda"]),
                                     (r["allocation"]["A"], w["PA"]),
                                     (r["allocation"]["B"], w["PB"]),
                                     (r["payments"]["A"], w["payA"]),
                                     (r["payments"]["B"], w["payB"]),
                                     (r["regret"]["A"], w["regret"]["A"]),
                                     (r["regret"]["B"], w["regret"]["B"]),
                                     (r["remapped_measure"], w["remappedMeasure"])):
                        self.assertAlmostEqual(got, exp, delta=1e-11 * (1 + abs(exp)))

    def test_agrees_with_js_on_synthetic_multi_seed_worlds(self):
        for world in ("city", "melody"):
            with self.subTest(world=world):
                atlas_set = oa.AtlasSet.from_docs(self.js[world]["docs"])
                self.assertEqual(atlas_set.mode, "synthetic_fixture")
                self.compare(oa.build_lab(atlas_set), self.js[world])

    def test_agrees_with_js_on_committed_model_test_atlases(self):
        self.compare(oa.build_lab(oa.load_atlases([TEXT_ATLAS])), self.js["text"])
        image = oa.load_atlases([IMAGE_ATLAS])
        self.assertEqual(image.mode, "synthetic_model_test")
        lab = oa.prepare_lab(image.atlases, lambda leaf: leaf["id"] % 2 == 0,
                             lambda leaf: leaf["id"] % 3 == 0)
        self.compare(lab, self.js["image"])

    def test_vcg_truthful_on_city_world(self):
        lab = oa.build_lab(oa.AtlasSet.from_docs(self.js["city"]["docs"]))
        for va, vb in VALUE_PAIRS[:3]:
            for mech in (oa.vcg_ex_ante, oa.leaf_vcg, oa.proportional_myerson):
                for side, value, rival in (("A", va, vb), ("B", vb, va)):
                    def util(z):
                        x, y = (z, rival) if side == "A" else (rival, z)
                        r = mech(lab, x, y)
                        return value * r["allocation"][side] - r["payments"][side]
                    truth = util(value)
                    gains = [util(z) - truth for z in misreports(value, lab, rival, side)[::7]
                             if z > 0]
                    self.assertLessEqual(max(gains), 1e-9)


if __name__ == "__main__":
    unittest.main()

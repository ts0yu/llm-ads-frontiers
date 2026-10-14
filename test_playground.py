"""Exactness checks for the browser Atlas Playground (synthetic generators).

The JavaScript auction port must match the Python reference on shared inputs,
and every atlas leaf must equal a direct generation at bids inside its interval.
"""

import json
import shutil
import subprocess
import unittest

import numpy as np

from auction import RandomTape, TwoBidderAuction
from token_auction_lab.walkthrough import ROOT

NODE = shutil.which("node")

ATLAS_CHECK = """
import { buildAtlas, runDirect, leafPayments, mulberry32 } from './walkthrough/atlas_core.mjs';
import { cityWalk, melody } from './walkthrough/playground_worlds.mjs';
const report = [];
for (const [name, gen, T] of [['city', cityWalk(), 80], ['melody', melody(), 32]]) {
  for (let seed = 1; seed <= 12; seed += 1) {
    const atlas = buildAtlas(gen, seed, T);
    const rand = mulberry32(1000 + seed);
    let lo = 0, mismatches = 0, maxPayError = 0;
    for (const leaf of atlas.leaves) {
      if (leaf.lo !== lo) mismatches += 1000;  // leaves must tile (0, 1]
      lo = leaf.hi;
      const inside = leaf.lo + (leaf.hi - leaf.lo) * (0.01 + 0.98 * rand());
      for (const lam of [(leaf.lo + leaf.hi) / 2, inside]) {
        const S = 0.5 + 3 * rand();
        const d = runDirect(gen, seed, T, lam * S, (1 - lam) * S);
        if (JSON.stringify(d.options) !== JSON.stringify(leaf.options)) mismatches += 1;
        const p = leafPayments(leaf, lam * S, (1 - lam) * S);
        maxPayError = Math.max(maxPayError, Math.abs(p.a - d.paidA), Math.abs(p.b - d.paidB));
      }
    }
    const endA = runDirect(gen, seed, T, 1, 0), endB = runDirect(gen, seed, T, 0, 1);
    report.push({ name, seed, leaves: atlas.leaves.length, bound: atlas.bound, last: lo,
      mismatches, maxPayError,
      endA: JSON.stringify(endA.options) === JSON.stringify(atlas.leaves.at(-1).options),
      endB: JSON.stringify(endB.options) === JSON.stringify(atlas.leaves[0].options) });
  }
}
console.log(JSON.stringify(report));
"""


GAME_CHECK = """
import { buildAtlas, runDirect, mulberry32 } from './walkthrough/atlas_core.mjs';
import { cityWalk } from './walkthrough/playground_worlds.mjs';
import { aggregate, piecesAt } from './walkthrough/bid_game.mjs';
const gen = cityWalk(), T = 80, seeds = 25;
const atlases = [];
for (let s = 1; s <= seeds; s += 1) atlases.push(buildAtlas(gen, s, T));
const game = aggregate(atlases, l => l.state.visited[0], l => l.state.visited[1]);
const rand = mulberry32(99);
let worst = 0;
for (let i = 0; i < 30; i += 1) {
  const lam = 0.001 + 0.998 * rand(), S = 0.5 + 2 * rand(), a = lam * S, b = (1 - lam) * S;
  let pa = 0, pb = 0, payA = 0, payB = 0;
  for (let s = 1; s <= seeds; s += 1) {
    const d = runDirect(gen, s, T, a, b);
    pa += d.state.visited[0] ? 1 : 0; pb += d.state.visited[1] ? 1 : 0;
    payA += d.paidA; payB += d.paidB;
  }
  const p = piecesAt(game, lam);
  worst = Math.max(worst, Math.abs(p.PA - pa / seeds), Math.abs(p.PB - pb / seeds),
    Math.abs(b * p.KA - payA / seeds), Math.abs(a * p.KB - payB / seeds));
}
console.log(JSON.stringify({ worst, pieces: game.pieces.length }));
"""


@unittest.skipUnless(NODE, "Node.js is required for browser checks")
class PlaygroundTests(unittest.TestCase):
    def run_node(self, source: str) -> str:
        return subprocess.run(
            [NODE, "--input-type=module"],
            input=source,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_js_auction_matches_python_reference(self):
        rng = np.random.default_rng(20261014)
        cases = []
        for _ in range(400):
            n = int(rng.integers(2, 9))
            p_a = rng.dirichlet(np.full(n, 0.6))
            p_b = rng.dirichlet(np.full(n, 0.6))
            if rng.random() < 0.2:
                p_b = p_a.copy()
            bids = [float(x) for x in rng.choice([0.0, 0.3, 1.0, 2.5], 2)]
            if sum(bids) == 0:
                bids[0] = 1.0
            tape = RandomTape.draw(rng)
            cases.append(
                {"pA": p_a.tolist(), "pB": p_b.tolist(), "bids": bids, "tape": tape.__dict__}
            )
        script = (
            "import { TwoBidderAuction } from './walkthrough/atlas_core.mjs';"
            f"const cases = {json.dumps(cases)};"
            "console.log(JSON.stringify(cases.map(c => new TwoBidderAuction(c.pA, c.pB)"
            ".resolve(c.bids[0], c.bids[1], c.tape))));"
        )
        js = json.loads(self.run_node(script))
        for case, got in zip(cases, js):
            ref = TwoBidderAuction(case["pA"], case["pB"]).resolve(
                *case["bids"], RandomTape(**case["tape"])
            )
            self.assertEqual(got["option"], ref.token_id)
            self.assertEqual(got["branch"], ref.branch)
            self.assertAlmostEqual(got["payment_a"], ref.payment_a, places=12)
            self.assertAlmostEqual(got["payment_b"], ref.payment_b, places=12)

    def test_bid_game_curves_equal_direct_averages(self):
        result = json.loads(self.run_node(GAME_CHECK))
        self.assertGreater(result["pieces"], 1)
        self.assertLess(result["worst"], 1e-9)

    def test_every_atlas_leaf_equals_a_direct_generation(self):
        report = json.loads(self.run_node(ATLAS_CHECK))
        self.assertEqual(len(report), 24)
        for row in report:
            with self.subTest(world=row["name"], seed=row["seed"]):
                self.assertEqual(row["mismatches"], 0)
                self.assertLess(row["maxPayError"], 1e-9)
                self.assertLessEqual(row["leaves"], row["bound"])
                self.assertEqual(row["last"], 1)
                self.assertTrue(row["endA"])
                self.assertTrue(row["endB"])


if __name__ == "__main__":
    unittest.main()

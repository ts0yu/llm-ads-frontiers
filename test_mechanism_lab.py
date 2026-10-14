"""Checks for walkthrough/mechanism_lab.mjs (outcome-level bidding experiments).

The lab must reduce to bid_game.mjs exactly when its modification is off, its
piecewise formulas must equal direct generations, and the outcome mechanisms must
satisfy the incentive properties claimed in docs/OUTCOME_BIDDING.md.
"""

import json
import shutil
import subprocess
import unittest

import numpy as np

from auction import RandomTape, TwoBidderAuction
from token_auction_lab.walkthrough import ROOT

NODE = shutil.which("node")

SETUP = """
import { buildAtlas, mulberry32 } from './walkthrough/atlas_core.mjs';
import { cityWalk, melody, blueShare } from './walkthrough/playground_worlds.mjs';
import { aggregate, dynamics, bestResponseA, bestResponseB } from './walkthrough/bid_game.mjs';
import * as L from './walkthrough/mechanism_lab.mjs';
const worlds = {
  city: [cityWalk(), 80, l => l.state.visited[0], l => l.state.visited[1]],
  melody: [melody(), 32, l => blueShare(l.state.notes) < 0.15, l => blueShare(l.state.notes) > 0.3],
};
function labFor(name, seeds) {
  const [gen, T, tA, tB] = worlds[name];
  const atlases = [];
  for (let s = 1; s <= seeds; s += 1) atlases.push(buildAtlas(gen, s, T));
  return { gen, T, tA, tB, atlases, lab: L.prepareLab(atlases, tA, tB) };
}
"""

OFF_CHECK = SETUP + """
const out = [];
for (const name of ['city', 'melody']) {
  const { atlases, tA, tB, lab } = labFor(name, 30);
  const game = aggregate(atlases, tA, tB);
  const P = lab.pieces;
  let pieceErr = game.pieces.length === lab.m ? 0 : Infinity;
  game.pieces.forEach((p, j) => {
    pieceErr = Math.max(pieceErr, Math.abs(p.lo - P.lo[j]), Math.abs(p.hi - P.hi[j]), Math.abs(p.PA - P.PA[j]),
      Math.abs(p.PB - P.PB[j]), Math.abs(p.KA - P.KA[j]) / (1 + p.KA), Math.abs(p.KB - P.KB[j]) / (1 + p.KB));
  });
  const rows = [];
  for (const [VA, VB, a0, b0] of [[40, 40, 1, 1], [40, 20, 0.3, 2], [10, 60, 5, 0.1]]) {
    const ref = JSON.stringify(dynamics(game, VA, VB, a0, b0, 50));
    const exact = [L.OFF, { kind: 'reserve', r: 0 }, { kind: 'phantom', r: 0 }]
      .map(mod => JSON.stringify(L.labDynamics(lab, VA, VB, mod, a0, b0, 50)) === ref);
    const brExact = [0.01, 0.7, 3].every(x =>
      L.bestResponse(lab, 'A', VA, x).bid === bestResponseA(game, VA, x).bid &&
      L.bestResponse(lab, 'B', VB, x).bid === bestResponseB(game, VB, x).bid);
    // The general code path (payment formulas with r = 0, extra zero-bid options).
    const ga = dynamics(game, VA, VB, a0, b0, 50).path;
    const gb = L.labDynamics(lab, VA, VB, { kind: 'none', general: true }, a0, b0, 50).path;
    let rel = ga.length === gb.length ? 0 : Infinity;
    ga.forEach((p, i) => { rel = Math.max(rel, Math.abs(p.a - gb[i].a) / p.a, Math.abs(p.b - gb[i].b) / p.b); });
    // Modifications with tiny r stay on the original path.
    let tiny = 0;
    for (const kind of ['reserve', 'phantom']) {
      const gt = L.labDynamics(lab, VA, VB, { kind, r: 1e-13 }, a0, b0, 12).path;
      gt.forEach((p, i) => { tiny = Math.max(tiny, Math.abs(p.a - ga[i].a) / ga[i].a, Math.abs(p.b - ga[i].b) / ga[i].b); });
    }
    rows.push({ exact, brExact, rel, tiny });
  }
  out.push({ name, pieceErr, rows });
}
console.log(JSON.stringify(out));
"""  # noqa: E501

DIRECT_CHECK = SETUP + """
const { gen, T, lab } = labFor('city', 20);
const rand = mulberry32(31);
let worst = 0, cases = 0;
const regimes = new Set();
for (const mod of [L.OFF, { kind: 'reserve', r: 0.6 }, { kind: 'phantom', r: 1.7 }, { kind: 'phantom', r: 25 }]) {
  for (let i = 0; i < 25; i += 1) {
    const pick = () => (rand() < 0.25 ? 0.5 * rand() : (rand() < 0.1 ? 0 : 4 * rand()));
    let a = pick(), b = pick();
    if (mod.kind === 'none' && a + b === 0) a = 1;
    const e = L.evaluateBids(lab, 40, 30, a, b, mod);
    regimes.add(`${mod.kind}:${e.regime}`);
    if (e.regime === 'noAd') { if (L.runDirectMod(gen, 1, T, a, b, mod) !== null) worst = Infinity; continue; }
    let pa = 0, pb = 0, payA = 0, payB = 0;
    for (let s = 1; s <= lab.n; s += 1) {
      const d = L.runDirectMod(gen, s, T, a, b, mod);
      pa += d.state.visited[0] ? 1 : 0; pb += d.state.visited[1] ? 1 : 0; payA += d.paidA; payB += d.paidB;
    }
    const n = lab.n;
    worst = Math.max(worst, Math.abs(e.PA - pa / n), Math.abs(e.PB - pb / n), Math.abs(e.payA - payA / n), Math.abs(e.payB - payB / n));
    cases += 1;
  }
}
console.log(JSON.stringify({ worst, cases, regimes: [...regimes] }));
"""  # noqa: E501

MECHANISM_CHECK = SETUP + """
const out = {};
for (const name of ['city', 'melody']) {
  const { lab } = labFor(name, 40);
  const rand = mulberry32(5);
  let vcgGain = -Infinity, postGain = -Infinity, envelopeMiss = 0, oneOutGain = -Infinity;
  for (let t = 0; t < 12; t += 1) {
    const VA = 1 + 80 * rand(), VB = 1 + 80 * rand();
    const truthA = L.vcgExAnte(lab, VA, VB), truthP = L.vcgExPost(lab, VA, VB);
    for (let k = 0; k < 12; k += 1) {
      const z = 120 * rand();
      const lie = L.vcgExAnte(lab, z, VB), lieP = L.vcgExPost(lab, z, VB);
      vcgGain = Math.max(vcgGain, (VA * lie.PA - lie.payA) - truthA.uA);
      postGain = Math.max(postGain, (VA * lieP.PA - lieP.payA) - truthP.uA);
      const lieB = L.vcgExAnte(lab, VA, z);
      vcgGain = Math.max(vcgGain, (VB * lieB.PB - lieB.payB) - truthA.uB);
    }
    // Upper-envelope intervals agree with brute-force best pieces.
    const c = Array.from(lab.pieces.PA, x => VA * x), s = Array.from(lab.pieces.KA);
    const iv = L.envelopeIntervals(c, s);
    for (let k = 0; k < 10; k += 1) {
      const b = 10 ** (-3 + 5 * rand());
      let best = -Infinity;
      for (let j = 0; j < lab.m; j += 1) best = Math.max(best, c[j] - s[j] * b);
      const hit = iv.some((I, j) => I && b >= I[0] - 1e-12 && b <= I[1] + 1e-12 && Math.abs(c[j] - s[j] * b - best) < 1e-9);
      if (!hit) envelopeMiss += 1;
    }
    // One-out equilibrium: A at 0 cannot gain against b just above the threshold.
    const eq = L.originalEquilibria(lab, VA, VB);
    const b = eq.oneOut.Bwins.bAtLeast * 1.0001 + 1e-9;
    for (let j = 0; j < lab.m; j += 1) oneOutGain = Math.max(oneOutGain, VA * lab.pieces.PA[j] - b * lab.pieces.KA[j] - VA * lab.pieces.PA[0]);
  }
  // Proportional Myerson: exact regret matches a brute-force report grid, and the
  // monotone-chain version is exactly truthful.
  const VA = 37, VB = 23;
  const pm = L.proportionalMyerson(lab, VA, VB), pmm = L.proportionalMyerson(lab, VA, VB, { monotone: true });
  const chain = L.monotoneChain(lab).map;
  let chainMonotone = true;
  for (let j = 1; j < lab.m; j += 1) {
    if (lab.pieces.PA[chain[j]] < lab.pieces.PA[chain[j - 1]] || lab.pieces.PB[chain[j]] > lab.pieces.PB[chain[j - 1]]) chainMonotone = false;
  }
  const P = lab.pieces;
  const xA = z => P.PA[L.pieceIndex(lab, z / (z + VB))];
  // Independent exact integral of x_A over [0, z], piece by piece.
  const integral = z => {
    let total = 0;
    for (let j = 0; j < lab.m; j += 1) {
      const zl = VB * P.lo[j] / (1 - P.lo[j]), zh = P.hi[j] >= 1 ? Infinity : VB * P.hi[j] / (1 - P.hi[j]);
      total += P.PA[j] * Math.max(0, Math.min(z, zh) - zl);
    }
    return total;
  };
  const utility = z => VA * xA(z) - (z * xA(z) - integral(z));
  let bruteBest = -Infinity;
  for (let i = 1; i <= 3000; i += 1) bruteBest = Math.max(bruteBest, utility(400 * i / 3000));
  const truthErr = Math.abs(utility(VA) - pm.uA);
  const budget = L.budgetShareSup(lab, 40, 10);
  out[name] = { vcgGain, postGain, envelopeMiss, oneOutGain, regret: pm.regret, regretMono: pmm.regret, chainMonotone,
    bruteGain: bruteBest - utility(VA), truthErr, budgetAttained: budget.attained, budgetPiece: budget.piece, top: lab.m - 1,
    payNonneg: Math.min(pm.payA, pm.payB, pmm.payA, pmm.payB), irA: pm.uA, irB: pm.uB };
}
console.log(JSON.stringify(out));
"""  # noqa: E501


@unittest.skipUnless(NODE, "Node.js is required for browser checks")
class MechanismLabTests(unittest.TestCase):
    def run_node(self, source: str) -> str:
        return subprocess.run(
            [NODE, "--input-type=module"],
            input=source,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_modification_off_recovers_bid_game_exactly(self):
        for world in json.loads(self.run_node(OFF_CHECK)):
            with self.subTest(world=world["name"]):
                self.assertLess(world["pieceErr"], 1e-9)
                for row in world["rows"]:
                    self.assertEqual(row["exact"], [True, True, True])
                    self.assertTrue(row["brExact"])
                    self.assertLess(row["rel"], 1e-9)
                    self.assertLess(row["tiny"], 1e-6)

    def test_piecewise_formulas_equal_direct_generations(self):
        result = json.loads(self.run_node(DIRECT_CHECK))
        self.assertGreater(result["cases"], 60)
        self.assertLess(result["worst"], 1e-9)
        for regime in ("reserve:contest", "phantom:contest", "none:contest"):
            self.assertIn(regime, result["regimes"])

    def test_phantom_step_is_reference_auction_at_shifted_bids(self):
        """Phantom bid r on the midpoint distribution == reference at (a+r/2, b+r/2).

        The output token matches auction.py; each real bidder pays the reference
        critical bid minus r/2, floored at zero, which is its own critical bid.
        """
        rng = np.random.default_rng(1014)
        cases = []
        for _ in range(300):
            n = int(rng.integers(2, 7))
            p_a, p_b = rng.dirichlet(np.full(n, 0.7)), rng.dirichlet(np.full(n, 0.7))
            a, b, r = (float(x) for x in rng.choice([0.0, 0.2, 1.0, 3.0], 3))
            r = r or 0.5
            tape = RandomTape.draw(rng)
            cases.append({"pA": p_a.tolist(), "pB": p_b.tolist(), "a": a, "b": b, "r": r,
                          "tape": tape.__dict__})
        script = (
            "import { resolveStep } from './walkthrough/mechanism_lab.mjs';"
            f"const cases = {json.dumps(cases)};"
            "console.log(JSON.stringify(cases.map(c => resolveStep(c.pA, c.pB, c.a, c.b, c.tape,"
            " { kind: 'phantom', r: c.r }))));"
        )
        for case, got in zip(cases, json.loads(self.run_node(script))):
            auction = TwoBidderAuction(case["pA"], case["pB"])
            r = case["r"]
            a, b = case["a"], case["b"]
            ref = auction.resolve(a + r / 2, b + r / 2, RandomTape(**case["tape"]))
            self.assertEqual(got["option"], ref.token_id)
            self.assertEqual(got["branch"], ref.branch)
            want_a = max(0.0, ref.payment_a - r / 2) if ref.branch == "A" else 0.0
            want_b = max(0.0, ref.payment_b - r / 2) if ref.branch == "B" else 0.0
            self.assertAlmostEqual(got["payment_a"], want_a, places=10)
            self.assertAlmostEqual(got["payment_b"], want_b, places=10)
            self.assertLessEqual(got["payment_a"], a + 1e-12)
            self.assertLessEqual(got["payment_b"], b + 1e-12)
            midpoint = (auction.p_a + auction.p_b) / 2
            mix = (a * auction.p_a + b * auction.p_b + r * midpoint) / (a + b + r)
            np.testing.assert_allclose(mix, auction.mixture(a + r / 2, b + r / 2), atol=1e-12)

    def test_step_modifications_are_monotone_with_critical_prices(self):
        """Raising your own bid only switches a step to you; you pay the switch point."""
        script = """
import { resolveStep } from './walkthrough/mechanism_lab.mjs';
import { mulberry32 } from './walkthrough/atlas_core.mjs';
const rand = mulberry32(77);
let violations = 0, priceErr = 0, checked = 0;
for (let t = 0; t < 300; t += 1) {
  const pA = [rand(), rand(), rand()], pB = [rand(), rand(), rand()];
  const sA = pA.reduce((x, y) => x + y), sB = pB.reduce((x, y) => x + y);
  const tape = { branch: 0.999, token_a: rand(), token_b: rand(), contest: 0.02 + 0.96 * rand() };
  const mod = t % 2 ? { kind: 'reserve', r: 0.4 } : { kind: 'phantom', r: 0.9 };
  const b = 0.05 + 2 * rand();
  const grid = Array.from({ length: 2001 }, (_, i) => 3 * i / 2000);
  let won = false, threshold = null;
  for (const a of grid) {
    const out = resolveStep(pA.map(x => x / sA), pB.map(x => x / sB), a, b, tape, mod);
    if (out === null || out.branch === 'shared') { won = false; continue; }
    const w = out.branch === 'A';
    if (won && !w) violations += 1;
    if (w && !won) threshold = a;
    if (w) { checked += 1; if (out.payment_a > a + 1e-12) violations += 1; if (threshold !== null) priceErr = Math.max(priceErr, Math.abs(out.payment_a - threshold) > 3 / 2000 + 1e-9 ? 1 : 0); }
    won = w;
  }
}
console.log(JSON.stringify({ violations, priceErr, checked }));
"""  # noqa: E501
        result = json.loads(self.run_node(script))
        self.assertGreater(result["checked"], 1000)
        self.assertEqual(result["violations"], 0)
        self.assertEqual(result["priceErr"], 0)

    def test_outcome_mechanism_incentives(self):
        for name, row in json.loads(self.run_node(MECHANISM_CHECK)).items():
            with self.subTest(world=name):
                self.assertLessEqual(row["vcgGain"], 1e-9)
                self.assertLessEqual(row["postGain"], 1e-9)
                self.assertEqual(row["envelopeMiss"], 0)
                self.assertLessEqual(row["oneOutGain"], 1e-9)
                self.assertLess(row["regretMono"]["A"], 1e-9)
                self.assertLess(row["regretMono"]["B"], 1e-9)
                self.assertTrue(row["chainMonotone"])
                # Exact regret is a supremum over reports; the grid cannot beat it.
                self.assertLess(row["truthErr"], 1e-9)
                self.assertLessEqual(row["bruteGain"], row["regret"]["A"] + 1e-9)
                self.assertFalse(row["budgetAttained"])
                self.assertEqual(row["budgetPiece"], row["top"])
                self.assertGreaterEqual(row["payNonneg"], -1e-9)
                self.assertGreaterEqual(min(row["irA"], row["irB"]), -1e-9)


if __name__ == "__main__":
    unittest.main()

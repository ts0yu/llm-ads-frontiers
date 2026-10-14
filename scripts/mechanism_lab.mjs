// Outcome-level bidding: original cycle vs modified mechanisms, on the
// synthetic playground worlds. All numbers are exact for the sampled seeds.
// Usage: node scripts/mechanism_lab.mjs [seeds] [rounds] > results/mechanism_lab.json
import { buildAtlas } from '../walkthrough/atlas_core.mjs';
import { cityWalk, melody, blueShare, PITCHES, REST } from '../walkthrough/playground_worlds.mjs';
import * as L from '../walkthrough/mechanism_lab.mjs';

const seeds = Number(process.argv[2] || 1000);
const rounds = Number(process.argv[3] || 200);
const worlds = [
  {
    name: 'city', gen: cityWalk(), T: 80,
    A: 'walk passes Brindlewick Coffee', B: 'walk passes Quillmere Books',
    testA: l => l.state.visited[0], testB: l => l.state.visited[1],
    extra: { 'reaches the venue': l => l.stop === 'done' },
  },
  {
    name: 'melody', gen: melody(), T: 32,
    A: 'blue share < 15% (sounds major)', B: 'blue share > 30% (sounds blue)',
    testA: l => blueShare(l.state.notes) < 0.15, testB: l => blueShare(l.state.notes) > 0.3,
    extra: { 'ends on C': l => { const s = l.state.notes.filter(n => n !== REST); return s.length > 0 && PITCHES[s.at(-1)] % 12 === 0; } },
  },
];
const valuePairs = [[40, 40], [40, 20], [20, 40], [100, 40]];
const starts = [[1, 1], [0.1, 3], [3, 0.1], [10, 10]];
const round = (x, d = 6) => (x === null || x === undefined ? x : Number(x.toPrecision(d)));

function dynamicsRow(lab, VA, VB, label, mod) {
  const lambdaRef = VA / (VA + VB);
  const runs = starts.map(([a0, b0], i) => {
    const d = L.labDynamics(lab, VA, VB, mod, a0, b0, rounds);
    const s = L.summarizePath(lab, VA, VB, mod, d);
    return {
      start: [a0, b0], status: s.status, cycleLength: s.cycleLength,
      medianGrowth: round(s.medianGrowth, 4), dropOuts: s.dropOuts, maxBid: round(s.maxBid),
      band: Object.fromEntries(Object.entries(s.band).map(([k, v]) => [k, round(v, 5)])),
      final: Object.fromEntries(Object.entries(s.final).map(([k, v]) => [k, typeof v === 'number' ? round(v) : v])),
      regret: { A: round(s.regret.A), B: round(s.regret.B) },
      average: Object.fromEntries(Object.entries(s.average).map(([k, v]) => [k, round(v)])),
      finalOutputChangedVsRef: s.final.lambda === null ? null : round(L.outputChangeRate(lab, s.final.lambda, lambdaRef)),
      trace: i === 0 ? d.path.slice(0, 16).map(p => [round(p.a, 4), round(p.b, 4), p.lambda === null ? null : round(p.lambda, 4)]) : undefined,
    };
  });
  return { label, mod: { ...mod }, runs };
}

function outcomeRow(lab, m, lambdaRef) {
  const out = { mechanism: m.mechanism };
  for (const k of ['lambda', 'implementedLambda', 'PA', 'PB', 'payA', 'payB', 'uA', 'uB', 'revenue', 'welfare']) if (m[k] !== undefined) out[k] = round(m[k]);
  if (m.regret) out.regret = { A: round(m.regret.A), B: round(m.regret.B) };
  if (m.extras) out.extras = Object.fromEntries(Object.entries(m.extras).map(([k, v]) => [k, round(v)]));
  const lam = m.implementedLambda ?? m.lambda;
  if (lam !== undefined && lam !== null) {
    out.lambdaDistanceFromRef = round(Math.abs(lam - lambdaRef));
    out.outputChangedVsRef = round(L.outputChangeRate(lab, lam, lambdaRef));
  }
  if (m.outputChangedVsCompare !== undefined) {
    out.outputChangedVsRef = round(m.outputChangedVsCompare);
    out.lambdaDistanceFromRef = round(m.meanLambdaDistance);
  }
  return out;
}

const report = {
  mode: 'synthetic_generators', seeds, rounds, prng: 'mulberry32', starts,
  note: 'Exact for the sampled seeds (atlas step functions). Simulated credits. Exploratory; no truthfulness claim about LLM ads.',
  worlds: {},
};
for (const w of worlds) {
  const atlases = [];
  for (let s = 1; s <= seeds; s += 1) atlases.push(buildAtlas(w.gen, s, w.T));
  const lab = L.prepareLab(atlases, w.testA, w.testB, w.extra);
  const entry = { outcomes: { A: w.A, B: w.B }, pieces: lab.m, monotonicity: L.monotonicity(lab), chainRemappedMeasure: round(L.monotoneChain(lab).remappedMeasure), values: [] };
  for (const [VA, VB] of valuePairs) {
    const scale = (VA + VB) / 80;
    const lambdaRef = VA / (VA + VB);
    const eq = L.originalEquilibria(lab, VA, VB);
    const dyn = [
      dynamicsRow(lab, VA, VB, 'original (bid_game.mjs)', L.OFF),
      dynamicsRow(lab, VA, VB, 'original + inertia', { kind: 'none', inertia: true }),
    ];
    for (const r of [0.5, 2, 5]) {
      dyn.push(dynamicsRow(lab, VA, VB, `reserve r=${round(r * scale, 3)}`, { kind: 'reserve', r: r * scale }));
      dyn.push(dynamicsRow(lab, VA, VB, `reserve r=${round(r * scale, 3)} + inertia`, { kind: 'reserve', r: r * scale, inertia: true }));
    }
    for (const r of [1, 5, 10, 20, 50]) {
      dyn.push(dynamicsRow(lab, VA, VB, `phantom r=${round(r * scale, 3)}`, { kind: 'phantom', r: r * scale }));
      dyn.push(dynamicsRow(lab, VA, VB, `phantom r=${round(r * scale, 3)} + inertia`, { kind: 'phantom', r: r * scale, inertia: true }));
    }
    const mechanisms = [
      L.naiveValueBids(lab, VA, VB),
      L.proportionalMyerson(lab, VA, VB),
      L.proportionalMyerson(lab, VA, VB, { monotone: true }),
      L.vcgExAnte(lab, VA, VB, { ref: lambdaRef }),
      L.vcgExPost(lab, VA, VB, { ref: lambdaRef, compareLambda: lambdaRef }),
    ].map(m => outcomeRow(lab, m, lambdaRef));
    const budget = L.budgetShareSup(lab, VA, 10);
    entry.values.push({
      VA, VB, lambdaRef,
      originalEquilibria: {
        interiorCount: eq.interior.length,
        interior: eq.interior.slice(0, 5),
        oneOut: eq.oneOut,
      },
      budgetShares: { S: 10, bestPieceForA: budget.piece, supUtilityA: round(budget.sup), attained: budget.attained, atLambda: budget.lambda, topPiece: lab.m - 1 },
      dynamics: dyn,
      outcomeMechanisms: mechanisms,
    });
    process.stderr.write(`${w.name} ${VA},${VB} done\n`);
  }
  report.worlds[w.name] = entry;
}
console.log(JSON.stringify(report));

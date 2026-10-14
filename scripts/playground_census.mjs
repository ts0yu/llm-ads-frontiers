// Census of exact atlases in the synthetic playground worlds.
// Usage: node scripts/playground_census.mjs [seeds] > results/playground_census.json
import { buildAtlas, outcomeProfile } from '../walkthrough/atlas_core.mjs';
import { cityWalk, melody, blueShare, PITCHES, REST } from '../walkthrough/playground_worlds.mjs';

const seeds = Number(process.argv[2] || 2000);
const worlds = [
  ['city', cityWalk(), 80, {
    'passes Rosso Roasters': l => l.state.visited[0],
    'passes Inkwell Books': l => l.state.visited[1],
    'reaches the venue': l => l.stop === 'done',
  }],
  ['melody', melody(), 32, {
    'blue share > 30%': l => blueShare(l.state.notes) > 0.3,
    'blue share < 15%': l => blueShare(l.state.notes) < 0.15,
    'ends on C': l => { const s = l.state.notes.filter(n => n !== REST); return s.length > 0 && PITCHES[s.at(-1)] % 12 === 0; },
  }],
];
const report = { mode: 'synthetic_generators', seeds, prng: 'mulberry32', worlds: {} };
for (const [name, gen, T, outcomes] of worlds) {
  const leaves = [], contestedShare = [];
  const stats = Object.fromEntries(Object.keys(outcomes).map(k => [k, { constant: 0, increasing: 0, decreasing: 0, 'non-monotone': 0, switches: [] }]));
  for (let s = 1; s <= seeds; s += 1) {
    const atlas = buildAtlas(gen, s, T);
    leaves.push(atlas.leaves.length);
    const mid = atlas.leaves.find(l => 0.5 > l.lo && 0.5 <= l.hi);
    contestedShare.push(mid.steps.filter(x => x.branch !== 'shared').length / mid.steps.length);
    for (const [k, test] of Object.entries(outcomes)) {
      const p = outcomeProfile(atlas, test);
      stats[k][p.shape] += 1;
      stats[k].switches.push(p.switches);
    }
  }
  const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
  const sorted = [...leaves].sort((a, b) => a - b);
  report.worlds[name] = {
    max_steps: T,
    bound: 1 + T,
    leaves: { mean: mean(leaves), median: sorted[Math.floor(sorted.length / 2)], max: sorted.at(-1), min: sorted[0] },
    contested_share_at_lambda_half: mean(contestedShare),
    outcomes: Object.fromEntries(Object.entries(stats).map(([k, v]) => [k, {
      constant: v.constant, increasing: v.increasing, decreasing: v.decreasing, non_monotone: v['non-monotone'],
      non_monotone_rate: v['non-monotone'] / seeds, mean_switches: mean(v.switches), max_switches: Math.max(...v.switches),
    }])),
  };
}
console.log(JSON.stringify(report, null, 2));

// Census of three-sponsor simplex atlases in the synthetic city walk.
// Usage: node scripts/simplex_census.mjs [seeds] > results/simplex_census.json
import { buildSimplexAtlas } from '../walkthrough/atlas_core_n.mjs';
import { CITY3, cityWalk } from '../walkthrough/playground_worlds.mjs';

const seeds = Number(process.argv[2] || 100);
const gen = cityWalk(CITY3);
const rows = [];
for (const T of [20, 40, 80]) {
  const cells = [], distinct = [], shares = [[], [], []];
  for (let s = 1; s <= seeds; s += 1) {
    const atlas = buildSimplexAtlas(gen, s, T);
    cells.push(atlas.cells.length);
    distinct.push(new Set(atlas.cells.map(c => c.options.join(','))).size);
    for (let i = 0; i < 3; i += 1) {
      shares[i].push(atlas.cells.filter(c => c.state.visited[i]).reduce((a, c) => a + c.area, 0) / 0.5);
    }
  }
  const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
  rows.push({
    max_steps: T, mean_cells: mean(cells), max_cells: Math.max(...cells), mean_distinct_routes: mean(distinct),
    mean_share_passing: CITY3.sponsors.map((sp, i) => ({ sponsor: sp.name, share: mean(shares[i]) })),
  });
}
console.log(JSON.stringify({ mode: 'synthetic_generators', seeds, world: 'city walk, three sponsors', rows }, null, 2));

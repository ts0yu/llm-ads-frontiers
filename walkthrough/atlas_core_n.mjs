// Exact three-bidder atlases over the bid simplex, in the browser.
//
// N-bidder exponential race (MOONSHOTS.md section 3, token_auction_lab/auction_n.py):
// the common mass min_i p_i is free; in a contested step each bidder draws
// E_i ~ Exp(1), the winner is argmin E_i / b_i and pays E_w * max_{j != w} b_j / E_j.
// For a fixed seed, the bid shares lambda (a point in the simplex) select one
// convex polygon cell per output. Cells are clipped with exact half-planes.
import { mulberry32 } from './atlas_core.mjs';

export function drawRaceTapes(seed, steps, n) {
  const rand = mulberry32(seed ^ 0x9e3779b9);
  const tapes = [];
  for (let k = 0; k < steps; k += 1) {
    const tape = { branch: rand(), shared: rand(), tokens: [], E: [] };
    for (let i = 0; i < n; i += 1) tape.tokens.push(rand());
    for (let i = 0; i < n; i += 1) tape.E.push(-Math.log(1 - rand()));  // 1 - U in (0, 1]
    tapes.push(tape);
  }
  return tapes;
}

function pickWeighted(weights, u) {
  let total = 0;
  for (const w of weights) total += w;
  let run = 0;
  for (let i = 0; i < weights.length; i += 1) {
    run += weights[i] / total;
    if (u < run) return i;
  }
  for (let i = weights.length - 1; i >= 0; i -= 1) if (weights[i] > 0) return i;
  throw new Error('No positive mass.');
}

// Bid-independent part of a step.
export function prepareRace(ps, tape) {
  const n = ps.length, k = ps[0].length;
  const common = new Array(k);
  let C = 0;
  for (let t = 0; t < k; t += 1) {
    let m = Infinity;
    for (let i = 0; i < n; i += 1) m = Math.min(m, ps[i][t]);
    common[t] = m;
    C += m;
  }
  const d = Math.max(0, 1 - C);
  if (tape.branch < C || d <= 1e-15) return { branch: 'shared', option: pickWeighted(common, tape.shared), sharedMass: C };
  const candidates = ps.map((p, i) => pickWeighted(p.map((x, t) => Math.max(0, x - common[t])), tape.tokens[i]));
  return { branch: 'contested', candidates, E: tape.E, sharedMass: C };
}

export function raceWinner(E, bids) {
  let best = -1, score = Infinity;
  for (let i = 0; i < bids.length; i += 1) {
    if (bids[i] > 0 && E[i] / bids[i] < score) { score = E[i] / bids[i]; best = i; }
  }
  return best;
}

export function criticalBid(E, bids, w) {
  let m = 0;
  for (let j = 0; j < bids.length; j += 1) if (j !== w && bids[j] > 0) m = Math.max(m, bids[j] / E[j]);
  return E[w] * m;
}

// ---- polygons in (x, y) = (lambda_0, lambda_1); lambda_2 = 1 - x - y ----
const lam = (p, i) => (i === 0 ? p[0] : i === 1 ? p[1] : 1 - p[0] - p[1]);

export function area(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i += 1) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}

export function centroid(poly) {
  let cx = 0, cy = 0, a = 0;
  for (let i = 0; i < poly.length; i += 1) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    const c = x1 * y2 - x2 * y1;
    a += c; cx += (x1 + x2) * c; cy += (y1 + y2) * c;
  }
  if (Math.abs(a) < 1e-18) return poly.reduce((s, p) => [s[0] + p[0] / poly.length, s[1] + p[1] / poly.length], [0, 0]);
  return [cx / (3 * a), cy / (3 * a)];
}

// Keep the part of `poly` where f(p) >= 0, f affine.
function clip(poly, f) {
  const out = [];
  for (let i = 0; i < poly.length; i += 1) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    const fp = f(p), fq = f(q);
    if (fp >= 0) out.push(p);
    if ((fp >= 0) !== (fq >= 0)) {
      const t = fp / (fp - fq);
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
    }
  }
  return out;
}

// Region of the simplex where bidder i wins the race with exponentials E.
export function winRegion(poly, E, i, n = 3) {
  let piece = poly;
  for (let j = 0; j < n && piece.length; j += 1) {
    if (j === i) continue;
    // lambda_i / E_i >= lambda_j / E_j  <=>  lambda_i * E_j - lambda_j * E_i >= 0
    piece = clip(piece, p => lam(p, i) * E[j] - lam(p, j) * E[i]);
  }
  return piece;
}

export const SIMPLEX = [[1, 0], [0, 1], [0, 0]];  // vertices: all-A, all-B, all-C

export function buildSimplexAtlas(generator, seed, maxSteps, { maxCells = 4000, minArea = 1e-12 } = {}) {
  const n = 3;
  const tapes = drawRaceTapes(seed, maxSteps, n);
  const cells = [];
  let slivers = 0, truncated = false;
  const stack = [{ poly: SIMPLEX, state: generator.init(), steps: [], options: [] }];
  while (stack.length) {
    const cell = stack.pop();
    let stop = 'steps';
    for (let k = cell.steps.length; k < maxSteps; k += 1) {
      if (generator.done(cell.state)) { stop = 'done'; break; }
      const step = prepareRace(generator.dists(cell.state), tapes[k]);
      if (step.branch === 'shared') {
        cell.steps.push({ step: k, branch: 'shared', option: step.option, shared_mass: step.sharedMass });
        cell.options.push(step.option);
        cell.state = generator.apply(cell.state, step.option);
        continue;
      }
      const pieces = [];
      for (let i = 0; i < n; i += 1) {
        const poly = winRegion(cell.poly, step.E, i, n);
        const a = poly.length >= 3 ? area(poly) : 0;
        if (a > minArea) pieces.push({ i, poly, a });
        else if (poly.length >= 3) slivers += 1;
      }
      pieces.sort((x, y) => y.a - x.a);
      if (!pieces.length) throw new Error('A contested step left no cell with positive area.');
      const [keep, ...rest] = pieces;
      for (const piece of rest) {
        if (cells.length + stack.length + 1 >= maxCells) { truncated = true; break; }
        const option = step.candidates[piece.i];
        stack.push({
          poly: piece.poly, state: generator.apply(cell.state, option),
          steps: [...cell.steps, { step: k, branch: 'contested', winner: piece.i, E: step.E, option, shared_mass: step.sharedMass }],
          options: [...cell.options, option],
        });
      }
      const option = step.candidates[keep.i];
      cell.poly = keep.poly;
      cell.steps.push({ step: k, branch: 'contested', winner: keep.i, E: step.E, option, shared_mass: step.sharedMass });
      cell.options.push(option);
      cell.state = generator.apply(cell.state, option);
    }
    if (stop === 'steps' && generator.done(cell.state)) stop = 'done';
    cells.push({ ...cell, stop, area: area(cell.poly) });
  }
  return { seed, maxSteps, cells, slivers, truncated };
}

export function runDirectRace(generator, seed, maxSteps, bids) {
  const tapes = drawRaceTapes(seed, maxSteps, bids.length);
  let state = generator.init();
  const options = [], paid = bids.map(() => 0);
  for (let k = 0; k < maxSteps && !generator.done(state); k += 1) {
    const step = prepareRace(generator.dists(state), tapes[k]);
    let option;
    if (step.branch === 'shared') option = step.option;
    else {
      const w = raceWinner(step.E, bids);
      option = step.candidates[w];
      paid[w] += criticalBid(step.E, bids, w);
    }
    options.push(option);
    state = generator.apply(state, option);
  }
  return { options, state, paid };
}

export function cellPayments(cell, bids) {
  const paid = bids.map(() => 0);
  for (const s of cell.steps) if (s.branch === 'contested') paid[s.winner] += criticalBid(s.E, bids, s.winner);
  return paid;
}

export function pointInPolygon(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function cellAt(atlas, p) {
  return atlas.cells.find(c => pointInPolygon(p, c.poly)) ?? atlas.cells.reduce((best, c) => {
    const [cx, cy] = centroid(c.poly);
    const d = (cx - p[0]) ** 2 + (cy - p[1]) ** 2;
    return !best || d < best.d ? { c, d } : best;
  }, null).c;
}

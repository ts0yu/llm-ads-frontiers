// Mechanism lab: can outcome-level bidding be made stable?
//
// Builds on bid_game.mjs. Sponsors value an OUTCOME of the whole generation.
// Under the reference token auction, alternating best responses cycle (see
// docs/OUTCOME_BIDDING.md). This module computes, exactly for a sample of seeds:
//
//   1. Token-level modifications that keep the per-step contest and its
//      threshold payments, with a parameter r >= 0 that recovers bid_game.mjs
//      exactly at r = 0:
//        'reserve'  each bidder must bid >= r to win a contested step; a winner
//                   pays max(r, critical bid). If neither bid reaches r, no
//                   sponsored generation runs (outcome values `noAd`, no charge).
//        'phantom'  the platform joins as a third bidder with fixed bid r whose
//                   distribution is the midpoint (pA+pB)/2. The common mass is
//                   unchanged and the output stays in the two-sponsor mixture
//                   family at lambda' = (a + r/2) / (a + b + r).
//   2. Outcome-level mechanisms that use the atlas directly: sponsors report a
//      value for their outcome and the platform chooses lambda:
//        vcgExAnte            lambda maximizing V_A P_A + V_B P_B, Clarke payments
//        vcgExPost            per seed, the atlas leaf maximizing V_A x_A + V_B x_B
//        proportionalMyerson  lambda = V_A/(V_A+V_B) (the original mixture at
//                             value bids), Myerson payments for P_i
//   3. Exact equilibrium analysis of the unmodified game (interior and
//      one-sponsor-out pure equilibria) and the fixed-budget variant.
//
// Everything here is synthetic-generator analysis in simulated credits. Seeds are
// a finite sample of tapes; the mechanisms that draw the final seed from the same
// sample see these curves as exact probabilities.
import { TwoBidderAuction, drawTapes } from './atlas_core.mjs';
import { aggregate, bestResponseA as baseBRA, bestResponseB as baseBRB, dynamics as baseDynamics } from './bid_game.mjs';

const EPS = 1e-12;

function prefixSums(values, f) {
  const out = new Float64Array(values.length + 1);
  for (let i = 0; i < values.length; i += 1) out[i + 1] = out[i] + f(values[i]);
  return out;
}

// Number of sorted values strictly below x.
function countBelow(sorted, x) {
  let lo = 0, hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// ---------- Preparation: per-seed leaf summaries on a common lambda grid ----------

// `extra` maps names to further leaf tests (side outcomes reported, not valued).
export function prepareLab(atlases, testA, testB, extra = {}) {
  const base = aggregate(atlases, testA, testB);
  const cuts = new Set([0, 1]);
  for (const atlas of atlases) for (const leaf of atlas.leaves) { cuts.add(leaf.lo); cuts.add(leaf.hi); }
  const edges = [...cuts].sort((x, y) => x - y);
  const lo = [], hi = [];
  for (let i = 0; i + 1 < edges.length; i += 1) if (edges[i + 1] > edges[i]) { lo.push(edges[i]); hi.push(edges[i + 1]); }
  const m = lo.length;
  const n = atlases.length;
  const sum = () => new Float64Array(m + 1);
  const acc = { PA: sum(), PB: sum(), KA: sum(), KB: sum(), NA: sum(), NB: sum() };
  const extraAcc = Object.fromEntries(Object.keys(extra).map(k => [k, sum()]));
  const seeds = atlases.map(atlas => {
    let p = 0;
    const leaves = atlas.leaves.map((leaf, index) => {
      const uA = [], uB = [];
      for (const s of leaf.steps) {
        if (s.branch === 'A') uA.push(s.u);
        else if (s.branch === 'B') uB.push(s.u);
      }
      uA.sort((x, y) => x - y);
      uB.sort((x, y) => x - y);
      const p0 = p;
      while (p < m && (lo[p] + hi[p]) / 2 <= leaf.hi) p += 1;
      const s = {
        index, lo: leaf.lo, hi: leaf.hi, p0, p1: p,
        xa: testA(leaf) ? 1 : 0, xb: testB(leaf) ? 1 : 0,
        extra: Object.fromEntries(Object.entries(extra).map(([k, t]) => [k, t(leaf) ? 1 : 0])),
        uA, uB,
        GA: prefixSums(uA, u => u / (1 - u)),
        HB: prefixSums(uB, u => (1 - u) / u),
      };
      s.ka = s.GA[uA.length];
      s.kb = s.HB[uB.length];
      const add = (arr, v) => { arr[p0] += v; arr[p] -= v; };
      add(acc.PA, s.xa); add(acc.PB, s.xb); add(acc.KA, s.ka); add(acc.KB, s.kb);
      add(acc.NA, uA.length); add(acc.NB, uB.length);
      for (const k of Object.keys(extra)) add(extraAcc[k], s.extra[k]);
      return s;
    });
    return { seed: atlas.seed, leaves };
  });
  const integrate = arr => {
    const out = new Float64Array(m);
    let run = 0;
    for (let i = 0; i < m; i += 1) { run += arr[i]; out[i] = run / n; }
    return out;
  };
  const pieces = { lo, hi };
  for (const [k, arr] of Object.entries(acc)) pieces[k] = integrate(arr);
  const extras = Object.fromEntries(Object.entries(extraAcc).map(([k, arr]) => [k, integrate(arr)]));
  return { n, m, pieces, extras, seeds, base };
}

export function pieceIndex(lab, lambda) {
  const { lo, hi } = lab.pieces;
  let a = 0, b = lab.m - 1;
  while (a < b) {
    const mid = (a + b) >> 1;
    if (lambda > hi[mid]) a = mid + 1; else b = mid;
  }
  return lambda > lo[a] ? a : 0;
}

export function leafIndexAt(seed, lambda) {
  for (const leaf of seed.leaves) if (lambda > leaf.lo && lambda <= leaf.hi) return leaf.index;
  return 0;
}

// ---------- Token-level modifications ----------

export const OFF = Object.freeze({ kind: 'none' });

function isOff(mod) {
  return !mod || mod.kind === 'none' || !(mod.r > 0);
}

// Off and not asking for the general code path or inertia: use bid_game.mjs.
function delegates(mod) {
  return isOff(mod) && !mod?.general && !mod?.inertia;
}

// One step under a modification, from the same tape the reference uses.
// Returns null in the reserve's no-ad regime (no sponsored step runs).
export function resolveStep(pA, pB, a, b, tape, mod = OFF) {
  const step = new TwoBidderAuction(pA, pB).prepare(tape);
  const { regime, lambda } = regimeAt(a, b, mod);
  if (regime === 'noAd') return null;
  if (step.branch === 'shared') return { option: step.option, branch: 'shared', payment_a: 0, payment_b: 0 };
  const u = step.u, r = isOff(mod) ? 0 : mod.r;
  if (u < lambda) {
    let price;
    if (isOff(mod)) price = b * (u / (1 - u));
    else if (mod.kind === 'phantom') price = Math.max(0, (b + r / 2) * (u / (1 - u)) - r / 2);
    else price = regime === 'soleA' ? r : Math.max(r, b * (u / (1 - u)));
    return { option: step.a, branch: 'A', payment_a: price, payment_b: 0 };
  }
  let price;
  if (isOff(mod)) price = a * ((1 - u) / u);
  else if (mod.kind === 'phantom') price = Math.max(0, (a + r / 2) * ((1 - u) / u) - r / 2);
  else price = regime === 'soleB' ? r : Math.max(r, a * ((1 - u) / u));
  return { option: step.b, branch: 'B', payment_a: 0, payment_b: price };
}

// Direct generation at one bid pair under a modification (verification oracle).
export function runDirectMod(generator, seed, maxSteps, a, b, mod = OFF) {
  if (regimeAt(a, b, mod).regime === 'noAd') return null;
  const tapes = drawTapes(seed, maxSteps);
  let state = generator.init();
  const options = [];
  let paidA = 0, paidB = 0;
  for (let k = 0; k < maxSteps && !generator.done(state); k += 1) {
    const [pA, pB] = generator.dists(state);
    const out = resolveStep(pA, pB, a, b, tapes[k], mod);
    options.push(out.option);
    paidA += out.payment_a;
    paidB += out.payment_b;
    state = generator.apply(state, out.option);
  }
  return { options, state, paidA, paidB };
}

// Payment of one side on one leaf at a fixed rival bid, when both compete.
function leafPay(s, side, rival, r, kind) {
  if (side === 'A') {
    const g = s.GA, total = g[s.uA.length];
    if (kind === 'none') return rival * total;
    if (kind === 'reserve') {
      // max(r, b u/(1-u)): the reserve binds when u < r/(r+b).
      const k = countBelow(s.uA, r / (r + rival));
      return r * k + rival * (total - g[k]);
    }
    // max(0, (b + r/2) u/(1-u) - r/2): free when u < (r/2)/(b + r).
    const k = countBelow(s.uA, (r / 2) / (rival + r));
    return (rival + r / 2) * (total - g[k]) - (r / 2) * (s.uA.length - k);
  }
  const h = s.HB, total = h[s.uB.length];
  if (kind === 'none') return rival * total;
  if (kind === 'reserve') {
    // max(r, a (1-u)/u): the reserve binds when u > a/(a+r).
    const k = countBelow(s.uB, rival / (rival + r));
    return rival * h[k] + r * (s.uB.length - k);
  }
  // max(0, (a + r/2)(1-u)/u - r/2): free when u > (a + r/2)/(a + r).
  const k = countBelow(s.uB, (rival + r / 2) / (rival + r));
  return (rival + r / 2) * h[k] - (r / 2) * k;
}

function modParams(mod) {
  return isOff(mod) ? { r: 0, kind: 'none' } : { r: mod.r, kind: mod.kind };
}

// Expected payment of one side on every piece, at a fixed rival bid, when both
// sponsors compete (for 'reserve': both bids >= r). Exact for the sampled seeds.
export function payVector(lab, side, rival, mod = OFF) {
  const m = lab.m;
  const diff = new Float64Array(m + 1);
  const { r, kind } = modParams(mod);
  for (const seed of lab.seeds) {
    for (const s of seed.leaves) {
      const pay = leafPay(s, side, rival, r, kind);
      diff[s.p0] += pay;
      diff[s.p1] -= pay;
    }
  }
  const out = new Float64Array(m);
  let run = 0;
  for (let i = 0; i < m; i += 1) { run += diff[i]; out[i] = run / lab.n; }
  return out;
}

// The same expectation at one lambda only (one leaf per seed).
export function payAt(lab, side, rival, mod, lambda) {
  const { r, kind } = modParams(mod);
  let total = 0;
  for (const seed of lab.seeds) {
    const leaves = seed.leaves;
    let a = 0, b = leaves.length - 1;
    while (a < b) {
      const mid = (a + b) >> 1;
      if (lambda > leaves[mid].hi) a = mid + 1; else b = mid;
    }
    total += leafPay(leaves[a], side, rival, r, kind);
  }
  return total / lab.n;
}

// Effective lambda and regime at a bid pair.
export function regimeAt(a, b, mod = OFF) {
  if (isOff(mod)) return { regime: 'contest', lambda: a / (a + b) };
  const r = mod.r;
  if (mod.kind === 'phantom') return { regime: 'contest', lambda: (a + r / 2) / (a + b + r) };
  const inA = a >= r, inB = b >= r;
  if (inA && inB) return { regime: 'contest', lambda: a / (a + b) };
  if (inA) return { regime: 'soleA', lambda: 1 };
  if (inB) return { regime: 'soleB', lambda: 0 };
  return { regime: 'noAd', lambda: null };
}

// Expected outcome probabilities, payments, utilities, revenue and welfare.
export function evaluateBids(lab, VA, VB, a, b, mod = OFF) {
  const { regime, lambda } = regimeAt(a, b, mod);
  const P = lab.pieces;
  let PA, PB, payA = 0, payB = 0, j = null;
  if (regime === 'noAd') {
    PA = mod.noAd?.PA ?? 0; PB = mod.noAd?.PB ?? 0;
  } else if (regime === 'soleA') {
    j = lab.m - 1; PA = P.PA[j]; PB = P.PB[j]; payA = mod.r * P.NA[j];
  } else if (regime === 'soleB') {
    j = 0; PA = P.PA[j]; PB = P.PB[j]; payB = mod.r * P.NB[j];
  } else {
    j = lambda >= 1 ? lab.m - 1 : pieceIndex(lab, lambda);
    PA = P.PA[j]; PB = P.PB[j];
    if (isOff(mod)) { payA = b * P.KA[j]; payB = a * P.KB[j]; }
    else { const at = (P.lo[j] + P.hi[j]) / 2; payA = payAt(lab, 'A', b, mod, at); payB = payAt(lab, 'B', a, mod, at); }
  }
  const uA = VA * PA - payA, uB = VB * PB - payB;
  return { regime, lambda, piece: j, PA, PB, payA, payB, uA, uB, revenue: payA + payB, welfare: VA * PA + VB * PB };
}

// Best response of `side` with value V to the rival's bid. With the modification
// off this is bid_game.mjs's best response, unchanged.
//
// `current` (own bid now) is used only with `mod.inertia`: a sponsor whose
// current bid is already a best response keeps it. The general code path with
// r = 0 (`{kind: 'none', general: true}`) is the unmodified game.
export function bestResponse(lab, side, V, rival, mod = OFF, current = null) {
  if (delegates(mod)) {
    const br = side === 'A' ? baseBRA(lab.base, V, rival) : baseBRB(lab.base, V, rival);
    return { utility: br.utility, lambda: br.lambda, bid: br.bid, option: 'contest' };
  }
  const P = lab.pieces, m = lab.m;
  const r = isOff(mod) ? 0 : mod.r;
  const kind = isOff(mod) ? 'phantom' : mod.kind; // phantom with r = 0 is the original
  const own = side === 'A' ? P.PA : P.PB;
  if (r === 0 && rival === 0) {
    // Any positive bid takes every contested step for free.
    const j = side === 'A' ? m - 1 : 0;
    return { utility: V * own[j], lambda: side === 'A' ? 1 : 0, bid: 1e-9, option: 'contest', piece: j };
  }
  const options = [];
  // Feasible lambda range when competing, as a function of own bid >= floor.
  let lamMin, lamMax, toBid;
  if (kind === 'phantom') {
    if (side === 'A') { lamMin = (r / 2) / (rival + r); lamMax = 1; toBid = l => Math.max(0, (l * (rival + r) - r / 2) / (1 - l)); }
    else { lamMin = 0; lamMax = (rival + r / 2) / (rival + r); toBid = l => Math.max(0, (rival + r / 2) / l - rival - r); }
  } else if (rival >= r) {
    if (side === 'A') { lamMin = r / (r + rival); lamMax = 1; toBid = l => rival * l / (1 - l); }
    else { lamMin = 0; lamMax = rival / (rival + r); toBid = l => rival * (1 - l) / l; }
  }
  if (lamMin !== undefined) {
    const pay = payVector(lab, side, rival, mod);
    for (let j = 0; j < m; j += 1) {
      const lo = Math.max(P.lo[j], lamMin), hi = Math.min(P.hi[j], lamMax);
      if (!(hi > lo)) continue;
      const utility = V * own[j] - pay[j];
      // Midpoint of the feasible part (bid_game's convention). A's top piece
      // excludes lambda = 1, which needs b = 0.
      const lambda = (lo + hi) / 2;
      options.push({ utility, lambda, bid: toBid(lambda), option: 'contest', piece: j });
    }
    if (kind === 'phantom' && side === 'A') {
      // Bidding zero is allowed: lambda' = lamMin exactly.
      const j = pieceIndex(lab, lamMin);
      options.push({ utility: V * own[j] - pay[j], lambda: lamMin, bid: 0, option: 'zero', piece: j });
    }
    if (kind === 'phantom' && side === 'B') {
      const j = pieceIndex(lab, lamMax);
      options.push({ utility: V * own[j] - pay[j], lambda: lamMax, bid: 0, option: 'zero', piece: j });
    }
  }
  if (kind === 'reserve') {
    // Stay out (bid 0 < r): the rival is sole bidder, or nobody is.
    const out = rival >= r
      ? evaluateBids(lab, 0, 0, side === 'A' ? 0 : rival, side === 'A' ? rival : 0, mod)
      : evaluateBids(lab, 0, 0, 0, 0, mod);
    options.push({ utility: V * (side === 'A' ? out.PA : out.PB) - 0, lambda: out.lambda, bid: 0, option: 'out' });
    if (rival < r) {
      // Sole eligible bidder at bid r: wins every contested step at price r.
      const j = side === 'A' ? m - 1 : 0;
      const utility = V * own[j] - r * (side === 'A' ? P.NA[j] : P.NB[j]);
      options.push({ utility, lambda: side === 'A' ? 1 : 0, bid: r, option: 'sole', piece: j });
    }
  }
  let best = null;
  for (const o of options) if (!best || o.utility > best.utility + EPS) best = o;
  if (mod.inertia && current !== null) {
    const e = side === 'A' ? evaluateBids(lab, V, 0, current, rival, mod) : evaluateBids(lab, 0, V, rival, current, mod);
    const now = side === 'A' ? e.uA : e.uB;
    if (now >= best.utility - EPS) return { utility: now, lambda: e.lambda, bid: current, option: 'stay', piece: e.piece };
  }
  return best;
}

// Alternating best responses, as in bid_game.dynamics. Off: bid_game's result.
export function labDynamics(lab, VA, VB, mod = OFF, a0 = 1, b0 = 1, rounds = 60) {
  if (delegates(mod)) return baseDynamics(lab.base, VA, VB, a0, b0, rounds);
  let a = a0, b = b0;
  const path = [{ a, b, lambda: regimeAt(a, b, mod).lambda }];
  const seen = new Map();
  for (let r = 0; r < rounds; r += 1) {
    a = bestResponse(lab, 'A', VA, b, mod, a).bid;
    b = bestResponse(lab, 'B', VB, a, mod, b).bid;
    const e = evaluateBids(lab, VA, VB, a, b, mod);
    path.push({ a, b, lambda: e.lambda, uA: e.uA, uB: e.uB });
    const key = `${a.toPrecision(8)}|${b.toPrecision(8)}`;
    if (seen.has(key)) {
      const start = seen.get(key);
      return { path, status: start === r - 1 ? 'converged' : 'cycle', cycleLength: r - start };
    }
    seen.set(key, r);
  }
  return { path, status: 'no convergence within rounds' };
}

// Summary of a best-response path: escalation, drop-outs, time averages, and
// each sponsor's regret at the final bids (0 at an exact equilibrium).
export function summarizePath(lab, VA, VB, mod, d) {
  const pts = d.path.slice(1);
  const evals = pts.map(p => evaluateBids(lab, VA, VB, p.a, p.b, mod));
  const growth = [];
  for (let i = 1; i < d.path.length; i += 1) {
    const prev = Math.max(d.path[i - 1].a, d.path[i - 1].b), cur = Math.max(d.path[i].a, d.path[i].b);
    if (cur > prev && prev > 0) growth.push(cur / prev);
  }
  growth.sort((x, y) => x - y);
  const mean = xs => xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
  const last = pts.at(-1);
  const brA = bestResponse(lab, 'A', VA, last.b, mod), brB = bestResponse(lab, 'B', VB, last.a, mod);
  const now = evals.at(-1);
  const levels = pts.map(p => Math.max(p.a, p.b));
  return {
    status: d.status,
    rounds: pts.length,
    cycleLength: d.cycleLength ?? null,
    medianGrowth: growth.length ? growth[Math.floor(growth.length / 2)] : 1,
    dropOuts: evals.filter(e => e.lambda === null || e.lambda > 0.995 || e.lambda < 0.005).length,
    maxBid: Math.max(...levels),
    minLevel: Math.min(...levels),
    final: { a: last.a, b: last.b, lambda: now.lambda, regime: now.regime, PA: now.PA, PB: now.PB, revenue: now.revenue, welfare: now.welfare },
    regret: { A: Math.max(0, brA.utility - now.uA), B: Math.max(0, brB.utility - now.uB) },
    // Range of lambda and bid level over the second half of the path.
    band: (() => {
      const tail = pts.slice(Math.floor(pts.length / 2));
      const lams = evals.slice(Math.floor(pts.length / 2)).map(e => e.lambda ?? 0.5);
      const lv = tail.map(p => Math.max(p.a, p.b));
      return { lambdaMin: Math.min(...lams), lambdaMax: Math.max(...lams), levelMin: Math.min(...lv), levelMax: Math.max(...lv) };
    })(),
    average: {
      revenue: mean(evals.map(e => e.revenue)),
      welfare: mean(evals.map(e => e.welfare)),
      PA: mean(evals.map(e => e.PA)),
      PB: mean(evals.map(e => e.PB)),
      uA: mean(evals.map(e => e.uA)),
      uB: mean(evals.map(e => e.uB)),
    },
  };
}

// ---------- Exact equilibrium analysis of the unmodified game ----------

// Upper envelope of lines y = c[j] - s[j] * x over x >= 0 (slopes s >= 0).
// Returns, per line index, the closed x-interval where it is maximal, or null.
export function envelopeIntervals(c, s) {
  // Slope -s ascending (s descending); equal slopes: larger intercept first.
  const idx = [...c.keys()].sort((i, j) => (s[j] - s[i]) || (c[j] - c[i]));
  const cross = (i, j) => (c[i] - c[j]) / (s[i] - s[j]);
  const hull = [];
  for (const i of idx) {
    if (hull.length && s[hull.at(-1)] === s[i]) continue;
    while (hull.length >= 2 && cross(hull.at(-2), i) <= cross(hull.at(-2), hull.at(-1))) hull.pop();
    hull.push(i);
  }
  const byLine = new Map();
  for (let t = 0; t < hull.length; t += 1) {
    const left = t === 0 ? 0 : Math.max(0, cross(hull[t - 1], hull[t]));
    const right = t + 1 < hull.length ? cross(hull[t], hull[t + 1]) : Infinity;
    if (right >= left) byLine.set(`${s[hull[t]]}|${c[hull[t]]}`, [left, right]);
  }
  // Lines identical to a hull line share its interval.
  return c.map((_, j) => byLine.get(`${s[j]}|${c[j]}`) ?? null);
}

// Pure Nash equilibria of the unmodified outcome game. Interior: both bids > 0
// and lambda inside a piece that is a best response for both. One-out: one
// sponsor bids 0, the other wins every contested step for free, and a large
// enough deterrent bid makes re-entry unprofitable.
export function originalEquilibria(lab, VA, VB) {
  const P = lab.pieces, m = lab.m;
  const IA = envelopeIntervals(Array.from(P.PA, x => VA * x), Array.from(P.KA));
  const IB = envelopeIntervals(Array.from(P.PB, x => VB * x), Array.from(P.KB));
  const interior = [];
  for (let j = 0; j < m; j += 1) {
    if (!IA[j] || !IB[j]) continue;
    const [bL, bH] = IA[j], [aL, aH] = IB[j];
    const rMin = bH === Infinity ? 0 : aL / bH;
    const rMax = bL === 0 ? Infinity : aH / bL;
    const pLo = P.lo[j] / (1 - P.lo[j]), pHi = P.hi[j] >= 1 ? Infinity : P.hi[j] / (1 - P.hi[j]);
    if (rMax > pLo && rMin <= pHi && aH > 0 && bH > 0) {
      interior.push({ piece: j, lo: P.lo[j], hi: P.hi[j], PA: P.PA[j], PB: P.PB[j], aRange: [aL, aH], bRange: [bL, bH] });
    }
  }
  // A out (a = 0, lambda = 0): stable iff b >= bStar.
  let bStar = 0, aStar = 0;
  for (let j = 0; j < m; j += 1) {
    if (P.KA[j] > 0) bStar = Math.max(bStar, VA * (P.PA[j] - P.PA[0]) / P.KA[j]);
    if (P.KB[j] > 0) aStar = Math.max(aStar, VB * (P.PB[j] - P.PB[m - 1]) / P.KB[j]);
  }
  return {
    interior,
    oneOut: {
      Bwins: { a: 0, bAtLeast: bStar, PA: P.PA[0], PB: P.PB[0], revenue: 0, welfare: VA * P.PA[0] + VB * P.PB[0] },
      Awins: { b: 0, aAtLeast: aStar, PA: P.PA[m - 1], PB: P.PB[m - 1], revenue: 0, welfare: VA * P.PA[m - 1] + VB * P.PB[m - 1] },
    },
  };
}

// ---------- Fixed total budget S (normalized shares) ----------

// Reports are normalized so that a + b = S, and charges use the normalized bids:
// E[pay_A] = S (1 - lambda) K_A(lambda). Within a piece this falls as lambda
// rises, so each piece's supremum is at its upper end, and the top piece's
// supremum (lambda -> 1) has zero charge and is not attained.
export function budgetShareSup(lab, VA, S) {
  const P = lab.pieces, m = lab.m;
  let best = null;
  for (let j = 0; j < m; j += 1) {
    const sup = VA * P.PA[j] - S * (1 - P.hi[j]) * P.KA[j];
    if (!best || sup > best.sup + EPS) best = { piece: j, sup, attained: j < m - 1, lambda: P.hi[j] };
  }
  return best;
}

// ---------- Outcome-level mechanisms ----------

function closestInPiece(lab, j, ref) {
  const lo = lab.pieces.lo[j], hi = lab.pieces.hi[j];
  if (ref > lo && ref <= hi) return ref;
  if (ref > hi) return hi;
  return lo + (hi - lo) * 1e-6;
}

// Ex-ante VCG over lambda: choose the piece maximizing expected welfare (ties to
// the piece nearest `ref`, and within it the lambda nearest `ref`), run the token
// sampler at that lambda on a seed drawn from the same sample, and charge Clarke
// payments. No token charges. Dominant-strategy truthful for risk-neutral
// sponsors whose value is V_i per occurrence of their outcome.
export function vcgExAnte(lab, VA, VB, { ref = 0.5 } = {}) {
  const P = lab.pieces, m = lab.m;
  const dist = j => (ref > P.lo[j] && ref <= P.hi[j]) ? 0 : Math.min(Math.abs(ref - P.lo[j]), Math.abs(ref - P.hi[j]));
  let j = 0, maxA = -Infinity, maxB = -Infinity;
  for (let k = 0; k < m; k += 1) {
    const w = VA * P.PA[k] + VB * P.PB[k], wj = VA * P.PA[j] + VB * P.PB[j];
    if (w > wj + EPS || (Math.abs(w - wj) <= EPS && dist(k) < dist(j))) j = k;
    maxA = Math.max(maxA, VA * P.PA[k]);
    maxB = Math.max(maxB, VB * P.PB[k]);
  }
  const payA = maxB - VB * P.PB[j], payB = maxA - VA * P.PA[j];
  const lambda = closestInPiece(lab, j, ref);
  return {
    mechanism: 'vcg-ex-ante', lambda, piece: j, PA: P.PA[j], PB: P.PB[j], payA, payB,
    uA: VA * P.PA[j] - payA, uB: VB * P.PB[j] - payB, revenue: payA + payB, welfare: VA * P.PA[j] + VB * P.PB[j],
    extras: Object.fromEntries(Object.entries(lab.extras).map(([k, v]) => [k, v[j]])),
  };
}

// Ex-post VCG: for each seed, after computing its atlas, choose the leaf
// maximizing V_A x_A + V_B x_B (ties to the leaf containing `ref`, then the
// nearest), with per-seed Clarke payments. Averages over the seeds.
export function vcgExPost(lab, VA, VB, { ref = 0.5, compareLambda = ref } = {}) {
  let PA = 0, PB = 0, payA = 0, payB = 0, changed = 0, moved = 0;
  const extras = Object.fromEntries(Object.keys(lab.extras).map(k => [k, 0]));
  const chosen = [];
  for (const seed of lab.seeds) {
    const dist = l => (ref > l.lo && ref <= l.hi) ? 0 : Math.min(Math.abs(ref - l.lo), Math.abs(ref - l.hi));
    let best = seed.leaves[0], maxA = 0, maxB = 0;
    for (const l of seed.leaves) {
      const w = VA * l.xa + VB * l.xb, wb = VA * best.xa + VB * best.xb;
      if (w > wb + EPS || (Math.abs(w - wb) <= EPS && dist(l) < dist(best))) best = l;
      maxA = Math.max(maxA, VA * l.xa);
      maxB = Math.max(maxB, VB * l.xb);
    }
    PA += best.xa; PB += best.xb;
    payA += maxB - VB * best.xb; payB += maxA - VA * best.xa;
    for (const k of Object.keys(extras)) extras[k] += best.extra[k];
    if (best.index !== leafIndexAt(seed, compareLambda)) changed += 1;
    moved += dist(best);
    chosen.push({ seed: seed.seed, leaf: best.index, lo: best.lo, hi: best.hi });
  }
  const n = lab.n;
  for (const k of Object.keys(extras)) extras[k] /= n;
  return {
    mechanism: 'vcg-ex-post', PA: PA / n, PB: PB / n, payA: payA / n, payB: payB / n,
    uA: VA * PA / n - payA / n, uB: VB * PB / n - payB / n,
    revenue: (payA + payB) / n, welfare: (VA * PA + VB * PB) / n,
    outputChangedVsCompare: changed / n, meanLambdaDistance: moved / n, extras, chosen,
  };
}

// Integral over reports z in [0, Z] of x(z), where x is P[j] on z-intervals.
function zIntervalsA(lab, rival) {
  const P = lab.pieces;
  return P.lo.map((lo, j) => [rival * lo / (1 - lo), P.hi[j] >= 1 ? Infinity : rival * P.hi[j] / (1 - P.hi[j])]);
}
function zIntervalsB(lab, rival) {
  const P = lab.pieces;
  return P.lo.map((lo, j) => [rival * (1 - P.hi[j]) / P.hi[j], lo === 0 ? Infinity : rival * (1 - lo) / lo]);
}

// Myerson payment and exact deviation regret for one side with allocation
// x(z) = own[j] on report intervals (zl, zh]. Utility is constant on each one.
function myersonSide(own, intervals, V, truthfulPiece) {
  const order = [...own.keys()].sort((i, j) => intervals[i][0] - intervals[j][0]);
  let integral = 0, truthful = null, best = -Infinity;
  for (const j of order) {
    const [zl, zh] = intervals[j];
    if (j === truthfulPiece) {
      const utility = own[j] * (V - zl) + integral;
      truthful = { piece: j, utility, pay: own[j] * V - utility };
    }
    if (!(zh > zl)) continue;
    best = Math.max(best, own[j] * (V - zl) + integral);
    integral += own[j] * (zh - zl);
    if (!Number.isFinite(integral)) break;
  }
  return { ...truthful, regret: Math.max(0, best - truthful.utility) };
}

// Proportional allocation lambda = V_A/(V_A+V_B) (the original mixture at bids
// equal to outcome values) with outcome-level Myerson payments. Truthful iff
// P_A is nondecreasing and P_B nonincreasing in lambda; `regret` is the exact
// gain from the best misreport against the rival's truthful report.
export function proportionalMyerson(lab, VA, VB, { monotone = false } = {}) {
  const P = lab.pieces;
  const lambda = VA / (VA + VB);
  const j = pieceIndex(lab, lambda);
  // With `monotone`, ratio pieces map onto a monotone chain of pieces, so the
  // outcome allocations are monotone and the Myerson payments exactly truthful.
  const k = monotone ? monotoneChain(lab).map : null;
  const PA = k ? k.map(i => P.PA[i]) : Array.from(P.PA);
  const PB = k ? k.map(i => P.PB[i]) : Array.from(P.PB);
  const A = myersonSide(PA, zIntervalsA(lab, VB), VA, j);
  const B = myersonSide(PB, zIntervalsB(lab, VA), VB, j);
  const used = k ? k[j] : j;
  return {
    mechanism: monotone ? 'proportional-myerson-monotone' : 'proportional-myerson', lambda,
    implementedLambda: k ? closestInPiece(lab, used, lambda) : lambda, piece: used,
    PA: PA[j], PB: PB[j], payA: A.pay, payB: B.pay, uA: A.utility, uB: B.utility, revenue: A.pay + B.pay,
    welfare: VA * PA[j] + VB * PB[j], regret: { A: A.regret, B: B.regret },
    monotone: monotonicity(lab), remappedMeasure: k ? monotoneChain(lab).remappedMeasure : 0,
    extras: Object.fromEntries(Object.entries(lab.extras).map(([key, v]) => [key, v[used]])),
  };
}

// Greedy chain of pieces, left to right, along which P_A never falls and P_B
// never rises. Every piece maps to the last chain piece at or before it.
export function monotoneChain(lab) {
  if (lab._chain) return lab._chain;
  const { PA, PB, lo, hi } = lab.pieces;
  const map = new Array(lab.m);
  let last = 0, remappedMeasure = 0;
  for (let j = 0; j < lab.m; j += 1) {
    if (PA[j] >= PA[last] && PB[j] <= PB[last]) last = j;
    else remappedMeasure += hi[j] - lo[j];
    map[j] = last;
  }
  lab._chain = { map, remappedMeasure };
  return lab._chain;
}

// Share of seeds whose output differs between two lambdas (same tapes).
export function outputChangeRate(lab, lambda, refLambda) {
  let changed = 0;
  for (const seed of lab.seeds) if (leafIndexAt(seed, lambda) !== leafIndexAt(seed, refLambda)) changed += 1;
  return changed / lab.n;
}

export function monotonicity(lab) {
  const { PA, PB } = lab.pieces;
  let dropA = 0, riseB = 0, nA = 0, nB = 0;
  for (let j = 1; j < lab.m; j += 1) {
    if (PA[j] < PA[j - 1]) { nA += 1; dropA = Math.max(dropA, PA[j - 1] - PA[j]); }
    if (PB[j] > PB[j - 1]) { nB += 1; riseB = Math.max(riseB, PB[j] - PB[j - 1]); }
  }
  return { PAnondecreasing: nA === 0, PBnonincreasing: nB === 0, violationsA: nA, violationsB: nB, maxDropA: dropA, maxRiseB: riseB };
}

// Naive reference: sponsors bid their outcome values as token bids in the
// unmodified mechanism. Same allocation as proportionalMyerson, token charges.
export function naiveValueBids(lab, VA, VB) {
  const e = evaluateBids(lab, VA, VB, VA, VB, OFF);
  return { mechanism: 'token-auction-at-value-bids', ...e, extras: Object.fromEntries(Object.entries(lab.extras).map(([k, v]) => [k, v[e.piece]])) };
}

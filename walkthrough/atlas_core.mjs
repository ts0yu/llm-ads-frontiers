// Exact bid atlases for any step-by-step generator, in the browser.
//
// A generator exposes:
//   init()            -> state
//   dists(state)      -> [pA, pB], two probability vectors over the same options
//   apply(state, k)   -> new state after option k (must not mutate `state`)
//   done(state)       -> boolean
// The auction is the two-bidder reference construction from auction.py. The
// per-step random tape is drawn from a seeded PRNG in step order, independent of
// bids, so the output is a step function of lambda = a/(a+b). See MOONSHOTS.md.

export function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function drawTapes(seed, steps) {
  const rand = mulberry32(seed);
  const tapes = [];
  for (let i = 0; i < steps; i += 1) {
    const tape = { branch: rand(), token_a: rand(), token_b: rand(), contest: rand() };
    if (tape.contest === 0) tape.contest = Number.MIN_VALUE;
    tapes.push(tape);
  }
  return tapes;
}

function cdf(weights) {
  let total = 0;
  for (const w of weights) total += w;
  if (total <= 0) return [];
  const out = new Array(weights.length);
  let run = 0;
  for (let i = 0; i < weights.length; i += 1) {
    run += weights[i] / total;
    out[i] = run;
  }
  out[out.length - 1] = 1;
  return out;
}

// numpy.searchsorted(cdf, u, side="right")
function pick(c, u) {
  if (!c.length) throw new Error('Attempted to sample zero probability mass.');
  let lo = 0;
  let hi = c.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (c[mid] <= u) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function normalize(scores) {
  const out = scores.map(x => Math.max(0, x));
  const total = out.reduce((s, x) => s + x, 0);
  if (!(total > 0)) throw new Error('A distribution needs positive mass.');
  return out.map(x => x / total);
}

export function softmax(logits, temperature = 1) {
  let max = -Infinity;
  for (const x of logits) if (x > max) max = x;
  return normalize(logits.map(x => (x === -Infinity ? 0 : Math.exp((x - max) / temperature))));
}

export class TwoBidderAuction {
  constructor(pA, pB) {
    if (pA.length !== pB.length) throw new Error('Distributions must share options.');
    const common = pA.map((x, i) => Math.min(x, pB[i]));
    let shared = 0;
    let tv = 0;
    for (let i = 0; i < pA.length; i += 1) {
      shared += common[i];
      tv += Math.abs(pA[i] - pB[i]);
    }
    this.pA = pA;
    this.pB = pB;
    this.sharedMass = Math.min(1, Math.max(0, shared));
    this.disagreement = 0.5 * tv;
    this.sharedCdf = cdf(common);
    this.aCdf = cdf(pA.map((x, i) => x - common[i]));
    this.bCdf = cdf(pB.map((x, i) => x - common[i]));
  }

  // Bid-independent part of a step: either a shared option or two candidates.
  prepare(tape) {
    if (tape.branch < this.sharedMass || this.disagreement === 0) {
      return { branch: 'shared', option: pick(this.sharedCdf, tape.token_a) };
    }
    return {
      branch: 'contested',
      a: pick(this.aCdf, tape.token_a),
      b: pick(this.bCdf, tape.token_b),
      u: tape.contest,
    };
  }

  resolve(bidA, bidB, tape) {
    const total = bidA + bidB;
    if (!(bidA >= 0 && bidB >= 0 && total > 0 && Number.isFinite(total))) {
      throw new Error('Bids must be finite, nonnegative and not both zero.');
    }
    const step = this.prepare(tape);
    if (step.branch === 'shared') {
      return { option: step.option, branch: 'shared', payment_a: 0, payment_b: 0 };
    }
    const u = step.u;
    if (u < bidA / total) {
      return { option: step.a, branch: 'A', payment_a: bidB * (u / (1 - u)), payment_b: 0 };
    }
    return { option: step.b, branch: 'B', payment_a: 0, payment_b: bidA * ((1 - u) / u) };
  }
}

// Exact interval search over lambda in (0, 1]. Returns leaves sorted by lo.
export function buildAtlas(generator, seed, maxSteps) {
  const tapes = drawTapes(seed, maxSteps);
  const leaves = [];
  const splits = [];
  const stack = [{ lo: 0, hi: 1, state: generator.init(), steps: [], options: [], parent: null, splitStep: null, id: 0 }];
  let nextId = 1;
  while (stack.length) {
    const leaf = stack.pop();
    let stop = 'steps';
    for (let k = leaf.steps.length; k < maxSteps; k += 1) {
      if (generator.done(leaf.state)) {
        stop = 'done';
        break;
      }
      const [pA, pB] = generator.dists(leaf.state);
      const auction = new TwoBidderAuction(pA, pB);
      const step = auction.prepare(tapes[k]);
      const record = { step: k, shared_mass: auction.sharedMass, disagreement: auction.disagreement };
      let option;
      if (step.branch === 'shared') {
        option = step.option;
        record.branch = 'shared';
      } else {
        record.u = step.u;
        if (step.u > leaf.lo && step.u < leaf.hi) {
          // Split: this leaf keeps (lo, u] with B's option; a child takes (u, hi] with A's.
          const child = {
            id: nextId++, lo: step.u, hi: leaf.hi, parent: leaf.id, splitStep: k,
            state: generator.apply(leaf.state, step.a),
            steps: [...leaf.steps, { ...record, branch: 'A', option: step.a }],
            options: [...leaf.options, step.a],
          };
          splits.push({ step: k, u: step.u, parent: leaf.id, child: child.id, a_option: step.a, b_option: step.b });
          stack.push(child);
          leaf.hi = step.u;
          option = step.b;
          record.branch = 'B';
        } else if (step.u <= leaf.lo) {
          option = step.a;
          record.branch = 'A';
        } else {
          option = step.b;
          record.branch = 'B';
        }
      }
      record.option = option;
      leaf.steps.push(record);
      leaf.options.push(option);
      leaf.state = generator.apply(leaf.state, option);
    }
    if (stop === 'steps' && generator.done(leaf.state)) stop = 'done';
    leaves.push({ ...leaf, stop });
  }
  leaves.sort((x, y) => x.lo - y.lo);
  return { seed, maxSteps, leaves, splits, bound: 1 + maxSteps };
}

// Direct generation at one bid pair, for verification.
export function runDirect(generator, seed, maxSteps, bidA, bidB) {
  const tapes = drawTapes(seed, maxSteps);
  let state = generator.init();
  const options = [];
  let paidA = 0;
  let paidB = 0;
  for (let k = 0; k < maxSteps && !generator.done(state); k += 1) {
    const [pA, pB] = generator.dists(state);
    const out = new TwoBidderAuction(pA, pB).resolve(bidA, bidB, tapes[k]);
    options.push(out.option);
    paidA += out.payment_a;
    paidB += out.payment_b;
    state = generator.apply(state, out.option);
  }
  return { options, state, paidA, paidB };
}

export function leafAt(atlas, lambda) {
  for (const leaf of atlas.leaves) if (lambda > leaf.lo && lambda <= leaf.hi) return leaf;
  return atlas.leaves[0];
}

// Exact simulated charges for a bid pair whose lambda lies in the leaf.
export function leafPayments(leaf, bidA, bidB) {
  let a = 0;
  let b = 0;
  for (const s of leaf.steps) {
    if (s.branch === 'A') a += bidB * (s.u / (1 - s.u));
    else if (s.branch === 'B') b += bidA * ((1 - s.u) / s.u);
  }
  return { a, b };
}

// Indicator intervals of a boolean outcome across leaves, and its monotonicity.
export function outcomeProfile(atlas, test) {
  const values = atlas.leaves.map(leaf => Boolean(test(leaf)));
  let measure = 0;
  let switches = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i]) measure += atlas.leaves[i].hi - atlas.leaves[i].lo;
    if (i && values[i] !== values[i - 1]) switches += 1;
  }
  let shape = 'constant';
  if (switches === 1) shape = values[0] ? 'decreasing' : 'increasing';
  else if (switches > 1) shape = 'non-monotone';
  const critical = switches === 1 ? atlas.leaves[values.indexOf(!values[0])].lo : null;
  return { values, measure, switches, shape, critical };
}

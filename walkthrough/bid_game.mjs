// Outcome-level bidding games on top of exact bid atlases.
//
// Each bidder values an OUTCOME of the whole generation (e.g. "the walk passes
// my storefront"), not individual steps. Over a set of seeds (independent
// tapes), the atlas gives, exactly as step functions of lambda = a/(a+b):
//   P_i(lambda)  probability that bidder i's outcome occurs
//   K_A(lambda)  mean of sum over A-won steps of u/(1-u)   so E[pay_A] = b * K_A
//   K_B(lambda)  mean of sum over B-won steps of (1-u)/u   so E[pay_B] = a * K_B
// The average over seeds is a Monte Carlo estimate of the expectation over
// tapes; for the sampled seeds the functions of lambda are exact.

export function aggregate(atlases, testA, testB) {
  const cuts = new Set([0, 1]);
  for (const atlas of atlases) for (const leaf of atlas.leaves) { cuts.add(leaf.lo); cuts.add(leaf.hi); }
  const edges = [...cuts].sort((x, y) => x - y);
  const pieces = [];
  for (let i = 0; i + 1 < edges.length; i += 1) {
    const lo = edges[i], hi = edges[i + 1];
    if (!(hi > lo)) continue;
    pieces.push({ lo, hi, mid: (lo + hi) / 2, PA: 0, PB: 0, KA: 0, KB: 0 });
  }
  for (const atlas of atlases) {
    const summary = atlas.leaves.map(leaf => {
      let ka = 0, kb = 0;
      for (const s of leaf.steps) {
        if (s.branch === 'A') ka += s.u / (1 - s.u);
        else if (s.branch === 'B') kb += (1 - s.u) / s.u;
      }
      return { lo: leaf.lo, hi: leaf.hi, xa: testA(leaf) ? 1 : 0, xb: testB(leaf) ? 1 : 0, ka, kb };
    });
    let j = 0;
    for (const piece of pieces) {
      while (!(piece.mid > summary[j].lo && piece.mid <= summary[j].hi)) j += 1;
      const s = summary[j];
      piece.PA += s.xa; piece.PB += s.xb; piece.KA += s.ka; piece.KB += s.kb;
    }
  }
  const n = atlases.length;
  for (const p of pieces) { p.PA /= n; p.PB /= n; p.KA /= n; p.KB /= n; }
  // Merge adjacent pieces with identical statistics for compact display.
  const merged = [];
  for (const p of pieces) {
    const last = merged.at(-1);
    if (last && last.PA === p.PA && last.PB === p.PB && last.KA === p.KA && last.KB === p.KB) {
      last.hi = p.hi; last.mid = (last.lo + last.hi) / 2;
    } else merged.push({ ...p });
  }
  return { pieces: merged, seeds: n };
}

// A's best response to a fixed rival bid b > 0. Utility VA*PA - b*KA is constant
// on each piece, so the maximum is exact. Within the best piece, A is
// indifferent; we report the piece and its midpoint bid.
export function bestResponseA(game, VA, b) {
  let best = null;
  for (const p of game.pieces) {
    const utility = VA * p.PA - b * p.KA;
    if (!best || utility > best.utility + 1e-12) best = { piece: p, utility };
  }
  const lam = best.piece.mid;
  return { ...best, lambda: lam, bid: b * lam / (1 - lam) };
}

export function bestResponseB(game, VB, a) {
  let best = null;
  for (const p of game.pieces) {
    const utility = VB * p.PB - a * p.KB;
    if (!best || utility > best.utility + 1e-12) best = { piece: p, utility };
  }
  const lam = best.piece.mid;
  return { ...best, lambda: lam, bid: a * (1 - lam) / lam };
}

// Alternating best responses from (a0, b0). Reports convergence or a cycle.
export function dynamics(game, VA, VB, a0 = 1, b0 = 1, rounds = 60) {
  let a = a0, b = b0;
  const path = [{ a, b, lambda: a / (a + b) }];
  const seen = new Map();
  for (let r = 0; r < rounds; r += 1) {
    const ra = bestResponseA(game, VA, b);
    a = ra.bid;
    const rb = bestResponseB(game, VB, a);
    b = rb.bid;
    const lambda = a / (a + b);
    path.push({ a, b, lambda, uA: VA * piecesAt(game, lambda).PA - b * piecesAt(game, lambda).KA, uB: VB * piecesAt(game, lambda).PB - a * piecesAt(game, lambda).KB });
    const key = `${a.toPrecision(8)}|${b.toPrecision(8)}`;
    if (seen.has(key)) {
      const start = seen.get(key);
      return { path, status: start === r - 1 ? 'converged' : 'cycle', cycleLength: r - start };
    }
    seen.set(key, r);
  }
  return { path, status: 'no convergence within rounds' };
}

export function piecesAt(game, lambda) {
  for (const p of game.pieces) if (lambda > p.lo && lambda <= p.hi) return p;
  return game.pieces[0];
}

// Rate card: for each target probability, the cheapest constant piece of lambda
// where the outcome's probability reaches the target, given the rival's bid.
// side 'A' uses (PA, KA) with E[pay] = rival * KA; side 'B' uses (PB, KB).
export function rateCard(game, side, rival, targets = [0.25, 0.5, 0.75, 0.9]) {
  return targets.map(target => {
    let best = null;
    for (const p of game.pieces) {
      const prob = side === 'A' ? p.PA : p.PB;
      if (prob + 1e-12 < target) continue;
      const cost = rival * (side === 'A' ? p.KA : p.KB);
      if (!best || cost < best.cost) best = { piece: p, prob, cost };
    }
    if (!best) return { target, reachable: false };
    const lam = best.piece.mid;
    const bid = side === 'A' ? rival * lam / (1 - lam) : rival * (1 - lam) / lam;
    return { target, reachable: true, probability: best.prob, expectedCharge: best.cost, lambda: lam, bid };
  });
}

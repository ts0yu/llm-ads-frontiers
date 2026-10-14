# Outcome-level bidding: why it cycles, and what fixes it

Status: exploratory research note. Synthetic generators only (the Atlas
Playground's city walk and melody), simulated credits, numbers exact for a
finite sample of seeds. Nothing here is a truthfulness, revenue or quality claim
about advertising in a real language model. `auction.py`, the two-bidder demo and
`bid_game.mjs` are unchanged.

Code: `walkthrough/mechanism_lab.mjs` (pure functions on top of `atlas_core.mjs`
and `bid_game.mjs`). Runs: `node scripts/mechanism_lab.mjs 1000 200 >
results/mechanism_lab.json` (1000 seeds, 200 best-response rounds, 4 starting
points, 4 value pairs, 2 worlds). Tests: `test_mechanism_lab.py`. Interactive:
the "Fixing the cycle" tab of `walkthrough/playground.html`.

## 1. Problem

The reference auction (MECHANISM.md) is a per-step mechanism. At each contested
step, A's candidate wins iff the contest uniform `u < lambda = a/(a+b)`, and the
winner pays its critical bid (`b u/(1-u)` for A). For a bidder that values each
contested win at `v`, bidding `v` is optimal at that step.

Sponsors usually value an *outcome* of the whole generation instead: "the walk
passes my storefront", "the jingle sounds like my label". For a fixed set of
seeds (random tapes), the atlas gives exactly, as step functions of lambda
(see `bid_game.mjs`):

```text
P_i(lambda)   probability of sponsor i's outcome
E[pay_A] = b * K_A(lambda),   K_A = mean over seeds of sum over A-won steps of u/(1-u)
E[pay_B] = a * K_B(lambda),   K_B = mean over seeds of sum over B-won steps of (1-u)/u
U_A(a, b) = V_A P_A(lambda) - b K_A(lambda)
```

The allocation depends on the bids only through the ratio lambda, but each
charge is proportional to the rival's bid level. The game is homogeneous: if
`(a, b)` are best responses for values `(V_A, V_B)`, then `(t a, t b)` are best
responses for `(t V_A, t V_B)`. With fixed values, nothing in the allocation
pins the bid level.

## 2. The cycle, measured

With 1000 seeds, alternating best responses (bid_game.mjs's rule: midpoint of
the best constant piece) never settle in any of the 8 value configurations or 4
starting points (200 rounds each).

| world, (V_A, V_B) | median growth per rising round | drop-outs in 200 rounds | lambda visited |
|---|---|---|---|
| city, (40, 40) | x4.36 | 14 | 0.315 to 0.99999 |
| city, (40, 20) | x4.36 | 24 | 0.315 to 0.99999 |
| melody, (40, 40) | x48.9 | 1 | 0.0085 to 0.208 |
| melody, (40, 20) | x5.81 | 71 | 0.0085 to 0.99999 |

City (40, 40), first rounds, as (a, b, lambda): `(1, 1, .5) -> (1.87, 3.86, .326)
-> (1.35e-4, 2.94e-4, .315) -> (5.9e-4, 1.28e-3, .315) -> ... -> (0.925, 1.96, .320)
-> (3.67, 4.4e-5, 1.0) -> (8.9e-5, 1.9e-4, .315) -> ...`. Both bids grow about
x4.4 per round while lambda stays in B's region. Then one sponsor's outcome no
longer pays for the charges, that sponsor drops to a near-zero bid, the other
then needs only a near-zero bid too, and the escalation restarts. This is an
Edgeworth-style cycle, like the sawtooth bidding Edelman and Ostrovsky (2007)
documented in first-price sponsored-search auctions.

**Exact equilibrium check (`originalEquilibria`).** For each constant piece j,
A's utility at rival bid b is the line `V_A P_A,j - b K_A,j`. The upper envelope
gives the exact interval of b on which piece j is A's best response, and
likewise for B. An interior pure equilibrium exists iff, for some piece, the bid
ratios allowed by those two intervals meet the piece's ratio range
`(lo/(1-lo), hi/(1-hi)]`.

- **Interior pure equilibria: none** in all 8 configurations (exact, for these
  seeds).
- **One-out equilibria do exist, with zero revenue.** If A bids 0, lambda = 0
  for every b > 0, so B gets its best outcome and pays `a K_B = 0`. A cannot
  profitably re-enter iff `b >= b* = max_j V_A (P_A,j - P_A,0)/K_A,j`. For city
  (40, 40), `b* = 3.79`. This is an equilibrium in which one sponsor deters the
  other with a large bid and nobody pays anything. Midpoint best responses never
  pick a deterrent bid, so the dynamics do not find it. Adding "inertia" (keep
  your bid if it is already a best response) changes nothing for the original
  game.

So the defect is not just the dynamics: the only pure equilibria in this sample
are degenerate.

Bidding the outcome values themselves as token bids (`a = V_A`, `b = V_B`) is
not a fix either: in city (40, 40) it charges 682.7 credits per generation in
expectation for 51.2 credits of outcome value. Token bids are prices per contested step, not
per outcome.

## 3. Candidate modifications

### (a1) Hard reserve per contested step

**Rule.** Bidder i is eligible iff `bid_i >= r`. If both are eligible, run the
reference contest and charge the winner `max(r, critical bid)`. If only one is
eligible, it wins every contested step at price r (lambda = 1 or 0). If neither
is, no sponsored generation runs (`noAd`: modeled as neither outcome occurring,
no charge; the playground worlds have no separate no-ad policy).

**Per-step incentives.** For fixed b and tape, A's win indicator is
`1[a >= r] * 1[u < a/(a+b)]` when b >= r, and `1[a >= r]` when b < r: monotone
in a. The threshold is `max(r, b u/(1-u))` or r, so the charge is the critical
bid and never exceeds the bid. Per-step truthfulness (value per contested win)
is kept. With r = 0 this is exactly the reference mechanism.

**Exactness.** For fixed b, A's expected charge on a piece is
`mean_seeds sum_{A-won} max(r, b u/(1-u))`. Each leaf keeps its A-won u's sorted
with prefix sums, so this is a count plus a prefix sum (`payVector`).
`test_mechanism_lab.py` checks it against direct generations to 1e-9.

**Result.** The reserve never produced an interior equilibrium in the runs:

- r = 0.5 (city 40/40): 2-cycles, cycle averages welfare 40.8, revenue 16.9.
- r = 2: settles at a monopoly. In city (40, 40), A takes lambda = 1 and pays
  r per contested step: revenue 20.4, welfare 40 (with inertia, B wins from
  other starts: revenue 25.5, welfare 40).
- r = 5: nobody can afford a sole win (`r * E[#contested steps] > V`). No ad,
  welfare 0.

Of the 192 reserve runs, 160 settled and 32 cycled. Every settled run ended at
a monopoly or no ad: `soleA` 75, `soleB` 37, `noAd` 48. The winner can depend on the starting bids. In melody (20, 40)
with r = 0.375 and inertia, A (value 20) wins from two of four starts: welfare
20 against 39.2 available. A reserve anchors the bid level at r, but it does
so by turning the contest into a winner-take-all purchase of the extreme mixture.

### (a2) Platform pseudo-bid on the midpoint distribution ("phantom")

**Rule.** The platform joins as a third bidder with fixed bid r. Its
distribution is the midpoint `m = (p_A + p_B)/2`.

**Why the midpoint.** The three-way linear mixture is

```text
(a p_A + b p_B + r m)/(a + b + r) = lambda' p_A + (1 - lambda') p_B,
lambda' = (a + r/2)/(a + b + r).
```

So the output stays in the original two-sponsor mixture family. Also
`min(p_A, p_B, m) = min(p_A, p_B)`, so the common (free) mass is unchanged. We
sample it with the reference coupling at lambda'. Equivalently, we run
`auction.py` at bids `(a + r/2, b + r/2)`. Bids still never enter the prompts.
The atlas machinery applies unchanged, with lambda replaced by lambda'.

**Per-step incentives.** A's candidate wins iff
`u < (a + r/2)/(a + b + r)`, which is increasing in a. The critical bid is

```text
A: max(0, (b + r/2) u/(1-u) - r/2)      B: max(0, (a + r/2)(1-u)/u - r/2)
```

This is the reference charge at the shifted bids, minus r/2, floored at 0. The
test cross-checks it against the Python `auction.py` on 300 random cases. It is
at most the bid, and r = 0 recovers the reference. This is not a pure subsidy.
For A, the charge is lower than at the unshifted bids on easy wins (u < 1/2) and
higher on hard wins (u > 1/2). A step with `u < (r/2)/(b+r)` is won for free.

**Result: bounded, mostly near-stationary bidding, rarely an exact equilibrium.**
In all 320 pseudo-bid runs, no sponsor dropped out: lambda' never left
[0.005, 0.995], and late-path bid levels stayed at or above 0.42. Exact fixed
points (pure equilibria) were reached in 40 of the 320 runs, mostly in melody. Elsewhere bids keep
moving inside a narrow band:

| city (40, 40) | lambda' late in path | cycle-avg welfare | cycle-avg revenue | exact NE? |
|---|---|---|---|---|
| r = 1 | 0.363 to 0.922 (3-cycle) | 43.9 | 14.2 | no |
| r = 10 | 0.490 to 0.512 | 51.8 | 25.6 | no (band) |
| r = 50 | 0.490 to 0.499 | 51.8 | 8.6 | no (band) |
| max over any single lambda | | 52.0 | | |

City's efficient mixture is near lambda = 1/2, which happens to be where the
platform's weight pushes lambda'. The pseudo-bid then does well. In melody, both
outcomes are rare at lambda = 1/2 and the efficient choice is extreme. There a
large r is costly. For melody (40, 40), r = 50 holds lambda' near 0.48, with
welfare 13.0 against 40 available. At r = 5 it settles at exact equilibria at
lambda' = 0.771 (welfare 32.3) or 0.258 (27.3), depending on the start. This is
the price of anchoring the level with a fixed neutral weight: it biases the
mixture toward the platform's default, by an amount the platform has to choose.

Not proved: existence of equilibrium, or boundedness of bids for this
mechanism. The bounded band is an observation on these runs.

### (b) Fixed total budget S (normalized shares)

Take any reading in which reports are normalized so that `a + b = S`.

1. If `b = S - a` is imposed, only one party chooses. That is not a two-bidder
   mechanism.
2. If both report shares `s_A`, `s_B` and the platform normalizes
   `a = S s_A/(s_A+s_B)`, the charges must be computed from something. Use the
   normalized bids: A pays `b u/(1-u) = S (1-lambda) u/(1-u)` when it wins. This
   charge *falls* as A's own report rises, so it is not a critical-bid payment.
   Per-step truthfulness is lost.
3. In the outcome game, `U_A = V_A P_A(lambda) - S (1-lambda) K_A(lambda)`.
   Within a piece K_A is constant, so utility rises with lambda. In the top piece
   `(1-lambda) K_A -> 0` as lambda -> 1. A's supremum utility is
   `V_A P_A(top piece)` and is not attained, so A has no best response. Each
   sponsor wants an unboundedly large share. `budgetShareSup` confirms this in
   every configuration: best piece = top piece, `attained: false`.

So a fixed budget does not define a mechanism with sensible incentives. Both
readings either remove a bidder's choice or reward claiming everything.

### (c) Outcome-level mechanisms on the atlas

Sponsors report a value `V_i` for their own (pre-specified, checkable) outcome.
The platform chooses lambda, or a leaf, using the atlas. Token charges are
replaced by outcome charges. The token sampler still produces the generation,
and bids still never enter prompts.

A Python port that runs (c1)-(c3) on stored `bid-atlas-v1` files (text or
image) is `token_auction_lab/outcome_auction.py`; see `docs/OUTCOME_AUCTION.md`.

**(c1) Proportional allocation with outcome-level Myerson payments.** Keep the
original allocation rule with bids equal to reported values,
`lambda = V_A/(V_A+V_B)`. Charge
`pay_A = V_A x_A(V_A) - integral_0^{V_A} x_A(z) dz`, with
`x_A(z) = P_A(z/(z+V_B))`, and symmetrically for B with
`x_B(z) = P_B(V_A/(V_A+z))`. The integral is exact: piece j maps to the report
interval `(V_B lo/(1-lo), V_B hi/(1-hi)]`. By Myerson's lemma this is
dominant-strategy truthful iff `x_A` is nondecreasing and `x_B` nonincreasing,
that is, iff the seed-averaged P_A is nondecreasing and P_B nonincreasing in
lambda.

- The averaged curves are *almost* monotone. City has 110 drops in P_A and 257
  rises in P_B among 23 854 pieces, each at most 0.001 (one seed in 1000). The
  exact gain from the best misreport is at most 0.0157 credits (city (100, 40),
  A, against utility 64.9).
- Monotone repair: map every piece to the last piece of a greedy chain along
  which P_A never falls and P_B never rises. Allocations become monotone, and
  measured regret is at most 1.1e-10. The chain remaps 16% (city) and 14%
  (melody) of the lambda-measure to a nearby earlier piece. At the configured
  values, the output at the chosen lambda changed in at most 1.8% of seeds.
  This is restriction, not Myerson's revenue ironing.
- Per seed, outcome indicators are often non-monotone (10 to 23% of seeds for
  the storefront outcomes, `results/playground_census.json`). A per-seed version
  of (c1) would need this repair much more.

The output distribution is exactly the original mixture at value-proportional
bids, and bidding is stable (dominant strategies). The weakness is welfare:
proportional allocation splits even when the outcomes are incompatible. In
melody (40, 40) it gets welfare 11.9 against 40 available.

**(c2) Ex-ante VCG over lambda (affine maximizer).** Choose the piece
maximizing `V_A P_A + V_B P_B`, breaking ties toward the reference
`V_A/(V_A+V_B)`, and within it the lambda nearest the reference. Generate with
the reference sampler at that lambda, on a seed drawn uniformly from the same N
seeds, so the curves are exact probabilities rather than estimates. Charge
Clarke payments `pay_A = max_lambda V_B P_B - V_B P_B(lambda*)`.

*Proof sketch of truthfulness.* For a report z, A's utility is
`V_A P_A(j(z)) + V_B P_B(j(z)) - max_k V_B P_B,k`. This is at most its value at
`j(V_A)`, the welfare maximizer. Nothing requires P_A to be monotone in lambda:
a non-monotone atlas does not need ironing here. Individually rational (utility
>= 0) and charges >= 0. The usual VCG caveats apply: low revenue in some
profiles, and collusion and false-name issues are not addressed.

**(c3) Ex-post VCG over leaves.** After drawing the seed and computing its atlas
(at most 1 + T leaves), choose the leaf maximizing `V_A x_A + V_B x_B`, with
per-seed Clarke payments. It is truthful seed by seed. It finds generations in
which *both* outcomes occur, which no single lambda can target. The cost: the
output is no longer a draw from one fixed mixture. The platform selects among
the leaves of each seed's atlas.

## 4. Results (1000 seeds; utilities, revenue and welfare in simulated credits per generation)

"Stable" means dominant-strategy truthful (outcome mechanisms) or an exact
fixed point of best responses (token-level dynamics). Dynamics rows report the
final state if settled, otherwise averages over the 200-round path. "Output
changed" is the share of seeds whose generation differs from the token
auction's output at the value-proportional reference `lambda = V_A/(V_A+V_B)`.
Per-step TV distance between two mixtures is `|lambda - lambda_ref| * TV(p_A, p_B)`.

**City walk, V = (40, 40)** (A: passes the café, B: passes the bookshop):

| mechanism | stable | lambda | P_A | P_B | revenue | welfare | output changed |
|---|---|---|---|---|---|---|---|
| original, best responses | no (cycle) | 0.315 to 1 | 0.12 | 0.93 | 4.9 avg | 41.9 avg | n/a |
| token auction at value bids | not a best response | 0.500 | 0.70 | 0.58 | 682.7 | 51.2 | 0% |
| reserve r = 2 | yes (monopoly) | 1 | 1.00 | 0.00 | 20.4 | 40.0 | 100% |
| pseudo-bid r = 10 | bounded band | 0.490 to 0.512 | 0.67 | 0.62 | 25.6 avg | 51.8 avg | n/a |
| proportional Myerson (chain) | yes | 0.500 | 0.70 | 0.58 | 38.2 | 51.2 | 0% |
| VCG ex ante | yes | 0.490 | 0.66 | 0.64 | 28.0 | 52.0 | 34% |
| VCG ex post (leaf per seed) | yes | per seed | 0.93 | 0.91 | 6.5 | 73.5 | 54% |

"Reaches the venue" moves little: 0.794 (reference), 0.800 (VCG ex ante), 0.791
(ex post).

**Melody, V = (40, 40)** (A: blue share < 15%, B: blue share > 30%):

| mechanism | stable | lambda | P_A | P_B | revenue | welfare | output changed |
|---|---|---|---|---|---|---|---|
| original, best responses | no | 0.009 to 0.21 | 0.01 | 0.89 | 7.4 avg | 36.0 avg | n/a |
| token auction at value bids | not a best response | 0.500 | 0.15 | 0.14 | 289.0 | 11.9 | 0% |
| reserve r = 2 | yes (monopoly) | 1 or 0 | 1 / 0 | 0 / 0.98 | 36.0 / 38.2 | 40.0 / 39.2 | 100% |
| pseudo-bid r = 5 | yes (2 equilibria) | 0.771 / 0.258 | | | 10.8 / 8.3 | 32.3 / 27.3 | |
| proportional Myerson (chain) | yes | 0.500 | 0.15 | 0.14 | 9.0 | 11.9 | 0% |
| VCG ex ante | yes | 0.983 | 1.00 | 0.00 | 39.2 | 40.0 | 100% |
| VCG ex post | yes | per seed | 0.51 | 0.49 | 39.4 | 40.0 | 70% |

In melody the two outcomes are essentially mutually exclusive. The
welfare-maximizing rules give the whole generation to one sponsor and charge
about the other's value, as a second-price auction would. Choosing per seed
adds no welfare.

**Asymmetric values, city (40, 20).** VCG ex ante: lambda = 0.567, welfare 42.6,
revenue 17.4. Ex post: welfare 56.8, revenue 3.2. Proportional Myerson: welfare
40.6, revenue 18.1. Reserve r = 1.5: A monopoly, welfare 40, revenue 15.3.
Pseudo-bid r = 7.5: lambda' 0.545 to 0.570, welfare 42.3 avg. The original game
cycles with 24 drop-outs in 200 rounds. Every configuration is in
`results/mechanism_lab.json`.

## 5. Reading the results

1. The cycle is structural. Allocation is scale-free in bids, while charges
   scale with the rival's level. The only pure equilibria found in these samples
   are one-sponsor-out profiles with zero revenue.
2. A reserve anchors the level, but every settled run was winner-take-all (or
   no ad). It earns revenue and gives up compromise mixtures such as city's
   lambda near 0.5, where both storefronts are often passed.
3. A platform pseudo-bid on the midpoint keeps the output in the original
   mixture family and kept bidding bounded in every run. Exact equilibria were
   rare, and it biases lambda toward 1/2 by an amount r that the platform has to
   choose. It is the best token-level option we found when the efficient
   mixture is near 1/2, and poor otherwise.
4. Asking sponsors for outcome values directly removes the bidding game. VCG
   over lambda is truthful without any monotonicity assumption and maximizes
   expected welfare over single mixtures. VCG over leaves gets more welfare when
   outcomes are compatible (73.5 against 52.0 in city), at the price of selecting
   generations rather than sampling one mixture. Proportional Myerson keeps the
   original mixture exactly and is truthful after a small monotone repair, but
   its welfare can be poor.

## 6. Relation to literature

These are cited from memory. The arXiv IDs other than 2310.10826 should be
checked before external use.

- Dütting, Mirrokni, Paes Leme, Xu, Zuo, *Mechanism Design for Large Language
  Models*, arXiv:2310.10826 (the token auction this repo implements).
- Hajiaghayi, Lahaie, Rezaei, Shin, *Ad Auctions for LLMs via Retrieval
  Augmented Generation*, arXiv:2406.09459. Segment-level auctions: the unit sold
  is a segment, not a token, closer in spirit to selling outcomes (c).
- Dubey, Feng, Kidambi, Mehta, Wang, *Auctions with LLM Summaries*,
  arXiv:2404.08126. Position-style auctions for LLM-generated summaries.
- Soumalias, Curry, Seuken, *Truthful Aggregation of LLMs with an Application to
  Online Advertising*, arXiv:2405.05905 (as I recall it, an aggregation-level
  truthfulness result; not checked in detail here).
- Edelman and Ostrovsky (2007), *Strategic bidder behavior in sponsored search
  auctions*, Decision Support Systems: sawtooth bid cycles in first-price
  sponsored search. Maskin and Tirole (1988, Econometrica): Edgeworth cycles.
- Tullock (1980) contests and the Kelly / Johari–Tsitsiklis (2004)
  proportional-allocation mechanism: the same a/(a+b) allocation with
  pay-your-bid charges. Pure equilibria there rely on concave valuations, which
  step-function outcome probabilities are not.
- Myerson (1981), Archer and Tardos (2001): monotone allocation plus threshold
  payments for single-parameter bidders (c1). Vickrey (1961), Clarke (1971),
  Groves (1973), Roberts (1979, affine maximizers): (c2), (c3).

## 7. Limitations and nonclaims

- Synthetic, hand-written generators. No language model, no measured latency
  or text quality.
- All curves are exact only for the sampled seeds (1000 per world; the
  playground tab uses up to 600). The ex-ante mechanisms are exact if the final
  seed is drawn from that same sample. Otherwise P_i are Monte Carlo estimates
  and truthfulness becomes approximate.
- Outcome mechanisms assume each sponsor values only its own pre-specified,
  checkable outcome, is risk-neutral, and reports one number. They need the
  atlas: up to 1 + T generations per seed, and many seeds for ex ante. For an
  LLM that is expensive.
- Dynamics results depend on the best-response rule (midpoint of the best
  piece; optional inertia) and on 200 rounds. "Did not settle" is not a proof of
  non-existence for the modified games. Only the unmodified game has an exact
  equilibrium search (`originalEquilibria`).
- The no-ad outcome for the reserve (neither outcome occurs) and the midpoint as
  the platform's distribution are modeling choices. A real no-ad distribution
  would take the mixture outside the two-sponsor family and require the simplex
  atlas (n-bidder race in MOONSHOTS.md).
- Simulated credits; floating-point arithmetic is not a payment ledger.

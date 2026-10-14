# Every answer money can buy: exact bid atlases for token auctions

Research note, Token Auction Lab, October 2026. Status: mechanism, algorithms
and tests are complete. Live-model atlases (Qwen3-4B text, three-bidder text,
SDXL images) were run on an H100 on 2026-10-14; see §7 and
[LIVE_RESULTS.md](LIVE_RESULTS.md). All charges are simulated credits. Nothing here claims truthful bidding over whole
answers, advertiser value, or production readiness.

## 1. Summary

The lab's two-bidder token auction (`auction.py`, after Dütting et al.,
arXiv:2310.10826) samples each token from the bid-weighted mixture
`q = lambda*p_A + (1-lambda)*p_B`, with `lambda = a/(a+b)`. It is stable for
each token: raising your bid only ever switches a contested token to your own
candidate, and the winner pays a threshold price.

This note shows that, for a fixed random seed, the auction's entire response
to bids can be computed exactly, not sampled:

1. **The answer is a step function of one number.** Bids enter only through
   `lambda`, in one comparison `u < lambda` per contested token. The random
   tape is drawn per step from the seed, independently of the bids.
2. **At most 1 + T distinct answers.** An interval search over `lambda`
   enumerates every answer the bids can buy for that seed. All branches read
   the same uniform at step k, so at most one branch splits per step. A
   96-token answer has at most 97 versions across all bid ratios.
3. **The expected number of answers is an integral.**
   `E[#answers] = 1 + integral_0^1 E[#contested steps at lambda] d lambda`.
4. **Exact charges and outcome curves.** Within each answer's interval, the
   charges have closed forms. Averaging over seeds gives the probability of any
   outcome, for example "mentions Bayview Hotel", as an exact step function of
   `lambda` for the sampled seeds.
5. **It is not specific to LLMs.** Any step-by-step sampler with two preferred
   next-step distributions qualifies. That includes diffusion models: their
   Euler-ancestral steps are equal-covariance Gaussians, which admit an exact
   maximal coupling with the same threshold payments. It also includes the
   synthetic city walk and melody in the browser playground.

Two empirical findings, both from synthetic generators:

- **Whole answers are often not monotone in the bid**, although every single
  token is. In 2000 seeds of the city walk, "the walk passes sponsor B's shop"
  is non-monotone in `lambda` for 22.6% of seeds.
- **Outcome-valuing sponsors do not settle.** When sponsors value an outcome
  rather than tokens, alternating best responses on exact outcome curves cycle.
  Bids escalate about ×4 per round, then one sponsor drops to almost zero
  (Edgeworth-style). Allocation depends only on the bid ratio while each charge
  scales with the rival's bid level, so nothing anchors the level.

## 2. Setting

At a fixed prefix with next-token distributions `p_A`, `p_B` over the same
vocabulary, let `c = min(p_A, p_B)`, `C = sum(c)`, and
`d = 1 - C = TV(p_A, p_B)`. With probability `C`, the token is drawn from
`c/C` for free. Otherwise each side draws a candidate from its residual
`(p_i - c)/d`, and a contest uniform `u` decides the winner:
- `u < lambda`: A's candidate, and A pays `b*u/(1-u)`;
- otherwise: B's candidate, and B pays `a*(1-u)/u`.

The marginal is exactly `q`, and each charge is the winner's critical bid. See
`MECHANISM.md`.

## 3. Exact bid atlases

**Algorithm (MOONSHOTS.md §1).** Decode once with the interval `(0, 1]`. At a
contested step with interval `(lo, hi]`:
- if `u <= lo`, A's candidate wins everywhere in the interval;
- if `u >= hi`, B's candidate wins everywhere;
- otherwise, fork the decoder state. `(lo, u]` continues with B's candidate and
  `(u, hi]` with A's.

**Bound.** Live branches partition `(0,1]` and share `u_k`, so at most one
branch splits at step k. That gives at most `1 + T` leaves.

**Expected count.** `#answers - 1` counts split steps. The branch containing
`u_k` is the generation at `lambda = u_k`. Its first k steps, and whether step
k is contested, are independent of `u_k`. Summing over k gives the integral
identity in §1. It holds in expectation, not per seed.

**Charges.** For `S = a + b` and `lambda` in a leaf, A pays
`S(1-lambda) * sum over A-won steps of u/(1-u)` and B pays
`S*lambda * sum over B-won steps of (1-u)/u`.

**Verification.** Every implementation checks leaves against direct
generations at bids inside each interval. Tokens and charges match exactly:
- browser playground: every leaf of 24 atlases;
- text atlas (`token_auction_lab/atlas.py`): every leaf, against
  `runner.run_round` on tiny random Qwen3 models;
- diffusion atlas: final latents of every leaf, on tiny SD and SDXL models.

## 4. Generalizations

**Diffusion (`gaussian_auction.py`, `diffusion_atlas.py`).** An
Euler-ancestral step draws `x' ~ N(mu_c(x), s^2 I)`, where `s` depends on the
noise schedule and not on the prompt. Two advertiser prompts give two Gaussians
with equal covariance. Their maximal coupling reduces to one dimension along
`mu_B - mu_A`, with `delta = |mu_B - mu_A|/s`:
- `TV = 2*Phi(delta/2) - 1`;
- the orthogonal noise is shared;
- branch, residual and contest uniforms, and the payments, are those of the
  token case.

The atlas forks only the latent. With bids (1,0), the loop reproduces the
diffusers scheduler step bit-for-bit. Numerical care was needed for tiny and
huge `delta`, using separate erf and erfc, quadrature below 1e-2 and clamped
uniforms.

**N bidders (`auction_n.py`, `simplex_atlas.py`).** The common mass
`min_i p_i` stays free, and every residual has the same mass `d`. In contested
steps an exponential race, `argmin E_i/b_i`, picks the winner:
- the marginal is exactly `sum(lambda_i p_i)`;
- the winner w pays `E_w * max_{j != w} b_j/E_j`;
- expected payment equals the two-bidder formula against one merged rival
  bidding `sum_{j != i} b_j`;
- for n = 2 it reproduces `auction.py` on 4,000 random cases: the same tokens
  and winners, and payments equal to 1e-12 relative (the race goes through
  exp and log).

Two differences appear for n ≥ 3:
- Residual supports can overlap, so a winner may pay for a race whose token a
  rival would also have produced.
- The 1 + T bound fails. Atlas cells are convex polygons in the bid simplex.
  Each contested step cuts along pieces of the hyperplanes
  `lambda_i/E_i = lambda_j/E_j` (for n = 3, three rays from one apex), so the
  count is at most `O(T^(n-1))` for fixed n. That is an upper bound, not a
  measured growth rate: in the three-sponsor city walk, mean cells go from
  116.5 at T = 20 to 461.6 at T = 80 (§5).

## 5. Findings on synthetic generators

These are hand-written stochastic policies, not models: a walker pulled toward
one of two sponsors' storefronts, and two "labels" composing a 32-note jingle.
They show the mechanism and the analysis, not advertising behaviour.

**Answers per seed (2000 seeds, `results/playground_census.json`).**

| World | T | Bound 1+T | Mean answers | Range |
|---|---:|---:|---:|---|
| City walk | 80 | 81 | 24.8 | 13–38 |
| Melody | 32 | 33 | 19.6 | 12–28 |

**Answer-level monotonicity (share of seeds with a non-monotone outcome).**

| World | Outcome | Non-monotone | Mean switches |
|---|---|---:|---:|
| City walk | passes A's shop | 10.1% | 1.21 |
| City walk | passes B's shop | 22.6% | 1.48 |
| City walk | reaches the venue within T | 70.9% | 1.78 |
| Melody | blue-note share > 30% | 11.5% | 1.23 |
| Melody | blue-note share < 15% | 7.1% | 1.14 |
| Melody | ends on C | 31.5% | 1.03 |

"Reaches the venue" is the clearest case. Middle bid ratios mix two coherent
walks into a dithering one that often runs out of steps. This is a non-language
analogue of incoherent text from evenly matched bids.

**Three sponsors (100 seeds, `results/simplex_census.json`).** A third shop
turns the bid ratio into a triangle of bid shares. Cells are convex polygons
and grow much faster than 1 + T:

| Max steps T | Mean cells | Max cells | Mean distinct routes |
|---:|---:|---:|---:|
| 20 | 116.5 | 151 | 60.4 |
| 40 | 295.8 | 448 | 167.8 |
| 80 | 461.6 | 778 | 276.6 |

Growth slows at T = 80 because most walks reach the venue earlier. Every cell
is checked against a direct race at its centroid in `test_playground.py`.

**Outcome-level bidding (`walkthrough/bid_game.mjs`).** Averaging 200 atlases
gives `P_i(lambda)` and the expected-charge factors `K_i(lambda)` exactly for
those seeds: `E[pay_A] = b*K_A`, `E[pay_B] = a*K_B`.

At one fixed prefix, a sponsor who values a contested win at v maximizes its
expected utility at that step by bidding v: the threshold payment is Myerson's
payment identity, and its expectation is `expected_payments`. This does not
extend to a whole generation, even for a sponsor that values every contested
win at v. The bid also moves the path, and with it which steps are contested
and how many. In the city walk (60 seeds, v = 2 against b = 1), bidding about
0.98 gives an expected utility of 41.5 credits, against 23.1 for bidding 2
(`test_playground.py`, checked by direct generations). Sponsors who value whole
outcomes face this:
- **City walk** (A wants the walk to pass its shop, B wants it to pass the
  other): best responses never settle within 60 rounds. A needs `lambda >= 0.67`
  and B needs `lambda <= 0.32`. Bids escalate about ×4.2 per round until one
  outcome stops paying for its charges, that sponsor drops to almost nothing
  (8 drop-outs in 60 rounds), and the cycle restarts.
- **Melody:** a short cycle among `lambda` of about 0.013, 0.19 and 0.20,
  with at most one early drop-out (V = 40, round 1).
- **Scale:** utilities are homogeneous of degree 1 in values and bids, so bid
  levels scale with values (max bid 1.8, 7.4, 28.1 for values 10, 40, 160).

**Fixing the cycle (`docs/OUTCOME_BIDDING.md`, `walkthrough/mechanism_lab.mjs`).**
Exact search over the step functions, on 1000 seeds per world, finds no
equilibrium of the unmodified token auction in which both sponsors bid positive
amounts, in any of 8 configurations. Results for the city walk with both outcomes worth 40:

| Mechanism | Bidding | Revenue | Welfare | Notes |
|---|---|---:|---:|---|
| Token auction, best responses | keeps moving | (cycles) | (cycles) | escalation and drop-out cycle |
| Hard reserve r = 2 | settles | 20.4 | 40.0 | one sponsor takes everything |
| Platform pseudo-bid r = 10 | bounded | 25.6 | 51.8 | output stays in the mixture family; few exact equilibria |
| Outcome VCG over one λ | truthful (dominant) | 28.0 | 52.0 | best single λ |
| Outcome VCG, leaf per seed | truthful (dominant) | 6.5 | 73.5 | selects generations instead of sampling one mixture |
| λ = V_A/(V_A+V_B), Myerson | truthful after repair | 38.2 | 51.2 | keeps the original mixture exactly |

A reserve stabilizes bidding by making one sponsor take everything. A
platform pseudo-bid keeps bids bounded but pulls λ toward ½. The
outcome-level mechanisms use the atlas to choose λ from reported values and
work best. Welfare is the expected value of the outcomes that occur.
Literature references in that document were written from memory and should
be checked before external use.

## 6. Software

| Piece | Where |
|---|---|
| Two-bidder text atlas + CLI | `token_auction_lab/atlas.py`, `test_atlas.py` |
| Gaussian coupling, diffusion atlas + CLI | `token_auction_lab/gaussian_auction.py`, `token_auction_lab/diffusion_atlas.py` |
| N-bidder race, simplex atlas + CLI | `token_auction_lab/auction_n.py`, `token_auction_lab/simplex_atlas.py`, `token_auction_lab/multi_decoder.py` |
| Atlas Explorer (replay viewer) | `walkthrough/atlas.html`, served at `/atlas.html` |
| Atlas Playground (live, synthetic) | `walkthrough/playground.html`, served at `/playground.html` |
| Census and bidding game | `scripts/playground_census.mjs`, `walkthrough/bid_game.mjs` |
| Live-model runbook | `scripts/run_moonshots.sh` |

The reference `auction.py`, the two-bidder decoder, Live Lab and all original
tests are unchanged.

## 7. Live-model runs

The first live-model atlases were run on a rented H100 on 2026-10-14 and are
published in the Atlas Explorer as `live-*` runs, labelled `live_model`. Full
results, figures and commands: [LIVE_RESULTS.md](LIVE_RESULTS.md). In short:

- Two-bidder Qwen3-4B text atlases have 17–39 answers per seed (bound 97 for
  T = 96). Every sampled leaf re-run directly matched its tokens and charges
  exactly, except the last seed of each run, whose checks were skipped for time
  and are labelled as skipped.
- SDXL atlases have 28–31 leaves of a possible 31. Images cross-fade between
  the two sponsors' scenes in exact steps.
- Between the extremes, the token mixture writes brand names that belong to
  neither sponsor ("Bayview Lodge", "Lantern Bayview") and star ratings that
  are in neither fact sheet. This happens on up to 78% of λ for some seeds.
  Both extremes name their sponsor exactly in all 20 two-bidder atlases.
- Three-bidder atlases exceed the two-bidder 1 + T bound on a real model. A
  largest-area-first search order (`--order area`) resolves far more of the
  triangle within a time budget than depth-first.

`scripts/run_moonshots.sh` still runs every stage on a Mac (MPS), a CUDA
machine or CPU.

## 8. Nonclaims

- No truthfulness or equilibrium claim for whole answers. §5 is exploratory and
  uses synthetic generators.
- One seed is one sample of the tape. Stored atlases can differ across
  hardware and software numerics.
- CLIP outcome scores for images are a noisy heuristic, and so are the
  string measures of `scripts/analyze_live_atlases.py`.
- Fictional brands only. Charges are simulated credits; nothing is billed,
  sent or settled.

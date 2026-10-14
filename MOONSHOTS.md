# Moonshots: exact bid atlases

Status: research extension, built after the Milestone 1 demo. The two-bidder
reference (`auction.py`), the shared-prefix decoder, Live Lab and the
walkthrough keep their current behaviour. Everything here lives in new modules
with its own tests. All charges are simulated credits.

## 1. The observation

At a fixed prefix the reference sampler uses bids only through one comparison:
`u < lambda`, with `lambda = a/(a+b)` and `u = tape.contest`. Whether the step
is shared depends on the two distributions, never on bids. The runner draws step
`k`'s `RandomTape` as the k-th draw from `numpy.default_rng(seed)`, regardless
of bids. So for a fixed seed, the whole generation is a deterministic function
of lambda alone (given model numerics), and it is piecewise constant.

**Interval search.** Start from the interval `(0, 1]`, decoding normally. At a
contested step with current interval `(lo, hi]`:

- `u <= lo`: A's candidate for the whole interval.
- `u >= hi`: B's candidate for the whole interval.
- otherwise split: `(lo, u]` takes B's candidate, `(u, hi]` takes A's.

Fork the decoder state at a split and continue each child. lambda = 1 is bids
(1,0). lambda -> 0+ reproduces bids (0,1), because `u > 0` strictly.

**Bound.** Every live branch at step k reads the same uniform `u_k`, and the
branches partition `(0,1]`, so `u_k` lies in at most one interval. At most one
branch splits per step: a T-step generation has **at most 1 + T distinct
outputs over all bid ratios**. This is deterministic, not a high-probability
estimate.

Each leaf records, per step, the branch label and `u`. The exact charges are
then available for any bid pair in the leaf: with `S = a+b` and `lambda`
inside the leaf, a contested step won by A charges A `S*(1-lambda)*u/(1-u)`,
and one won by B charges B `S*lambda*(1-u)/u`.

## 2. What it buys

- Exact bid-response curves (no grid) for any outcome function of the output,
  for example "mentions brand X exactly".
- An **answer-level monotonicity audit**. The per-token construction is stable.
  Whether an answer-level outcome is monotone in the bid ratio is an empirical
  question, measured exactly for each seed.
- **Outcome-level threshold prices**. When an outcome indicator is monotone in
  lambda for a seed, its critical lambda is a leaf boundary. Exploratory, with
  no truthfulness claim.
- **Outcome-level bidding.** When sponsors value outcomes, best responses in the
  token auction cycle. `docs/OUTCOME_BIDDING.md` and
  `walkthrough/mechanism_lab.mjs` analyze reserves, a platform pseudo-bid and
  outcome-level mechanisms (VCG over lambda or leaves, Myerson on the atlas).

## 3. Generalizations

- **Continuous steps (diffusion).** Stochastic samplers such as Euler-ancestral
  draw `x' ~ N(mu_c(x), s^2 I)`, where `s` comes from the noise schedule and
  does not depend on the conditioning. Two advertiser conditionings give two
  Gaussians with equal covariance. Their maximal coupling reduces to one
  dimension along `e = (mu_B - mu_A)/|mu_B - mu_A|`. With
  `delta = |mu_B - mu_A|/s`:
  `TV = 2*Phi(delta/2) - 1`, common mass `C = 2*Phi(-delta/2)`.
  The orthogonal noise is shared. The branch, residual and contest uniforms,
  and the threshold payments, are exactly those of the token case. See
  `token_auction_lab/gaussian_auction.py`.
- **N bidders (exponential race).** Common mass `c = min_i p_i` is free. Every
  residual `(p_i - c)/d` has the same mass `d = 1 - sum(c)`. In the contested
  branch, each bidder draws `E_i ~ Exp(1)` and the winner is
  `argmin E_i / b_i`, so bidder i wins with probability `b_i / sum(b)`, and the
  marginal is exactly `sum(lambda_i p_i)`. Raising `b_i` can only switch the
  winner to i. The winner w pays its critical bid `E_w * max_{j != w} b_j/E_j`
  (w wins iff `E_w/b_w < min_j E_j/b_j`). An earlier draft of this note said
  `min`; the two agree only for n = 2. The implementation and tests use `max`.
  Expected payment depends only on the bidder's own bid and the sum of the
  others' bids: `d*B*[log(1+b_i/B) - b_i/(b_i+B)]` with `B = sum_{j != i} b_j`,
  the two-bidder formula against one merged rival.
  Unlike n = 2, residual supports can overlap for n >= 3, so a winner can pay
  for winning the race even when a rival's win would have produced the same
  token. The 1 + T bound also does not carry over: each contested step adds a
  fan of three rays, and cell counts can grow like T^(n-1).
  For n = 2, setting `u = E_A/(E_A+E_B)` (which is Uniform(0,1)) recovers
  `auction.py` exactly, payments included. In the simplex, the winner regions
  at a step are convex cones meeting at `lambda ∝ E`, so atlas cells are convex
  polygons. See `token_auction_lab/auction_n.py`.

## 4. Atlas JSON schema (`bid-atlas-v1`)

```json
{
  "schema": "bid-atlas-v1",
  "kind": "text | image",
  "mode": "live_model | synthetic_model_test | synthetic_fixture",
  "bidders": ["Brand A", "Brand B"],
  "seed": 7,
  "parameter": "lambda = bid_A / (bid_A + bid_B)",
  "max_steps": 96,
  "manifest": {"model": "...", "revision": "...", "prompts": "...", "temperature": 0.8},
  "leaves": [
    {
      "id": 0,
      "lo": 0.0,
      "hi": 0.31,
      "parent": null,
      "split_step": null,
      "token_ids": [1, 2],
      "text": "...",
      "image": "relative/path.jpg",
      "stop_reason": "eos | length | steps",
      "steps": [{"step": 0, "branch": "shared|A|B", "u": 0.42, "shared_mass": 0.7}],
      "outcomes": {"mentions:Brand A": true}
    }
  ],
  "splits": [
    {"step": 5, "u": 0.31, "parent": 0, "child": 1, "b_token": 11, "a_token": 12,
     "prefix_text": "..."}
  ],
  "verification": [{"leaf": 0, "lambda": 0.15, "matches_direct_run": true}]
}
```

`leaves` partition `(0,1]` in order of `lo`. A split's `parent` keeps
`(lo, u]` (B's candidate) and the new `child` takes `(u, hi]` (A's candidate).
The parent's interval is recorded after all splits. N-bidder atlases use
`"schema": "simplex-atlas-v1"`, replace `lo/hi` with `polygon` (a list of
barycentric points), and record per-step `winner` and exponentials `E`.

## 5. Labels

- Live atlases come from real model inference: `mode: live_model`.
- Viewing a stored atlas is REPLAY.
- Tiny random models are `synthetic_model_test`. Hand-made fixtures are
  `synthetic_fixture`.
- Image "brand visible" scores come from CLIP zero-shot similarity: a noisy
  heuristic, not ground truth.

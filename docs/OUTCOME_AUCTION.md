# Outcome auctions on stored atlases

Status: research tool. Python, pure numpy, no model loaded. All charges are
simulated credits. Inputs can be synthetic (`synthetic_fixture`,
`synthetic_model_test`) or live-model atlases; for a live run see
[LIVE_RESULTS.md](LIVE_RESULTS.md) and `results/live_outcome_auction_competing.json`.

`token_auction_lab/outcome_auction.py` sells **outcomes** instead of tokens.
Each sponsor reports one value for one checkable outcome of the whole
generation, for example "the answer mentions my brand". The platform then
chooses the bid ratio (or a leaf) from stored `bid-atlas-v1` files, one per seed
for the same prompt, text or image. It is the Python port of the outcome-level
mechanisms in `walkthrough/mechanism_lab.mjs`. The theory and the synthetic
results are in `docs/OUTCOME_BIDDING.md` §3(c).

## What it computes

From the atlases it builds the exact step functions of
`lambda = bid_A/(bid_A+bid_B)` on the common grid of all leaf boundaries
(`prepare_lab`, a port of `prepareLab`):

- `P_i(lambda)`: the share of seeds whose leaf at lambda satisfies sponsor i's
  predicate;
- `K_A(lambda)`, `K_B(lambda)`: the token-auction charge factors
  (`E[pay_A] = b K_A`, `E[pay_B] = a K_B`), kept for comparison.

Mechanisms (each returns the chosen lambda or the leaf per seed, the allocation
probabilities, payments, revenue, welfare and each sponsor's utility at the
reported values):

| `--mechanism` | rule | payments | incentive property |
|---|---|---|---|
| `vcg` | one lambda maximizing `V_A P_A + V_B P_B` (ex ante). Ties go toward `V_A/(V_A+V_B)` | Clarke | truthful whatever the shape of `P_i`, if the generation's seed is drawn uniformly from the same seeds |
| `proportional` | `lambda = V_A/(V_A+V_B)`, after the monotone-chain repair | Myerson | truthful after the repair. The output reports the repair's remapped lambda-measure and the unrepaired regret |
| `leaf-vcg` | per seed, the leaf maximizing `V_A x_A + V_B x_B` (ex post) | per-seed Clarke | truthful seed by seed. The platform selects among leaves instead of sampling one mixture |

Default predicates are the stored, exact, case-sensitive `mentions:<bidder>`
leaf outcomes. A leaf without the key is an error, not a False. `--outcome
NAME=KEY` picks another stored key. `KEY>=x` (or `>`, `<=`, `<`) thresholds a
numeric outcome such as a stored CLIP score. In Python, any `leaf -> bool`
callable works (`prepare_lab(atlases, test_a, test_b)`).

Loading refuses mixed prompts. It requires the same mode, kind, bidders,
`max_steps`, model, tokenizer, prompts and generation parameters, plus distinct
seeds, complete atlases and leaves that partition `(0, 1]`. The mode is printed
and saved as stored (`MODE: synthetic_model_test ...`) and is never relabeled.

## CLI

```text
python -m token_auction_lab.outcome_auction ATLAS_DIR_OR_FILES... \
    --value "Bayview Hotel=40" --value "Lantern House=40" \
    [--outcome "NAME=KEY"] [--mechanism vcg|proportional|leaf-vcg|all] \
    [--curves] [--json OUT]
```

On the committed model-test text atlas (one seed, random-weight tiny Qwen3,
bidders HarborHop and Lantern House):

```text
$ python -m token_auction_lab.outcome_auction walkthrough/atlas_data/model-test-text \
    --value "HarborHop=40" --value "Lantern House=40"
MODE: synthetic_model_test -- SYNTHETIC MODEL TEST (random-weight model: checks the pipeline, not model quality)
Outcome auction on 1 stored bid-atlas-v1 atlas(es), kind text, 'A shuttle and a hotel'
Seeds (1): 3; leaves: 13; pieces: 13
...
mechanism        lambda     P(A)     P(B)    pay A    pay B  revenue  welfare   util A   util B
vcg              0.7733   1.0000   0.0000   0.0000   0.0000   0.0000  40.0000  40.0000   0.0000
proportional     0.5000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000
leaf-vcg       per seed   1.0000   0.0000   0.0000   0.0000   0.0000  40.0000  40.0000   0.0000
```

The model's weights are random, so only one leaf (lambda in (0.773, 0.820])
mentions HarborHop, and no leaf mentions Lantern House. VCG finds that leaf and
charges nothing, because the rival's outcome is unreachable. The output checks
the pipeline. It is not a result about advertising. `--json` also saves the
curves, per-seed choices, source paths with sha256, seeds, model and tokenizer
fields and verification status.

## Tests

`test_outcome_auction.py`:

- **Agreement with the JS.** It matches `mechanism_lab.mjs` (pieces, `P_i`,
  `K_i`, all three mechanisms and the unrepaired variant) to 1e-12 (1e-11 for
  Myerson) on four inputs: the committed model-test text atlas, the committed
  model-test image atlas with custom predicates, and 40-seed city-walk and
  30-seed melody atlases from the playground, exported as `bid-atlas-v1`.
- **Truthfulness by brute force.** Misreports are tried over log-spaced grids
  and every piece boundary. VCG, leaf VCG and repaired proportional Myerson
  allow no gain above 1e-9. For unrepaired Myerson, the brute-force gain never
  exceeds the reported exact regret.
- **Payments.** Payments are nonnegative and at most value × probability,
  overall and per seed for leaf VCG.
- **Exactness.** `P_i(lambda)` equals the share of seeds whose direct leaf at
  lambda satisfies the predicate, with exact float equality. `b K_A` and
  `a K_B` equal the direct charges, and the stored `run_round` verification
  payments.

## Assumptions and nonclaims

- Values are reported per outcome occurrence. Sponsors are risk-neutral and
  value only their own pre-specified predicate.
- Probabilities are exact only for the loaded seeds. The ex-ante mechanisms are
  exact if the final generation uses a seed drawn uniformly from that set.
  Otherwise `P_i` are Monte Carlo estimates and truthfulness becomes
  approximate.
- Truthfulness holds for these seeds and these outcome predicates only. It is
  not truthfulness of the token auction. Nothing here addresses collusion or
  false-name bids, and there is no revenue guarantee: VCG can earn little or
  nothing.
- No claim about real advertiser utility, answer quality, latency or on-chain
  verification. A substring mention is not a measure of favourable coverage.
- The monotone-chain repair is a restriction, not Myerson ironing. It changes
  the implemented lambda on part of the lambda-range.

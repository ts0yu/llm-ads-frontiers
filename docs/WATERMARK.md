# Sponsor watermark: who is this ad from?

Status: an opt-in research sampler (`token_auction_lab/watermark.py`,
`python -m token_auction_lab.atlas --watermark demo`). It combines with span
hold. The reference auction is unchanged. Charges are simulated credits. The
keys used here are **public demo keys**, derived from fixed strings so that
anyone can reproduce the results. A real deployment would use secret keys.

## The question

Given only the text of an answer, can someone holding the keys tell which
sponsor's tokens it contains, without the model or the prompts?

Yes, if each sponsor's tokens are drawn with that sponsor's own key.

## How it works

The auction already separates who wrote what. At each step it either draws a
*shared* token from `min(p_A, p_B)`, or runs a contest between a candidate from
A's leftover probability and one from B's. The contest uniform `u` decides the
winner, and the winner pays the critical bid.

The watermark keeps all of that and changes only the token draws:

| Token drawn from | Reference sampler | Watermarked sampler |
|---|---|---|
| shared part `min(p_A, p_B)` | inverse CDF, tape uniform | keyed race, **platform** key |
| A's leftover (A's candidate) | inverse CDF, tape uniform | keyed race, **A's** key |
| B's leftover (B's candidate) | inverse CDF, tape uniform | keyed race, **B's** key |
| span-hold proposal of owner W | inverse CDF, hold uniform | keyed race, **W's** key |

**The keyed race.** To draw from `q`, compute one uniform `r_v` per vocabulary
entry from Philox, keyed by (role key, last 4 generated tokens), and take
`argmin_v −ln(r_v)/q_v`. This is the exponential race used by distortion-free
LLM watermarks (Aaronson's proposal; Kuditipudi et al.'s "exponential minimum"
sampler).

- If `r` is uniform, the race picks `v` with probability exactly `q_v`. Averaged
  over keys, nothing about the auction changes: the same branch probabilities,
  the same λ-mixture, the same contest uniform, the same payments.
- Candidates still don't depend on the bids, so the atlas stays exact.
- The 1 + T bound is kept, because every branch reads the same contest uniform
  at step k.
- The argmin runs over random keyed numbers, not over probabilities. Nothing is
  argmaxed, filtered or reweighted.

**Detection.** For each generated token and each role, recompute `r` for the
token that appeared and score `−ln(1 − r)`. Repeated (context, token) pairs are
skipped.

- If the role's key had no hand in the token, the score is an exponential with
  mean 1.
- If the role drew it, the score is larger on average. How much larger depends
  on the entropy of the distribution it was drawn from.
- Summed over n tokens, the null is exactly Gamma(n, 1), so each role gets an
  exact p-value.

The detector needs only the token IDs and the keys: no model, no prompts, no
log.

**What it tells you, per answer:**

- **A's p-value:** does the answer contain tokens drawn on A's behalf, meaning
  contests A won and words A held?
- **B's p-value:** the same for B.
- **The platform's p-value:** did the answer come out of this ad system at all?
  This covers the shared tokens.

Per-token scores are noisy. The test is strong over many tokens and weak for a
handful.

## Results

All results below are LIVE MODEL OUTPUT. The model is Qwen3-4B-Instruct-2507 at
revision `cdbee75f…`, fp32, temperature 0.8, 96 tokens. The runs used span hold
plus the watermark on the same prompts and seeds as the reference and span-hold
runs: Fort Mason (5 seeds), transit + hotel (5) and competing hotels (10). Every
watermarked atlas matched direct watermarked re-runs, in tokens, steps, keys and
payments. The keys are the public demo keys. The run was on an H100 on
2026-10-14.

**1. The mechanism works.** Using the recorded ground truth (which key drew each
token), the mean score under each key was:

| Token drawn with | score under platform key | under A's key | under B's key |
|---|---|---|---|
| platform key (6,127 tokens) | **1.21** | 0.99 | 1.01 |
| A's key (3,131) | 1.00 | **1.39** | 1.00 |
| B's key (3,566) | 0.99 | 1.02 | **1.37** |

Unrelated keys see about 1.00, the Exp(1) null. The drawing key sees more: about
21 standard deviations over A's 3,131 tokens.

**2. One answer is not enough, from text alone.** A detector does not know which
tokens a sponsor drew, so it scores all of them. A sponsor draws only some of an
answer's 96 tokens, and those draws are mostly near-certain. Detected share of λ
at α = 0.01, mean over atlases:

| | platform | A | B |
|---|---|---|---|
| watermarked (20 atlases) | 3.7% | 9.7% | 3.5% |
| unwatermarked, same prompts (20 span-hold atlases) | 0.8% | 0.8% | 0.5% |

That is above the false-alarm rate, but far from usable per answer. Random edits
(10–25% of tokens replaced or deleted, truncation) were also measured. With so
little detected even on clean answers, those numbers say nothing about
robustness.

**3. A campaign is enough.** Pool every distinct answer of a scenario, scoring each
(context, token) once, and test each key:

| Scenario (answers) | A's key | B's key | platform key | same pooling, unwatermarked text (A / B / platform) |
|---|---|---|---|---|
| competing hotels (212) | p = 1e-8 | p = 6e-14 | p = 3e-6 | 0.70 / 0.20 / 0.06 |
| Fort Mason (139) | p = 1e-6 | p = 2e-6 | p = 2e-16 | 0.79 / 0.52 / 0.10 |
| transit + hotel (140) | p = 3e-20 | p = 7e-20 | p = 5e-3 | 0.74 / 0.42 / 0.58 |

Each pool holds about 2,700–3,500 distinct scored tokens. The "answers" are the
atlas leaves: different bid ratios of a few seeds, not independent ad
impressions. So read this as "a few thousand tokens of a sponsor's answers
suffice", not as a count of impressions.

**4. Why per-answer attribution is capped: entropy.** A distortion-free watermark
can carry at most the randomness of the draws it signs. When A wins a contest,
its token comes from A's leftover distribution, which is usually concentrated on
A's preferred token. That token is predictable, so it carries little signal.

**5. Replay with the model does not remove that cap.** `token_auction_lab/watermark_replay.py`
is for an auditor who holds the keys, the model and both advertiser prompts. It
replays each answer, recomputes every role's keyed draws, and checks whether the
emitted token equals one of them.

- **No dilution.** The two residuals have disjoint supports, so B's tokens cannot
  match A's residual draw.
- **No misses.** All 3,336 keyed tokens in the 60 replayed answers matched their
  own key's draw.
- **Exact test.** It counts matches only at informative steps (null match
  probability ≤ 0.5, fixed in advance) with an exact Poisson-binomial tail.

| Replay, answers at λ = 0.1 / 0.5 / 0.9 | answers | detected per answer (A, B, platform) | pooled p (A / B / platform) |
|---|---|---|---|
| competing hotels, watermarked | 30 | 5, 0, 1 | 4e-15 / 2e-6 / 2e-6 |
| competing hotels, unwatermarked | 30 | 0, 0, 0 | 0.26 / 0.86 / 0.40 |
| Fort Mason, watermarked | 15 | 1, 0, 4 | 0.05 / 0.03 / 5e-15 |
| Fort Mason, unwatermarked | 15 | 0, 0, 0 | 0.18 / 0.10 / 0.34 |
| transit, watermarked | 15 | 2, 0, 0 | 7e-9 / 1e-6 / 0.21 |
| transit, unwatermarked | 15 | 0, 0, 0 | 0.91 / 0.61 / 0.74 |

There were about 12–18 informative steps per role per answer. Most were tokens
the sponsor did *not* draw, so even with the model a single answer rarely
convicts. The replay's real gains are elsewhere:

- **Per-token map.** Which key could have drawn each word.
- **Zero false alarms here.** None in the 60 unwatermarked answers.

**6. Floor keying strengthens the campaign test, not the per-answer one.**
Drawing shared tokens with the most recent contest winner's key (`--watermark
demo-floor`; 20 more atlases, same seeds) puts about 5,500–6,600 tokens per
sponsor under that sponsor's key, scoring about 1.30 each. Per answer, A is
detected on 8.0% of λ and B on 8.9%, against 9.7% and 3.5% before. The campaign
p-values strengthen:

| | Fort Mason (A / B) | transit (A / B) | competing hotels (A / B) |
|---|---|---|---|
| plain keying | 1e-6 / 2e-6 | 3e-20 / 7e-20 | 1e-8 / 6e-14 |
| floor keying | 6e-19 / 3e-17 | 3e-27 / 4e-22 | 4e-8 / 3e-20 |

The unwatermarked controls stay above p = 0.05. Without span hold (competing hotels,
10 seeds) the picture is the same: campaign p = 2e-11 (A) and 1e-17 (B); per
answer, 11% (A) and 1.8% (B).

All 50 watermarked atlases (plain with span hold, floor with span hold, plain
without hold) matched direct re-runs: 300 checks, tokens, steps, keys and
payments. Results: `results/watermark/` (evaluation reports and the replay
output). The raw watermarked atlases (about 30 MB) are not committed. They
regenerate exactly with the commands below on the same hardware and software,
for example:

```sh
python -m token_auction_lab.atlas --scenario competing-hotels --span-hold --watermark demo \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --dtype float32 --seeds 1 2 3 4 5 6 7 8 9 10 \
  --max-tokens 96 --temperature 0.8 --verify 2 --output out/wm/competing-hotels-span
python scripts/watermark_eval.py --watermarked out/wm --null walkthrough/atlas_data/live-competing-hotels-span
python -m token_auction_lab.watermark_replay out/wm --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --output out/replay.json
```

**What this means.**

- **Per-answer "who is this ad from" from text alone** is not reliable at these
  settings. Use a signed disclosure record per answer for that; it is cheap and
  exact.
- **The watermark answers the campaign-level question,** with no logs or model:
  "do these texts carry sponsor X's key?"
- **More randomness helps.** Longer answers, higher temperature or larger
  vocabularies in play all raise the cap. Watermarks that tilt the probabilities
  (green-list schemes) would be stronger, but they change the auction's inputs,
  and the project rules exclude that in the reference mode.

GPU cost of the watermark runs and replay: about 37 minutes of one H100
(16:30–17:07 UTC), about $3.20 in simulated-research spend on the owner's
prepaid balance.

## Limits

- **Power depends on how many tokens a sponsor drew and how uncertain those draws
  were.** A sponsor that wins few contests, or wins only near-certain tokens,
  leaves little signal.
- **Synthetic edits are not paraphrases.** Random substitutions and deletions
  remove part of the signal; rewriting the text, by a person or another model,
  is expected to remove most of it. That is the known weakness of text
  watermarks, and it is not measured here.
- **Detection needs the right keys,** and the demo keys are public. With public
  keys anyone can also *forge* a watermark. Real use needs secret, per-sponsor
  keys held by the platform or an auditor.
- **It shows whose key shaped the text, not what the sponsor intended.** A
  shared token gets no sponsor attribution even when it is the sponsor's
  message.

## Tests

`test_watermark.py` covers:

- keyed draws are deterministic, keyed and exact samples;
- the keyed auction gives the reference branch and payments tape by tape, and the
  λ-mixture in distribution;
- the Gamma tail is exact;
- manifests carry only key fingerprints;
- detection rates and false alarms on a random toy world;
- wrong keys see nothing;
- watermarked atlases on a tiny random Qwen3, with and without span hold, match
  direct runs at every leaf;
- the evaluation script runs end to end;
- replay attribution recovers every keyed token, with both keying variants and
  with and without span hold, and the Poisson-binomial tail equals brute force.

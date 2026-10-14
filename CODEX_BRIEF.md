# Build brief: Token Auction Lab

## Outcome and timebox

Proposed timebox: roughly 24 hours, two human builders using Codex. Build an
interactive demonstration in which two fictional advertisers jointly generate a
short sponsored message. Their scalar bids affect each next-token distribution;
an auditable threshold-payment mechanism charges for contested outcomes.

The project is successful when a viewer can change bids, see genuine model output
change, inspect the actual next-token probabilities and payment calculation,
and distinguish a fixed-prefix counterfactual from a new full generation.

Do not turn this into an ad exchange, a general agent framework, an autonomous
shopping agent, a training project, or a blockchain inference system.

## Proposed stack, subject to the initial smoke test

Use one local/self-hosted model instance, initially
`Qwen/Qwen3-4B-Instruct-2507`. Its model card describes a 4B-parameter, non-thinking
causal model and provides Transformers examples [2]. Use Python, PyTorch, and
Transformers with direct forward calls. Transformers exposes vocabulary logits
and cached state for incremental decoding [3,4]. Pin a working model revision and
library versions after the smoke test rather than inventing a tested lockfile.

Plan for one 24 GB NVIDIA GPU. This is a planning allowance, not a measured
requirement: 4 billion two-byte parameters alone are about 8 GB in decimal units,
with extra memory needed for caches, activations, and runtime overhead. The model
need not be loaded twice. Keep both prompts short. The initial smoke runs used a
96-token cap. The live interface now runs until EOS, with a configurable safety
ceiling of at most 4,096 tokens and a manual stop control (user-requested update,
October 14, 2026). A safety cutoff or manual stop is labeled as partial output.
The user also requested experimental winner continuation: a contested winner
may receive up to three additional tokens at no charge when its next-token
probability is at least 98%. The live controls expose the limit and threshold;
zero extra tokens selects the original reference auction. See MECHANISM.md for
the exact state transitions and limits of the reference guarantees.

Proposed web layer: FastAPI, a React/TypeScript client, server-sent events, and
JSONL traces. Start with one live generation at a time. Browser clients must not
receive credentials or full model weights. No external data feed is needed;
curate fictional product fact sheets in the repository.

## Milestone 1: real decoding before UI

1. Run the existing CPU reference tests.
2. Load the model and tokenizer once. Put the model in evaluation/inference mode.
3. Prepare two complete prompt prefixes ending at the start of an assistant
   answer. Include the same user task and factual constraints in both; vary only
   the advertiser objective and fact sheet. Do not include bids.
4. Run a prefill for each context, with distinct KV state, masks, and positions.
5. At every step, obtain each context's full next-token logits, apply an agreed
   fixed temperature before mixing, and compute normalized probabilities.
6. Feed those vectors into the reference auction and obtain one token ID plus
   charges. Float32 probabilities should be explicitly cast to float64 and
   renormalized before calling the strict NumPy reference validator.
7. Advance both contexts with EXACTLY that same token ID, even when the token
   strongly favors the other advertiser. Do not re-tokenize decoded text to
   rebuild the assistant suffix. Do not turn every token into a new chat turn.
8. Stop on a configured EOS token or the shared token cap. Decode the shared token
   sequence with correct handling of tokens that split UTF-8 characters.

Initially, sequential forwards with separate caches are acceptable. Optimize
with batching or a GPU-resident sampler only after correctness is demonstrated.
Add a test comparing a cached continuation with an uncached full-prefix forward
within an appropriate numeric tolerance.

Exit criteria: generate several short texts with bid pairs `(1,0)`, `(1,1)`, and
`(0,1)`; log real examples, seconds per token, time to first token, peak memory,
and EOS/length behavior. Verify live distributions are meaningfully different.
No fabricated throughput targets or cherry-picked success claims.

## Milestone 2: events, ledger, and replay

Freeze each bidder's scalar bid for the entire round. Reserve at most
`max_tokens * bid_i` credits per bidder before beginning, because each actual
step charge is bounded by that bidder's bid. Release unused reservation at the
end. Do not introduce adaptive pacing into the initial mechanism.

Define the event schema before assigning frontend work. A token event contains:

```json
{
  "round_id": "example",
  "step": 7,
  "token_id": 123,
  "decoded_text_so_far": "illustrative, not a live result",
  "bid_a": 1.0,
  "bid_b": 1.0,
  "branch": "A",
  "shared_mass": 0.5,
  "candidate_a": 123,
  "candidate_b": 456,
  "payment_a": 0.6666666667,
  "payment_b": 0.0,
  "cumulative_a": 0.8,
  "cumulative_b": 0.4,
  "top_tokens": [],
  "mode": "synthetic_schema_example"
}
```

A real event's values must be calculated, not copied from this example. The top
rows should be the union of each distribution's displayed top candidates, with
p_A, p_B, and q shown for every row. Displaying top-k is fine; sampling from a
renormalized displayed table is not.

Keep full arrays for a small number of selected inspection checkpoints, not every
run forever. Store the random tape per step and a run manifest containing model
revision, prompt hashes, full prompts, tokenizer, temperature, dependency
versions, and seeds. A seed alone is not enough for bit-identical replay across
hardware or software changes. Stored traces should remain replayable without a
GPU, with a persistent REPLAY label.

A deterministic simulation RNG is adequate for a local demo, not a public money
market. In the demo, lock bids before exposing the next random tape. A trusted
operator could otherwise reroll outcomes; settlement hashes alone do not solve
that problem.

## Milestone 3: interface

Use bidder panels on the left/right and a shared generated message in the center.
Show a permanent SPONSORED SIMULATION label. Provide preset fact sheets and a
separate no-ad baseline. Changing bid controls starts a new round; it does not
rewrite historical charges.

Clicking a token opens an inspector: both probability distributions, the mixture,
common/disagreement mass, sampled residual candidates, threshold, and charge.
Explain labels as sampling branches, not semantic authorship or ownership.

In the inspector only, allow changing bids while freezing the prefix,
distributions, and tape. Demonstrate the single threshold switch. This is a
local counterfactual, not a claim about an entire answer's causal history.

Begin with complementary fictional products, such as a transit service and a
hotel. Add a second preset with competing hotels. Curate explicit allowed facts
and prohibit invented prices, discounts, endorsements, or real transactions.

## Milestone 4: evaluation and demo

Run a small evaluation sweep: two scenarios, five bid ratios, ten seeds, and an
output cap. Keep a no-ad baseline and each single-advertiser endpoint. A prompt-
weighted generation baseline is optional but useful for comparing complexity.

Report brand inclusion, exact brand-name correctness, unsupported factual claims,
coherence, tokens and seconds generated, mean payments, and variance. Human-read
several samples. If an LLM judge is used, label it as a noisy heuristic rather
than ground truth. Preserve failures and do not show only favorable seeds.

Verify mechanism invariants separately from text metrics. More favorable
next-token distributions do not imply monotonically improving full-answer
quality, business value, or advertiser conversion.

Do not silently rewrite the final text with a third model: that breaks the trace
between what was auctioned and what was displayed. Flag failures in the research
demo. A deployed rejection/refund policy would need separate analysis.

## Work allocation and schedule

First 0–2 hours: one builder tackles inference and the other locks the event
schema plus starts a synthetic UI. Stop and reassess if the decoder cannot
produce real shared-prefix output.

Hours 2–6: complete live CLI and reference integration. Keep changes isolated:
backend/model loop versus frontend/traces. Integrate one real token stream early.

Hours 6–12: interactive inspector, credit ledger, frozen-prefix counterfactuals,
and replay. Hours 12–18: evaluation, numerical tests, prompt/fact-sheet fixes.
Reserve the final hours for deployment, latency work, a recorded fallback, and
the presentation. Do not spend those hours starting a new research mechanism.

Codex supports repository guidance in AGENTS.md and isolated worktree chats [5,6].
Use one worktree per bounded component and a human-owned integration branch.

## Optional crypto stretch goal

Only after the live demo works, add testnet round-level escrow: reserve a bounded
credit equivalent, generate off-chain, settle aggregate charges once, and record
a trace hash. Require explicit authorization before any transaction. Include
round IDs, replay prevention, caps, and an expiry/refund route.

Label the inference operator as trusted. A receipt signature or on-chain hash
does not establish that the claimed model logits, random draws, or generation
were correct. Do not do per-token chain transactions or claim trustless inference.

## Important boundaries

Do not approximate full distributions from a top-logprob list without labeling
and evaluating the approximation. Do not average logits and describe it as
linear probability mixing. Do not replace sampling by argmax. Do not insert a
third bidder, reserve, budget-responsive weight, post-mixture filter, or
LLM-generated valuation and assume the existing payment tests still suffice.

Do not claim that an advertiser model's preferred distribution is its actual
business utility, that bids have a uniquely truthful interpretation, or that
strategic prompts/model submissions are handled.

## Sources

[2] Qwen model card: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

[3] Transformers model outputs:
https://huggingface.co/docs/transformers/main_classes/output

[4] Transformers cache explanation:
https://huggingface.co/docs/transformers/cache_explanation

[5] Official Codex AGENTS.md documentation:
https://developers.openai.com/codex/guides/agents-md

[6] Official Codex worktree documentation:
https://developers.openai.com/codex/app/worktrees

# Span hold: keeping words and names whole

Status: an opt-in research sampler (`token_auction_lab/span_hold.py`,
`python -m token_auction_lab.atlas --span-hold`). The reference auction
(`auction.py`, `runner.run_round`, the Live Lab) is unchanged. Charges are simulated
credits.

## The problem it addresses

The reference auction runs one contest per token. Brand names span several tokens,
so a contest can fall inside a name, and the other advertiser's context then
finishes it. On Qwen3-4B, between the two extreme bids, this produced names that
belong to neither sponsor, such as "Lantern Bayview" and "Bay Area’s Lantern Hotel".
It also produced star ratings that appear in no fact sheet
([LIVE_RESULTS.md](LIVE_RESULTS.md), finding 2).

## The rule (span-hold-v1)

1. **Spans.** Text is cut into spans: a word, a number, or a run of capitalised words
   (a name). A token *continues* the current span if one of these holds:
   - it has no leading space and extends a word or number: "Lan"+"tern",
     "Hotel"+"’s", "4"+"."+"9";
   - it is a single space plus a capitalised word, and the span is a name:
     "Lantern"+" House".

   Anything else starts a new span: a lowercase word, punctuation, a newline or
   EOS. See `continues_span`.
2. **Ownership.** When a contested step is won by advertiser W, W owns the span that
   its token starts or joins. Shared (uncontested) steps give no ownership.
3. **Holding.** While W owns a span, at each step W *proposes* a token, sampled from
   its own full next-token distribution `p_W`. The draw uses inverse CDF with that
   step's hold uniform.
   - If the proposal continues the span, it is emitted: no contest, no charge.
   - Otherwise the span is released and that step is an ordinary reference auction
     step, with the ordinary tape.
   - A hold lasts at most 16 tokens.
4. **Randomness.** The hold uniforms come from their own generator,
   `numpy.random.default_rng([seed, 0x5BA9])`, one per step. The reference tape
   (one `RandomTape` per step from `default_rng(seed)`) is drawn exactly as before.

## What is preserved

- **Project rules.** Probabilities are sampled; nothing is argmaxed, averaged as
  logits or filtered after mixing. The exact selected token ID is appended to both
  contexts, each with its own prompt, mask, positions and KV cache. Bids never
  enter a prompt.
- **Charges.** Bids enter only through `u < lambda` at contested steps, as in the
  reference. A contested step charges the reference critical bid: A pays
  `b·u/(1−u)` and B pays `a·(1−u)/u`. Held steps charge nothing.
- **The atlas.** A held step depends only on the owner, the shared prefix and the
  hold uniform, never on the bids. So for one seed the output is still a step
  function of `lambda`. The interval search is unchanged, and the 1 + T bound holds
  with the same proof: all branches read the same contest uniform at step k, so at
  most one branch splits per step. Every leaf is checked against the direct
  sampler `span_hold.generate` at the same bids.
- **Per-contest threshold structure.** Fix the prefix and the tapes. Winning a
  contest decides the token and the continuation the winner then holds, and you
  win iff `u < lambda` (A) or `u ≥ lambda` (B). So each contest is still a
  monotone allocation with a threshold price, with "the span" as the item.
  As with the reference auction, this says nothing about truthful bidding over a
  whole answer.

## What changes

- One contest can now decide several tokens: the winner writes the rest of the
  span from its own distribution. Per token, the answer is no longer a sample
  from the bid-weighted mixture. Roughly, at each span start the span is drawn
  from one advertiser's model, chosen by the auction.
- A span that starts on a shared token has no owner until its first contest. So a
  word both contexts agreed to begin can still be finished by either side.
- After a release, the next step is an ordinary contest. A losing context can
  still begin a new span right after the owner's span: for example, a
  capitalised word after a released name.

## Tests

`test_span_hold.py`:

- **Rule.** Unit tests of `continues_span` and the ownership state.
- **Toy world.** Hand-written policies in which A wants "Bayview Hotel" and B
  wants "Lantern House". Holds are disabled (`MAX_HOLD = 0`) in some tests.
  - With holds off, the sampler reproduces the reference sampler exactly, tokens
    and payments.
  - With holds off, it writes "Lantern Bayview Hotel" and "Bay Area" for some bids.
  - With holds on, over 40 seeds × 39 bid ratios every output is an exact name.
    Charges equal the reference charges at the contested steps.
- **Atlas on a tiny random Qwen3.**
  - Span-hold atlases partition (0, 1] within 1 + T leaves.
  - Every leaf matches a direct span-hold run at its midpoint and upper end,
    tokens, steps and payments.
  - With holds off, the span-hold atlas equals the reference atlas.

These are synthetic tests of the mechanism. Results on Qwen3-4B are in
[LIVE_RESULTS.md](LIVE_RESULTS.md).

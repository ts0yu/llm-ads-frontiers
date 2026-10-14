# Overnight run, 2026-10-14: what happened

Branch: `claude/vibrant-maxwell-pimdne`. Everything is committed and pushed.
The two-bidder reference (`auction.py`), the demo and every original test are
unchanged.

## Try it first

Private pages (only you can open them until you share them):
- **What can a pair of bids buy?** The story, with live demos:
  https://claude.ai/artifact/QQTMQr2LNRsgPcxf9pueEX
- **Atlas Playground.** Every outcome bids can buy, computed live: city walk,
  melody, three-sponsor triangle, bidding game, fixing the cycle:
  https://claude.ai/artifact/2giVL2vMCaWVdkf58EK8wP
- **Atlas Explorer.** Replay viewer for stored atlases (text, image,
  three-bidder): https://claude.ai/artifact/6GZLETMppahxgsvV6E5yNt

Locally: `python -m token_auction_lab.walkthrough`, then open `/story.html`,
`/playground.html` or `/atlas.html`.

## Morning update: the live-model run (H100, 14:00 UTC)

After your approval, the code ran on `tal-qwen-h100`. Results are published in
the Atlas Explorer (live atlases open first) and written up in
[LIVE_RESULTS.md](LIVE_RESULTS.md):

- **SDXL image atlases** (café, beach, park; 12 seeds). Moving λ cross-fades
  between the two sponsors' scenes, one exact leaf at a time: a teal café turns
  red, a purple kiosk turns yellow. There are 28–31 leaves out of a possible 31,
  and the re-run leaf of each seed was bit-identical.
- **Qwen3-4B text atlases** (Fort Mason defaults, transit + hotel, competing
  hotels; 20 seeds). Re-run leaves matched exactly, with 0.0 payment error;
  the last seed of each run skipped its check for time and is labelled so.
- **Finding:** the middle of the bid range writes brand names neither sponsor
  owns ("Lantern Bayview", "Bay Area’s Lantern Hotel") and star ratings in
  neither fact sheet (4.3 where the sheet says 4.9). At the extremes, all 20
  two-bidder text atlases name their sponsor exactly.
- **Three bidders on a real model** exceed the 1 + T bound (276–310 cells
  for T = 64). A new largest-area-first search resolves 65–75% of the bid
  triangle in 300 s, where depth-first resolved 8–52% in 240 s.
- **Outcome auction on live atlases:** per-seed leaf VCG finds leaves that
  mention both hotels in 3 of 10 seeds.

GPU spend this morning: about 46 minutes of H100 time, about $3.90. The node is
stopped. The provider showed $9.56 spent this month and $20.08 of credit left.

## The idea

For one random seed, the token auction's response to bids is a step function
of `lambda = a/(a+b)`. An exact interval search lists every answer any bids can
buy: at most 1 + T of them, each with exact charges. The research note is
[docs/ATLAS_NOTE.md](ATLAS_NOTE.md) and the mechanism study is
[docs/OUTCOME_BIDDING.md](OUTCOME_BIDDING.md).

## What was built

- **Exact atlases:**
  - two-bidder text (`token_auction_lab/atlas.py`);
  - three-bidder simplex (`simplex_atlas.py`, `auction_n.py`,
    `multi_decoder.py`);
  - diffusion images, from an exact Gaussian coupling (`gaussian_auction.py`,
    `diffusion_atlas.py`).
  Each has a CLI and checks every leaf or cell against direct runs.
- **Browser playground** on synthetic non-LLM generators, plus a census over
  2000 seeds (`results/`).
- **Atlas Explorer,** and a publish step (`scripts/publish_atlases.py`) that
  turns run outputs into explorer data.
- **Outcome-level bidding:** the escalation and drop-out cycle, and mechanisms
  that fix it. Outcome-level VCG over λ works best.
- **Outcome auction CLI** (`python -m token_auction_lab.outcome_auction`):
  sells outcomes such as a brand mention from stored atlases, using VCG,
  proportional Myerson or per-seed VCG. See [OUTCOME_AUCTION.md](OUTCOME_AUCTION.md).
- **Theory:**
  - the 1 + T bound;
  - `E[#answers] = 1 + integral of E[#contested steps] d lambda`;
  - for N ≥ 3 bidders, the critical bid is `E_w * max_j b_j/E_j`, and the 1 + T
    bound fails.
- The full test suite grew from 38 to 160 tests at the last full run. All pass,
  with none skipped for import reasons.

## Still open

- Ruff 0.17 reports 34 style findings in files this work did not touch
  (`live.py`, `walkthrough.py`, `runner.py` and their tests).
- The node's venv uses `transformers==4.57.6` with `huggingface_hub==0.36.2`
  and `diffusers==0.37.0`; newer `diffusers` pulls a hub version that breaks
  transformers 4.x.

## Caveats worth knowing

- **Corrected claim:** the per-token truthfulness of the auction holds at one
  fixed prefix only. Over a whole generation, a sponsor's bid changes which
  steps get contested, so bidding its value per won step is not optimal
  (city walk: bidding value 2 earns 23.1 expected credits, bidding about 0.98
  earns 41.5). An independent audit caught this; the docs and pages now say
  so, and a regression test pins it.

- All playground and census results use hand-written synthetic generators,
  not language models. Charges are simulated credits.
- `test_atlas_explorer.py` (a test file added tonight) skips bid pairs whose
  floating-point ratio `S*lambda/(S*lambda + S*(1-lambda))` differs from
  `lambda`. At an interval's exact upper boundary, which is a contest uniform,
  one ulp of rounding legitimately selects the neighbouring answer.
- `docs/OUTCOME_BIDDING.md` cites arXiv IDs from memory. Check them before
  citing externally.
- Sponsor names in the playground and fixtures were renamed to clearly coined
  ones after some sounded like real businesses. They have not been checked against trademark registers.

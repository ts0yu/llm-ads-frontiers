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
- The full test suite grew from 38 to 154 tests at the last full run. All pass,
  with none skipped for import reasons.

## What did not happen, and needs you

- **No live-model atlases yet.** This container cannot download weights.
  Copying the code to the rented H100 (`tal-qwen-h100`) was blocked by the
  auto-mode permission check as data exfiltration, so I stopped. Choose one:
  - Run on your Mac, with no approval needed:
    `bash scripts/run_moonshots.sh` (or `STAGES=text` first). Outputs are
    published into the explorer automatically.
  - Or tell a future session explicitly that copying this repository to the
    givemeanode H100 is approved.
- **GPU spend:** one wake of about 17 minutes (setup and downloads only),
  roughly $1.45 of your $10.59 balance. The node auto-stopped. Stopping it by
  hand was also blocked by the permission check; it stopped on its own idle
  timer.
- **Node environment:** the setup step downgraded `huggingface_hub` in the
  node's venv, which breaks its transformers 5.19 there. Before any node run,
  `pip install "transformers==4.57.6"` (matches the Mac smoke run) or
  reinstall `huggingface_hub>=1.31`.

## Caveats worth knowing

- All playground and census results use hand-written synthetic generators,
  not language models. Charges are simulated credits.
- `test_atlas_explorer.py` (a test file added tonight) skips bid pairs whose
  floating-point ratio `S*lambda/(S*lambda + S*(1-lambda))` differs from
  `lambda`. At an interval's exact upper boundary, which is a contest uniform,
  one ulp of rounding legitimately selects the neighbouring answer.
- `docs/OUTCOME_BIDDING.md` cites arXiv IDs from memory. Check them before
  citing externally.
- Sponsor names were renamed to clearly coined ones after one sounded like a
  real company. They have not been checked against trademark registers.

# Token Auction Lab

A two-bidder token-auction reference, shared-prefix LLM decoder, and local web interface.
Exactly two fictional advertisers use one model and tokenizer, with independent
prompt contexts and KV caches. Every selected token ID is fed to both contexts.
Bids stay fixed for a round and never enter either prompt. All charges are
simulated credits.

The mathematical sampler is unchanged from the supplied reference. The decoder
uses full-vocabulary probabilities at a fixed temperature, without argmax,
logit averaging, or top-k/top-p filtering. The top-token table is for inspection
only. See [MECHANISM.md](MECHANISM.md) for the payment derivation and its limits.

## Moonshots: exact bid atlases

A research extension, built on top of the unchanged two-bidder demo. For one
random seed, the auction's whole response to bids can be computed exactly:
every answer the bids can buy, at most 1 + T of them for a T-token answer, with
exact charges and outcome curves. Start here:

- [docs/LIVE_RESULTS.md](docs/LIVE_RESULTS.md): the first live-model atlases
  (Qwen3-4B text, three bidders, SDXL images; H100 run), with figures.
- [docs/ATLAS_NOTE.md](docs/ATLAS_NOTE.md): the research note, with results,
  proofs and nonclaims.
- [docs/SPAN_HOLD.md](docs/SPAN_HOLD.md): an opt-in variant (`--span-hold`) that
  keeps words and names whole, so a contest cannot land inside a brand name.
- [docs/WATERMARK.md](docs/WATERMARK.md): a keyed sponsor watermark (`--watermark
  demo`) for checking whose key shaped a text. It is reliable for a campaign's
  answers, not for a single answer.
- [MOONSHOTS.md](MOONSHOTS.md): the design, the JSON schemas and the labels.
- **Atlas Playground**, `/playground.html`: computed live in the browser on
  synthetic, non-LLM generators. It has a sponsored city walk, a two-label
  melody you can play, a three-sponsor bid triangle and an outcome-level
  bidding game.
- **Atlas Explorer**, `/atlas.html`: replays stored atlases (text, image and
  three-bidder), including the live-model runs.
- Generators with CLIs:
  - `python -m token_auction_lab.atlas` (two-bidder text);
  - `python -m token_auction_lab.simplex_atlas` (three-bidder text);
  - `python -m token_auction_lab.diffusion_atlas` (two-bidder SDXL images).
- `scripts/run_moonshots.sh` runs all live-model stages on a Mac, a CUDA
  machine or CPU.

## Open the local interface

From this checkout:

```bash
python -m token_auction_lab.walkthrough
```

Open **http://127.0.0.1:8765/**. Use `--port 8766` if needed. Stop the server
with Ctrl-C. The HTTP server uses Python's standard library, binds to loopback,
and serves the walkthrough, live inference API, and an allowlist of experiment
evidence. Reading the walkthrough needs no model, tokenizer, GPU, or downloads.

Seven guided chapters explain the question, auction, separate decoder contexts,
all 21 archived tests, recorded runs, hallucinations, numerical failures, and
remaining work. Click any recorded token to inspect its probabilities, random
tape, and payments. The frozen-prefix bid controls preview one alternative
decision from the recorded residual candidates; they never rewrite the trace or
generate a new answer. The separate three-token teaching example is labeled
synthetic. Original manifests, logs, and test sources open from evidence links.

The interface lives in `walkthrough/` and needs no JavaScript build or packages.
The original smoke report remains an immutable account of the 21-test run;
the current suite also includes walkthrough and live-service checks below.

### Live Lab

Open **http://127.0.0.1:8765/#live** to edit the shared prompt and instructions,
both advertisers' brands, facts, objectives and bids, seed, temperature, and
token cap. Changes mark the result stale. Only **Refresh inference** starts a
new model run; opening the page or editing a field does not. Inputs are locked
and a spinner is shown during generation and cache validation.
The saved defaults use the Fort Mason conference prompt, Bayview Hotel, and
Lantern House from the captured user run.

Each advertiser context receives the shared task and instructions, plus **only
its own brand, fact sheet, and objective**. Editing one advertiser's brief does
not change the other's starting prompt. Both still receive every generated
token, so information can cross through the shared response. Shared instructions
and the user task are visible to both; put private advertiser material in its
own panel. This matches the private-prompt structure of the paper's
[Section 5 demonstration](https://arxiv.org/html/2310.10826v3#S5.SS1).
It does not guarantee that brand tokens are contested or paid.

Live Lab and the CLI record `private-advertiser-briefs-v1` in new manifests.
The archived smoke runs used both fact sheets in both starting prompts; their
recordings and captured test output remain unchanged and describe that earlier
setup. Changing prompts changes the model distributions, not the auction math.

Live inference requires the inference dependencies below and a locally cached
`Qwen/Qwen3-4B-Instruct-2507` model and tokenizer at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. The smoke command below can populate
that cache. The web service uses `local_files_only=True`, loads the model on
the first refresh, and reuses it for subsequent runs. No paid infrastructure or
wallet is used.

There is one active set of inputs. Each refresh generates the ad response and
a separate prompt-only answer using the same model, tokenizer, seed, temperature,
and token cap. The no-ad prompt contains only the user's message: shared ad
instructions, advertiser facts, objectives, and bids are excluded. This also
removes any general constraints in those shared instructions; it is a reference
to the unconditioned user request, not an estimate of just the objective's effect.

Inspect each ad-selected token, the full vocabulary, random tape, and simulated
charges for both advertisers. **Ad mixture q** is the bid-weighted probability
distribution actually sampled by the auction; it is not the no-ad baseline.
**No-ad p** comes from an independent context that receives the same exact
auction-selected token IDs. The inspector compares distributions at that fixed
prefix, including percentage-point changes and full-vocabulary total variation.
The prefix can already contain ad content. This observer does not bid, sample,
or change the two-bidder mechanism. Its cache is separately validated.

The separate no-ad answer follows its own sampled prefix and never incurs
advertiser charges. Its text is useful for comparing whole responses, while the
observer is needed for comparing next-token probabilities at the same prefix.
Counterfactual bid controls freeze that token's prefix, distributions, and tape;
they leave the generated message and its charges unchanged. Setting both main
bids to zero generates only the prompt-only answer on the next refresh.

Manifests, traces, results, and failed attempts are saved in
`artifacts/live-lab/`, which is excluded from Git. Full probability vectors are
retained in memory for the active run's inspector and replaced on the next run.
Generated text is displayed without repair, including any hallucinations.
The separate no-ad manifest, trace, and result are saved in the run's `no-ad/`
directory. Main-run checkpoints include `p_no_ad` for the shared-prefix observer.

### Atlas Explorer

Open **http://127.0.0.1:8765/atlas.html** to replay a stored bid atlas (schema in
MOONSHOTS.md section 4): for one seed, every bid ratio λ and the exact answer,
charges, contest boundaries and outcome measures it produces. Two-bidder text and
image atlases use a λ scrubber; three-bidder `simplex-atlas-v1` atlases use a
draggable triangle. No inference runs on this page. The bundled atlases in
`walkthrough/atlas_data/` are labeled `synthetic_fixture`: toy weighted-corpus and
placeholder-image data written by `python -m scripts.make_atlas_fixtures`
(Pillow is needed only to redraw the placeholder JPEGs). They exercise the UI and
its arithmetic; they say nothing about model behaviour. The `model-test-*` atlases
are labeled `synthetic_model_test`: the real generators (`token_auction_lab.atlas`,
`simplex_atlas`, `diffusion_atlas`) run on tiny random models by
`python scripts/make_model_test_atlases.py`, so the text is word salad and the
images are noise by design.

To view real generator output, publish its `--output` directories:

```bash
python scripts/publish_atlases.py artifacts/moonshots/text/*/ artifacts/moonshots/simplex/*/ \
  artifacts/moonshots/image/*/ --prefix live- --max-image-side 768
python -m token_auction_lab.walkthrough --port 8790   # then open /atlas.html
```

This copies each atlas (and its images) to `walkthrough/atlas_data/<run-name>/` and
rewrites `index.json`, keeping other entries. Mode labels are copied verbatim; it only
fills display fields older runs lack, renames simplex `cells` to `leaves`, and records
every change in the file's `published` block. `bash scripts/run_moonshots.sh` runs the
live stages and this publish step. You can also open
`/atlas.html?atlas=<relative path>.json`. Image paths are relative to the atlas file.

## Install and test

The inference package requires Python 3.11 or newer. With `uv`:

```bash
uv sync --frozen --extra inference --extra dev
source .venv/bin/activate
python -m unittest -v
ruff check .
```

Alternatively, install into a Python 3.11+ virtual environment with
`python -m pip install -e '.[inference,dev]'`. `uv.lock` records exact resolved
versions; pip resolves within the ranges in `pyproject.toml`.

The CPU auction reference only needs `pip install -r requirements.txt`.
Without the inference dependencies, decoder tests are explicitly skipped.
The complete suite has 160 tests. The 38 Milestone 1 tests are described here:
all 11 original auction tests, two prompt tests,
and eleven decoder tests using tiny randomly initialized Qwen models. These
synthetic models test mechanics and cache correctness, not text quality.
Three additional walkthrough tests check archived evidence fidelity, the source
file allowlist, and browser counterfactuals against 312 Python-reference cases.
The last check uses Node.js and is explicitly skipped when Node is unavailable.
Eleven live-service tests cover editable prompts and bid isolation, validation,
immutable round inputs and concurrency, full-vector inspection, preservation of
failed attempts, and rejection of cross-origin inference requests. These use
synthetic fixtures and do not establish model quality.
Prompt checks retain the shared-task, own-facts, and bid-exclusion assertions;
the former shared-facts expectation is now private-brief isolation. New checks
cover both directions of private edits and common task/instruction edits.
The new checks verify that the no-ad observer matches a full-prefix forward,
leaves auction tokens/tapes/payments unchanged, includes no-ad candidates in the
display table, and preserves both outputs if the independent answer fails.
Seven Atlas Explorer tests check that the synthetic fixtures partition (0, 1] (or
tile the simplex) consistently with every recorded contest uniform, that the
`/atlas_data/` route serves only data files inside that folder, and (with Node)
that the browser's charges, (lo, hi] leaf lookup, outcome switch counts and
three-bidder race charges match a Python reference built on `auction.py`.
The remaining moonshot tests (`test_atlas.py`, `test_auction_n.py`,
`test_simplex_atlas.py`, `test_gaussian_auction.py`, `test_diffusion_atlas.py`,
`test_playground.py`, `test_mechanism_lab.py`, `test_publish_atlases.py`) cover
the atlas generators, the N-bidder race, the Gaussian coupling and the
browser playground; see MOONSHOTS.md.

## Run a live smoke test

The proposed model is
[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
This command freezes its revision and runs the three bid pairs `(1,0)`, `(1,1)`,
and `(0,1)`, followed by a separate no-ad baseline:

```bash
python -m token_auction_lab.cli \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --max-tokens 96 --temperature 0.8 --seed 7 --baseline \
  --output artifacts/smoke
```

The default device selection is CUDA, then Apple MPS, then CPU. MPS defaults to
float32 after the live float16 run failed cache validation. Override with
`--device` and `--dtype`. The model is loaded once for the bid-pair sweep.
Use `--scenario competing-hotels` for the second preset, and repeat `--bids A,B`
to specify a different sweep. The token cap must be between 1 and 96.
Zero/zero bids are only valid through the no-ad baseline.

Once the model is cached, add `--local-files-only` with the full revision SHA
to run without contacting Hugging Face. Model downloads stay in the normal
Hugging Face cache, outside the repository.

Each uniquely named round directory contains:

- `manifest.json`: model and tokenizer revision, dependency versions, prompts
  and prompt token IDs, source hashes, seeds, fixed bids, and generation settings.
- `events.jsonl`: exact token IDs and cumulative decoded text, auction branches,
  probabilities for displayed candidates, random tapes, and simulated charges.
- `checkpoint-*.npz`: full probability vectors and the frozen generated prefix
  for at most four selected steps. These are inspection data, not model weights.
- `result.json`: unchanged generated text, EOS/length stop reason, timing,
  cache-comparison errors, suffix verification, payments, and memory measurements.

The enclosing `summary.json` contains all completed rounds in the invocation.
An exception leaves any existing manifest and partial trace intact; a directory
without `result.json` is incomplete and must not be presented as a successful run.
Live inference is labeled `live_model` or `live_baseline`; synthetic test runs
use `synthetic_model_test`. Reading a stored trace for display must be labeled
REPLAY. A seed alone does not guarantee identical regeneration on other hardware.

Generation timing includes prefill, sequential forwards for both contexts,
sampling, synchronization, decoding, and trace/checkpoint I/O. The selected token
is fed to both contexts before its event is emitted. Cache validation is measured
separately. Process peak RSS is a process-lifetime high-water mark; CUDA allocator
peak is reset per round. MPS reports current and sampled allocator memory, not a
true transient device peak.

Measured live output and preserved failures are in [SMOKE_RESULTS.md](SMOKE_RESULTS.md).
The completed sweep and an eight-token length-cap run passed trace auditing:

```bash
python -m scripts.audit_smoke artifacts/smoke
```

This audit uses the recorded full checkpoints and cached tokenizer without
model inference. It prints a REPLAY AUDIT label. The source prompts and seeds
remain in each manifest; archived JSON evidence is under `smoke_results/`.

## Scope

Milestone 1 of [CODEX_BRIEF.md](CODEX_BRIEF.md) is demonstrated locally. The guided
walkthrough inspects its evidence, and Live Lab runs editable local inference
with token inspection and frozen-prefix counterfactuals. A credit reservation
ledger, broader evaluation sweep, and blockchain integration remain outside the
implemented scope.

This independent implementation covers a narrow two-bidder construction. It does
not establish arbitrary-utility truthful bidding, incentives over complete
answers, advertiser conversion, or production advertising economics. Generated
text is never repaired by a third model. Inspect and preserve failures.

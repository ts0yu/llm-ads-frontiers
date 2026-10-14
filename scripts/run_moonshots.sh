#!/usr/bin/env bash
# Live-model runs for the bid-atlas moonshots (see MOONSHOTS.md), then publish them
# into the Atlas Explorer.
#
# Runs on a Mac (MPS), a CUDA box or CPU from a checkout with the inference
# extra installed and the pinned Qwen revision cached. Each stage has its own
# time budget and writes results incrementally under $OUT. Stages whose module
# is missing are skipped. Nothing here spends money or touches a wallet.
#
#   bash scripts/run_moonshots.sh               # all stages, default budgets
#   STAGES="text" bash scripts/run_moonshots.sh  # only the text atlases
#   STAGES="publish" bash scripts/run_moonshots.sh  # only (re)publish existing outputs
#   OUT=artifacts/moonshots-mac bash scripts/run_moonshots.sh
#
# Then: python -m token_auction_lab.walkthrough --port 8790 and open /atlas.html.
# The image stage needs SDXL weights (downloaded unless cached) and, realistically,
# a CUDA GPU; IMAGE_DEVICE=mps is passed through but has not been tested.
set -uo pipefail

OUT=${OUT:-artifacts/moonshots}
STAGES=${STAGES:-"text simplex image publish"}
PYTHON=${PYTHON:-python}
QWEN_REV=cdbee75f17c01a7cc42f958dc650907174af0554
TEXT_SEEDS=${TEXT_SEEDS:-"$(seq -s ' ' 1 24)"}
SIMPLEX_SEEDS=${SIMPLEX_SEEDS:-"1 2 3 4"}
SIMPLEX_SCENARIO=${SIMPLEX_SCENARIO:-three-hotels}  # or fort-mason-three
IMAGE_SEEDS=${IMAGE_SEEDS:-"1 2 3"}
IMAGE_MODEL=${IMAGE_MODEL:-stabilityai/stable-diffusion-xl-base-1.0}
IMAGE_SCENES=${IMAGE_SCENES:-cafe}
IMAGE_DEVICE=${IMAGE_DEVICE:-}  # empty: the CLI default (cuda if available, else cpu)
TEXT_BUDGET=${TEXT_BUDGET:-1800}
SIMPLEX_BUDGET=${SIMPLEX_BUDGET:-1200}
IMAGE_BUDGET=${IMAGE_BUDGET:-1500}
DTYPE=${DTYPE:-float32}
PUBLISH_PREFIX=${PUBLISH_PREFIX:-live-}
PUBLISH_IMAGE_SIDE=${PUBLISH_IMAGE_SIDE:-768}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
has() { "$PYTHON" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)"; }
count() { set -- $1; echo $#; }

for stage in $STAGES; do
  case $stage in
    text)
      # name|arguments: one output directory per prompt set.
      for spec in "live-defaults|--live-defaults" "transit-hotel|--scenario transit-hotel" \
                  "competing-hotels|--scenario competing-hotels"; do
        name=${spec%%|*} target=${spec#*|}
        log "text atlas $name"
        # shellcheck disable=SC2086
        "$PYTHON" -m token_auction_lab.atlas $target --revision "$QWEN_REV" --local-files-only \
          --dtype "$DTYPE" --seeds $TEXT_SEEDS --max-tokens 96 --temperature 0.8 --verify 2 \
          --time-budget-seconds $((TEXT_BUDGET / 3)) --output "$OUT/text/$name" 2>&1 | tee -a "$OUT/text.log"
      done ;;
    simplex)
      if has token_auction_lab.simplex_atlas; then
        # The simplex CLI's budget is per seed.
        per_seed=$((SIMPLEX_BUDGET / $(count "$SIMPLEX_SEEDS")))
        log "three-bidder simplex atlas $SIMPLEX_SCENARIO (${per_seed}s per seed)"
        # shellcheck disable=SC2086
        "$PYTHON" -m token_auction_lab.simplex_atlas --scenario "$SIMPLEX_SCENARIO" \
          --revision "$QWEN_REV" --local-files-only --dtype "$DTYPE" \
          --seeds $SIMPLEX_SEEDS --max-tokens 64 --temperature 0.8 --verify 3 --max-cells 400 \
          --time-budget-seconds "$per_seed" --output "$OUT/simplex/$SIMPLEX_SCENARIO" 2>&1 | tee -a "$OUT/simplex.log"
      else log "skip simplex: module not present"; fi ;;
    image)
      if has token_auction_lab.diffusion_atlas; then
        for scene in $IMAGE_SCENES; do
          log "diffusion atlas $scene ($IMAGE_MODEL)"
          # shellcheck disable=SC2086
          "$PYTHON" -m token_auction_lab.diffusion_atlas --model "$IMAGE_MODEL" --scene "$scene" \
            --seeds $IMAGE_SEEDS --steps 30 --verify 1 --time-budget-seconds "$IMAGE_BUDGET" \
            ${IMAGE_DEVICE:+--device "$IMAGE_DEVICE"} --output "$OUT/image/$scene" 2>&1 | tee -a "$OUT/image.log"
        done
      else log "skip image: module not present"; fi ;;
    publish)
      # Copies every finished atlas into walkthrough/atlas_data/<prefix><stage>-<name>/ and
      # rewrites its index.json. Mode labels are copied verbatim; incomplete image atlases
      # are skipped (re-run publish_atlases.py with --allow-incomplete to show them).
      log "publish atlases into walkthrough/atlas_data"
      for kind in text simplex image; do
        [ -d "$OUT/$kind" ] || continue
        "$PYTHON" scripts/publish_atlases.py --skip-missing --prefix "$PUBLISH_PREFIX$kind-" \
          --max-image-side "$PUBLISH_IMAGE_SIDE" "$OUT/$kind"/*/ 2>&1 | tee -a "$OUT/publish.log"
      done ;;
    *) log "unknown stage $stage" ;;
  esac
done
log "done; results under $OUT; explorer: python -m token_auction_lab.walkthrough --port 8790, open /atlas.html"

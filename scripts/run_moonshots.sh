#!/usr/bin/env bash
# Live-model runs for the bid-atlas moonshots (see MOONSHOTS.md).
#
# Runs on a Mac (MPS), a CUDA box or CPU from a checkout with the inference
# extra installed and the pinned Qwen revision cached. Each stage has its own
# time budget and writes results incrementally under $OUT. Stages whose module
# is missing are skipped. Nothing here spends money or touches a wallet.
#
#   bash scripts/run_moonshots.sh               # all stages, default budgets
#   STAGES="text" bash scripts/run_moonshots.sh  # only the text atlases
#   OUT=artifacts/moonshots-mac bash scripts/run_moonshots.sh
set -uo pipefail

OUT=${OUT:-artifacts/moonshots}
STAGES=${STAGES:-"text simplex image"}
QWEN_REV=cdbee75f17c01a7cc42f958dc650907174af0554
TEXT_SEEDS=${TEXT_SEEDS:-"$(seq -s ' ' 1 24)"}
SIMPLEX_SEEDS=${SIMPLEX_SEEDS:-"1 2 3 4"}
IMAGE_SEEDS=${IMAGE_SEEDS:-"1 2 3"}
TEXT_BUDGET=${TEXT_BUDGET:-1800}
SIMPLEX_BUDGET=${SIMPLEX_BUDGET:-1200}
IMAGE_BUDGET=${IMAGE_BUDGET:-1500}
DTYPE=${DTYPE:-float32}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
has() { python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)"; }

for stage in $STAGES; do
  case $stage in
    text)
      for target in --live-defaults "--scenario transit-hotel" "--scenario competing-hotels"; do
        name=$(echo "$target" | tr -d '-' | tr ' ' '_')
        log "text atlas $target"
        # shellcheck disable=SC2086
        python -m token_auction_lab.atlas $target --revision "$QWEN_REV" --local-files-only \
          --dtype "$DTYPE" --seeds $TEXT_SEEDS --max-tokens 96 --temperature 0.8 --verify 2 \
          --time-budget-seconds $((TEXT_BUDGET / 3)) --output "$OUT/text/$name" 2>&1 | tee -a "$OUT/text.log"
      done ;;
    simplex)
      if has token_auction_lab.simplex_atlas; then
        log "three-bidder simplex atlas"
        # shellcheck disable=SC2086
        python -m token_auction_lab.simplex_atlas --revision "$QWEN_REV" --local-files-only \
          --seeds $SIMPLEX_SEEDS --max-tokens 64 --temperature 0.8 --verify 3 \
          --time-budget-seconds "$SIMPLEX_BUDGET" --output "$OUT/simplex" 2>&1 | tee -a "$OUT/simplex.log"
      else log "skip simplex: module not present"; fi ;;
    image)
      if has token_auction_lab.diffusion_atlas; then
        for scene in cafe; do
          log "diffusion atlas $scene"
          # shellcheck disable=SC2086
          python -m token_auction_lab.diffusion_atlas --scene "$scene" --seeds $IMAGE_SEEDS \
            --steps 30 --verify 1 --time-budget-seconds "$IMAGE_BUDGET" \
            --output "$OUT/image/$scene" 2>&1 | tee -a "$OUT/image.log"
        done
      else log "skip image: module not present"; fi ;;
  esac
done
log "done; results under $OUT"

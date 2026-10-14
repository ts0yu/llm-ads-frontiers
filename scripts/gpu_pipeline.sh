#!/usr/bin/env bash
# One-shot H100 pipeline for the bid-atlas moonshots.
#
# Spend guard: a hard wall-clock budget for the whole script and a budget per
# stage. Optional self-stop through the node metadata endpoint at the end. Each stage writes its results incrementally
# and tars them into ~/out/<stage>.tar so they can be exported at any time.
#
# Usage on the node: bash ~/tal/scripts/gpu_pipeline.sh [TOTAL_MINUTES]
set -uo pipefail

TOTAL_MIN=${1:-95}
START=$(date +%s)
DEADLINE=$((START + TOTAL_MIN * 60))
ROOT=~/tal
OUT=~/out
QWEN_REV=cdbee75f17c01a7cc42f958dc650907174af0554
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false

log() { echo "[$(date -u +%H:%M:%S) +$(( ($(date +%s) - START) / 60 ))m] $*" | tee -a "$OUT/pipeline.log"; }
left() { echo $((DEADLINE - $(date +%s))); }
status() { printf '{"stage":"%s","state":"%s","elapsed_s":%d}\n' "$1" "$2" $(( $(date +%s) - START )) > "$OUT/status.json"; }

# Runs "$@" for at most min(budget, time left - 4 min reserve) seconds.
stage() {
  local name=$1 budget=$2; shift 2
  local cap=$(( $(left) - 240 ))
  (( cap < budget )) && budget=$cap
  if (( budget < 120 )); then log "SKIP $name: not enough time left"; status "$name" skipped; return; fi
  log "START $name (budget ${budget}s)"; status "$name" running
  BUDGET=$budget timeout --signal=INT --kill-after=30 "$budget" "$@" >> "$OUT/$name.log" 2>&1
  local rc=$?
  log "END $name rc=$rc"; status "$name" "rc=$rc"
  [ -d "$OUT/$name" ] && tar -C "$OUT" -cf "$OUT/$name.tar" "$name" "$name.log" 2>/dev/null
}

setup() {
  python -m pip install -q "transformers>=4.51,<5" "diffusers>=0.35" accelerate safetensors \
    scipy shapely pillow hf_transfer "huggingface_hub[hf_transfer]" || return 1
  python -m pip install -q -e . --no-deps || return 1
  python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554")
snapshot_download("stabilityai/stable-diffusion-xl-base-1.0",
                  allow_patterns=["*.json", "*.txt", "tokenizer*/*", "*fp16.safetensors"])
snapshot_download("openai/clip-vit-large-patch14",
                  allow_patterns=["*.json", "*.txt", "model.safetensors"])
print("downloads ok")
EOF
}

bench() {
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  python -m token_auction_lab.cli --revision "$QWEN_REV" --local-files-only --device cuda \
    --dtype float32 --max-tokens 96 --temperature 0.8 --seed 7 --baseline --output "$OUT/bench"
}

log "pipeline start, total budget ${TOTAL_MIN} min"
stage setup 900 setup
stage bench 300 bench
# Stage commands are filled in from the module CLIs (see MOONSHOTS.md):
if [ -f "$ROOT/scripts/gpu_stages.sh" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/scripts/gpu_stages.sh"
fi
log "pipeline done; packing and self-stopping"
tar -C "$OUT" -cf ~/out-all.tar . 2>/dev/null
status done done
sync
# Default: leave the node to its 15-minute idle grace so results can be
# exported, then stop_node. SELF_STOP=1 parks it immediately instead.
if [ "${SELF_STOP:-0}" = 1 ]; then
  curl -s -X POST -H 'Metadata-Flavor: givemeanode' http://169.254.42.1/v1/stop || true
fi

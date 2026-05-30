#!/bin/bash
# Run ONE ablation on a single GPU, reading hidden states from the cache (no live vLLM server).
# Usage:
#   bash ablation/run.sh <name> <gpu_id> [-- <extra train.py args that override defaults>]
# Examples:
#   bash ablation/run.sh baseline-cache 0 -- --epochs 5
#   bash ablation/run.sh lr1e-3 1 -- --lr 0.001
#   bash ablation/run.sh layers-late 2 -- --target-layer-ids 25 29 34
set -euo pipefail
source "$(dirname "$0")/env.sh"

NAME="$1"; GPU="$2"; shift 2
[ "${1:-}" = "--" ] && shift

# Baseline defaults (epochs=2 for screening). Anything in "$@" overrides (argparse: last wins).
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/scripts/train.py" \
  --verifier-name-or-path "$VERIFIER" \
  --data-path "$DATASET" \
  --hidden-states-path "$CACHE_DIR" \
  --on-missing raise \
  --speculator-type dflash \
  --draft-vocab-size "$DRAFT_VOCAB" \
  --max-anchors "$MAX_ANCHORS" \
  --scheduler-type cosine \
  --target-layer-ids $BASELINE_AUX \
  --num-layers 5 \
  --lr 0.0006 \
  --epochs 2 \
  --seed 42 \
  --save-path "$REPO/output_dir/abl_ckpts/$NAME" \
  --logger wandb \
  --run-name "abl-$NAME" \
  "$@" 2>&1 | tee "$ABL/logs/$NAME.log"

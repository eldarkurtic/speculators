#!/bin/bash
# Build an MRCR acceptance cache for ONE length bucket.
# Usage: bash ablation/build_mrcr_cache.sh <bucket> <total_seq_len> <max_model_len> [max_samples]
#   e.g. bash ablation/build_mrcr_cache.sh 4k-8k 8192 12288 0   (0 = all rows in the bucket)
# Output: $REPO/output_dir/mrcr_<bucket>_dense
set -euo pipefail
source "$(dirname "$0")/env.sh"
BUCKET="$1"; TSL="$2"; MML="$3"; NMAX="${4:-0}"
FILE="/data-tier-1/machine/eldar/to_share/openai_mrcr/2needle/openai_mrcr_2needle_${BUCKET}.jsonl"

export BENCH="$REPO/output_dir/mrcr_${BUCKET}"
export DATASET="$BENCH" CACHE_DIR="$BENCH/hidden_states" HS_SCRATCH="$BENCH/hs_scratch"
export TOTAL_SEQ_LEN="$TSL" MAX_MODEL_LEN="$MML" REQUEST_TIMEOUT=900
export VOCAB_SRC="$REPO/output_dir/Qwen3-8B_magpie_5k_dense"
mkdir -p "$CACHE_DIR" "$HS_SCRATCH"

NMAX_ARG=""; [ "$NMAX" -gt 0 ] && NMAX_ARG="--max-samples $NMAX"
echo ">> [1/4] verifier continuations ($BUCKET, max_model_len=$MML) ..."
"$VLLM_PY" "$ABL/_gen_mrcr.py" --file "$FILE" --out "$BENCH/continuations.jsonl" \
  --k 512 --max-model-len "$MML" --bucket "$BUCKET" $NMAX_ARG

echo ">> [2/4] arrow (TOTAL_SEQ_LEN=$TSL) ..."
"$PY" "$ABL/_build_benchmark_arrow.py"

echo ">> [3/4] aux hidden states (max_model_len=$MML) ..."
bash "$ABL/gen_cache.sh"

echo ">> [4/4] validate + densify ..."
"$PY" "$ABL/_validate_cache.py" --dir "$CACHE_DIR" --delete || true
"$PY" "$ABL/_densify_cache.py"
echo ">> DONE -> ${BENCH}_dense"

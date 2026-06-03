#!/bin/bash
# Build the acceptance-benchmark cache end-to-end:
#   1. verifier greedy continuations (vLLM offline, .venv_vllm)
#   2. arrow dataset in training format (.venv)
#   3. aux hidden-state extraction (reuses gen_cache.sh; superset layers 1 9 17 25 34 36)
#   4. validate + densify (eval uses on_missing=raise -> needs a gap-free cache)
# Usage:  bash ablation/build_benchmark_cache.sh [--thinking]
#   --thinking  enable Qwen3 reasoning (default OFF; else the 256-tok budget is eaten by <think>)
set -euo pipefail
source "$(dirname "$0")/env.sh"
BENCH="$REPO/output_dir/Qwen3-8B_bench"
THINK="${1:-}"

echo ">> [1/4] verifier greedy continuations (vLLM offline) ..."
CUDA_VISIBLE_DEVICES=0 "$VLLM_PY" "$ABL/_gen_benchmark.py" $THINK --out "$BENCH/continuations.jsonl"

echo ">> [2/4] building arrow dataset ..."
"$PY" "$ABL/_build_benchmark_arrow.py"

echo ">> [3/4] extracting aux hidden states (DATASET=bench via gen_cache.sh) ..."
DATASET="$BENCH" REQUEST_TIMEOUT=600 GEN_CONCURRENCY=16 bash "$ABL/gen_cache.sh"

echo ">> [4/4] validate + densify ..."
"$PY" "$ABL/_validate_cache.py" --dir "$BENCH/hidden_states" --delete || true
DATASET="$BENCH" CACHE_DIR="$BENCH/hidden_states" "$PY" "$ABL/_densify_cache.py"
echo ">> DONE -> ${BENCH}_dense"
echo ">> eval:  .venv/bin/python ablation/eval_acceptance.py --ckpt output_dir/abl_ckpts/<run>/checkpoint_best"

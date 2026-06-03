#!/bin/bash
# Build the acceptance-benchmark cache end-to-end (one per verifier):
#   1. verifier greedy continuations (vLLM offline, .venv_vllm)
#   2. arrow dataset in training format (.venv)
#   3. aux hidden-state extraction (reuses gen_cache.sh; SUPERSET_LAYERS)
#   4. validate + densify (eval uses on_missing=raise -> needs a gap-free cache)
# Output: ${BENCH}_dense   (BENCH is set by env.sh from your VERIFIER)
# Usage:  bash build_benchmark_cache.sh [--thinking]
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo ">> [1/4] verifier greedy continuations (vLLM offline) ..."
"$VLLM_PY" "$ABL/_gen_benchmark.py" ${1:+"$1"} --out "$BENCH/continuations.jsonl"

echo ">> [2/4] building arrow dataset ..."
"$PY" "$ABL/_build_benchmark_arrow.py"

echo ">> [3/4] extracting aux hidden states (gen_cache.sh over BENCH) ..."
bash "$ABL/gen_cache.sh"

echo ">> [4/4] validate + densify ..."
"$PY" "$ABL/_validate_cache.py" --dir "$BENCH/hidden_states" --delete || true
"$PY" "$ABL/_densify_cache.py"
echo ">> DONE -> ${BENCH}_dense"
echo ">> eval:  .venv/bin/python $ABL/eval_acceptance.py --ckpt <draft>/checkpoint_best --bench ${BENCH}_dense"

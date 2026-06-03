#!/bin/bash
# Central config for the acceptance-eval bundle. Sourced by every script.
# >>> EDIT the four lines marked EDIT for your box, then follow README.md. <<<

# EDIT: your speculators repo (provides .venv, .venv_vllm, scripts/launch_vllm.py,
#       scripts/data_generation_offline.py, and output_dir/ for the cache).
export REPO="${REPO:-/path/to/your/speculators}"
# EDIT: your verifier / target model (the one your drafts speculate for).
export VERIFIER="${VERIFIER:-/path/to/Qwen3-8B}"
# EDIT: cache layer ids stored per sample; the LAST id is the verifier TARGET layer.
#       Must match what your drafts were trained with.
export SUPERSET_LAYERS="${SUPERSET_LAYERS:-1 9 17 25 34 36}"
# EDIT: the aux layers your draft fuses (a subset of SUPERSET_LAYERS, without the target).
export BASELINE_AUX="${BASELINE_AUX:-1 9 17 25 34}"

# --- usually no edit below ---
export PY="${PY:-$REPO/.venv/bin/python}"                  # training venv (speculators+torch; NO vllm)
export VLLM_PY="${VLLM_PY:-$REPO/.venv_vllm/bin/python}"   # vllm venv (HS server + continuations)
# this bundle's own dir (self-locating, so it can live anywhere):
export ABL="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# where the benchmark cache is built (one per verifier):
export BENCH="${BENCH:-$REPO/output_dir/$(basename "$VERIFIER")_bench}"
# dir containing the d2t.npy/t2d.npy/token_freq.pt your drafts were trained with
# (e.g. your training dataset dir). EDIT if your vocab map lives elsewhere.
export VOCAB_SRC="${VOCAB_SRC:-$REPO/output_dir}"
export DATASET="$BENCH"                                    # gen_cache.sh/densify operate on the bench set
export CACHE_DIR="$BENCH/hidden_states"
export HS_SCRATCH="$BENCH/hs_scratch"
export MAX_ANCHORS="${MAX_ANCHORS:-3072}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"           # raise the 15s default; avoids dropped samples
export GEN_CONCURRENCY="${GEN_CONCURRENCY:-16}"

mkdir -p "$ABL/logs" "$CACHE_DIR" "$HS_SCRATCH"

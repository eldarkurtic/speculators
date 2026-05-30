#!/bin/bash
# Central paths/config for the DFlash ablation study. Source this in every ablation script:
#   source "$(dirname "$0")/env.sh"
# If the repo or model paths differ on a new server, edit REPO / VERIFIER below (only these two).

export REPO="${REPO:-/home/eldarkurtic/github/speculators}"
export PY="$REPO/venv_spec/bin/python"                       # training venv (editable speculators)
export VERIFIER="${VERIFIER:-/home/eldarkurtic/hf_models/Qwen/Qwen3-8B}"

# Dataset: NON-FP8 magpie 5k (user switched from the FP8 variant).
export DATASET="$REPO/output_dir/Qwen3-8B_magpie_5k"
export CACHE_DIR="$DATASET/hidden_states"                    # hs_0..hs_4999.safetensors live here
export HS_SCRATCH="$DATASET/hs_scratch"                      # vLLM server scratch dir

# Hidden-state superset: last id (36) becomes the verifier target (loss); the first 9 are the
# aux-layer pool the draft fc() fuses. Baseline aux = 1 9 17 25 34 (all present in this superset).
export SUPERSET_LAYERS="1 5 9 13 17 21 25 29 34 36"
export BASELINE_AUX="1 9 17 25 34"

export DRAFT_VOCAB=32000
export MAX_ANCHORS=3072
export ABL="$REPO/ablation"

mkdir -p "$ABL/logs" "$CACHE_DIR" "$HS_SCRATCH"

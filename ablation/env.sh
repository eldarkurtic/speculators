#!/bin/bash
# Central paths/config for the DFlash ablation study. Source this in every ablation script:
#   source "$(dirname "$0")/env.sh"
# If the repo or model paths differ on a new server, edit REPO / VERIFIER below (only these two).

export REPO="${REPO:-/home/eldarkurtic/github/eldarkurtic/speculators}"
export PY="$REPO/.venv/bin/python"                           # training venv (editable speculators; NO vllm)
# vLLM lives in a separate env. launch_vllm.py execs `sys.executable -m vllm...`, so the HS SERVER
# must run under a vllm-capable python. The data-gen CLIENT (data_generation_offline.py) needs
# speculators+openai (not vllm) → runs under $PY. Override VLLM_PY if it moves.
export VLLM_PY="${VLLM_PY:-$REPO/.venv_vllm/bin/python}"
export VERIFIER="${VERIFIER:-/home/eldarkurtic/hf_models/Qwen/Qwen3-8B}"

# Dataset: NON-FP8 magpie 5k (user switched from the FP8 variant). The offline generator skipped
# ~28% of samples ("incomplete metadata"), so we use the DENSE variant: the 3613 rows that have
# hidden states, re-indexed contiguously with symlinked hs files (built by _densify_cache.py). This
# lets training use --on-missing raise (no gaps -> no empty-batch crash). To regenerate the cache
# from scratch, point DATASET back to ".../Qwen3-8B_magpie_5k" and rerun gen_cache.sh.
export DATASET="${DATASET:-$REPO/output_dir/Qwen3-8B_magpie_5k_dense}"
export CACHE_DIR="$DATASET/hidden_states"                    # contiguous hs_0..hs_3612 (symlinks)
export HS_SCRATCH="$DATASET/hs_scratch"                      # vLLM server scratch dir (gen only)

# Hidden-state superset: last id (36) becomes the verifier target (loss); the rest are the aux-layer
# pool the draft fc() fuses. For the ARCHITECTURE ablation we hold the layer set fixed at the baseline
# (aux 1 9 17 25 34 + target 36), so the cache is exactly these 6 layers (~1TB) and every arch variant
# reads identical inputs. (A wider superset is only needed for a layer-selection sweep, which we dropped.)
export SUPERSET_LAYERS="1 9 17 25 34 36"
export BASELINE_AUX="1 9 17 25 34"

export DRAFT_VOCAB=32000
export MAX_ANCHORS=3072
export ABL="$REPO/ablation"

# wandb: group every ablation run under one project so they're trackable together. run.sh passes
# --logger wandb; the WandbHandler doesn't set project, so wandb.init() picks these env vars up.
export WANDB_PROJECT="${WANDB_PROJECT:-speculators-scripts-v2}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-arch-ablation}"

mkdir -p "$ABL/logs" "$CACHE_DIR" "$HS_SCRATCH"

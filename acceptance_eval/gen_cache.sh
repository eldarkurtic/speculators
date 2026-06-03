#!/bin/bash
# Phase 1: generate the hidden-state cache ONCE (superset layers) so all training runs read from disk.
# Usage:
#   bash ablation/gen_cache.sh            # full dataset (5000 samples) -> ~1.8TB
#   bash ablation/gen_cache.sh 8          # dry-run on first 8 samples (validate pipeline first!)
#
# Flow: launch vLLM hidden-state server (dp=8) -> offline-generate + validate -> kill server.
# NOTE: server flags mirror the user's working vllm_serve_for_hidden_states.sh. Verify on first run.
set -euo pipefail
source "$(dirname "$0")/env.sh"

MAX_SAMPLES="${1:-}"          # empty => all
SERVER_LOG="$ABL/logs/hs_server.log"

echo ">> Launching vLLM hidden-state server (layers: $SUPERSET_LAYERS) via VLLM_PY ..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$VLLM_PY" "$REPO/scripts/launch_vllm.py" "$VERIFIER" \
  --hidden-states-path "$HS_SCRATCH" \
  --target-layer-ids $SUPERSET_LAYERS \
  -- --max-model-len 12288 -tp 1 --data-parallel-size 8 --gpu-memory-utilization 0.9 \
     --no-enable-chunked-prefill \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'echo ">> Killing server $SERVER_PID"; kill $SERVER_PID 2>/dev/null || true; pkill -f "VLLM::Worker" 2>/dev/null || true' EXIT

echo ">> Waiting for server on :8000 (log: $SERVER_LOG) ..."
for i in $(seq 1 120); do
  if curl -s -m 3 http://localhost:8000/v1/models >/dev/null 2>&1; then echo ">> Server up."; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "!! Server died early; see $SERVER_LOG"; exit 1; fi
  sleep 10
done

echo ">> Generating hidden states -> $CACHE_DIR ..."
# request-timeout default raised to 600s: the stock 15s caused ~28% of samples to
# time out under dp=8 + concurrency-32 (4k-token prefills queue past 15s) and get
# skipped. Override via REQUEST_TIMEOUT / GEN_CONCURRENCY env vars.
GEN_ARGS=(--model "$VERIFIER" --endpoint http://localhost:8000/v1 \
          --preprocessed-data "$DATASET" --output "$CACHE_DIR" --validate-outputs \
          --request-timeout "${REQUEST_TIMEOUT:-600}" --concurrency "${GEN_CONCURRENCY:-32}")
[ -n "$MAX_SAMPLES" ] && GEN_ARGS+=(--max-samples "$MAX_SAMPLES")
"$PY" "$REPO/scripts/data_generation_offline.py" "${GEN_ARGS[@]}" 2>&1 | tee "$ABL/logs/gen_client.log"

echo ">> Done. Cache files: $(ls "$CACHE_DIR" 2>/dev/null | grep -c '^hs_.*\.safetensors$' || echo 0)"
echo ">> Sanity-check a sample shape with: $PY ablation/check_cache.py"

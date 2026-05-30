#!/bin/bash
# Fan out a queue of ablations across N GPUs (default 8), 1 run per GPU, launching the next as each
# GPU frees. Reads ablation/queue.txt; each non-comment line is:  <name> | <extra train.py args>
# Usage:  bash ablation/queue.sh [num_gpus]   (default 8)
set -uo pipefail
source "$(dirname "$0")/env.sh"

NGPU="${1:-8}"
QUEUE="$ABL/queue.txt"
[ -f "$QUEUE" ] || { echo "no $QUEUE"; exit 1; }

# Build job list (skip blank/# lines).
mapfile -t JOBS < <(grep -vE '^\s*(#|$)' "$QUEUE")
echo ">> ${#JOBS[@]} jobs across $NGPU GPUs"

declare -A GPU_PID   # gpu_id -> bg pid

launch() { # $1=gpu  $2=job-line
  local gpu="$1" line="$2"
  local name="${line%%|*}"; name="$(echo "$name" | xargs)"
  local args="${line#*|}"
  echo ">> [gpu $gpu] $name :: $args"
  bash "$ABL/run.sh" "$name" "$gpu" -- $args > /dev/null 2>&1 &
  GPU_PID[$gpu]=$!
}

ji=0
# Prime each GPU.
for ((g=0; g<NGPU && ji<${#JOBS[@]}; g++)); do launch "$g" "${JOBS[$ji]}"; ji=$((ji+1)); done
# As any job finishes, launch the next on that GPU.
while ((ji < ${#JOBS[@]})) || ((${#GPU_PID[@]} > 0)); do
  for g in "${!GPU_PID[@]}"; do
    if ! kill -0 "${GPU_PID[$g]}" 2>/dev/null; then
      wait "${GPU_PID[$g]}" 2>/dev/null
      unset 'GPU_PID[$g]'
      if ((ji < ${#JOBS[@]})); then launch "$g" "${JOBS[$ji]}"; ji=$((ji+1)); fi
    fi
  done
  sleep 15
done
echo ">> queue drained. Summarize: $PY ablation/eal.py --scan ablation/logs"

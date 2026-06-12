#!/usr/bin/env bash
# Sequential V2 training on GPU 1: breakout then boxing (300 min each, same as pong_v2).
set -uo pipefail

export HOME="${HOME:-/mnt/server12_hard0/kiseol}"
export CUDA_VISIBLE_DEVICES=1
export DISPLAY="${DISPLAY:-:99}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${HOME}/.conda/envs/dreamerv3/bin/python"
MINUTES="${V2_MINUTES:-300}"
MEM="${V2_MEM_FRACTION:-0.85}"
LOGDIR="${PROJECT}/logs/atari_train_4gpu"
mkdir -p "$LOGDIR"

cd "$PROJECT"
export PYTHONPATH="${PROJECT}/vendor/dreamerv3:${PROJECT}:${PYTHONPATH:-}"

# Ensure Minecraft stays off
bash scripts/pause_minecraft.sh >> "$LOGDIR/v2_gpu1.log" 2>&1

echo "=== V2 GPU1 pipeline start $(date) minutes=${MINUTES} ===" | tee -a "$LOGDIR/v2_gpu1.log"

run_one() {
  local game="$1"
  local log="$LOGDIR/v2_gpu1_${game}.log"
  echo "=== ${game}_v2 start $(date) ===" | tee -a "$LOGDIR/v2_gpu1.log" "$log"
  if ! "$PYTHON" -u scripts/run_atari_single_job.py \
      --gpu 1 --version v2 --game "$game" \
      --minutes "$MINUTES" --mem-fraction "$MEM" 2>&1 | tee -a "$log"; then
    echo "=== ${game}_v2 FAILED $(date) ===" | tee -a "$LOGDIR/v2_gpu1.log"
    return 1
  fi
  echo "=== ${game}_v2 done $(date) ===" | tee -a "$LOGDIR/v2_gpu1.log"
}

run_one breakout
run_one boxing

echo "=== V2 GPU1 pipeline finished $(date) ===" | tee -a "$LOGDIR/v2_gpu1.log"

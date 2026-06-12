#!/usr/bin/env bash
# Wait for in-flight V3 jobs, then run pending V2 jobs with fixed config.
set -uo pipefail

export HOME="${HOME:-/mnt/server12_hard0/kiseol}"
export DISPLAY="${DISPLAY:-:99}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${HOME}/.conda/envs/dreamerv3/bin/python"
LOGDIR="${PROJECT}/logs/atari_train_4gpu"
mkdir -p "$LOGDIR"
cd "$PROJECT"
export PYTHONPATH="${PROJECT}/vendor/dreamerv3:${PROJECT}:${PYTHONPATH:-}"

PONG_V3_PID="${1:-1799193}"
BREAKOUT_V3_PID="${2:-1799194}"
MEM="${3:-0.42}"
MINUTES="${4:-300}"

run_v2() {
  local gpu="$1"
  local game="$2"
  local log="$LOGDIR/gpu${gpu}_v2_sidecar.log"
  echo "=== GPU${gpu} ${game}_v2 start $(date) ===" | tee -a "$log"
  "$PYTHON" -u scripts/run_atari_single_job.py \
    --gpu "$gpu" --version v2 --game "$game" \
    --minutes "$MINUTES" --mem-fraction "$MEM" 2>&1 | tee -a "$log"
  echo "=== GPU${gpu} ${game}_v2 end $(date) exit=$? ===" | tee -a "$log"
}

(
  echo "sidecar: waiting for pong_v3 pid $PONG_V3_PID"
  while kill -0 "$PONG_V3_PID" 2>/dev/null; do sleep 30; done
  echo "sidecar: pong_v3 done, starting breakout_v2 on GPU2"
  run_v2 2 breakout
) >> "$LOGDIR/sidecar_gpu2.log" 2>&1 &

(
  echo "sidecar: waiting for breakout_v3 pid $BREAKOUT_V3_PID"
  while kill -0 "$BREAKOUT_V3_PID" 2>/dev/null; do sleep 30; done
  echo "sidecar: breakout_v3 done, starting boxing_v2 on GPU4"
  run_v2 4 boxing
) >> "$LOGDIR/sidecar_gpu4.log" 2>&1 &

echo "V2 sidecars launched (GPU2 breakout, GPU4 boxing)"

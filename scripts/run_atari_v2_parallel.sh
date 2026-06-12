#!/usr/bin/env bash
# Parallel V2: breakout on GPU1, boxing on GPU2 (300 min each).
set -uo pipefail

export HOME="${HOME:-/mnt/server12_hard0/kiseol}"
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

bash scripts/pause_minecraft.sh >> "$LOGDIR/v2_parallel.log" 2>&1

launch() {
  local gpu="$1"
  local game="$2"
  local log="$LOGDIR/v2_gpu${gpu}_${game}.log"
  echo "=== ${game}_v2 GPU${gpu} start $(date) ${MINUTES}min ===" | tee -a "$LOGDIR/v2_parallel.log" "$log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -u scripts/run_atari_single_job.py \
    --gpu "$gpu" --version v2 --game "$game" \
    --minutes "$MINUTES" --mem-fraction "$MEM" >> "$log" 2>&1 &
  echo "  pid=$!" | tee -a "$LOGDIR/v2_parallel.log"
}

# Skip if already running
if pgrep -f "dreamerv2/train.py.*atari_breakout_v2" >/dev/null; then
  echo "breakout_v2 already running, skip launch" | tee -a "$LOGDIR/v2_parallel.log"
else
  launch 1 breakout
fi

if pgrep -f "dreamerv2/train.py.*atari_boxing_v2" >/dev/null; then
  echo "boxing_v2 already running, skip launch" | tee -a "$LOGDIR/v2_parallel.log"
else
  launch 2 boxing
fi

echo "=== parallel V2 launched $(date) ===" | tee -a "$LOGDIR/v2_parallel.log"

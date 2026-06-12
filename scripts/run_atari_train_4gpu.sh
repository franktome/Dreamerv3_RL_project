#!/usr/bin/env bash
# Launch 6×300min Atari jobs across GPUs 2–5 (does not touch GPU 1 / Minecraft).
set -uo pipefail

export HOME="${HOME:-/mnt/server12_hard0/kiseol}"
export DISPLAY="${DISPLAY:-:99}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-${HOME}/.conda/envs/dreamerv3/bin/python}"
[ -x "$PYTHON" ] || PYTHON="/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3/bin/python"

mkdir -p "${PROJECT}/logs/atari_train_4gpu"
cd "$PROJECT"
export PYTHONPATH="${PROJECT}/vendor/dreamerv3:${PROJECT}:${PYTHONPATH:-}"

MASTER_LOG="${PROJECT}/logs/atari_train_4gpu/master.log"
echo "=== atari 4gpu train start $(date) ===" | tee "$MASTER_LOG"

exec "$PYTHON" -u scripts/run_atari_train_4gpu.py \
  --minutes 300 \
  --mem-fraction 0.42 \
  "$@" 2>&1 | tee -a "$MASTER_LOG"

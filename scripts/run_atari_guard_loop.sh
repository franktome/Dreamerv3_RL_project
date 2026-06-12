#!/usr/bin/env bash
# Periodic guard: keep Minecraft paused, heal stale Atari jobs.
set -uo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${HOME:-/mnt/server12_hard0/kiseol}/.conda/envs/dreamerv3/bin/python"
INTERVAL="${ATARI_GUARD_INTERVAL:-900}"

cd "$PROJECT"
export PYTHONPATH="${PROJECT}/vendor/dreamerv3:${PROJECT}:${PYTHONPATH:-}"

while true; do
  "$PYTHON" -u scripts/run_atari_guard.py 2>&1 | tee -a logs/atari_train_4gpu/guard_loop.log
  sleep "$INTERVAL"
done

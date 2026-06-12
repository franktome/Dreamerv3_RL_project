#!/usr/bin/env bash
# 12h pipeline: smoke → 3 games × (V2 90m + V3 90m) → infer + anim + report
set -uo pipefail

export HOME="${HOME:-/mnt/server12_hard0/kiseol}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DISPLAY="${DISPLAY:-:99}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-${HOME}/.conda/envs/dreamerv3/bin/python}"
[ -x "$PYTHON" ] || PYTHON="/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3/bin/python"
LOG="${PROJECT}/logs/atari_compare_12h.log"
mkdir -p "${PROJECT}/logs" "${PROJECT}/report/atari_compare"

cd "$PROJECT"
export PYTHONPATH="${PROJECT}/vendor/dreamerv3:${PROJECT}:${PYTHONPATH:-}"

echo "=== atari compare start $(date) ===" | tee "$LOG"

"$PYTHON" scripts/smoke_atari_compare.py --gpu 1 --mem-fraction "${XLA_PYTHON_CLIENT_MEM_FRACTION}" 2>&1 | tee -a "$LOG" || exit 1

"$PYTHON" -u - <<'PY' 2>&1 | tee -a "$LOG"
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from seollab.paths import default_paths
from seollab import atari_compare

cfg = default_paths(Path('.'), gpu='1')
cfg.apply_env(mem_fraction=float(__import__('os').environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.25')))

# Full training (skip if checkpoints exist)
atari_compare.run_timed_training(cfg, minutes_per_run=90, skip_existing=True)

# Inference + anim + report
result = atari_compare.run_full_pipeline(cfg, smoke=False, skip_train=True)
print('Report:', result.get('report'))
print('Done:', result.get('finished'))
PY

echo "=== atari compare end $(date) ===" | tee -a "$LOG"

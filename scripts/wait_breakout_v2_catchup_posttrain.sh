#!/usr/bin/env bash
# Wait for breakout V2 catchup, regenerate alignment/GIFs, sync notebook inputs.
set -u
WS=/mnt/server12_hard0/kiseol/Dreamerv3
PY=/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3/bin/python
LOGS=$WS/logs/atari_breakout_v2_catchup
SP=/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3/lib/python3.11/site-packages
export LD_LIBRARY_PATH="$WS/.cache/cudnn8/nvidia/cudnn/lib:$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')"

echo "$(date '+%F %T') waiting for breakout V2 catchup..."
while true; do
  pid=$(cat "$LOGS/breakout_v2.pid" 2>/dev/null || echo "")
  alive=0
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then alive=1; fi
  n=$(ps aux | grep -cE '[d]reamerv2/train.py.*atari_breakout' || true)
  if [ "$alive" -eq 0 ] && [ "$n" -eq 0 ]; then
    echo "$(date '+%F %T') training finished"
    break
  fi
  sleep 120
done

echo "$(date '+%F %T') posttrain (alignment + GIFs + reports)..."
"$PY" "$WS/scripts/run_fair_atari_posttrain.py" 2>&1 | tee "$LOGS/posttrain.log"

echo "$(date '+%F %T') notebook cells 22/24 equivalent..."
"$PY" "$WS/scripts/run_notebook_atari_cells.py" 2>&1 | tee "$LOGS/notebook_cells.log"

echo "$(date '+%F %T') HF upload (post-catchup, full replace)..."
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" "$PY" "$WS/scripts/upload_fair_atari_hf.py" \
  --label "Breakout V2 catchup" 2>&1 | tee "$LOGS/hf_upload_after_catchup.log"

echo "$(date '+%F %T') catchup pipeline complete"

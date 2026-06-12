#!/usr/bin/env bash
# Wait for 6 extend jobs, then posttrain + wallclock + HF + notebook sync.
set -u
WS=/mnt/server12_hard0/kiseol/Dreamerv3
PY=/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3/bin/python
LOGS=$WS/logs/atari_fair_extend
SP=/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3/lib/python3.11/site-packages
export LD_LIBRARY_PATH="$WS/.cache/cudnn8/nvidia/cudnn/lib:$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')"

echo "$(date '+%F %T') waiting for extend jobs..."
while true; do
  alive=0
  for f in pong_v2 breakout_v2 boxing_v2 pong_v3 breakout_v3 boxing_v3; do
    pid=$(cat "$LOGS/${f}.pid" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  if [ "$alive" -eq 0 ]; then
    # also check train subprocesses
    n=$(ps aux | grep -cE '[d]reamerv2/train.*atari_(pong|breakout|boxing)|[d]reamerv3/main.py.*atari_(pong|breakout|boxing)' || true)
    if [ "$n" -eq 0 ]; then
      echo "$(date '+%F %T') all jobs finished"
      break
    fi
  fi
  sleep 120
done

echo "$(date '+%F %T') running posttrain..."
"$PY" "$WS/scripts/run_fair_atari_posttrain.py" 2>&1 | tee "$LOGS/posttrain.log"
"$PY" "$WS/scripts/compute_atari_wallclock.py" 2>&1 | tee -a "$LOGS/posttrain.log"

echo "$(date '+%F %T') HF upload..."
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" "$PY" "$WS/scripts/upload_fair_atari_hf.py" 2>&1 | tee "$LOGS/hf_upload.log"

# Sync notebook + code to RL project and push
DST=/mnt/server12_hard0/kiseol/Dreamerv3_RL_project
rsync -a "$WS/seollab/" "$DST/seollab/"
rsync -a "$WS/dvbench/" "$DST/dvbench/"
rsync -a "$WS/dreamerv3_team11.ipynb" "$DST/"
rsync -a "$WS/scripts/" "$DST/scripts/"
rsync -a "$WS/report/atari_compare/" "$DST/report/atari_compare/" 2>/dev/null || true

cd "$DST" && git add dreamerv3_team11.ipynb dvbench/ seollab/ scripts/ 2>/dev/null
git add -f report/atari_compare/*.md report/atari_compare/*.json 2>/dev/null || true
git -c user.name="kiseol" -c user.email="kiseol@users.noreply.github.com" commit -m "$(cat <<'EOF'
Update fair Atari extended training results and wall-clock report.

500k-step extended run: alignment, metrics, V2/V3 wall-clock comparison, and refreshed checkpoints.
EOF
)" 2>/dev/null || echo "nothing to commit"
git push origin dreamerv2-v3 2>&1 | tee "$LOGS/git_push.log"

echo "$(date '+%F %T') pipeline complete"

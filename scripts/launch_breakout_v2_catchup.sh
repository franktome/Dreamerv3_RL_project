#!/usr/bin/env bash
# Resume breakout V2 until env-step cap matches V3 (fair GIF at min(V2,V3)).
set -u
WS=/mnt/server12_hard0/kiseol/Dreamerv3
PY=${PY:-/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3/bin/python}
LOGS=$WS/logs/atari_breakout_v2_catchup
mkdir -p "$LOGS"

STEPS=500000
MINUTES=420   # ~7h buffer (extend rate ~28k env steps/h → ~6h for +165k–176k)

SP=$(dirname "$PY")/../lib/python3.11/site-packages
export LD_LIBRARY_PATH="$WS/.cache/cudnn8/nvidia/cudnn/lib:$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup "$PY" "$WS/scripts/run_atari_single_job.py" \
  --gpu 0 --version v2 --game breakout \
  --minutes "$MINUTES" --steps "$STEPS" --mem-fraction 0.25 --jit True \
  >> "$LOGS/breakout_v2.log" 2>&1 &
echo "launched breakout_v2 catchup on GPU0 pid=$! (${MINUTES}min cap, ${STEPS} env steps)"
echo "$!" > "$LOGS/breakout_v2.pid"

nohup "$PY" "$WS/scripts/v2_ckpt_sidecar.py" --games breakout \
  --interval 600 --hours 7.5 \
  >> "$LOGS/v2_sidecar.log" 2>&1 &
echo "sidecar pid=$!"
echo "$!" > "$LOGS/v2_sidecar.pid"

echo "$(date '+%F %T') catchup started" | tee "$LOGS/started.txt"

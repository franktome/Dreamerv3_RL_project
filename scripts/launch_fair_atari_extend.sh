#!/usr/bin/env bash
# Resume fair Atari training: +8h wall-clock, unified env-step cap per game.
# GPU0: V2 x3 (TF)  |  GPU1: V3 x3 (JAX)  — same --steps for V2/V3 on each game.
set -u
WS=/mnt/server12_hard0/kiseol/Dreamerv3
PY=${PY:-/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3/bin/python}
LOGS=$WS/logs/atari_fair_extend
mkdir -p "$LOGS"

# Unified cap (env interactions, post action-repeat). ~100k done; +8h targets ~400–500k.
STEPS=500000
MINUTES=480   # 8 hours

SP=$(dirname "$PY")/../lib/python3.11/site-packages
export LD_LIBRARY_PATH="$WS/.cache/cudnn8/nvidia/cudnn/lib:$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

launch() {
  local gpu=$1 ver=$2 game=$3
  local extra=()
  [ "$ver" = v2 ] && extra=(--jit True)
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup "$PY" "$WS/scripts/run_atari_single_job.py" \
    --gpu "$gpu" --version "$ver" --game "$game" \
    --minutes "$MINUTES" --steps "$STEPS" --mem-fraction 0.25 "${extra[@]}" \
    >> "$LOGS/${game}_${ver}.log" 2>&1 &
  echo "launched ${game}_${ver} on GPU$gpu pid=$! (resume → ${STEPS} env steps, ${MINUTES}min)"
  echo "$!" > "$LOGS/${game}_${ver}.pid"
}

for game in pong breakout boxing; do
  launch 0 v2 "$game"
done
for game in pong breakout boxing; do
  launch 1 v3 "$game"
done

nohup "$PY" "$WS/scripts/v2_ckpt_sidecar.py" --interval 600 --hours 8.5 \
  >> "$LOGS/v2_sidecar.log" 2>&1 &
echo "sidecar pid=$!"
echo "$!" > "$LOGS/v2_sidecar.pid"

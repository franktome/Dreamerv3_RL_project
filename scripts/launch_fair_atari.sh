#!/usr/bin/env bash
# Fair Atari retrain: 6 jobs, GPU0 = V2 x3 (TF), GPU1 = V3 x3 (JAX).
# Budget: 100k env steps (= 400k frames, Atari-100k) with 3.5h wall-clock cap.
set -u
WS=/mnt/server12_hard0/kiseol/Dreamerv3
PY=${PY:-/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3/bin/python}
LOGS=$WS/logs/atari_fair
mkdir -p "$LOGS"

STEPS=100000   # env interactions (post-repeat); logged step counter reads 400k
MINUTES=210    # 3.5h timeout per job

# TF 2.15 needs cuDNN 8 (JAX uses cuDNN 9); side-installed copy keeps env intact.
SP=$(dirname "$PY")/../lib/python3.11/site-packages
export LD_LIBRARY_PATH="$WS/.cache/cudnn8/nvidia/cudnn/lib:$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"

launch() {
  local gpu=$1 ver=$2 game=$3
  local extra=()
  # jit only changes execution speed (graph compilation), not the model.
  [ "$ver" = v2 ] && extra=(--jit True)
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup "$PY" "$WS/scripts/run_atari_single_job.py" \
    --gpu "$gpu" --version "$ver" --game "$game" \
    --minutes "$MINUTES" --steps "$STEPS" --mem-fraction 0.25 "${extra[@]}" \
    > "$LOGS/${game}_${ver}.log" 2>&1 &
  echo "launched ${game}_${ver} on GPU$gpu pid=$!"
  echo "$!" > "$LOGS/${game}_${ver}.pid"
}

for game in pong breakout boxing; do
  launch 0 v2 "$game"
done
for game in pong breakout boxing; do
  launch 1 v3 "$game"
done

# V2 snapshot sidecar (10-min interval, labels by env steps)
nohup "$PY" "$WS/scripts/v2_ckpt_sidecar.py" --interval 600 --hours 4.5 \
  > "$LOGS/v2_sidecar.log" 2>&1 &
echo "sidecar pid=$!"
echo "$!" > "$LOGS/v2_sidecar.pid"

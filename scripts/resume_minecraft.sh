#!/usr/bin/env bash
# Resume Minecraft training after Atari jobs finish.
set -uo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
FLAG="${PROJECT}/logs/minecraft_paused.flag"
rm -f "$FLAG"

export HOME="${HOME:-/mnt/server12_hard0/kiseol}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$PROJECT"
mkdir -p logs
nohup bash run_minecraft_full_gpu1.sh >> logs/mc_full_gpu1.log 2>&1 &
echo "Minecraft resumed (PID $!)"

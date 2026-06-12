#!/usr/bin/env bash
# Pause Minecraft training to free RAM for Atari jobs.
set -uo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
FLAG="${PROJECT}/logs/minecraft_paused.flag"
mkdir -p "${PROJECT}/logs"
date '+%Y-%m-%d %H:%M:%S %Z' > "$FLAG"
echo "paused" >> "$FLAG"

pkill -f "dreamerv3/main.py.*minecraft_diamond_full" 2>/dev/null || true
pkill -f "scripts/run_minecraft_full.py" 2>/dev/null || true
pkill -f "run_minecraft_full_gpu1.sh" 2>/dev/null || true
sleep 2
# MineRL Java children
pkill -f "MalmoMod-0.37.0-fat.jar" 2>/dev/null || true

echo "Minecraft paused (flag: $FLAG)"
free -h | head -2

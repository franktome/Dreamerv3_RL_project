#!/usr/bin/env python3
"""Periodically snapshot DreamerV2 variables.pkl, labeled by env steps.

V2 keeps a single variables.pkl, so without snapshots we cannot pick an
aligned checkpoint after training. This sidecar copies the file every
`--interval` seconds into <logdir>/snapshots/variables_<env_steps>.pkl.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

LOGROOT = Path(__file__).resolve().parents[1] / 'vendor' / 'dreamerv3' / 'logdir'


def latest_env_steps(logdir: Path) -> int:
    metrics = logdir / 'metrics.jsonl'
    steps = 0
    if metrics.exists():
        for line in metrics.read_text().strip().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'train_total_steps' in row:
                steps = int(row['train_total_steps'])
    return steps


def snapshot(game: str) -> str:
    logdir = LOGROOT / f'atari_{game}_v2'
    src = logdir / 'variables.pkl'
    if not src.exists():
        return f'{game}: no ckpt yet'
    steps = latest_env_steps(logdir)
    snapdir = logdir / 'snapshots'
    snapdir.mkdir(exist_ok=True)
    dst = snapdir / f'variables_{steps:07d}.pkl'
    if dst.exists():
        return f'{game}: snapshot {steps} exists'
    tmp = snapdir / '.tmp.pkl'
    shutil.copy2(src, tmp)
    tmp.rename(dst)
    return f'{game}: saved snapshot at {steps} env steps'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+', default=['pong', 'breakout', 'boxing'])
    parser.add_argument('--interval', type=int, default=600)
    parser.add_argument('--hours', type=float, default=4.5)
    args = parser.parse_args()

    deadline = time.time() + args.hours * 3600
    while time.time() < deadline:
        for game in args.games:
            try:
                print(time.strftime('%H:%M:%S'), snapshot(game), flush=True)
            except Exception as e:  # keep sidecar alive on transient copy errors
                print(time.strftime('%H:%M:%S'), f'{game}: error {e}', flush=True)
        time.sleep(args.interval)
    print('sidecar done', flush=True)


if __name__ == '__main__':
    main()

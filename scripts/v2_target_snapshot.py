#!/usr/bin/env python3
"""Snapshot V2 variables.pkl as soon as env steps cross a per-game target.

Used to grab a V2 checkpoint matching the final V3 checkpoint step, so the
side-by-side GIF uses checkpoints at (nearly) the same training progress.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

LOGROOT = Path(__file__).resolve().parents[1] / 'vendor' / 'dreamerv3' / 'logdir'

TARGETS = {'pong': 90330, 'breakout': 90870, 'boxing': 91240}


def env_steps(logdir: Path) -> int:
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


def main() -> None:
    pending = dict(TARGETS)
    deadline = time.time() + 4 * 3600
    while pending and time.time() < deadline:
        for game, target in list(pending.items()):
            logdir = LOGROOT / f'atari_{game}_v2'
            steps = env_steps(logdir)
            src = logdir / 'variables.pkl'
            if steps >= target and src.exists():
                snapdir = logdir / 'snapshots'
                snapdir.mkdir(exist_ok=True)
                dst = snapdir / f'variables_{steps:07d}.pkl'
                if not dst.exists():
                    tmp = snapdir / '.tmp_target.pkl'
                    shutil.copy2(src, tmp)
                    tmp.rename(dst)
                print(time.strftime('%H:%M:%S'), f'{game}: target snapshot at {steps} (target {target})', flush=True)
                del pending[game]
        time.sleep(20)
    print('target watcher done; remaining:', pending, flush=True)


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Minecraft full training with checkpoint resume (fresh process each launch)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from seollab.env_setup import clone_dreamerv3, ensure_xvfb
from seollab.paths import default_paths
from seollab import minecraft


def _latest_step(logdir: Path) -> int:
    metrics = logdir / 'metrics.jsonl'
    if not metrics.exists() or not metrics.read_text().strip():
        return 0
    return int(json.loads(metrics.read_text().strip().splitlines()[-1]).get('step', 0))


def _latest_ckpt(logdir: Path) -> str:
    latest = logdir / 'ckpt' / 'latest'
    if latest.exists():
        return latest.read_text().strip()
    return '(none)'


def main() -> int:
    import os

    pause_flag = WORKSPACE / 'logs' / 'minecraft_paused.flag'
    if pause_flag.exists():
        print(f'Minecraft paused ({pause_flag}); exit.')
        return 0

    cfg = default_paths(WORKSPACE, gpu=os.environ.get('MC_GPU', '1'))
    logdir = cfg.minecraft_full_logdir
    steps = int(os.environ.get('MC_STEPS', '100000000'))

    ensure_xvfb(cfg.display)
    clone_dreamerv3(cfg)

    step = _latest_step(logdir)
    ckpt = _latest_ckpt(logdir)
    print(f'Resume logdir: {logdir}')
    print(f'  metrics step: {step:,}')
    print(f'  latest ckpt:  {ckpt}')

    if step >= steps:
        print(f'Target reached ({step:,} >= {steps:,})')
        return 0

    # skip_if_ckpt=False: always train; DreamerV3 load_or_save() resumes in-logdir.
    minecraft.train(
        cfg,
        mode='full',
        skip_if_ckpt=False,
        logdir=logdir,
        steps=steps,
        envs=int(os.environ.get('MC_ENVS', '4')),
        batch_size=int(os.environ.get('MC_BATCH', '8')),
    )
    return 0


if __name__ == '__main__':
    while True:
        try:
            raise SystemExit(main())
        except SystemExit:
            raise
        except Exception as exc:
            print(f'Error: {exc} — retry in 90s', flush=True)
            time.sleep(90)

#!/usr/bin/env python3
"""Run Minecraft multi-rollout inference on GPU1, then resume training in background."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONDA_ENV = Path(os.environ.get(
    'DREAMERV3_CONDA',
    '/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3',
))

from dvbench.paths import default_paths
from dvbench.hf_assets import resolve_minecraft_logdir
from dvbench import inference_demo
from dvbench import minecraft


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default='1')
    p.add_argument('--n-rollouts', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=3600)
    p.add_argument('--mem-fraction', type=float, default=0.42)
    p.add_argument('--resume-train', action='store_true', help='Start training after rollouts')
    p.add_argument('--train-envs', type=int, default=6)
    args = p.parse_args()

    cfg = default_paths(ROOT, gpu=args.gpu)
    if CONDA_ENV.exists():
        cfg.conda_env = CONDA_ENV
    cfg.apply_env(mem_fraction=args.mem_fraction)
    logdir = resolve_minecraft_logdir(cfg)
    print('logdir:', logdir)

    result = inference_demo.run_minecraft_multi_rollouts(
        cfg,
        logdir=logdir,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
        top_k=3,
    )
    print(json.dumps({
        'ok': result.get('ok'),
        'best': result.get('best', {}).get('max_milestone') if result.get('best') else None,
        'milestones': result.get('best', {}).get('milestones_reached') if result.get('best') else [],
        'index': result.get('index_path'),
    }, indent=2))

    if args.resume_train:
        proc = minecraft.resume_train(
            cfg, logdir=logdir, envs=args.train_envs, mem_fraction=args.mem_fraction,
        )
        print('Training resumed pid=', proc.pid)
        print('Log:', ROOT / 'logs' / 'minecraft_diamond_full' / 'train_gpu1.log')


if __name__ == '__main__':
    main()

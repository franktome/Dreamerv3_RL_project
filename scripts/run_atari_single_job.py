#!/usr/bin/env python3
"""Run one Atari V2/V3 timed training job (fresh code load per invocation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--version', choices=('v2', 'v3'), required=True)
    parser.add_argument('--game', required=True)
    parser.add_argument('--minutes', type=int, default=300)
    parser.add_argument('--steps', type=int, default=None,
                        help='env-step cap (post-repeat interactions)')
    parser.add_argument('--mem-fraction', type=float, default=0.42)
    parser.add_argument('--jit', choices=('True', 'False'), default=None,
                        help='override V2 jit config (speed only, same model)')
    args = parser.parse_args()

    from seollab import atari, atari_v2
    from seollab.paths import default_paths

    cfg = default_paths(WORKSPACE, gpu=args.gpu)
    cfg.apply_env(mem_fraction=args.mem_fraction)

    if args.version == 'v2':
        atari_v2.train_v2_timed(
            cfg, args.game, minutes=args.minutes, steps=args.steps,
            mem_fraction=args.mem_fraction, resume=True,
            jit=None if args.jit is None else args.jit == 'True',
        )
    else:
        atari.train_v3_timed(
            cfg, args.game, minutes=args.minutes, steps=args.steps,
            mem_fraction=args.mem_fraction, resume=True,
        )


if __name__ == '__main__':
    main()

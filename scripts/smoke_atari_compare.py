#!/usr/bin/env python3
"""Smoke test: 3 Atari games × V2/V3 mini train + infer (OOM-safe)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seollab.paths import default_paths
from seollab import atari_compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default='1')
    p.add_argument('--mem-fraction', type=float, default=0.25)
    args = p.parse_args()
    cfg = default_paths(ROOT, gpu=args.gpu)
    cfg.apply_env(mem_fraction=args.mem_fraction)
    print('Smoke test on GPU', args.gpu, 'mem_fraction', args.mem_fraction)
    result = atari_compare.run_smoke(cfg, mem_fraction=args.mem_fraction)
    print(result)
    if not result['ok']:
        print('SMOKE FAILED — try --mem-fraction 0.20')
        sys.exit(1)
    print('SMOKE OK')


if __name__ == '__main__':
    main()

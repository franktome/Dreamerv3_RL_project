#!/usr/bin/env python3
"""Regenerate V2|V3 compare + anim GIFs for all compare games."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seollab.paths import default_paths
from seollab import gif_compare, atari_anim, atari_align, atari_compare, viz_atari_compare


def main() -> int:
    cfg = default_paths(ROOT, gpu='1')
    cfg.apply_env(mem_fraction=0.25)
    games = atari_compare.GAMES
    print('Regenerating step-aligned GIFs for:', ', '.join(games))
    report = atari_align.write_alignment_report(cfg, games)
    print('Alignment report:', report)
    for row in atari_align.alignment_table(cfg, games):
        print(
            f"  {row['game']}: aligned={row['aligned_env_steps']:,} "
            f"(v2={row['v2_env_steps']:,}, v3={row['v3_env_steps']:,})"
        )

    for game in games:
        print(f'\n=== {game}: compare ===')
        r = gif_compare.infer_both(cfg, game, max_steps=1500)
        print('compare ok:', r.get('ok'), 'delta:', r.get('score_delta'))

    for game in games:
        for preset in ('fast', 'sluggish'):
            print(f'\n=== {game}: anim {preset} ===')
            a = atari_anim.run_anim_compare(cfg, game, preset=preset, max_steps=1200)
            print('anim ok:', a.get('ok'), 'grid:', a.get('grid_gif'))

    metrics = atari_compare.collect_metrics(cfg)
    viz_atari_compare.generate_all(cfg, metrics, games)
    viz_atari_compare.write_atari_compare_report(cfg, metrics, {'refreshed': True})
    print('\nDone. Artifacts under highlights/inference/ and report/atari_compare/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

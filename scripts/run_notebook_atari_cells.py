#!/usr/bin/env python3
"""Run dreamerv3_team11.ipynb §2.1b + §2.2 Atari cells (22, 24) headless."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))


def cell_22_24(cfg, games):
    import pandas as pd
    from IPython.display import Image, Markdown, display
    from dvbench import atari_align, atari_compare, gif_compare

    align_df = pd.DataFrame(atari_align.alignment_table(cfg, games))
    print('=== Cell 22: Alignment ===')
    show = align_df[
        ['game', 'v2_env_steps', 'v2_episodes', 'v3_env_steps', 'v3_episodes',
         'aligned_env_steps', 'aligned_episodes', 'fair_gif']
    ].copy()
    print(show.to_string(index=False))

    metrics = atari_compare.collect_metrics(cfg)
    print('\n=== Cell 22: Metrics at aligned steps ===')
    print(metrics['summary'].to_string(index=False))

    lc = cfg.report_dir / 'atari_compare' / 'learning_curves_3games.png'
    print('\nlearning_curves:', lc, 'exists' if lc.exists() else 'missing')

    print('\n=== Cell 22/24: Side-by-side GIFs ===')
    for game in games:
        row = align_df[align_df['game'] == game].iloc[0]
        al = {r['game']: r for r in atari_align.alignment_table(cfg, games)}[game]
        cmp = gif_compare.infer_both(cfg, game, max_steps=1500, align=True)
        v2s = cmp.get('v2', {}).get('score') if cmp.get('ok') else None
        v3s = cmp.get('v3', {}).get('score') if cmp.get('ok') else None
        fair = '✅ fair GIF' if al['fair_gif'] and cmp.get('ok') else '⚠️ check alignment'
        score_txt = f' | rollout V2={v2s} V3={v3s}' if v2s is not None and v3s is not None else ''
        print(
            f"{game}: aligned {int(row['aligned_env_steps']):,} env steps "
            f"(V2 ep {int(row['v2_episodes'])}, V3 ep {int(row['v3_episodes'])}) "
            f"{fair}{score_txt}"
        )
        if cmp.get('ok'):
            print('  compare_gif:', cmp.get('compare_gif'))


def main() -> None:
    from seollab import atari_compare
    from seollab.paths import default_paths

    cfg = default_paths(WORKSPACE, gpu='1')
    cfg.apply_env(mem_fraction=0.25)
    games = atari_compare.GAMES
    cell_22_24(cfg, games)


if __name__ == '__main__':
    main()

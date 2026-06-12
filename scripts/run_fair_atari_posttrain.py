#!/usr/bin/env python3
"""Post-training pipeline for fair Atari retrain (plan step: align-results)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))


def main() -> None:
    from seollab import atari_align, atari_compare, gif_compare, viz_atari_compare
    from seollab.paths import default_paths

    cfg = default_paths(WORKSPACE, gpu='1')
    cfg.apply_env(mem_fraction=0.25)
    games = atari_compare.GAMES

    align_rows = atari_align.alignment_table(cfg, games)
    print('=== Alignment ===')
    for r in align_rows:
        print(json.dumps(r, ensure_ascii=False))

    align_path = atari_align.write_alignment_report(cfg, games)
    print('Wrote', align_path)

    refresh = {'compare': {}, 'anim': {}}
    for game in games:
        print(f'GIF compare: {game}')
        refresh['compare'][game] = gif_compare.infer_both(cfg, game, max_steps=1500)
        print(' ', refresh['compare'][game].get('ok'), refresh['compare'][game].get('compare_gif'))

    metrics = atari_compare.collect_metrics(cfg)
    plots = viz_atari_compare.generate_all(cfg, metrics, games)
    report = viz_atari_compare.write_atari_compare_report(cfg, metrics, refresh)
    print('Report:', report)
    print('Plots:', plots)

    out = cfg.report_dir / 'atari_compare' / 'fair_retrain_result.json'
    payload = {
        'alignment': align_rows,
        'compare': {g: refresh['compare'][g] for g in games},
        'report': str(report),
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print('Saved', out)


if __name__ == '__main__':
    main()

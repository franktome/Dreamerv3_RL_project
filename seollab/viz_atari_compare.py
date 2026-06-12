"""Visualizations for 3-game DreamerV2 vs V3 local comparison."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import PathConfig
from . import atari, atari_v2, atari_align, gif_compare

COLORS = {'DreamerV2': '#ff7f0e', 'DreamerV3': '#2ca02c'}


def plot_learning_curves(cfg: PathConfig, games: tuple[str, ...]) -> str:
    fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 4))
    if len(games) == 1:
        axes = [axes]
    align_map = {r['game']: r['aligned_env_steps'] for r in atari_align.alignment_table(cfg, games)}
    for ax, game in zip(axes, games):
        aligned = align_map.get(game, 0)
        for ver, loader, label in (
            ('v2', atari_v2.load_v2_scores, 'DreamerV2'),
            ('v3', atari.load_v3_scores, 'DreamerV3'),
        ):
            rows = loader(cfg.atari_logdir(game, ver))
            ys, xs = [], []
            for i, s in enumerate(rows):
                step = int(s.get('step', 0))
                env_step = (
                    int(s['train_total_steps']) if 'train_total_steps' in s
                    else step // atari_align.ATARI_REPEAT
                )
                if aligned and env_step > aligned:
                    continue
                if 'train_return' in s:
                    ys.append(float(s['train_return']))
                    xs.append(env_step if ver == 'v2' else i)
                elif 'episode/score' in s:
                    ys.append(float(s['episode/score']))
                    xs.append(env_step)
            if ys:
                ax.plot(xs, ys, label=label, color=COLORS[label], alpha=0.8)
        ax.axvline(aligned, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'{game} (aligned ≤ {aligned:,})')
        ax.set_xlabel('Episode / step')
        ax.set_ylabel('Return')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Local training — episode returns', y=1.02)
    plt.tight_layout()
    out = cfg.report_dir / 'atari_compare' / 'learning_curves_3games.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(out)


def plot_gif_strips(cfg: PathConfig, games: tuple[str, ...]) -> dict[str, str]:
    paths = {}
    for game in games:
        gif = cfg.highlights_dir / 'inference' / f'atari_{game}_v2v3_compare.gif'
        if not gif.exists():
            continue
        frames = gif_compare.load_gif_frames(gif)
        if not frames:
            continue
        picks = [frames[0], frames[len(frames) // 2], frames[-1]]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, fr, title in zip(axes, picks, ['Start', 'Mid', 'End']):
            ax.imshow(fr)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        fig.suptitle(f'{game} — V2 | V3 compare', fontsize=10)
        plt.tight_layout()
        out = cfg.report_dir / 'atari_compare' / f'{game}_v2v3_strip.png'
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        paths[game] = str(out)
    return paths


def plot_metrics_summary(cfg: PathConfig, metrics: dict) -> str:
    df = metrics.get('summary')
    if df is None or df.empty:
        return ''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    games = df['game'].unique()
    x = np.arange(len(games))
    w = 0.35
    for i, model in enumerate(['DreamerV2', 'DreamerV3']):
        sub = df[df['model'] == model].set_index('game').reindex(games)
        axes[0].bar(x + (i - 0.5) * w, sub['mean_return'].fillna(0), w, label=model, color=COLORS[model])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(games)
    axes[0].set_ylabel('Mean training return')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    hns = metrics.get('hns')
    if hns is not None and not hns.empty:
        for i, model in enumerate(['DreamerV2', 'DreamerV3']):
            sub = hns[hns['model'] == model].set_index('game').reindex(games)
            axes[1].bar(x + (i - 0.5) * w, sub['hns'].fillna(0), w, label=model, color=COLORS[model])
        axes[1].set_ylabel('HNS (local max return)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(games)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = cfg.report_dir / 'atari_compare' / 'metrics_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(out)


def plot_anim_grid(cfg: PathConfig, games: tuple[str, ...]) -> dict[str, str]:
    paths = {}
    for game in games:
        gifs = list((cfg.highlights_dir / 'inference' / 'anim').glob(f'atari_{game}_*_v2v3.gif'))
        if not gifs:
            continue
        fig, axes = plt.subplots(1, len(gifs), figsize=(4 * len(gifs), 3))
        if len(gifs) == 1:
            axes = [axes]
        for ax, gif in zip(axes, sorted(gifs)):
            frames = gif_compare.load_gif_frames(gif)
            if frames:
                ax.imshow(frames[len(frames) // 2])
            ax.set_title(gif.stem.replace(f'atari_{game}_', ''), fontsize=8)
            ax.axis('off')
        fig.suptitle(f'{game} — anim perturbations', fontsize=10)
        plt.tight_layout()
        out = cfg.report_dir / 'atari_compare' / f'{game}_anim_grid.png'
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        paths[game] = str(out)
    return paths


def generate_all(cfg: PathConfig, metrics: dict, games: tuple[str, ...]) -> dict:
    return {
        'learning_curves': plot_learning_curves(cfg, games),
        'gif_strips': plot_gif_strips(cfg, games),
        'metrics_summary': plot_metrics_summary(cfg, metrics),
        'anim_grids': plot_anim_grid(cfg, games),
    }


def write_atari_compare_report(cfg: PathConfig, metrics: dict, pipeline: dict) -> str:
    df = metrics.get('summary')
    align = metrics.get('alignment')
    lines = [
        '# Atari 3-Game Compare — DreamerV2 vs DreamerV3',
        f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}',
        '',
        '## Setup',
        '- Games: pong, breakout, boxing',
        '- Equal wall-clock training per model (90 min default)',
        '- GPU1 shared with ongoing Minecraft training',
        '',
        '## Training summary',
        '',
    ]
    if df is not None and not df.empty:
        lines.append('| Game | Model | Mean return | Max return | Episodes |')
        lines.append('|------|-------|------------:|-----------:|---------:|')
        for _, r in df.iterrows():
            lines.append(
                f"| {r['game']} | {r['model']} | {r['mean_return']:.2f} | {r['max_return']:.2f} | {int(r['episodes'])} |"
            )
    lines += [
        '',
        '## Alignment basis (fair cutoff)',
        '',
        '- Metric curves and summary above are truncated to each game\'s `aligned_env_steps = min(v2, v3)`.',
        '- `fair_gif = False` means current V3 main checkpoint is beyond aligned cutoff (GIF may be less fair).',
        '',
    ]
    if align is not None and not align.empty:
        lines.append('| Game | V2 env steps | V2 episodes | V3 env steps | V3 episodes | Aligned env steps | Aligned episodes | fair_gif |')
        lines.append('|------|-------------:|------------:|-------------:|------------:|------------------:|-----------------:|:--------:|')
        for _, r in align.iterrows():
            lines.append(
                f"| {r['game']} | {int(r['v2_env_steps'])} | {int(r['v2_episodes'])} | "
                f"{int(r['v3_env_steps'])} | {int(r['v3_episodes'])} | "
                f"{int(r['aligned_env_steps'])} | {int(r['aligned_episodes'])} | "
                f"{'✅' if bool(r['fair_gif']) else '⚠️'} |"
            )
    lines += [
        '',
        '## Inference GIFs',
        '',
        'Side-by-side rollouts: `highlights/inference/atari_{game}_v2v3_compare.gif`',
        '',
        '## Anim perturbations',
        '',
        '| Preset | repeat | sticky | Effect |',
        '|--------|--------|--------|--------|',
        '| baseline | 4 | 0.25 | Training-matched dynamics |',
        '| fast | 2 | 0.25 | Faster game speed (frame skip) |',
        '| sluggish | 4 | 0.50 | Higher action stickiness / control lag |',
        '',
        '## Analysis',
        '',
    ]
    if df is not None and not df.empty:
        for game in df['game'].unique():
            sub = df[df['game'] == game]
            v2 = sub[sub['model'] == 'DreamerV2']
            v3 = sub[sub['model'] == 'DreamerV3']
            if len(v2) and len(v3):
                d = float(v3['mean_return'].iloc[0]) - float(v2['mean_return'].iloc[0])
                lines.append(f'- **{game}**: V3 mean return − V2 = {d:+.2f}')
    lines += [
        '',
        '## Plots',
        '',
        '![Learning curves](learning_curves_3games.png)',
        '![Metrics](metrics_summary.png)',
    ]
    path = cfg.report_dir / 'atari_compare' / 'ATARI_COMPARE.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    return str(path)

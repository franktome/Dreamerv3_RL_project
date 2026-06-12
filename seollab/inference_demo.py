"""Inference demo: env GIFs, local score distributions, training health."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import PathConfig
from . import minecraft, atari, viz

MILESTONE_LABELS = minecraft.MILESTONES


def analyze_training_health(logdir: Path) -> dict:
    """Summarize ongoing Minecraft full training without stopping it."""
    logdir = Path(logdir)
    out = {'logdir': str(logdir), 'issues': [], 'ok': True}

    metrics = logdir / 'metrics.jsonl'
    scores = logdir / 'scores.jsonl'
    if not metrics.exists():
        out['ok'] = False
        out['issues'].append('metrics.jsonl missing')
        return out

    lines = metrics.read_text().strip().splitlines()
    last = json.loads(lines[-1])
    out['step'] = last.get('step', 0)
    out['fps_policy'] = last.get('fps/policy', 0)
    out['fps_train'] = last.get('fps/train', 0)

    if out['fps_policy'] and out['fps_policy'] < 1.0:
        out['issues'].append('Very low policy fps — check Xvfb / Malmo DISPLAY')
    if len(lines) >= 2:
        prev = json.loads(lines[-2])
        if last.get('step', 0) <= prev.get('step', 0):
            out['issues'].append('Step not advancing — possible stall or checkpoint rollback')

    if scores.exists():
        ep_scores = []
        for line in scores.read_text().strip().splitlines():
            d = json.loads(line)
            ep_scores.append(float(d.get('episode/score', d.get('score', 0))))
        out['episodes'] = len(ep_scores)
        out['max_milestone'] = int(round(max(ep_scores))) if ep_scores else 0
        out['max_milestone_name'] = (
            MILESTONE_LABELS[min(out['max_milestone'], len(MILESTONE_LABELS) - 1)]
            if ep_scores else 'none'
        )
        out['mean_episode_score'] = float(np.mean(ep_scores))
        out['max_episode_score'] = float(max(ep_scores))

    ckpt = logdir / 'ckpt'
    if ckpt.exists():
        latest = ckpt / 'latest'
        out['checkpoint'] = latest.read_text().strip() if latest.exists() else 'unknown'

    return out


def plot_local_minecraft_scores(
    cfg: PathConfig,
    logdir: Path | None = None,
    show: bool = True,
) -> dict:
    """Episode score distribution from local training scores.jsonl."""
    logdir = Path(logdir or cfg.minecraft_full_logdir)
    scores_path = logdir / 'scores.jsonl'
    if not scores_path.exists():
        return {'ok': False, 'error': f'No {scores_path}'}

    rows = []
    for line in scores_path.read_text().strip().splitlines():
        d = json.loads(line)
        rows.append({
            'step': d.get('step', 0),
            'score': float(d.get('episode/score', d.get('score', 0))),
        })
    df = pd.DataFrame(rows)
    df['milestone'] = df['score'].round().astype(int).clip(0, 12)

    out_dir = cfg.report_dir / 'inference'
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(df['milestone'], bins=np.arange(-0.5, 13.5, 1), color='#2ca02c', alpha=0.85, edgecolor='white')
    axes[0].set(
        xlabel='Milestone index',
        ylabel='Episode count',
        title='Local DreamerV3 — milestone distribution',
    )
    axes[0].set_xticks(range(0, 13, 2))

    axes[1].scatter(df['step'] / 1e6, df['score'], alpha=0.6, s=28, c='#1f77b4', label='Episodes')
    rolling = df['score'].rolling(5, min_periods=1).mean()
    axes[1].plot(df['step'] / 1e6, rolling, color='black', lw=1.5, alpha=0.6, label='Rolling mean (5 ep)')
    axes[1].set(xlabel='Training steps (M)', ylabel='Episode score', title='Episode scores over training')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    health = analyze_training_health(logdir)
    fig.suptitle(
        f"Minecraft full train @ step {health.get('step', 0):,} | "
        f"max milestone: {health.get('max_milestone_name', '?')}",
        y=1.02,
        fontsize=11,
    )
    plt.tight_layout()
    png = out_dir / 'minecraft_local_scores.png'
    fig.savefig(png, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    summary = {
        'ok': True,
        'png': str(png),
        'episodes': len(df),
        'max_milestone': int(df['milestone'].max()),
        'mean_score': float(df['score'].mean()),
    }
    return summary


def plot_atari_comparison(cfg: PathConfig, show: bool = True) -> pd.DataFrame:
    """Atari V2 vs V3 curves (report/atari_v2_v3.png) + HNS distribution."""
    compare = viz.plot_atari_v2_v3(cfg, show=show)
    plot_atari_hns_distribution(cfg, show=show, compare=compare)
    return compare


def plot_atari_hns_distribution(
    cfg: PathConfig, show: bool = True, compare: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-game HNS distribution for DreamerV2 vs DreamerV3 (official scores)."""
    if compare is None:
        compare = viz.plot_atari_v2_v3(cfg, show=False)

    out_dir = cfg.report_dir / 'inference'
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(compare['DreamerV2'].dropna(), bins=20, alpha=0.6, label='DreamerV2', color='#ff7f0e')
    axes[0].hist(compare['DreamerV3'].dropna(), bins=20, alpha=0.6, label='DreamerV3', color='#2ca02c')
    axes[0].axvline(compare['DreamerV2'].median(), color='#ff7f0e', ls='--', lw=1.5)
    axes[0].axvline(compare['DreamerV3'].median(), color='#2ca02c', ls='--', lw=1.5)
    axes[0].set(xlabel='HNS @ 50M steps', ylabel='Games', title='Atari 57 — HNS distribution')
    axes[0].legend()

    delta = compare['delta'].dropna()
    axes[1].hist(delta, bins=20, color='#9467bd', alpha=0.85, edgecolor='white')
    axes[1].axvline(0, color='black', lw=1)
    axes[1].set(
        xlabel='HNS delta (V3 − V2)',
        ylabel='Games',
        title=f'V3 wins on {(delta > 0).mean() * 100:.0f}% of games',
    )
    plt.tight_layout()
    png = out_dir / 'atari_hns_distribution.png'
    fig.savefig(png, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    compare.to_csv(out_dir / 'atari_hns_compare.csv')
    return compare


def run_minecraft_env_gif(
    cfg: PathConfig,
    logdir: Path | None = None,
    max_steps: int = 800,
    train_mode: str = 'full',
) -> dict:
    """Policy rollout GIF from latest checkpoint (safe while training)."""
    logdir = Path(logdir or cfg.minecraft_full_logdir)
    out_dir = cfg.highlights_dir / 'inference'
    out_dir.mkdir(parents=True, exist_ok=True)
    gif = out_dir / 'minecraft_diamond_rollout.gif'
    return minecraft.infer(
        cfg, logdir=logdir, gif_path=gif,
        max_steps=max_steps, train_mode=train_mode,
    )


def run_atari_v3_env_gif(cfg: PathConfig, game: str = 'pong', max_steps: int = 1500) -> dict:
    """DreamerV3 Atari env GIF from local debug checkpoint if present."""
    logdir = cfg.dreamerv3_root / 'logdir' / f'atari_{game}'
    ckpt = logdir / 'ckpt'
    if not ckpt.exists() or not any(ckpt.iterdir()):
        return {
            'ok': False,
            'error': f'No local atari_{game} checkpoint — train Step 5 first or use official scores',
            'game': game,
        }
    out_dir = cfg.highlights_dir / 'inference'
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    dest = out_dir / f'atari_{game}_rollout.gif'
    result = atari.infer_v3_debug(cfg, game=game, max_steps=max_steps)
    src = Path(result.get('gif', ''))
    if result.get('ok') and src.exists():
        shutil.copy2(src, dest)
        result['gif'] = str(dest)
    return result


def run_full_demo(
    cfg: PathConfig,
    minecraft_steps: int = 800,
    atari_game: str = 'pong',
    show: bool = False,
) -> dict:
    """Run all inference artifacts (GPU shared with ongoing training)."""
    cfg.apply_env()
    results = {
        'health': analyze_training_health(cfg.minecraft_full_logdir),
        'minecraft_scores': plot_local_minecraft_scores(cfg, show=show),
        'minecraft_gif': run_minecraft_env_gif(cfg, max_steps=minecraft_steps),
        'minecraft_baselines': viz.plot_minecraft_baselines(cfg, show=show),
        'atari_compare': plot_atari_hns_distribution(cfg, show=show),
        'atari_gif': run_atari_v3_env_gif(cfg, game=atari_game),
    }
    return results

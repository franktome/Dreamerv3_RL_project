"""Extended visualizations: 10-game Atari panels, Minecraft overlays, executive summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import PathConfig
from . import viz

COLORS = {'DreamerV2': '#ff7f0e', 'DreamerV3': '#2ca02c', 'PPO': '#e377c2', 'IMPALA': '#1f77b4'}


def load_atari_compare(cfg: PathConfig) -> pd.DataFrame:
    return viz.plot_atari_v2_v3(cfg, show=False)


def select_10_games(compare: pd.DataFrame) -> list[str]:
    """8 largest V3 gains + 2 largest V2 gains for balanced comparison."""
    c = compare.dropna(subset=['DreamerV2', 'DreamerV3'])
    top_v3 = c.nlargest(8, 'delta').index.tolist()
    rest = c.drop(index=top_v3, errors='ignore')
    top_v2 = rest.nsmallest(2, 'delta').index.tolist()
    games = top_v3 + top_v2
    return [g.replace('atari_', '') for g in games[:10]]


def _load_minecraft_runs(cfg: PathConfig) -> dict:
    urls = {
        'DreamerV3': 'https://raw.githubusercontent.com/danijar/dreamerv3/main/scores/minecraft_diamond-dreamerv3.json.gz',
        'PPO': 'https://raw.githubusercontent.com/danijar/dreamerv3/main/scores/minecraft_diamond-ppo_fixhp.json.gz',
        'IMPALA': 'https://raw.githubusercontent.com/danijar/dreamerv3/main/scores/minecraft_diamond-impala.json.gz',
    }
    return {n: viz._load_cached(cfg.scores_cache, f'minecraft_{n.lower()}.json', u, gz=True) for n, u in urls.items()}


def plot_minecraft_baselines_enhanced(cfg: PathConfig, show: bool = False) -> dict:
    """Task-success curves + mean max milestone (official scores)."""
    runs = _load_minecraft_runs(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    summary = {}

    def task_success_curve(rs, n=80):
        max_x = max(max(r['xs']) for r in rs)
        xs = np.unique(np.geomspace(max(1, min(min(r['xs']) for r in rs)), max_x, n).astype(int))
        rates = []
        for x in xs:
            vals = []
            for run in rs:
                valid = [i for i, xv in enumerate(run['xs']) if xv <= x]
                if valid:
                    best = max(run['ys'][i] for i in valid)
                    vals.append(1.0 if best >= 12 else 0.0)
            rates.append(np.mean(vals) if vals else np.nan)
        return xs, np.array(rates)

    def mean_milestone_curve(rs, n=80):
        max_x = max(max(r['xs']) for r in rs)
        xs = np.unique(np.geomspace(max(1, min(min(r['xs']) for r in rs)), max_x, n).astype(int))
        means = []
        for x in xs:
            ms = []
            for run in rs:
                valid = [i for i, xv in enumerate(run['xs']) if xv <= x]
                if valid:
                    ms.append(min(12, max(run['ys'][: max(valid) + 1])))
            means.append(np.mean(ms) if ms else np.nan)
        return xs, np.array(means)

    for name, rs in runs.items():
        xs, succ = task_success_curve(rs)
        m = ~np.isnan(succ)
        axes[0].plot(xs[m] / 1e6, succ[m] * 100, label=name, color=COLORS.get(name, 'gray'), lw=2)
        xs2, mm = mean_milestone_curve(rs)
        m2 = ~np.isnan(mm)
        axes[1].plot(xs2[m2] / 1e6, mm[m2], label=name, color=COLORS.get(name, 'gray'), lw=2)
        summary[name] = {
            'final_task_success_pct': float(np.mean([1.0 if max(r['ys']) >= 12 else 0.0 for r in rs]) * 100),
            'mean_max_milestone': float(np.mean([min(12, max(r['ys'])) for r in rs])),
        }

    axes[0].set(xlabel='Steps (M)', ylabel='Task success (%)', title='Minecraft — task completion rate')
    axes[1].set(xlabel='Steps (M)', ylabel='Mean max milestone index', title='Minecraft — progress depth')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = cfg.report_dir / 'minecraft_baselines_enhanced.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return summary


def plot_minecraft_local_overlay(cfg: PathConfig, logdir: Path, show: bool = False) -> str:
    """Local episode scores over training steps."""
    scores_path = Path(logdir) / 'scores.jsonl'
    if not scores_path.exists():
        return ''
    rows = [json.loads(l) for l in scores_path.read_text().strip().splitlines()]
    steps = [r.get('step', 0) for r in rows]
    scores = [float(r.get('episode/score', r.get('score', 0))) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(np.array(steps) / 1e6, scores, alpha=0.7, s=40, c=COLORS['DreamerV3'], label='Local V3 episodes')
    ax.plot(np.array(steps) / 1e6, pd.Series(scores).rolling(5, min_periods=1).mean(),
            color='black', lw=1.5, alpha=0.6, label='Rolling mean (5 ep)')
    ax.set(xlabel='Training steps (M)', ylabel='Episode score (milestone index)', title='Local Minecraft training progress')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = cfg.report_dir / 'inference' / 'minecraft_local_overlay.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


def plot_atari_heatmap(cfg: PathConfig, compare: pd.DataFrame | None = None, show: bool = False) -> str:
    compare = compare if compare is not None else load_atari_compare(cfg)
    c = compare.sort_values('delta', ascending=True)
    data = c[['DreamerV2', 'DreamerV3']].values
    fig, ax = plt.subplots(figsize=(6, 14))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=-2, vmax=10)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels([t.replace('atari_', '') for t in c.index], fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['DreamerV2', 'DreamerV3'])
    ax.set_title('Atari 57 — HNS per game @ 50M steps')
    plt.colorbar(im, ax=ax, label='HNS')
    plt.tight_layout()
    out = cfg.report_dir / 'atari_hns_heatmap.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


def _game_runs(cfg: PathConfig, game: str) -> tuple[list, list, dict]:
    v3 = viz._load_cached(cfg.scores_cache, 'atari57-dreamerv3.json',
        'https://raw.githubusercontent.com/danijar/dreamerv3/main/scores/atari57-dreamerv3.json.gz', gz=True)
    v2 = viz._load_cached(cfg.scores_cache, 'atari-dreamerv2.json',
        'https://raw.githubusercontent.com/danijar/dreamerv2/main/scores/atari-dreamerv2.json')
    baselines = viz._load_cached(cfg.scores_cache, 'baselines.json',
        'https://raw.githubusercontent.com/danijar/dreamerv2/main/scores/baselines.json')
    task = f'atari_{game}'
    r2 = [r for r in v2 if r['task'] == task]
    r3 = [r for r in v3 if r['task'] == task]
    return r2, r3, baselines


def plot_atari_10game_curves(cfg: PathConfig, games: list[str], show: bool = False) -> str:
    n = len(games)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    for ax, game in zip(axes, games):
        r2, r3, baselines = _game_runs(cfg, game)
        task = f'atari_{game}'
        for runs, label, color in [(r2, 'V2', COLORS['DreamerV2']), (r3, 'V3', COLORS['DreamerV3'])]:
            if not runs:
                continue
            run = runs[0]
            hns = [viz.hns(y, task, baselines) for y in run['ys']]
            ax.plot(np.array(run['xs']) / 1e6, hns, label=label, color=color, lw=1.5)
        ax.set_title(game, fontsize=9)
        ax.set_xlabel('Steps (M)', fontsize=7)
        ax.set_ylabel('HNS', fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    for j in range(len(games), len(axes)):
        axes[j].axis('off')
    fig.suptitle('Atari — 10 representative games (official learning curves)', y=1.02)
    plt.tight_layout()
    out = cfg.report_dir / 'atari_10game_curves.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


def plot_atari_10game_bars(cfg: PathConfig, games: list[str], compare: pd.DataFrame, show: bool = False) -> str:
    rows = compare.loc[[f'atari_{g}' for g in games if f'atari_{g}' in compare.index]]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(rows))
    w = 0.35
    ax.bar(x - w / 2, rows['DreamerV2'], w, label='DreamerV2', color=COLORS['DreamerV2'])
    ax.bar(x + w / 2, rows['DreamerV3'], w, label='DreamerV3', color=COLORS['DreamerV3'])
    ax.set_xticks(x)
    ax.set_xticklabels([i.replace('atari_', '') for i in rows.index], rotation=35, ha='right')
    ax.set_ylabel('HNS @ 50M steps')
    ax.set_title('Atari — 10 games head-to-head')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = cfg.report_dir / 'atari_10game_bars.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


def plot_atari_10game_panels(cfg: PathConfig, games: list[str], show: bool = False) -> str:
    """Side-by-side score trajectories: V2 (left column) vs V3 (right column) per game."""
    n = len(games)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.2 * n))
    for i, game in enumerate(games):
        r2, r3, baselines = _game_runs(cfg, game)
        task = f'atari_{game}'
        for col, (runs, title) in enumerate([(r2, 'DreamerV2'), (r3, 'DreamerV3')]):
            ax = axes[i, col]
            if runs:
                run = runs[0]
                hns = [viz.hns(y, task, baselines) for y in run['ys']]
                ax.plot(np.array(run['xs']) / 1e6, hns, color=COLORS[title], lw=1.5)
                ax.scatter(np.array(run['xs'][-1:]) / 1e6, hns[-1:], s=30, color=COLORS[title], zorder=5)
            ax.set_ylabel('HNS', fontsize=7)
            if i == 0:
                ax.set_title(title, fontsize=9)
            if i == n - 1:
                ax.set_xlabel('Steps (M)', fontsize=7)
            ax.grid(alpha=0.3)
        axes[i, 0].set_ylabel(f'{game}\nHNS', fontsize=7)
    fig.suptitle('Atari — V2 vs V3 score trajectories (10 games)', y=1.01, fontsize=11)
    plt.tight_layout()
    out = cfg.report_dir / 'atari_10game_panels.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


def plot_executive_summary(cfg: PathConfig, compare: pd.DataFrame, mc_summary: dict, show: bool = False) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    delta = compare['delta'].dropna()
    axes[0].bar(['V2 median', 'V3 median'], [compare['DreamerV2'].median(), compare['DreamerV3'].median()],
                color=[COLORS['DreamerV2'], COLORS['DreamerV3']])
    axes[0].set_ylabel('HNS @ 50M')
    axes[0].set_title('Atari 57 — median HNS')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].pie(
        [ (delta > 0).sum(), (delta <= 0).sum()],
        labels=['V3 higher', 'V2 ≥ V3'],
        colors=[COLORS['DreamerV3'], COLORS['DreamerV2']],
        autopct='%1.0f%%',
    )
    axes[1].set_title(f'Per-game winners (n={len(delta)})')

    names = list(mc_summary.keys())
    vals = [mc_summary[n].get('mean_max_milestone', 0) if isinstance(mc_summary[n], dict) else 0 for n in names]
    axes[2].bar(names, vals, color=[COLORS.get(n, 'gray') for n in names])
    axes[2].set_ylabel('Mean max milestone')
    axes[2].set_title('Minecraft — official progress depth')
    axes[2].tick_params(axis='x', rotation=15)
    plt.tight_layout()
    out = cfg.report_dir / 'executive_summary.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out)


def run_advanced_viz(cfg: PathConfig, logdir: Path | None = None, show: bool = False) -> dict:
    """Generate all extended static visualizations."""
    logdir = Path(logdir or cfg.minecraft_full_logdir)
    compare = load_atari_compare(cfg)
    games = select_10_games(compare)
    mc_summary = plot_minecraft_baselines_enhanced(cfg, show=show)
    viz.plot_minecraft_baselines(cfg, show=False)
    viz.plot_atari_v2_v3(cfg, show=False)
    from .inference_demo import plot_atari_hns_distribution, plot_local_minecraft_scores
    plot_atari_hns_distribution(cfg, show=False, compare=compare)
    if (logdir / 'scores.jsonl').exists():
        plot_local_minecraft_scores(cfg, logdir=logdir, show=show)
    return {
        'games': games,
        'compare': compare,
        'mc_summary': mc_summary,
        'minecraft_enhanced': str(cfg.report_dir / 'minecraft_baselines_enhanced.png'),
        'minecraft_overlay': plot_minecraft_local_overlay(cfg, logdir, show=show),
        'atari_heatmap': plot_atari_heatmap(cfg, compare, show=show),
        'atari_10_curves': plot_atari_10game_curves(cfg, games, show=show),
        'atari_10_bars': plot_atari_10game_bars(cfg, games, compare, show=show),
        'atari_10_panels': plot_atari_10game_panels(cfg, games, show=show),
        'executive': plot_executive_summary(cfg, compare, mc_summary, show=show),
        'median_hns_v2': float(compare['DreamerV2'].median()),
        'median_hns_v3': float(compare['DreamerV3'].median()),
        'v3_win_pct': float((compare['delta'] > 0).mean() * 100),
    }

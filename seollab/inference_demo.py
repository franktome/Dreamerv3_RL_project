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


def training_milestone_chains(logdir: Path, top_n: int = 3) -> list[dict]:
    """Top training episodes as milestone chains (from scores.jsonl)."""
    scores_path = Path(logdir) / 'scores.jsonl'
    if not scores_path.exists():
        return []
    rows = [
        json.loads(line)
        for line in scores_path.read_text().strip().splitlines()
        if line.strip()
    ]
    ranked = sorted(
        rows,
        key=lambda r: float(r.get('episode/score', r.get('score', 0))),
        reverse=True,
    )
    chains = []
    for row in ranked[:top_n]:
        score = float(row.get('episode/score', row.get('score', 0)))
        idx = max(0, min(int(round(score)), len(MILESTONE_LABELS) - 1))
        chains.append({
            'score': score,
            'step': int(row.get('step', 0)),
            'milestones': MILESTONE_LABELS[: idx + 1],
        })
    return chains


def ensure_minecraft_training_plots(cfg: PathConfig, logdir: Path | None = None) -> dict:
    """Refresh local Minecraft training PNGs used in the notebook."""
    from .viz_advanced import plot_minecraft_local_overlay

    logdir = Path(logdir or cfg.minecraft_full_logdir)
    out = {'scores_png': '', 'overlay_png': ''}
    if (logdir / 'scores.jsonl').exists():
        scores = plot_local_minecraft_scores(cfg, logdir=logdir, show=False)
        if scores.get('ok'):
            out['scores_png'] = scores.get('png', '')
    overlay = plot_minecraft_local_overlay(cfg, logdir, show=False)
    if overlay:
        out['overlay_png'] = overlay
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
    seed: int | None = None,
) -> dict:
    """Policy rollout GIF from latest checkpoint (safe while training)."""
    logdir = Path(logdir or cfg.minecraft_full_logdir)
    out_dir = cfg.highlights_dir / 'inference'
    out_dir.mkdir(parents=True, exist_ok=True)
    gif = out_dir / 'minecraft_diamond_rollout.gif'
    return minecraft.infer(
        cfg, logdir=logdir, gif_path=gif,
        max_steps=max_steps, train_mode=train_mode, seed=seed,
    )


def run_minecraft_multi_rollouts(
    cfg: PathConfig,
    logdir: Path | None = None,
    n_rollouts: int = 6,
    max_steps: int = 3600,
    seeds: list[int] | None = None,
    top_k: int = 3,
) -> dict:
    """Several rollout GIFs; best milestone chain copied to main inference paths."""
    return minecraft.infer_multi_rollouts(
        cfg,
        logdir=logdir or cfg.minecraft_full_logdir,
        n_rollouts=n_rollouts,
        max_steps=max_steps,
        seeds=seeds,
        top_k=top_k,
    )


def load_minecraft_rollout_index(
    cfg: PathConfig,
    logdir: Path | None = None,
    top_k: int = 3,
) -> dict:
    """Rebuild rollout index from disk and merge saved reward/milestone metadata."""
    logdir = Path(logdir or cfg.minecraft_full_logdir)
    index_path = cfg.highlights_dir / 'inference' / 'minecraft_rollouts_index.json'
    saved = json.loads(index_path.read_text()) if index_path.exists() else {}
    scanned = minecraft.scan_rollout_cache(cfg, logdir=logdir, top_k=top_k)
    if not scanned.get('ok'):
        return saved if saved.get('ok') else scanned
    if saved.get('ok') and saved.get('all_rollouts'):
        meta = {r['gif']: r for r in saved['all_rollouts']}
        for row in scanned['all_rollouts']:
            old = meta.get(row['gif'])
            if not old:
                continue
            for key in (
                'reward', 'steps', 'milestones_reached', 'milestone_count',
                'max_milestone', 'gallery_png', 'milestone_pngs', 'milestones_dir',
            ):
                if old.get(key):
                    row[key] = old[key]
        ranked = sorted(
            scanned['all_rollouts'],
            key=lambda r: (r['milestone_count'], r.get('reward', 0), r['seed']),
            reverse=True,
        )
        scanned['all_rollouts'] = ranked
        scanned['best'] = ranked[0]
        scanned['top_k'] = minecraft._diverse_top_k(ranked, k=top_k)
        for key in ('checkpoint_id', 'training_step', 'max_steps', 'seeds'):
            if saved.get(key):
                scanned[key] = saved[key]
        index_path.write_text(json.dumps(scanned, indent=2))
    return scanned


def display_minecraft_inference(
    cfg: PathConfig,
    mc_index: dict,
    health: dict,
    logdir: Path | None = None,
) -> dict:
    """Render Minecraft checkpoint + rollout GIFs / milestone galleries in the notebook."""
    from IPython.display import Image, Markdown, display

    logdir = Path(logdir or cfg.minecraft_full_logdir)
    inf = cfg.highlights_dir / 'inference'
    best = mc_index.get('best') or {}
    best_chains = training_milestone_chains(logdir, top_n=3)

    ckpt_id = mc_index.get('checkpoint_id') or health.get('checkpoint', '?')
    train_step = mc_index.get('training_step') or health.get('step', '?')

    mc_result = {
        'ok': mc_index.get('ok', False),
        'gif': mc_index.get('main_gif', str(inf / 'minecraft_diamond_rollout.gif')),
        'strip_gif': mc_index.get('main_strip', str(inf / 'minecraft_milestone_strip.gif')),
        'gallery_png': mc_index.get('main_gallery', str(inf / 'minecraft_milestone_gallery.png')),
        'reward': best.get('reward', health.get('max_episode_score', 0)),
        'max_milestone': best.get('max_milestone', health.get('max_milestone_name', '?')),
        'milestones_reached': best.get('milestones_reached', []),
        'checkpoint_id': ckpt_id,
        'training_step': train_step,
        'training_episodes': health.get('episodes', 0),
        'mean_episode_score': health.get('mean_episode_score', 0),
    }

    if not mc_result['ok']:
        display(Markdown(f"*{mc_index.get('error', 'No checkpoint / rollouts')}*"))
        return mc_result

    cached = ' (rebuilt from cache)' if mc_index.get('cached') else ''
    display(Markdown(
        '### Checkpoint & training status\n'
        f"- **Checkpoint:** `{ckpt_id}` @ training step **{train_step:,}**\n"
        f"- **Logdir:** `{logdir}`\n"
        f"- **Rollouts indexed:** {mc_index.get('n_rollouts', '?')}{cached}\n"
        f"- **Policy fps:** {health.get('fps_policy', '?')}"
    ))
    if health.get('issues'):
        display(Markdown('**Health notes:** ' + '; '.join(health['issues'])))

    display(Markdown(
        '### Training summary (`scores.jsonl`)\n'
        f"- Episodes: **{mc_result['training_episodes']}**\n"
        f"- Max training score: **{health.get('max_episode_score', 0):.1f}** "
        f"(milestone **{health.get('max_milestone_name', '?')}**)\n"
        f"- Mean episode score: **{mc_result['mean_episode_score']:.2f}**"
    ))
    if best_chains:
        lines = [
            f"{i}. score **{c['score']:.1f}** @ step {c['step']:,} — "
            + ' → '.join(c['milestones'])
            for i, c in enumerate(best_chains, 1)
        ]
        display(Markdown('**Top training episodes**\n' + '\n'.join(f'- {ln}' for ln in lines)))

    display(Markdown('### Best inference rollout (deepest milestone chain)'))
    if mc_result.get('milestones_reached'):
        display(Markdown(
            '**Milestones:** ' + ' → '.join(mc_result['milestones_reached'])
            + f"  \n**Reward:** {mc_result.get('reward', 0):.2f}"
        ))

    gallery = mc_result.get('gallery_png', '')
    if gallery and Path(gallery).exists():
        display(Markdown('**Milestone event frames (best rollout)**'))
        display(Image(filename=gallery))
    elif mc_result.get('strip_gif') and Path(mc_result['strip_gif']).exists():
        display(Markdown('**Milestone strip (best rollout)**'))
        display(Image(filename=mc_result['strip_gif']))

    gif = mc_result.get('gif', '')
    if gif and Path(gif).exists():
        display(Markdown('**Full episode GIF (best rollout)**'))
        display(Image(filename=gif))

    top = mc_index.get('top_k') or []
    if top:
        display(Markdown('### Diverse rollouts (distinct milestone chains when available)'))
        for item in top:
            chain = ' → '.join(item.get('milestones_reached', [])) or '(none)'
            display(Markdown(
                f"**#{item.get('rank', '?')}** seed={item.get('seed')} — "
                f"**{item.get('max_milestone')}** "
                f"({item.get('milestone_count', 0)} events, reward={item.get('reward', 0):.2f})\n"
                f"{chain}"
            ))
            if item.get('gallery_png') and Path(item['gallery_png']).exists():
                display(Image(filename=item['gallery_png']))
            elif item.get('strip_gif') and Path(item['strip_gif']).exists():
                display(Image(filename=item['strip_gif']))
            pngs = item.get('milestone_pngs') or {}
            if pngs:
                ordered = [pngs[n] for n in item.get('milestones_reached', []) if n in pngs]
                for p in ordered[:6]:
                    if Path(p).exists():
                        display(Image(filename=p, width=140))
            if item.get('gif') and Path(item['gif']).exists():
                display(Image(filename=item['gif'], width=360))

    scores = plot_local_minecraft_scores(cfg, logdir=logdir, show=False)
    if scores.get('ok') and Path(scores.get('png', '')).exists():
        display(Markdown('### Training milestone distribution'))
        display(Image(filename=scores['png']))

    return mc_result


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

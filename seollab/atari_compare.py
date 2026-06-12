"""Orchestrate 3-game DreamerV2 vs V3 timed training, inference, anim, and metrics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .paths import PathConfig
from . import atari, atari_v2, atari_anim, atari_align, gif_compare
from . import viz_atari_compare

GAMES = atari.COMPARE_GAMES


def run_smoke(cfg: PathConfig, games: tuple[str, ...] = GAMES, mem_fraction: float = 0.25) -> dict:
    """Mini train+infer for each game and model."""
    results = {'ok': True, 'games': {}}
    for game in games:
        g = {'v2': {}, 'v3': {}}
        try:
            v2_ckpt = cfg.atari_logdir(game, 'v2') / 'variables.pkl'
            if not v2_ckpt.exists():
                atari_v2.train_v2_timed(cfg, game, smoke=True, mem_fraction=mem_fraction)
            g['v2']['train'] = 'ok' if v2_ckpt.exists() or True else 'ran'
            g['v2']['infer'] = atari_v2.infer_v2_gif(cfg, game, max_steps=100)
            if not g['v2']['infer'].get('ok'):
                results['ok'] = False
        except Exception as e:
            g['v2']['error'] = str(e)[:500]
            results['ok'] = False
        try:
            v3_ckpt = cfg.atari_logdir(game, 'v3') / 'ckpt'
            if not (v3_ckpt.exists() and any(v3_ckpt.iterdir())):
                atari.train_v3_timed(cfg, game, smoke=True, mem_fraction=mem_fraction)
            g['v3']['train'] = 'ok'
            g['v3']['infer'] = atari.infer_v3(cfg, game, max_steps=100)
            if not g['v3']['infer'].get('ok'):
                results['ok'] = False
        except Exception as e:
            g['v3']['error'] = str(e)[:500]
            results['ok'] = False
        results['games'][game] = g
    return results


def run_timed_training(
    cfg: PathConfig,
    games: tuple[str, ...] = GAMES,
    minutes_per_run: int = 90,
    mem_fraction: float = 0.25,
    skip_existing: bool = True,
) -> dict:
    """Sequential V2 then V3 training per game (equal wall-clock)."""
    log = {}
    for game in games:
        log[game] = {}
        v2_dir = cfg.atari_logdir(game, 'v2')
        if skip_existing and (v2_dir / 'variables.pkl').exists():
            log[game]['v2'] = {'skipped': True, 'logdir': str(v2_dir)}
        else:
            atari_v2.train_v2_timed(cfg, game, minutes=minutes_per_run, mem_fraction=mem_fraction)
            log[game]['v2'] = {'logdir': str(v2_dir)}
        v3_dir = cfg.atari_logdir(game, 'v3')
        if skip_existing and (v3_dir / 'ckpt').exists() and any((v3_dir / 'ckpt').iterdir()):
            log[game]['v3'] = {'skipped': True, 'logdir': str(v3_dir)}
        else:
            atari.train_v3_timed(cfg, game, minutes=minutes_per_run, mem_fraction=mem_fraction)
            log[game]['v3'] = {'logdir': str(v3_dir)}
    return log


def run_inference_and_anim(
    cfg: PathConfig,
    games: tuple[str, ...] = GAMES,
    max_steps: int = 1500,
    run_anim: bool = True,
) -> dict:
    """Generate compare GIFs and anim perturbation runs."""
    out = {'compare': {}, 'anim': {}}
    for game in games:
        out['compare'][game] = gif_compare.infer_both(cfg, game, max_steps=max_steps)
        if run_anim:
            out['anim'][game] = {}
            for preset in ('fast', 'sluggish'):
                out['anim'][game][preset] = atari_anim.run_anim_compare(
                    cfg, game, preset=preset, max_steps=max_steps)
    if run_anim and out['anim']:
        from . import atari_anim_ui
        atari_anim_ui.merge_anim_results(cfg, out['anim'])
    return out


def collect_metrics(cfg: PathConfig, games: tuple[str, ...] = GAMES) -> dict:
    """Aggregate training and inference metrics (truncated to aligned env steps)."""
    import pandas as pd
    import numpy as np
    from . import viz

    align_rows = atari_align.alignment_table(cfg, games)
    align_map = {r['game']: r['aligned_env_steps'] for r in align_rows}

    rows = []
    for game in games:
        aligned = align_map.get(game, 0)
        for ver, loader, tag in (
            ('v2', atari_v2.load_v2_scores, 'DreamerV2'),
            ('v3', atari.load_v3_scores, 'DreamerV3'),
        ):
            logdir = cfg.atari_logdir(game, ver)
            scores = loader(logdir)
            ep_returns = []
            seen_steps = set()
            for s in scores:
                step = int(s.get('step', 0))
                env_step = (
                    int(s['train_total_steps']) if 'train_total_steps' in s
                    else step // atari_align.ATARI_REPEAT
                )
                if aligned and env_step > aligned:
                    continue
                if 'train_return' in s and s.get('step') not in seen_steps:
                    ep_returns.append(float(s['train_return']))
                    seen_steps.add(s.get('step'))
                elif 'episode/score' in s:
                    ep_returns.append(float(s['episode/score']))
                elif 'score' in s and 'episode' in str(s):
                    ep_returns.append(float(s['score']))
            infer_gif = cfg.highlights_dir / 'inference' / f'atari_{game}_{ver}_rollout.gif'
            compare = cfg.highlights_dir / 'inference' / f'atari_{game}_v2v3_compare.gif'
            row = {
                'game': game,
                'model': tag,
                'episodes': len(ep_returns),
                'max_return': max(ep_returns) if ep_returns else 0,
                'mean_return': float(np.mean(ep_returns)) if ep_returns else 0,
                'aligned_env_steps': aligned,
                'logdir': str(logdir),
                'gif': str(infer_gif) if infer_gif.exists() else '',
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    baselines = viz._load_cached(
        cfg.scores_cache, 'baselines.json',
        'https://raw.githubusercontent.com/danijar/dreamerv2/main/scores/baselines.json',
    )
    hns_rows = []
    for game in games:
        task = f'atari_{game}'
        if task not in baselines:
            continue
        for ver, tag in (('v2', 'DreamerV2'), ('v3', 'DreamerV3')):
            scores = atari_v2.load_v2_scores(cfg.atari_logdir(game, ver)) if ver == 'v2' else atari.load_v3_scores(cfg.atari_logdir(game, ver))
            ep_returns = [float(s.get('train_return', s.get('episode/score', s.get('score', 0)))) for s in scores]
            if not ep_returns:
                continue
            y = max(ep_returns)
            hns = viz.hns(y, task, baselines)
            hns_rows.append({'game': game, 'model': tag, 'hns': hns, 'raw_score': y})
    return {
        'summary': df,
        'hns': pd.DataFrame(hns_rows) if hns_rows else None,
        'alignment': pd.DataFrame(align_rows),
    }


def run_full_pipeline(
    cfg: PathConfig,
    smoke: bool = False,
    train_minutes: int = 90,
    skip_train: bool = False,
) -> dict:
    """Full compare pipeline: smoke (optional) → train → infer → viz → report."""
    cfg.apply_env(mem_fraction=0.25)
    result = {'started': datetime.now().isoformat(), 'games': list(GAMES)}
    if smoke:
        result['smoke'] = run_smoke(cfg)
        if not result['smoke']['ok']:
            return result
    if not skip_train:
        result['training'] = run_timed_training(cfg, minutes_per_run=train_minutes)
    result['inference'] = run_inference_and_anim(cfg)
    metrics = collect_metrics(cfg)
    result['metrics'] = metrics
    result['plots'] = viz_atari_compare.generate_all(cfg, metrics, GAMES)
    result['report'] = viz_atari_compare.write_atari_compare_report(cfg, metrics, result)
    result['finished'] = datetime.now().isoformat()
    cache = cfg.report_dir / 'atari_compare' / 'pipeline_result.json'
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({k: v for k, v in result.items() if k != 'metrics'}, default=str, indent=2))
    return result

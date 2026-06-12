"""Visualization from official score JSON files."""

import gzip
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import PathConfig


def _load_cached(cache_dir: Path, name: str, url: str, gz: bool = False):
    cache = cache_dir / name
    if not cache.exists():
        with urllib.request.urlopen(url, timeout=180) as resp:
            data = resp.read()
        if gz or url.endswith('.gz'):
            data = gzip.decompress(data)
        cache.write_bytes(data)
    return json.loads(cache.read_text())


def hns(score, task, baselines):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return np.nan
    b = baselines.get(task, {})
    human, random = b.get('human_gamer'), b.get('random')
    if human is None or random is None or human == random:
        return np.nan
    return (score - random) / (human - random)


def plot_minecraft_baselines(cfg: PathConfig, show: bool = True) -> dict:
    urls = {
        'DreamerV3': 'https://raw.githubusercontent.com/danijar/dreamerv3/main/scores/minecraft_diamond-dreamerv3.json.gz',
        'PPO': 'https://raw.githubusercontent.com/danijar/dreamerv3/main/scores/minecraft_diamond-ppo_fixhp.json.gz',
        'IMPALA': 'https://raw.githubusercontent.com/danijar/dreamerv3/main/scores/minecraft_diamond-impala.json.gz',
    }
    runs = {n: _load_cached(cfg.scores_cache, f'minecraft_{n.lower()}.json', u, gz=True) for n, u in urls.items()}

    def curve(rs, n=80):
        max_x = max(max(r['xs']) for r in rs)
        xs = np.unique(np.geomspace(max(1, min(min(r['xs']) for r in rs)), max_x, n).astype(int))
        rates = []
        for x in xs:
            succ = []
            for run in rs:
                valid = [i for i, xv in enumerate(run['xs']) if xv <= x]
                if not valid:
                    continue
                best = max(run['ys'][i] for i in valid)
                succ.append(1.0 if best >= 12 else 0.0)
            rates.append(np.mean(succ) if succ else np.nan)
        return xs, np.array(rates)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'DreamerV3': '#2ca02c', 'PPO': '#ff7f0e', 'IMPALA': '#1f77b4'}
    summary = {}
    for name, rs in runs.items():
        xs, rates = curve(rs)
        mask = ~np.isnan(rates)
        axes[0].plot(xs[mask] / 1e6, rates[mask] * 100, label=name, color=colors[name], lw=2)
        summary[name] = float(np.mean([1.0 if max(r['ys']) >= 12 else 0.0 for r in rs]) * 100)
    axes[0].set(xlabel='Steps (M)', ylabel='Task success (%)', title='Minecraft — official scores')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for name, rs in runs.items():
        ms = [min(12, round(max(r['ys']))) for r in rs]
        axes[1].hist(ms, bins=range(0, 14), alpha=0.5, label=name, color=colors[name])
    axes[1].set(xlabel='Max milestone', ylabel='Seeds', title='Milestone distribution')
    axes[1].legend()
    plt.suptitle('DreamerV2 on Minecraft: N/A', y=1.02)
    plt.tight_layout()
    out = cfg.report_dir / 'minecraft_baselines.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return summary


def plot_atari_v2_v3(cfg: PathConfig, show: bool = True) -> pd.DataFrame:
    v3 = _load_cached(cfg.scores_cache, 'atari57-dreamerv3.json',
        'https://raw.githubusercontent.com/danijar/dreamerv3/main/scores/atari57-dreamerv3.json.gz', gz=True)
    v2 = _load_cached(cfg.scores_cache, 'atari-dreamerv2.json',
        'https://raw.githubusercontent.com/danijar/dreamerv2/main/scores/atari-dreamerv2.json')
    baselines = _load_cached(cfg.scores_cache, 'baselines.json',
        'https://raw.githubusercontent.com/danijar/dreamerv2/main/scores/baselines.json')

    def final_hns(runs, budget=50_000_000):
        rows = []
        for run in runs:
            valid = [i for i, xv in enumerate(run['xs']) if xv <= budget]
            if not valid:
                continue
            idx = max(valid)
            rows.append((run['task'], hns(run['ys'][idx], run['task'], baselines)))
        return pd.Series(dict(rows), name='hns')

    v3s = final_hns(v3).rename('DreamerV3')
    v2s = final_hns(v2).rename('DreamerV2')
    compare = pd.concat([v2s, v3s], axis=1).dropna()
    compare['delta'] = compare['DreamerV3'] - compare['DreamerV2']
    compare['Δ'] = compare['delta']

    cs = compare.sort_values('DreamerV3')
    n = len(cs)
    fig_h = max(12.0, 4.0 + n * 0.17)
    fig, axes = plt.subplots(
        2, 1, figsize=(14, fig_h),
        gridspec_kw={'height_ratios': [1.2, max(4.0, n * 0.14)]},
    )
    def med_curve(runs):
        tasks = sorted({r['task'] for r in runs})
        max_x = min(50_000_000, max(max(r['xs']) for r in runs))
        xs = np.unique(np.geomspace(max(1, min(min(r['xs']) for r in runs)), max_x, 50).astype(int))
        meds = []
        for x in xs:
            vals = []
            for task in tasks:
                sc = []
                for run in runs:
                    if run['task'] != task:
                        continue
                    valid = [i for i, xv in enumerate(run['xs']) if xv <= x]
                    if valid:
                        sc.append(hns(run['ys'][max(valid)], task, baselines))
                if sc:
                    vals.append(np.nanmean(sc))
            meds.append(np.nanmedian(vals) if vals else np.nan)
        return xs, np.array(meds)

    xs2, m2 = med_curve(v2)
    xs3, m3 = med_curve(v3)
    axes[0].plot(xs2 / 1e6, m2, label='DreamerV2', lw=2)
    axes[0].plot(xs3 / 1e6, m3, label='DreamerV3', lw=2)
    axes[0].axhline(1.0, color='gray', ls='--', alpha=0.5)
    axes[0].set(xlabel='Steps (M)', ylabel='Median HNS', title='Atari 57')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    y = np.arange(n)
    w = 0.35
    axes[1].barh(y - w / 2, cs['DreamerV2'], height=w, label='DreamerV2', color='#1f77b4', alpha=0.85)
    axes[1].barh(y + w / 2, cs['DreamerV3'], height=w, label='DreamerV3', color='#ff7f0e', alpha=0.85)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([t.replace('atari_', '') for t in cs.index], fontsize=6)
    axes[1].set_xlabel('HNS @ 50M')
    axes[1].set_title(f'Per-game HNS @ 50M ({n} games)')
    axes[1].legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(cfg.report_dir / 'atari_v2_v3.png', dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    compare.to_csv(cfg.report_dir / 'atari_per_game_hns.csv')
    print('Median HNS V2:', compare['DreamerV2'].median(), 'V3:', compare['DreamerV3'].median())
    print('V3 wins:', f"{(compare['delta'] > 0).mean() * 100:.1f}%")
    return compare

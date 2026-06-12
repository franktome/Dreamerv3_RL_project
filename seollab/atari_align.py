"""Align DreamerV2 vs V3 training progress for fair comparison."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

from .paths import PathConfig

ATARI_REPEAT = 4


@dataclass
class TrainProgress:
    version: str
    logdir: str
    logged_step: int
    env_steps: int
    episodes: int
    ckpt_path: str
    ckpt_env_steps: int | None = None


@dataclass
class AlignResult:
    game: str
    v2: TrainProgress
    v3: TrainProgress
    aligned_env_steps: int
    aligned_episodes: int
    fair_gif: bool
    note: str


def v2_progress(logdir: Path) -> TrainProgress:
    metrics = logdir / 'metrics.jsonl'
    best = {}
    if metrics.exists():
        for line in metrics.read_text().strip().splitlines():
            row = json.loads(line)
            if 'train_total_steps' in row:
                best = row
    env_steps = int(best.get('train_total_steps', 0))
    episodes = int(best.get('train_total_episodes', 0))
    logged = int(best.get('step', env_steps * ATARI_REPEAT))
    ckpt = logdir / 'variables.pkl'
    return TrainProgress(
        version='v2',
        logdir=str(logdir),
        logged_step=logged,
        env_steps=env_steps,
        episodes=episodes,
        ckpt_path=str(ckpt) if ckpt.exists() else '',
        ckpt_env_steps=env_steps if ckpt.exists() else None,
    )


def _read_v3_ckpts(logdir: Path) -> list[tuple[int, Path]]:
    ckpt_root = logdir / 'ckpt'
    out = []
    if not ckpt_root.exists():
        return out
    for d in ckpt_root.iterdir():
        if not d.is_dir():
            continue
        sp = d / 'step.pkl'
        if sp.exists():
            step = int(pickle.loads(sp.read_bytes()))
            out.append((step, d))
    out.sort(key=lambda x: x[0])
    return out


def v3_progress(logdir: Path) -> TrainProgress:
    scores = logdir / 'scores.jsonl'
    logged = env_steps = episodes = 0
    if scores.exists():
        lines = [json.loads(l) for l in scores.read_text().strip().splitlines() if l.strip()]
        episodes = len(lines)
        if lines:
            logged = int(lines[-1].get('step', 0))
            env_steps = logged // ATARI_REPEAT
    ckpts = _read_v3_ckpts(logdir)
    ckpt_path = ''
    ckpt_env_steps = None
    if ckpts:
        ckpt_env_steps, folder = ckpts[-1]
        ckpt_path = str(folder)
        # scores.jsonl only records completed episodes, so it can lag the
        # checkpoint's true step counter; the ckpt step is also real progress.
        env_steps = max(env_steps, ckpt_env_steps)
        logged = max(logged, ckpt_env_steps * ATARI_REPEAT)
    return TrainProgress(
        version='v3',
        logdir=str(logdir),
        logged_step=logged,
        env_steps=env_steps,
        episodes=episodes,
        ckpt_path=ckpt_path,
        ckpt_env_steps=ckpt_env_steps,
    )


def select_v2_snapshot(logdir: Path, target_env_steps: int) -> tuple[Path | None, int | None]:
    """Pick the V2 sidecar snapshot closest to target (prefer <= target)."""
    snapdir = logdir / 'snapshots'
    cands: list[tuple[int, Path]] = []
    if snapdir.exists():
        for p in snapdir.glob('variables_*.pkl'):
            try:
                cands.append((int(p.stem.split('_')[1]), p))
            except (IndexError, ValueError):
                continue
    if not cands:
        return None, None
    below = [c for c in cands if c[0] <= target_env_steps]
    step, path = max(below) if below else min(cands, key=lambda c: abs(c[0] - target_env_steps))
    return path, step


def select_v3_ckpt_dir(logdir: Path, target_env_steps: int) -> tuple[Path | None, int | None, str]:
    """Pick the newest V3 checkpoint with env step <= target."""
    ckpts = _read_v3_ckpts(logdir)
    if not ckpts:
        return None, None, 'no V3 checkpoints'
    best = None
    for step, folder in ckpts:
        if step <= target_env_steps:
            best = (folder, step)
    if best:
        return best[0], best[1], 'ok'
    if ckpts:
        step, folder = ckpts[0]
        return folder, step, f'no ckpt <= {target_env_steps}; using earliest ({step})'
    return None, None, 'no checkpoints'


def aligned_logdir(cfg: PathConfig, game: str) -> Path:
    return cfg.dreamerv3_root / 'logdir' / f'atari_{game}_v3_aligned'


def align_game(cfg: PathConfig, game: str) -> AlignResult:
    v2_dir = cfg.atari_logdir(game, 'v2')
    v3_dir = cfg.atari_logdir(game, 'v3')
    v2 = v2_progress(v2_dir)
    v3 = v3_progress(v3_dir)
    aligned = min(v2.env_steps, v3.env_steps)
    aligned_ep = min(v2.episodes, v3.episodes) if v2.episodes and v3.episodes else 0

    v3_ckpt_dir, v3_ckpt_step, _ = select_v3_ckpt_dir(v3_dir, aligned)
    align_dir = aligned_logdir(cfg, game)
    align_prog = v3_progress(align_dir) if align_dir.exists() else None

    fair = False
    notes = []

    if not v2.ckpt_path:
        notes.append('V2 checkpoint missing')
    elif v2.env_steps < aligned:
        notes.append(f'V2 behind aligned target ({v2.env_steps} < {aligned})')
    elif v2.env_steps > aligned:
        snap_path, snap_step = select_v2_snapshot(v2_dir, aligned)
        if snap_path:
            v2.ckpt_path = str(snap_path)
            v2.ckpt_env_steps = snap_step
            notes.append(f'V2 using sidecar snapshot at {snap_step} env steps (aligned {aligned})')
        else:
            notes.append(
                f'V2 latest ckpt is at {v2.env_steps} env steps (>{aligned}); '
                'only one variables.pkl — using latest (may overshoot aligned step)'
            )

    if align_prog and align_prog.ckpt_env_steps and align_prog.ckpt_env_steps <= aligned:
        fair = True
        notes.append(f'Using aligned V3 logdir at {align_prog.ckpt_env_steps} env steps')
    elif v3_ckpt_dir and v3_ckpt_step is not None and v3_ckpt_step <= aligned:
        fair = True
        notes.append(f'Using main V3 ckpt at {v3_ckpt_step} env steps')
    else:
        main_step = v3.ckpt_env_steps or 0
        notes.append(
            f'V3 main ckpt at {main_step} env steps > aligned {aligned}; '
            f'need aligned retrain in {align_dir.name}'
        )
        if align_prog and align_prog.ckpt_path:
            notes.append(f'Aligned logdir exists at step {align_prog.ckpt_env_steps}')

    if v2.env_steps >= aligned and (
        (align_prog and align_prog.ckpt_env_steps and align_prog.ckpt_env_steps <= aligned)
        or (v3_ckpt_step is not None and v3_ckpt_step <= aligned)
    ):
        fair = True

    return AlignResult(
        game=game,
        v2=v2,
        v3=v3,
        aligned_env_steps=aligned,
        aligned_episodes=aligned_ep,
        fair_gif=fair,
        note=' | '.join(notes),
    )


def alignment_table(cfg: PathConfig, games: tuple[str, ...]) -> list[dict]:
    rows = []
    for game in games:
        a = align_game(cfg, game)
        rows.append({
            'game': game,
            'v2_env_steps': a.v2.env_steps,
            'v2_episodes': a.v2.episodes,
            'v3_env_steps': a.v3.env_steps,
            'v3_episodes': a.v3.episodes,
            'v3_ckpt_env_steps': a.v3.ckpt_env_steps,
            'aligned_env_steps': a.aligned_env_steps,
            'aligned_episodes': a.aligned_episodes,
            'fair_gif': a.fair_gif,
            'note': a.note,
        })
    return rows


def ensure_v3_aligned_ckpt(cfg: PathConfig, game: str, env_steps: int) -> Path:
    """Train or reuse V3 checkpoint at <= aligned env steps."""
    from . import atari

    align_dir = aligned_logdir(cfg, game)
    ckpt_dir, ckpt_step, _ = select_v3_ckpt_dir(align_dir, env_steps)
    if ckpt_dir and ckpt_step is not None and ckpt_step >= env_steps * 0.9:
        return align_dir
    return atari.train_v3_to_env_steps(cfg, game, env_steps, logdir=align_dir)


def resolve_v3_infer_paths(cfg: PathConfig, game: str, aligned_env_steps: int) -> tuple[Path, Path | None]:
    """Return (logdir, ckpt_dir) for aligned V3 inference."""
    main_dir = cfg.atari_logdir(game, 'v3')
    ckpt_dir, ckpt_step, _ = select_v3_ckpt_dir(main_dir, aligned_env_steps)
    if ckpt_dir and ckpt_step is not None:
        return main_dir, ckpt_dir
    align_dir = aligned_logdir(cfg, game)
    ensure_v3_aligned_ckpt(cfg, game, aligned_env_steps)
    ckpt_dir, _, _ = select_v3_ckpt_dir(align_dir, aligned_env_steps)
    return align_dir, ckpt_dir


def write_alignment_report(cfg: PathConfig, games: tuple[str, ...]) -> str:
    rows = alignment_table(cfg, games)
    lines = [
        '# Atari V2 vs V3 — Step Alignment',
        '',
        'Comparable unit: **environment steps** (`train_total_steps` for V2, `scores step / 4` for V3).',
        'Aligned step = `min(V2, V3)` per game. GIFs should use checkpoints at or before this step.',
        '',
        '| Game | V2 steps | V2 ep | V3 steps | V3 ckpt | Aligned | Fair GIF? | Note |',
        '|------|----------|-------|----------|---------|---------|-----------|------|',
    ]
    for r in rows:
        lines.append(
            f"| {r['game']} | {r['v2_env_steps']:,} | {r['v2_episodes']} | "
            f"{r['v3_env_steps']:,} | {r['v3_ckpt_env_steps'] or '—'} | "
            f"**{r['aligned_env_steps']:,}** | {'✅' if r['fair_gif'] else '⚠️'} | {r['note']} |"
        )
    lines += [
        '',
        '## Step-by-step',
        '',
        '1. Read latest V2 `metrics.jsonl` → `train_total_steps`, `train_total_episodes`.',
        '2. Read V3 `scores.jsonl` (last step ÷ 4) and `ckpt/*/step.pkl`.',
        '3. `aligned_env_steps = min(v2, v3)`.',
        '4. V2: load `variables.pkl` (single file; equals aligned when V2 is the limiter).',
        '5. V3: load newest ckpt with `step <= aligned`; if none, train `atari_{game}_v3_aligned`.',
        '6. Run inference/GIF only when both checkpoints are at or before `aligned`.',
        '',
    ]
    path = cfg.report_dir / 'atari_compare' / 'ALIGNMENT.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
    return str(path)

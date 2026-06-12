"""Side-by-side GIF comparison for DreamerV2 vs DreamerV3 rollouts."""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataclasses import asdict

from .paths import PathConfig
from . import atari, atari_v2, atari_align
from .frame_utils import resize_frame, to_rgb_uint8

DISPLAY_SIZE = (96, 96)


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    return to_rgb_uint8(frame)


def stack_side_by_side(
    frames_l: list,
    frames_r: list,
    labels: tuple[str, str] = ('DreamerV2', 'DreamerV3'),
    scores: tuple[float, float] | None = None,
) -> list[np.ndarray]:
    """Horizontally stack two frame lists with labels."""
    n = max(len(frames_l), len(frames_r))
    out = []
    for i in range(n):
        fl = resize_frame(
            _to_rgb(frames_l[min(i, len(frames_l) - 1)]),
            DISPLAY_SIZE,
        )
        fr = resize_frame(
            _to_rgb(frames_r[min(i, len(frames_r) - 1)]),
            DISPLAY_SIZE,
        )
        h = DISPLAY_SIZE[1]
        combined = np.concatenate([fl, fr], axis=1)
        tile = Image.fromarray(combined)
        draw = ImageDraw.Draw(tile)
        txt = labels[0]
        if scores:
            txt += f' R={scores[0]:.1f}'
        draw.rectangle([0, 0, fl.shape[1], 14], fill=(0, 0, 0))
        draw.text((4, 1), txt, fill=(255, 200, 100))
        txt2 = labels[1]
        if scores:
            txt2 += f' R={scores[1]:.1f}'
        draw.rectangle([fl.shape[1], 0, combined.shape[1], 14], fill=(0, 0, 0))
        draw.text((fl.shape[1] + 4, 1), txt2, fill=(100, 255, 150))
        out.append(np.asarray(tile))
    return out


def load_gif_frames(path: Path) -> list[np.ndarray]:
    if not path.exists():
        return []
    return [np.asarray(f) for f in imageio.mimread(str(path))]


def save_compare_gif(frames: list, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, duration=125)
    return str(out_path)


def infer_both(
    cfg: PathConfig,
    game: str,
    max_steps: int = 1500,
    env_overrides: dict | None = None,
    tag: str = '',
    align: bool = True,
) -> dict:
    """Run V2 and V3 inference, save side-by-side GIF (optionally step-aligned)."""
    suffix = f'_{tag}' if tag else ''
    alignment = atari_align.align_game(cfg, game) if align else None
    aligned_steps = alignment.aligned_env_steps if alignment else None

    v2_out = cfg.highlights_dir / 'inference' / f'atari_{game}_v2_rollout{suffix}.gif'
    v3_out = cfg.highlights_dir / 'inference' / f'atari_{game}_v3_rollout{suffix}.gif'

    v2_vars = None
    if alignment and alignment.v2.ckpt_path and 'snapshots' in alignment.v2.ckpt_path:
        v2_vars = Path(alignment.v2.ckpt_path)
    v2 = atari_v2.infer_v2_gif(
        cfg, game, max_steps=max_steps, env_overrides=env_overrides,
        out_path=v2_out, variables_path=v2_vars,
    )

    v3_logdir, v3_ckpt = None, None
    if align and alignment:
        v3_logdir, v3_ckpt = atari_align.resolve_v3_infer_paths(cfg, game, alignment.aligned_env_steps)
    v3 = atari.infer_v3(
        cfg, game, max_steps=max_steps, env_overrides=env_overrides, out_path=v3_out,
        logdir=v3_logdir, ckpt_dir=v3_ckpt,
    )
    if not (v2.get('ok') and v3.get('ok')):
        return {
            'ok': False, 'v2': v2, 'v3': v3,
            'alignment': asdict(alignment) if alignment else None,
        }

    fl = load_gif_frames(Path(v2['gif']))
    fr = load_gif_frames(Path(v3['gif']))
    labels = ('DreamerV2', 'DreamerV3')
    if aligned_steps:
        labels = (
            f'DreamerV2 @{aligned_steps:,}',
            f'DreamerV3 @{aligned_steps:,}',
        )
    stacked = stack_side_by_side(
        fl, fr, labels=labels,
        scores=(v2.get('score', 0), v3.get('score', 0)),
    )
    compare_path = cfg.highlights_dir / 'inference' / f'atari_{game}_v2v3_compare{suffix}.gif'
    save_compare_gif(stacked, compare_path)
    scores_payload = {
        'v2': v2.get('score'),
        'v3': v3.get('score'),
        'delta': round(v3.get('score', 0) - v2.get('score', 0), 3),
        'tag': tag or 'default',
        'env_overrides': env_overrides or {},
    }
    compare_path.with_suffix('.scores.json').write_text(
        __import__('json').dumps(scores_payload, indent=2),
    )
    return {
        'ok': True,
        'game': game,
        'v2': v2,
        'v3': v3,
        'compare_gif': str(compare_path),
        'score_delta': round(v3.get('score', 0) - v2.get('score', 0), 3),
        'aligned_env_steps': aligned_steps,
        'alignment': asdict(alignment) if alignment else None,
    }

"""Atari inference perturbations (anim): env speed / control lag during rollout."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from .paths import PathConfig
from . import gif_compare


@dataclass
class AnimSpec:
  repeat: int = 4
  sticky: float = 0.25
  noops: int = 30
  seed: int | None = None

  def as_env_overrides(self) -> dict:
    return {'repeat': self.repeat, 'sticky': self.sticky, 'noops': self.noops}


ANIM_PRESETS: dict[str, AnimSpec] = {
    'baseline': AnimSpec(repeat=4, sticky=0.25, noops=30),
    'fast': AnimSpec(repeat=2, sticky=0.25, noops=30),
    'sluggish': AnimSpec(repeat=4, sticky=0.5, noops=30),
}


def run_anim_compare(
    cfg: PathConfig,
    game: str,
    preset: str = 'fast',
    max_steps: int = 1200,
    baseline: AnimSpec | None = None,
    perturbed: AnimSpec | None = None,
) -> dict:
    """Compare baseline vs perturbed inference for V2 and V3."""
    if perturbed is None:
        if preset not in ANIM_PRESETS:
            raise ValueError(f'Unknown preset {preset}; choose from {list(ANIM_PRESETS)}')
        perturbed = ANIM_PRESETS[preset]
    if baseline is None:
        baseline = ANIM_PRESETS['baseline']

    tag_base = 'baseline'
    tag_pert = preset if preset in ANIM_PRESETS else 'custom'
    base = gif_compare.infer_both(
        cfg, game, max_steps=max_steps,
        env_overrides=baseline.as_env_overrides(), tag=tag_base)
    pert = gif_compare.infer_both(
        cfg, game, max_steps=max_steps,
        env_overrides=perturbed.as_env_overrides(), tag=tag_pert)

    out_dir = cfg.highlights_dir / 'inference' / 'anim'
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_path = out_dir / f'atari_{game}_{tag_pert}_v2v3.gif'
    if base.get('ok') and pert.get('ok'):
        fl = gif_compare.load_gif_frames(Path(base['compare_gif']))
        fr = gif_compare.load_gif_frames(Path(pert['compare_gif']))
        stacked = fl + fr
        gif_compare.save_compare_gif(stacked, grid_path)

    return {
        'ok': base.get('ok') and pert.get('ok'),
        'game': game,
        'preset': tag_pert,
        'baseline': base,
        'perturbed': pert,
        'baseline_spec': asdict(baseline),
        'perturbed_spec': asdict(perturbed),
        'grid_gif': str(grid_path) if grid_path.exists() else '',
        'score_delta_v2': _delta(base, pert, 'v2'),
        'score_delta_v3': _delta(base, pert, 'v3'),
    }


def run_all_anim(cfg: PathConfig, games: tuple[str, ...], max_steps: int = 1200) -> dict:
    results = {}
    for game in games:
        results[game] = {}
        for preset in ('fast', 'sluggish'):
            results[game][preset] = run_anim_compare(cfg, game, preset=preset, max_steps=max_steps)
    return results


def _delta(base: dict, pert: dict, key: str) -> float:
    try:
        b = base.get(key, {})
        p = pert.get(key, {})
        if isinstance(b, dict) and isinstance(p, dict):
            return round(float(p.get('score', 0)) - float(b.get('score', 0)), 3)
    except (KeyError, TypeError, ValueError):
        pass
    return 0.0

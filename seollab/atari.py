"""Atari: DreamerV3 training/inference + compare helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .paths import PathConfig

COMPARE_GAMES = ('pong', 'breakout', 'boxing')


def train_v3_timed(
    cfg: PathConfig,
    game: str,
    minutes: int = 300,
    steps: int | None = None,
    smoke: bool = False,
    mem_fraction: float = 0.25,
    resume: bool = True,
) -> Path:
    """Train DreamerV3 on Atari (atari_compare preset) with wall-clock timeout."""
    cfg.apply_env(mem_fraction=mem_fraction)
    logdir = cfg.atari_logdir(game, 'v3')
    ckpt = logdir / 'ckpt'
    if ckpt.exists() and any(ckpt.iterdir()) and not smoke and not resume:
        print(f'V3 ckpt exists (skip): {logdir}')
        return logdir
    if ckpt.exists() and resume and not smoke:
        print(f'V3 resume from: {logdir}')

    step_cap = steps or (500 if smoke else int(1e9))
    timeout = 300 if smoke else minutes * 60
    platforms = ('cuda', 'cpu') if not smoke else ('cuda',)
    last_err: Exception | None = None
    for platform in platforms:
        cmd = [
            sys.executable, str(cfg.dreamerv3_root / 'dreamerv3' / 'main.py'),
            '--logdir', str(logdir),
            '--configs', 'atari', 'atari_compare',
            '--task', f'atari_{game}',
            '--script', 'train',
            '--run.steps', str(step_cap),
            '--run.envs', '1',
            '--run.save_every', '300',
            '--run.log_every', '60',
            '--batch_size', '2' if smoke else '4',
            '--jax.platform', platform,
            '--jax.prealloc', 'False',
        ]
        print('V3 train:', ' '.join(cmd), f'(timeout={timeout}s)')
        env = _train_env(cfg)
        try:
            subprocess.run(
                cmd, cwd=str(cfg.dreamerv3_root), check=True,
                timeout=timeout, env=env,
            )
            return logdir
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            last_err = e
            if platform == 'cuda' and not smoke:
                print(f'V3 GPU train ended ({e}); not falling back to CPU after timed run')
            raise
    if last_err:
        raise last_err
    return logdir


def train_v3_debug(cfg: PathConfig, game: str = 'pong', steps: int = 5000) -> Path:
    """Short DreamerV3 Atari training (debug preset)."""
    return train_v3_timed(cfg, game, minutes=30, steps=steps, smoke=True)


def train_v3_to_env_steps(
    cfg: PathConfig,
    game: str,
    env_steps: int,
    logdir: Path | None = None,
    mem_fraction: float = 0.25,
) -> Path:
    """Train DreamerV3 until env-step counter reaches `env_steps` (for aligned compare)."""
    from .atari_align import aligned_logdir, select_v3_ckpt_dir

    cfg.apply_env(mem_fraction=mem_fraction)
    logdir = Path(logdir) if logdir else aligned_logdir(cfg, game)
    logdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir, ckpt_step, msg = select_v3_ckpt_dir(logdir, env_steps)
    if ckpt_dir and ckpt_step is not None and ckpt_step >= env_steps * 0.95:
        print(f'V3 aligned ckpt exists ({ckpt_step} steps): {ckpt_dir}')
        return logdir

    timeout = max(600, int(env_steps / 50))
    cmd = [
        sys.executable, str(cfg.dreamerv3_root / 'dreamerv3' / 'main.py'),
        '--logdir', str(logdir),
        '--configs', 'atari', 'atari_compare',
        '--task', f'atari_{game}',
        '--script', 'train',
        '--run.steps', str(env_steps),
        '--run.envs', '1',
        '--run.save_every', '60',
        '--run.log_every', '30',
        '--batch_size', '4',
        '--jax.platform', 'cuda',
        '--jax.prealloc', 'False',
    ]
    print('V3 aligned train:', ' '.join(cmd), f'(timeout={timeout}s)')
    subprocess.run(
        cmd, cwd=str(cfg.dreamerv3_root), check=True,
        timeout=timeout, env=_train_env(cfg),
    )
    return logdir


def infer_v3(
    cfg: PathConfig,
    game: str,
    max_steps: int = 1500,
    env_overrides: dict | None = None,
    out_path: Path | None = None,
    version: str = 'v3',
    logdir: Path | None = None,
    ckpt_dir: Path | None = None,
) -> dict:
    """DreamerV3 Atari inference from checkpoint."""
    import imageio
    import numpy as np
    import ruamel.yaml as yaml
    import elements

    cfg.apply_env(mem_fraction=0.25)
    sys.path.insert(0, str(cfg.dreamerv3_root))
    from dreamerv3.main import make_env
    from dreamerv3.agent import Agent

    logdir = Path(logdir) if logdir else cfg.atari_logdir(game, version)
    folder = cfg.dreamerv3_root / 'dreamerv3'
    configs = yaml.YAML(typ='safe').load((folder / 'configs.yaml').read_text())
    config = elements.Config(configs['defaults'])
    if (logdir / 'config.yaml').exists():
        config = config.update(yaml.YAML(typ='safe').load((logdir / 'config.yaml').read_text()))
    else:
        config = config.update(configs['atari'], configs['atari_compare'], task=f'atari_{game}', logdir=str(logdir))

    overrides = env_overrides or {}
    env_kwargs = {}
    for key in ('repeat', 'noops'):
        if key in overrides:
            env_kwargs[key] = overrides[key]
    if 'sticky' in overrides:
        val = overrides['sticky']
        env_kwargs['sticky'] = val if isinstance(val, bool) else float(val)

    env = make_env(config, 0, **env_kwargs)
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    agent = Agent(
        obs_space, act_space,
        elements.Config(
            **config.agent, logdir=config.logdir, seed=config.seed, jax=config.jax,
            batch_size=config.batch_size, batch_length=config.batch_length,
            replay_context=config.replay_context, report_length=config.report_length,
            replica=config.replica, replicas=config.replicas,
        ),
    )
    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.agent = agent
    if ckpt_dir:
        cp.load(Path(ckpt_dir))
    else:
        cp.load()

    frames, actions = [], []
    carry = agent.init_policy(1)
    act = {'reset': True}
    if 'action' in env.act_space:
        act['action'] = env.act_space['action'].sample()
    obs = env.step(act)
    total = 0.0

    for step in range(max_steps):
        if 'image' in obs and step % 3 == 0:
            img = np.asarray(obs['image'])
            if img.ndim == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            frames.append(img if img.dtype == np.uint8 else np.clip(img * 255, 0, 255).astype(np.uint8))
        policy_obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
        carry, act, _ = agent.policy(
            carry,
            {k: (np.asarray(v)[None] if np.asarray(v).ndim else np.array([v])) for k, v in policy_obs.items()},
            mode='eval',
        )
        step_act = {k: (np.int32(v[0]) if k == 'action' else v[0]) for k, v in act.items()}
        step_act['reset'] = False
        obs = env.step(step_act)
        actions.append(int(step_act['action']))
        total += float(np.asarray(obs.get('reward', 0)))
        if bool(np.asarray(obs.get('is_last', False))):
            act = {'reset': True}
            if 'action' in env.act_space:
                act['action'] = env.act_space['action'].sample()
            obs = env.step(act)
            carry = agent.init_policy(1)

    env.close()
    out = out_path or (cfg.highlights_dir / 'inference' / f'atari_{game}_v3_rollout.gif')
    out.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(str(out), frames, duration=125)
    ent = _action_entropy(actions)
    return {
        'ok': True, 'game': game, 'score': round(total, 3),
        'gif': str(out), 'steps': len(actions), 'action_entropy': ent,
    }


def infer_v3_debug(cfg: PathConfig, game: str = 'pong', max_steps: int = 2000) -> dict:
    return infer_v3(cfg, game, max_steps=max_steps)


def load_v3_scores(logdir: Path) -> list[dict]:
    path = logdir / 'scores.jsonl'
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().strip().splitlines() if l.strip()]


def _action_entropy(actions: list[int]) -> float:
    import numpy as np
    if not actions:
        return 0.0
    counts = np.bincount(np.asarray(actions, dtype=np.int64))
    p = counts[counts > 0] / len(actions)
    return float(-(p * np.log(p + 1e-12)).sum())


def _train_env(cfg: PathConfig) -> dict:
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{cfg.dreamerv3_root}:{cfg.workspace}:{env.get('PYTHONPATH', '')}"
    return env

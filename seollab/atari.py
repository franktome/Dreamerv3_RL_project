"""Atari: official scores comparison + optional DreamerV3 debug training/inference."""

import subprocess
import sys
from pathlib import Path

from .paths import PathConfig


def train_v3_debug(cfg: PathConfig, game: str = 'pong', steps: int = 5000) -> Path:
    """Short DreamerV3 Atari training (debug). DreamerV2 needs TF — use scores compare instead."""
    cfg.apply_env()
    logdir = cfg.dreamerv3_root / 'logdir' / f'atari_{game}'
    ckpt = logdir / 'ckpt'
    if ckpt.exists() and any(ckpt.iterdir()):
        print(f'Atari ckpt exists: {logdir}')
        return logdir
    cmd = [
        sys.executable, str(cfg.dreamerv3_root / 'dreamerv3' / 'main.py'),
        '--logdir', str(logdir),
        '--configs', 'atari', 'debug',
        '--task', f'atari_{game}',
        '--script', 'train',
        '--run.steps', str(steps),
        '--run.envs', '2',
        '--batch_size', '4',
        '--jax.platform', 'cuda',
        '--jax.prealloc', 'False',
    ]
    print('Atari train:', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(cfg.dreamerv3_root), check=True)
    return logdir


def infer_v3_debug(cfg: PathConfig, game: str = 'pong', max_steps: int = 2000) -> dict:
    """DreamerV3 Atari inference from debug checkpoint."""
    import imageio
    import numpy as np
    import ruamel.yaml as yaml
    import elements

    cfg.apply_env()
    sys.path.insert(0, str(cfg.dreamerv3_root))
    from dreamerv3.main import make_env
    from dreamerv3.agent import Agent

    logdir = cfg.dreamerv3_root / 'logdir' / f'atari_{game}'
    folder = cfg.dreamerv3_root / 'dreamerv3'
    configs = yaml.YAML(typ='safe').load((folder / 'configs.yaml').read_text())
    config = elements.Config(configs['defaults'])
    if (logdir / 'config.yaml').exists():
        config = config.update(yaml.YAML(typ='safe').load((logdir / 'config.yaml').read_text()))
    else:
        config = config.update(configs['atari'], configs['debug'], task=f'atari_{game}', logdir=str(logdir))

    env = make_env(config, 0)
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
    cp.load()

    frames, carry = [], agent.init_policy(1)
    act = {'reset': True}
    if 'action' in env.act_space:
        act['action'] = env.act_space['action'].sample()
    obs = env.step(act)
    total = 0.0
    for step in range(max_steps):
        if 'image' in obs and step % 3 == 0:
            img = np.asarray(obs['image'])
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            frames.append(img if img.dtype == np.uint8 else np.clip(img * 255, 0, 255).astype(np.uint8))
        policy_obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
        carry, act, _ = agent.policy(carry, {k: (np.asarray(v)[None] if np.asarray(v).ndim else np.array([v])) for k, v in policy_obs.items()}, mode='eval')
        step_act = {k: (np.int32(v[0]) if k == 'action' else v[0]) for k, v in act.items()}
        step_act['reset'] = False
        obs = env.step(step_act)
        total += float(np.asarray(obs.get('reward', 0)))
        if bool(np.asarray(obs.get('is_last', False))):
            break
    env.close()
    out = cfg.workspace / f'atari_{game}_inference.gif'
    if frames:
        imageio.mimsave(str(out), frames, duration=125)
    return {'ok': True, 'game': game, 'score': total, 'gif': str(out)}

"""Minecraft diamond: deps check, train, inference."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .paths import PathConfig

MILESTONES = [
    'log', 'planks', 'stick', 'crafting_table', 'wooden_pickaxe',
    'cobblestone', 'stone_pickaxe', 'iron_ore', 'furnace',
    'iron_ingot', 'iron_pickaxe', 'diamond',
]


def check_deps() -> tuple[bool, list[str]]:
    issues = []
    try:
        import embodied.envs.minecraft  # noqa: F401
    except Exception as exc:
        issues.append(f'embodied.envs.minecraft: {exc}')
    if shutil.which('java') is None:
        issues.append('Java not found (MineRL requires Java 8+)')
    return len(issues) == 0, issues


def train(
    cfg: PathConfig,
    mode: str = 'debug',
    skip_if_ckpt: bool = True,
    logdir: Path | None = None,
    steps: int | None = None,
    envs: int = 4,
    batch_size: int = 8,
) -> bool:
    """Train DreamerV3 on minecraft_diamond. Returns True if training ran or ckpt exists."""
    from .env_setup import ensure_xvfb

    logdir = Path(logdir or cfg.minecraft_logdir)
    ckpt = logdir / 'ckpt'
    if skip_if_ckpt and ckpt.exists() and any(ckpt.iterdir()):
        print(f'Checkpoint exists: {ckpt}')
        return True

    ensure_xvfb(cfg.display)
    cfg.apply_env()
    logdir.mkdir(parents=True, exist_ok=True)

    # 'full' is a train-mode alias only — never pass it as a config preset name.
    if mode in ('full', 'size200m'):
        presets = ['minecraft', 'size200m']
    elif mode == 'debug':
        presets = ['minecraft', 'debug']
    else:
        presets = ['minecraft', mode]
    assert 'full' not in presets, presets

    cmd = [
        sys.executable, str(cfg.dreamerv3_root / 'dreamerv3' / 'main.py'),
        '--logdir', str(logdir),
        '--configs', *presets,
        '--task', 'minecraft_diamond',
        '--script', 'train',
        '--run.envs', str(envs),
        '--batch_size', str(batch_size),
        '--jax.platform', 'cuda',
        '--jax.prealloc', 'False',
    ]
    if steps is not None:
        cmd += ['--run.steps', str(steps)]
    elif mode == 'debug':
        cmd += ['--run.steps', '5000', '--run.save_every', '60']

    print('Train:', ' '.join(cmd))
    subprocess.run(cmd, cwd=str(cfg.dreamerv3_root), check=True)
    return True


def infer(
    cfg: PathConfig,
    logdir: Path | None = None,
    gif_path: Path | None = None,
    max_steps: int = 500,
    train_mode: str = 'debug',
) -> dict:
    """Run policy inference and save GIF."""
    import imageio
    import ruamel.yaml as yaml
    import elements

    from .env_setup import ensure_xvfb

    logdir = Path(logdir or cfg.minecraft_logdir)
    gif_path = Path(gif_path or cfg.workspace / 'minecraft_diamond_inference.gif')
    ckpt = logdir / 'ckpt'
    if not ckpt.exists() or not any(ckpt.iterdir()):
        return {'ok': False, 'error': f'No checkpoint at {ckpt}'}

    sys.path.insert(0, str(cfg.dreamerv3_root))
    ready, issues = check_deps()
    if not ready:
        return {'ok': False, 'error': '; '.join(issues)}

    ensure_xvfb(cfg.display)
    cfg.apply_env()
    sys.path.insert(0, str(cfg.dreamerv3_root))

    from dreamerv3.main import make_env
    from dreamerv3.agent import Agent

    folder = cfg.dreamerv3_root / 'dreamerv3'
    configs = yaml.YAML(typ='safe').load((folder / 'configs.yaml').read_text())
    config = elements.Config(configs['defaults'])
    saved = logdir / 'config.yaml'
    if saved.exists():
        config = config.update(yaml.YAML(typ='safe').load(saved.read_text()))
    else:
        preset = 'size200m' if train_mode == 'full' else train_mode
        for name in ['minecraft', preset]:
            config = config.update(configs[name])
        config = config.update(logdir=str(logdir), task='minecraft_diamond')

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
    cp = elements.Checkpoint(ckpt)
    cp.agent = agent
    cp.load()

    def batch_obs(obs):
        out = {}
        for key, val in obs.items():
            arr = np.asarray(val)
            out[key] = np.array([arr]) if arr.ndim == 0 else arr[None, ...]
        return out

    def unbatch_act(act):
        out = {}
        for k, v in act.items():
            out[k] = np.int32(v[0]) if k == 'action' else v[0]
        return out

    frames, milestone_frames = [], []
    reached = []
    carry = agent.init_policy(1)
    act = {'reset': True}
    if 'action' in env.act_space:
        act['action'] = env.act_space['action'].sample()
    obs = env.step(act)
    total_reward, last_score = 0.0, 0

    def _capture_frame():
        if 'image' not in obs:
            return None
        img = np.asarray(obs['image'])
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    for step in range(max_steps):
        img = _capture_frame()
        if img is not None and step % 3 == 0:
            frames.append(img)
        policy_obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
        carry, act, _ = agent.policy(carry, batch_obs(policy_obs), mode='eval')
        step_act = unbatch_act(act)
        step_act['reset'] = False
        obs = env.step(step_act)
        total_reward += float(np.asarray(obs.get('reward', 0)))
        score = int(round(total_reward))
        if score > last_score:
            for idx in range(last_score, min(score, len(MILESTONES))):
                name = MILESTONES[idx]
                if name not in reached:
                    reached.append(name)
                    snap = _capture_frame()
                    if snap is not None:
                        milestone_frames.append((name, snap))
            last_score = score
        if bool(np.asarray(obs.get('is_last', False))):
            act = {'reset': True}
            if 'action' in env.act_space:
                act['action'] = env.act_space['action'].sample()
            obs = env.step(act)

    env.close()
    out_dir = gif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if frames:
        imageio.mimsave(str(gif_path), frames, duration=125)
    strip_path = out_dir / 'minecraft_milestone_strip.gif'
    if milestone_frames:
        from PIL import Image, ImageDraw
        tiles = []
        for name, img in milestone_frames:
            tile = Image.fromarray(img).resize((128, 128))
            draw = ImageDraw.Draw(tile)
            draw.rectangle([0, 0, 127, 14], fill=(0, 0, 0))
            draw.text((4, 1), name[:14], fill=(255, 255, 255))
            tiles.append(np.asarray(tile))
        strip = np.concatenate(tiles, axis=1)
        imageio.mimsave(str(strip_path), [strip], duration=500)
    result = {
        'ok': True,
        'gif': str(gif_path),
        'strip_gif': str(strip_path) if milestone_frames else '',
        'steps': step + 1,
        'reward': round(total_reward, 3),
        'max_milestone': MILESTONES[min(last_score, len(MILESTONES) - 1)] if last_score >= 0 else 'none',
        'milestones_reached': reached,
    }
    return result

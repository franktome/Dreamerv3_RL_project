"""Minecraft diamond: deps check, train, inference."""

import json
import os
import re
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


def _load_agent(cfg: PathConfig, logdir: Path, train_mode: str = 'full'):
    import ruamel.yaml as yaml
    import elements
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
    cp = elements.Checkpoint(logdir / 'ckpt')
    cp.agent = agent
    cp.load()
    return agent, env, config, make_env


def _checkpoint_info(logdir: Path) -> dict:
    logdir = Path(logdir)
    latest = logdir / 'ckpt' / 'latest'
    ckpt_id = latest.read_text().strip() if latest.exists() else ''
    ckpt_dir = logdir / 'ckpt' / ckpt_id if ckpt_id else None
    step = 0
    metrics = logdir / 'metrics.jsonl'
    if metrics.exists():
        step = json.loads(metrics.read_text().strip().splitlines()[-1]).get('step', 0)
    return {
        'checkpoint_id': ckpt_id,
        'checkpoint_dir': str(ckpt_dir) if ckpt_dir and ckpt_dir.exists() else '',
        'training_step': int(step),
    }


def _labeled_tile(name: str, img: np.ndarray, size: int = 128):
    from PIL import Image, ImageDraw

    tile = Image.fromarray(img).resize((size, size))
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, size - 1, 16], fill=(0, 0, 0))
    draw.text((4, 2), name[:18], fill=(255, 255, 255))
    return tile


def _save_milestone_strip(milestone_frames: list[tuple[str, np.ndarray]], strip_path: Path) -> str:
    import imageio

    if not milestone_frames:
        return ''
    tiles = [np.asarray(_labeled_tile(name, img)) for name, img in milestone_frames]
    strip = np.concatenate(tiles, axis=1)
    imageio.mimsave(str(strip_path), [strip], duration=500)
    return str(strip_path)


def _save_milestone_assets(
    milestone_frames: list[tuple[str, np.ndarray]],
    base_path: Path,
    *,
    gallery_cols: int = 4,
    tile_size: int = 160,
) -> dict:
    """Per-milestone PNGs, horizontal strip GIF, and labeled grid PNG."""
    import imageio
    from PIL import Image

    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    stem = base_path.stem
    parent = base_path.parent

    strip_path = parent / f'{stem}_strip.gif'
    gallery_path = parent / f'{stem}_gallery.png'
    milestones_dir = parent / f'{stem}_milestones'
    milestones_dir.mkdir(parents=True, exist_ok=True)

    milestone_pngs: dict[str, str] = {}
    for name, img in milestone_frames:
        png = milestones_dir / f'{name}.png'
        _labeled_tile(name, img, size=tile_size).save(png)
        milestone_pngs[name] = str(png)

    strip = _save_milestone_strip(milestone_frames, strip_path)

    gallery = ''
    if milestone_frames:
        tiles = [_labeled_tile(name, img, size=tile_size) for name, img in milestone_frames]
        cols = min(gallery_cols, len(tiles))
        rows = (len(tiles) + cols - 1) // cols
        label_h = 0
        canvas = Image.new(
            'RGB',
            (cols * tile_size, rows * (tile_size + label_h)),
            (24, 24, 24),
        )
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            canvas.paste(tile, (c * tile_size, r * (tile_size + label_h)))
        canvas.save(gallery_path)
        gallery = str(gallery_path)

    return {
        'strip_gif': strip,
        'gallery_png': gallery,
        'milestones_dir': str(milestones_dir),
        'milestone_pngs': milestone_pngs,
    }


def _diverse_top_k(ranked: list[dict], k: int = 3) -> list[dict]:
    """Pick top rollouts with distinct milestone chains when possible."""
    picked: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for row in ranked:
        sig = tuple(row.get('milestones_reached') or [])
        if not picked or sig not in seen:
            picked.append(row)
            seen.add(sig)
        if len(picked) >= k:
            break
    for row in ranked:
        if row in picked:
            continue
        picked.append(row)
        if len(picked) >= k:
            break
    for i, row in enumerate(picked[:k], 1):
        row['rank'] = i
    return picked[:k]


def _rollout_episode(agent, env, max_steps: int, frame_skip: int = 2) -> dict:
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
        if img is not None and step % frame_skip == 0:
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
            break

    return {
        'frames': frames,
        'milestone_frames': milestone_frames,
        'steps': step + 1,
        'reward': round(total_reward, 3),
        'max_milestone': MILESTONES[min(last_score, len(MILESTONES) - 1)] if last_score >= 0 else 'none',
        'milestones_reached': reached,
        'milestone_count': len(reached),
    }


def infer(
    cfg: PathConfig,
    logdir: Path | None = None,
    gif_path: Path | None = None,
    max_steps: int = 500,
    train_mode: str = 'debug',
    seed: int | None = None,
) -> dict:
    """Run policy inference and save GIF."""
    import imageio
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

    agent, env, config, make_env = _load_agent(cfg, logdir, train_mode=train_mode)
    if seed is not None:
        env.close()
        config = config.update(seed=int(seed))
        env = make_env(config, 0)

    rollout = _rollout_episode(agent, env, max_steps=max_steps)
    env.close()

    out_dir = gif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if rollout['frames']:
        imageio.mimsave(str(gif_path), rollout['frames'], duration=125)
    assets = _save_milestone_assets(
        rollout['milestone_frames'],
        gif_path.with_suffix(''),
    )
    ckpt = _checkpoint_info(logdir)
    return {
        'ok': True,
        'gif': str(gif_path),
        'strip_gif': assets.get('strip_gif', ''),
        'gallery_png': assets.get('gallery_png', ''),
        'milestone_pngs': assets.get('milestone_pngs', {}),
        'milestones_dir': assets.get('milestones_dir', ''),
        'seed': seed,
        **ckpt,
        **{k: rollout[k] for k in ('steps', 'reward', 'max_milestone', 'milestones_reached', 'milestone_count')},
    }


def infer_multi_rollouts(
    cfg: PathConfig,
    logdir: Path | None = None,
    out_dir: Path | None = None,
    n_rollouts: int = 6,
    max_steps: int = 3600,
    seeds: list[int] | None = None,
    train_mode: str = 'full',
    top_k: int = 3,
) -> dict:
    """Run several rollouts, keep the best milestone chains, export gallery GIFs."""
    import imageio
    import json
    from .env_setup import ensure_xvfb

    logdir = Path(logdir or cfg.minecraft_full_logdir)
    out_dir = Path(out_dir or cfg.highlights_dir / 'inference' / 'minecraft_rollouts')
    out_dir.mkdir(parents=True, exist_ok=True)
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

    seeds = seeds or list(range(100, 100 + n_rollouts))
    agent, env, config, make_env = _load_agent(cfg, logdir, train_mode=train_mode)
    env.close()

    rollouts = []
    for i, seed in enumerate(seeds[:n_rollouts]):
        env = make_env(config.update(seed=int(seed)), 0)
        rollout = _rollout_episode(agent, env, max_steps=max_steps)
        env.close()
        gif_path = out_dir / f'rollout_{i:02d}_seed{seed}.gif'
        if rollout['frames']:
            imageio.mimsave(str(gif_path), rollout['frames'], duration=125)
        assets = _save_milestone_assets(rollout['milestone_frames'], gif_path.with_suffix(''))
        rollouts.append({
            'index': i,
            'seed': seed,
            'gif': str(gif_path),
            'strip_gif': assets.get('strip_gif', ''),
            'gallery_png': assets.get('gallery_png', ''),
            'milestone_pngs': assets.get('milestone_pngs', {}),
            'milestones_dir': assets.get('milestones_dir', ''),
            'steps': rollout['steps'],
            'reward': rollout['reward'],
            'max_milestone': rollout['max_milestone'],
            'milestones_reached': rollout['milestones_reached'],
            'milestone_count': rollout['milestone_count'],
        })
        print(
            f'Rollout {i+1}/{n_rollouts} seed={seed}: '
            f'{len(rollout["milestones_reached"])} milestones -> {rollout["max_milestone"]} '
            f'(reward={rollout["reward"]})',
        )

    ranked = sorted(
        rollouts,
        key=lambda r: (r['milestone_count'], r['reward'], r['steps']),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    main_dir = cfg.highlights_dir / 'inference'
    main_dir.mkdir(parents=True, exist_ok=True)
    if best:
        main_gif = main_dir / 'minecraft_diamond_rollout.gif'
        main_strip = main_dir / 'minecraft_milestone_strip.gif'
        main_gallery = main_dir / 'minecraft_milestone_gallery.png'
        shutil.copy2(best['gif'], main_gif)
        if best.get('strip_gif'):
            shutil.copy2(best['strip_gif'], main_strip)
        if best.get('gallery_png'):
            shutil.copy2(best['gallery_png'], main_gallery)

    gallery = _diverse_top_k(ranked, k=top_k)
    ckpt = _checkpoint_info(logdir)

    index = {
        'ok': True,
        'logdir': str(logdir),
        'n_rollouts': len(rollouts),
        'max_steps': max_steps,
        'seeds': seeds[:n_rollouts],
        **ckpt,
        'best': best,
        'top_k': gallery,
        'all_rollouts': ranked,
    }
    index_path = main_dir / 'minecraft_rollouts_index.json'
    index_path.write_text(json.dumps(index, indent=2))
    index['index_path'] = str(index_path)
    index['main_gif'] = str(main_dir / 'minecraft_diamond_rollout.gif')
    index['main_strip'] = str(main_dir / 'minecraft_milestone_strip.gif')
    index['main_gallery'] = str(main_dir / 'minecraft_milestone_gallery.png')
    return index


def scan_rollout_cache(
    cfg: PathConfig,
    logdir: Path | None = None,
    out_dir: Path | None = None,
    top_k: int = 3,
) -> dict:
    """Rebuild rollout index from on-disk GIFs / milestone PNG folders."""
    logdir = Path(logdir or cfg.minecraft_full_logdir)
    out_dir = Path(out_dir or cfg.highlights_dir / 'inference' / 'minecraft_rollouts')
    main_dir = cfg.highlights_dir / 'inference'
    if not out_dir.exists():
        return {'ok': False, 'error': f'No rollout cache at {out_dir}'}

    pat = re.compile(r'^rollout_(\d+)_seed(\d+)\.gif$')
    rollouts = []
    for gif in sorted(out_dir.glob('rollout_*_seed*.gif')):
        m = pat.match(gif.name)
        if not m:
            continue
        i, seed = int(m.group(1)), int(m.group(2))
        stem = gif.with_suffix('')
        strip = Path(str(stem) + '_strip.gif')
        gallery = Path(str(stem) + '_gallery.png')
        milestones_dir = Path(str(stem) + '_milestones')
        pngs = sorted(milestones_dir.glob('*.png')) if milestones_dir.exists() else []
        reached = [p.stem for p in pngs]
        milestone_pngs = {p.stem: str(p) for p in pngs}
        if pngs:
            mcount = len(pngs)
            max_ms = pngs[-1].stem
        elif strip.exists():
            mcount = max(1, strip.stat().st_size // 12_000)
            max_ms = 'see_strip'
            reached = ['(legacy strip — re-run RUN_MC_MULTI=1 for per-event PNGs)']
        else:
            mcount = 0
            max_ms = 'none'
        rollouts.append({
            'index': i,
            'seed': seed,
            'gif': str(gif),
            'strip_gif': str(strip) if strip.exists() else '',
            'gallery_png': str(gallery) if gallery.exists() else '',
            'milestones_dir': str(milestones_dir) if milestones_dir.exists() else '',
            'milestone_pngs': milestone_pngs,
            'milestones_reached': reached,
            'milestone_count': mcount,
            'max_milestone': max_ms,
            'reward': 0.0,
            'steps': 0,
        })

    if not rollouts:
        return {'ok': False, 'error': 'No rollout GIFs found in cache'}

    ranked = sorted(rollouts, key=lambda r: (r['milestone_count'], r['seed']), reverse=True)
    best = ranked[0]
    ckpt = _checkpoint_info(logdir)
    gallery = _diverse_top_k(ranked, k=top_k)

    index = {
        'ok': True,
        'cached': True,
        'logdir': str(logdir),
        'n_rollouts': len(rollouts),
        **ckpt,
        'best': best,
        'top_k': gallery,
        'all_rollouts': ranked,
        'main_gif': str(main_dir / 'minecraft_diamond_rollout.gif'),
        'main_strip': str(main_dir / 'minecraft_milestone_strip.gif'),
        'main_gallery': str(main_dir / 'minecraft_milestone_gallery.png'),
    }
    index_path = main_dir / 'minecraft_rollouts_index.json'
    index_path.write_text(json.dumps(index, indent=2))
    index['index_path'] = str(index_path)
    return index


def resume_train(
    cfg: PathConfig,
    logdir: Path | None = None,
    envs: int = 6,
    batch_size: int = 8,
    mem_fraction: float = 0.42,
) -> subprocess.Popen:
    """Resume full training on the configured GPU (non-blocking subprocess)."""
    from .env_setup import ensure_xvfb

    logdir = Path(logdir or cfg.minecraft_full_logdir)
    ensure_xvfb(cfg.display)
    cfg.apply_env(mem_fraction=mem_fraction)
    logdir.mkdir(parents=True, exist_ok=True)

    log_path = cfg.workspace / 'logs' / 'minecraft_diamond_full' / 'train_gpu1.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    if cfg.conda_env and (cfg.conda_env / 'bin' / 'python').exists():
        py = str(cfg.conda_env / 'bin' / 'python')
    cmd = [
        py, str(cfg.dreamerv3_root / 'dreamerv3' / 'main.py'),
        '--logdir', str(logdir),
        '--configs', 'minecraft', 'size200m',
        '--task', 'minecraft_diamond',
        '--script', 'train',
        '--run.envs', str(envs),
        '--batch_size', str(batch_size),
        '--jax.platform', 'cuda',
        '--jax.prealloc', 'False',
    ]
    print('Resume train:', ' '.join(cmd))
    with log_path.open('a') as logf:
        logf.write('\n--- resume ---\n')
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cfg.dreamerv3_root),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    return proc

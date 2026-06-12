"""DreamerV2 Atari training and inference (TensorFlow, subprocess-isolated)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .paths import PathConfig

COMPARE_GAMES = ('pong', 'breakout', 'boxing')


def _python(cfg: PathConfig) -> str:
    if cfg.conda_env and (cfg.conda_env / 'bin' / 'python').exists():
        return str(cfg.conda_env / 'bin' / 'python')
    return sys.executable


def train_v2_timed(
    cfg: PathConfig,
    game: str,
    minutes: int = 300,
    steps: int | None = None,
    smoke: bool = False,
    mem_fraction: float = 0.25,
    resume: bool = True,
    jit: bool | None = None,
) -> Path:
    """Train DreamerV2 on Atari with wall-clock timeout."""
    logdir = cfg.atari_logdir(game, 'v2')
    ckpt = logdir / 'variables.pkl'
    if ckpt.exists() and not smoke and not resume:
        print(f'V2 ckpt exists (skip): {logdir}')
        return logdir
    if ckpt.exists() and resume and not smoke:
        print(f'V2 resume from: {logdir}')

    logdir.mkdir(parents=True, exist_ok=True)
    train_py = cfg.dreamerv2_root / 'dreamerv2' / 'train.py'
    if not train_py.exists():
        raise FileNotFoundError(f'Missing {train_py}')

    step_cap = steps or (600 if smoke else int(1e9))
    # Smoke runs use the debug architecture; real runs (fresh or resumed)
    # always use the atari_compare architecture so resume stays consistent.
    configs = 'atari debug' if smoke else 'atari atari_compare'
    cmd = [
        _python(cfg), str(train_py),
        '--logdir', str(logdir),
        '--configs', *configs.split(),
        '--task', f'atari_{game}',
        '--prefill', '500' if smoke else '5000',
        '--steps', str(step_cap),
        '--envs', '1',
        '--eval_every', '100' if smoke else '5000',
        '--log_every', '100' if smoke else '500',
        '--precision', '32',
    ]
    if jit is not None:
        cmd += ['--jit', str(jit)]
    env = _subprocess_env(cfg, mem_fraction)
    env['TF_NUM_INTRAOP_THREADS'] = '4'
    env['TF_NUM_INTEROP_THREADS'] = '2'
    env['OMP_NUM_THREADS'] = '4'
    timeout = 300 if smoke else minutes * 60
    print('V2 train:', ' '.join(cmd), f'(timeout={timeout}s)')
    subprocess.run(cmd, cwd=str(cfg.dreamerv2_root), env=env, check=True, timeout=timeout)
    if not ckpt.exists():
        raise RuntimeError(f'V2 training finished without checkpoint: {ckpt}')
    return logdir


def infer_v2_gif(
    cfg: PathConfig,
    game: str,
    max_steps: int = 1500,
    env_overrides: dict | None = None,
    out_path: Path | None = None,
    variables_path: Path | None = None,
) -> dict:
    """Run V2 policy rollout and save GIF (optionally from a sidecar snapshot)."""
    script = Path(__file__).parent / '_v2_infer_worker.py'
    logdir = cfg.atari_logdir(game, 'v2')
    out = out_path or (cfg.highlights_dir / 'inference' / f'atari_{game}_v2_rollout.gif')
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'dreamerv2_root': str(cfg.dreamerv2_root),
        'logdir': str(logdir),
        'game': game,
        'max_steps': max_steps,
        'out_gif': str(out),
        'env_overrides': env_overrides or {},
        'variables': str(variables_path) if variables_path else None,
    }
    env = _subprocess_env(cfg, mem_fraction=0.25)
    proc = subprocess.run(
        [_python(cfg), str(script), json.dumps(payload)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        return {'ok': False, 'error': proc.stderr[-2000:] or proc.stdout[-2000:]}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {'ok': False, 'error': proc.stdout + proc.stderr}


def load_v2_scores(logdir: Path) -> list[dict]:
    path = logdir / 'metrics.jsonl'
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().strip().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _subprocess_env(cfg: PathConfig, mem_fraction: float) -> dict:
    import os
    env = os.environ.copy()
    cfg.apply_env(mem_fraction=mem_fraction)
    for key in (
        'CUDA_VISIBLE_DEVICES', 'DISPLAY', 'TMPDIR', 'TEMP', 'TMP',
        'TF_FORCE_GPU_ALLOW_GROWTH', 'TF_CPP_MIN_LOG_LEVEL',
        'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'JAX_COMPILATION_CACHE_DIR', 'HOME',
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    env['PYTHONPATH'] = f"{cfg.workspace}:{cfg.dreamerv2_root}:{env.get('PYTHONPATH', '')}"
    return env

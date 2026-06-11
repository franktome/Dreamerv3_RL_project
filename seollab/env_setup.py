"""Environment setup: DreamerV3 clone, JAX, Xvfb."""

import os
import subprocess
import sys
import time

from .paths import PathConfig


def clone_dreamerv3(cfg: PathConfig) -> None:
    main_py = cfg.dreamerv3_root / 'dreamerv3' / 'main.py'
    if main_py.exists():
        print(f'DreamerV3 found: {cfg.dreamerv3_root}')
    else:
        cfg.dreamerv3_root.parent.mkdir(parents=True, exist_ok=True)
        print(f'Cloning DreamerV3 -> {cfg.dreamerv3_root}')
        subprocess.run(
            ['git', 'clone', '--depth', '1', 'https://github.com/danijar/dreamerv3.git', str(cfg.dreamerv3_root)],
            check=True,
        )
    if str(cfg.dreamerv3_root) not in sys.path:
        sys.path.insert(0, str(cfg.dreamerv3_root))


def setup_jax(cfg: PathConfig):
    cfg.apply_env()
    if cfg.conda_env:
        nv = cfg.conda_env / 'lib' / 'python3.11' / 'site-packages' / 'nvidia'
        libs = [
            nv / 'cublas' / 'lib', nv / 'cusolver' / 'lib', nv / 'cusparse' / 'lib',
            nv / 'cufft' / 'lib', nv / 'cudnn' / 'lib', cfg.conda_env / 'lib',
        ]
        existing = [str(p) for p in libs if p.exists()]
        if existing:
            prev = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['LD_LIBRARY_PATH'] = ':'.join(existing + ([prev] if prev else []))
    import jax
    print('JAX devices:', jax.devices())
    return jax


def ensure_xvfb(display: str = ':99') -> bool:
    num = display.lstrip(':')
    if subprocess.run(['pgrep', '-f', f'Xvfb :{num}'], capture_output=True).returncode == 0:
        return True
    xvfb = '/mnt/server12_hard0/kiseol/tools/usr/bin/Xvfb'
    if not os.path.isfile(xvfb):
        xvfb = 'Xvfb'
    subprocess.Popen([xvfb, display, '-screen', '0', '1024x768x24', '-ac', '+extension', 'GLX', '+render', '-noreset'])
    time.sleep(2)
    chk = subprocess.run(['/usr/bin/xdpyinfo', '-display', display], capture_output=True)
    ok = chk.returncode == 0
    print(f'Xvfb {display}:', 'OK' if ok else 'FAILED')
    return ok

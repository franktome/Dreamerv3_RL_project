"""Workspace path configuration."""

import os
import pathlib
import shutil
from dataclasses import dataclass, field


def ensure_tmpdir(workspace: pathlib.Path | None = None, min_free_gb: float = 1.0) -> pathlib.Path:
    """Use a writable temp dir when /tmp is full (JAX/XLA writes PTX there)."""
    min_free = int(min_free_gb * 1024**3)

    def _free(path: pathlib.Path) -> int:
        try:
            return shutil.disk_usage(path).free
        except OSError:
            return 0

    current = pathlib.Path(os.environ.get('TMPDIR', '/tmp'))
    if _free(current) >= min_free:
        return current

    candidates = []
    if workspace is not None:
        candidates.append(pathlib.Path(workspace).resolve() / 'tmp')
    alt_home = pathlib.Path('/mnt/server12_hard0/kiseol/tmp')
    if alt_home.parent.exists():
        candidates.append(alt_home)
    candidates.append(pathlib.Path.home() / 'tmp')

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if _free(candidate) >= min_free // 2:
            os.environ['TMPDIR'] = str(candidate)
            os.environ['TEMP'] = str(candidate)
            os.environ['TMP'] = str(candidate)
            print(f'TMPDIR -> {candidate} (was {current}, low disk space)')
            return candidate

    candidate.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(candidate)
    os.environ['TEMP'] = str(candidate)
    os.environ['TMP'] = str(candidate)
    print(f'TMPDIR -> {candidate} (fallback)')
    return candidate


@dataclass
class PathConfig:
    workspace: pathlib.Path
    dreamerv3_root: pathlib.Path
    conda_env: pathlib.Path | None = None
    scores_cache: pathlib.Path = field(init=False)
    report_dir: pathlib.Path = field(init=False)
    highlights_dir: pathlib.Path = field(init=False)
    minecraft_logdir: pathlib.Path = field(init=False)
    minecraft_full_logdir: pathlib.Path = field(init=False)
    gpu: str = '0'
    display: str = ':99'

    def __post_init__(self):
        self.workspace = pathlib.Path(self.workspace).resolve()
        self.dreamerv3_root = pathlib.Path(self.dreamerv3_root).resolve()
        self.scores_cache = self.workspace / 'scores_cache'
        self.report_dir = self.workspace / 'report'
        self.highlights_dir = self.workspace / 'highlights'
        self.minecraft_logdir = self.dreamerv3_root / 'logdir' / 'minecraft_diamond'
        self.minecraft_full_logdir = self.dreamerv3_root / 'logdir' / 'minecraft_diamond_full'
        self.scores_cache.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.highlights_dir.mkdir(parents=True, exist_ok=True)

    @property
    def minecraft_ckpt(self) -> pathlib.Path:
        return self.minecraft_logdir / 'ckpt'

    def apply_env(self) -> None:
        ensure_tmpdir(self.workspace)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['JAX_PLATFORMS'] = 'cuda'
        os.environ['DISPLAY'] = self.display
        if self.conda_env:
            tools = self.conda_env.parent.parent / 'tools' / 'usr' / 'bin'
            if tools.exists():
                os.environ['PATH'] = f'{tools}:{os.environ.get("PATH", "")}'
            jre = self.conda_env / 'bin'
            if jre.exists():
                os.environ['JAVA_HOME'] = str(self.conda_env)
                os.environ['PATH'] = f'{jre}:{os.environ.get("PATH", "")}'


def _resolve_dreamerv3_root(ws: pathlib.Path) -> pathlib.Path:
    """Support repo layout (dreamerv3/ at root) and vendor/dreamerv3 layout."""
    for candidate in (ws / 'vendor' / 'dreamerv3', ws):
        if (candidate / 'dreamerv3' / 'main.py').exists():
            return candidate
    return ws / 'vendor' / 'dreamerv3'


def default_paths(
    workspace: str | pathlib.Path | None = None,
    gpu: str = '0',
) -> PathConfig:
    ws = pathlib.Path(workspace or pathlib.Path.cwd())
    home = pathlib.Path.home()
    d3 = _resolve_dreamerv3_root(ws)
    conda = home / '.conda' / 'envs' / 'dreamerv3'
    alt = pathlib.Path('/mnt/server12_hard0/kiseol/.conda/envs/dreamerv3')
    if not conda.exists() and alt.exists():
        conda = alt
    if not conda.exists():
        conda = None
    return PathConfig(workspace=ws, dreamerv3_root=d3, conda_env=conda, gpu=gpu)

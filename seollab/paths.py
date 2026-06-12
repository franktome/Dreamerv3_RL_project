"""Workspace path configuration."""

import os
import pathlib
import tempfile
from dataclasses import dataclass, field


def ensure_scratch_dirs(workspace: pathlib.Path) -> pathlib.Path:
    """Redirect temp/JAX cache off root /tmp (often full on shared servers)."""
    root = pathlib.Path(workspace).resolve()
    tmpdir = root / 'tmp'
    tmpdir.mkdir(parents=True, exist_ok=True)
    for key in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ[key] = str(tmpdir)
    jax_cache = pathlib.Path(workspace).resolve() / '.jax_cache'
    jax_cache.mkdir(parents=True, exist_ok=True)
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(jax_cache)
    tempfile.tempdir = str(tmpdir)
    return tmpdir


@dataclass
class PathConfig:
    workspace: pathlib.Path
    dreamerv3_root: pathlib.Path
    dreamerv2_root: pathlib.Path
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
        self.dreamerv2_root = pathlib.Path(self.dreamerv2_root).resolve()
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

    def atari_logdir(self, game: str, version: str) -> pathlib.Path:
        return self.dreamerv3_root / 'logdir' / f'atari_{game}_{version}'

    def apply_env(self, mem_fraction: float | None = None) -> None:
        ensure_scratch_dirs(self.workspace)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        os.environ['JAX_PLATFORMS'] = 'cuda'
        os.environ['DISPLAY'] = self.display
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        if mem_fraction is not None:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mem_fraction)
        if self.conda_env:
            tools = self.conda_env.parent.parent / 'tools' / 'usr' / 'bin'
            if tools.exists():
                os.environ['PATH'] = f'{tools}:{os.environ.get("PATH", "")}'
            jre = self.conda_env / 'bin'
            if jre.exists():
                os.environ['JAVA_HOME'] = str(self.conda_env)
                os.environ['PATH'] = f'{jre}:{os.environ.get("PATH", "")}'


def _resolve_dreamerv2_root(ws: pathlib.Path) -> pathlib.Path:
    candidate = ws / 'vendor' / 'dreamerv2'
    if (candidate / 'dreamerv2' / 'train.py').exists():
        return candidate
    return candidate


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
    d2 = _resolve_dreamerv2_root(ws)
    conda = home / '.conda' / 'envs' / 'dreamerv3'
    if not conda.exists():
        conda = None
    return PathConfig(workspace=ws, dreamerv3_root=d3, dreamerv2_root=d2, conda_env=conda, gpu=gpu)

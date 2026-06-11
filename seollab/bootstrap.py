"""Download seollab package from GitHub branch (for ipynb-only environments)."""

import io
import pathlib
import sys
import urllib.request
import zipfile

DEFAULT_REPO = 'franktome/Dreamerv3_RL_project'
DEFAULT_BRANCH = 'kihyun-dreamerv3'
PACKAGE_DIR = 'seollab'


def ensure_package(
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    cache_dir: str | pathlib.Path | None = None,
    force: bool = False,
) -> pathlib.Path:
    """Download branch zip and add seollab/ to sys.path."""
    cache = pathlib.Path(cache_dir or pathlib.Path.home() / '.cache' / 'seollab_pkg')
    cache.mkdir(parents=True, exist_ok=True)
    pkg_root = cache / f'{repo.split("/")[-1]}-{branch}' / PACKAGE_DIR
    init_file = pkg_root / '__init__.py'

    if init_file.exists() and not force:
        _add_to_path(pkg_root.parent)
        print(f'Using cached package: {pkg_root}')
        return pkg_root

    url = f'https://github.com/{repo}/archive/refs/heads/{branch}.zip'
    print(f'Downloading {repo}@{branch} ...')
    with urllib.request.urlopen(url, timeout=180) as resp:
        data = resp.read()

    extract_root = cache / f'{repo.split("/")[-1]}-{branch}'
    if extract_root.exists():
        import shutil
        shutil.rmtree(extract_root)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(cache)

    # GitHub zip root: Dreamerv3_RL_project-dreamerv2-v3/
    candidates = list(cache.glob(f'*-{branch}'))
    if not candidates:
        raise FileNotFoundError(f'Extracted archive missing *-{branch} under {cache}')
    src = candidates[0] / PACKAGE_DIR
    if not (src / '__init__.py').exists():
        raise FileNotFoundError(f'{PACKAGE_DIR}/ not found in {candidates[0]}')

    _add_to_path(candidates[0])
    print(f'Installed package from {url}')
    print(f'Package path: {src}')
    return src


def _add_to_path(root: pathlib.Path) -> None:
    root = str(root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

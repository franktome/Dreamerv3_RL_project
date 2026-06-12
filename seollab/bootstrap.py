"""Download dvbench package from GitHub branch (for ipynb-only environments)."""

import io
import pathlib
import sys
import urllib.request
import zipfile

DEFAULT_REPO = 'franktome/Dreamerv3_RL_project'
DEFAULT_BRANCH = 'dreamerv2-v3'


def ensure_package(
    repo: str = DEFAULT_REPO,
    branch: str = DEFAULT_BRANCH,
    cache_dir: str | pathlib.Path | None = None,
    force: bool = False,
) -> pathlib.Path:
    """Download branch zip and add repo root to sys.path (dvbench + seollab)."""
    cache = pathlib.Path(cache_dir or pathlib.Path.home() / '.cache' / 'dvbench_pkg')
    cache.mkdir(parents=True, exist_ok=True)
    extract_root = cache / f'{repo.split("/")[-1]}-{branch}'
    init_file = extract_root / 'dvbench' / '__init__.py'

    if init_file.exists() and not force:
        _add_to_path(extract_root)
        print(f'Using cached package: {extract_root}')
        return extract_root

    url = f'https://github.com/{repo}/archive/refs/heads/{branch}.zip'
    print(f'Downloading {repo}@{branch} ...')
    with urllib.request.urlopen(url, timeout=180) as resp:
        data = resp.read()

    if extract_root.exists():
        import shutil
        shutil.rmtree(extract_root)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(cache)

    candidates = list(cache.glob(f'*-{branch}'))
    if not candidates:
        raise FileNotFoundError(f'Extracted archive missing *-{branch} under {cache}')
    root = candidates[0]
    if not (root / 'dvbench' / '__init__.py').exists():
        raise FileNotFoundError(f'dvbench/ not found in {root}')

    _add_to_path(root)
    print(f'Installed package from {url}')
    print(f'Package path: {root / "dvbench"}')
    return root


def _add_to_path(root: pathlib.Path) -> None:
    root = str(root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

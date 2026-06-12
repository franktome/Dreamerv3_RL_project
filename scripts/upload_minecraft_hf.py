#!/usr/bin/env python3
"""Upload Minecraft diamond checkpoint to Hugging Face."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

HF_REPO = 'HyunseoYun/dreamerv3-custom-envs'
PATH_IN_REPO = 'checkpoints/minecraft_diamond_full'
PATTERNS = ['scores.jsonl', 'metrics.jsonl', 'config.yaml', 'ckpt/**']


def main() -> None:
    from huggingface_hub import HfApi
    from dvbench.hf_assets import login_if_needed, resolve_minecraft_logdir
    from dvbench.paths import default_paths

    login_if_needed()
    cfg = default_paths(WORKSPACE, gpu='1')
    logdir = resolve_minecraft_logdir(cfg)
    if not (logdir / 'ckpt').exists():
        raise SystemExit(f'No checkpoint at {logdir / "ckpt"}')

    latest = (logdir / 'ckpt' / 'latest').read_text().strip() if (logdir / 'ckpt' / 'latest').exists() else '?'
    print(f'Uploading {logdir} (latest={latest}) -> {HF_REPO}/{PATH_IN_REPO}')

    api = HfApi()
    api.upload_folder(
        folder_path=str(logdir),
        path_in_repo=PATH_IN_REPO,
        repo_id=HF_REPO,
        repo_type='model',
        allow_patterns=PATTERNS,
        commit_message=f'Update minecraft_diamond_full checkpoint ({latest})',
    )
    print('Upload complete.')


if __name__ == '__main__':
    main()

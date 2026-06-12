"""Hugging Face checkpoint helpers for dreamerv3-custom-envs."""

from __future__ import annotations

import os
import pathlib

HF_REPO = 'HyunseoYun/dreamerv3-custom-envs'


def login_if_needed() -> None:
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not token:
        return
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)


def ensure_checkpoints(
    workspace: pathlib.Path | str,
    patterns: list[str] | None = None,
) -> pathlib.Path:
    """Download HF checkpoints into workspace/dreamerv3_checkpoints if missing."""
    from huggingface_hub import snapshot_download

    login_if_needed()
    workspace = pathlib.Path(workspace)
    dest = workspace / 'dreamerv3_checkpoints'
    marker = dest / 'checkpoints' / 'highway_roundabout' / 'latest'
    if marker.exists():
        print('HF checkpoints already present:', dest)
        return dest

    patterns = patterns or ['checkpoints/**']
    print(f'Downloading {HF_REPO} -> {dest}')
    snapshot_download(HF_REPO, local_dir=str(dest), repo_type='model', allow_patterns=patterns)
    print('Download complete.')
    return dest


def upload_minecraft_checkpoint(cfg, repo_id: str = HF_REPO) -> str:
    """Upload local minecraft_diamond_full logdir to Hugging Face."""
    from huggingface_hub import HfApi

    login_if_needed()
    logdir = resolve_minecraft_logdir(cfg)
    if not (logdir / 'ckpt').exists():
        raise FileNotFoundError(f'No checkpoint at {logdir / "ckpt"}')
    latest = (logdir / 'ckpt' / 'latest').read_text().strip() if (logdir / 'ckpt' / 'latest').exists() else 'unknown'
    path_in_repo = 'checkpoints/minecraft_diamond_full'
    HfApi().upload_folder(
        folder_path=str(logdir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type='model',
        allow_patterns=['scores.jsonl', 'metrics.jsonl', 'config.yaml', 'ckpt/**'],
        commit_message=f'Update minecraft_diamond_full ({latest})',
    )
    return f'{repo_id}/{path_in_repo}'


def resolve_minecraft_logdir(cfg) -> pathlib.Path:
    """Prefer vendor/dreamerv3 trained logdir when present."""
    candidates = [
        cfg.minecraft_full_logdir,
        cfg.workspace / 'vendor' / 'dreamerv3' / 'logdir' / 'minecraft_diamond_full',
        cfg.workspace / 'logdir' / 'minecraft_diamond_full',
    ]
    for p in candidates:
        if (p / 'ckpt').exists():
            return p
    return cfg.minecraft_full_logdir

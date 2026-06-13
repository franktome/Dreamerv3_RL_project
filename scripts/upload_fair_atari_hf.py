#!/usr/bin/env python3
"""Upload fair Atari checkpoints to Hugging Face (full replace under checkpoints/atari_*)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

GAMES = ('pong', 'breakout', 'boxing')
HF_REPO = 'HyunseoYun/dreamerv3-custom-envs'

V2_PATTERNS = ['variables.pkl', 'metrics.jsonl', 'config.yaml', 'snapshots/**']
V3_PATTERNS = ['scores.jsonl', 'metrics.jsonl', 'config.yaml', 'ckpt/**']


def _env_steps(cfg, game: str, ver: str) -> int | None:
    from seollab import atari_align

    a = atari_align.align_game(cfg, game)
    p = a.v2 if ver == 'v2' else a.v3
    return p.env_steps or p.ckpt_env_steps


def upload_game_ver(api, cfg, game: str, ver: str, label: str) -> None:
    logdir = cfg.atari_logdir(game, ver)
    path_in_repo = f'checkpoints/atari_{game}_{ver}'
    if ver == 'v2':
        if not (logdir / 'variables.pkl').exists():
            print(f'SKIP {game}_{ver}: no variables.pkl')
            return
        patterns = V2_PATTERNS
    else:
        if not (logdir / 'ckpt').exists():
            print(f'SKIP {game}_{ver}: no ckpt dir')
            return
        patterns = V3_PATTERNS

    steps = _env_steps(cfg, game, ver)
    step_txt = f'{steps:,} env steps' if steps else 'checkpoint refresh'
    commit = f'{label}: {game} {ver} ({step_txt})'

    print(f'Replacing {HF_REPO}/{path_in_repo} ...')
    try:
        api.delete_folder(
            path_in_repo=path_in_repo,
            repo_id=HF_REPO,
            repo_type='model',
            commit_message=f'Remove stale {path_in_repo} before re-upload',
        )
        print('  deleted remote folder')
    except Exception as e:
        print(f'  delete skipped ({e})')

    print(f'Uploading {logdir} -> {path_in_repo}')
    api.upload_folder(
        folder_path=str(logdir),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO,
        repo_type='model',
        allow_patterns=patterns,
        commit_message=commit,
    )
    print('  done')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+', default=list(GAMES))
    parser.add_argument(
        '--label', default='Fair Atari retrain',
        help='Commit message prefix (e.g. "Breakout V2 catchup")',
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi
    from seollab.hf_assets import login_if_needed
    from seollab.paths import default_paths

    login_if_needed()
    cfg = default_paths(WORKSPACE, gpu='0')
    api = HfApi()

    for game in args.games:
        for ver in ('v2', 'v3'):
            upload_game_ver(api, cfg, game, ver, args.label)
    print('All uploads complete.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Upload fair Atari checkpoints to Hugging Face (overwrite checkpoints/atari_*)."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

GAMES = ('pong', 'breakout', 'boxing')
HF_REPO = 'HyunseoYun/dreamerv3-custom-envs'

V2_PATTERNS = ['variables.pkl', 'metrics.jsonl', 'config.yaml']
V3_PATTERNS = ['scores.jsonl', 'metrics.jsonl', 'config.yaml', 'ckpt/**']


def main() -> None:
    from huggingface_hub import HfApi
    from seollab.hf_assets import login_if_needed
    from seollab.paths import default_paths

    login_if_needed()
    cfg = default_paths(WORKSPACE, gpu='0')
    api = HfApi()

    for game in GAMES:
        for ver in ('v2', 'v3'):
            logdir = cfg.atari_logdir(game, ver)
            path_in_repo = f'checkpoints/atari_{game}_{ver}'
            if ver == 'v2':
                if not (logdir / 'variables.pkl').exists():
                    print(f'SKIP {game}_{ver}: no variables.pkl')
                    continue
                patterns = V2_PATTERNS
            else:
                if not (logdir / 'ckpt').exists():
                    print(f'SKIP {game}_{ver}: no ckpt dir')
                    continue
                patterns = V3_PATTERNS
            print(f'Uploading {logdir} -> {HF_REPO}/{path_in_repo}')
            api.upload_folder(
                folder_path=str(logdir),
                path_in_repo=path_in_repo,
                repo_id=HF_REPO,
                repo_type='model',
                allow_patterns=patterns,
                commit_message=f'Fair Atari retrain: {game} {ver} (100k env steps)',
            )
            print('  done')
    print('All uploads complete.')


if __name__ == '__main__':
    main()

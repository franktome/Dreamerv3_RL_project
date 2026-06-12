#!/usr/bin/env python3
"""Ensure Atari jobs finish on schedule: pause Minecraft, heal stale jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SCHEDULE = WORKSPACE / 'scripts' / 'atari_job_schedule.json'
LOG = WORKSPACE / 'logs' / 'atari_train_4gpu' / 'guard.log'
STALE_SEC = 600  # restart if metrics idle >10 min
MEM_FRACTION = 0.42
PYTHON = Path.home() / '.conda' / 'envs' / 'dreamerv3' / 'bin' / 'python'


def log(msg: str) -> None:
    line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(line + '\n')


def pause_minecraft() -> None:
    script = WORKSPACE / 'scripts' / 'pause_minecraft.sh'
    subprocess.run(['bash', str(script)], check=False)


def metrics_age_sec(game: str, version: str) -> float | None:
    m = WORKSPACE / 'vendor' / 'dreamerv3' / 'logdir' / f'atari_{game}_{version}' / 'metrics.jsonl'
    if not m.exists() or not m.read_text().strip():
        return None
    return time.time() - m.stat().st_mtime


def remaining_minutes(deadline_str: str) -> int:
    deadline = datetime.fromisoformat(deadline_str)
    return max(0, int((deadline.timestamp() - time.time()) / 60))


def is_completed(job: dict) -> bool:
    """Job finished if past deadline and no active trainer."""
    if remaining_minutes(job['deadline']) > 0:
        return False
    return not find_train_pid(job['game'], job['version'])


def find_train_pid(game: str, version: str) -> list[str]:
    pat = f'atari_{game}_{version}'
    out = subprocess.run(['pgrep', '-af', pat], capture_output=True, text=True)
    pids = []
    for line in out.stdout.splitlines():
        if 'grep' in line or 'run_atari_guard' in line:
            continue
        if pat in line and ('dreamerv3/main' in line or 'dreamerv2/train' in line):
            pids.append(line.split()[0])
    return pids


def restart_job(gpu: str, version: str, game: str, minutes: int) -> None:
    for pid in find_train_pid(game, version):
        subprocess.run(['kill', pid], check=False)
    time.sleep(3)
    cmd = [
        str(PYTHON), '-u', str(WORKSPACE / 'scripts' / 'run_atari_single_job.py'),
        '--gpu', gpu, '--version', version, '--game', game,
        '--minutes', str(minutes), '--mem-fraction', str(MEM_FRACTION),
    ]
    logdir = WORKSPACE / 'logs' / 'atari_train_4gpu'
    logfile = logdir / f'restart_{game}_{version}.log'
    env = {
        **os.environ,
        'HOME': str(Path.home()),
        'PYTHONPATH': f'{WORKSPACE / "vendor" / "dreamerv3"}:{WORKSPACE}',
        'DISPLAY': ':99',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
    }
    with logfile.open('a') as lf:
        lf.write(f'\n=== restart {game}_{version} {minutes}min {datetime.now()} ===\n')
        lf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(WORKSPACE), env=env,
            stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f'restarted {game}_{version} gpu={gpu} minutes={minutes} pid={proc.pid}')


def launch_sidecars(v3_pids: dict[str, str]) -> None:
    """Start V2 sidecars for pending jobs after V3 (skip if already waiting)."""
    chk = subprocess.run(['pgrep', '-f', 'run_atari_v2_sidecars.sh'], capture_output=True)
    if chk.returncode == 0:
        log('sidecars already running')
        return
    script = WORKSPACE / 'scripts' / 'run_atari_v2_sidecars.sh'
    pong_pid = v3_pids.get('pong', '0')
    breakout_pid = v3_pids.get('breakout', '0')
    if pong_pid == '0' and breakout_pid == '0':
        log('no V3 pids for sidecars')
        return
    subprocess.Popen(
        ['bash', str(script), pong_pid, breakout_pid, str(MEM_FRACTION), '300'],
        cwd=str(WORKSPACE),
        stdout=open(WORKSPACE / 'logs' / 'atari_train_4gpu' / 'sidecars_launch.log', 'a'),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f'sidecars launched (pong_v3={pong_pid}, breakout_v3={breakout_pid})')


def main() -> None:
    import os
    schedule = json.loads(SCHEDULE.read_text())
    pause_minecraft()
    log('Minecraft paused; checking Atari jobs')

    v3_pids: dict[str, str] = {}
    for job in schedule['jobs']:
        if job.get('after'):
            continue
        tag = job['tag']
        game, version = job['game'], job['version']
        age = metrics_age_sec(game, version)
        mins_left = remaining_minutes(job['deadline'])
        pids = find_train_pid(game, version)

        if version == 'v3':
            v3_pids[game] = pids[0] if pids else '0'

        if is_completed(job):
            log(f'{tag}: completed (deadline passed, no process)')
            continue

        if mins_left <= 0:
            log(f'{tag}: deadline passed, not restarting')
            continue

        if age is None:
            log(f'{tag}: no metrics, starting {mins_left} min')
            restart_job(job['gpu'], version, game, mins_left)
            continue

        if age > STALE_SEC:
            log(f'{tag}: stale {age:.0f}s, restarting with {mins_left} min left')
            restart_job(job['gpu'], version, game, mins_left)
        elif not pids:
            log(f'{tag}: idle after deadline window, not restarting')
        else:
            log(f'{tag}: OK (age={age:.0f}s, left={mins_left}min, pid={pids[0]})')

    launch_sidecars(v3_pids)
    log('guard complete')


if __name__ == '__main__':
    main()

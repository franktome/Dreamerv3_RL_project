#!/usr/bin/env python3
"""Run 6 Atari V2/V3 jobs (300 min each) across GPUs 2–5.

GPU 2: pong_v3 → breakout_v2
GPU 3: pong_v2
GPU 4: breakout_v3 → boxing_v2
GPU 5: boxing_v3

Jobs on the same GPU run sequentially (OOM-safe). Different GPUs run in parallel.
V2 uses CPU (TF GPU libs unavailable); V3 uses JAX on the assigned GPU.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

# GPU id -> ordered list of (version, game)
GPU_QUEUES: dict[str, list[tuple[str, str]]] = {
    '2': [('v3', 'pong'), ('v2', 'breakout')],
    '3': [('v2', 'pong')],
    '4': [('v3', 'breakout'), ('v2', 'boxing')],
    '5': [('v3', 'boxing')],
}

DEFAULT_MINUTES = 300
# ~20 GiB of 49 GiB per GPU (sequential jobs, one at a time)
DEFAULT_MEM_FRACTION = 0.42


def _gpu_worker(
    gpu: str,
    jobs: list[tuple[str, str]],
    minutes: int,
    mem_fraction: float,
    workspace: Path,
    log_path: Path,
    python: str,
) -> None:
    import subprocess as sp

    job_script = workspace / 'scripts' / 'run_atari_single_job.py'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        def writeln(msg: str) -> None:
            line = f'[GPU{gpu}] {msg}'
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()

        for i, (version, game) in enumerate(jobs, 1):
            tag = f'{game}_{version}'
            writeln(f'=== job {i}/{len(jobs)}: {tag} start ({minutes} min) ===')
            t0 = time.time()
            try:
                sp.run(
                    [
                        python, str(job_script),
                        '--gpu', gpu, '--version', version, '--game', game,
                        '--minutes', str(minutes),
                        '--mem-fraction', str(mem_fraction),
                    ],
                    cwd=str(workspace), check=True,
                )
                elapsed = (time.time() - t0) / 60
                writeln(f'=== job {i}/{len(jobs)}: {tag} done ({elapsed:.1f} min) ===')
            except Exception:
                writeln(f'=== job {i}/{len(jobs)}: {tag} FAILED ===')
                writeln(traceback.format_exc())
                raise

        writeln('=== all jobs on this GPU finished ===')


def main() -> None:
    parser = argparse.ArgumentParser(description='Parallel Atari training on GPUs 2–5')
    parser.add_argument('--minutes', type=int, default=DEFAULT_MINUTES)
    parser.add_argument('--mem-fraction', type=float, default=DEFAULT_MEM_FRACTION)
    parser.add_argument('--gpu', action='append', dest='gpus', help='Run only this GPU queue (repeatable)')
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    logs_dir = workspace / 'logs' / 'atari_train_4gpu'
    logs_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    conda_py = Path.home() / '.conda' / 'envs' / 'dreamerv3' / 'bin' / 'python'
    if conda_py.exists():
        python = str(conda_py)

    queues = GPU_QUEUES
    if args.gpus:
        queues = {g: GPU_QUEUES[g] for g in args.gpus if g in GPU_QUEUES}

    print(f'Starting {sum(len(v) for v in queues.values())} jobs on GPUs {list(queues)}')
    print(f'  {args.minutes} min/job, mem_fraction={args.mem_fraction}')
    for gpu, jobs in queues.items():
        print(f'  GPU {gpu}:', ' → '.join(f'{g}_{v}' for v, g in jobs))

    ctx = mp.get_context('spawn')
    procs: list[mp.Process] = []
    for gpu, jobs in queues.items():
        log_path = logs_dir / f'gpu{gpu}.log'
        log_path.write_text(f'=== GPU {gpu} worker start {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        p = ctx.Process(
            target=_gpu_worker,
            args=(gpu, jobs, args.minutes, args.mem_fraction, workspace, log_path, python),
            name=f'atari-gpu{gpu}',
        )
        p.start()
        procs.append(p)

    failed = []
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append(p.name)

    if failed:
        print(f'FAILED workers: {failed}', file=sys.stderr)
        sys.exit(1)
    print('All GPU workers finished successfully.')


if __name__ == '__main__':
    main()

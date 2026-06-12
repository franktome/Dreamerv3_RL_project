#!/usr/bin/env python3
"""Estimate cumulative wall-clock training time per game/model."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
GAMES = ('pong', 'breakout', 'boxing')
FAIR_START = datetime(2026, 6, 12, 17, 20, tzinfo=timezone.utc).astimezone()
EXTEND_START = datetime(2026, 6, 12, 21, 40, tzinfo=timezone.utc).astimezone()


def _load_meta(name: str) -> datetime | None:
    p = WORKSPACE / 'logs' / name / 'meta.json'
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    key = 'extend_started' if 'extend' in name else 'fair_started'
    return datetime.fromisoformat(data[key])


def _proc_etime_seconds(pid: int) -> float | None:
    try:
        out = subprocess.check_output(['ps', '-o', 'etime=', '-p', str(pid)], text=True).strip()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    parts = out.split('-')
    if len(parts) == 2:
        days, rest = int(parts[0]), parts[1]
    else:
        days, rest = 0, parts[0]
    h, m, s = (rest.split(':') + ['0', '0'])[:3]
    if ':' not in rest and len(rest.split(':')) == 1:
        m, s = h, m
        h = '0'
    return days * 86400 + int(h) * 3600 + int(m) * 60 + int(float(s))


def _pid_alive(pid: int) -> bool:
    try:
        subprocess.check_call(['kill', '-0', str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def _phase_hours(log_dir: Path, game: str, ver: str, phase_start: datetime, phase_name: str) -> float:
    pid_file = log_dir / f'{game}_{ver}.pid'
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        if _pid_alive(pid):
            etime = _proc_etime_seconds(pid)
            if etime is not None:
                return etime / 3600
    log = log_dir / f'{game}_{ver}.log'
    if log.exists():
        end = datetime.fromtimestamp(log.stat().st_mtime).astimezone()
        return max(0.0, (end - phase_start).total_seconds() / 3600)
    return 0.0


def main() -> Path:
    sys.path.insert(0, str(WORKSPACE))
    from seollab.paths import default_paths

    cfg = default_paths(WORKSPACE, gpu='0')
    fair_start = _load_meta('atari_fair') or FAIR_START
    extend_start = _load_meta('atari_fair_extend') or EXTEND_START
    now = datetime.now().astimezone()

    rows = []
    for game in GAMES:
        for ver in ('v2', 'v3'):
            h_fair = _phase_hours(WORKSPACE / 'logs' / 'atari_fair', game, ver, fair_start, 'fair')
            h_ext = _phase_hours(WORKSPACE / 'logs' / 'atari_fair_extend', game, ver, extend_start, 'extend')
            # If extend job still running, use now as end for extend phase
            pid_file = WORKSPACE / 'logs' / 'atari_fair_extend' / f'{game}_{ver}.pid'
            if pid_file.exists() and _pid_alive(int(pid_file.read_text().strip())):
                h_ext = max(0.0, (now - extend_start).total_seconds() / 3600)
            total = h_fair + h_ext
            rows.append({
                'game': game,
                'model': f'Dreamer{ver.upper()}',
                'version': ver,
                'phase1_fair_hours': round(h_fair, 2),
                'phase2_extend_hours': round(h_ext, 2),
                'wall_clock_hours': round(total, 2),
                'logdir': str(cfg.atari_logdir(game, ver)),
            })

    out = cfg.report_dir / 'atari_compare' / 'WALLCLOCK.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Atari Fair Retrain — Wall-Clock Comparison',
        '',
        f'Generated: {now.strftime("%Y-%m-%d %H:%M:%S %Z")}',
        '',
        'Phase 1: fair 100k run | Phase 2: extend 500k / 8h resume.',
        'Per-job wall-clock (phase1 + phase2). Jobs on GPU0 (V2) and GPU1 (V3) run **in parallel**.',
        '',
        '| Game | Model | Phase1 (h) | Phase2 (h) | **Total (h)** |',
        '|------|-------|----------:|----------:|--------------:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['game']} | {r['model']} | {r['phase1_fair_hours']:.2f} | "
            f"{r['phase2_extend_hours']:.2f} | **{r['wall_clock_hours']:.2f}** |"
        )
    lines += ['', '## V2 vs V3 (total hours per game)', '']
    for game in GAMES:
        v2 = next(x for x in rows if x['game'] == game and x['version'] == 'v2')
        v3 = next(x for x in rows if x['game'] == game and x['version'] == 'v3')
        delta = v3['wall_clock_hours'] - v2['wall_clock_hours']
        tag = 'V3 faster' if delta < -0.05 else ('V2 faster' if delta > 0.05 else '~same')
        lines.append(
            f"- **{game}**: V2 **{v2['wall_clock_hours']:.2f}h**, V3 **{v3['wall_clock_hours']:.2f}h** "
            f"(Δ {delta:+.2f}h, {tag})"
        )
    v2t = sum(r['wall_clock_hours'] for r in rows if r['version'] == 'v2')
    v3t = sum(r['wall_clock_hours'] for r in rows if r['version'] == 'v3')
    lines += [
        '',
        f'**Sum over 3 games**: V2 {v2t:.2f}h, V3 {v3t:.2f}h (Δ {v3t - v2t:+.2f}h)',
        '',
        '> Interpretation: lower hours at similar env steps ⇒ higher sample throughput.',
        '> Cluster wall-clock until all jobs done ≈ max(per-job total), not sum.',
        '',
    ]
    out.write_text('\n'.join(lines), encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(out)
    return out


if __name__ == '__main__':
    main()

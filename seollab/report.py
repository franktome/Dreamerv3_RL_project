"""Generate markdown report from viz outputs."""

from datetime import datetime

from .paths import PathConfig


def write_report(cfg: PathConfig, mc_summary: dict, atari_compare, local_mc: dict | None = None) -> str:
    med_v2 = float(atari_compare['DreamerV2'].median())
    med_v3 = float(atari_compare['DreamerV3'].median())
    v3_wins = float((atari_compare['delta'] > 0).mean() * 100)
    lines = [
        '# DreamerV3 Evaluation Report',
        f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}',
        '',
        '## Minecraft Diamond (official scores)',
        '| Method | Diamond success (%) |',
        '|--------|--------------------:|',
    ]
    for name, pct in mc_summary.items():
        lines.append(f'| {name} | {pct:.1f} |')
    lines += ['', 'DreamerV2: N/A', '', '![Minecraft](minecraft_baselines.png)', '']
    if local_mc and local_mc.get('ok'):
        lines += [
            '### Local run',
            f"- GIF: `{local_mc.get('gif', '')}`",
            f"- Milestones: {local_mc.get('milestones', [])}",
            f"- Diamond: {local_mc.get('diamond', False)}",
            '',
        ]
    lines += [
        '## Atari 57 — DreamerV2 vs DreamerV3',
        f'- Median HNS: V2={med_v2:.3f}, V3={med_v3:.3f}',
        f'- V3 wins: {v3_wins:.1f}%',
        '',
        '![Atari](atari_v2_v3.png)',
    ]
    path = cfg.report_dir / 'REPORT.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print('Report:', path)
    return str(path)

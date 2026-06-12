"""Generate markdown report from viz outputs."""

from datetime import datetime

from .paths import PathConfig


def write_analysis(
    cfg: PathConfig,
    mc_summary: dict,
    atari_compare,
    local_mc: dict | None = None,
    advanced: dict | None = None,
) -> str:
    """English analysis of metrics shown in the notebook."""
    med_v2 = float(atari_compare['DreamerV2'].median())
    med_v3 = float(atari_compare['DreamerV3'].median())
    v3_wins = float((atari_compare['delta'] > 0).mean() * 100)
    delta = atari_compare['delta'].dropna()
    top5 = delta.nlargest(5)
    bottom2 = delta.nsmallest(2)

    lines = [
        '# Results Analysis',
        '',
        '## Atari (official 50M-step scores)',
        '',
        f'- Median HNS rises from **{med_v2:.2f}** (DreamerV2) to **{med_v3:.2f}** (DreamerV3).',
        f'- DreamerV3 exceeds DreamerV2 on **{v3_wins:.0f}%** of the 57 games.',
        f'- Mean HNS delta (V3 − V2): **{delta.mean():.2f}**.',
        '',
        'Largest V3 gains:',
    ]
    for game, val in top5.items():
        lines.append(f'- `{game.replace("atari_", "")}`: +{val:.2f} HNS')
    lines += ['', 'Games where V2 stays ahead or close:']
    for game, val in bottom2.items():
        lines.append(f'- `{game.replace("atari_", "")}`: {val:+.2f} HNS')

    if advanced and advanced.get('games'):
        lines += [
            '',
            '## 10-game subset',
            '',
            'Selected for visualization: eight largest V3 improvements plus two V2-favorable titles.',
            f'- Games: {", ".join(advanced["games"])}',
        ]

    lines += ['', '## Minecraft (official baselines)', '']
    for name, stats in mc_summary.items():
        if isinstance(stats, dict):
            lines.append(
                f'- **{name}**: task success {stats.get("final_task_success_pct", 0):.0f}%, '
                f'mean max milestone {stats.get("mean_max_milestone", 0):.1f}'
            )

    if local_mc and local_mc.get('ok'):
        lines += [
            '',
            '## Local Minecraft rollout',
            '',
            f'- Max episode reward: **{float(local_mc.get("reward", 0)):.2f}**',
            f'- Mean episode score: **{float(local_mc.get("mean_episode_score", 0)):.2f}**',
            f'- Max milestone reached: **{local_mc.get("max_milestone", "?")}**',
        ]
        if local_mc.get('training_episodes'):
            lines.append(f'- Training episodes: **{local_mc["training_episodes"]}**')
        reached = local_mc.get('milestones_reached') or []
        if reached:
            lines.append(f'- Progress chain: {" → ".join(reached)}')
        elif local_mc.get('strip_gif'):
            lines.append('- Milestone strip GIF available (see notebook A2).')

    lines += [
        '',
        '## Takeaways',
        '',
        '- DreamerV3 is the stronger generalist on Atari under the published score protocol.',
        '- On Minecraft, V3 reaches deeper inventory milestones faster than PPO/IMPALA on official runs.',
        '- Local rollout metrics (reward, milestone chain) situate the trained checkpoint against those curves.',
    ]
    path = cfg.report_dir / 'ANALYSIS.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return str(path)


def write_report(
    cfg: PathConfig,
    mc_summary: dict,
    atari_compare,
    local_mc: dict | None = None,
    advanced: dict | None = None,
) -> str:
    med_v2 = float(atari_compare['DreamerV2'].median())
    med_v3 = float(atari_compare['DreamerV3'].median())
    v3_wins = float((atari_compare['delta'] > 0).mean() * 100)
    lines = [
        '# DreamerV3 Evaluation Report',
        f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}',
        '',
        '## Minecraft (official scores)',
        '',
        '| Method | Task success (%) | Mean max milestone |',
        '|--------|-----------------:|-------------------:|',
    ]
    for name, stats in mc_summary.items():
        if isinstance(stats, dict):
            lines.append(
                f'| {name} | {stats.get("final_task_success_pct", 0):.1f} | '
                f'{stats.get("mean_max_milestone", 0):.2f} |'
            )
    lines += [
        '',
        '![Minecraft enhanced](minecraft_baselines_enhanced.png)',
        '',
        '## Atari 57 — DreamerV2 vs DreamerV3',
        f'- Median HNS: V2={med_v2:.3f}, V3={med_v3:.3f}',
        f'- V3 higher HNS on {v3_wins:.1f}% of games',
        '',
        '![Atari](atari_v2_v3.png)',
        '![Atari 10 games](atari_10game_panels.png)',
        '![Atari 10 bars](atari_10game_bars.png)',
        '![Executive summary](executive_summary.png)',
    ]
    if local_mc and local_mc.get('ok'):
        lines += [
            '',
            '### Local Minecraft rollout',
            f"- Max episode reward: {local_mc.get('reward', 0)}",
        ]
        if local_mc.get('mean_episode_score') is not None:
            lines.append(f"- Mean episode score: {local_mc.get('mean_episode_score', 0):.2f}")
        lines.append(f"- Max milestone: {local_mc.get('max_milestone', '?')}")
        if local_mc.get('training_episodes'):
            lines.append(f"- Training episodes logged: {local_mc.get('training_episodes')}")
        reached = local_mc.get('milestones_reached') or []
        if reached:
            lines.append(f"- Milestones: {' → '.join(reached)}")
    if advanced:
        lines += [
            '',
            '## Summary',
            f"- Atari median HNS V2 → V3: {advanced.get('median_hns_v2', 0):.2f} → {advanced.get('median_hns_v3', 0):.2f}",
            f"- 10-game set: {', '.join(advanced.get('games', []))}",
        ]
    path = cfg.report_dir / 'REPORT.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    write_analysis(cfg, mc_summary, atari_compare, local_mc, advanced)
    return str(path)

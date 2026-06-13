"""Interactive explorer for cached Atari perturbation inference (no live GIF regen)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, Markdown, clear_output, display

from .atari_anim import ANIM_PRESETS, AnimSpec
from .paths import PathConfig

PRESETS = ('baseline', 'fast', 'sluggish')


def _index_path(cfg: PathConfig) -> Path:
    return cfg.report_dir / 'atari_compare' / 'anim_index.json'


def _scores_sidecar(compare_gif: Path) -> Path:
    return compare_gif.with_suffix('.scores.json')


def _read_scores_sidecar(path: Path) -> dict | None:
    sidecar = _scores_sidecar(path)
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return None


def nearest_preset(repeat: int, sticky: float) -> str:
    """Map slider values to the closest cached preset."""
    def dist(spec: AnimSpec) -> float:
        return abs(spec.repeat - repeat) + abs(spec.sticky - sticky) * 6

    ranked = sorted(ANIM_PRESETS.items(), key=lambda kv: dist(kv[1]))
    return ranked[0][0]


def build_anim_index(cfg: PathConfig, games: tuple[str, ...]) -> dict:
    """Collect cached GIF paths and optional score sidecars (no inference)."""
    index = {
        'presets': {name: asdict(spec) for name, spec in ANIM_PRESETS.items()},
        'games': {},
    }
    for game in games:
        game_entry = {}
        base_cmp = cfg.highlights_dir / 'inference' / f'atari_{game}_v2v3_compare_baseline.gif'
        if not base_cmp.exists():
            base_cmp = cfg.highlights_dir / 'inference' / f'atari_{game}_v2v3_compare.gif'
        base_scores = _read_scores_sidecar(base_cmp) if base_cmp.exists() else None
        for preset in PRESETS:
            tag = preset
            grid = cfg.highlights_dir / 'inference' / 'anim' / f'atari_{game}_{tag}_v2v3.gif'
            if preset == 'baseline':
                cmp_path = base_cmp
                scores = {'baseline': base_scores} if base_scores else None
            else:
                cmp_path = cfg.highlights_dir / 'inference' / f'atari_{game}_v2v3_compare_{tag}.gif'
                pert_scores = _read_scores_sidecar(cmp_path) if cmp_path.exists() else None
                scores = None
                if base_scores or pert_scores:
                    scores = {'baseline': base_scores, 'perturbed': pert_scores}
                    if base_scores and pert_scores:
                        scores['delta_v2'] = round(
                            (pert_scores.get('v2') or 0) - (base_scores.get('v2') or 0), 3,
                        )
                        scores['delta_v3'] = round(
                            (pert_scores.get('v3') or 0) - (base_scores.get('v3') or 0), 3,
                        )
            game_entry[preset] = {
                'repeat': ANIM_PRESETS[preset].repeat,
                'sticky': ANIM_PRESETS[preset].sticky,
                'grid_gif': str(grid) if grid.exists() else '',
                'baseline_compare_gif': str(cmp_path if preset == 'baseline' else base_cmp) if base_cmp.exists() else '',
                'perturbed_compare_gif': '' if preset == 'baseline' else (str(cmp_path) if cmp_path.exists() else ''),
                'scores': scores,
            }
        index['games'][game] = game_entry
    _index_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    _index_path(cfg).write_text(json.dumps(index, indent=2))
    return index


def _index_paths_stale(cfg: PathConfig, index: dict, games: tuple[str, ...]) -> bool:
    """True when cached JSON points at GIFs outside this workspace clone."""
    for game in games:
        for preset in PRESETS:
            entry = index.get('games', {}).get(game, {}).get(preset, {})
            for key in ('grid_gif', 'baseline_compare_gif', 'perturbed_compare_gif'):
                raw = entry.get(key, '')
                if not raw:
                    continue
                p = Path(raw)
                if not p.is_absolute():
                    p = cfg.workspace / p
                if not p.exists():
                    return True
                try:
                    p.resolve().relative_to(cfg.workspace.resolve())
                except ValueError:
                    return True
    return False


def load_anim_index(cfg: PathConfig, games: tuple[str, ...]) -> dict:
    path = _index_path(cfg)
    if path.exists():
        index = json.loads(path.read_text())
        if not _index_paths_stale(cfg, index, games):
            return index
    return build_anim_index(cfg, games)


def merge_anim_results(cfg: PathConfig, anim_results: dict) -> dict:
    """Update anim_index.json with scores from a fresh run_inference_and_anim() payload."""
    index = load_anim_index(cfg, tuple(anim_results.keys()))
    for game, presets in anim_results.items():
        if game not in index['games']:
            index['games'][game] = {}
        for preset, result in presets.items():
            if preset not in index['games'][game]:
                index['games'][game][preset] = {}
            entry = index['games'][game][preset]
            entry['grid_gif'] = result.get('grid_gif', entry.get('grid_gif', ''))
            base = result.get('baseline', {})
            pert = result.get('perturbed', {})
            entry['scores'] = {
                'baseline': {
                    'v2': base.get('v2', {}).get('score'),
                    'v3': base.get('v3', {}).get('score'),
                    'delta': base.get('score_delta'),
                },
                'perturbed': {
                    'v2': pert.get('v2', {}).get('score'),
                    'v3': pert.get('v3', {}).get('score'),
                    'delta': pert.get('score_delta'),
                },
                'delta_v2': result.get('score_delta_v2'),
                'delta_v3': result.get('score_delta_v3'),
            }
    _index_path(cfg).write_text(json.dumps(index, indent=2))
    return index


def _score_table(game: str, preset: str, entry: dict) -> str:
    scores = entry.get('scores') or {}
    base = scores.get('baseline', {})
    pert = scores.get('perturbed', {})
    if not base and not pert:
        return (
            f'_Scores not cached for **{game} / {preset}**. '
            'Re-run section B1 (`REFRESH_ATARI_GIFS=1`) to populate `anim_index.json`._'
        )
    lines = [
        f'**{game} — {preset}** (`repeat={entry.get("repeat")}`, `sticky={entry.get("sticky")}`)',
        '',
        '| Condition | V2 score | V3 score | V3−V2 |',
        '|-----------|---------:|---------:|------:|',
    ]
    for label, block in [('baseline env', base), ('perturbed env', pert)]:
        v2 = block.get('v2')
        v3 = block.get('v3')
        delta = block.get('delta')
        if v2 is None and v3 is None:
            continue
        lines.append(
            f"| {label} | {v2 if v2 is not None else '—'} "
            f"| {v3 if v3 is not None else '—'} "
            f"| {delta if delta is not None else '—'} |"
        )
    dv2 = scores.get('delta_v2')
    dv3 = scores.get('delta_v3')
    if dv2 is not None or dv3 is not None:
        lines += [
            '',
            f'- Perturbation effect: ΔV2={dv2 if dv2 is not None else "—"}, '
            f'ΔV3={dv3 if dv3 is not None else "—"} (perturbed − baseline rollout return)',
        ]
    return '\n'.join(lines)


def _plot_score_bars(game: str, game_data: dict):
    labels, v2_base, v3_base, v2_pert, v3_pert = [], [], [], [], []
    for preset in PRESETS:
        scores = (game_data.get(preset) or {}).get('scores') or {}
        base = scores.get('baseline', {})
        pert = scores.get('perturbed', {})
        if base.get('v2') is None and pert.get('v2') is None:
            continue
        labels.append(preset)
        v2_base.append(base.get('v2', 0))
        v3_base.append(base.get('v3', 0))
        v2_pert.append(pert.get('v2', 0))
        v3_pert.append(pert.get('v3', 0))
    if not labels:
        display(Markdown('_No score sidecars yet — GIF view still works._'))
        return

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(x - 1.5 * w, v2_base, w, label='V2 baseline', color='#1f77b4')
    ax.bar(x - 0.5 * w, v3_base, w, label='V3 baseline', color='#ff7f0e')
    ax.bar(x + 0.5 * w, v2_pert, w, label='V2 perturbed', color='#1f77b4', alpha=0.45)
    ax.bar(x + 1.5 * w, v3_pert, w, label='V3 perturbed', color='#ff7f0e', alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Rollout return')
    ax.set_title(f'{game} — cached inference scores by preset')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    display(fig)
    plt.close(fig)


def show_perturb_explorer(cfg: PathConfig, games: tuple[str, ...]):
    """ipywidgets UI: pick game/preset, tune repeat/sticky, inspect cached GIFs + scores."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise ImportError('ipywidgets is required for the perturbation explorer') from exc

    index = load_anim_index(cfg, games)

    game_dd = widgets.Dropdown(options=list(games), description='Game', layout=widgets.Layout(width='220px'))
    preset_btns = widgets.ToggleButtons(
        options=list(PRESETS),
        description='Preset',
        style={'button_width': '90px'},
    )
    repeat_sl = widgets.IntSlider(value=4, min=2, max=6, step=1, description='repeat', continuous_update=True)
    sticky_sl = widgets.FloatSlider(
        value=0.25, min=0.0, max=0.75, step=0.05, description='sticky', continuous_update=True,
    )
    view_mode = widgets.RadioButtons(
        options=[('Stacked anim GIF', 'grid'), ('Baseline compare only', 'baseline')],
        description='View',
    )
    nearest_lbl = widgets.HTML()
    out = widgets.Output()

    def _sync_sliders_from_preset(preset: str):
        spec = ANIM_PRESETS[preset]
        repeat_sl.value = spec.repeat
        sticky_sl.value = spec.sticky

    def _update_nearest_label():
        near = nearest_preset(repeat_sl.value, sticky_sl.value)
        spec = ANIM_PRESETS[near]
        nearest_lbl.value = (
            f'<b>Closest cached preset:</b> <code>{near}</code> '
            f'(repeat={spec.repeat}, sticky={spec.sticky}) — showing its cached GIF/scores'
        )

    def _render(*_):
        with out:
            clear_output(wait=True)
            game = game_dd.value
            preset = preset_btns.value
            _update_nearest_label()
            entry = index['games'].get(game, {}).get(preset, {})
            display(nearest_lbl)
            display(Markdown(
                '**What changes:** `repeat` = frame skip (lower → faster physics); '
                '`sticky` = action repeat probability (higher → control lag). '
                'Sliders snap to the nearest cached preset — no live re-inference.'
            ))
            gif = ''
            if view_mode.value == 'grid':
                gif = entry.get('grid_gif', '')
            else:
                gif = entry.get('baseline_compare_gif', '')
            if gif and Path(gif).exists():
                display(Markdown(f'**Viewing:** `{Path(gif).name}`'))
                display(Image(filename=gif))
            else:
                display(Markdown(
                    f'*Missing GIF for {game}/{preset}. Run B1 with `REFRESH_ATARI_GIFS=1`.*'
                ))
            display(Markdown(_score_table(game, preset, entry)))
            _plot_score_bars(game, index['games'].get(game, {}))

    def _on_preset(change):
        if change['name'] == 'value':
            _sync_sliders_from_preset(change['new'])
            _render()

    def _on_slider(change):
        if change['name'] == 'value':
            near = nearest_preset(repeat_sl.value, sticky_sl.value)
            if preset_btns.value != near:
                preset_btns.value = near
            else:
                _render()

    preset_btns.observe(_on_preset, names='value')
    repeat_sl.observe(_on_slider, names='value')
    sticky_sl.observe(_on_slider, names='value')
    game_dd.observe(lambda _: _render(), names='value')
    view_mode.observe(lambda _: _render(), names='value')

    _sync_sliders_from_preset(preset_btns.value)
    controls = widgets.VBox([
        widgets.HBox([game_dd, preset_btns]),
        widgets.HBox([repeat_sl, sticky_sl]),
        view_mode,
        out,
    ])
    display(controls)
    _render()

# seollab — DreamerV2/V3 evaluation helpers

Used by `highway_inference.ipynb` (imported from GitHub branch `kihyun-dreamerv3`).

## Modules

| Module | Role |
|--------|------|
| `bootstrap.py` | Download package from GitHub |
| `paths.py` | Workspace / logdir paths |
| `env_setup.py` | DreamerV3 clone, JAX, Xvfb |
| `minecraft.py` | Minecraft train + inference |
| `atari.py` | Atari debug train/infer + scores |
| `viz.py` | Official score plots |
| `report.py` | Markdown report |

## Publish to GitHub

```bash
# On franktome/Dreamerv3_RL_project, branch kihyun-dreamerv3
git checkout -b kihyun-dreamerv3
cp -r /path/to/seollab ./
git add seollab/
git commit -m "Add seollab evaluation package for notebook import"
git push -u origin kihyun-dreamerv3
```

Author: SeolLab (seolpark731@gmail.com)

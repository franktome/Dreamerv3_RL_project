# seollab — DreamerV2/V3 evaluation helpers

Used by notebooks via **`dvbench`** (thin re-export of this package) on GitHub branch `dreamerv2-v3`.

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
# On franktome/Dreamerv3_RL_project, branch dreamerv2-v3
git checkout -b dreamerv2-v3
cp -r /path/to/seollab ./
git add seollab/
git commit -m "Add seollab evaluation package for notebook import"
git push -u origin dreamerv2-v3
```

Author: SeolLab (seolpark731@gmail.com)

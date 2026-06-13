# Atari V2 vs V3 — Step Alignment

Comparable unit: **environment steps** (`train_total_steps` for V2, `scores step / 4` for V3).
Aligned step = `min(V2, V3)` per game. GIFs should use checkpoints at or before this step.

| Game | V2 steps | V2 ep | V3 steps | V3 ckpt | Aligned | Fair GIF? | Note |
|------|----------|-------|----------|---------|---------|-----------|------|
| pong | 503,224 | 619 | 498,561 | 497390 | **498,561** | ✅ | V2 using sidecar snapshot at 480476 env steps (aligned 498561) | Using main V3 ckpt at 497390 env steps |
| breakout | 498,236 | 960 | 499,883 | 489000 | **498,236** | ✅ | Using main V3 ckpt at 489000 env steps |
| boxing | 502,097 | 281 | 498,924 | 475050 | **498,924** | ✅ | V2 using sidecar snapshot at 498533 env steps (aligned 498924) | Using main V3 ckpt at 475050 env steps |

## Step-by-step

1. Read latest V2 `metrics.jsonl` → `train_total_steps`, `train_total_episodes`.
2. Read V3 `scores.jsonl` (last step ÷ 4) and `ckpt/*/step.pkl`.
3. `aligned_env_steps = min(v2, v3)`.
4. V2: load `variables.pkl` (single file; equals aligned when V2 is the limiter).
5. V3: load newest ckpt with `step <= aligned`; if none, train `atari_{game}_v3_aligned`.
6. Run inference/GIF only when both checkpoints are at or before `aligned`.

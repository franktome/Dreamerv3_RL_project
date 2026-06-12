# Atari Fair Retrain — Wall-Clock Comparison

Generated: 2026-06-13 05:47:44 KST

Phase 1: fair 100k run | Phase 2: extend 500k / 8h resume.
Per-job wall-clock (phase1 + phase2). Jobs on GPU0 (V2) and GPU1 (V3) run **in parallel**.

| Game | Model | Phase1 (h) | Phase2 (h) | **Total (h)** |
|------|-------|----------:|----------:|--------------:|
| pong | DreamerV2 | 1.43 | 3.69 | **5.12** |
| pong | DreamerV3 | 0.39 | 1.44 | **1.83** |
| breakout | DreamerV2 | 0.55 | 8.01 | **8.55** |
| breakout | DreamerV3 | 0.39 | 1.47 | **1.86** |
| boxing | DreamerV2 | 1.52 | 6.38 | **7.90** |
| boxing | DreamerV3 | 0.39 | 1.51 | **1.89** |

## V2 vs V3 (total hours per game)

- **pong**: V2 **5.12h**, V3 **1.83h** (Δ -3.29h, V3 faster)
- **breakout**: V2 **8.55h**, V3 **1.86h** (Δ -6.69h, V3 faster)
- **boxing**: V2 **7.90h**, V3 **1.89h** (Δ -6.01h, V3 faster)

**Sum over 3 games**: V2 21.57h, V3 5.58h (Δ -15.99h)

> Interpretation: lower hours at similar env steps ⇒ higher sample throughput.
> Cluster wall-clock until all jobs done ≈ max(per-job total), not sum.

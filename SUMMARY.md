# clean-FINO SUPP arm

M+ + tss(M-). Same as BASE arm plus transcriptome start site (M-).

## 3-seed sweep (s2026/s2027/s2028)


| Seed | Job | linear | knn | 16shot | segmentation | progression | mutation | survival | robustness | **mean** |
|------|-----|--------|-----|--------|-------------|-------------|----------|----------|------------|----------|
| s2026 | 99357 | 0.7564 | 0.6728 | 0.6133 | 0.2904 | 0.6251 | 0.6101 | 0.5917 | 0.8998 | **0.6325** |
| s2027 | 99358 | 0.7713 | 0.6896 | 0.6401 | 0.3003 | 0.6249 | 0.5929 | 0.6014 | 0.9002 | **0.6401** |
| s2028 | 99359 | 0.7633 | 0.6730 | 0.6266 | 0.3063 | 0.6604 | 0.6189 | 0.5818 | 0.9022 | **0.6416** |

## Baseline comparison (dinov2 / base)


| Probe | Baseline |
|-------|----------|
| linear | 0.7555 |
| knn | 0.6839 |
| 16shot | 0.5890 |
| segmentation | 0.3089 |
| progression | 0.6418 |
| mutation | 0.5594 |
| survival | 0.5894 |
| robustness | 0.8531 |
| **mean (Labless score)** | **0.6277** |

## Per-seed vs baseline


### s2026 (job 99357)


| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7564 | 0.7555 | +0.0009 |
| knn | 0.6728 | 0.6839 | -0.0111 |
| 16shot | 0.6133 | 0.5890 | +0.0243 |
| segmentation | 0.2904 | 0.3089 | -0.0185 |
| progression | 0.6251 | 0.6418 | -0.0167 |
| mutation | 0.6101 | 0.5594 | +0.0507 |
| survival | 0.5917 | 0.5894 | +0.0023 |
| robustness | 0.8998 | 0.8531 | +0.0467 |
| **mean (Labless score)** | **0.6325** | **0.6277** | **+0.0048** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/fah4rlbz
Labless: FAILED: interactive GitHub OAuth required (no github_token_file available)

### s2027 (job 99358)


| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7713 | 0.7555 | +0.0158 |
| knn | 0.6896 | 0.6839 | +0.0057 |
| 16shot | 0.6401 | 0.5890 | +0.0511 |
| segmentation | 0.3003 | 0.3089 | -0.0086 |
| progression | 0.6249 | 0.6418 | -0.0169 |
| mutation | 0.5929 | 0.5594 | +0.0335 |
| survival | 0.6014 | 0.5894 | +0.0120 |
| robustness | 0.9002 | 0.8531 | +0.0471 |
| **mean (Labless score)** | **0.6401** | **0.6277** | **+0.0124** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/bgibjgyd
Labless: FAILED: interactive GitHub OAuth required (no github_token_file available)

### s2028 (job 99359)


| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7633 | 0.7555 | +0.0078 |
| knn | 0.6730 | 0.6839 | -0.0109 |
| 16shot | 0.6266 | 0.5890 | +0.0376 |
| segmentation | 0.3063 | 0.3089 | -0.0026 |
| progression | 0.6604 | 0.6418 | +0.0186 |
| mutation | 0.6189 | 0.5594 | +0.0595 |
| survival | 0.5818 | 0.5894 | -0.0076 |
| robustness | 0.9022 | 0.8531 | +0.0491 |
| **mean (Labless score)** | **0.6416** | **0.6277** | **+0.0139** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/z08z0y0u
Labless: FAILED: interactive GitHub OAuth required (no github_token_file available)


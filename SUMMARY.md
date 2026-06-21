# clean-FINO BASE arm

M+ = morphology, expr512, fga, purity, tmb, ploidy, msi_mantis. grad_equalize on. Reads new_fino_meta.json.

## 3-seed sweep (s2026/s2027/s2028)


| Seed | Job | linear | knn | 16shot | segmentation | progression | mutation | survival | robustness | **mean** |
|------|-----|--------|-----|--------|-------------|-------------|----------|----------|------------|----------|
| s2026 | 99350 | 0.7666 | 0.7147 | 0.6569 | 0.2992 | 0.6216 | 0.6066 | 0.5729 | 0.8951 | **0.6417** |
| s2027 | 99351 | 0.7898 | 0.7220 | 0.6504 | 0.2982 | 0.6153 | 0.6116 | 0.6003 | 0.8889 | **0.6471** |
| s2028 | 99352 | 0.7798 | 0.7194 | 0.6621 | 0.2986 | 0.6187 | 0.6089 | 0.5673 | 0.8868 | **0.6427** |

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


### s2026 (job 99350)


| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7666 | 0.7555 | +0.0111 |
| knn | 0.7147 | 0.6839 | +0.0308 |
| 16shot | 0.6569 | 0.5890 | +0.0679 |
| segmentation | 0.2992 | 0.3089 | -0.0097 |
| progression | 0.6216 | 0.6418 | -0.0202 |
| mutation | 0.6066 | 0.5594 | +0.0472 |
| survival | 0.5729 | 0.5894 | -0.0165 |
| robustness | 0.8951 | 0.8531 | +0.0420 |
| **mean (Labless score)** | **0.6417** | **0.6277** | **+0.0140** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/kpcd864e
Labless: FAILED: interactive GitHub OAuth required (no github_token_file available)

### s2027 (job 99351)


| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7898 | 0.7555 | +0.0343 |
| knn | 0.7220 | 0.6839 | +0.0381 |
| 16shot | 0.6504 | 0.5890 | +0.0614 |
| segmentation | 0.2982 | 0.3089 | -0.0107 |
| progression | 0.6153 | 0.6418 | -0.0265 |
| mutation | 0.6116 | 0.5594 | +0.0522 |
| survival | 0.6003 | 0.5894 | +0.0109 |
| robustness | 0.8889 | 0.8531 | +0.0358 |
| **mean (Labless score)** | **0.6471** | **0.6277** | **+0.0194** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/c1ilolo3
Labless: FAILED: interactive GitHub OAuth required (no github_token_file available)

### s2028 (job 99352)


| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7798 | 0.7555 | +0.0243 |
| knn | 0.7194 | 0.6839 | +0.0355 |
| 16shot | 0.6621 | 0.5890 | +0.0731 |
| segmentation | 0.2986 | 0.3089 | -0.0103 |
| progression | 0.6187 | 0.6418 | -0.0231 |
| mutation | 0.6089 | 0.5594 | +0.0495 |
| survival | 0.5673 | 0.5894 | -0.0221 |
| robustness | 0.8868 | 0.8531 | +0.0337 |
| **mean (Labless score)** | **0.6427** | **0.6277** | **+0.0150** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/8e7kzjct
Labless: FAILED: interactive GitHub OAuth required (no github_token_file available)


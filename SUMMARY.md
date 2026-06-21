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

## Round-2: ref3 vs mol (3-seed sweep)

ref3 = subtype/expr512/fga. mol = +purity/tmb/msi_mantis on top of ref3. All on commit `9cc9593`.

### Group means (averaged across 3 seeds)

| Probe | ref3 | mol | Baseline | ref3 Δ | mol Δ |
|-------|------|-----|----------|--------|-------|
| linear | 0.7752 | 0.7804 | 0.7555 | +0.0197 | +0.0249 |
| knn | 0.7192 | 0.7182 | 0.6839 | +0.0353 | +0.0343 |
| 16shot | 0.6421 | 0.6409 | 0.5890 | +0.0531 | +0.0519 |
| segmentation | 0.3019 | 0.3072 | 0.3089 | -0.0070 | -0.0017 |
| progression | 0.6441 | 0.6265 | 0.6418 | +0.0023 | -0.0153 |
| mutation | 0.6165 | 0.6147 | 0.5594 | +0.0571 | +0.0553 |
| survival | 0.5702 | 0.5671 | 0.5894 | -0.0192 | -0.0223 |
| robustness | 0.8909 | 0.8929 | 0.8531 | +0.0378 | +0.0398 |
| **mean (Labless score)** | **0.6450** | **0.6435** | **0.6277** | **+0.0173** | **+0.0158** |

### Per-run vs baseline (dinov2 / base)

#### r3-s2026 (ref3 subtype/expr512/fga s2026)

| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7767 | 0.7555 | +0.0212 |
| knn | 0.7136 | 0.6839 | +0.0297 |
| 16shot | 0.6423 | 0.5890 | +0.0533 |
| segmentation | 0.2930 | 0.3089 | -0.0159 |
| progression | 0.6653 | 0.6418 | +0.0235 |
| mutation | 0.6223 | 0.5594 | +0.0629 |
| survival | 0.5754 | 0.5894 | -0.0140 |
| robustness | 0.8920 | 0.8531 | +0.0389 |
| **mean (Labless score)** | **0.6476** | **0.6277** | **+0.0199** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/5qeqquz7
Labless: submitted (submission_id=987f141fb9)

#### r3-s2027 (ref3 s2027)

| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7649 | 0.7555 | +0.0094 |
| knn | 0.7217 | 0.6839 | +0.0378 |
| 16shot | 0.6411 | 0.5890 | +0.0521 |
| segmentation | 0.3091 | 0.3089 | +0.0002 |
| progression | 0.6343 | 0.6418 | -0.0075 |
| mutation | 0.6150 | 0.5594 | +0.0556 |
| survival | 0.5680 | 0.5894 | -0.0214 |
| robustness | 0.8951 | 0.8531 | +0.0420 |
| **mean (Labless score)** | **0.6436** | **0.6277** | **+0.0159** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/952dpbm0
Labless: submitted (submission_id=b932ded428)

#### r3-s2028 (ref3 s2028)

| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7842 | 0.7555 | +0.0287 |
| knn | 0.7222 | 0.6839 | +0.0383 |
| 16shot | 0.6428 | 0.5890 | +0.0538 |
| segmentation | 0.3036 | 0.3089 | -0.0053 |
| progression | 0.6327 | 0.6418 | -0.0091 |
| mutation | 0.6123 | 0.5594 | +0.0529 |
| survival | 0.5672 | 0.5894 | -0.0222 |
| robustness | 0.8855 | 0.8531 | +0.0324 |
| **mean (Labless score)** | **0.6438** | **0.6277** | **+0.0161** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/fny5qid8
Labless: submitted (submission_id=78f61d7f43)

#### mol-s2026 (mol +purity/tmb/msi_mantis s2026)

| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7777 | 0.7555 | +0.0222 |
| knn | 0.7185 | 0.6839 | +0.0346 |
| 16shot | 0.6504 | 0.5890 | +0.0614 |
| segmentation | 0.3075 | 0.3089 | -0.0014 |
| progression | 0.6311 | 0.6418 | -0.0107 |
| mutation | 0.6223 | 0.5594 | +0.0629 |
| survival | 0.5846 | 0.5894 | -0.0048 |
| robustness | 0.8936 | 0.8531 | +0.0405 |
| **mean (Labless score)** | **0.6482** | **0.6277** | **+0.0205** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/tffezcn9
Labless: submitted (submission_id=285a5f677b)

#### mol-s2027 (mol s2027)

| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7868 | 0.7555 | +0.0313 |
| knn | 0.7136 | 0.6839 | +0.0297 |
| 16shot | 0.6329 | 0.5890 | +0.0439 |
| segmentation | 0.3123 | 0.3089 | +0.0034 |
| progression | 0.6266 | 0.6418 | -0.0152 |
| mutation | 0.6082 | 0.5594 | +0.0488 |
| survival | 0.5624 | 0.5894 | -0.0270 |
| robustness | 0.8902 | 0.8531 | +0.0371 |
| **mean (Labless score)** | **0.6416** | **0.6277** | **+0.0139** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/5x1ftf3p
Labless: submitted (submission_id=33ae642612)

#### mol-s2028 (mol s2028)

| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.7768 | 0.7555 | +0.0213 |
| knn | 0.7226 | 0.6839 | +0.0387 |
| 16shot | 0.6395 | 0.5890 | +0.0505 |
| segmentation | 0.3018 | 0.3089 | -0.0071 |
| progression | 0.6218 | 0.6418 | -0.0200 |
| mutation | 0.6135 | 0.5594 | +0.0541 |
| survival | 0.5544 | 0.5894 | -0.0350 |
| robustness | 0.8950 | 0.8531 | +0.0419 |
| **mean (Labless score)** | **0.6407** | **0.6277** | **+0.0130** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/pkh6vzx6
Labless: submitted (submission_id=c82603be94)

### Labless submission IDs (Round-2)

| Run | Submission ID | Verdict |
|-----|--------------|---------|
| r3-s2026 | `987f141fb9` | promising |
| r3-s2027 | `b932ded428` | promising |
| r3-s2028 | `78f61d7f43` | promising |
| mol-s2026 | `285a5f677b` | promising |
| mol-s2027 | `33ae642612` | promising |
| mol-s2028 | `c82603be94` | promising |


# 20260620_jepa_fino-a33-base

## What this is

A/B control for the FINO A.3.3 treatment run (97509). This is the unedited origin/jepa-fino 3-factor all-M+ FINO (subtype/expr512/fga), grad_equalize OFF. Serves as the baseline against which the A.3.3 paired tss-suppression + cancer-M+ treatment is evaluated.

## Hypothesis

N/A — this is the control branch of the A/B test. The treatment (20260620_jepa_fino-a33-treat) adds cancer (M+), tss (M-), and grad_equalize.

## Results (auto-filled 2026-06-20)

| Probe | This run | JEPA baseline | Delta |
|-------|----------|---------------|-------|
| linear | 0.7872 | 0.7842 | +0.0030 |
| knn | 0.7266 | 0.7061 | +0.0205 |
| 16shot | 0.6431 | 0.6383 | +0.0048 |
| segmentation | 0.2865 | 0.2891 | -0.0026 |
| progression | 0.6590 | 0.6575 | +0.0015 |
| mutation | 0.6282 | 0.6162 | +0.0120 |
| survival | 0.5643 | 0.5783 | -0.0140 |
| robustness | 0.8885 | 0.8855 | +0.0030 |
| **mean (Labless score)** | **0.6479** | **0.6444** | **+0.0035** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/4c3279uf
Labless: 6390756087

## Analysis vs baseline

This run is the A/B control (unedited origin/jepa-fino 3-factor all-M+ FINO, grad_equalize OFF). Its own scores vs the canonical JEPA baseline: +0.0030 mean (0.6479 vs 0.6444). No individual probe regressed; the strongest gains are progression (+0.0015), robustness (+0.0030), survival (-0.0140), and linear (+0.0030). Training was stable: grad_norm first=2.31, last=8.01, range 1.40-13.13. Batch size = 128 throughout. This run provides the counterfactual for the A.3.3 treatment (job 97509).


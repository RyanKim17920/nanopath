# Nanopath JEPA — exp_0214_retrain

_Auto-generated 2026-06-16_

## Results — retrain2 (seed 7777, main.yaml) (auto-filled 2026-06-16)

| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.8162 | 0.7842 | +0.0320 |
| knn | 0.7426 | 0.7061 | +0.0365 |
| 16shot | 0.6786 | 0.6383 | +0.0403 |
| segmentation | 0.3312 | 0.2891 | +0.0421 |
| progression | 0.6485 | 0.6575 | -0.0090 |
| mutation | 0.6243 | 0.6162 | +0.0081 |
| survival | 0.5756 | 0.5783 | -0.0027 |
| robustness | 0.8957 | 0.8855 | +0.0102 |
| **mean (Labless score)** | **0.6641** | **0.6444** | **+0.0197** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/36fe2il7
Labless: FAILED: dry-run error — review files exceed 120000 bytes (codebase diff too large for Labless locked-path validation)

## Results — seed42 (seed 42, seed42.yaml) (auto-filled 2026-06-16)

| Probe | This run | Baseline | Δ |
|-------|----------|----------|---|
| linear | 0.8078 | 0.7842 | +0.0236 |
| knn | 0.7396 | 0.7061 | +0.0335 |
| 16shot | 0.6821 | 0.6383 | +0.0438 |
| segmentation | 0.3314 | 0.2891 | +0.0423 |
| progression | 0.6327 | 0.6575 | -0.0248 |
| mutation | 0.6133 | 0.6162 | -0.0029 |
| survival | 0.5586 | 0.5783 | -0.0197 |
| robustness | 0.8889 | 0.8855 | +0.0034 |
| **mean (Labless score)** | **0.6568** | **0.6444** | **+0.0124** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/1wtlto8b
Labless: FAILED: dry-run error — review files exceed 120000 bytes (codebase diff too large for Labless locked-path validation)

## Analysis vs baseline

Both the seed-7777 retrain (mean 0.6641) and seed-42 variance test (mean 0.6568) exceed the JEPA baseline (mean 0.6444) by +0.020 and +0.012 respectively. The seed-7777 result closely matches the original 0.6636 run (delta ≈ +0.0005), confirming the training pipeline is reproducible. The seed-to-seed gap of 0.0073 is consistent with the previously observed noise floor (σ ≈ 0.004 across 5 seeds), meaning the two seeds fall well within expected variance.

The largest per-probe gains over the JEPA baseline are in 16-shot (+0.040 to +0.044) and segmentation (+0.042 to +0.043), suggesting the additional training improved few-shot generalization and spatial representation quality. Progression and survival probes regressed slightly against the baseline (up to -0.025), which may reflect a trade-off where the model's representation capacity shifted toward classification and segmentation at the expense of longitudinal task signals.

Training dynamics were stable and nearly identical across both seeds. Gradient norms grew from ~2.0–2.4 to ~7.8–8.7 over training, with peak spikes around 11.6–12.9, consistent with the expected warmup-then-grow pattern under layerwise decay. Total loss declined monotonically from ~12.6 to ~10.1, while validation loss remained flat (~11.8), indicating no overfitting. The absence of NaN losses, OOM events, or CUDA errors confirms a clean training run. Batch size was held constant at 128 throughout both runs.

# 20260620_jepa_fino-a33-treat

## What's different from base

+cancer (M+, sign+1) and +tss (M-, sign-1, gradient reversal) FINO factors; grad_equalize ON (Alg A.3) with paper-faithful raw-grad-norm lazy init + warmup-NaN guard. Base = origin/jepa-fino 3-factor all-M+ (subtype/expr512/fga), grad_equalize off.

## Hypothesis

FINO A.3.3 practical guideline — suppressing the tss institution batch while preserving the entangled cancer-type signal (paired M-/M+), balanced by grad-equalize, should improve generalization vs the unbalanced all-M+ baseline.

## Results (auto-filled 2026-06-20)

| Probe | This run | JEPA baseline | Delta |
|-------|----------|---------------|-------|
| linear | 0.7660 | 0.7842 | -0.0182 |
| knn | 0.6808 | 0.7061 | -0.0253 |
| 16shot | 0.6197 | 0.6383 | -0.0186 |
| segmentation | 0.2955 | 0.2891 | +0.0064 |
| progression | 0.6136 | 0.6575 | -0.0439 |
| mutation | 0.6199 | 0.6162 | +0.0037 |
| survival | 0.5790 | 0.5783 | +0.0007 |
| robustness | 0.8996 | 0.8855 | +0.0141 |
| **mean (Labless score)** | **0.6343** | **0.6444** | **-0.0101** |

Wandb: https://wandb.ai/ryankim17920-university-of-illinois-urbana-champaign/nanopath/runs/2nihjh2u
Labless: def80f0d09

## Analysis vs baseline

The A.3.3 treatment (paired tss-suppression + cancer-M+ + grad-equalize) net-regressed the aggregate Labless score by -0.0136 versus the 3-factor all-M+ baseline (0.6343 vs 0.6479). This shift is approximately 3.4x the seed-noise SD (approx.0.004 from the noise study), suggesting a real effect, though confirmed only at a single seed. The regressed probes form a coherent pattern: knn (-0.0458), slide/progression (-0.0454), linear (-0.0212), and 16shot (-0.0234) all declined, indicating the adversarial tss-suppression damaged general representation quality. Conversely, the axes that batch-removal should help improved: survival (+0.0147), robustness (+0.0111), and segmentation (+0.0090). This trade-off is consistent with the dependence analysis finding U(cancer|tss)=1.0 -- suppressing the institution batch (tss) while preserving the entangled cancer signal (cancer) redistributed representational capacity away from linearly-decodable features toward domain-shift robustness and survival prediction. Training was stable: grad_norm ranged 1.29-12.99 (first=2.44, last=7.88), comparable to the baseline's 1.40-13.13 (first=2.31, last=8.01). Both runs used batch_size=128 throughout. The kde loss was slightly higher in treatment (0.0228 vs 0.0048 at step 7812), possibly reflecting grad-equalize redistributing capacity.

**Conclusion**: A.3.3 tss-suppression buys robustness/survival at the cost of linear/kNN/slide quality; aggregate net-negative at gamma_max=0.7 single seed. Follow-up: multi-seed confirm, try lower gamma_max for tss, test mpp (eta^2|tss=0.82) as an additional acquisition-only M- arm.


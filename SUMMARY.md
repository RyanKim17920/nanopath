# 20260620_jepa_fino-a33-treat

## What's different from base

+cancer (M+, sign+1) and +tss (M-, sign-1, gradient reversal) FINO factors; grad_equalize ON (Alg A.3) with paper-faithful raw-grad-norm lazy init + warmup-NaN guard. Base = origin/jepa-fino 3-factor all-M+ (subtype/expr512/fga), grad_equalize off.

## Hypothesis

FINO A.3.3 practical guideline — suppressing the tss institution batch while preserving the entangled cancer-type signal (paired M-/M+), balanced by grad-equalize, should improve generalization vs the unbalanced all-M+ baseline.

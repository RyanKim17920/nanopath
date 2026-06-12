exp_0037 — seg decode-geometry: SUM fusion of last-4 block patch tokens

Sweep axis: vary exp_0032's readout-geometry FUSION OPERATOR (concat -> sum), n held at 4.

Surface: model.py `encode_image` (single caller = probe.py:_seg_extract_features, segmentation
probe ONLY). New helper `_intermediate_patch_tokens_sum(x, n=4)`:
  - re-runs the trunk, takes patch tokens from the last 4 blocks (each through shared final
    LayerNorm), and SUMS them elementwise -> (B, n_patches, dim).
  - encode_image returns [zero register-pad || fused patches]; probe slices [:, registers:].

Why sum vs exp_0032 concat: concat widens the seg head's d_encoder 4x (dim -> 4*dim). The
MaskTransformer head projects whatever d_encoder it gets into d_model=768, so the 4x-wide
concat input can overfit/underfit the tiny per-fold head and is the likely source of the
~0.017 pathorob-dip-coincident readout variance the brief flags as training-variance. Sum keeps
the encoder dim at `dim` (native width = baseline), averaging complementary depth information
in place — predicted to preserve most of exp_0032's seg gain with a cleaner aggregate.

Orthogonality (same as exp_0032): NO pretraining loss term added/changed. forward() (training)
and probe_features() (CLS classification probes) untouched; the x_norm_patchtokens / x_norm_clstoken
dict that pathorob, robustness, classification read is NOT modified. Only the dense features the
seg probe consumes change. Cannot perturb linear/knn/16shot/progression/mutation/survival/robustness.

Invariants preserved:
- patch-token COUNT unchanged (256 for 224px/14): grid size gs=int(n**0.5) inferred from count.
- feature dim STAYS dim (sum); head d_encoder = train_feats.shape[-1] adapts automatically.
- register prefix kept as zero padding so probe slice `[:, registers:]` still lands on patches.

No probe.py / benchmarking / labless edits. Loss/optimizer/aug/budget unchanged.
Budget invariants (max_train_samples 1e6, tile_presentations<=1e6, max_train_flops 1e18) untouched.

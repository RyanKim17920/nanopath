exp_0032 — seg decode-geometry (multi-layer patch tokens)

Surface: model.py `encode_image` (single caller = probe.py:423, segmentation probe ONLY).
Change: seg head now reads patch tokens concatenated from the last 4 transformer blocks
(each passed through the shared final LayerNorm), i.e. DINOv2 get_intermediate_layers(n=4)
dense recipe — instead of only the final CLS-optimized layer.

Why orthogonal to all 8+ failed SSL-loss children: no pretraining loss term added/changed.
forward() (used by training) and probe_features() (CLS classification probes) are untouched,
so this cannot perturb linear/knn/16shot/progression/mutation/survival/robustness — it only
changes the dense features the seg probe consumes. Sidesteps the in-trunk seg-vs-CLS wall.

Invariants preserved:
- patch-token COUNT unchanged (256 for 224px/14): MaskTransformer infers grid size gs=int(n**0.5)
  from token count — must stay 256. Concat is along CHANNEL dim only.
- feature dim widens dim -> 4*dim; head d_encoder = train_feats.shape[-1] adapts automatically.
- register prefix kept as zero padding so probe slice `[:, registers:]` still lands on patches.

No probe.py / benchmarking / labless edits. Loss/optimizer/aug/budget unchanged.
Budget invariants (max_train_samples 1e6, tile_presentations<=1e6, max_train_flops 1e18) untouched.

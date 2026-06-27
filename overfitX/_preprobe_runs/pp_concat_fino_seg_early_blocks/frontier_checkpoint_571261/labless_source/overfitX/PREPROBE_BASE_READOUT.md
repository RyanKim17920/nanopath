# Canonical Standing-Base Pre-Probe Readout (IMPLEMENTED)

The pre-probe TRACK searches improvements **on top of** this standing base readout.
It is EVAL-ONLY (in `model.py` `probe_features()` / `encode_image()`), legal (no eval-set-fitted
statistics — the JBU guide is the per-image input only). NOT in the RAW training path.

## Classification readout (linear / knn / fewshot / slide / auc / survival / robustness)
Use the **single final CLS token** (last transformer block) — NOT multi-depth concatenation.

**Why not multi-depth concat:** The multi-depth CLS-concat ("cls stack") was empirically
evaluated on the 3-checkpoint panel and found HARMFUL: panel mean -0.0149 vs plain final CLS.
The standing baseline (plain CLS + JBU-last4 seg) achieves panel mean 0.657, while the concat
composite scored ~0.642. Multi-depth concat is NOT part of the standing base.

## Segmentation readout (pannuke / monusac / consep)
- Concatenate the **last four normalized block outputs** (patch tokens).
- **Keep register tokens.**
- Reshape patch tokens into a denser **32×32 feature grid**.
- Apply a lightweight **JBU-style (Joint Bilateral Upsampling) image-guided upsample/sharpen**
  (guidance = the input image; bilateral weights from the guide; fixed sigma_spatial/sigma_range;
  no eval-set statistics).

## Future pre-probe EXPERIMENT idea (NOT the base)
Mean of patch tokens concatenated with CLS — carry as a preprobe-track experiment candidate,
not part of the standing base.

## Status (2026-06-21) — IMPLEMENTED & ACTIVE
Implemented as the DEFAULT (non-gated) readout in preprobe `model.py`:
- `probe_features()` — classification: **plain final CLS token** (last block), single `[B, embed_dim]`
  vector. (Multi-depth concat was retired — empirically harmful, -0.0149 panel vs plain CLS.)
- `encode_image()` — segmentation/dense: last-four block patch tokens
  (`last_four = (depth-3, depth-2, depth-1, depth)` → (9,10,11,12) for a 12-block model),
  concatenated over the feature dim, register tokens kept, reshaped to a 32×32 grid, then
  `_jbu_upsample(grid, x, SEG_GRID=32)` (fixed sigma_spatial=1.0 / sigma_range=0.1; guide = input
  image; no eval-set statistics).
The docstring marks it explicitly: "this is the DEFAULT readout, not a gated overlay."
Validated on the rolling panel (frontier + reseed-anchor 314159 + consolidated baseline 220794):
panel mean 0.657, per-ckpt scores 0.6583 / 0.6601 / 0.6525.
Recorded in `MANIFEST.json` (`baseline_readout`), `batch02/baseline_readout_panel.json`.
The preprobe directions (AnyUp, register fusion, pooled-patch, deterministic centering) are
VARIATIONS layered on top, all fixed-feed-forward (no TTO).

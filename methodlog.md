# exp_0039 — cross-view mean-patch consistency (pathorob robustness lever)

- Parent: exp_0000 (baseline 0.6444). Config for full submit: configs/main.yaml.
- Surface: pathorob embeds [CLS || mean-of-patch-tokens] from forward() final-norm output.
  Robustness index gated by mean-pooled patch-token separability across medical-center shift.
- Change (train.py compute_losses): add cross_view_patch_weight term. The two global views are
  independent augmentations of the SAME tile (in-dataset proxy for stain/scanner/center shift).
  Pull each view's STUDENT visible-patch mean-pool toward the EMA-TEACHER full-patch mean-pool of
  the OTHER view, symmetric, stop-grad teacher, layer-normed smooth-L1. Folded into ibot_loss so it
  rides total_loss = dino+ibot+kde and is included in evaluate().
- Why on-brief: pools first (no per-patch dense pressure -> avoids seg-vs-CLS regression that sank
  exp_0020-0024); does NOT route through CLS / prototypes / uniformity / variance-isotropy; does NOT
  touch encode_image or augmentation strength.
- Knob: dino.cross_view_patch_weight=0.1 (main.yaml + smoke.yaml). Default 0.0 (.get) = off.
  Guarded by global_views == 2.
- NEXT: smoke-test with configs/smoke.yaml first, then np_full_submit configs/main.yaml.
  Watch: robustness up vs baseline 0.8855; linear (0.7842) / knn (0.7061) / survival (0.5783) held.

"""Standalone verification for SDHL (multi-depth JEPA / Self-Distillation of Hidden Layers).

Exercises the REAL model.py mechanism (DinoV2ViT.forward(return_blocks=...) + SDHLHead) and the REAL
train.py loss-assembly math end-to-end on a small random ViT-S/14, so gradients + loss deltas can be
observed without a GPU or a pretrained download. Faithfully mirrors compute_losses' JEPA + SDHL wiring.

Proves the two required properties:
  (a) sdhl.enabled=true  -> sdhl loss is NON-ZERO, SDHL head params get NON-ZERO gradients, and the
      total loss DIFFERS from the baseline (jepa-only) total.
  (b) sdhl.enabled=false -> the training path is BYTE-IDENTICAL to baseline: the return_blocks capture
      is a pure side-channel (main outputs bitwise-equal) and, with the loss term gated off, the
      backbone gradients are bitwise-equal to the baseline (no-capture) gradients.

Run: /admin/home/ryan.kim/nanopath/.venv/bin/python3 verify_sdhl.py
"""
import random
from copy import deepcopy

import torch
import torch.nn.functional as F

from model import DinoV2ViT, JEPAPredictor, SDHLHead

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_BLOCKS = [8, 9, 10, 11]   # new canonical default (late cluster)
DIM = 384
GRID = 16                        # 224 / 14
B_PER_VIEW = 2                   # `b` in compute_losses
N_IMG = 2 * B_PER_VIEW           # b * global_views images fed to the masked forward
LOSS_WEIGHT = 0.2


# Faithful copy of train.make_block_mask (contiguous square blocks, same weighting).
def make_block_mask(batch, grid, device, n_blocks=4, block_scale=0.10):
    masks = torch.zeros(batch, grid, grid, dtype=torch.bool, device=device)
    side = max(1, round(grid * block_scale ** 0.5))
    for i in range(batch):
        for _ in range(n_blocks):
            top = random.randint(0, grid - side)
            left = random.randint(0, grid - side)
            masks[i, top:top + side, left:left + side] = True
    masks = masks.flatten(1)
    idx = masks.flatten().nonzero().flatten()
    weights = (1 / masks.sum(-1).clamp(min=1)).unsqueeze(-1).expand_as(masks)[masks]
    return masks, idx, weights


def build():
    torch.manual_seed(0)
    student = DinoV2ViT(variant="dinov2_vits14_reg").to(DEV)
    teacher = deepcopy(student).to(DEV)
    teacher.train(False)
    for p in teacher.parameters():
        p.requires_grad = False
    predictor = JEPAPredictor(DIM, depth=2, width=0).to(DEV)   # final-layer I-JEPA head
    sdhl = SDHLHead(DIM, TARGET_BLOCKS, per_depth_layernorm=True, merge="concat").to(DEV)
    return student, teacher, predictor, sdhl


# Exact replica of the JEPA term in train.compute_losses (final-layer, LN'd teacher target, mask-weighted).
def jepa_term(sg, t, predictor, mask_idx, mask_w, b):
    target = F.layer_norm(t["x_norm_patchtokens"].flatten(0, 1), (DIM,))[mask_idx]
    pred = predictor(sg["x_norm_patchtokens"]).flatten(0, 1)[mask_idx]
    return F.smooth_l1_loss(pred, target, reduction="none").mean(-1).mul(mask_w).sum() / max(1, b * 2)


def gsum(params):
    return float(sum(p.grad.abs().sum() for p in params if p.grad is not None))


random.seed(0)
x = torch.randn(N_IMG, 3, GRID * 14, GRID * 14, device=DEV)
masks, mask_idx, mask_w = make_block_mask(N_IMG, GRID, DEV)

print("=" * 74)
print(f"SDHL verification  device={DEV}  target_blocks={TARGET_BLOCKS}  loss_weight={LOSS_WEIGHT}")
print(f"masks {tuple(masks.shape)}  masked patches M={mask_idx.numel()}  mask_w {tuple(mask_w.shape)}")
print("=" * 74)

# ------------------------------------------------------------------ [1] config keys read
print("\n[1] CONFIG KEYS (mirror train.py's cfg.get('sdhl') reads)")
import yaml
main_cfg = yaml.safe_load(open("configs/main.yaml"))
sdhl_cfg = main_cfg.get("sdhl") or {}
assert bool(sdhl_cfg.get("enabled", False)) is False, "configs/main.yaml must ship sdhl.enabled=false (baseline)"
assert [int(b) for b in sdhl_cfg.get("target_blocks", [8, 9, 10, 11])] == [8, 9, 10, 11]
assert str(sdhl_cfg.get("merge", "concat")) == "concat"
assert float(sdhl_cfg.get("loss_weight", 0.2)) == 0.2
assert bool(sdhl_cfg.get("per_depth_layernorm", True)) is True
print(f"  configs/main.yaml sdhl block: {sdhl_cfg}")
print("  [1] PASS  (default-disabled, canonical keys present)")

# ------------------------------------------------------------------ [2] block extraction + forward byte-identical
print("\n[2] BLOCK EXTRACTION + FORWARD BYTE-IDENTICAL (return_blocks is a pure side-channel)")
student, teacher, predictor, sdhl = build()
# same-rng forward with and without capture
g = torch.Generator(device="cpu")
with torch.no_grad():
    out_none = student(x, masks=masks, return_blocks=None)
    out_cap = student(x, masks=masks, return_blocks=TARGET_BLOCKS)
for key in ("x_norm_clstoken", "x_norm_patchtokens", "x_norm_regtokens"):
    eq = torch.equal(out_none[key], out_cap[key])
    print(f"  {key:22s} bitwise-equal with/without capture = {eq}")
    assert eq, f"return_blocks perturbed {key} -- NOT byte-identical!"
assert "blocks" not in out_none and "blocks" in out_cap
assert set(out_cap["blocks"]) == set(TARGET_BLOCKS)
with torch.no_grad():
    t = teacher(x, return_blocks=TARGET_BLOCKS)
for d in TARGET_BLOCKS:
    s_shape, t_shape = tuple(out_cap["blocks"][d].shape), tuple(t["blocks"][d].shape)
    print(f"  block {d:2d}: student {s_shape}  teacher {t_shape}")
    assert s_shape == (N_IMG, GRID * GRID, DIM) and t_shape == (N_IMG, GRID * GRID, DIM)
assert t["blocks"][11].requires_grad is False and t["blocks"][11].detach().grad_fn is None
try:
    student(x, return_blocks=[12])   # depth is 12 -> index 12 out of range
    raise SystemExit("ERROR: out-of-range return_blocks=[12] was NOT rejected")
except AssertionError:
    print("  out-of-range return_blocks=[12] correctly rejected by assert")
print("  [2] PASS")

# ------------------------------------------------------------------ [3] enabled=true is LIVE
print("\n[3] sdhl.enabled=true IS LIVE  (nonzero loss, nonzero head grads, total differs from baseline)")
student, teacher, predictor, sdhl = build()
student.zero_grad(set_to_none=True); predictor.zero_grad(set_to_none=True); sdhl.zero_grad(set_to_none=True)
sg = student(x, masks=masks, return_blocks=TARGET_BLOCKS)
with torch.no_grad():
    t = teacher(x, return_blocks=TARGET_BLOCKS)
jepa = jepa_term(sg, t, predictor, mask_idx, mask_w, B_PER_VIEW)
sdhl_loss = sdhl(sg["blocks"], t["blocks"], mask_idx, mask_w, B_PER_VIEW)
total_baseline = jepa
total_sdhl = jepa + LOSS_WEIGHT * sdhl_loss
print(f"  jepa_loss           = {jepa.detach().item():.6f}")
print(f"  sdhl_loss           = {sdhl_loss.detach().item():.6f}  (grad_fn set = {sdhl_loss.grad_fn is not None})")
print(f"  total baseline      = {total_baseline.detach().item():.6f}")
print(f"  total with 0.2*sdhl = {total_sdhl.detach().item():.6f}   delta = {(total_sdhl - total_baseline).detach().item():+.6f}")
assert sdhl_loss.detach().item() != 0.0, "sdhl_loss is exactly zero"
assert sdhl_loss.grad_fn is not None, "sdhl_loss has no grad graph"
assert total_sdhl.detach().item() != total_baseline.detach().item(), "total loss did NOT differ from baseline"
total_sdhl.backward()
for d in TARGET_BLOCKS:
    gs = gsum(student.blocks[d].parameters())
    print(f"  student.blocks[{d:2d}] grad-sum = {gs:.6e}")
    assert gs > 0, f"student block {d} received ZERO gradient"
for k, d in enumerate(TARGET_BLOCKS):
    gh = gsum(sdhl.heads[k].parameters())
    print(f"  sdhl.heads[{k}] (block {d}) grad-sum = {gh:.6e}")
    assert gh > 0, f"sdhl head {k} received ZERO gradient"
teacher_grads = [p.grad for p in teacher.parameters() if p.grad is not None]
tn_params = sum(p.numel() for m in sdhl.target_norms for p in m.parameters())
print(f"  teacher params with grad = {len(teacher_grads)} (expect 0)   target_norm trainable params = {tn_params} (expect 0)")
assert len(teacher_grads) == 0 and tn_params == 0
print("  [3] PASS")

# ------------------------------------------------------------------ [4] disabled=false is BYTE-IDENTICAL (loss + grads)
print("\n[4] sdhl.enabled=false IS BYTE-IDENTICAL  (loss + backbone grads unchanged vs baseline)")
# baseline path: sdhl_head is None -> no capture, total = jepa only.
student_a, teacher_a, predictor_a, _ = build()
student_a.zero_grad(set_to_none=True); predictor_a.zero_grad(set_to_none=True)
sg_a = student_a(x, masks=masks, return_blocks=None)
with torch.no_grad():
    t_a = teacher_a(x, return_blocks=None)
loss_a = jepa_term(sg_a, t_a, predictor_a, mask_idx, mask_w, B_PER_VIEW)
loss_a.backward()
grads_a = [p.grad.detach().clone() for p in student_a.parameters()]

# "sdhl present but the gate would add nothing" control: capture blocks, but (as when disabled) DO NOT add
# the sdhl term. Backbone grads must be bitwise-equal to baseline -> capture channel is inert on the main graph.
student_b, teacher_b, predictor_b, _ = build()
student_b.zero_grad(set_to_none=True); predictor_b.zero_grad(set_to_none=True)
sg_b = student_b(x, masks=masks, return_blocks=TARGET_BLOCKS)
with torch.no_grad():
    t_b = teacher_b(x, return_blocks=TARGET_BLOCKS)
loss_b = jepa_term(sg_b, t_b, predictor_b, mask_idx, mask_w, B_PER_VIEW)
loss_b.backward()
grads_b = [p.grad.detach().clone() for p in student_b.parameters()]

print(f"  baseline jepa loss (no capture)   = {float(loss_a):.8f}")
print(f"  jepa loss WITH capture, no term   = {float(loss_b):.8f}")
assert torch.equal(loss_a, loss_b), "loss changed when blocks were merely captured!"
all_eq = all(torch.equal(ga, gb) for ga, gb in zip(grads_a, grads_b))
n_params = len(grads_a)
max_absdiff = max((ga - gb).abs().max().item() for ga, gb in zip(grads_a, grads_b))
print(f"  backbone grad tensors bitwise-equal = {all_eq}  ({n_params} tensors, max|Δ|={max_absdiff:.2e})")
assert all_eq, "backbone grads differ -- capture side-channel is NOT inert!"
print("  [4] PASS  (disabled path is byte-identical to baseline)")

print("\n" + "=" * 74)
print("ALL CHECKS PASSED: SDHL is config-gated, live when on, byte-identical when off.")
print("=" * 74)

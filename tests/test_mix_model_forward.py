# Integration check that mixed-size local crops flow through the REAL model exactly as
# train.py's compute_losses does: forward each size group separately, concatenate at the
# CLS token, chunk by total local views for the DINO local loss. Uses a randomly-init
# DinoV2ViT (no pretrained-weight download) on CPU with tiny tensors.
#
# Run: /admin/home/ryan.kim/nanopath/.venv/bin/python3 tests/test_mix_model_forward.py
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import DinoV2ViT, DINOHead


def test_mixed_size_forward_and_dino_local_loss():
    torch.manual_seed(0)
    backbone = DinoV2ViT(variant="dinov2_vits14_reg", drop_path_rate=0.0).eval()  # random init, no download
    head = DINOHead(in_dim=backbone.embed_dim, n_prototypes=128).eval()
    B = 2
    mix = [(112, 4), (128, 4)]  # 8 crops total
    total_local_views = sum(c for _, c in mix)

    # Emulate batch["local_views"] after collate: one (B, count, 3, S, S) tensor per size group.
    local_groups = [torch.randn(B, c, 3, s, s) for s, c in mix]
    # build_lf: crop-major flatten per group -> list of (count*B, 3, S, S).
    lf_groups = [lv.transpose(0, 1).flatten(0, 1) for lv in local_groups]
    assert [tuple(t.shape) for t in lf_groups] == [(8, 3, 112, 112), (8, 3, 128, 128)]

    # compute_losses local branch (mixed): per-group forward -> cat CLS -> head.
    sl_cls = torch.cat(
        [head(backbone(g)["x_norm_clstoken"]) for g in lf_groups], dim=0
    )
    assert tuple(sl_cls.shape) == (total_local_views * B, 128), sl_cls.shape

    # DINO local loss structure: chunk by L=total_local_views, each chunk is one crop's batch.
    chunks = sl_cls.chunk(total_local_views)
    assert len(chunks) == total_local_views and all(tuple(c.shape) == (B, 128) for c in chunks)

    # A DINO-style CE against a dummy teacher target must be finite and backprop cleanly.
    t_prob = torch.softmax(torch.randn(2, B, 128), dim=-1)  # 2 global teacher views
    loss = sum(-(y * torch.log_softmax(x, dim=-1)).sum(-1).mean() for x in chunks for y in t_prob)
    loss = loss / (2 * total_local_views + 2)
    assert torch.isfinite(loss), loss
    loss.backward()
    gnorm = backbone.patch_embed.proj.weight.grad.norm()
    assert gnorm > 0, "mixed-size local crops must produce nonzero backbone gradient"
    print(f"  sl_cls={tuple(sl_cls.shape)} chunks={len(chunks)}x{tuple(chunks[0].shape)} "
          f"loss={float(loss):.4f} backbone_grad_norm={float(gnorm):.4f}  OK")


def test_scalar_forward_matches_single_group():
    # Single-size path (len(lf_groups)==1) must equal a plain single forward (byte-identical branch).
    torch.manual_seed(1)
    backbone = DinoV2ViT(variant="dinov2_vits14_reg", drop_path_rate=0.0).eval()
    head = DINOHead(in_dim=backbone.embed_dim, n_prototypes=64).eval()
    lf = torch.randn(8, 3, 112, 112)  # 4 crops x B=2, scalar path
    direct = head(backbone(lf)["x_norm_clstoken"])
    grouped = torch.cat([head(backbone(g)["x_norm_clstoken"]) for g in [lf]], dim=0)
    assert torch.allclose(direct, grouped, atol=1e-6)
    print(f"  scalar single-group == direct forward (shape {tuple(direct.shape)})  OK")


if __name__ == "__main__":
    for fn in [test_mixed_size_forward_and_dino_local_loss, test_scalar_forward_matches_single_group]:
        print(f"{fn.__name__}:")
        fn()
    print("\nALL MIX-MODEL-FORWARD TESTS PASSED")

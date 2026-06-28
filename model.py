# DinoV2ViT: clean ViT + 4 register tokens that loads Meta's `dinov2_vit{s,b,g}14_reg`
# pretrained weights via state_dict (no xformers, no dinov2 codebase imports).
# Attention runs on `F.scaled_dot_product_attention` so we get FlashAttention-2
# on H100 bf16 with no third-party kernel dependency. Module names below match
# Meta's checkpoint key layout exactly, so `load_dinov2_pretrained(model)` does
# a strict load.
#
# DINOHead is the small MLP + weight-normed classifier used by train.py for the
# DINO CLS self-distillation loss. It is intentionally trivial
# (~15 lines) so we have zero runtime dependency on the dinov2 codebase.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# (dim, depth, heads, pretrain_grid, ffn, pos_has_cls, weight URL[, registers]) for each supported variant.
DINOV2_VARIANTS = {
    "dinov2_vits14_reg": (384, 12, 6, 37, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"),
    "dinov2_vitb14_reg": (768, 12, 12, 37, "mlp", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"),
    "dinov2_vitg14_reg": (1536, 40, 24, 37, "swiglu", True, "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth"),
}


def probe_transforms():
    # Default for Nanopath-trained checkpoints; baseline scripts override this in their request config.
    transform = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor()])
    # Keep the two return slots because probe.py separates tile-image and slide/patch-bag probes.
    return transform, transform


# Stochastic depth: keep_prob bernoulli on the residual branch, scaled to preserve mean.
class DropPath(nn.Module):
    def __init__(self, p): super().__init__(); self.p = float(p)
    def forward(self, x):
        if self.p == 0.0 or not self.training: return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep)
        return x * mask / keep


# Per-channel learnable scale on residual branches; matches Meta's `ls1.gamma`/`ls2.gamma`.
class LayerScale(nn.Module):
    def __init__(self, dim): super().__init__(); self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x * self.gamma


# FINO gradient gate: identity forward, scales the gradient by `scale` on backward. sign>0 encourages the
# encoder to predict a metadata factor (M+); sign<0 reverses the gradient to suppress it (M-, DANN-style).
class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale): ctx.scale = scale; return x
    @staticmethod
    def backward(ctx, g): return g * ctx.scale, None


_diags = []

def clear_diags():
    _diags.clear()

def collect_diags():
    d = _diags.copy()
    _diags.clear()
    return d


# Attention with single qkv Linear + F.scaled_dot_product_attention (Flash-2 backend on H100 bf16).
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, log_diagnostics=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        if log_diagnostics:
            _diags.append({
                "q_norm": float(q.norm(dim=-1).mean().detach()),
                "k_norm": float(k.norm(dim=-1).mean().detach()),
                "v_norm": float(v.norm(dim=-1).mean().detach()),
                "qk_sim_max": float((q @ k.transpose(-2, -1)).amax(dim=-1).mean().detach()),
            })
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        hidden = (int(hidden * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(dim, 2 * hidden, bias=True)
        self.w3 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


# Standard pre-LN block: attn + ls1 + drop_path, then mlp + ls2 + drop_path.
class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, drop_path_p, ffn="mlp"):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.ls1 = LayerScale(dim)
        self.drop_path1 = DropPath(drop_path_p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, hidden) if ffn == "swiglu" else nn.Sequential()
        if ffn == "mlp":
            self.mlp.fc1 = nn.Linear(dim, hidden, bias=True)
            self.mlp.fc2 = nn.Linear(hidden, dim, bias=True)
        self.ls2 = LayerScale(dim)
        self.drop_path2 = DropPath(drop_path_p)

    def _ff(self, x): return self.mlp(x) if isinstance(self.mlp, SwiGLU) else self.mlp.fc2(F.gelu(self.mlp.fc1(x)))

    def forward(self, x, log_diagnostics=False):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), log_diagnostics=log_diagnostics)))
        x = x + self.drop_path2(self.ls2(self._ff(self.norm2(x))))
        return x


# ViT-S/B-14 with 4 register tokens; key layout matches Meta's DINOv2 register checkpoints
# (cls_token, register_tokens, pos_embed (1, 1+37^2, dim), mask_token (1, dim), patch_embed.proj,
# blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,ls1,ls2,mlp.fc1,mlp.fc2}, norm).
# Pos embed is bicubically interpolated at runtime to the current patch grid.
# Meta DINOv2 includes a cls pos and uses 37x37 patches; variant_cfg can override this for probes.
class DinoV2ViT(nn.Module):
    def __init__(self, variant="dinov2_vits14_reg", drop_path_rate=0.0, variant_cfg=None):
        super().__init__()
        cfg = variant_cfg or DINOV2_VARIANTS[variant]
        dim, depth, heads, pretrain_grid, ffn, pos_has_cls, _ = cfg[:7]
        mlp_ratio, patch, registers = 4.0, 14, cfg[7] if len(cfg) > 7 else 4
        self.variant = variant
        self.patch_size, self.registers, self.embed_dim = patch, registers, dim
        self._pretrain_grid, self._pos_has_cls = pretrain_grid, pos_has_cls
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, registers, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, int(self._pos_has_cls) + self._pretrain_grid**2, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        rates = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio, p, ffn=ffn) for p in rates)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    # Bicubic resample of the checkpoint patch-pos grid to the current (h, w) grid.
    def _interpolate_pos_embed(self, h, w):
        cls_pos = self.pos_embed[:, :1] if self._pos_has_cls else None
        g = self._pretrain_grid
        patch_pos = self.pos_embed[:, int(self._pos_has_cls):].reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        # antialias=True matches Meta's default for DINOv2 `_reg` variants.
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1).to(self.pos_embed.dtype)
        return torch.cat([cls_pos, patch_pos], dim=1) if cls_pos is not None else patch_pos

    # Build [cls, registers, patches] tokens; masked patch positions are replaced by mask_token.
    def _prepare_tokens(self, x, masks=None):
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).expand_as(x), x)
        cls = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        if self._pos_has_cls:
            x = torch.cat([cls, x], dim=1) + self._interpolate_pos_embed(h, w)
            return torch.cat([x[:, :1], regs, x[:, 1:]], dim=1)
        return torch.cat([cls, regs, x + self._interpolate_pos_embed(h, w)], dim=1)

    # Returns the dict shape Meta's `forward_features` returns; used by train.py and probe.py.
    # `checkpoint=True` re-runs each block under torch.utils.checkpoint to trade compute for memory;
    # useful when the 1-GPU batch of 128 (2 globals + 8 locals) does not fit in 80 GB.
    def forward(self, x, masks=None, checkpoint=False, log_diagnostics=False):
        x = self._prepare_tokens(x, masks)
        if log_diagnostics:
            clear_diags()
            _token_norms = []
            for blk in self.blocks:
                x = blk(x, log_diagnostics=True)
                r = self.registers
                _token_norms.append({
                    "cls_norm":   float(x[:, 0].norm(dim=-1).mean().detach()),
                    "reg_norm":   float(x[:, 1:1 + r].norm(dim=-1).mean().detach()) if r > 0 else 0.0,
                    "patch_norm": float(x[:, 1 + r:].norm(dim=-1).mean().detach()),
                })
            x = self.norm(x)
            result = {
                "x_norm_clstoken": x[:, 0],
                "x_norm_regtokens": x[:, 1 : 1 + self.registers],
                "x_norm_patchtokens": x[:, 1 + self.registers :],
            }
            _layers = collect_diags()
            for i, tn in enumerate(_token_norms):
                if i < len(_layers):
                    _layers[i].update(tn)
            result["_diagnostics"] = _layers
            result["_final_norm_cls"] = float(x[:, 0].norm(dim=-1).mean().detach())
            result["_final_norm_reg"] = float(x[:, 1 : 1 + self.registers].norm(dim=-1).mean().detach())
            result["_final_norm_patch"] = float(x[:, 1 + self.registers :].norm(dim=-1).mean().detach())
            return result
        for blk in self.blocks:
            if checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        return {
            "x_norm_clstoken": x[:, 0],
            "x_norm_regtokens": x[:, 1 : 1 + self.registers],
            "x_norm_patchtokens": x[:, 1 + self.registers :],
        }


    def _forward_intermediates(self, x, depths):
        x_tokens = self._prepare_tokens(x)
        want = set(depths)
        captured = {}
        for i, blk in enumerate(self.blocks):
            x_tokens = blk(x_tokens)
            if i in want:  # 0-indexed: matches deployed probe.py
                captured[i] = x_tokens
        return captured

    CLS_DEPTHS = (4, 6, 8, 11)

    def probe_features(self, x):
        # OVFX_CLS selects the classification readout (pre-probe baseline sweep).
        # Unset/"concat" = the current default (multi-depth CLS-concat).
        #   plain            -> final CLS only (x_norm_clstoken)
        #   concat           -> multi-depth CLS-concat (CLS_DEPTHS), L2-normed per depth
        #   meanpatch        -> [final CLS || mean patch token], each L2-normed
        #   concat_meanpatch -> multi-depth CLS-concat || mean patch token
        #   quantile         -> [CLS || per-dim Q0.1,Q0.5,Q0.9 of patches] (heterogeneity)
        #   normweight       -> [CLS || norm-weighted patch mean] (high-norm = informative)
        #   gram             -> [CLS || compact second-order sketch of patch features] (texture)
        #   concat_quantile  -> CLS-stack || per-dim Q0.1,Q0.5,Q0.9 of patches
        #   concat_normweight -> CLS-stack || norm-weighted patch mean
        #   concat_gram      -> CLS-stack || compact second-order sketch of patch features
        #   sinkhorn_cls     -> concat_fino 4-depth stack || Sinkhorn-centered (K=5 iters) projection
        # All are fixed feed-forward functions of frozen features (no per-image opt,
        # no eval-set stats) -> legal under the eval-fairness rule.
        mode = os.environ.get("OVFX_CLS", "plain")
        if mode == "plain":
            return self(x)["x_norm_clstoken"]
        if mode == "meanpatch":
            out = self(x)
            cls = F.normalize(out["x_norm_clstoken"], dim=-1, p=2)
            mp = F.normalize(out["x_norm_patchtokens"].mean(dim=1), dim=-1, p=2)
            return torch.cat([cls, mp], dim=-1)  # [B, 2 * embed_dim]
        if mode == "quantile":
            out = self(x)
            cls = F.normalize(out["x_norm_clstoken"], dim=-1, p=2)
            p = out["x_norm_patchtokens"]  # [B, n_patch, D]
            qs = torch.tensor([0.1, 0.5, 0.9], device=p.device, dtype=p.dtype)
            q = torch.quantile(p, qs, dim=1)  # [3, B, D]
            parts = [cls] + [F.normalize(q[i], dim=-1, p=2) for i in range(q.shape[0])]
            return torch.cat(parts, dim=-1)  # [B, 4 * embed_dim]
        if mode == "normweight":
            out = self(x)
            cls = F.normalize(out["x_norm_clstoken"], dim=-1, p=2)
            p = out["x_norm_patchtokens"]  # [B, n_patch, D]
            w = p.norm(dim=-1, keepdim=True)  # [B, n_patch, 1]
            wp = (p * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)  # [B, D]
            return torch.cat([cls, F.normalize(wp, dim=-1, p=2)], dim=-1)
        if mode == "gram":
            out = self(x)
            cls = F.normalize(out["x_norm_clstoken"], dim=-1, p=2)
            p = out["x_norm_patchtokens"]  # [B, n_patch, D]
            # Compact second-order: project patches by a FIXED random R [D, m] (seed 0,
            # deterministic), then per-projected-dim second moment E[(Rp)^2] over patches.
            # Each projected dim mixes original feature pairs, so the energies capture
            # cross-feature (texture) structure the mean discards. Bounded m -> usable dim.
            D = p.shape[-1]; m = 96
            gen = torch.Generator(device=p.device).manual_seed(0)
            R = torch.randn(D, m, generator=gen, device=p.device, dtype=p.dtype) / (D ** 0.5)
            z = p @ R  # [B, n_patch, m]
            second = (z * z).mean(dim=1)  # [B, m] diagonal of the projected gram
            return torch.cat([cls, F.normalize(second, dim=-1, p=2)], dim=-1)
        if mode == "concat_fino":
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            feats = [F.normalize(self.norm(caps[d])[:, 0], dim=-1, p=2) for d in self.CLS_DEPTHS]
            return torch.cat(feats, dim=-1)  # [B, 4 * embed_dim]
        if mode == "concat_quantile":
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            feats = [F.normalize(self.norm(caps[d])[:, 0], dim=-1, p=2) for d in self.CLS_DEPTHS]
            p = self(x)["x_norm_patchtokens"]  # [B, n_patch, D]
            qs = torch.tensor([0.1, 0.5, 0.9], device=p.device, dtype=p.dtype)
            q = torch.quantile(p, qs, dim=1)  # [3, B, D]
            feats = feats + [F.normalize(q[i], dim=-1, p=2) for i in range(q.shape[0])]
            return torch.cat(feats, dim=-1)  # [B, (4+3) * embed_dim]
        if mode == "concat_normweight":
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            feats = [F.normalize(self.norm(caps[d])[:, 0], dim=-1, p=2) for d in self.CLS_DEPTHS]
            p = self(x)["x_norm_patchtokens"]  # [B, n_patch, D]
            w = p.norm(dim=-1, keepdim=True)  # [B, n_patch, 1]
            wp = (p * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)  # [B, D]
            feats = feats + [F.normalize(wp, dim=-1, p=2)]
            return torch.cat(feats, dim=-1)  # [B, (4+1) * embed_dim]
        if mode == "concat_gram":
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            feats = [F.normalize(self.norm(caps[d])[:, 0], dim=-1, p=2) for d in self.CLS_DEPTHS]
            p = self(x)["x_norm_patchtokens"]  # [B, n_patch, D]
            D = p.shape[-1]; m = 96
            gen = torch.Generator(device=p.device).manual_seed(0)
            R = torch.randn(D, m, generator=gen, device=p.device, dtype=p.dtype) / (D ** 0.5)
            z = p @ R  # [B, n_patch, m]
            second = (z * z).mean(dim=1)  # [B, m] diagonal of the projected gram
            feats = feats + [F.normalize(second, dim=-1, p=2)]
            return torch.cat(feats, dim=-1)  # [B, (4*embed_dim + 96)]
        if mode == "cls_attn_depth":
            # Build ON TOP of concat_fino: keep the full 4-depth CLS concat (preserves
            # the depth diversity knn/16shot exploit) AND append an input-dependent
            # attention-weighted depth summary (the non-linear cross-depth feature that
            # lifted progression/survival in the collapse-to-2D variant).
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            d_cls = [self.norm(caps[d])[:, 0] for d in self.CLS_DEPTHS]  # each [B, D]
            final_cls = self(x)["x_norm_clstoken"]
            # concat_fino base (L2-normed per depth) -- identical to mode "concat_fino"
            feats = [F.normalize(c, dim=-1, p=2) for c in d_cls]  # 4 x [B, D]
            # cosine sim between each depth CLS and final CLS as attention weight
            sims = torch.stack([F.cosine_similarity(c, final_cls, dim=-1) for c in d_cls], dim=-1)  # [B, K]
            w = F.softmax(sims / 0.1, dim=-1)  # [B, K]
            d_stack = torch.stack(d_cls, dim=1)  # [B, K, D]
            attn_weighted = (d_stack * w.unsqueeze(-1)).sum(dim=1)  # [B, D]
            feats = feats + [F.normalize(attn_weighted, dim=-1, p=2)]  # 5th block
            return torch.cat(feats, dim=-1)  # [B, 5 * D]
        if mode == "cls_depth_stats":
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            d_cls = [self.norm(caps[d])[:, 0] for d in self.CLS_DEPTHS]  # each [B, D]
            d_stack = torch.stack(d_cls, dim=1)  # [B, K, D]
            # per-dimension mean and std across depths
            d_mean = d_stack.mean(dim=1)  # [B, D]
            d_std = d_stack.std(dim=1)    # [B, D]
            return torch.cat([F.normalize(d_mean, dim=-1, p=2), F.normalize(d_std, dim=-1, p=2)], dim=-1)
        if mode == "cls_depth_stats_concat":
            # Build-on-top of cls_depth_stats (trial 1). Trial-1 finding: the depth-MEAN
            # DILUTES the discriminative final-block CLS (linear 0.733 vs concat_fino 0.773),
            # while the cross-depth STD added complementary heterogeneity (survival/knn gains).
            # So KEEP the full concat_fino depth stack [4*D] (preserves classification) and
            # ADD ONLY the per-dim STD across depths [D] (the genuinely new signal). Same
            # self.norm + [:,0] convention as concat_fino -> verbatim-faithful to probe.py.
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            d_cls = [self.norm(caps[d])[:, 0] for d in self.CLS_DEPTHS]  # each [B, D]
            feats = [F.normalize(c, dim=-1, p=2) for c in d_cls]          # concat_fino stack
            d_stack = torch.stack(d_cls, dim=1)  # [B, K, D]
            d_std = d_stack.std(dim=1)           # [B, D] per-dim cross-depth std
            feats = feats + [F.normalize(d_std, dim=-1, p=2)]
            return torch.cat(feats, dim=-1)  # [B, (4+1) * embed_dim]
        if mode == "patch_magnitude":
            # GeM (generalized-mean) pooling of patch tokens, concatenated onto the
            # VALIDATED concat_fino multi-depth CLS base (the "concat_meanpatch family").
            # GeM_p(x) = (mean_i x_i^p)^(1/p): p=1 -> mean, p->inf -> max. p>1 emphasizes the
            # most-activated (salient) patches per channel -- a salience pool that the mean /
            # norm-weight / quantile pools (all already tested, all diluted concat_fino) do NOT
            # capture. Signed power preserves the sign of LayerNorm'd (negative-capable) features;
            # per-image, feed-forward, no eval-set stats -> legal under the eval-fairness rule.
            # Tunables (env): OVFX_GEM_P exponent (default 3), OVFX_GEM_ALPHA scale on the
            # L2-normed pool block to control dilution of the classification heads (default 1),
            # OVFX_GEM_BASE cls base ("concat" multi-depth default | "single" final CLS only).
            gem_p = float(os.environ.get("OVFX_GEM_P", "3.0"))
            gem_alpha = float(os.environ.get("OVFX_GEM_ALPHA", "1.0"))
            gem_base = os.environ.get("OVFX_GEM_BASE", "concat")
            out = self(x)
            p = out["x_norm_patchtokens"]  # [B, n_patch, D]
            sp = p.sign() * p.abs().clamp_min(1e-6).pow(gem_p)  # signed power [B, n_patch, D]
            m = sp.mean(dim=1)  # [B, D]
            gem = m.sign() * m.abs().clamp_min(1e-6).pow(1.0 / gem_p)  # [B, D]
            pool = gem_alpha * F.normalize(gem, dim=-1, p=2)  # [B, D]
            if gem_base == "single":
                cls = F.normalize(out["x_norm_clstoken"], dim=-1, p=2)
                return torch.cat([cls, pool], dim=-1)  # [B, 2 * embed_dim]
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            feats = [F.normalize(self.norm(caps[d])[:, 0], dim=-1, p=2) for d in self.CLS_DEPTHS]
            return torch.cat(feats + [pool], dim=-1)  # [B, (4 + 1) * embed_dim]
        if mode == "reg_mean":
            out = self(x)
            cls = F.normalize(out["x_norm_clstoken"], dim=-1, p=2)
            reg = out["x_norm_regtokens"].mean(dim=1)  # [B, D]
            return torch.cat([cls, F.normalize(reg, dim=-1, p=2)], dim=-1)
        if mode == "concat_reg_mean":
            # Build ON TOP of concat_fino (trial-2). Trial-1 finding (bare reg_mean = [final CLS ||
            # mean register], score 0.6527, delta -0.0036): dropping the multi-depth concat HURT
            # classification (16shot -0.025, linear/knn -0.010 each) but the mean-register block
            # WON survival (+0.0165) and progression (+0.0047). So KEEP the full concat_fino
            # 4-depth CLS stack [4*D] (recovers the classification richness) and APPEND ONLY the
            # L2-normed mean register-token block [D] (the genuinely new slide-level signal).
            # Same self.norm + [:,0] convention as concat_fino -> verbatim-faithful to probe.py;
            # mean over register tokens is per-image, feed-forward, no eval-set stats -> legal.
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            feats = [F.normalize(self.norm(caps[d])[:, 0], dim=-1, p=2) for d in self.CLS_DEPTHS]
            reg = self(x)["x_norm_regtokens"].mean(dim=1)  # [B, D]
            feats = feats + [F.normalize(reg, dim=-1, p=2)]
            return torch.cat(feats, dim=-1)  # [B, (4+1) * embed_dim]
        if mode == "sinkhorn_cls":
            # Build ON TOP of concat_fino: KEEP the 4-depth CLS concat (preserves the
            # classification/knn/16shot richness) AND APPEND a Sinkhorn-Knopp doubly-stochastic
            # centered CLS as a 5th block. Concretely: stack the 4 depth-CLS into [B, K, D] then
            # apply K_ITER Sinkhorn-Knopp iterations alternating col(K)-then-row(B) normalization.
            # After convergence the B x K matrix (per output dim) is doubly-stochastic -- the
            # marginal batch bias is removed, leaving a centered projection. Sum over the K=depth
            # axis gives [B, D]; L2-normed and appended to the 4-depth concat_fino stack.
            # Per-image, feed-forward, no eval-set stats -> legal under the eval-fairness rule.
            # Tunables (env): OVFX_SINKHORN_K iterations (default 5),
            # OVFX_SINKHORN_BASE "concat" (default, build-on-top) | "single" (final-CLS ablated).
            caps = self._forward_intermediates(x, self.CLS_DEPTHS)
            d_cls = [self.norm(caps[d])[:, 0] for d in self.CLS_DEPTHS]  # each [B, D]
            feats = [F.normalize(c, dim=-1, p=2) for c in d_cls]          # concat_fino base (4 blocks)
            sk_k = int(os.environ.get("OVFX_SINKHORN_K", "5"))
            sk_base = os.environ.get("OVFX_SINKHORN_BASE", "concat")
            if sk_base == "single":
                # Ablation: use only the final CLS as the Sinkhorn input (1-iter centering is
                # trivial; this path is only used for sanity checks. Default path is "concat").
                sinkhorn_out = self(x)["x_norm_clstoken"]
            else:
                d_stack = torch.stack(d_cls, dim=1)  # [B, K, D]
                # Sinkhorn-Knopp across the K=depth axis: alternate col-norm (K) and row-norm (B).
                for _ in range(sk_k):
                    d_stack = d_stack / d_stack.sum(dim=1, keepdim=True).clamp_min(1e-12)  # col-norm (K)
                    d_stack = d_stack / d_stack.sum(dim=0, keepdim=True).clamp_min(1e-12)  # row-norm (B)
                sinkhorn_out = d_stack.sum(dim=1)  # [B, D] — marginal-preserving projection
            feats = feats + [F.normalize(sinkhorn_out, dim=-1, p=2)]
            return torch.cat(feats, dim=-1)  # [B, 5 * embed_dim]
        caps = self._forward_intermediates(x, self.CLS_DEPTHS)
        feats = [F.normalize(self.norm(caps[d])[:, 0], dim=-1, p=2) for d in self.CLS_DEPTHS]
        if mode == "concat_meanpatch":
            mp = F.normalize(self(x)["x_norm_patchtokens"].mean(dim=1), dim=-1, p=2)
            feats = feats + [mp]
        return torch.cat(feats, dim=-1)  # [B, k * embed_dim]

    # SEGMENTATION / dense readout used by the seg probes via encode_image.
    # 1. Concatenate the LAST FOUR normalized block patch-token outputs.
    # 2. KEEP the register tokens (returned ahead of the patch tokens, matching
    #    the [registers || patches] contract probe.py slices with [:, registers:]).
    # 3. Reshape patch tokens to a denser 32x32 grid (densify the native 16x16
    #    patch grid via the JBU upsample below).
    # 4. Apply a lightweight JBU-style (joint bilateral upsampling) image-guided
    #    upsample/sharpen, guide = the input image. No eval-set statistics.
    SEG_DEPTHS = (8, 9, 10, 11)  # 0-indexed last four blocks
    SEG_GRID = 32

    def encode_image(self, x, checkpoint=False):
        # `checkpoint` is accepted for call-site compatibility; the dense readout
        # is eval-only (no_grad in probe.py) so activation checkpointing is moot.
        # OVFX_SEG selects the segmentation readout (pre-probe baseline sweep).
        # Unset/"jbu_last4" = the current default (JBU on last-4-block concat).
        #   plain     -> [registers || patch tokens], native grid, NO JBU
        #   jbu_last1 -> JBU densify of the LAST block only (no depth concat)
        #   jbu_last3 -> JBU densify of the last-3-block patch concat
        #   jbu_last4 -> JBU densify of the last-4-block patch concat (default)
        seg_mode = os.environ.get("OVFX_SEG", "jbu_last4")
        if seg_mode == "plain":
            out = self(x, checkpoint=checkpoint)
            return torch.cat([out["x_norm_regtokens"], out["x_norm_patchtokens"]], dim=1)
        if seg_mode == "seg_early_blocks":
            depth = len(self.blocks)
            # early blocks at 0-indexed indices (0,2) = old (1,3) 1-indexed + last 4 blocks (8,9,10,11)
            early = tuple(k for k in (0, 2) if k < depth)
            late = tuple(depth - 1 - k for k in (3, 2, 1, 0))
            all_depths = early + late
            caps = self._forward_intermediates(x, all_depths)
            reg = self.registers
            patch_parts, reg_parts = [], []
            for d in all_depths:
                normed = self.norm(caps[d])
                reg_parts.append(normed[:, 1 : 1 + reg])
                patch_parts.append(normed[:, 1 + reg :])
            patches = torch.cat(patch_parts, dim=-1)
            regs = torch.cat(reg_parts, dim=-1)
            B, n_patch, C = patches.shape
            g = int(round(n_patch ** 0.5))
            grid = patches.transpose(1, 2).reshape(B, C, g, g)
            dense = self._jbu_upsample(grid, x, self.SEG_GRID)
            dense = dense.flatten(2).transpose(1, 2)
            return torch.cat([regs, dense], dim=1)
        if seg_mode == "seg_jbu_last3":
            depth = len(self.blocks)
            last_three = tuple(depth - 1 - k for k in (2, 1, 0))
            caps = self._forward_intermediates(x, last_three)
            reg = self.registers
            patch_parts, reg_parts = [], []
            for d in last_three:
                normed = self.norm(caps[d])
                reg_parts.append(normed[:, 1 : 1 + reg])
                patch_parts.append(normed[:, 1 + reg :])
            patches = torch.cat(patch_parts, dim=-1)
            regs = torch.cat(reg_parts, dim=-1)
            B, n_patch, C = patches.shape
            g = int(round(n_patch ** 0.5))
            grid = patches.transpose(1, 2).reshape(B, C, g, g)
            dense = self._jbu_upsample(grid, x, self.SEG_GRID)
            dense = dense.flatten(2).transpose(1, 2)
            return torch.cat([regs, dense], dim=1)
        if seg_mode == "seg_jbu_last2":
            depth = len(self.blocks)
            last_two = tuple(depth - 1 - k for k in (1, 0))
            caps = self._forward_intermediates(x, last_two)
            reg = self.registers
            patch_parts, reg_parts = [], []
            for d in last_two:
                normed = self.norm(caps[d])
                reg_parts.append(normed[:, 1 : 1 + reg])
                patch_parts.append(normed[:, 1 + reg :])
            patches = torch.cat(patch_parts, dim=-1)
            regs = torch.cat(reg_parts, dim=-1)
            B, n_patch, C = patches.shape
            g = int(round(n_patch ** 0.5))
            grid = patches.transpose(1, 2).reshape(B, C, g, g)
            dense = self._jbu_upsample(grid, x, self.SEG_GRID)
            dense = dense.flatten(2).transpose(1, 2)
            return torch.cat([regs, dense], dim=1)
        depth = len(self.blocks)
        # 0-indexed: last block = depth-1, last four = depth-4 .. depth-1
        last_four = (depth - 1,) if seg_mode == "jbu_last1" else tuple(depth - 1 - k for k in (3, 2, 1, 0))
        caps = self._forward_intermediates(x, last_four)
        # Per-depth final-LayerNorm of each captured block, then concat patch
        # tokens over depth -> [B, n_patch, 4 * embed_dim]. Registers handled below.
        reg = self.registers
        patch_per_depth, reg_per_depth = [], []
        for d in last_four:
            normed = self.norm(caps[d])
            reg_per_depth.append(normed[:, 1 : 1 + reg])
            patch_per_depth.append(normed[:, 1 + reg :])
        patches = torch.cat(patch_per_depth, dim=-1)   # [B, n_patch, 4C]
        regs = torch.cat(reg_per_depth, dim=-1)        # [B, registers, 4C]

        # Native patch grid (square) -> dense 32x32 grid via image-guided JBU.
        B, n_patch, C = patches.shape
        g = int(round(n_patch ** 0.5))
        assert g * g == n_patch, f"patch count {n_patch} is not square"
        grid = patches.transpose(1, 2).reshape(B, C, g, g)  # [B, 4C, g, g]
        dense = self._jbu_upsample(grid, x, self.SEG_GRID)  # [B, 4C, 32, 32]
        dense = dense.flatten(2).transpose(1, 2)            # [B, 32*32, 4C] = [B, 1024, 4C]

        # [registers || dense-patches]; probe.py drops the first `registers`
        # tokens, leaving 1024 = 32*32 patch tokens for the seg head's gs=32 reshape.
        return torch.cat([regs, dense], dim=1)

    # Lightweight JBU (Joint Bilateral Upsampling) variant. Bilinearly upsample
    # the low-res patch-feature grid to the target grid, then run ONE guided
    # pass: the guide is the input image downsampled to the target grid, and each
    # output cell is a fixed-3x3-neighborhood blend of the upsampled features
    # weighted by (a) a fixed spatial Gaussian and (b) a range Gaussian on guide-
    # intensity similarity. Sigmas are fixed constants -> no eval-set statistics,
    # the guide is purely the per-image input. Bounded: single pass, 3x3 kernel.
    SIGMA_SPATIAL = 1.0
    SIGMA_RANGE = 0.1

    def _jbu_upsample(self, feat_grid, image, out_size):
        B, C, _, _ = feat_grid.shape
        # (1) bilinear upsample features to the dense target grid.
        feat_up = F.interpolate(feat_grid, size=(out_size, out_size), mode="bilinear", align_corners=False)
        # (2) build the guide: input image -> grayscale -> downsample to target grid.
        guide = image.mean(dim=1, keepdim=True)  # [B,1,H,W] per-image, no dataset stats
        guide = F.interpolate(guide, size=(out_size, out_size), mode="bilinear", align_corners=False)  # [B,1,32,32]

        # Fixed 3x3 spatial weights (Gaussian over a unit-spaced grid).
        offs = [-1, 0, 1]
        device, dtype = feat_up.device, feat_up.dtype
        spatial = torch.tensor(
            [[(dy * dy + dx * dx) for dx in offs] for dy in offs],
            device=device, dtype=torch.float32,
        )
        spatial = torch.exp(-spatial / (2.0 * self.SIGMA_SPATIAL ** 2))  # [3,3]

        # (3) one guided-filter pass: accumulate neighbor features weighted by
        # spatial * range(guide-intensity similarity). Reflect-pad so border
        # cells keep full 3x3 support without dataset-dependent fill.
        g_pad = F.pad(guide, (1, 1, 1, 1), mode="reflect")     # [B,1,34,34]
        f_pad = F.pad(feat_up, (1, 1, 1, 1), mode="reflect")   # [B,C,34,34]
        out = torch.zeros_like(feat_up)
        wsum = torch.zeros(B, 1, out_size, out_size, device=device, dtype=dtype)
        sr2 = 2.0 * self.SIGMA_RANGE ** 2
        for iy, dy in enumerate(offs):
            for ix, dx in enumerate(offs):
                g_shift = g_pad[:, :, 1 + dy : 1 + dy + out_size, 1 + dx : 1 + dx + out_size]
                f_shift = f_pad[:, :, 1 + dy : 1 + dy + out_size, 1 + dx : 1 + dx + out_size]
                rng = torch.exp(-((g_shift - guide) ** 2) / sr2)         # [B,1,32,32]
                w = (spatial[iy, ix].to(dtype) * rng)                    # [B,1,32,32]
                out = out + w * f_shift
                wsum = wsum + w
        return out / wsum.clamp_min(1e-6)


# Strict-load Meta's pretrained weights for the model's declared variant.
# Strict matches our key layout against Meta's; any drift fails loudly per AGENTS.md.
def load_dinov2_pretrained(model):
    *_, url = DINOV2_VARIANTS[model.variant]
    state = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


# DINO projection head: 3-layer MLP (in -> hidden -> hidden -> bottleneck) + L2 norm +
# weight-normed Linear(bottleneck -> n_prototypes) with weight_g frozen at 1, matching the
# behaviour of dinov2.layers.DINOHead. Standalone reimplementation (no xformers, no fvcore).
class DINOHead(nn.Module):
    def __init__(self, in_dim, n_prototypes, hidden_dim=2048, bottleneck_dim=384, nlayers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, n_prototypes, bias=False))
        # weight-norm under torch.nn.utils.parametrizations exposes `parametrizations.weight.original0/1`;
        # original0 is the magnitude vector (size n_prototypes). Freeze it at 1 to match dinov2's recipe.
        with torch.no_grad():
            self.last_layer.parametrizations.weight.original0.fill_(1.0)
        self.last_layer.parametrizations.weight.original0.requires_grad_(False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


# I-JEPA predictor head: regresses EMA-teacher patch representations at masked target blocks from the student's
# block-masked patch tokens. FINO/JEPA-T option: n_cond>0 adds a learned per-class embedding (idx 0 = missing/-1)
# of a discrete metadata factor to every patch token, so the latent-regression target is metadata-aware
# (a dense-path alternative to CLS-token steering). n_cond=0 is plain I-JEPA.
class JEPAPredictor(nn.Module):
    def __init__(self, dim, depth=4, width=0, heads=6, n_cond=0):
        super().__init__()
        width = width or dim
        self.proj_in = nn.Linear(dim, width) if width != dim else nn.Identity()
        self.cond_emb = nn.Embedding(n_cond + 1, width) if n_cond else None
        self.blocks = nn.ModuleList(Block(width, heads, 4.0, 0.0) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.proj = nn.Linear(width, dim, bias=True)

    def forward(self, patch_tokens, cond=None):
        x = self.proj_in(patch_tokens)
        if self.cond_emb is not None and cond is not None:
            x = x + self.cond_emb(cond + 1).unsqueeze(1)  # broadcast factor embedding over patches; cond=-1 -> idx 0
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.norm(x))

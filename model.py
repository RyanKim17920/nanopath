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

    # EVAL-TIME-ONLY dense readout for the segmentation probe (exp_0114).
    # `encode_image` is never called in train.py (the trunk + the trained checkpoint are
    # untouched by this method) and is hit at exactly ONE probe-time callsite
    # (probe.py: `model.encode_image(...)[:, model.registers:]` -> seg MaskTransformer).
    # `probe_features` (CLS) is unchanged, so the 7 CLS-pooled probes stay byte-identical.
    #
    # Canonical DINOv2/DPT dense-readout recipe: instead of feeding only the last block's
    # patch tokens to the frozen-trunk seg decoder, FUSE the last `n_fuse` block outputs.
    # Each collected layer is passed through the trained final LayerNorm `self.norm` (the
    # `get_intermediate_layers(norm=True)` convention) so every fused layer is in the same
    # normalized scale the seg head's `proj_dec` expects; we then concat on the CHANNEL dim.
    # Token axis (=[cls, regs, patches]) is preserved, so probe.py's `[:, registers:]` slice
    # still drops the register tokens correctly. d_encoder widens 384 -> 384*n_fuse; the
    # locked seg MaskTransformer auto-adapts via `d_encoder=train_feats.shape[-1]`.
    # Shallower blocks carry finer spatial / boundary detail (lost after the last block's
    # global mixing) -> pure seg upside with zero seg/CLS trunk lockstep.
    DENSE_FUSE_LAYERS = 4

    # exp_0176 (port of exp_0167): EVAL-ONLY image-guided feature DENSIFIER on top of the
    # exp_0114 fusion. The bottleneck for dense seg (worst probe @0.289) is the COARSE 16x16
    # patch grid, not which blocks are read. We densify the FROZEN feature map itself (no trunk
    # retrain, no input-upsample-through-patch-embed -- that path regressed CLS via seed variance
    # in exp_0038) with a parameter-free Joint Bilateral Upsampler (JBU / FeatUp-style): the
    # 16x16 fused feature grid is upsampled to UPSAMPLE_GRID x UPSAMPLE_GRID using bilateral
    # weights = spatial Gaussian x range Gaussian on the INPUT IMAGE's high-res structure.
    # This injects real boundary/edge detail from the guidance image into the feature map and
    # denoises across patch boundaries, instead of the information-free bilinear interp the seg
    # head already does post-decode. Pure eval-only: trunk runs ONCE at native resolution, CLS
    # path (probe_features) untouched, so the 7 CLS probes are byte-identical. The locked
    # MaskTransformer auto-adapts: gs=int(n**0.5) reads the new 32x32 grid and d_encoder reads
    # the channel dim. Token axis stays [regs || patches] for probe.py's `[:, registers:]`.
    UPSAMPLE_GRID = 32          # target dense grid (16 -> 32, the seg head infers gs from n)
    JBU_SIGMA_RANGE = 0.1       # guidance-intensity range bandwidth (image normalized to ~[0,1])

    def _jbu_upsample(self, feat_hw, guide_lr, guide_hr):
        # feat_hw: [B,C,h,w] source feature map; guide_lr/guide_hr: [B,1,h,w]/[B,1,H,W] grayscale
        # guidance at source / target resolution. Returns [B,C,H,W] image-guided dense features.
        B, C, h, w = feat_hw.shape
        H = Wt = self.UPSAMPLE_GRID
        # Bilinear feature prior (spatial smoothness term) at the target grid.
        up = F.interpolate(feat_hw, size=(H, Wt), mode="bilinear", align_corners=False)
        # Range term: for each target cell, the guidance intensity gap to the nearest-source cell
        # it was drawn from. Nearest-up the source guidance to the target grid, compare to the
        # true high-res guidance; small gap (same tissue) -> trust the feature, large gap (across
        # a boundary) -> sharpen toward the high-res structure by blending in the high-res guide's
        # local detail. Parameter-free Gaussian on the gap.
        guide_src_up = F.interpolate(guide_lr, size=(H, Wt), mode="nearest")
        gap = (guide_hr - guide_src_up).abs()
        w_range = torch.exp(-(gap ** 2) / (2.0 * self.JBU_SIGMA_RANGE ** 2))  # [B,1,H,W] in (0,1]
        # High-frequency feature detail recovered by sharpening the bilinear prior where the
        # guidance says there is a boundary: detail = bilinear_up - blur(bilinear_up); add it
        # back weighted by (1 - w_range) so boundaries get sharpened, flat regions stay smooth.
        blur = F.avg_pool2d(F.pad(up, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
        detail = up - blur
        return up + (1.0 - w_range) * detail

    def encode_image(self, x, checkpoint=False):
        n_fuse = min(self.DENSE_FUSE_LAYERS, len(self.blocks))
        take_from = len(self.blocks) - n_fuse  # indices [take_from, depth) inclusive of last
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        # Grayscale guidance from the input image, normalized to ~[0,1] for the range bandwidth.
        guide = x.mean(dim=1, keepdim=True)
        guide = (guide - guide.amin(dim=(2, 3), keepdim=True)) / (
            guide.amax(dim=(2, 3), keepdim=True) - guide.amin(dim=(2, 3), keepdim=True) + 1e-6)
        guide_lr = F.interpolate(guide, size=(h, w), mode="area")  # source-grid guidance
        guide_hr = F.interpolate(guide, size=(self.UPSAMPLE_GRID, self.UPSAMPLE_GRID), mode="area")
        xt = self._prepare_tokens(x)
        collected = []
        for i, blk in enumerate(self.blocks):
            if checkpoint and self.training:
                xt = torch.utils.checkpoint.checkpoint(blk, xt, use_reentrant=False)
            else:
                xt = blk(xt)
            if i >= take_from:
                # Norm a clone so the running trunk activation is untouched across layers.
                xn = self.norm(xt)
                collected.append(xn[:, 1:])  # drop cls; keep [regs || patches]
        # Channel-concat the fused layers (shallow -> deep): [B, regs+h*w, C], C = 384*n_fuse.
        fused = torch.cat(collected, dim=-1)
        regs = fused[:, : self.registers]                 # register tokens (non-spatial, kept as-is)
        patches = fused[:, self.registers :]              # [B, h*w, C] spatial tokens
        C = patches.shape[-1]
        feat_hw = patches.transpose(1, 2).reshape(B, C, h, w)
        dense = self._jbu_upsample(feat_hw.float(), guide_lr.float(), guide_hr.float())
        dense = dense.flatten(2).transpose(1, 2).to(fused.dtype)  # [B, UPSAMPLE_GRID^2, C]
        # Token axis stays [regs || dense patches] so probe.py's `[:, registers:]` slice is correct.
        return torch.cat([regs, dense], dim=1)

    # exp_0179: EVAL-ONLY multi-depth CLS readout for the classification probes
    # (linear/knn/16shot -- and, structurally, mutation/survival, which the locked probe.py
    # also routes through probe_features). The trunk + checkpoint are UNTOUCHED (this is hit
    # only at probe time; train.py never calls probe_features), and encode_image / the seg
    # densifier path is byte-identical -> segmentation stays at exp_0176's 0.33090.
    #
    # Bottleneck: the live mover for CLS classification is the EVAL-SIDE readout geometry, not
    # the weights. The known ceiling is the val_dino back-75% late-rise -- the DINO head sits on
    # the FINAL CLS, so the DINO/CLS overfit corruption concentrates in the last blocks. The
    # final-block normed CLS alone is what gives exp_0176's best-yet linear=0.7694; we AUGMENT it
    # with an INTERMEDIATE-depth normed CLS taken BEFORE the heavy overfit band, recovering linear
    # discriminability the late overfit erased. Each collected CLS is passed through the trained
    # final LayerNorm `self.norm` (the get_intermediate_layers(norm=True) convention) so both
    # depths sit in the same normalized scale before channel-concat.
    #
    # Depth selection: {block 8, block 11}. Block 11 (final) preserves the current best signal;
    # block 8 (2/3 through the 12-block trunk) is the deepest PRE-final-specialization layer --
    # still highly discriminative but not yet collapsed onto the DINO prototypes. SPARSE 2-depth
    # (one final + one intermediate) deliberately AVOIDS exp_0048's blind last-4 stack
    # (blocks 8-11 -> progression -0.0479): we EXCLUDE the corrupted blocks 9,10 band rather than
    # stacking it. Follows exp_0164's intermediate-depth-selection precedent (applied there to the
    # dense readout; applied here to CLS). Feature dim widens 384 -> 768; the locked linear/knn/
    # 16shot probes read e.shape[1] dynamically and auto-adapt.
    CLS_READOUT_DEPTHS = (8, 11)

    def probe_features(self, x):
        depths = set(d % len(self.blocks) for d in self.CLS_READOUT_DEPTHS)
        last = len(self.blocks) - 1
        xt = self._prepare_tokens(x)
        collected = []
        for i, blk in enumerate(self.blocks):
            xt = blk(xt)
            if i in depths:
                # Norm a clone so the running trunk activation is untouched across layers.
                collected.append(self.norm(xt)[:, 0])
        if last not in depths:  # safety: always keep the final-block CLS signal
            collected.append(self.norm(xt)[:, 0])
        return torch.cat(collected, dim=-1)


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

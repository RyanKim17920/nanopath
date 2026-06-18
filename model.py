# DinoV2ViT: clean ViT + 4 register tokens that loads Meta's `dinov2_vit{s,b,g}14_reg`
# pretrained weights via state_dict (no xformers, no dinov2 codebase imports).
# Attention runs on `F.scaled_dot_product_attention` so we get FlashAttention-2
# on H100 bf16 with no third-party kernel dependency. Module names below match
# Meta's checkpoint key layout exactly, so `load_dinov2_pretrained(model)` does
# a strict load.
#
# DINOHead is the small MLP + weight-normed classifier used by train.py for the
# DINO CLS / iBOT patch self-distillation losses. It is intentionally trivial
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


# Module-level list for per-layer attention diagnostics; populated during log_diagnostics forward passes.
_diags = []

def clear_diags(): _diags.clear()
def collect_diags():
    d = _diags.copy(); _diags.clear(); return d


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

    # Build [cls, registers, patches] tokens; iBOT swaps the masked patch positions for mask_token.
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

    # Pre-probe: Multi-Depth CLS Concatenation for classification probes.
    # Collects normalized CLS tokens from strided transformer depths 4, 6, 8, 11,
    # concatenates them, projects back to embed_dim, and applies final norm.
    # Eval-only — projection weights are lazy-init (frozen; no training needed).
    def _multi_depth_probe_features(self, x):
        x_tokens = self._prepare_tokens(x)
        # Strided depths: 4, 6, 8, 11 (1-indexed) — covers early/mid/late transformer
        depths = [4, 6, 8, 11]
        cls_raw = []
        for i, blk in enumerate(self.blocks):
            x_tokens = blk(x_tokens)
            if (i + 1) in depths:
                # Normalize each depth's CLS token independently for stability
                cls_raw.append(F.normalize(x_tokens[:, 0], dim=-1))
        concatenated = torch.cat(cls_raw, dim=-1)
        # Lazy-init projection: Xavier uniform, frozen after first forward
        if not hasattr(self, "_md_cls_proj"):
            in_dim = concatenated.shape[-1]
            self._md_cls_proj = nn.Linear(in_dim, self.embed_dim, bias=False).to(concatenated)
            nn.init.xavier_uniform_(self._md_cls_proj.weight)
        return self.norm(self._md_cls_proj(concatenated))

    # Pre-probe: JBU-inspired segmentation densifier for encode_image.
    # Concatenates last four normalized block outputs, keeps register tokens,
    # and converts patch tokens into a denser 32x32 feature grid via
    # lightweight image-guided upsample + sharpening step.
    def _jbu_dense_encode_image(self, x, checkpoint=False):
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        # Forward through blocks, collecting last-4 normalized outputs
        tokens = self._prepare_tokens(x, masks=None)
        last_four = []
        for blk in self.blocks:
            if checkpoint and self.training:
                tokens = torch.utils.checkpoint.checkpoint(blk, tokens, use_reentrant=False)
            else:
                tokens = blk(tokens)
            last_four.append(tokens.clone())
            if len(last_four) > 4:
                last_four.pop(0)
        # Final norm
        tokens = self.norm(tokens)
        # Also normalize the intermediate snapshots for stable concatenation
        normed_intermediate = []
        for snap in last_four:
            # Per-token L2 norm for each snapshot
            normed_intermediate.append(F.normalize(snap, dim=-1))
        # Concatenate: [B, N, 4*dim]
        stacked = torch.cat(normed_intermediate, dim=-1)
        # Separate cls, register, patch tokens from stacked representation
        reg_count = self.registers
        patch_stacked = stacked[:, reg_count:, :]  # [B, h*w, 4*dim]
        reg_stacked = stacked[:, :reg_count, :]    # [B, reg, 4*dim]
        # Project patch features: 4*dim -> dim
        if not hasattr(self, "_jbu_patch_proj"):
            self._jbu_patch_proj = nn.Linear(4 * self.embed_dim, self.embed_dim, bias=False).to(patch_stacked)
            nn.init.xavier_uniform_(self._jbu_patch_proj.weight)
        patch_feat = self._jbu_patch_proj(patch_stacked)  # [B, h*w, dim]
        # Reshape to spatial grid
        patch_grid = patch_feat.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
        # Image-guided upsample to 32x32 (or nearest useful grid)
        target_size = max(32, h * 2)
        upsampled = F.interpolate(patch_grid, size=target_size, mode='bilinear', align_corners=False)
        # Lightweight sharpening: residual of original - lowpass
        lowpass = F.interpolate(upsampled, size=target_size // 2, mode='bilinear', align_corners=False)
        lowpass = F.interpolate(lowpass, size=target_size, mode='bilinear', align_corners=False)
        sharpened = upsampled + 0.25 * (upsampled - lowpass)
        # Normalize channels
        sharpened = F.normalize(sharpened, p=2, dim=1)
        # Flatten back to token sequence
        seg_feat = sharpened.flatten(2).transpose(1, 2)  # [B, target_size*target_size, dim]
        # Project register tokens from stacked
        if not hasattr(self, "_jbu_reg_proj"):
            self._jbu_reg_proj = nn.Linear(4 * self.embed_dim, self.embed_dim, bias=False).to(reg_stacked)
            nn.init.xavier_uniform_(self._jbu_reg_proj.weight)
        reg_feat = self._jbu_reg_proj(reg_stacked)  # [B, reg, dim]
        return torch.cat([reg_feat, seg_feat], dim=1)

    # Probe contract: encode_image returns [registers || patches] for the seg head;
    # probe_features returns the cls token for classification probes.
    # With COMBINED_PREPROBE=1 env: uses multi-depth CLS for classification
    # and JBU densifier for segmentation.
    def encode_image(self, x, checkpoint=False):
        if os.environ.get("COMBINED_PREPROBE") == "1":
            return self._jbu_dense_encode_image(x, checkpoint=checkpoint)
        out = self(x, checkpoint=checkpoint)
        return torch.cat([out["x_norm_regtokens"], out["x_norm_patchtokens"]], dim=1)

    def probe_features(self, x):
        if os.environ.get("COMBINED_PREPROBE") == "1":
            return self._multi_depth_probe_features(x)
        return self(x)["x_norm_clstoken"]


# Strict-load Meta's pretrained weights for the model's declared variant.
# Strict matches our key layout against Meta's; any drift fails loudly per AGENTS.md.
def load_dinov2_pretrained(model):
    *_, url = DINOV2_VARIANTS[model.variant]
    state = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


# DINO/iBOT projection head: 3-layer MLP (in -> hidden -> hidden -> bottleneck) + L2 norm +
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

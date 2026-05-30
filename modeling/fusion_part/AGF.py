import torch
import torch.nn as nn

from ..backbones.vit_pytorch import DropPath
from ..backbones.vit_pytorch import Mlp
from ..backbones.vit_pytorch import trunc_normal_


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = y.shape
        q = self.q_(x).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(self.normy(y)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GatedCrossAttn(nn.Module):
    """
    Cross-attention block with the first residual replaced by an adaptive
    gated convex combination (HTL paper §3.4):

        delta = MHA(LN(x), y)
        lam   = sigmoid(Linear([x; delta]))      # per-channel, [B, D]
        x     = (1 - lam) * x + lam * delta
        x     = x + drop_path(MLP(LN(x)))

    The MLP/norm2 residual is kept as a standard pre-norm residual; the
    paper does not specify it but removing it collapses the block into a
    pure linear convex combination with no non-linear capacity.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.gate = nn.Linear(2 * dim, dim)
        # bias = 0 → sigmoid(0) = 0.5: balanced start, symmetric gradient through
        # both x and delta. Weight stays at PyTorch default (kaiming uniform).
        nn.init.zeros_(self.gate.bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        delta = self.attn(self.norm1(x), y)
        lam = torch.sigmoid(self.gate(torch.cat([x, delta], dim=-1)))
        x = (1.0 - lam) * x + lam * delta
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AGF(nn.Module):
    """
    Quality-aware graph fusion for nighttime RGB/NIR/TIR ReID.

    Each modality is a graph node. For every target node, a shared gated
    cross-attention reads both other modalities, while a small edge gate and
    the per-modality quality scores determine which neighbor should dominate.
    The final [B, 3D] descriptor preserves the original AGF output contract.
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.cross = GatedCrossAttn(dim, num_heads, mlp_ratio=4., qkv_bias=False,
                                    qk_scale=None, drop=0., attn_drop=0.,
                                    drop_path=0., act_layer=nn.GELU,
                                    norm_layer=nn.LayerNorm)
        self.self_aggr = GatedCrossAttn(dim, num_heads, mlp_ratio=4., qkv_bias=False,
                                        qk_scale=None, drop=0., attn_drop=0.,
                                        drop_path=0., act_layer=nn.GELU,
                                        norm_layer=nn.LayerNorm)
        hidden = max(dim // 4, 64)
        self.edge_gate = nn.Sequential(
            nn.LayerNorm(2 * dim + 2),
            nn.Linear(2 * dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.edge_gate[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _quality_tensor(x, quality_scores):
        if quality_scores is None:
            return torch.ones(x.size(0), 3, device=x.device, dtype=x.dtype)
        return quality_scores.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def _edge_weights(self, cls_tokens, quality, target_idx):
        src_indices = [i for i in range(3) if i != target_idx]
        target = cls_tokens[target_idx]
        q_t = quality[:, target_idx:target_idx + 1]
        logits = []
        valid = []
        for src_idx in src_indices:
            source = cls_tokens[src_idx]
            q_s = quality[:, src_idx:src_idx + 1]
            edge_in = torch.cat([target, source, q_t, q_s], dim=-1)
            logit = self.edge_gate(edge_in).squeeze(-1)
            logit = logit + torch.log(q_s.squeeze(-1).clamp_min(1e-6))
            logits.append(logit)
            valid.append(q_s.squeeze(-1) > 0)

        logits = torch.stack(logits, dim=1)
        valid = torch.stack(valid, dim=1)
        very_neg = torch.finfo(logits.dtype).min
        logits = torch.where(valid, logits, torch.full_like(logits, very_neg))
        all_invalid = ~valid.any(dim=1, keepdim=True)
        logits = torch.where(all_invalid, torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1) * valid.to(logits.dtype)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        return src_indices, weights

    def _fuse_node(self, feats, cls_tokens, quality, target_idx):
        target_feat = feats[target_idx]
        target_cls = cls_tokens[target_idx]
        self_cls = self.self_aggr(target_cls, target_feat[:, 1:, :])
        src_indices, weights = self._edge_weights(cls_tokens, quality, target_idx)

        cross_cls = torch.zeros_like(self_cls)
        other_quality = torch.zeros_like(quality[:, target_idx])
        valid_count = torch.zeros_like(quality[:, target_idx])
        for pos, src_idx in enumerate(src_indices):
            src_feat = feats[src_idx]
            candidate = self.cross(target_cls, src_feat[:, 1:, :])
            w = weights[:, pos:pos + 1]
            cross_cls = cross_cls + w * candidate
            is_valid = (quality[:, src_idx] > 0).to(quality.dtype)
            other_quality = other_quality + quality[:, src_idx] * is_valid
            valid_count = valid_count + is_valid

        other_quality = other_quality / valid_count.clamp_min(1.0)
        q_t = quality[:, target_idx]
        cross_mix = other_quality / (q_t + other_quality + 1e-6)
        fused = (1.0 - cross_mix.unsqueeze(-1)) * self_cls + cross_mix.unsqueeze(-1) * cross_cls
        return fused * (q_t > 0).to(fused.dtype).unsqueeze(-1)

    def forward(self, x, y, z, quality_scores=None):
        feats = [x, y, z]
        cls_tokens = [feat[:, 0, :] for feat in feats]
        quality = self._quality_tensor(x, quality_scores)
        fused = [
            self._fuse_node(feats, cls_tokens, quality, target_idx=i)
            for i in range(3)
        ]
        return torch.cat(fused, dim=-1)

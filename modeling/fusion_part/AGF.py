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


class AGFBlock(nn.Module):
    """
    One rotation block. mode=0: each modality's cls token attends to a
    neighbor's patches (cross-modal exchange). mode=1: each modality's cls
    attends to its own patches (self-aggregation), and the three cls tokens
    are concatenated to form the fused representation.

    All three calls within a block share a single GatedCrossAttn — the paper
    does not require modality-specific gating, and sharing keeps the
    parameter count aligned with the previous TPM design.
    """

    def __init__(self, dim, num_heads, mode=0):
        super().__init__()
        self.gca = GatedCrossAttn(dim, num_heads, mlp_ratio=4., qkv_bias=False,
                                  qk_scale=None, drop=0., attn_drop=0.,
                                  drop_path=0., act_layer=nn.GELU,
                                  norm_layer=nn.LayerNorm)
        self.mode = mode

    def forward(self, x, y, z):
        if self.mode == 0:
            x_cls = self.gca(x[:, 0, :], y[:, 1:, :])
            y_cls = self.gca(y[:, 0, :], z[:, 1:, :])
            z_cls = self.gca(z[:, 0, :], x[:, 1:, :])
            x = torch.cat([x_cls.unsqueeze(1), x[:, 1:, :]], dim=-2)
            y = torch.cat([y_cls.unsqueeze(1), y[:, 1:, :]], dim=-2)
            z = torch.cat([z_cls.unsqueeze(1), z[:, 1:, :]], dim=-2)
            return x, y, z
        else:
            x_cls = self.gca(x[:, 0, :], x[:, 1:, :])
            y_cls = self.gca(y[:, 0, :], y[:, 1:, :])
            z_cls = self.gca(z[:, 0, :], z[:, 1:, :])
            cls = torch.cat([x_cls, y_cls, z_cls], dim=-1)
            return cls


class AGF(nn.Module):
    """
    Adaptive Gated Fusion (AGF). Three rotation blocks with independent
    weights stage cross-modal exchange, followed by a self-aggregation that
    yields the final fused [B, 3D] cls representation. Replaces TPM; the
    only architectural difference is the gated convex-combination residual
    inside each cross-attn (HTL paper §3.4).
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.start = AGFBlock(dim, num_heads)
        self.middle = AGFBlock(dim, num_heads)
        self.end = AGFBlock(dim, num_heads, mode=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y, z):
        x, y, z = self.start(x=x, y=y, z=z)
        x, z, y = self.middle(x=x, y=z, z=y)
        cls = self.end(x=x, y=y, z=z)
        return cls

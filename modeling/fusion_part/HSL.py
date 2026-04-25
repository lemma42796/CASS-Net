import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d


class WTransform2d(nn.Module):
    """Cholesky decomposition whitening for 4D input (B, C, H, W)."""

    def __init__(self, num_features, group_size=1, momentum=0.1, eps=1e-3):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.group_size = min(num_features, group_size)
        self.num_groups = num_features // self.group_size
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_variance', torch.eye(self.group_size).unsqueeze(0).repeat(self.num_groups, 1, 1))

    def forward(self, x):
        assert x.dim() == 4, f'expected 4D input, got {x.dim()}D'
        assert self.num_features % self.group_size == 0

        m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)
        if not self.training:
            m = self.running_mean
        xn = x - m

        T = xn.permute(1, 0, 2, 3).contiguous().view(self.num_groups, self.group_size, -1)
        f_cov = torch.bmm(T, T.permute(0, 2, 1)) / T.shape[-1]
        eye = torch.eye(self.group_size, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(self.num_groups, 1, 1)
        f_cov_shrinked = (1 - self.eps) * f_cov + self.eps * eye

        if not self.training:
            f_cov_shrinked = (1 - self.eps) * self.running_variance + self.eps * eye

        L = torch.linalg.cholesky(f_cov_shrinked)
        inv_sqrt = torch.linalg.inv(L).contiguous().view(self.num_features, self.group_size, 1, 1)
        decorrelated = conv2d(xn, inv_sqrt, groups=self.num_groups)

        if self.training:
            self.running_mean = self.momentum * m.detach() + (1 - self.momentum) * self.running_mean
            self.running_variance = self.momentum * f_cov.detach() + (1 - self.momentum) * self.running_variance

        return decorrelated


class HypergraphConv(nn.Module):
    """Hypergraph convolution for 4D input (B, C, H, W)."""

    def __init__(self, in_features=768, out_features=768,
                 features_height=16, features_width=8,
                 edges=128, filters=128, apply_bias=True, theta1=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.features_height = features_height
        self.features_width = features_width
        self.vertices = features_height * features_width
        self.edges = edges
        self.filters = filters
        self.apply_bias = apply_bias
        self.theta1 = theta1

        self.phi_conv = nn.Conv2d(in_features, filters, kernel_size=1)
        self.A_conv = nn.Conv2d(in_features, filters, kernel_size=1)
        self.M_conv = nn.Conv2d(in_features, edges, kernel_size=7, padding=3)

        self.weight_2 = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_normal_(self.weight_2)

        if apply_bias:
            self.bias_2 = nn.Parameter(torch.empty(1, out_features))
            nn.init.xavier_normal_(self.bias_2)

    def forward(self, x):
        phi = self.phi_conv(x)
        phi = phi.permute(0, 2, 3, 1).contiguous().view(-1, self.vertices, self.filters)

        A = F.avg_pool2d(x, kernel_size=(self.features_height, self.features_width))
        A = self.A_conv(A)                          # [B, filters, 1, 1]
        A = torch.diag_embed(A.squeeze(-1).squeeze(-1))   # [B, filters, filters]

        M = self.M_conv(x)
        M = M.permute(0, 2, 3, 1).contiguous().view(-1, self.vertices, self.edges)

        H = torch.matmul(phi, torch.matmul(A, torch.matmul(phi.transpose(1, 2), M)))
        H = torch.abs(H)

        if self.theta1 != 0.0:
            mean_H = self.theta1 * torch.mean(H, dim=[1, 2], keepdim=True)
            H = torch.where(H < mean_H, 0.0, H)

        D = H.sum(dim=2)
        D_H = torch.mul(torch.unsqueeze(torch.pow(D + 1e-10, -0.5), dim=-1), H)
        B = H.sum(dim=1)
        B = torch.diag_embed(torch.pow(B + 1e-10, -1))

        features = x.permute(0, 2, 3, 1).contiguous().view(-1, self.vertices, self.in_features)

        out = features - torch.matmul(D_H, torch.matmul(B, torch.matmul(D_H.transpose(1, 2), features)))
        out = torch.matmul(out, self.weight_2)

        if self.apply_bias:
            out = out + self.bias_2

        out = out.permute(0, 2, 1).contiguous().view(-1, self.out_features, self.features_height, self.features_width)
        return out


class HSLModule(nn.Module):
    """High-Order Structure Learning module for ViT patch tokens.

    Input:  (B, N, C) patch tokens, N = feat_h * feat_w
    Output: (B, N, C) relation-enhanced patch tokens
    """

    def __init__(self, in_features=768, edges=128, filters=128,
                 feat_h=16, feat_w=8, group_size=1, graphw=1.0, theta1=0.5):
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.graphw = graphw

        self.whiten = WTransform2d(in_features, group_size)
        self.gamma = nn.Parameter(torch.ones(in_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(in_features, 1, 1))

        self.hypergraph = HypergraphConv(
            in_features=in_features, out_features=in_features,
            features_height=feat_h, features_width=feat_w,
            edges=edges, filters=filters, theta1=theta1
        )

    def forward(self, patch_tokens):
        B, N, C = patch_tokens.shape
        x = patch_tokens.permute(0, 2, 1).contiguous().view(B, C, self.feat_h, self.feat_w)

        # Whitening with affine residual
        x_wh = self.whiten(x) * self.gamma + self.beta + x

        # Hypergraph convolution with residual
        x = x + self.graphw * self.hypergraph(x_wh)

        return x.view(B, C, N).permute(0, 2, 1).contiguous()

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_ORDER = ('RGB', 'NIR', 'TIR')
PAIR_ORDER = ((0, 1), (0, 2), (1, 2))


def _largest_divisor_at_most(value, limit):
    limit = max(1, min(value, int(limit)))
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class TokenWhitening2d(nn.Module):
    """Group whitening adapted from HOS-Net's 2D whitening block."""

    def __init__(self, channels, group_size=16, momentum=0.1, eps=1e-3):
        super().__init__()
        group_size = _largest_divisor_at_most(channels, group_size)
        self.channels = channels
        self.group_size = group_size
        self.num_groups = channels // group_size
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer('running_mean', torch.zeros(1, channels, 1, 1))
        self.register_buffer('running_cov', torch.eye(group_size).repeat(self.num_groups, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError('expected {} channels, got {}'.format(self.channels, c))

        orig_dtype = x.dtype
        x_float = x.float()
        if self.training:
            mean = x_float.mean(dim=(0, 2, 3), keepdim=True)
            centered = x_float - mean
            flat = centered.permute(1, 0, 2, 3).contiguous()
            flat = flat.view(self.num_groups, self.group_size, -1)
            cov = torch.bmm(flat, flat.transpose(1, 2)) / max(flat.size(-1), 1)
            with torch.no_grad():
                self.running_mean.mul_(1.0 - self.momentum).add_(mean.detach() * self.momentum)
                self.running_cov.mul_(1.0 - self.momentum).add_(cov.detach() * self.momentum)
        else:
            mean = self.running_mean.to(device=x.device, dtype=x_float.dtype)
            centered = x_float - mean
            flat = centered.permute(1, 0, 2, 3).contiguous()
            flat = flat.view(self.num_groups, self.group_size, -1)
            cov = self.running_cov.to(device=x.device, dtype=x_float.dtype)

        eye = torch.eye(self.group_size, device=x.device, dtype=x_float.dtype).unsqueeze(0)
        cov = (1.0 - self.eps) * cov + self.eps * eye
        chol = torch.linalg.cholesky(cov)
        inv_chol = torch.linalg.inv(chol)
        decorrelated = torch.bmm(inv_chol, flat)
        decorrelated = decorrelated.view(c, b, h, w).permute(1, 0, 2, 3).contiguous()
        return decorrelated.to(orig_dtype)


class WhiteningScaleShift(nn.Module):
    def __init__(self, channels, group_size, momentum, eps):
        super().__init__()
        self.whitening = TokenWhitening2d(channels, group_size, momentum, eps)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return self.whitening(x) * self.gamma + self.beta + x


class HypergraphConv2d(nn.Module):
    """HOS-Net-style hypergraph convolution with configurable ViT grid sizes."""

    def __init__(self, dim, feat_h, feat_w, edges=128, filters=128, theta=0.0, bias=True):
        super().__init__()
        self.dim = dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.vertices = feat_h * feat_w
        self.edges = int(edges)
        self.filters = int(filters)
        self.theta = float(theta)
        self.phi_conv = nn.Conv2d(dim, self.filters, kernel_size=1, stride=1, padding=0)
        self.metric_conv = nn.Conv2d(dim, self.filters, kernel_size=1, stride=1, padding=0)
        self.assign_conv = nn.Conv2d(dim, self.edges, kernel_size=7, stride=1, padding=3)
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.dim or h != self.feat_h or w != self.feat_w:
            raise ValueError(
                'expected [B, {}, {}, {}], got {}'.format(
                    self.dim, self.feat_h, self.feat_w, tuple(x.shape)))

        phi = self.phi_conv(x).permute(0, 2, 3, 1).contiguous()
        phi = phi.view(b, self.vertices, self.filters)

        metric = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        metric = self.metric_conv(metric).view(b, self.filters)
        metric = torch.diag_embed(metric)

        assign = self.assign_conv(x).permute(0, 2, 3, 1).contiguous()
        assign = assign.view(b, self.vertices, self.edges)

        incidence = torch.matmul(phi, torch.matmul(metric, torch.matmul(phi.transpose(1, 2), assign))).abs()
        if self.theta > 0.0:
            threshold = self.theta * incidence.mean(dim=(1, 2), keepdim=True)
            incidence = torch.where(incidence < threshold, torch.zeros_like(incidence), incidence)

        node_degree = incidence.sum(dim=2)
        incidence_norm = node_degree.add(1e-10).pow(-0.5).unsqueeze(-1) * incidence
        edge_degree = incidence.sum(dim=1)
        edge_degree = torch.diag_embed(edge_degree.add(1e-10).pow(-1.0))

        features = x.permute(0, 2, 3, 1).contiguous().view(b, self.vertices, self.dim)
        propagated = torch.matmul(incidence_norm, torch.matmul(edge_degree, torch.matmul(
            incidence_norm.transpose(1, 2), features)))
        out = torch.matmul(features - propagated, self.weight)
        if self.bias is not None:
            out = out + self.bias
        out = out.permute(0, 2, 1).contiguous().view(b, self.dim, self.feat_h, self.feat_w)
        return out


class HighOrderStructureSynergy(nn.Module):
    def __init__(self, dim, feat_h, feat_w, cfg):
        super().__init__()
        self.dim = dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.use_whitening = bool(cfg.MODEL.CASS_HSS_WHITEN)
        self.graph_weight = float(cfg.MODEL.CASS_HSS_GRAPH_WEIGHT)
        if self.use_whitening:
            self.whitening = WhiteningScaleShift(
                dim,
                cfg.MODEL.CASS_WHITEN_GROUP_SIZE,
                cfg.MODEL.CASS_WHITEN_MOMENTUM,
                cfg.MODEL.CASS_WHITEN_EPS,
            )
        self.hypergraph = HypergraphConv2d(
            dim=dim,
            feat_h=feat_h,
            feat_w=feat_w,
            edges=cfg.MODEL.CASS_HSS_EDGES,
            filters=cfg.MODEL.CASS_HSS_FILTERS,
            theta=cfg.MODEL.CASS_HSS_THETA,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat):
        cls_token = feat[:, :1, :]
        patches = feat[:, 1:, :]
        b, n, c = patches.shape
        expected = self.feat_h * self.feat_w
        if n != expected:
            raise ValueError('expected {} patch tokens, got {}'.format(expected, n))

        x = patches.transpose(1, 2).contiguous().view(b, c, self.feat_h, self.feat_w)
        graph_input = self.whitening(x) if self.use_whitening else x
        graph_out = self.hypergraph(graph_input)
        enhanced = x + self.graph_weight * graph_out
        enhanced = enhanced.view(b, c, n).transpose(1, 2).contiguous()
        enhanced = self.norm(enhanced)
        out = torch.cat([cls_token, enhanced], dim=1)
        score_self = F.cosine_similarity(enhanced, cls_token.expand(-1, n, -1), dim=-1)
        return out, score_self


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.norm_context = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, context):
        b, n, c = context.shape
        context = self.norm_context(context)
        q = self.q(query).view(b, 1, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).view(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).view(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, c)
        return self.proj_drop(self.proj(out))


class SynergyQueryToken(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.query_token = nn.Parameter(torch.zeros(1, dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats):
        all_patches = torch.cat([feat[:, 1:, :] for feat in feats], dim=1)
        query = self.query_token.expand(all_patches.size(0), -1)
        prototype = self.norm(self.attn(query, all_patches))
        scores = []
        for feat in feats:
            patches = feat[:, 1:, :]
            scores.append(F.cosine_similarity(patches, prototype.unsqueeze(1), dim=-1))
        return scores, prototype


class NeighborhoodGuidedAdapter(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.dim = dim
        self.knn = int(cfg.MODEL.CASS_NGA_KNN)
        self.gate_groups = _largest_divisor_at_most(dim, cfg.MODEL.CASS_NGA_GATE_GROUPS)
        hidden = int(cfg.MODEL.CASS_NGA_HIDDEN)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.gate_groups),
        )
        self.gate_scale = nn.Parameter(torch.tensor(float(cfg.MODEL.CASS_NGA_GATE_SCALE)))
        self.memory_ready = False
        self.memory_keys = []
        self.key_to_index = {}
        self.memory_neighbors = {}

    def set_memory(self, features, keys, modality_names):
        if features.numel() == 0:
            self.memory_ready = False
            return
        features = F.normalize(features.detach().float(), dim=-1).cpu()
        full = torch.zeros(features.size(0), len(MODALITY_ORDER), features.size(-1))
        for pos, name in enumerate(modality_names):
            full[:, MODALITY_ORDER.index(name), :] = features[:, pos, :]

        self.memory_keys = [str(k) for k in keys]
        self.key_to_index = {key: idx for idx, key in enumerate(self.memory_keys)}
        self.memory_neighbors = {}
        n = full.size(0)
        k = min(self.knn, max(n - 1, 0))
        for idx, name in enumerate(MODALITY_ORDER):
            if k == 0:
                self.memory_neighbors[idx] = torch.empty(n, 0, dtype=torch.long)
                continue
            sim = full[:, idx, :] @ full[:, idx, :].t()
            sim.fill_diagonal_(float('-inf'))
            self.memory_neighbors[idx] = sim.topk(k, dim=1).indices
        self.memory_ready = True

    def _fallback_jaccard(self, cos_values):
        return [(cos.clamp(-1.0, 1.0) + 1.0) * 0.5 for cos in cos_values]

    def _memory_jaccard(self, keys, device, dtype, batch_size):
        if (not self.memory_ready) or keys is None:
            return None
        out = torch.zeros(batch_size, len(PAIR_ORDER), device=device, dtype=dtype)
        for b, key in enumerate(keys):
            idx = self.key_to_index.get(str(key))
            if idx is None:
                continue
            for p, (i, j) in enumerate(PAIR_ORDER):
                a = self.memory_neighbors[i][idx]
                b_set = self.memory_neighbors[j][idx]
                if a.numel() == 0 or b_set.numel() == 0:
                    continue
                inter = torch.isin(a, b_set).sum().item()
                union = a.numel() + b_set.numel() - inter
                out[b, p] = float(inter) / max(float(union), 1.0)
        return out

    def forward(self, cls_list, modality_names, keys=None):
        batch_size = cls_list[0].size(0)
        device = cls_list[0].device
        dtype = cls_list[0].dtype
        cls_by_index = {}
        for name, cls in zip(modality_names, cls_list):
            cls_by_index[MODALITY_ORDER.index(name)] = cls

        cos_values = []
        for i, j in PAIR_ORDER:
            if i in cls_by_index and j in cls_by_index:
                cos = F.cosine_similarity(cls_by_index[i], cls_by_index[j], dim=-1)
            else:
                cos = torch.zeros(batch_size, device=device, dtype=dtype)
            cos_values.append(cos)
        jaccard = self._memory_jaccard(keys, device, dtype, batch_size)
        if jaccard is None:
            jaccard_values = self._fallback_jaccard(cos_values)
            jaccard = torch.stack(jaccard_values, dim=1)

        cos_tensor = torch.stack(cos_values, dim=1)
        context = torch.cat([jaccard, cos_tensor], dim=1)
        alpha_dyn = torch.sigmoid(self.alpha_mlp(context))
        group_bias = torch.tanh(self.gate_mlp(context)) * self.gate_scale
        repeat = self.dim // self.gate_groups
        gate_bias = group_bias.repeat_interleave(repeat, dim=1)
        if gate_bias.size(1) < self.dim:
            pad = self.dim - gate_bias.size(1)
            gate_bias = F.pad(gate_bias, (0, pad))
        return alpha_dyn, gate_bias[:, :self.dim], context


class DynamicCollaborativeSelector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.topk = int(cfg.MODEL.CASS_TOPK)
        self.use_ste = bool(cfg.MODEL.CASS_STE)
        self.ste_tau = float(cfg.MODEL.CASS_STE_TAU)

    @staticmethod
    def _minmax(score, eps=1e-8):
        lo = score.min(dim=1, keepdim=True).values
        hi = score.max(dim=1, keepdim=True).values
        return (score - lo) / (hi - lo + eps)

    def forward(self, feat, score_self, score_structure, alpha_dyn):
        cls_token = feat[:, :1, :]
        patches = feat[:, 1:, :]
        b, n, _ = patches.shape
        s_self = self._minmax(score_self)
        s_structure = self._minmax(score_structure)
        score = (1.0 - alpha_dyn) * s_self + alpha_dyn * s_structure
        k = min(self.topk, n)
        topk_idx = score.topk(k, dim=1).indices
        mask = torch.zeros(b, n, device=patches.device, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)

        hard = mask.to(patches.dtype)
        if self.use_ste and self.training:
            soft = F.softmax(score / self.ste_tau, dim=1)
            gate = hard + soft - soft.detach()
        else:
            gate = hard
        selected = torch.cat([cls_token, patches * gate.unsqueeze(-1)], dim=1)
        return selected, mask


class ContextAwareGatedFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.cross_blocks = nn.ModuleDict()
        for target in MODALITY_ORDER:
            for source in MODALITY_ORDER:
                if target != source:
                    key = '{}<-{}'.format(target, source)
                    self.cross_blocks[key] = CrossAttention(dim, num_heads=num_heads, qkv_bias=True)
        self.self_blocks = nn.ModuleDict({
            name: CrossAttention(dim, num_heads=num_heads, qkv_bias=True)
            for name in MODALITY_ORDER
        })
        self.gates = nn.ModuleDict()
        for target in MODALITY_ORDER:
            for source in MODALITY_ORDER:
                if target != source:
                    key = '{}<-{}'.format(target, source)
                    self.gates[key] = nn.Sequential(
                        nn.Linear(2 * dim, dim),
                        nn.LayerNorm(dim),
                    )
        self.priority = ('NIR', 'TIR', 'RGB')

    def _ordered_sources(self, target, modality_names):
        return [name for name in self.priority if name != target and name in modality_names]

    def forward(self, selected, gate_bias, modality_names):
        fused = {}
        for target in modality_names:
            cls = selected[target][:, 0, :]
            for source in self._ordered_sources(target, modality_names):
                key = '{}<-{}'.format(target, source)
                delta = self.cross_blocks[key](cls, selected[source][:, 1:, :])
                gate = torch.sigmoid(self.gates[key](torch.cat([cls, delta], dim=-1)) + gate_bias)
                cls = (1.0 - gate) * cls + gate * delta
            cls = cls + self.self_blocks[target](cls, selected[target][:, 1:, :])
            fused[target] = cls
        return fused


class CASSModule(nn.Module):
    def __init__(self, dim, cfg, feat_h, feat_w):
        super().__init__()
        self.dim = dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        heads = int(cfg.MODEL.CASS_NUM_HEADS)
        self.hss = HighOrderStructureSynergy(dim, feat_h, feat_w, cfg)
        self.sqt = SynergyQueryToken(dim, num_heads=heads)
        self.nga = NeighborhoodGuidedAdapter(dim, cfg)
        self.selector = DynamicCollaborativeSelector(cfg)
        self.fusion = ContextAwareGatedFusion(dim, num_heads=heads)

    def set_memory(self, features, keys, modality_names):
        self.nga.set_memory(features, keys, modality_names)

    def forward(self, features, img_path=None):
        modality_names = list(features.keys())
        enhanced = {}
        self_scores = {}
        for name in modality_names:
            enhanced[name], self_scores[name] = self.hss(features[name])

        enhanced_list = [enhanced[name] for name in modality_names]
        structure_scores, _ = self.sqt(enhanced_list)
        cls_list = [features[name][:, 0, :] for name in modality_names]
        alpha_dyn, gate_bias, _ = self.nga(cls_list, modality_names, keys=img_path)

        selected = {}
        masks = {}
        for idx, name in enumerate(modality_names):
            selected[name], masks[name] = self.selector(
                enhanced[name], self_scores[name], structure_scores[idx], alpha_dyn)

        fused = self.fusion(selected, gate_bias, modality_names)
        template = next(iter(fused.values()))
        descriptor_parts = []
        for name in MODALITY_ORDER:
            if name in fused:
                descriptor_parts.append(fused[name])
            else:
                descriptor_parts.append(torch.zeros_like(template))
        descriptor = torch.cat(descriptor_parts, dim=-1)
        return selected, masks, fused, descriptor

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
        # Avoid explicitly inverting ill-conditioned whitening groups under AMP.
        decorrelated = torch.linalg.solve_triangular(chol, flat, upper=False)
        decorrelated = torch.nan_to_num(decorrelated, nan=0.0, posinf=0.0, neginf=0.0)
        decorrelated = decorrelated.reshape(c, b, h, w).permute(1, 0, 2, 3).contiguous()
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

        orig_dtype = x.dtype
        x_float = x.float()
        phi = self.phi_conv(x).float().permute(0, 2, 3, 1).contiguous()
        phi = phi.view(b, self.vertices, self.filters)

        metric = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        metric = self.metric_conv(metric).float().view(b, self.filters)
        metric = torch.diag_embed(metric)

        assign = self.assign_conv(x).float().permute(0, 2, 3, 1).contiguous()
        assign = assign.view(b, self.vertices, self.edges)

        incidence = torch.matmul(phi, torch.matmul(metric, torch.matmul(phi.transpose(1, 2), assign))).abs()
        if self.theta > 0.0:
            threshold = self.theta * incidence.mean(dim=(1, 2), keepdim=True)
            incidence = torch.where(incidence < threshold, torch.zeros_like(incidence), incidence)

        node_degree = incidence.sum(dim=2)
        incidence_norm = node_degree.clamp_min(1e-6).pow(-0.5).unsqueeze(-1) * incidence
        edge_degree = incidence.sum(dim=1)
        edge_degree = torch.diag_embed(edge_degree.clamp_min(1e-6).pow(-1.0))

        features = x_float.permute(0, 2, 3, 1).contiguous().view(b, self.vertices, self.dim)
        propagated = torch.matmul(incidence_norm, torch.matmul(edge_degree, torch.matmul(
            incidence_norm.transpose(1, 2), features)))
        out = torch.matmul(features - propagated, self.weight.float())
        if self.bias is not None:
            out = out + self.bias.float()
        out = out.permute(0, 2, 1).contiguous().view(b, self.dim, self.feat_h, self.feat_w)
        return out.to(orig_dtype)


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
        self.use_modal_alpha = bool(cfg.MODEL.CASS_MODAL_ALPHA)
        self.use_dynamic_topk = bool(cfg.MODEL.CASS_DYNAMIC_TOPK)
        self.memory_warmup_epochs = int(cfg.MODEL.CASS_NGA_WARMUP_EPOCHS)
        self.memory_ema_momentum = float(cfg.MODEL.CASS_NGA_EMA_MOMENTUM)
        self.use_prototype_memory = bool(cfg.MODEL.CASS_NGA_USE_PROTOTYPE)
        hidden = int(cfg.MODEL.CASS_NGA_HIDDEN)
        context_dim = len(PAIR_ORDER) * 2 + len(MODALITY_ORDER)
        alpha_out = len(MODALITY_ORDER) if self.use_modal_alpha else 1
        self.alpha_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, alpha_out),
        )
        if self.use_dynamic_topk:
            self.topk_mlp = nn.Sequential(
                nn.Linear(context_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, len(MODALITY_ORDER)),
            )
        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.gate_groups),
        )
        self.gate_scale = nn.Parameter(torch.tensor(float(cfg.MODEL.CASS_NGA_GATE_SCALE)))
        self.memory_ready = False
        self.memory_tensor = None
        self.memory_keys = []
        self.key_to_index = {}
        self.key_to_pid = {}
        self.memory_neighbors = {}
        self.pid_to_proto_index = {}
        self.prototype_neighbors = {}

    def set_memory(self, features, keys, modality_names, labels=None, epoch=None):
        if features.numel() == 0 or (
                epoch is not None and int(epoch) <= self.memory_warmup_epochs):
            self.memory_ready = False
            return
        features = F.normalize(features.detach().float(), dim=-1).cpu()
        new_full = torch.zeros(features.size(0), len(MODALITY_ORDER), features.size(-1))
        for pos, name in enumerate(modality_names):
            new_full[:, MODALITY_ORDER.index(name), :] = features[:, pos, :]
        new_full = F.normalize(new_full, dim=-1)

        keys = [str(k) for k in keys]
        if (self.memory_tensor is not None and self.memory_keys == keys and
                self.memory_tensor.shape == new_full.shape):
            momentum = min(max(self.memory_ema_momentum, 0.0), 1.0)
            full = momentum * self.memory_tensor + (1.0 - momentum) * new_full
            full = F.normalize(full, dim=-1)
        else:
            full = new_full
        self.memory_tensor = full

        self.memory_keys = keys
        self.key_to_index = {key: idx for idx, key in enumerate(self.memory_keys)}
        self.key_to_pid = {}
        if labels is not None:
            self.key_to_pid = {
                key: int(labels[idx])
                for idx, key in enumerate(self.memory_keys)
            }
        self._build_sample_neighbors(full)
        self._build_prototype_neighbors(full, labels)
        self.memory_ready = True

    def _build_sample_neighbors(self, full):
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

    def _build_prototype_neighbors(self, full, labels):
        self.pid_to_proto_index = {}
        self.prototype_neighbors = {}
        if (not self.use_prototype_memory) or labels is None:
            return
        label_tensor = torch.tensor([int(x) for x in labels], dtype=torch.long)
        unique = torch.unique(label_tensor, sorted=True)
        if unique.numel() == 0:
            return
        prototypes = torch.zeros(unique.numel(), len(MODALITY_ORDER), full.size(-1))
        for proto_idx, pid in enumerate(unique):
            mask = label_tensor == pid
            prototypes[proto_idx] = full[mask].mean(dim=0)
            self.pid_to_proto_index[int(pid.item())] = proto_idx
        prototypes = F.normalize(prototypes, dim=-1)
        k = min(self.knn, max(prototypes.size(0) - 1, 0))
        for idx, name in enumerate(MODALITY_ORDER):
            if k == 0:
                self.prototype_neighbors[idx] = torch.empty(
                    prototypes.size(0), 0, dtype=torch.long)
                continue
            sim = prototypes[:, idx, :] @ prototypes[:, idx, :].t()
            sim.fill_diagonal_(float('-inf'))
            self.prototype_neighbors[idx] = sim.topk(k, dim=1).indices

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
            pid = self.key_to_pid.get(str(key))
            proto_idx = self.pid_to_proto_index.get(pid)
            for p, (i, j) in enumerate(PAIR_ORDER):
                if proto_idx is not None and self.prototype_neighbors:
                    a = self.prototype_neighbors[i][proto_idx]
                    b_set = self.prototype_neighbors[j][proto_idx]
                else:
                    a = self.memory_neighbors[i][idx]
                    b_set = self.memory_neighbors[j][idx]
                if a.numel() == 0 or b_set.numel() == 0:
                    continue
                inter = torch.isin(a, b_set).sum().item()
                union = a.numel() + b_set.numel() - inter
                out[b, p] = float(inter) / max(float(union), 1.0)
        return out

    def forward(self, cls_list, modality_names, keys=None, quality_scores=None):
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
        if quality_scores is None:
            quality_context = torch.zeros(
                batch_size, len(MODALITY_ORDER), device=device, dtype=dtype)
            for name in modality_names:
                quality_context[:, MODALITY_ORDER.index(name)] = 1.0
        else:
            quality_context = quality_scores.to(device=device, dtype=dtype)
        context = torch.cat([jaccard, cos_tensor, quality_context], dim=1)
        alpha_dyn = torch.sigmoid(self.alpha_mlp(context))
        if alpha_dyn.size(1) == 1:
            alpha_dyn = alpha_dyn.expand(-1, len(MODALITY_ORDER))
        keep_ratio = torch.sigmoid(self.topk_mlp(context)) if self.use_dynamic_topk else None
        group_bias = torch.tanh(self.gate_mlp(context)) * self.gate_scale
        repeat = self.dim // self.gate_groups
        gate_bias = group_bias.repeat_interleave(repeat, dim=1)
        if gate_bias.size(1) < self.dim:
            pad = self.dim - gate_bias.size(1)
            gate_bias = F.pad(gate_bias, (0, pad))
        return alpha_dyn, keep_ratio, gate_bias[:, :self.dim], context


class DynamicCollaborativeSelector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.topk = int(cfg.MODEL.CASS_TOPK)
        self.use_dynamic_topk = bool(cfg.MODEL.CASS_DYNAMIC_TOPK)
        self.min_topk = int(cfg.MODEL.CASS_MIN_TOPK)
        self.max_topk = int(cfg.MODEL.CASS_MAX_TOPK)
        self.use_ste = bool(cfg.MODEL.CASS_STE)
        self.ste_tau = float(cfg.MODEL.CASS_STE_TAU)
        self.soft_residual_weight = float(cfg.MODEL.CASS_SOFT_RESIDUAL_WEIGHT)

    @staticmethod
    def _minmax(score, eps=1e-8):
        orig_dtype = score.dtype
        score_float = score.float()
        lo = score_float.min(dim=1, keepdim=True).values
        hi = score_float.max(dim=1, keepdim=True).values
        return ((score_float - lo) / (hi - lo).clamp_min(eps)).to(orig_dtype)

    def _topk_bounds(self, num_tokens):
        base = max(1, min(self.topk, num_tokens))
        if not self.use_dynamic_topk:
            return base, base
        min_topk = self.min_topk if self.min_topk > 0 else max(1, base // 2)
        max_topk = self.max_topk if self.max_topk > 0 else base + max(1, base // 2)
        min_topk = max(1, min(min_topk, num_tokens))
        max_topk = max(min_topk, min(max_topk, num_tokens))
        return min_topk, max_topk

    def forward(self, feat, score_self, score_structure, alpha_dyn, keep_ratio=None, quality=None):
        cls_token = feat[:, :1, :]
        patches = feat[:, 1:, :]
        b, n, _ = patches.shape
        s_self = self._minmax(score_self)
        s_structure = self._minmax(score_structure)
        score = (1.0 - alpha_dyn) * s_self + alpha_dyn * s_structure
        min_k, max_k = self._topk_bounds(n)
        if self.use_dynamic_topk and keep_ratio is not None:
            keep_ratio = keep_ratio.view(b, 1).clamp(0.0, 1.0)
            if quality is not None:
                quality = quality.view(b, 1).clamp(0.0, 1.0)
                keep_ratio = (keep_ratio * (0.5 + 0.5 * quality)).clamp(0.0, 1.0)
            k_float = min_k + keep_ratio.squeeze(1) * float(max_k - min_k)
            k_hard = k_float.round().long().clamp(min_k, max_k)
        else:
            k_float = torch.full((b,), float(max_k), device=patches.device, dtype=patches.dtype)
            k_hard = torch.full((b,), max_k, device=patches.device, dtype=torch.long)

        topk_count = int(k_hard.max().item())
        topk_idx = score.topk(topk_count, dim=1).indices
        active = torch.arange(topk_count, device=patches.device).view(1, -1)
        active = active < k_hard.view(-1, 1)
        mask = torch.zeros(b, n, device=patches.device, dtype=torch.bool)
        mask.scatter_(1, topk_idx, active)

        hard = mask.to(patches.dtype)
        residual = min(max(self.soft_residual_weight, 0.0), 1.0)
        if residual > 0.0:
            hard = residual + (1.0 - residual) * hard
        if self.use_ste and self.training:
            soft = F.softmax(score / self.ste_tau, dim=1) * k_float.view(b, 1).to(score.dtype)
            soft = soft.clamp(max=1.0)
            if residual > 0.0:
                soft = residual + (1.0 - residual) * soft
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

    @staticmethod
    def _quality(quality_scores, name, batch_size, device, dtype):
        if quality_scores is None:
            return torch.ones(batch_size, 1, device=device, dtype=dtype)
        idx = MODALITY_ORDER.index(name)
        return quality_scores[:, idx:idx + 1].to(device=device, dtype=dtype)

    def forward(self, selected, gate_bias, modality_names, quality_scores=None):
        fused = {}
        for target in modality_names:
            cls = selected[target][:, 0, :]
            batch_size = cls.size(0)
            q_target = self._quality(
                quality_scores, target, batch_size, cls.device, cls.dtype)
            for source in self._ordered_sources(target, modality_names):
                key = '{}<-{}'.format(target, source)
                q_source = self._quality(
                    quality_scores, source, batch_size, cls.device, cls.dtype)
                context = selected[source][:, 1:, :] * q_source.unsqueeze(-1)
                delta = self.cross_blocks[key](cls, context)
                gate = torch.sigmoid(self.gates[key](torch.cat([cls, delta], dim=-1)) + gate_bias)
                gate = gate * q_source
                cls = (1.0 - gate) * cls + gate * delta
            cls = cls + q_target * self.self_blocks[target](cls, selected[target][:, 1:, :])
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

    def set_memory(self, features, keys, modality_names, labels=None, epoch=None):
        self.nga.set_memory(features, keys, modality_names, labels=labels, epoch=epoch)

    def forward(self, features, img_path=None, quality_scores=None):
        modality_names = list(features.keys())
        enhanced = {}
        self_scores = {}
        for name in modality_names:
            enhanced[name], self_scores[name] = self.hss(features[name])

        enhanced_list = [enhanced[name] for name in modality_names]
        structure_scores, _ = self.sqt(enhanced_list)
        cls_list = [features[name][:, 0, :] for name in modality_names]
        if quality_scores is not None:
            template = cls_list[0]
            quality_scores = quality_scores.to(device=template.device, dtype=template.dtype)
        alpha_dyn, keep_ratio, gate_bias, _ = self.nga(
            cls_list, modality_names, keys=img_path, quality_scores=quality_scores)

        selected = {}
        masks = {}
        for idx, name in enumerate(modality_names):
            modality_idx = MODALITY_ORDER.index(name)
            alpha = alpha_dyn[:, modality_idx:modality_idx + 1]
            ratio = keep_ratio[:, modality_idx:modality_idx + 1] if keep_ratio is not None else None
            quality = quality_scores[:, modality_idx:modality_idx + 1] \
                if quality_scores is not None else None
            selected[name], masks[name] = self.selector(
                enhanced[name], self_scores[name], structure_scores[idx], alpha, ratio, quality)

        fused = self.fusion(selected, gate_bias, modality_names, quality_scores=quality_scores)
        template = next(iter(fused.values()))
        descriptor_parts = []
        for name in MODALITY_ORDER:
            if name in fused:
                part = fused[name]
                if quality_scores is not None:
                    idx = MODALITY_ORDER.index(name)
                    part = part * quality_scores[:, idx:idx + 1]
                descriptor_parts.append(part)
            else:
                descriptor_parts.append(torch.zeros_like(template))
        descriptor = torch.cat(descriptor_parts, dim=-1)
        return selected, masks, fused, descriptor

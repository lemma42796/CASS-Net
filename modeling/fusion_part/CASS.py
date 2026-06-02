from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_ORDER = ('RGB', 'NIR', 'TIR')
PAIR_ORDER = ((0, 1), (0, 2), (1, 2))
CASS_ABLATION_STAGES = (
    'baseline',
    'hss',
    'hss_nga',
    'hss_nga_cagf',
    'hss_sqt',
    'hss_sqt_nga',
    'hss_sqt_nga_cagf',
    'full',
)
CASS_STAGE_RANK = {name: idx for idx, name in enumerate(CASS_ABLATION_STAGES)}
CASS_STAGE_FEATURES = {
    'baseline': (),
    'hss': ('hss',),
    'hss_nga': ('hss', 'nga'),
    'hss_nga_cagf': ('hss', 'nga', 'cagf'),
    'hss_sqt': ('hss', 'sqt'),
    'hss_sqt_nga': ('hss', 'sqt', 'nga'),
    'hss_sqt_nga_cagf': ('hss', 'sqt', 'nga', 'cagf'),
    'full': ('hss', 'sqt', 'nga', 'cagf'),
}


def _canonical_stage(stage):
    name = str(stage).strip().lower()
    aliases = {
        'none': 'baseline',
        'base': 'baseline',
        'hss+nga': 'hss_nga',
        'hss+nga+cagf': 'hss_nga_cagf',
        'hss_nga_ca_gf': 'hss_nga_cagf',
        'hss+sqt': 'hss_sqt',
        'hss+sqt+nga': 'hss_sqt_nga',
        'hss+sqt+nga+cagf': 'hss_sqt_nga_cagf',
        'hss_sqt_nga_ca_gf': 'hss_sqt_nga_cagf',
        'core': 'hss_sqt_nga_cagf',
    }
    name = aliases.get(name, name)
    if name not in CASS_STAGE_RANK:
        raise ValueError(
            'Unsupported CASS_ABLATION_STAGE "{}". Use one of: {}'.format(
                stage, ', '.join(CASS_ABLATION_STAGES)))
    return name


def _largest_divisor_at_most(value, limit):
    limit = max(1, min(value, int(limit)))
    for candidate in range(limit, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _disable_cuda_autocast(x):
    if x.is_cuda:
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            return torch.amp.autocast('cuda', enabled=False)
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


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

        assign = self.assign_conv(x).float().permute(0, 2, 3, 1).contiguous()
        assign = assign.view(b, self.vertices, self.edges)

        assign_projected = torch.matmul(phi.transpose(1, 2), assign)
        assign_projected = metric.unsqueeze(-1) * assign_projected
        incidence = torch.matmul(phi, assign_projected).abs()
        incidence = torch.nan_to_num(incidence, nan=0.0, posinf=0.0, neginf=0.0)
        if self.theta > 0.0:
            threshold = self.theta * incidence.mean(dim=(1, 2), keepdim=True)
            incidence = torch.where(incidence < threshold, torch.zeros_like(incidence), incidence)

        node_degree = incidence.sum(dim=2)
        incidence_norm = node_degree.clamp_min(1e-6).pow(-0.5).unsqueeze(-1) * incidence
        incidence_norm = torch.nan_to_num(incidence_norm, nan=0.0, posinf=0.0, neginf=0.0)
        edge_degree = incidence.sum(dim=1)
        edge_weight = edge_degree.clamp_min(1e-6).pow(-1.0)

        features = x_float.permute(0, 2, 3, 1).contiguous().view(b, self.vertices, self.dim)
        propagated = torch.matmul(incidence_norm.transpose(1, 2), features)
        propagated = edge_weight.unsqueeze(-1) * propagated
        propagated = torch.matmul(incidence_norm, propagated)
        propagated = torch.nan_to_num(propagated, nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.matmul(features - propagated, self.weight.float())
        if self.bias is not None:
            out = out + self.bias.float()
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = out.permute(0, 2, 1).contiguous().view(b, self.dim, self.feat_h, self.feat_w)
        return out.to(orig_dtype)


class TokenResidualAdapter(nn.Module):
    def __init__(self, dim, hidden_dim=0, use_norm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        hidden_dim = int(hidden_dim)
        if hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
        else:
            self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        tokens = self.proj(self.norm(tokens))
        return tokens.transpose(1, 2).contiguous().view(b, c, h, w)


class HighOrderStructureSynergy(nn.Module):
    def __init__(self, dim, feat_h, feat_w, cfg):
        super().__init__()
        self.dim = dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.use_whitening = bool(cfg.MODEL.CASS_HSS_WHITEN)
        self.graph_weight = float(cfg.MODEL.CASS_HSS_GRAPH_WEIGHT)
        self.graph_warmup_epochs = int(cfg.MODEL.CASS_HSS_GRAPH_WARMUP_EPOCHS)
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
        self.use_residual_adapter = bool(cfg.MODEL.CASS_HSS_RESIDUAL_ADAPTER)
        if self.use_residual_adapter:
            self.residual_adapter = TokenResidualAdapter(
                dim,
                cfg.MODEL.CASS_HSS_ADAPTER_DIM,
                bool(cfg.MODEL.CASS_HSS_ADAPTER_NORM),
            )
        else:
            self.residual_adapter = nn.Identity()
        gate_init = float(cfg.MODEL.CASS_HSS_GATE_INIT)
        self.residual_gate = nn.Parameter(torch.full((dim,), gate_init))
        self.gate_floor = float(cfg.MODEL.CASS_HSS_GATE_FLOOR)
        self.gate_floor_warmup_epochs = int(cfg.MODEL.CASS_HSS_GATE_FLOOR_WARMUP_EPOCHS)
        self.score_mix = float(cfg.MODEL.CASS_HSS_SCORE_MIX)
        self.score_source = str(cfg.MODEL.CASS_HSS_SCORE_SOURCE).lower()
        self.score_detach = bool(cfg.MODEL.CASS_HSS_SCORE_DETACH)
        if not 0.0 <= self.score_mix <= 1.0:
            raise ValueError('CASS_HSS_SCORE_MIX must be in [0, 1], got {}'.format(self.score_mix))
        if self.score_source not in ('residual', 'graph'):
            raise ValueError("CASS_HSS_SCORE_SOURCE must be 'residual' or 'graph', got {}".format(
                self.score_source))
        self.norm = nn.LayerNorm(dim)

    def current_graph_weight(self, epoch=None):
        if self.graph_warmup_epochs <= 0 or epoch is None:
            return self.graph_weight
        scale = min(1.0, max(0.0, float(epoch)) / float(self.graph_warmup_epochs))
        return self.graph_weight * scale

    def current_gate_floor(self, epoch=None):
        if self.gate_floor_warmup_epochs <= 0 or epoch is None:
            return self.gate_floor
        scale = min(1.0, max(0.0, float(epoch)) / float(self.gate_floor_warmup_epochs))
        return self.gate_floor * scale

    def effective_residual_gate(self, epoch=None):
        return self.residual_gate.float() + self.current_gate_floor(epoch)

    def current_residual_gate(self, epoch=None):
        gate = self.effective_residual_gate(epoch).detach().float()
        return gate.mean().item(), gate.abs().mean().item()

    @staticmethod
    def _minmax_token_score(score, eps=1e-8):
        score = score.float()
        lo = score.min(dim=1, keepdim=True).values
        hi = score.max(dim=1, keepdim=True).values
        return (score - lo) / (hi - lo).clamp_min(eps)

    def _structure_score(self, graph_out, graph_residual):
        source = graph_out if self.score_source == 'graph' else graph_residual
        if self.score_detach:
            source = source.detach()
        tokens = source.float().flatten(2).transpose(1, 2).contiguous()
        return self._minmax_token_score(tokens.norm(dim=-1))

    def _self_score(self, enhanced, cls_token, graph_out, graph_residual):
        n = enhanced.size(1)
        cls_score = F.cosine_similarity(
            enhanced, cls_token.float().expand(-1, n, -1), dim=-1)
        if self.score_mix <= 0.0:
            return cls_score
        cls_score = self._minmax_token_score(cls_score)
        structure_score = self._structure_score(graph_out, graph_residual)
        return (1.0 - self.score_mix) * cls_score + self.score_mix * structure_score

    def forward(self, feat, epoch=None):
        cls_token = feat[:, :1, :]
        patches = feat[:, 1:, :]
        b, n, c = patches.shape
        expected = self.feat_h * self.feat_w
        if n != expected:
            raise ValueError('expected {} patch tokens, got {}'.format(expected, n))

        x = patches.transpose(1, 2).contiguous().view(b, c, self.feat_h, self.feat_w)
        orig_dtype = x.dtype
        with _disable_cuda_autocast(x):
            x_float = x.float()
            graph_input = self.whitening(x_float) if self.use_whitening else x_float
            graph_out = self.hypergraph(graph_input)
            graph_residual = self.residual_adapter(graph_out)
            gate = self.effective_residual_gate(epoch).view(1, c, 1, 1).to(device=x.device)
            enhanced = x_float + self.current_graph_weight(epoch) * gate * graph_residual
            enhanced = enhanced.view(b, c, n).transpose(1, 2).contiguous()
            enhanced = self.norm(enhanced)
            score_self = self._self_score(enhanced, cls_token, graph_out, graph_residual)
        out = torch.cat([cls_token, enhanced.to(dtype=orig_dtype)], dim=1)
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
        single_query = query.dim() == 2
        if single_query:
            query = query.unsqueeze(1)
        q_len = query.size(1)
        q = self.q(query).view(b, q_len, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).view(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).view(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, q_len, c)
        out = self.proj_drop(self.proj(out))
        return out.squeeze(1) if single_query else out


class SynergyQueryToken(nn.Module):
    def __init__(self, dim, num_heads, cfg):
        super().__init__()
        self.num_queries = int(cfg.MODEL.CASS_SQT_NUM_QUERIES)
        if self.num_queries < 1:
            raise ValueError('CASS_SQT_NUM_QUERIES must be >= 1, got {}'.format(
                self.num_queries))
        self.query_token = nn.Parameter(torch.zeros(1, self.num_queries, dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.score_norm = nn.LayerNorm(dim)
        self.cls_query = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.cls_query_weight = float(cfg.MODEL.CASS_SQT_CLS_QUERY_WEIGHT)
        self.cls_score_weight = float(cfg.MODEL.CASS_SQT_CLS_SCORE_WEIGHT)
        for name, value in (
                ('CASS_SQT_CLS_QUERY_WEIGHT', self.cls_query_weight),
                ('CASS_SQT_CLS_SCORE_WEIGHT', self.cls_score_weight)):
            if value < 0.0 or value > 1.0:
                raise ValueError('{} must be in [0, 1], got {}'.format(name, value))

    def forward(self, feats):
        all_patches = torch.cat([feat[:, 1:, :] for feat in feats], dim=1)
        cls_context = torch.stack([feat[:, 0, :] for feat in feats], dim=1).mean(dim=1)
        query = self.query_token.expand(all_patches.size(0), -1, -1)
        cls_delta = self.cls_query(cls_context).unsqueeze(1)
        query = self.query_norm(query + self.cls_query_weight * cls_delta)
        prototype = self.norm(self.attn(query, all_patches))
        proto_norm = F.normalize(prototype, dim=-1)
        scores = []
        for feat in feats:
            patches = feat[:, 1:, :]
            patch_norm = F.normalize(patches, dim=-1)
            proto_score = (patch_norm @ proto_norm.transpose(1, 2)).max(dim=-1).values
            cls_anchor = self.score_norm(feat[:, 0, :])
            cls_score = F.cosine_similarity(patches, cls_anchor.unsqueeze(1), dim=-1)
            score = ((1.0 - self.cls_score_weight) * proto_score
                     + self.cls_score_weight * cls_score)
            scores.append(score)
        if self.num_queries == 1:
            diversity = prototype.new_zeros(())
        else:
            sim = proto_norm @ proto_norm.transpose(1, 2)
            eye = torch.eye(self.num_queries, device=sim.device, dtype=sim.dtype).unsqueeze(0)
            diversity = ((sim * (1.0 - eye)).pow(2).sum(dim=(1, 2)) /
                         float(self.num_queries * (self.num_queries - 1))).mean()
        return scores, prototype, diversity


class NeighborhoodGuidedAdapter(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.dim = dim
        self.knn = int(cfg.MODEL.CASS_NGA_KNN)
        self.gate_groups = _largest_divisor_at_most(dim, cfg.MODEL.CASS_NGA_GATE_GROUPS)
        self.use_modal_alpha = bool(cfg.MODEL.CASS_MODAL_ALPHA)
        self.use_dynamic_topk = bool(cfg.MODEL.CASS_DYNAMIC_TOPK)
        self.use_query_anchor = bool(cfg.MODEL.CASS_NGA_QUERY_ANCHOR)
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
        self.prototype_tensor = None

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
        self.prototype_tensor = None
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
        self.prototype_tensor = prototypes
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

    def _infer_query_anchor(self, cls_by_index, source):
        if source is None or source.numel() == 0:
            return None
        scores = None
        with torch.no_grad():
            for idx, query in cls_by_index.items():
                if idx >= source.size(1):
                    continue
                query_cpu = F.normalize(query.detach().float(), dim=-1).cpu()
                ref = source[:, idx, :].float()
                sim = query_cpu @ ref.t()
                scores = sim if scores is None else scores + sim
        if scores is None:
            return None
        return scores.argmax(dim=1)

    def _memory_jaccard(self, keys, cls_by_index, device, dtype, batch_size):
        if (not self.memory_ready) or keys is None:
            return None
        out = torch.zeros(batch_size, len(PAIR_ORDER), device=device, dtype=dtype)
        proto_anchor = None
        if (self.use_query_anchor and self.use_prototype_memory and
                self.prototype_tensor is not None and self.prototype_neighbors):
            proto_anchor = self._infer_query_anchor(cls_by_index, self.prototype_tensor)
        sample_anchor = None
        if self.use_query_anchor and self.memory_tensor is not None and self.memory_neighbors:
            sample_anchor = self._infer_query_anchor(cls_by_index, self.memory_tensor)

        for b, key in enumerate(keys):
            idx = self.key_to_index.get(str(key))
            if idx is None:
                idx = int(sample_anchor[b].item()) if sample_anchor is not None else None
            pid = self.key_to_pid.get(str(key))
            proto_idx = self.pid_to_proto_index.get(pid)
            if proto_idx is None and proto_anchor is not None:
                proto_idx = int(proto_anchor[b].item())
            for p, (i, j) in enumerate(PAIR_ORDER):
                if proto_idx is not None and self.prototype_neighbors:
                    a = self.prototype_neighbors[i][proto_idx]
                    b_set = self.prototype_neighbors[j][proto_idx]
                elif idx is None:
                    continue
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
        jaccard = self._memory_jaccard(keys, cls_by_index, device, dtype, batch_size)
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
        self.use_soft_gate = bool(cfg.MODEL.CASS_SELECTOR_SOFT_GATE)
        self.soft_gate_tau = float(cfg.MODEL.CASS_SOFT_GATE_TAU)
        if self.soft_gate_tau <= 0:
            raise ValueError('CASS_SOFT_GATE_TAU must be > 0, got {}'.format(
                self.soft_gate_tau))

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

        residual = min(max(self.soft_residual_weight, 0.0), 1.0)
        if self.use_soft_gate:
            soft = torch.sigmoid((score - 0.5) / self.soft_gate_tau)
            if residual > 0.0:
                soft = residual + (1.0 - residual) * soft
            gate = soft.to(patches.dtype)
        else:
            hard = mask.to(patches.dtype)
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
        return selected, mask, gate


class ContextAwareGatedFusion(nn.Module):
    def __init__(self, dim, num_heads, cfg):
        super().__init__()
        self.mode = str(cfg.MODEL.CASS_CAGF_MODE).strip().lower()
        self.residual_weight = float(cfg.MODEL.CASS_CAGF_RESIDUAL_WEIGHT)
        self.self_weight = float(cfg.MODEL.CASS_CAGF_SELF_WEIGHT)
        self.min_agree = float(cfg.MODEL.CASS_CAGF_MIN_AGREE)
        self.agree_tau = float(cfg.MODEL.CASS_CAGF_AGREE_TAU)
        self.max_gate = float(cfg.MODEL.CASS_CAGF_MAX_GATE)
        self.max_residual_norm = float(cfg.MODEL.CASS_CAGF_MAX_RESIDUAL_NORM)
        self.detach_context = bool(cfg.MODEL.CASS_CAGF_DETACH_CONTEXT)
        self.warmup_epochs = int(cfg.MODEL.CASS_CAGF_WARMUP_EPOCHS)
        self.ramp_epochs = int(cfg.MODEL.CASS_CAGF_RAMP_EPOCHS)
        if self.mode not in ('attention', 'agreement'):
            raise ValueError("CASS_CAGF_MODE must be 'attention' or 'agreement', got {}".format(
                self.mode))
        if self.residual_weight < 0.0:
            raise ValueError('CASS_CAGF_RESIDUAL_WEIGHT must be >= 0, got {}'.format(
                self.residual_weight))
        if self.self_weight < 0.0:
            raise ValueError('CASS_CAGF_SELF_WEIGHT must be >= 0, got {}'.format(
                self.self_weight))
        if self.agree_tau <= 0.0:
            raise ValueError('CASS_CAGF_AGREE_TAU must be > 0, got {}'.format(
                self.agree_tau))
        if self.max_gate < 0.0 or self.max_gate > 1.0:
            raise ValueError('CASS_CAGF_MAX_GATE must be in [0, 1], got {}'.format(
                self.max_gate))
        if self.max_residual_norm < 0.0:
            raise ValueError('CASS_CAGF_MAX_RESIDUAL_NORM must be >= 0, got {}'.format(
                self.max_residual_norm))
        if self.warmup_epochs < 0:
            raise ValueError('CASS_CAGF_WARMUP_EPOCHS must be >= 0, got {}'.format(
                self.warmup_epochs))
        if self.ramp_epochs < 0:
            raise ValueError('CASS_CAGF_RAMP_EPOCHS must be >= 0, got {}'.format(
                self.ramp_epochs))
        if self.mode == 'attention':
            self.cross_blocks = nn.ModuleDict()
            for target in MODALITY_ORDER:
                for source in MODALITY_ORDER:
                    if target != source:
                        key = '{}<-{}'.format(target, source)
                        self.cross_blocks[key] = CrossAttention(
                            dim, num_heads=num_heads, qkv_bias=True)
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

    def _epoch_scale(self, epoch):
        if self.residual_weight <= 0.0:
            return 0.0
        if epoch is None:
            return 1.0
        epoch = int(epoch)
        if epoch <= self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch - self.warmup_epochs) / float(self.ramp_epochs))

    def _forward_attention(self, selected, gate_bias, modality_names, quality_scores=None,
                           base_fused=None):
        fused = {}
        for target in modality_names:
            token_cls = selected[target][:, 0, :]
            cls = base_fused[target] if base_fused is not None else token_cls
            batch_size = token_cls.size(0)
            q_target = self._quality(
                quality_scores, target, batch_size, cls.device, cls.dtype)
            residual = torch.zeros_like(cls)
            source_count = 0
            for source in self._ordered_sources(target, modality_names):
                key = '{}<-{}'.format(target, source)
                q_source = self._quality(
                    quality_scores, source, batch_size, cls.device, cls.dtype)
                context = selected[source][:, 1:, :] * q_source.unsqueeze(-1)
                delta = self.cross_blocks[key](cls, context)
                gate = torch.sigmoid(self.gates[key](torch.cat([cls, delta], dim=-1)) + gate_bias)
                gate = gate * q_source
                residual = residual + gate * (delta - cls)
                source_count += 1
            if source_count > 0:
                residual = residual / float(source_count)
            if self.self_weight > 0.0:
                self_delta = self.self_blocks[target](cls, selected[target][:, 1:, :])
                residual = residual + self.self_weight * q_target * (self_delta - cls)
            fused[target] = cls + self.residual_weight * residual
        return fused

    def _forward_agreement(self, selected, gate_bias, modality_names, quality_scores=None,
                           base_fused=None, epoch=None):
        epoch_scale = self._epoch_scale(epoch)
        base = {
            name: base_fused[name] if base_fused is not None else selected[name][:, 0, :]
            for name in modality_names
        }
        if epoch_scale <= 0.0 or len(modality_names) < 2:
            noop = gate_bias.sum(dim=1, keepdim=True).to(
                device=next(iter(base.values())).device,
                dtype=next(iter(base.values())).dtype) * 0.0
            return {name: value + noop for name, value in base.items()}

        fused = {}
        normed = {name: F.normalize(base[name].float(), dim=-1) for name in modality_names}
        bias_mod = 1.0 + 0.25 * torch.tanh(gate_bias.float().mean(dim=-1, keepdim=True))
        for target in modality_names:
            cls = base[target]
            cls_float = cls.float()
            batch_size = cls.size(0)
            q_target = self._quality(
                quality_scores, target, batch_size, cls.device, cls.dtype).float()
            weighted_sum = torch.zeros_like(cls_float)
            weight_sum = torch.zeros(batch_size, 1, device=cls.device, dtype=cls_float.dtype)
            source_count = 0
            for source in self._ordered_sources(target, modality_names):
                source_vec = base[source]
                if self.detach_context:
                    source_vec = source_vec.detach()
                source_float = source_vec.float()
                q_source = self._quality(
                    quality_scores, source, batch_size, cls.device, cls.dtype).float()
                cosine = (normed[target] * F.normalize(source_float, dim=-1)).sum(
                    dim=-1, keepdim=True)
                agreement = torch.sigmoid((cosine - self.min_agree) / self.agree_tau)
                weight = agreement * q_source * bias_mod
                weighted_sum = weighted_sum + weight * source_float
                weight_sum = weight_sum + weight
                source_count += 1

            if source_count == 0:
                fused[target] = cls
                continue
            confidence = (weight_sum / float(source_count)).clamp(0.0, self.max_gate) * q_target
            consensus = weighted_sum / weight_sum.clamp_min(1e-6)
            target_norm = cls_float.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            consensus = F.normalize(consensus, dim=-1) * target_norm
            residual = consensus - cls_float
            if self.max_residual_norm > 0.0:
                residual_norm = residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                max_norm = self.max_residual_norm * target_norm
                residual = residual * (max_norm / residual_norm).clamp(max=1.0)
            update = epoch_scale * self.residual_weight * confidence * residual
            fused[target] = (cls_float + update).to(dtype=cls.dtype)
        return fused

    def forward(self, selected, gate_bias, modality_names, quality_scores=None, base_fused=None,
                epoch=None):
        if self.mode == 'attention':
            return self._forward_attention(
                selected, gate_bias, modality_names,
                quality_scores=quality_scores, base_fused=base_fused)
        return self._forward_agreement(
            selected, gate_bias, modality_names,
            quality_scores=quality_scores, base_fused=base_fused, epoch=epoch)


class CASSModule(nn.Module):
    def __init__(self, dim, cfg, feat_h, feat_w):
        super().__init__()
        self.dim = dim
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.stage = _canonical_stage(getattr(cfg.MODEL, 'CASS_ABLATION_STAGE', 'full'))
        self.stage_rank = CASS_STAGE_RANK[self.stage]
        self.descriptor_mode = str(
            getattr(cfg.MODEL, 'CASS_DESCRIPTOR_MODE', 'summary')).strip().lower()
        if self.descriptor_mode not in ('summary', 'cls'):
            raise ValueError(
                "CASS_DESCRIPTOR_MODE must be 'summary' or 'cls', got {}".format(
                    self.descriptor_mode))
        self.cls_context_weight = float(
            getattr(cfg.MODEL, 'CASS_CLS_CONTEXT_WEIGHT', 0.0))
        if self.cls_context_weight < 0.0:
            raise ValueError('CASS_CLS_CONTEXT_WEIGHT must be >= 0, got {}'.format(
                self.cls_context_weight))
        self.sqt_fallback_alpha = float(cfg.MODEL.CASS_SQT_FALLBACK_ALPHA)
        if self.sqt_fallback_alpha < 0.0 or self.sqt_fallback_alpha > 1.0:
            raise ValueError(
                'CASS_SQT_FALLBACK_ALPHA must be in [0, 1], got {}'.format(
                    self.sqt_fallback_alpha))
        self.sqt_diversity_weight = float(cfg.MODEL.CASS_SQT_DIVERSITY_WEIGHT)
        self.sqt_use_selector = bool(cfg.MODEL.CASS_SQT_USE_SELECTOR)
        self.sqt_fusion_weight = float(cfg.MODEL.CASS_SQT_FUSION_WEIGHT)
        self.sqt_summary_tau = float(cfg.MODEL.CASS_SQT_SUMMARY_TAU)
        self.sqt_agreement_gate = bool(cfg.MODEL.CASS_SQT_AGREEMENT_GATE)
        self.sqt_max_residual_norm = float(cfg.MODEL.CASS_SQT_MAX_RESIDUAL_NORM)
        self.sqt_learnable_gate = bool(cfg.MODEL.CASS_SQT_LEARNABLE_GATE)
        self.sqt_gate_init = float(cfg.MODEL.CASS_SQT_GATE_INIT)
        self.sqt_warmup_epochs = int(cfg.MODEL.CASS_SQT_WARMUP_EPOCHS)
        self.sqt_ramp_epochs = int(cfg.MODEL.CASS_SQT_RAMP_EPOCHS)
        self.nga_residual_weight = float(cfg.MODEL.CASS_NGA_RESIDUAL_WEIGHT)
        self.nga_residual_mode = str(cfg.MODEL.CASS_NGA_RESIDUAL_MODE).strip().lower()
        if self.sqt_fusion_weight < 0.0:
            raise ValueError('CASS_SQT_FUSION_WEIGHT must be >= 0, got {}'.format(
                self.sqt_fusion_weight))
        if self.sqt_summary_tau <= 0.0:
            raise ValueError('CASS_SQT_SUMMARY_TAU must be > 0, got {}'.format(
                self.sqt_summary_tau))
        if self.sqt_max_residual_norm < 0.0:
            raise ValueError('CASS_SQT_MAX_RESIDUAL_NORM must be >= 0, got {}'.format(
                self.sqt_max_residual_norm))
        if self.sqt_warmup_epochs < 0:
            raise ValueError('CASS_SQT_WARMUP_EPOCHS must be >= 0, got {}'.format(
                self.sqt_warmup_epochs))
        if self.sqt_ramp_epochs < 0:
            raise ValueError('CASS_SQT_RAMP_EPOCHS must be >= 0, got {}'.format(
                self.sqt_ramp_epochs))
        if self.nga_residual_weight < 0.0:
            raise ValueError('CASS_NGA_RESIDUAL_WEIGHT must be >= 0, got {}'.format(
                self.nga_residual_weight))
        if self.nga_residual_mode not in ('cross_mean', 'sqt_gate'):
            raise ValueError(
                "CASS_NGA_RESIDUAL_MODE must be 'cross_mean' or 'sqt_gate', got {}".format(
                    self.nga_residual_mode))
        heads = int(cfg.MODEL.CASS_NUM_HEADS)
        self.hss = HighOrderStructureSynergy(dim, feat_h, feat_w, cfg)
        self.sqt = SynergyQueryToken(dim, num_heads=heads, cfg=cfg)
        self.nga = NeighborhoodGuidedAdapter(dim, cfg)
        self.selector = DynamicCollaborativeSelector(cfg)
        self.fusion = ContextAwareGatedFusion(dim, num_heads=heads, cfg=cfg)
        self.sqt_fusion_norm = nn.LayerNorm(dim)
        self.sqt_residual_gate = nn.Parameter(torch.tensor(self.sqt_gate_init)) \
            if self.uses_sqt and self.sqt_learnable_gate else None
        self.sqt_aux_loss = None

    @property
    def uses_hss(self):
        return 'hss' in CASS_STAGE_FEATURES[self.stage]

    @property
    def uses_sqt(self):
        return 'sqt' in CASS_STAGE_FEATURES[self.stage]

    @property
    def uses_nga(self):
        return 'nga' in CASS_STAGE_FEATURES[self.stage]

    @property
    def uses_cagf(self):
        return 'cagf' in CASS_STAGE_FEATURES[self.stage]

    @property
    def uses_full_design(self):
        return self.stage == 'full'

    def set_memory(self, features, keys, modality_names, labels=None, epoch=None):
        if not self.uses_nga:
            self.nga.memory_ready = False
            return
        self.nga.set_memory(features, keys, modality_names, labels=labels, epoch=epoch)

    @staticmethod
    def _all_token_masks(features):
        masks = {}
        for name, feat in features.items():
            masks[name] = torch.ones(
                feat.size(0), feat.size(1) - 1,
                device=feat.device, dtype=torch.bool)
        return masks

    @staticmethod
    def _token_summary(feat, gate=None):
        patches = feat[:, 1:, :]
        if gate is not None:
            denom = gate.sum(dim=1, keepdim=True).to(
                device=patches.device, dtype=patches.dtype).clamp_min(1e-6)
            return feat[:, 0, :] + patches.sum(dim=1) / denom
        return feat[:, 0, :] + patches.mean(dim=1)

    def _summary_dict(self, token_features, gates=None):
        return {
            name: self._token_summary(feat, None if gates is None else gates.get(name))
            for name, feat in token_features.items()
        }

    def _cls_context_summary(self, feat, gate=None):
        cls_token = feat[:, 0, :]
        if self.cls_context_weight <= 0.0:
            return cls_token
        patches = feat[:, 1:, :]
        if gate is not None:
            denom = gate.sum(dim=1, keepdim=True).to(
                device=patches.device, dtype=patches.dtype).clamp_min(1e-6)
            context = patches.sum(dim=1) / denom
        else:
            context = patches.mean(dim=1)
        return cls_token + self.cls_context_weight * context

    def _fused_dict(self, token_features, gates=None):
        if self.descriptor_mode == 'cls':
            return {
                name: self._cls_context_summary(
                    feat, None if gates is None else gates.get(name))
                for name, feat in token_features.items()
            }
        return self._summary_dict(token_features, gates)

    def _sqt_summary(self, feat, score):
        patches = feat[:, 1:, :]
        weight = F.softmax(score.float() / self.sqt_summary_tau, dim=1)
        summary = (patches.float() * weight.unsqueeze(-1)).sum(dim=1)
        return self.sqt_fusion_norm(summary).to(dtype=patches.dtype)

    def _sqt_epoch_scale(self, epoch):
        if self.sqt_fusion_weight <= 0.0:
            return 0.0
        if epoch is None:
            return 1.0
        epoch = int(epoch)
        if epoch <= self.sqt_warmup_epochs:
            return 0.0
        if self.sqt_ramp_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch - self.sqt_warmup_epochs) / float(self.sqt_ramp_epochs))

    def _sqt_gate_scale(self, device, dtype):
        if self.sqt_residual_gate is None:
            return torch.ones((), device=device, dtype=dtype)
        return torch.sigmoid(self.sqt_residual_gate).to(device=device, dtype=dtype)

    def _safe_sqt_residual(self, base, summary, weight):
        residual = weight * summary
        if self.sqt_agreement_gate:
            cosine = F.cosine_similarity(base.float(), summary.float(), dim=-1).unsqueeze(-1)
            residual = residual * cosine.clamp_min(0.0).to(dtype=residual.dtype)
        if self.sqt_max_residual_norm > 0.0:
            base_norm = base.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            residual_float = residual.float()
            residual_norm = residual_float.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            max_norm = self.sqt_max_residual_norm * base_norm
            residual = (
                residual_float * (max_norm / residual_norm).clamp(max=1.0)
            ).to(dtype=residual.dtype)
        return residual

    def _apply_sqt_residual(self, fused, enhanced, structure_scores, modality_names,
                            alpha_dyn=None, epoch=None):
        epoch_scale = self._sqt_epoch_scale(epoch)
        if epoch_scale <= 0.0:
            return fused
        out = dict(fused)
        template = next(iter(fused.values()))
        gate_scale = self._sqt_gate_scale(template.device, template.dtype)
        for idx, name in enumerate(modality_names):
            weight = self.sqt_fusion_weight * epoch_scale * gate_scale
            if alpha_dyn is not None and self.nga_residual_weight > 0.0:
                modality_idx = MODALITY_ORDER.index(name)
                alpha = alpha_dyn[:, modality_idx:modality_idx + 1].to(
                    device=fused[name].device, dtype=fused[name].dtype).clamp(0.0, 1.0)
                weight = weight * (1.0 + self.nga_residual_weight * (2.0 * alpha - 1.0))
                weight = weight.clamp(0.0, 2.0 * self.sqt_fusion_weight)
            summary = self._sqt_summary(enhanced[name], structure_scores[idx])
            out[name] = out[name] + self._safe_sqt_residual(out[name], summary, weight)
        return out

    def _apply_nga_cross_mean_residual(self, fused, alpha_dyn, modality_names):
        if self.nga_residual_weight <= 0.0 or len(modality_names) < 2:
            return fused

        stacked = torch.stack([fused[name] for name in modality_names], dim=1)
        total = stacked.sum(dim=1)
        denom = float(len(modality_names) - 1)
        out = {}
        for local_idx, name in enumerate(modality_names):
            modality_idx = MODALITY_ORDER.index(name)
            other_mean = (total - stacked[:, local_idx, :]) / denom
            alpha = alpha_dyn[:, modality_idx:modality_idx + 1].to(
                device=fused[name].device, dtype=fused[name].dtype).clamp(0.0, 1.0)
            residual = other_mean - fused[name]
            out[name] = fused[name] + self.nga_residual_weight * alpha * residual
        return out

    @staticmethod
    def _descriptor_from_fused(fused, quality_scores=None):
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
        return torch.cat(descriptor_parts, dim=-1)

    def _base_keep_ratio(self, batch_size, num_tokens, device, dtype):
        if self.selector is None or not self.selector.use_dynamic_topk:
            return None
        min_k, max_k = self.selector._topk_bounds(num_tokens)
        if max_k == min_k:
            return None
        base = max(1, min(self.selector.topk, num_tokens))
        ratio = (float(base) - float(min_k)) / float(max_k - min_k)
        ratio = min(max(ratio, 0.0), 1.0)
        return torch.full(
            (batch_size, len(MODALITY_ORDER)),
            ratio, device=device, dtype=dtype)

    def auxiliary_loss(self, device, dtype):
        if self.sqt_aux_loss is None:
            return torch.zeros((), device=device, dtype=dtype)
        return self.sqt_aux_loss.to(device=device, dtype=dtype)

    def forward(self, features, img_path=None, quality_scores=None, epoch=None):
        self.sqt_aux_loss = None
        modality_names = list(features.keys())

        if self.uses_hss:
            enhanced = {}
            self_scores = {}
            for name in modality_names:
                enhanced[name], self_scores[name] = self.hss(features[name], epoch=epoch)
        else:
            enhanced = dict(features)
            self_scores = {}

        cls_list = [features[name][:, 0, :] for name in modality_names]
        if quality_scores is not None:
            template = cls_list[0]
            quality_scores = quality_scores.to(device=template.device, dtype=template.dtype)

        structure_scores = None
        if self.uses_sqt:
            enhanced_list = [enhanced[name] for name in modality_names]
            structure_scores, _, sqt_diversity = self.sqt(enhanced_list)
            self.sqt_aux_loss = self.sqt_diversity_weight * sqt_diversity

        if self.uses_nga:
            alpha_dyn, keep_ratio, gate_bias, _ = self.nga(
                cls_list, modality_names, keys=img_path, quality_scores=quality_scores)
        elif self.uses_sqt:
            template = cls_list[0]
            alpha_dyn = torch.full(
                (template.size(0), len(MODALITY_ORDER)),
                self.sqt_fallback_alpha,
                device=template.device, dtype=template.dtype)
            keep_ratio = self._base_keep_ratio(
                template.size(0), enhanced_list[0].size(1) - 1,
                template.device, template.dtype)
            gate_bias = torch.zeros_like(template)
        else:
            template = cls_list[0]
            alpha_dyn = None
            keep_ratio = None
            gate_bias = torch.zeros_like(template)

        if self.uses_sqt and self.sqt_use_selector:
            if self.selector is None:
                raise RuntimeError('CASS SQT selector is enabled but was not constructed')
            selected = {}
            masks = {}
            gates = {}
            for idx, name in enumerate(modality_names):
                modality_idx = MODALITY_ORDER.index(name)
                alpha = alpha_dyn[:, modality_idx:modality_idx + 1]
                ratio = keep_ratio[:, modality_idx:modality_idx + 1] if keep_ratio is not None else None
                quality = quality_scores[:, modality_idx:modality_idx + 1] \
                    if quality_scores is not None else None
                selected[name], masks[name], gates[name] = self.selector(
                    enhanced[name], self_scores[name], structure_scores[idx], alpha, ratio, quality)
        else:
            selected = enhanced
            masks = self._all_token_masks(selected)
            gates = None

        fused = self._fused_dict(selected, gates)
        if self.uses_nga and self.nga_residual_mode == 'cross_mean':
            fused = self._apply_nga_cross_mean_residual(fused, alpha_dyn, modality_names)
        if self.uses_sqt:
            sqt_alpha = alpha_dyn if self.uses_nga and self.nga_residual_mode == 'sqt_gate' else None
            fused = self._apply_sqt_residual(
                fused, enhanced, structure_scores, modality_names, alpha_dyn=sqt_alpha, epoch=epoch)
        if self.uses_cagf and self.fusion is not None:
            fused = self.fusion(
                selected, gate_bias, modality_names,
                quality_scores=quality_scores, base_fused=fused, epoch=epoch)
        descriptor = self._descriptor_from_fused(fused, quality_scores=quality_scores)
        return selected, masks, fused, descriptor

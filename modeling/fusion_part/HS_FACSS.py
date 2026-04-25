import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRollout(nn.Module):
    """
    Hierarchical Token Selection (HS).

    Cumulative attention rollout with head-mean and 0.5*I residual mixing
    (Abnar & Zuidema, 2020). Captures top-k indices at multiple ViT depths
    and unions them per modality.
    """

    def __init__(self, layers=(4, 8, 12), k=16):
        super().__init__()
        self.layers = tuple(sorted(layers))
        self.k = k

    def forward(self, attn_list):
        """
        attn_list: list of L tensors, each [B, H, M, M] where M = N + 1.

        Returns:
          union_mask: [B, N] bool, True at positions in J_m^hier.
          s_final:    [B, N] cumulative attention score at the deepest
                      requested layer; reused as S_self in FACSS.
          per_depth_scores: dict {layer: [B, N]} for visualization/debug.
        """
        max_layer = max(self.layers)
        device = attn_list[0].device
        B, _, M, _ = attn_list[0].shape
        N = M - 1

        eye = torch.eye(M, device=device).unsqueeze(0)  # [1, M, M]
        cum = None
        snapshots = {}
        for l in range(1, max_layer + 1):
            A = attn_list[l - 1].mean(dim=1)            # head-mean: [B, M, M]
            A_hat = 0.5 * A + 0.5 * eye                 # residual mixing
            cum = A_hat if cum is None else torch.matmul(A_hat, cum)
            if l in self.layers:
                snapshots[l] = cum[:, 0, 1:]            # [B, N]: class-token row

        union_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for l in self.layers:
            s_l = snapshots[l]
            k = min(self.k, N)
            topk_idx = s_l.topk(k, dim=1).indices       # [B, k]
            mask_l = torch.zeros_like(union_mask)
            mask_l.scatter_(1, topk_idx, True)
            union_mask = union_mask | mask_l

        s_final = snapshots[max_layer]
        return union_mask, s_final, snapshots


class HSFACSS(nn.Module):
    """
    Hierarchical Selection (HS) + Fusion-Aware Synergistic Selection (FACSS).

    Replaces SFTS. Per-modality independent selection: HS produces a
    candidate set, FACSS rescues fusion-aware top-K via intra-modal
    discriminability + cross-modal cosine consensus modulated by an
    environment-aware scalar alpha.
    """

    def __init__(self, dim, cfg):
        super().__init__()
        self.dim = dim
        self.hs = HierarchicalRollout(
            layers=tuple(cfg.MODEL.HS_LAYERS),
            k=cfg.MODEL.HS_K,
        )
        self.facss_k = cfg.MODEL.FACSS_K
        self.cross_pool = cfg.MODEL.FACSS_CROSS_POOL
        self.cross_topk = cfg.MODEL.FACSS_CROSS_TOPK
        self.cross_lse_tau = cfg.MODEL.FACSS_CROSS_LSE_TAU
        self.alpha_grain = cfg.MODEL.FACSS_ALPHA_GRANULARITY
        self.norm_mode = cfg.MODEL.FACSS_NORM
        self.use_ste = bool(cfg.MODEL.FACSS_STE)
        self.ste_tau = float(cfg.MODEL.FACSS_STE_TAU)

        hidden = cfg.MODEL.FACSS_ALPHA_HIDDEN
        in_dim = 3 * dim if self.alpha_grain == 'sample' else 4 * dim
        self.alpha_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.alpha_mlp[-1].bias)         # initial alpha ~= 0.5

    @staticmethod
    def _normalize(s, mode, mask=None, eps=1e-6):
        """
        Normalize s [B, N] per-sample, optionally restricted to candidate
        positions via bool mask [B, N] (non-candidate values are excluded
        from min/max/mean/std and zeroed in the output).
        """
        if mask is not None:
            very_neg = torch.finfo(s.dtype).min
            very_pos = torch.finfo(s.dtype).max
            s_for_max = torch.where(mask, s, torch.full_like(s, very_neg))
            s_for_min = torch.where(mask, s, torch.full_like(s, very_pos))
        else:
            s_for_max = s
            s_for_min = s

        if mode == 'minmax':
            lo = s_for_min.min(dim=1, keepdim=True).values
            hi = s_for_max.max(dim=1, keepdim=True).values
            out = (s - lo) / (hi - lo + eps)
        elif mode == 'robust':
            # quantile across candidate positions only
            if mask is not None:
                out = torch.zeros_like(s)
                for b in range(s.size(0)):
                    sb = s[b][mask[b]]
                    if sb.numel() == 0:
                        continue
                    lo = torch.quantile(sb, 0.05)
                    hi = torch.quantile(sb, 0.95)
                    sb_clipped = sb.clamp(lo, hi)
                    out[b][mask[b]] = (sb_clipped - lo) / (hi - lo + eps)
                return out
            lo = torch.quantile(s, 0.05, dim=1, keepdim=True)
            hi = torch.quantile(s, 0.95, dim=1, keepdim=True)
            s_clip = s.clamp(lo, hi)
            out = (s_clip - lo) / (hi - lo + eps)
        elif mode == 'zscore':
            if mask is not None:
                cnt = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
                mu = (s * mask).sum(dim=1, keepdim=True) / cnt
                var = (((s - mu) * mask) ** 2).sum(dim=1, keepdim=True) / cnt
                sd = var.sqrt()
            else:
                mu = s.mean(dim=1, keepdim=True)
                sd = s.std(dim=1, keepdim=True)
            out = (s - mu) / (sd + eps)
        else:
            raise ValueError(mode)

        if mask is not None:
            out = out * mask.to(out.dtype)
        return out

    def _cross_score(self, p_feats, others_feats):
        """
        p_feats: [B, N, D] modality m's full patch features at layer 12.
        others_feats: list of [B, N, D] other-modality patch features.
        Returns S_cross: [B, N].
        """
        p_norm = F.normalize(p_feats, dim=-1)
        scores = []
        for q in others_feats:
            q_norm = F.normalize(q, dim=-1)
            cos = torch.matmul(p_norm, q_norm.transpose(-1, -2))   # [B, N, N]
            cos = F.relu(cos)
            if self.cross_pool == 'max':
                pooled = cos.max(dim=-1).values
            elif self.cross_pool == 'topk':
                k = min(self.cross_topk, cos.size(-1))
                pooled = cos.topk(k, dim=-1).values.mean(dim=-1)
            elif self.cross_pool == 'lse':
                tau = self.cross_lse_tau
                pooled = (1.0 / tau) * torch.logsumexp(tau * cos, dim=-1)
            else:
                raise ValueError(self.cross_pool)
            scores.append(pooled)
        return torch.stack(scores, dim=0).mean(dim=0)              # [B, N]

    def _alpha(self, cls_list, p_feats=None):
        """
        cls_list: list of [B, D] class tokens from each modality.
        p_feats:  [B, N, D] only used when granularity == 'token'.
        Returns alpha: [B, 1] (sample) or [B, N] (token).
        """
        ctx = torch.cat(cls_list, dim=-1)                          # [B, kD]
        if self.alpha_grain == 'sample':
            return torch.sigmoid(self.alpha_mlp(ctx))              # [B, 1]
        # token-level: broadcast ctx and concat per-token feature
        B, N, D = p_feats.shape
        ctx_exp = ctx.unsqueeze(1).expand(-1, N, -1)               # [B, N, kD]
        inp = torch.cat([p_feats, ctx_exp], dim=-1)                # [B, N, (k+1)D]
        return torch.sigmoid(self.alpha_mlp(inp).squeeze(-1))      # [B, N]

    def _select_one_modality(self, m_feat, m_attn, others_full,
                             cls_list, mask_fre):
        """
        Run HS + FACSS for one modality.
        m_feat: [B, M, D] full token sequence (cls + N patches).
        m_attn: list of L attention tensors.
        others_full: list of [B, N, D] other-modality patches at layer 12.
        cls_list: [cls_rgb, cls_nir, cls_tir] from un-modified backbone.
        mask_fre: [B, N] bool or None.
        Returns: feat_out [B, M, D] (zero-padded outside selection),
                 sel_mask [B, N] bool of finally selected positions.
        """
        cls_token = m_feat[:, :1, :]                               # [B, 1, D]
        patches = m_feat[:, 1:, :]                                 # [B, N, D]
        B, N, D = patches.shape

        hs_mask, s_self_full, _ = self.hs(m_attn)                  # both per modality
        cand_mask = hs_mask
        if mask_fre is not None:
            cand_mask = cand_mask | mask_fre.bool()

        s_cross_full = self._cross_score(patches, others_full)     # [B, N]

        s_self_norm = self._normalize(s_self_full, self.norm_mode, mask=cand_mask)
        s_cross_norm = self._normalize(s_cross_full, self.norm_mode, mask=cand_mask)

        if self.alpha_grain == 'sample':
            alpha = self._alpha(cls_list)                          # [B, 1]
            s_facss = s_self_norm + alpha * s_cross_norm
        else:
            alpha = self._alpha(cls_list, p_feats=patches)         # [B, N]
            s_facss = s_self_norm + alpha * s_cross_norm

        # Restrict scoring to candidate positions: non-candidates get -inf
        very_neg = torch.finfo(s_facss.dtype).min
        s_facss = torch.where(cand_mask, s_facss, torch.full_like(s_facss, very_neg))

        # Per-sample candidate counts may be < facss_k → clamp k accordingly
        k = min(self.facss_k, N)
        topk_idx = s_facss.topk(k, dim=1).indices                  # [B, k]
        sel_mask = torch.zeros(B, N, dtype=torch.bool, device=patches.device)
        sel_mask.scatter_(1, topk_idx, True)

        # Build the gating tensor applied to patches.
        # STE: forward = hard sel_mask; backward = softmax over candidate scores
        # (so alpha_mlp and the rollout-derived scores receive gradient).
        sel_mask_f = sel_mask.float()
        if self.use_ste and self.training:
            soft = F.softmax(s_facss / self.ste_tau, dim=1)        # [B, N]
            gate = sel_mask_f + soft - soft.detach()
        else:
            gate = sel_mask_f

        feat_out = torch.cat(
            [cls_token, patches * gate.unsqueeze(-1)],
            dim=1,
        )
        return feat_out, sel_mask

    def forward(self, RGB_feat, RGB_attn, NIR_feat=None, NIR_attn=None,
                TIR_feat=None, TIR_attn=None, img_path=None,
                writer=None, epoch=None, mask_fre=None):
        cls_rgb = RGB_feat[:, 0, :]
        patches_rgb = RGB_feat[:, 1:, :]

        modalities = [('RGB', RGB_feat, RGB_attn, patches_rgb, cls_rgb)]
        if NIR_feat is not None:
            modalities.append(('NIR', NIR_feat, NIR_attn, NIR_feat[:, 1:, :], NIR_feat[:, 0, :]))
        if TIR_feat is not None:
            modalities.append(('TIR', TIR_feat, TIR_attn, TIR_feat[:, 1:, :], TIR_feat[:, 0, :]))

        cls_list = [m[4] for m in modalities]
        # alpha_mlp is sized for 3 modalities; pad with a zero cls token when
        # fewer modalities are present (e.g. 2-modal datasets like RGBN300).
        # Zero context degrades alpha gracefully toward its bias-only response.
        while len(cls_list) < 3:
            cls_list.append(torch.zeros_like(cls_list[0]))
        full_patches = {m[0]: m[3] for m in modalities}

        outs = {}
        masks = {}
        for name, feat, attn, _, _ in modalities:
            others = [full_patches[k] for k in full_patches if k != name]
            feat_out, sel_mask = self._select_one_modality(
                feat, attn, others, cls_list, mask_fre,
            )
            outs[name] = feat_out
            masks[name] = sel_mask

        # Maintain SFTS-style return tuple ordering for make_model.py.
        if 'TIR' in outs:
            return outs['RGB'], outs['NIR'], outs['TIR'], (masks['RGB'], masks['NIR'], masks['TIR'])
        return outs['RGB'], outs['NIR'], (masks['RGB'], masks['NIR'])

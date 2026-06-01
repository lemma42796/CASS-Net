"""
End-to-end pipeline smoke test for HTL-ReID.

Runs on CPU by default. Verifies that a fresh checkout with the published
configs can build the model and complete forward + backward without CUDA, real
data, or pretrained weights. On a GPU server, set CASS_TEST_DEVICE=cuda or
CASS_TEST_DEVICE=auto to run the model tests on CUDA.

Coverage:
  1. cfg defaults load and freeze
  2. Each shipped yml config merges cleanly into defaults
  3. Model builds for each yml (PRETRAIN_CHOICE forced off so no .pth needed)
  4. 3-modal training forward returns the right tuple length and shapes
  5. 3-modal eval forward
  6. 2-modal forward_two_modalities (RGBN300-style path)
  7. Backward populates non-NaN gradients on every trainable parameter
 8. Loss assembly matches engine/processor.py's odd/even pairing rule
 9. state_dict save -> reload -> identical forward output
 10. local/timm pretrained source loader
 11. Ablation switches (AGF=0, OCFR=1) each produce a usable model
 12. NGA memory warmup skips feature extraction
 13. HSS gate floor unblocks graph-branch gradients
 14. HSS structural score mix changes selector signal
 15. Hypergraph broadcast path matches the old diagonal math
  16. Staged resume weight loader, including full training checkpoint payloads
  17. Full training checkpoint save -> resume round-trip
  18. AMP dtype selection keeps bf16 unscaled and HSS autocast-safe
 19. SQT fallback alpha and gate-normalized summaries avoid hard SQT-only selection
 20. SQT additive summary keeps tokens intact while preserving SQT gradients
 21. NGA residual keeps the additive SQT repair path active when selector is off
 22. CA-GF residual keeps the repaired A3 descriptor as the identity path
 23. CA-GF agreement mode suppresses conflicting cross-modal context

Run:
    python3 test_pipeline.py
    CASS_TEST_DEVICE=cuda python3 test_pipeline.py
"""
import copy
import io
import os
import sys
import types
import torch
import torch.nn.functional as F

from config import cfg as default_cfg
from modeling.make_model import make_model


YMLS = [
    'configs/RGBNT201/default.yml',
    'configs/Market1501-MM/default.yml',
    'configs/MSVR310/default.yml',
    'configs/RGBNT100/default.yml',
]

NUM_CLASSES = 8
BATCH = 2


def _resolve_device():
    requested = os.environ.get('CASS_TEST_DEVICE', 'cpu').strip().lower()
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CASS_TEST_DEVICE={} requested, but CUDA is unavailable'.format(requested))
    return torch.device(requested)


DEVICE = _resolve_device()


def _make_cfg(yml=None, **overrides):
    c = default_cfg.clone()
    if yml is not None:
        c.merge_from_file(yml)
    c.MODEL.PRETRAIN_CHOICE = 'self'        # skip ImageNet weight load
    c.MODEL.SIE_CAMERA = False              # avoid camera_num plumbing in tests
    c.INPUT.SIZE_TRAIN = [128, 64]           # keep CPU smoke tests quick
    c.INPUT.SIZE_TEST = [128, 64]
    c.MODEL.CASS_HSS_EDGES = 16
    c.MODEL.CASS_HSS_FILTERS = 32
    c.MODEL.CASS_TOPK = 8
    c.MODEL.CASS_NGA_MEMORY = 0
    for k, v in overrides.items():
        # supports dotted MODEL.XYZ overrides
        node = c
        parts = k.split('.')
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    return c


def _expected_train_outputs(cfg, modalities=3):
    if cfg.MODEL.AL:
        base = 5
    else:
        base = 2 + 2 * modalities + 1
    if cfg.MODEL.METHOD.upper() == 'CASS' and cfg.MODEL.CASS_PART_BRANCH:
        base += 2
    if cfg.MODEL.METHOD.upper() != 'CASS' and cfg.MODEL.PART_BRANCH:
        base += 2
    return base


def _dummy_batch(cfg, modalities=('RGB', 'NI', 'TI'), device=None):
    H, W = cfg.INPUT.SIZE_TRAIN
    device = DEVICE if device is None else device
    return {m: torch.randn(BATCH, 3, H, W, device=device) for m in modalities}


def _assert_finite(t, name):
    assert torch.isfinite(t).all(), '{} has NaN/Inf'.format(name)


def _cuda_autocast(enabled, dtype):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda', enabled=enabled, dtype=dtype)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def _cuda_grad_scaler(enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler('cuda', enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _loss_assembly_like_processor(output):
    """Mirror engine/processor.py:80-92 pairing rule (odd len => last is aux)."""
    first = output[0]
    loss = torch.zeros((), device=first.device, dtype=first.dtype)
    if len(output) % 2 == 1:
        for i in range(0, len(output) - 1, 2):
            loss = loss + output[i].sum() + output[i + 1].sum()
        loss = loss + output[-1]
    else:
        for i in range(0, len(output), 2):
            loss = loss + output[i].sum() + output[i + 1].sum()
    return loss


def test_defaults_load():
    print('[1] cfg defaults clone+freeze')
    c = default_cfg.clone()
    c.freeze()
    assert c.MODEL.AGF in (0, 1)
    print('     OK')


def test_yml_configs_merge():
    print('[2] each yml merges into defaults')
    for y in YMLS:
        c = default_cfg.clone()
        c.merge_from_file(y)
        c.freeze()
        print('     OK: {}  AGF={} AL={}'.format(
            y, c.MODEL.AGF, c.MODEL.AL))


def test_three_modal_pipeline(yml):
    print('[3] 3-modal train+eval | {}'.format(yml))
    cfg = _make_cfg(yml)
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    model.train()

    x = _dummy_batch(cfg)
    label = torch.randint(0, NUM_CLASSES, (BATCH,), device=DEVICE)

    output = model(x, cam_label=None, label=label, epoch=0)
    assert isinstance(output, tuple), 'expected tuple, got {}'.format(type(output))

    expected_len = _expected_train_outputs(cfg, modalities=3)
    assert len(output) == expected_len, 'expected {} outputs, got {}'.format(
        expected_len, len(output))

    score, cls4t = output[0], output[1]
    assert score.shape == (BATCH, NUM_CLASSES), 'score shape {}'.format(score.shape)
    assert cls4t.shape == (BATCH, 3 * model.BACKBONE.token_dim), \
        'cls4t shape {}'.format(cls4t.shape)
    for i, t in enumerate(output):
        _assert_finite(t, 'output[{}]'.format(i))

    # Backward via the same loss assembly the trainer uses
    loss = _loss_assembly_like_processor(output)
    loss.backward()

    bad = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            bad.append((n, 'grad is None'))
        elif not torch.isfinite(p.grad).all():
            bad.append((n, 'NaN/Inf in grad'))
    # Some params are intentionally unused in this forward path. Filter them.
    # - AL_HEAD/BN: only used when MODEL.AL=1
    # - BACKBONE_HEAD/BN: only used when MODEL.AL=0
    # - BACKBONE.base.fc: ViT's ImageNet classifier, never used (we use embeddings)
    # - CASS.nga.topk_mlp: predicts a hard keep count, so the count path is non-differentiable
    # - CASS.sqt_fusion_norm: only active when CASS_SQT_FUSION_WEIGHT > 0
    unused_ok = (
        'AL_HEAD', 'AL_BN', 'BACKBONE_HEAD', 'BACKBONE_BN', 'BACKBONE.base.fc',
        'CASS.nga.topk_mlp', 'CASS.sqt_fusion_norm',
    )
    bad = [b for b in bad
           if not (b[1] == 'grad is None' and any(u in b[0] for u in unused_ok))]
    assert not bad, 'gradient issues:\n  ' + '\n  '.join('{} -- {}'.format(*b) for b in bad)
    print('     train fwd+bwd OK ({} outputs, all grads finite)'.format(len(output)))

    # Eval
    model.eval()
    with torch.no_grad():
        cls_eval = model(x, cam_label=None, epoch=0)
    assert cls_eval.shape == (BATCH, 3 * model.BACKBONE.token_dim)
    _assert_finite(cls_eval, 'eval cls4t')
    print('     eval fwd OK shape={}'.format(tuple(cls_eval.shape)))


def test_two_modal_pipeline():
    print('[4] 2-modal forward_two_modalities (AL=0)')
    cfg = _make_cfg('configs/RGBNT201/default.yml', **{'MODEL.AL': 0})
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    model.train()

    x = _dummy_batch(cfg, modalities=('RGB', 'NI'))
    label = torch.randint(0, NUM_CLASSES, (BATCH,), device=DEVICE)
    output = model.forward_two_modalities(x, cam_label=None, label=label, epoch=0)

    expected_len = _expected_train_outputs(cfg, modalities=2)
    assert len(output) == expected_len, 'expected {} outputs, got {}'.format(expected_len, len(output))
    score, cls4t = output[0], output[1]
    assert score.shape == (BATCH, NUM_CLASSES)
    assert cls4t.shape == (BATCH, 3 * model.BACKBONE.token_dim)
    for i, t in enumerate(output):
        _assert_finite(t, 'output[{}]'.format(i))
    loss = _loss_assembly_like_processor(output)
    loss.backward()
    print('     train fwd+bwd OK ({} outputs)'.format(len(output)))

    model.eval()
    with torch.no_grad():
        cls_eval = model.forward_two_modalities(x, cam_label=None, epoch=0)
    assert cls_eval.shape == (BATCH, 3 * model.BACKBONE.token_dim)
    print('     eval fwd OK')


def test_save_load_roundtrip():
    print('[5] state_dict save/load round-trip preserves output')
    cfg = _make_cfg('configs/RGBNT201/default.yml')
    torch.manual_seed(42)
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    model.eval()
    x = _dummy_batch(cfg)
    with torch.no_grad():
        out_before = model(x, cam_label=None, epoch=0)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, map_location='cpu', weights_only=True)

    torch.manual_seed(42)
    model2 = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    # Use the model's own load_param so we exercise the public API
    buf2 = io.BytesIO()
    torch.save(sd, buf2)
    buf2.seek(0)
    # write to a temp file because load_param expects a path-like
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as fh:
        torch.save(sd, fh.name)
        path = fh.name
    model2.load_param(path)
    model2.eval()
    with torch.no_grad():
        out_after = model2(x, cam_label=None, epoch=0)
    diff = (out_before - out_after).abs().max().item()
    assert diff < 1e-5, 'roundtrip diff {} > 1e-5'.format(diff)
    print('     OK (max abs diff = {:.2e})'.format(diff))


def test_timm_pretrain_source_loader():
    print('[6] timm:// pretrained source loader')
    from modeling.backbones.vit_pytorch import _load_pretrained_state_dict

    expected_name = 'vit_base_patch16_224.augreg2_in21k_ft_in1k'

    class FakeTimmModel:
        def state_dict(self):
            return {
                'cls_token': torch.ones(1, 1, 768),
                'head.weight': torch.ones(1000, 768),
            }

    def create_model(name, pretrained=True):
        assert name == expected_name
        assert pretrained is True
        return FakeTimmModel()

    old_timm = sys.modules.get('timm')
    sys.modules['timm'] = types.SimpleNamespace(create_model=create_model)
    try:
        state = _load_pretrained_state_dict('timm://{}'.format(expected_name))
    finally:
        if old_timm is None:
            del sys.modules['timm']
        else:
            sys.modules['timm'] = old_timm

    assert state['cls_token'].shape == (1, 1, 768)
    assert state['head.weight'].shape == (1000, 768)
    print('     OK')


def test_local_pretrain_source_loader():
    print('[7] local pretrained source loader')
    from pathlib import Path
    import tempfile
    from modeling.backbones.vit_pytorch import _load_pretrained_state_dict

    with tempfile.TemporaryDirectory() as td:
        bin_path = Path(td) / 'pytorch_model.bin'
        torch.save({'model': {'cls_token': torch.ones(1, 1, 768)}}, str(bin_path))
        state = _load_pretrained_state_dict(td)
        assert state['cls_token'].shape == (1, 1, 768)

    expected = {'pos_embed': torch.ones(1, 197, 768)}

    def load_file(path):
        assert path.endswith('model.safetensors')
        return expected

    old_pkg = sys.modules.get('safetensors')
    old_mod = sys.modules.get('safetensors.torch')
    safetensors_pkg = types.ModuleType('safetensors')
    safetensors_torch = types.ModuleType('safetensors.torch')
    safetensors_torch.load_file = load_file
    sys.modules['safetensors'] = safetensors_pkg
    sys.modules['safetensors.torch'] = safetensors_torch
    try:
        with tempfile.TemporaryDirectory() as td:
            safe_path = Path(td) / 'model.safetensors'
            safe_path.write_bytes(b'fake')
            state = _load_pretrained_state_dict(td)
    finally:
        if old_pkg is None:
            del sys.modules['safetensors']
        else:
            sys.modules['safetensors'] = old_pkg
        if old_mod is None:
            del sys.modules['safetensors.torch']
        else:
            sys.modules['safetensors.torch'] = old_mod

    assert state['pos_embed'].shape == (1, 197, 768)
    print('     OK')


def test_ablation_switches():
    print('[8] ablation switches')
    base_yml = 'configs/RGBNT201/default.yml'
    matrix = [
        {'MODEL.AGF': 0, 'MODEL.OCFR': 0},
        {'MODEL.AGF': 1, 'MODEL.OCFR': 1},
    ]
    for m in matrix:
        cfg = _make_cfg(base_yml, **m)
        model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
        model.train()
        x = _dummy_batch(cfg)
        label = torch.randint(0, NUM_CLASSES, (BATCH,), device=DEVICE)
        out = model(x, cam_label=None, label=label, epoch=0)
        loss = _loss_assembly_like_processor(out)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print('     OK  AGF={} OCFR={}  params={:.2f}M'.format(
            m['MODEL.AGF'], m['MODEL.OCFR'], n_params))


def test_nga_memory_stabilization():
    print('[9] NGA memory warmup+EMA+prototype')
    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_NGA_MEMORY': 1,
            'MODEL.CASS_NGA_WARMUP_EPOCHS': 2,
            'MODEL.CASS_NGA_EMA_MOMENTUM': 0.5,
            'MODEL.CASS_NGA_USE_PROTOTYPE': 1,
        }
    )
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    dim = model.BACKBONE.token_dim
    features = torch.randn(4, 3, dim, device=DEVICE)
    keys = ['a', 'b', 'c', 'd']
    labels = [0, 0, 1, 1]

    model.CASS.set_memory(features, keys, ['RGB', 'NIR', 'TIR'], labels=labels, epoch=1)
    assert not model.CASS.nga.memory_ready

    model.CASS.set_memory(features, keys, ['RGB', 'NIR', 'TIR'], labels=labels, epoch=3)
    assert model.CASS.nga.memory_ready
    assert model.CASS.nga.prototype_neighbors
    before = model.CASS.nga.memory_tensor.clone()

    model.CASS.set_memory(features + 0.1, keys, ['RGB', 'NIR', 'TIR'], labels=labels, epoch=4)
    after = model.CASS.nga.memory_tensor
    assert torch.isfinite(after).all()
    assert (before - after).abs().sum().item() > 0
    print('     OK')


def test_nga_warmup_skips_loader_iteration():
    print('[10] NGA warmup skips memory feature extraction')
    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_NGA_MEMORY': 1,
            'MODEL.CASS_NGA_WARMUP_EPOCHS': 2,
        }
    )
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    model.train()
    model.CASS.nga.memory_ready = True

    class RaisingLoader:
        def __iter__(self):
            raise AssertionError('loader should not be iterated during NGA warmup')

    model.refresh_nga_memory(RaisingLoader(), device=DEVICE, epoch=1)
    assert model.training
    assert not model.CASS.nga.memory_ready
    print('     OK')


def test_hss_graph_warmup():
    print('[12] HSS graph residual warmup')
    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_HSS_GRAPH_WEIGHT': 0.6,
            'MODEL.CASS_HSS_GRAPH_WARMUP_EPOCHS': 3,
        }
    )
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    hss = model.CASS.hss
    expected = [
        (None, 0.6),
        (0, 0.0),
        (1, 0.2),
        (2, 0.4),
        (3, 0.6),
        (10, 0.6),
    ]
    for epoch, target in expected:
        got = hss.current_graph_weight(epoch)
        assert abs(got - target) < 1e-6, 'epoch {}: {} != {}'.format(epoch, got, target)

    model.train()
    x = _dummy_batch(cfg)
    label = torch.randint(0, NUM_CLASSES, (BATCH,), device=DEVICE)
    out = model(x, cam_label=None, label=label, epoch=1)
    assert isinstance(out, tuple)
    print('     OK')


def test_hss_zero_gate_matches_graph0():
    print('[11] HSS zero-gate residual starts as graph0')
    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_HSS_GRAPH_WEIGHT': 0.6,
            'MODEL.CASS_HSS_GATE_INIT': 0.0,
            'MODEL.CASS_HSS_GATE_FLOOR': 0.0,
        }
    )
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    hss = model.CASS.hss
    dim = model.BACKBONE.token_dim
    patches = model.feat_h * model.feat_w
    feat = torch.randn(BATCH, patches + 1, dim, device=DEVICE)

    out_zero_gate, score_zero_gate = hss(feat, epoch=1)
    old_weight = hss.graph_weight
    hss.graph_weight = 0.0
    out_graph0, score_graph0 = hss(feat, epoch=1)
    hss.graph_weight = old_weight

    assert torch.allclose(out_zero_gate, out_graph0, atol=1e-6, rtol=1e-5)
    assert torch.allclose(score_zero_gate, score_graph0, atol=1e-6, rtol=1e-5)
    gate_mean, gate_abs = hss.current_residual_gate()
    assert abs(gate_mean) < 1e-12 and abs(gate_abs) < 1e-12
    print('     OK')


def test_hss_gate_floor_unblocks_graph_gradients():
    print('[13] HSS gate floor unblocks graph-branch gradients')
    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_HSS_GATE_INIT': 0.0,
            'MODEL.CASS_HSS_GATE_FLOOR': 0.02,
            'MODEL.CASS_HSS_GATE_FLOOR_WARMUP_EPOCHS': 2,
        }
    )
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    hss = model.CASS.hss
    dim = model.BACKBONE.token_dim
    patches = model.feat_h * model.feat_w
    feat = torch.randn(BATCH, patches + 1, dim, device=DEVICE)

    assert abs(hss.current_gate_floor(0)) < 1e-12
    assert abs(hss.current_gate_floor(1) - 0.01) < 1e-6
    assert abs(hss.current_gate_floor(2) - 0.02) < 1e-6
    gate_mean, gate_abs = hss.current_residual_gate(epoch=2)
    assert abs(gate_mean - 0.02) < 1e-6
    assert abs(gate_abs - 0.02) < 1e-6

    out, score = hss(feat, epoch=2)
    loss = out[:, 1:, :].float().pow(2).mean() + score.float().pow(2).mean()
    loss.backward()
    grad_total = 0.0
    for name, param in hss.named_parameters():
        if ('hypergraph' in name or 'residual_adapter' in name) and param.grad is not None:
            grad_total += param.grad.detach().abs().sum().item()
    assert grad_total > 0.0
    print('     OK')


def test_hss_structure_score_mix_changes_self_score():
    print('[14] HSS structural score mix changes selector signal')
    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_HSS_SCORE_MIX': 0.5,
            'MODEL.CASS_HSS_SCORE_SOURCE': 'residual',
            'MODEL.CASS_HSS_SCORE_DETACH': 0,
        }
    )
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    hss = model.CASS.hss
    dim = model.BACKBONE.token_dim
    patches = model.feat_h * model.feat_w
    feat = torch.randn(BATCH, patches + 1, dim, device=DEVICE)

    hss.score_mix = 0.0
    _, cls_score = hss(feat, epoch=1)
    hss.score_mix = 0.5
    _, mixed_score = hss(feat, epoch=1)

    _assert_finite(cls_score, 'hss cls-only score')
    _assert_finite(mixed_score, 'hss mixed structure score')
    assert mixed_score.min().item() >= -1e-6
    assert mixed_score.max().item() <= 1.0 + 1e-6
    assert not torch.allclose(cls_score, mixed_score, atol=1e-6, rtol=1e-5)
    print('     OK')


def test_sqt_fallback_alpha_and_weighted_summary():
    print('[19] SQT fallback alpha and gate-normalized summary')
    from modeling.fusion_part.CASS import CASSModule, DynamicCollaborativeSelector

    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_ABLATION_STAGE': 'hss_sqt',
            'MODEL.CASS_SQT_FALLBACK_ALPHA': 0.35,
            'MODEL.CASS_DYNAMIC_TOPK': 0,
            'MODEL.CASS_TOPK': 2,
            'MODEL.CASS_SOFT_RESIDUAL_WEIGHT': 0.25,
        }
    )
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    assert abs(model.CASS.sqt_fallback_alpha - 0.35) < 1e-12

    selector = DynamicCollaborativeSelector(cfg).to(DEVICE)
    feat = torch.zeros(1, 5, 2, device=DEVICE)
    feat[:, 1:, 0] = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=DEVICE)
    score_self = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE)
    score_structure = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=DEVICE)
    alpha = torch.full((1, 1), 0.5, device=DEVICE)
    selected, mask, gate = selector(feat, score_self, score_structure, alpha)

    assert mask.sum().item() == 2
    assert gate.min().item() >= 0.25 - 1e-6
    summary = CASSModule._token_summary(selected, gate)
    expected = selected[:, 1:, :].sum(dim=1) / gate.sum(dim=1, keepdim=True)
    assert torch.allclose(summary, expected, atol=1e-6, rtol=1e-5)
    print('     OK')


def test_sqt_additive_summary_keeps_tokens():
    print('[20] SQT additive summary keeps tokens intact')
    from modeling.fusion_part.CASS import CASSModule

    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_ABLATION_STAGE': 'hss_sqt',
            'MODEL.CASS_NUM_HEADS': 4,
            'MODEL.CASS_HSS_EDGES': 4,
            'MODEL.CASS_HSS_FILTERS': 8,
            'MODEL.CASS_SQT_USE_SELECTOR': 0,
            'MODEL.CASS_SQT_FUSION_WEIGHT': 0.2,
            'MODEL.CASS_SQT_DIVERSITY_WEIGHT': 0.0,
        }
    )
    cass = CASSModule(dim=24, cfg=cfg, feat_h=2, feat_w=2).to(DEVICE)
    cass.train()
    features = {
        name: torch.randn(BATCH, 5, 24, device=DEVICE)
        for name in ('RGB', 'NIR', 'TIR')
    }
    selected, masks, fused, descriptor = cass(features, epoch=1)

    assert descriptor.shape == (BATCH, 72)
    for name in ('RGB', 'NIR', 'TIR'):
        assert selected[name].shape == features[name].shape
        assert masks[name].all(), '{} mask should keep every token'.format(name)
        assert fused[name].shape == (BATCH, 24)

    loss = descriptor.sum()
    loss.backward()
    grad = cass.sqt.query_token.grad
    assert grad is not None and torch.isfinite(grad).all()
    print('     OK')


def test_nga_residual_keeps_additive_path_active():
    print('[21] NGA residual affects additive SQT path')
    from modeling.fusion_part.CASS import CASSModule

    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_ABLATION_STAGE': 'hss_sqt_nga',
            'MODEL.CASS_NUM_HEADS': 4,
            'MODEL.CASS_HSS_EDGES': 4,
            'MODEL.CASS_HSS_FILTERS': 8,
            'MODEL.CASS_SQT_USE_SELECTOR': 0,
            'MODEL.CASS_SQT_FUSION_WEIGHT': 0.2,
            'MODEL.CASS_SQT_DIVERSITY_WEIGHT': 0.0,
            'MODEL.CASS_NGA_MEMORY': 1,
            'MODEL.CASS_NGA_WARMUP_EPOCHS': 0,
            'MODEL.CASS_NGA_EMA_MOMENTUM': 0.0,
            'MODEL.CASS_NGA_USE_PROTOTYPE': 0,
            'MODEL.CASS_NGA_RESIDUAL_WEIGHT': 0.2,
            'MODEL.CASS_NGA_RESIDUAL_MODE': 'sqt_gate',
        }
    )
    torch.manual_seed(23)
    features = {
        name: torch.randn(BATCH, 5, 24, device=DEVICE)
        for name in ('RGB', 'NIR', 'TIR')
    }
    memory = torch.randn(BATCH, 3, 24, device=DEVICE)
    keys = ['a', 'b']

    torch.manual_seed(31)
    no_memory = CASSModule(dim=24, cfg=cfg, feat_h=2, feat_w=2).to(DEVICE)
    torch.manual_seed(31)
    with_memory = CASSModule(dim=24, cfg=cfg, feat_h=2, feat_w=2).to(DEVICE)
    with_memory.set_memory(memory, keys, ['RGB', 'NIR', 'TIR'], labels=[0, 1], epoch=1)

    no_memory.eval()
    with_memory.eval()
    with torch.no_grad():
        desc_no_memory = no_memory(features, img_path=keys, epoch=1)[3]
        desc_with_memory = with_memory(features, img_path=keys, epoch=1)[3]
        desc_with_query_keys = with_memory(features, img_path=['query_a', 'query_b'], epoch=1)[3]

    delta = (desc_no_memory - desc_with_memory).abs().max().item()
    assert delta > 1e-7, 'NGA memory should affect descriptor when residual is enabled'
    delta_query = (desc_no_memory - desc_with_query_keys).abs().max().item()
    assert delta_query > 1e-7, 'NGA memory should affect descriptor for unseen query keys'
    _assert_finite(desc_with_memory, 'NGA residual descriptor')
    _assert_finite(desc_with_query_keys, 'NGA query-anchor residual descriptor')
    print('     OK delta={:.6f}, query_delta={:.6f}'.format(delta, delta_query))


def test_cagf_residual_preserves_repaired_a3_identity():
    print('[22] CA-GF residual preserves repaired A3 identity path')
    from modeling.fusion_part.CASS import CASSModule

    base_overrides = {
        'MODEL.CASS_NUM_HEADS': 4,
        'MODEL.CASS_HSS_EDGES': 4,
        'MODEL.CASS_HSS_FILTERS': 8,
        'MODEL.CASS_SQT_USE_SELECTOR': 0,
        'MODEL.CASS_SQT_FUSION_WEIGHT': 0.2,
        'MODEL.CASS_SQT_DIVERSITY_WEIGHT': 0.0,
        'MODEL.CASS_NGA_MEMORY': 1,
        'MODEL.CASS_NGA_WARMUP_EPOCHS': 0,
        'MODEL.CASS_NGA_EMA_MOMENTUM': 0.0,
        'MODEL.CASS_NGA_USE_PROTOTYPE': 0,
        'MODEL.CASS_NGA_RESIDUAL_WEIGHT': 0.2,
        'MODEL.CASS_NGA_RESIDUAL_MODE': 'sqt_gate',
        'MODEL.CASS_CAGF_MODE': 'agreement',
        'MODEL.CASS_CAGF_MIN_AGREE': -1.0,
        'MODEL.CASS_CAGF_AGREE_TAU': 0.25,
        'MODEL.CASS_CAGF_MAX_GATE': 0.5,
        'MODEL.CASS_CAGF_MAX_RESIDUAL_NORM': 1.0,
        'MODEL.CASS_CAGF_WARMUP_EPOCHS': 0,
        'MODEL.CASS_CAGF_RAMP_EPOCHS': 0,
    }
    cfg_a3 = _make_cfg(
        'configs/RGBNT201/default.yml',
        **dict(base_overrides, **{
            'MODEL.CASS_ABLATION_STAGE': 'hss_sqt_nga',
            'MODEL.CASS_CAGF_RESIDUAL_WEIGHT': 0.0,
        })
    )
    cfg_cagf_zero = _make_cfg(
        'configs/RGBNT201/default.yml',
        **dict(base_overrides, **{
            'MODEL.CASS_ABLATION_STAGE': 'hss_sqt_nga_cagf',
            'MODEL.CASS_CAGF_RESIDUAL_WEIGHT': 0.0,
        })
    )
    cfg_cagf_on = _make_cfg(
        'configs/RGBNT201/default.yml',
        **dict(base_overrides, **{
            'MODEL.CASS_ABLATION_STAGE': 'hss_sqt_nga_cagf',
            'MODEL.CASS_CAGF_RESIDUAL_WEIGHT': 0.2,
        })
    )
    torch.manual_seed(43)
    features = {
        name: torch.randn(BATCH, 5, 24, device=DEVICE)
        for name in ('RGB', 'NIR', 'TIR')
    }
    keys = ['a', 'b']
    memory = torch.randn(BATCH, 3, 24, device=DEVICE)

    torch.manual_seed(47)
    a3 = CASSModule(dim=24, cfg=cfg_a3, feat_h=2, feat_w=2).to(DEVICE)
    torch.manual_seed(47)
    cagf_zero = CASSModule(dim=24, cfg=cfg_cagf_zero, feat_h=2, feat_w=2).to(DEVICE)
    torch.manual_seed(47)
    cagf_on = CASSModule(dim=24, cfg=cfg_cagf_on, feat_h=2, feat_w=2).to(DEVICE)
    for module in (a3, cagf_zero, cagf_on):
        module.set_memory(memory, keys, ['RGB', 'NIR', 'TIR'], labels=[0, 1], epoch=1)
        module.eval()

    with torch.no_grad():
        desc_a3 = a3(features, img_path=keys, epoch=1)[3]
        desc_zero = cagf_zero(features, img_path=keys, epoch=1)[3]
        desc_on = cagf_on(features, img_path=keys, epoch=1)[3]

    assert torch.allclose(desc_a3, desc_zero, atol=1e-6, rtol=1e-5)
    delta = (desc_on - desc_a3).abs().max().item()
    assert delta > 1e-7, 'CA-GF residual should affect descriptor when enabled'
    _assert_finite(desc_on, 'CA-GF residual descriptor')
    print('     OK delta={:.6f}'.format(delta))


def test_cagf_agreement_rejects_conflicting_context():
    print('[23] CA-GF agreement rejects conflicting context')
    from modeling.fusion_part.CASS import ContextAwareGatedFusion

    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_CAGF_MODE': 'agreement',
            'MODEL.CASS_CAGF_RESIDUAL_WEIGHT': 1.0,
            'MODEL.CASS_CAGF_MIN_AGREE': 0.25,
            'MODEL.CASS_CAGF_AGREE_TAU': 0.05,
            'MODEL.CASS_CAGF_MAX_GATE': 1.0,
            'MODEL.CASS_CAGF_MAX_RESIDUAL_NORM': 1.0,
            'MODEL.CASS_CAGF_WARMUP_EPOCHS': 0,
            'MODEL.CASS_CAGF_RAMP_EPOCHS': 0,
        }
    )
    fusion = ContextAwareGatedFusion(dim=4, num_heads=1, cfg=cfg).to(DEVICE)
    selected = {
        name: torch.zeros(BATCH, 2, 4, device=DEVICE)
        for name in ('RGB', 'NIR')
    }
    gate_bias = torch.zeros(BATCH, 4, device=DEVICE)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0]], device=DEVICE)
    aligned = torch.tensor([[0.8, 0.6, 0.0, 0.0],
                            [0.8, 0.6, 0.0, 0.0]], device=DEVICE)
    opposite = -target

    good = fusion(selected, gate_bias, ['RGB', 'NIR'],
                  base_fused={'RGB': target, 'NIR': aligned}, epoch=1)['RGB']
    bad = fusion(selected, gate_bias, ['RGB', 'NIR'],
                 base_fused={'RGB': target, 'NIR': opposite}, epoch=1)['RGB']
    good_delta = (good - target).norm(dim=-1).mean().item()
    bad_delta = (bad - target).norm(dim=-1).mean().item()
    assert good_delta > 1e-3, 'aligned source should produce a visible CA-GF update'
    assert bad_delta < 0.1 * good_delta, 'conflicting source should be strongly gated'
    print('     OK aligned_delta={:.6f}, conflicting_delta={:.6f}'.format(
        good_delta, bad_delta))


def test_hypergraph_broadcast_matches_diag_math():
    print('[15] Hypergraph broadcast matches diagonal math')
    from modeling.fusion_part.CASS import HypergraphConv2d

    torch.manual_seed(13)
    hypergraph = HypergraphConv2d(
        dim=8,
        feat_h=4,
        feat_w=2,
        edges=5,
        filters=6,
        theta=0.25,
    ).to(DEVICE)
    x = torch.randn(2, 8, 4, 2, device=DEVICE)

    b, c, h, w = x.shape
    vertices = h * w
    x_float = x.float()
    phi = hypergraph.phi_conv(x).float().permute(0, 2, 3, 1).contiguous()
    phi = phi.view(b, vertices, hypergraph.filters)
    metric = F.adaptive_avg_pool2d(x, output_size=(1, 1))
    metric = hypergraph.metric_conv(metric).float().view(b, hypergraph.filters)
    metric_diag = torch.diag_embed(metric)
    assign = hypergraph.assign_conv(x).float().permute(0, 2, 3, 1).contiguous()
    assign = assign.view(b, vertices, hypergraph.edges)

    incidence = torch.matmul(phi, torch.matmul(metric_diag, torch.matmul(
        phi.transpose(1, 2), assign))).abs()
    threshold = hypergraph.theta * incidence.mean(dim=(1, 2), keepdim=True)
    incidence = torch.where(incidence < threshold, torch.zeros_like(incidence), incidence)
    node_degree = incidence.sum(dim=2)
    incidence_norm = node_degree.clamp_min(1e-6).pow(-0.5).unsqueeze(-1) * incidence
    incidence_norm = torch.nan_to_num(incidence_norm, nan=0.0, posinf=0.0, neginf=0.0)
    edge_degree = torch.diag_embed(incidence.sum(dim=1).clamp_min(1e-6).pow(-1.0))
    features = x_float.permute(0, 2, 3, 1).contiguous().view(b, vertices, hypergraph.dim)
    propagated = torch.matmul(incidence_norm, torch.matmul(edge_degree, torch.matmul(
        incidence_norm.transpose(1, 2), features)))
    propagated = torch.nan_to_num(propagated, nan=0.0, posinf=0.0, neginf=0.0)
    expected = torch.matmul(features - propagated, hypergraph.weight.float())
    if hypergraph.bias is not None:
        expected = expected + hypergraph.bias.float()
    expected = expected.permute(0, 2, 1).contiguous().view(b, c, h, w)

    actual = hypergraph(x)
    assert torch.allclose(actual, expected.to(actual.dtype), atol=1e-5, rtol=1e-5)
    print('     OK')


def test_amp_dtype_and_hss_autocast_path():
    print('[18] AMP dtype selection and HSS autocast-safe path')
    from engine.processor import _resolve_amp_settings

    cfg = _make_cfg('configs/RGBNT201/default.yml')
    cfg.SOLVER.AMP = True
    cfg.SOLVER.AMP_DTYPE = 'bf16'
    enabled, dtype, name, scaler_enabled = _resolve_amp_settings(cfg, 'cpu')
    assert enabled and dtype == torch.bfloat16 and name == 'bf16'
    assert not scaler_enabled

    cfg.SOLVER.AMP_DTYPE = 'fp16'
    enabled, dtype, name, scaler_enabled = _resolve_amp_settings(cfg, 'cpu')
    assert enabled and dtype == torch.float16 and name == 'fp16'
    assert scaler_enabled

    cfg.SOLVER.AMP_DTYPE = 'fp32'
    enabled, dtype, name, scaler_enabled = _resolve_amp_settings(cfg, 'cpu')
    assert not enabled and dtype is None and name == 'fp32'
    assert not scaler_enabled

    if DEVICE.type != 'cuda':
        print('     OK (CUDA autocast branch skipped on {})'.format(DEVICE))
        return
    if not getattr(torch.cuda, 'is_bf16_supported', lambda: False)():
        print('     OK (bf16 CUDA autocast branch skipped: unsupported device)')
        return

    cfg = _make_cfg('configs/RGBNT201/default.yml')
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    hss = model.CASS.hss
    dim = model.BACKBONE.token_dim
    patches = model.feat_h * model.feat_w
    feat = torch.randn(BATCH, patches + 1, dim, device=DEVICE, dtype=torch.bfloat16)
    with _cuda_autocast(enabled=True, dtype=torch.bfloat16):
        out, score = hss(feat, epoch=1)
    assert out.dtype == torch.bfloat16
    assert score.dtype == torch.float32
    _assert_finite(out, 'hss bf16 autocast output')
    _assert_finite(score, 'hss bf16 autocast score')
    print('     OK')


def test_resume_weight_loader():
    print('[16] staged resume weight loader')
    import tempfile
    from utils.checkpoint import load_resume_weights

    cfg = _make_cfg('configs/RGBNT201/default.yml')
    torch.manual_seed(7)
    source = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    source_state = copy.deepcopy(source.state_dict())

    target = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    first_key = next(iter(source_state))
    with torch.no_grad():
        target.state_dict()[first_key].zero_()
    assert not torch.allclose(target.state_dict()[first_key], source_state[first_key])

    with tempfile.NamedTemporaryFile(suffix='.pth') as fh:
        torch.save(source_state, fh.name)
        cfg.MODEL.RESUME_PATH = fh.name
        loaded = load_resume_weights(cfg, target)

    assert loaded
    assert torch.allclose(target.state_dict()[first_key], source_state[first_key])

    target_full = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).to(DEVICE)
    with torch.no_grad():
        target_full.state_dict()[first_key].zero_()
    with tempfile.NamedTemporaryFile(suffix='.pth') as fh:
        torch.save({'epoch': 3, 'model': source_state}, fh.name)
        cfg.MODEL.RESUME_PATH = fh.name
        loaded = load_resume_weights(cfg, target_full)

    assert loaded
    assert torch.allclose(target_full.state_dict()[first_key], source_state[first_key])
    print('     OK')


def test_training_checkpoint_roundtrip():
    print('[17] full training checkpoint round-trip')
    import tempfile
    from utils.checkpoint import load_training_checkpoint, save_training_checkpoint

    cfg = _make_cfg('configs/RGBNT201/default.yml')
    model = torch.nn.Linear(3, 2).to(DEVICE)
    center = torch.nn.Linear(2, 2).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer_center = torch.optim.SGD(center.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    scaler = _cuda_grad_scaler(enabled=False)

    loss = model(torch.randn(4, 3, device=DEVICE)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    saved_weight = model.weight.detach().clone()

    best_index = {'mAP': 0.72, 'Rank-1': 0.68, 'Rank-5': 0.70, 'Rank-10': 0.73}
    with tempfile.NamedTemporaryFile(suffix='.pth') as fh:
        save_training_checkpoint(
            fh.name,
            cfg,
            model,
            center,
            optimizer,
            optimizer_center,
            scheduler,
            scaler,
            epoch=3,
            best_index=best_index,
        )
        with torch.no_grad():
            model.weight.add_(10.0)
        optimizer.param_groups[0]['lr'] = 0.001

        info = load_training_checkpoint(
            fh.name,
            model,
            center,
            optimizer,
            optimizer_center,
            scheduler,
            scaler,
        )

    assert info['start_epoch'] == 4
    assert abs(info['best_index']['mAP'] - best_index['mAP']) < 1e-12
    assert torch.allclose(model.weight, saved_weight)
    assert abs(optimizer.param_groups[0]['lr'] - 0.01) < 1e-12
    print('     OK')


def main():
    print('Using test device: {}'.format(DEVICE))
    test_defaults_load()
    test_yml_configs_merge()
    for y in YMLS:
        test_three_modal_pipeline(y)
    test_two_modal_pipeline()
    test_save_load_roundtrip()
    test_timm_pretrain_source_loader()
    test_local_pretrain_source_loader()
    test_ablation_switches()
    test_nga_memory_stabilization()
    test_nga_warmup_skips_loader_iteration()
    test_hss_zero_gate_matches_graph0()
    test_hss_graph_warmup()
    test_hss_gate_floor_unblocks_graph_gradients()
    test_hss_structure_score_mix_changes_self_score()
    test_sqt_fallback_alpha_and_weighted_summary()
    test_sqt_additive_summary_keeps_tokens()
    test_nga_residual_keeps_additive_path_active()
    test_cagf_residual_preserves_repaired_a3_identity()
    test_cagf_agreement_rejects_conflicting_context()
    test_hypergraph_broadcast_matches_diag_math()
    test_resume_weight_loader()
    test_training_checkpoint_roundtrip()
    test_amp_dtype_and_hss_autocast_path()
    print('\n=== ALL PIPELINE TESTS PASSED ===')


if __name__ == '__main__':
    main()

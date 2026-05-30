"""
End-to-end pipeline smoke test for HTL-ReID.

Runs on CPU. Verifies that a fresh checkout with the published configs can
build the model and complete forward + backward without CUDA, real data, or
pretrained weights. This is the test others should run after `pip install
-r requirements.txt` to confirm their setup.

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
 10. Ablation switches (AGF=0, OCFR=1) each produce a usable model

Run:
    python3 test_pipeline.py
"""
import copy
import io
import sys
import torch

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


def _dummy_batch(cfg, modalities=('RGB', 'NI', 'TI')):
    H, W = cfg.INPUT.SIZE_TRAIN
    return {m: torch.randn(BATCH, 3, H, W) for m in modalities}


def _assert_finite(t, name):
    assert torch.isfinite(t).all(), '{} has NaN/Inf'.format(name)


def _loss_assembly_like_processor(output):
    """Mirror engine/processor.py:80-92 pairing rule (odd len => last is aux)."""
    loss = torch.zeros(())
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
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
    model.train()

    x = _dummy_batch(cfg)
    label = torch.randint(0, NUM_CLASSES, (BATCH,))

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
    unused_ok = ('AL_HEAD', 'AL_BN', 'BACKBONE_HEAD', 'BACKBONE_BN', 'BACKBONE.base.fc')
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
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
    model.train()

    x = _dummy_batch(cfg, modalities=('RGB', 'NI'))
    label = torch.randint(0, NUM_CLASSES, (BATCH,))
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
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
    model.eval()
    x = _dummy_batch(cfg)
    with torch.no_grad():
        out_before = model(x, cam_label=None, epoch=0)

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, map_location='cpu', weights_only=True)

    torch.manual_seed(42)
    model2 = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
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


def test_ablation_switches():
    print('[6] ablation switches')
    base_yml = 'configs/RGBNT201/default.yml'
    matrix = [
        {'MODEL.AGF': 0, 'MODEL.OCFR': 0},
        {'MODEL.AGF': 1, 'MODEL.OCFR': 1},
    ]
    for m in matrix:
        cfg = _make_cfg(base_yml, **m)
        model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
        model.train()
        x = _dummy_batch(cfg)
        label = torch.randint(0, NUM_CLASSES, (BATCH,))
        out = model(x, cam_label=None, label=label, epoch=0)
        loss = _loss_assembly_like_processor(out)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print('     OK  AGF={} OCFR={}  params={:.2f}M'.format(
            m['MODEL.AGF'], m['MODEL.OCFR'], n_params))


def test_nga_memory_stabilization():
    print('[7] NGA memory warmup+EMA+prototype')
    cfg = _make_cfg(
        'configs/RGBNT201/default.yml',
        **{
            'MODEL.CASS_NGA_MEMORY': 1,
            'MODEL.CASS_NGA_WARMUP_EPOCHS': 2,
            'MODEL.CASS_NGA_EMA_MOMENTUM': 0.5,
            'MODEL.CASS_NGA_USE_PROTOTYPE': 1,
        }
    )
    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=0).cpu()
    dim = model.BACKBONE.token_dim
    features = torch.randn(4, 3, dim)
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


def main():
    test_defaults_load()
    test_yml_configs_merge()
    for y in YMLS:
        test_three_modal_pipeline(y)
    test_two_modal_pipeline()
    test_save_load_roundtrip()
    test_ablation_switches()
    test_nga_memory_stabilization()
    print('\n=== ALL PIPELINE TESTS PASSED ===')


if __name__ == '__main__':
    main()

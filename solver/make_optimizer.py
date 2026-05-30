import torch


def _is_new_module(name):
    prefixes = (
        'CASS', 'HS_FACSS', 'AGF', 'QUALITY_HEAD', 'MODALITY_ADAPTERS',
        'PART_', 'FUSE_', 'BACKBONE_HEAD', 'BACKBONE_BN', 'AL_'
    )
    return name.startswith(prefixes)


def make_optimizer(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if key.startswith("BACKBONE."):
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_LR_FACTOR
        elif _is_new_module(key):
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.NEW_MODULE_LR_FACTOR
        if "bias" in key:
            lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if "norm" in key.lower() or "bn" in key.lower():
            weight_decay = 0.0
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center

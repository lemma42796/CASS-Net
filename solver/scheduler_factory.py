""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler


def create_scheduler(cfg, optimizer, num_batches=None):
    unit = str(getattr(cfg.SOLVER, 'SCHEDULER_UNIT', 'epoch')).lower()
    if unit not in ('iteration', 'epoch'):
        raise ValueError(
            "SOLVER.SCHEDULER_UNIT must be 'iteration' or 'epoch', got {}".format(unit))
    t_in_epochs = unit == 'epoch'
    if t_in_epochs:
        t_initial = cfg.SOLVER.MAX_EPOCHS
    else:
        if num_batches is None:
            raise ValueError(
                "num_batches is required when SOLVER.SCHEDULER_UNIT='iteration'")
        t_initial = cfg.SOLVER.MAX_EPOCHS * num_batches
    lr_min = 0.001 * cfg.SOLVER.BASE_LR
    warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    warmup_t = cfg.SOLVER.WARMUP_ITERS
    noise_range = None

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=t_initial,
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=t_in_epochs,
        noise_range_t=noise_range,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )

    return lr_scheduler

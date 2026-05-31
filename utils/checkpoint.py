import os
import random

import numpy as np
import torch


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _torch_load(path, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _scalar(value):
    if hasattr(value, 'item'):
        return value.item()
    return value


def _extract_model_state_dict(payload, source):
    if isinstance(payload, dict) and isinstance(payload.get('model'), dict):
        return payload['model'], 'training checkpoint'
    if not isinstance(payload, dict):
        raise TypeError('Checkpoint is not a state dict: {}'.format(source))
    return payload, 'state dict'


def _load_model_weights(model, state_dict, logger=None, source='checkpoint'):
    target = _unwrap_model(model)
    model_dict = target.state_dict()
    loaded = []
    skipped = []
    for key, value in state_dict.items():
        clean_key = key.replace('module.', '')
        if clean_key not in model_dict:
            skipped.append(clean_key)
            continue
        try:
            model_dict[clean_key].copy_(value)
            loaded.append(clean_key)
        except Exception as exc:
            skipped.append('{} ({})'.format(clean_key, exc))
    missing = [key for key in model_dict if key not in loaded]
    if not loaded:
        raise RuntimeError('No model parameters were loaded from {}'.format(source))
    if logger is not None:
        logger.info('Loaded {}/{} model keys from {}'.format(
            len(loaded), len(model_dict), source))
        if skipped:
            logger.info('Skipped keys while loading {}: {}'.format(source, skipped))
        if missing:
            logger.info('Missing model keys after loading {}: {}'.format(source, missing))
    return loaded, skipped, missing


def _rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'].detach().cpu())
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all([s.detach().cpu() for s in state['cuda']])


def load_resume_weights(cfg, model, logger=None):
    resume_path = str(getattr(cfg.MODEL, 'RESUME_PATH', '')).strip()
    if not resume_path:
        return False
    if not os.path.isfile(resume_path):
        raise FileNotFoundError('MODEL.RESUME_PATH does not exist: {}'.format(resume_path))
    checkpoint = _torch_load(resume_path, map_location='cpu')
    state_dict, source_type = _extract_model_state_dict(checkpoint, resume_path)
    if logger is not None:
        logger.info('Loading staged fine-tune weights from {} ({})'.format(
            resume_path, source_type))
    _load_model_weights(model, state_dict, logger=logger, source='MODEL.RESUME_PATH {}'.format(resume_path))
    return True


def save_training_checkpoint(path,
                             cfg,
                             model,
                             center_criterion,
                             optimizer,
                             optimizer_center,
                             scheduler,
                             scaler,
                             epoch,
                             best_index):
    checkpoint = {
        'epoch': int(epoch),
        'model': _unwrap_model(model).state_dict(),
        'center_criterion': center_criterion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_center': optimizer_center.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_index': {k: _scalar(v) for k, v in best_index.items()},
        'rng_state': _rng_state(),
        'cfg': str(cfg),
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = path + '.tmp'
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_training_checkpoint(path,
                             model,
                             center_criterion,
                             optimizer,
                             optimizer_center,
                             scheduler,
                             scaler,
                             logger=None,
                             map_location='cpu'):
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError('SOLVER.RESUME_CHECKPOINT does not exist: {}'.format(path))
    if logger is not None:
        logger.info('Loading training checkpoint from {}'.format(path))
    checkpoint = _torch_load(path, map_location=map_location)
    if 'model' not in checkpoint or 'epoch' not in checkpoint:
        raise KeyError('Training checkpoint must contain at least "model" and "epoch": {}'.format(path))

    _unwrap_model(model).load_state_dict(checkpoint['model'])
    if 'center_criterion' in checkpoint:
        center_criterion.load_state_dict(checkpoint['center_criterion'])
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'optimizer_center' in checkpoint:
        optimizer_center.load_state_dict(checkpoint['optimizer_center'])
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    _restore_rng_state(checkpoint.get('rng_state'))

    epoch = int(checkpoint['epoch'])
    best_index = checkpoint.get('best_index') or {}
    if logger is not None:
        logger.info('Resumed from epoch {}; next epoch is {}'.format(epoch, epoch + 1))
    return {
        'epoch': epoch,
        'start_epoch': epoch + 1,
        'best_index': best_index,
    }

import os


def load_resume_weights(cfg, model, logger=None):
    resume_path = str(getattr(cfg.MODEL, 'RESUME_PATH', '')).strip()
    if not resume_path:
        return False
    if not os.path.isfile(resume_path):
        raise FileNotFoundError('MODEL.RESUME_PATH does not exist: {}'.format(resume_path))
    if logger is not None:
        logger.info('Loading staged fine-tune weights from {}'.format(resume_path))
    model.load_param(resume_path)
    return True

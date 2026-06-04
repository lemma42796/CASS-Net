import logging
import os
import time
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from utils.checkpoint import load_training_checkpoint, save_training_checkpoint


def _cfg_enabled(value):
    if isinstance(value, str):
        return value.lower() in ('yes', 'true', '1', 'on')
    return bool(value)


def _is_two_modal(cfg):
    return int(getattr(cfg.DATASETS, 'MODALITIES', 3)) == 2


def _is_cross_modal_eval(cfg):
    return _cfg_enabled(getattr(cfg.DATASETS, 'CROSS_MODAL_EVAL', False))


def _prepare_img_batch(img, cfg, device):
    batch = {
        'RGB': img['RGB'].to(device),
        'NI': img['NI'].to(device),
    }
    if not _is_two_modal(cfg):
        batch['TI'] = img['TI'].to(device)
    return batch


def _model_forward_train(model, cfg, img, target, target_cam, target_view,
                         imgpath, writer, epoch):
    cross_type = 'two_modal' if _is_two_modal(cfg) else None
    return model(
        img, label=target, cam_label=target_cam, view_label=target_view,
        img_path=imgpath, writer=writer, epoch=epoch, cross_type=cross_type)


def _model_forward_eval(model, cfg, img, camids, target_view, img_paths, epoch=None):
    if _is_two_modal(cfg) and _is_cross_modal_eval(cfg):
        return model(
            img, cam_label=camids, view_label=target_view, img_path=img_paths,
            mode=1, epoch=epoch, cross_type='cross_modal_eval',
            modality_label=target_view)
    if _is_two_modal(cfg):
        return model(
            img, cam_label=camids, view_label=target_view, img_path=img_paths,
            mode=1, epoch=epoch, cross_type='two_modal')
    return model(
        img, cam_label=camids, view_label=target_view,
        mode=1, img_path=img_paths, epoch=epoch)


def _test_refresh_mode(value):
    mode = str(value).strip().lower()
    aliases = {
        'true': 'yes',
        '1': 'yes',
        'on': 'yes',
        'false': 'no',
        '0': 'no',
        'off': 'no',
    }
    mode = aliases.get(mode, mode)
    if mode not in ('auto', 'yes', 'no'):
        raise ValueError(
            "Unsupported TEST.REFRESH_CASS_NGA_MEMORY '{}'. Use auto, yes, or no.".format(value))
    return mode


def refresh_cass_nga_memory_for_test(cfg, model, train_loader_normal, device='cuda', logger=None):
    mode = _test_refresh_mode(getattr(cfg.TEST, 'REFRESH_CASS_NGA_MEMORY', 'auto'))
    if mode == 'no':
        if logger is not None:
            logger.info('Skipped CASS NGA memory refresh before inference: disabled by TEST.REFRESH_CASS_NGA_MEMORY')
        return False

    should_refresh = (
        mode == 'yes' or (
            getattr(cfg.MODEL, 'METHOD', 'HTL').upper() == 'CASS' and
            _cfg_enabled(getattr(cfg.MODEL, 'CASS_NGA_MEMORY', 0))
        )
    )
    if not should_refresh:
        if logger is not None:
            logger.info('Skipped CASS NGA memory refresh before inference: not a CASS NGA-memory run')
        return False

    model_for_memory = model.module if hasattr(model, 'module') else model
    if not hasattr(model_for_memory, 'refresh_nga_memory'):
        if logger is not None:
            logger.warning('Skipped CASS NGA memory refresh before inference: model has no refresh_nga_memory method')
        return False

    epoch = int(getattr(cfg.TEST, 'CASS_NGA_MEMORY_EPOCH', -1))
    refresh_epoch = None if epoch < 0 else epoch
    if logger is not None:
        epoch_msg = 'fully enabled inference state' if refresh_epoch is None else 'epoch {}'.format(refresh_epoch)
        logger.info('Refreshing CASS NGA memory before inference ({})'.format(epoch_msg))
    model_for_memory.refresh_nga_memory(
        train_loader_normal, device=device, logger=logger, epoch=refresh_epoch)
    return True


def _is_main_process(cfg):
    if not cfg.MODEL.DIST_TRAIN:
        return True
    return dist.is_available() and dist.is_initialized() and dist.get_rank() == 0


def _amp_dtype_from_cfg(value):
    name = str(value).strip().lower()
    if name in ('bf16', 'bfloat16'):
        return torch.bfloat16, 'bf16'
    if name in ('fp16', 'float16', 'half'):
        return torch.float16, 'fp16'
    if name in ('fp32', 'float32', 'none', 'off', 'false', '0'):
        return None, 'fp32'
    raise ValueError(
        "Unsupported SOLVER.AMP_DTYPE '{}'. Use one of: bf16, fp16, fp32.".format(value))


def _resolve_amp_settings(cfg, device):
    amp_requested = bool(getattr(cfg.SOLVER, 'AMP', True))
    if not amp_requested:
        return False, None, 'off', False
    amp_dtype, amp_dtype_name = _amp_dtype_from_cfg(getattr(cfg.SOLVER, 'AMP_DTYPE', 'bf16'))
    amp_enabled = amp_requested and amp_dtype is not None
    if amp_enabled and amp_dtype == torch.bfloat16 and str(device).startswith('cuda'):
        bf16_supported = getattr(torch.cuda, 'is_bf16_supported', lambda: False)()
        if not bf16_supported:
            raise RuntimeError(
                'SOLVER.AMP_DTYPE=bf16 was requested, but this CUDA device does not report bf16 support. '
                'Use SOLVER.AMP_DTYPE fp16 or SOLVER.AMP False on this machine.')
    scaler_enabled = amp_enabled and amp_dtype == torch.float16
    return amp_enabled, amp_dtype, amp_dtype_name, scaler_enabled


def _cuda_autocast(enabled, dtype):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        if enabled:
            return torch.amp.autocast('cuda', enabled=True, dtype=dtype)
        return torch.amp.autocast('cuda', enabled=False)
    if enabled:
        return amp.autocast(enabled=True, dtype=dtype)
    return amp.autocast(enabled=False)


def _cuda_grad_scaler(enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler('cuda', enabled=enabled)
        except TypeError:
            pass
    return amp.GradScaler(enabled=enabled)


def _stage_scale(epoch, warmup_epochs):
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))


def _branch_loss_weights(cfg, num_pairs):
    if getattr(cfg.MODEL, 'METHOD', 'HTL').upper() != 'CASS':
        if num_pairs <= 0:
            return []
        fused_weight = float(getattr(cfg.MODEL, 'FUSE_LOSS_WEIGHT', 1.0))
        branch_weight = float(getattr(cfg.MODEL, 'BRANCH_LOSS_WEIGHT', 0.5))
        part_weight = float(getattr(cfg.MODEL, 'PART_LOSS_WEIGHT', 0.25))
        weights = [fused_weight]
        if bool(getattr(cfg.MODEL, 'PART_BRANCH', 0)) and num_pairs > 1:
            weights.extend([branch_weight] * max(num_pairs - 2, 0))
            weights.append(part_weight)
        else:
            weights.extend([branch_weight] * (num_pairs - 1))
        return weights

    fused_weight = float(getattr(cfg.MODEL, 'CASS_FUSED_LOSS_WEIGHT', 1.0))
    modal_weight = float(getattr(cfg.MODEL, 'CASS_MODAL_LOSS_WEIGHT', 1.0))
    part_weight = float(getattr(cfg.MODEL, 'CASS_PART_LOSS_WEIGHT', 1.0))
    if num_pairs <= 0:
        return []

    weights = [fused_weight]
    if bool(getattr(cfg.MODEL, 'AL', 0)):
        if num_pairs > 1:
            weights.append(modal_weight)
        weights.extend([part_weight] * max(num_pairs - len(weights), 0))
        return weights

    modal_count = min(3, max(num_pairs - 1, 0))
    weights.extend([modal_weight] * modal_count)
    weights.extend([part_weight] * max(num_pairs - len(weights), 0))
    return weights


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             train_loader_normal,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    save_latest_checkpoint = _cfg_enabled(getattr(cfg.SOLVER, 'SAVE_LATEST_CHECKPOINT', True))
    device = "cuda"
    scheduler_epochs = int(cfg.SOLVER.MAX_EPOCHS)
    train_epochs = int(getattr(cfg.SOLVER, 'TRAIN_EPOCHS', 0))
    epochs = train_epochs if train_epochs > 0 else scheduler_epochs
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("HTL-ReID.train")
    logger.info('start training')
    if train_epochs > 0 and train_epochs != scheduler_epochs:
        logger.info('Training will stop at epoch {} while scheduler MAX_EPOCHS is {}.'.format(
            train_epochs, scheduler_epochs))
    # Create SummaryWriter
    writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs'))

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)
    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator_m = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator_m = R1_mAP_eval(
            num_query,
            max_rank=50,
            feat_norm=cfg.TEST.FEAT_NORM,
            reranking=_cfg_enabled(cfg.TEST.RE_RANKING),
            rerank_k1=cfg.TEST.RERANK_K1,
            rerank_k2=cfg.TEST.RERANK_K2,
            rerank_lambda=cfg.TEST.RERANK_LAMBDA,
        )
    evaluator_m.reset()


    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    amp_enabled, amp_dtype, amp_dtype_name, scaler_enabled = _resolve_amp_settings(cfg, device)
    logger.info('AMP enabled: {} dtype: {} scaler: {}'.format(
        amp_enabled, amp_dtype_name, scaler_enabled))
    if getattr(cfg.MODEL, 'METHOD', 'HTL').upper() == 'CASS':
        logger.info('CASS branch loss weights: fused {:.3f}, modal {:.3f}, part {:.3f}'.format(
            float(cfg.MODEL.CASS_FUSED_LOSS_WEIGHT),
            float(cfg.MODEL.CASS_MODAL_LOSS_WEIGHT),
            float(cfg.MODEL.CASS_PART_LOSS_WEIGHT),
        ))
    else:
        logger.info('HTL branch loss weights: fused {:.3f}, branch {:.3f}, part {:.3f}, aux {:.3f}'.format(
            float(cfg.MODEL.FUSE_LOSS_WEIGHT),
            float(cfg.MODEL.BRANCH_LOSS_WEIGHT),
            float(cfg.MODEL.PART_LOSS_WEIGHT),
            float(cfg.MODEL.AUX_LOSS_WEIGHT),
        ))
    scaler = _cuda_grad_scaler(enabled=scaler_enabled)
    scheduler_in_epochs = getattr(scheduler, 't_in_epochs', True)
    updates_per_epoch = len(train_loader)

    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    start_epoch = 1
    resume_checkpoint = str(getattr(cfg.SOLVER, 'RESUME_CHECKPOINT', '')).strip()
    if resume_checkpoint:
        resume_info = load_training_checkpoint(
            resume_checkpoint,
            model,
            center_criterion,
            optimizer,
            optimizer_center,
            scheduler,
            scaler,
            logger=logger,
            map_location=device,
        )
        start_epoch = resume_info['start_epoch']
        best_index.update(resume_info.get('best_index') or {})
    if start_epoch > epochs:
        logger.info('Resume checkpoint is already at epoch {}; MAX_EPOCHS is {}. Nothing to train.'.format(
            start_epoch - 1, epochs))
        writer.close()
        return None

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        evaluator_m.reset()
        acc_meter.reset()
        if scheduler_in_epochs:
            scheduler.step(epoch)
        model_for_memory = model.module if hasattr(model, 'module') else model
        hss = getattr(getattr(model_for_memory, 'CASS', None), 'hss', None)
        if hss is not None and hasattr(hss, 'current_graph_weight'):
            logger.info('CASS HSS graph weight: {:.4f} / {:.4f}'.format(
                hss.current_graph_weight(epoch), hss.graph_weight))
            if hasattr(hss, 'current_residual_gate'):
                gate_mean, gate_abs = hss.current_residual_gate(epoch)
                if hasattr(hss, 'current_gate_floor'):
                    logger.info('CASS HSS residual gate: mean {:.4f}, abs-mean {:.4f}, floor {:.4f}'.format(
                        gate_mean, gate_abs, hss.current_gate_floor(epoch)))
                else:
                    logger.info('CASS HSS residual gate: mean {:.4f}, abs-mean {:.4f}'.format(
                        gate_mean, gate_abs))
            if hasattr(hss, 'score_mix') and hss.score_mix > 0.0:
                logger.info('CASS HSS score mix: {:.4f}, source {}, detach {}'.format(
                    hss.score_mix, hss.score_source, hss.score_detach))
        refresh_period = max(1, int(getattr(cfg.MODEL, 'CASS_NGA_REFRESH_PERIOD', 1)))
        if getattr(cfg.MODEL, 'METHOD', 'HTL').upper() == 'CASS' and (epoch - 1) % refresh_period == 0:
            model_for_memory.refresh_nga_memory(
                train_loader_normal, device=device, logger=logger, epoch=epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, imgpath) in enumerate(train_loader):
            if not scheduler_in_epochs:
                num_updates = (epoch - 1) * updates_per_epoch + n_iter
                scheduler.step_update(num_updates)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = _prepare_img_batch(img, cfg, device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with _cuda_autocast(amp_enabled, amp_dtype):
                output = _model_forward_train(
                    model, cfg, img, target, target_cam, target_view,
                    imgpath, writer, epoch)
                loss = 0
                index = len(output) - 1 if len(output) % 2 == 1 else len(output)
                branch_weights = _branch_loss_weights(cfg, index // 2)
                for pair_idx, i in enumerate(range(0, index, 2)):
                    loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                    loss = loss + branch_weights[pair_idx] * loss_tmp
                if len(output) % 2 == 1:
                    if getattr(cfg.MODEL, 'METHOD', 'HTL').upper() == 'CASS':
                        loss = loss + output[-1]
                    else:
                        aux_weight = float(cfg.MODEL.AUX_LOSS_WEIGHT) * _stage_scale(
                            epoch, cfg.MODEL.AUX_WARMUP_EPOCHS)
                        loss = loss + aux_weight * output[-1]
            writer.add_scalar('Loss', loss.item(), epoch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if output[0].shape[0] != target.shape[0]:
                target = torch.cat((target, target), dim=0)
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                # print(scheduler._get_lr(epoch))
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, optimizer.param_groups[0]['lr']))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print('!!!Mutil-Modal Testing!!!')
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = _prepare_img_batch(img, cfg, device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = _model_forward_eval(
                                model, cfg, img, camids, target_view, _, epoch=epoch)
                            if cfg.DATASETS.NAMES == "MSVR310":
                                evaluator_m.update((feat, vid, camid, target_view, _))
                            else:
                                evaluator_m.update((feat, vid, camid))


                    # 计算多模态性能
                    cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.2%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
                    writer.add_scalar('MM/mAP', mAP.item(), epoch)
                    writer.add_scalar('MM/Rank-1', cmc[0].item(), epoch)

                    if mAP >= best_index['mAP']:
                        best_index['mAP'] = mAP
                        best_index['Rank-1'] = cmc[0]
                        best_index['Rank-5'] = cmc[4]
                        best_index['Rank-10'] = cmc[9]
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                    logger.info("Best Multi-Modal mAP: {:.2%}".format(best_index['mAP']))
                    logger.info("Best Multi-Modal Rank-1: {:.2%}".format(best_index['Rank-1']))
                    logger.info("Best Multi-Modal Rank-5: {:.2%}".format(best_index['Rank-5']))
                    logger.info("Best Multi-Modal Rank-10: {:.2%}".format(best_index['Rank-10']))
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    torch.cuda.empty_cache()

            else:
                model.eval()
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('!!!Mutil-Modal Testing!!!')
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = _prepare_img_batch(img, cfg, device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = _model_forward_eval(
                            model, cfg, img, camids, target_view, _, epoch=epoch)
                        if cfg.DATASETS.NAMES == "MSVR310":
                            evaluator_m.update((feat, vid, camid, target_view, _))
                        else:
                            evaluator_m.update((feat, vid, camid))

                # 计算多模态性能
                cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.2%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
                writer.add_scalar('MM/mAP', mAP.item(), epoch)
                writer.add_scalar('MM/Rank-1', cmc[0].item(), epoch)


                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                logger.info("Best Multi-Modal mAP: {:.2%}".format(best_index['mAP']))
                logger.info("Best Multi-Modal Rank-1: {:.2%}".format(best_index['Rank-1']))
                logger.info("Best Multi-Modal Rank-5: {:.2%}".format(best_index['Rank-5']))
                logger.info("Best Multi-Modal Rank-10: {:.2%}".format(best_index['Rank-10']))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                torch.cuda.empty_cache()

        if save_latest_checkpoint and _is_main_process(cfg):
            latest_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_latest.pth')
            save_training_checkpoint(
                latest_path,
                cfg,
                model,
                center_criterion,
                optimizer,
                optimizer_center,
                scheduler,
                scaler,
                epoch,
                best_index,
            )
            logger.info('Saved latest training checkpoint to {}'.format(latest_path))


    writer.close()
    return None


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("HTL-ReID.test")
    logger.info("Enter inferencing")
    if not _is_main_process(cfg):
        return None

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator_m = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator_m = R1_mAP_eval(
            num_query,
            max_rank=50,
            feat_norm=cfg.TEST.FEAT_NORM,
            reranking=_cfg_enabled(cfg.TEST.RE_RANKING),
            rerank_k1=cfg.TEST.RERANK_K1,
            rerank_k2=cfg.TEST.RERANK_K2,
            rerank_lambda=cfg.TEST.RERANK_LAMBDA,
        )
    evaluator_m.reset()

    model.eval()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('!!!Mutil-Modal Testing!!!')
    with torch.inference_mode():
        for n_iter, (img, vid, camid, camids, target_view, img_paths) in enumerate(val_loader):
            img = _prepare_img_batch(img, cfg, device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = _model_forward_eval(
                model, cfg, img, camids, target_view, img_paths)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator_m.update((feat, vid, camid, target_view, img_paths))
            else:
                evaluator_m.update((feat, vid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator_m.compute(cfg)
    logger.info("Inference Results")
    logger.info("mAP: {:.2%}".format(float(mAP)))
    results = {'mAP': float(mAP)}
    for r in [1, 5, 10]:
        if len(cmc) >= r:
            value = float(cmc[r - 1])
            logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, value))
            results['Rank-{}'.format(r)] = value
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    torch.cuda.empty_cache()
    return results

"""
Run a re-ranking grid after extracting validation features once.

This is the fast counterpart to tools/run_rerank_grid.py: it keeps the model
and validation features in memory, then evaluates many TEST.RERANK_* settings.
"""
import argparse
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import cfg
from data import make_dataloader
from engine.processor import refresh_cass_nga_memory_for_test
from modeling import make_model
from utils.logger import setup_logger
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking


def parse_float_list(raw):
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def enabled(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def extract_features(cfg, model, val_loader, num_query, device="cuda"):
    feats = []
    pids = []
    camids = []
    model.eval()
    with torch.inference_mode():
        for img, vid, camid, camids_batch, target_view, img_paths in val_loader:
            img = {
                "RGB": img["RGB"].to(device),
                "NI": img["NI"].to(device),
                "TI": img["TI"].to(device),
            }
            camids_batch = camids_batch.to(device)
            target_view = target_view.to(device)
            feat = model(
                img,
                cam_label=camids_batch,
                view_label=target_view,
                mode=1,
                img_path=img_paths,
            )
            feats.append(feat.cpu())
            pids.extend(np.asarray(vid))
            camids.extend(np.asarray(camid))

    feats = torch.cat(feats, dim=0)
    if enabled(cfg.TEST.FEAT_NORM):
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    qf = feats[:num_query]
    gf = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])
    return qf, gf, q_pids, g_pids, q_camids, g_camids


def write_summary(rows, output_dir):
    header = ["k1", "k2", "lambda", "mAP", "Rank-1", "Rank-5", "Rank-10"]
    os.makedirs(output_dir, exist_ok=True)
    for name, rows_to_write in (
        ("summary.tsv", rows),
        ("summary_sorted.tsv", sorted(rows, key=lambda r: (r["mAP"], r["Rank-1"]), reverse=True)),
    ):
        with open(os.path.join(output_dir, name), "w") as handle:
            handle.write("\t".join(header) + "\n")
            for row in rows_to_write:
                handle.write(
                    "{k1}\t{k2}\t{lam:.3g}\t{mAP:.2f}\t{r1:.2f}\t{r5:.2f}\t{r10:.2f}\n".format(
                        k1=row["k1"],
                        k2=row["k2"],
                        lam=row["lambda"],
                        mAP=100.0 * row["mAP"],
                        r1=100.0 * row["Rank-1"],
                        r5=100.0 * row["Rank-5"],
                        r10=100.0 * row["Rank-10"],
                    )
                )


def main():
    parser = argparse.ArgumentParser(description="Run cached TEST.RERANK_* grid")
    parser.add_argument("--config_file", action="append", default=[], required=True)
    parser.add_argument("--checkpoint", "--weight", dest="checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--k1", default="40,45,50,55")
    parser.add_argument("--k2", default="15,20")
    parser.add_argument("--lambda_values", "--lambda-values", dest="lambda_values",
                        default="0.08,0.10,0.12,0.15")
    parser.add_argument("--include_plain", action="store_true")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[])
    args, trailing_opts = parser.parse_known_args()
    args.opts.extend(trailing_opts)

    for config_path in args.config_file:
        cfg.merge_from_file(config_path)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.TEST.WEIGHT = args.checkpoint
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("HTL-ReID", cfg.OUTPUT_DIR, if_train=False)
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, _ = \
        make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
    model.cuda()
    model.load_param(cfg.TEST.WEIGHT)
    refresh_cass_nga_memory_for_test(
        cfg, model, train_loader_normal, device="cuda", logger=logger)
    qf, gf, q_pids, g_pids, q_camids, g_camids = extract_features(
        cfg, model, val_loader, num_query, device="cuda")

    rows = []
    if args.include_plain:
        cmc, mAP = eval_func(
            euclidean_distance(qf, gf), q_pids, g_pids, q_camids, g_camids)
        rows.append({
            "k1": 0,
            "k2": 0,
            "lambda": 1.0,
            "mAP": float(mAP),
            "Rank-1": float(cmc[0]),
            "Rank-5": float(cmc[4]),
            "Rank-10": float(cmc[9]),
        })
        write_summary(rows, cfg.OUTPUT_DIR)

    for k1 in parse_int_list(args.k1):
        for k2 in parse_int_list(args.k2):
            for lam in parse_float_list(args.lambda_values):
                logger.info("Evaluating rerank k1=%s k2=%s lambda=%s", k1, k2, lam)
                distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lam)
                cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
                row = {
                    "k1": k1,
                    "k2": k2,
                    "lambda": lam,
                    "mAP": float(mAP),
                    "Rank-1": float(cmc[0]),
                    "Rank-5": float(cmc[4]),
                    "Rank-10": float(cmc[9]),
                }
                rows.append(row)
                logger.info(
                    "k1=%s k2=%s lambda=%.3g mAP=%.2f Rank-1=%.2f Rank-5=%.2f Rank-10=%.2f",
                    k1, k2, lam,
                    100.0 * row["mAP"], 100.0 * row["Rank-1"],
                    100.0 * row["Rank-5"], 100.0 * row["Rank-10"],
                )
                write_summary(rows, cfg.OUTPUT_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

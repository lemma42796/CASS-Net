import os
from config import cfg
import argparse
from data import make_dataloader
from modeling import make_model
from engine.processor import do_inference, refresh_cass_nga_memory_for_test
from utils.logger import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTL-ReID Testing")
    parser.add_argument(
        "--config_file", default=[], action='append', type=str,
        help="path to config file; pass multiple times to chain merges"
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    for config_path in args.config_file:
        cfg.merge_from_file(config_path)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("HTL-ReID", output_dir, if_train=False)
    logger.info(args)

    for config_path in args.config_file:
        logger.info("Loaded configuration file {}".format(config_path))
        with open(config_path, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
    model.cuda()
    if not cfg.TEST.WEIGHT:
        raise ValueError(
            "cfg.TEST.WEIGHT is empty. Pass the trained checkpoint path via the "
            "yml config or CLI, e.g.:\n"
            "    python test_net.py --config_file <yml> TEST.WEIGHT /path/to/ckpt.pth"
        )
    model.load_param(cfg.TEST.WEIGHT)
    refresh_cass_nga_memory_for_test(
        cfg, model, train_loader_normal, device="cuda", logger=logger)
    do_inference(cfg, model, val_loader, num_query)

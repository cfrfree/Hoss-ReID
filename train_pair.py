from utils.logger import setup_logger
from datasets import make_dataloader_pair
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train_pair
import random
import torch
import numpy as np
import os
import re
import argparse
from config import cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TransOSS Pretraining")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local-rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    train_loader_pair, num_classes, camera_num = make_dataloader_pair(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)
    start_epoch = 1
    if cfg.SOLVER.RESUME_PATH != "":
        logger.info(f"Resuming from checkpoint: {cfg.SOLVER.RESUME_PATH}")
        # 加载模型权重
        model.load_param(cfg.SOLVER.RESUME_PATH)

        # 从文件名中提取 epoch 数
        # 例如: 从 '.../transformer_20.pth' 中提取 '20'
        try:
            epoch_str = re.search(r"_(\d+)\.pth$", cfg.SOLVER.RESUME_PATH).group(1)
            start_epoch = int(epoch_str) + 1
            logger.info(f"Will start training from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Could not parse epoch from filename. Starting from epoch 1. Error: {e}")
            start_epoch = 1
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train_pair(cfg, model, train_loader_pair, optimizer, scheduler, args.local_rank, start_epoch)

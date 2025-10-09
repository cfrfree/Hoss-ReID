# train_mae.py
import os
import argparse
import torch
import random
import numpy as np
import logging
import time

from config import cfg
from datasets import make_dataloader_pair
from model.backbones.vit_transoss_mae import TransOSS_MAE  # <-- 导入新的MAE模型
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="TransOSS MAE Pre-training")
    parser.add_argument("--config_file", default="configs/pretrain_transoss_mae.yml", help="path to config file", type=str)
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

    logger = setup_logger("transreid_mae", output_dir, if_train=True)
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # --- 数据加载 ---
    # 复用原来的dataloader，但注意collate_fn可能需要调整
    train_loader_pair, _, _ = make_dataloader_pair(cfg)

    # --- 模型、优化器、调度器 ---
    model = TransOSS_MAE(cfg)
    optimizer, _ = make_optimizer(cfg, model, None)  # Center loss is not used
    scheduler = create_scheduler(cfg, optimizer)

    # --- 训练核心逻辑 ---
    do_train_mae(cfg, model, train_loader_pair, optimizer, scheduler, args.local_rank)


def do_train_mae(cfg, model, train_loader, optimizer, scheduler, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid_mae.train")
    logger.info("Start MAE pre-training")

    device = "cuda"
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        for n_iter, (img_batch, _, _) in enumerate(train_loader):
            optimizer.zero_grad()

            # 从batch中分离光学和SAR图像
            # 注意: 这里的img_batch是拼接后的，需要根据dataloader的实现来正确分离
            b_s = img_batch.shape[0] // 2
            imgs_opt = img_batch[:b_s].to(device)
            imgs_sar = img_batch[b_s:].to(device)

            with amp.autocast(enabled=True):
                loss = model(imgs_opt, imgs_sar)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), imgs_opt.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader), loss_meter.avg, scheduler._get_lr(epoch)[0]
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s]".format(epoch, time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    # 保存编码器权重用于后续微调
                    torch.save(model.module.encoder.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"mae_encoder_{epoch}.pth"))
            else:
                torch.save(model.encoder.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"mae_encoder_{epoch}.pth"))


if __name__ == "__main__":
    main()

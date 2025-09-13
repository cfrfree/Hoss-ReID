import os
import torch
import argparse
import random
import numpy as np
import logging
import sys
import xml.etree.ElementTree as ET

from config import cfg
from datasets.hjj import CustomReIDDataset, InferenceGalleryDataset, InferenceQueryDataset
from datasets.bases import ImageDataset
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from utils.logger import setup_logger

# **关键修改：导入新的评估器**
from utils.custom_metrics import CustomClassificationEvaluator


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_collate_fn(batch):
    imgs, pids, camids, _, _, img_sizes = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    img_wh_tensors = [torch.tensor(s, dtype=torch.float32) for s in img_sizes]
    img_wh = torch.stack(img_wh_tensors, dim=0)
    return torch.stack(imgs, dim=0), pids, camids, None, img_wh


def val_collate_fn(batch):
    imgs, pids, camids, _, img_paths, img_sizes = zip(*batch)
    camids_tensor = torch.tensor(camids, dtype=torch.int64)
    img_wh_tensors = [torch.tensor(s, dtype=torch.float32) for s in img_sizes]
    img_wh = torch.stack(img_wh_tensors, dim=0)
    return torch.stack(imgs, dim=0), pids, camids, camids_tensor, None, img_paths, img_wh


def parse_xml_for_pids(xml_file):
    """解析XML文件以创建文件名到目标类别的映射。"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return {obj.find("filename").text: obj.find("type").text for obj in root.findall("object")}
    except (ET.ParseError, FileNotFoundError):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom ReID Task Training with Periodic Validation")
    parser.add_argument("--config_file", default="configs/hjj.yml", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local-rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(1234)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    # --- 训练数据加载器 ---
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
        ]
    )

    custom_dataset = CustomReIDDataset(data_path=cfg.DATASETS.TRAIN_PATH)
    num_classes = custom_dataset.num_train_pids
    train_set = ImageDataset(custom_dataset.train, train_transforms)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(custom_dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=train_collate_fn,
    )

    val_loader = None
    num_query = 0
    if cfg.DATASETS.VAL_PATH and cfg.DATASETS.GT_XML_PATH:
        val_transforms = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
        gt_map = parse_xml_for_pids(cfg.DATASETS.GT_XML_PATH)
        if gt_map is None:
            logger.error(f"无法加载真值XML文件: {cfg.DATASETS.GT_XML_PATH}. 跳过验证。")
        else:
            query_path = os.path.join(cfg.DATASETS.VAL_PATH, "Sar文件夹")
            gallery_path = os.path.join(cfg.DATASETS.VAL_PATH, "光文件夹")
            try:
                query_dataset = InferenceQueryDataset(query_path, transform=val_transforms)
                gallery_dataset = InferenceGalleryDataset(gallery_path, transform=val_transforms)
                val_set_list = []
                for img_path, filename in query_dataset.img_items:
                    if filename in gt_map:
                        pid = gt_map[filename]
                        val_set_list.append((img_path, pid, 1, 1))
                num_query = len(val_set_list)
                for img_path, target_name in gallery_dataset.img_items:
                    pid = target_name
                    val_set_list.append((img_path, pid, 0, 1))
                val_set = ImageDataset(val_set_list, val_transforms)
                val_loader = DataLoader(
                    val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn
                )
            except Exception as e:
                logger.warning(f"无法创建验证数据加载器，将跳过周期性验证。错误: {e}")
                val_loader = None
                num_query = 0

    # --- 初始化模型、损失、优化器等 (不变) ---
    model = make_model(cfg, num_class=num_classes, camera_num=2)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    # **关键修改：实例化新的评估器**
    evaluator = CustomClassificationEvaluator(
        num_query=num_query, gt_xml_path=cfg.DATASETS.GT_XML_PATH, top_k=cfg.TEST.TOP_K, feat_norm=cfg.TEST.FEAT_NORM  # 从配置中读取K值
    )

    # --- 开始训练 ---
    logger.info("Starting ReID training with periodic validation...")
    do_train(
        cfg=cfg,
        model=model,
        center_criterion=center_criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        optimizer_center=optimizer_center,
        scheduler=scheduler,
        loss_fn=loss_func,
        num_query=num_query,
        local_rank=args.local_rank,
        evaluator=evaluator,  # **关键修改：将评估器实例传入**
    )
    logger.info("ReID training finished.")

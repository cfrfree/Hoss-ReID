import os
import torch
import argparse
import time
from collections import Counter
import xml.etree.ElementTree as ET
from xml.dom import minidom
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from model import make_model

# 确保导入的是为您的任务定制的数据集加载器
from datasets.hjj import InferenceGalleryDataset, InferenceQueryDataset
from utils.reranking import re_ranking


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Custom Inference Script for Test Set")
    parser.add_argument("--config_file", default="configs/hjj.yml", help="path to config file for the task")
    parser.add_argument("--test_dir", default=None, help="Path to the test data directory (e.g., '赛道4测试数据/'). Overrides the path in the config file.")
    parser.add_argument("--output_path", required=True, help="Path to save the final result.xml file")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # --- 1. 模型加载 ---
    # print("Building model for weight loading...")

    # 我们知道训练好的模型是基于3个类别的。
    # 因此，我们必须用 num_class=3 来初始化模型，以确保分类层的形状匹配，从而成功加载权重。
    # 如果您的训练集类别数发生变化，只需修改这里的数字即可。
    NUM_CLASSES_IN_CHECKPOINT = 3
    model = make_model(cfg, num_class=NUM_CLASSES_IN_CHECKPOINT, camera_num=2)

    # print(f"Loading weights from: {cfg.TEST.WEIGHT}")
    # 此时，由于模型形状与权重文件完全匹配，加载会成功
    model.load_param(cfg.TEST.WEIGHT)
    # print("Model weights loaded successfully.")

    # 加载权重后，我们就不再需要分类相关的层了。
    # 将它们替换为 Identity 层，使模型成为一个纯粹的特征提取器。
    model.classifier = nn.Identity()
    model.bottleneck = nn.Identity()  # 同样移除 bottleneck 层
    # print("Classifier and bottleneck layers removed. Model is now in feature extraction mode.")

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model.eval()

    # --- 2. 数据加载和特征提取 ---
    val_transforms = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])

    if args.test_dir:
        test_path = args.test_dir
        print(f"Using test directory provided via command line: {test_path}")
    else:
        test_path = cfg.DATASETS.TEST_PATH
        print(f"Using test directory from config file: {test_path}")

    if not os.path.isdir(test_path):
        print(f"错误: 测试集路径不存在: {test_path}")
        return

    # 提取光学参考图像 (Gallery) 的特征
    gallery_path = os.path.join(test_path, "光文件夹")
    gallery_dataset = InferenceGalleryDataset(gallery_path, transform=val_transforms)
    gallery_loader = DataLoader(gallery_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS)

    all_gallery_feats = []
    all_gallery_labels = []
    # print(f"Extracting features from optical gallery at: {gallery_path}")
    with torch.no_grad():
        for imgs, target_names, img_sizes in gallery_loader:
            imgs = imgs.to(device)
            img_wh = torch.stack(img_sizes, dim=1).float().to(device)
            # =======================================================
            feats = model(imgs, cam_label=torch.zeros(imgs.size(0), dtype=torch.long).to(device), img_wh=img_wh)
            all_gallery_feats.append(feats)
            all_gallery_labels.extend(target_names)
    all_gallery_feats = torch.cat(all_gallery_feats, dim=0)
    # print(f"Extracted {all_gallery_feats.shape[0]} features from the gallery.")

    # 提取SAR待查询图像 (Query) 的特征
    query_path = os.path.join(test_path, "Sar文件夹")
    query_dataset = InferenceQueryDataset(query_path, transform=val_transforms)
    query_loader = DataLoader(query_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS)

    all_query_feats = []
    all_query_filenames = []
    # print(f"Extracting features from SAR queries at: {query_path}")
    with torch.no_grad():
        for imgs, filenames, img_sizes in query_loader:
            imgs = imgs.to(device)
            img_wh = torch.stack(img_sizes, dim=1).float().to(device)
            # =======================================================
            query_feats = model(imgs, cam_label=torch.ones(imgs.size(0), dtype=torch.long).to(device), img_wh=img_wh)
            all_query_feats.append(query_feats)
            all_query_filenames.extend(filenames)
    all_query_feats = torch.cat(all_query_feats, dim=0)
    # print(f"Extracted {all_query_feats.shape[0]} features from the queries.")

    # --- 3. 距离计算和分类 ---
    if cfg.TEST.RE_RANKING:
        print("\nRe-ranking is enabled...")
        dist_matrix_reranked = re_ranking(all_query_feats, all_gallery_feats, k1=20, k2=6, lambda_value=0.3)
        dist_matrix = torch.from_numpy(dist_matrix_reranked).to(device)
    else:
        print("\nUsing standard Euclidean distance.")
        dist_matrix = torch.cdist(all_query_feats, all_gallery_feats)

    top_k = cfg.TEST.TOP_K
    print(f"Classification started using Top-K majority vote (K={top_k}).")
    _, top_k_indices = torch.topk(dist_matrix, k=top_k, dim=1, largest=False)

    results = []
    for i in range(top_k_indices.shape[0]):
        neighbor_labels = [all_gallery_labels[idx] for idx in top_k_indices[i]]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        results.append((all_query_filenames[i], most_common_label))

    print(f"Classification finished for {len(results)} SAR images.")

    # --- 4. 生成并保存XML文件 ---
    root = ET.Element("annotation")
    for filename, type_name in results:
        obj_elem = ET.SubElement(root, "object")
        ET.SubElement(obj_elem, "filename").text = filename
        ET.SubElement(obj_elem, "type").text = type_name
    xml_str = ET.tostring(root, "utf-8")
    reparsed = minidom.parseString(xml_str)
    pretty_xml_str = reparsed.toprettyxml(indent="  ", encoding="utf-8")
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "wb") as f:
        f.write(pretty_xml_str)

    print(f"Result XML saved to {args.output_path}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal inference time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()

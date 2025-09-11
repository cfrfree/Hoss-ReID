import os
import torch
import argparse
from config import cfg
from model import make_model
from torch.utils.data import DataLoader
from collections import Counter
import xml.etree.ElementTree as ET
from xml.dom import minidom
import torch.nn as nn
import torchvision.transforms as T

from utils.reranking import re_ranking
from datasets.hjj import OpticalGalleryDataset, SarQueryDataset


def main():
    parser = argparse.ArgumentParser(description="Custom Inference Task with KNN and Re-ranking")
    parser.add_argument("--config_file", default="configs/hoss_transoss.yml", help="path to config file")
    parser.add_argument("--input_dir", required=True, help="Path to the test data directory (e.g., '赛道4测试数据/')")
    parser.add_argument("--output_path", required=True, help="Path to save the output result.xml file")
    parser.add_argument("--top_k", type=int, default=5, help="Value of K for KNN majority voting")

    # --- 新增 Rerank 开关 ---
    parser.add_argument("--rerank", action="store_true", help="Enable k-reciprocal re-ranking to improve results")
    # Rerank 的超参数
    parser.add_argument("--k1", type=int, default=20, help="k1 parameter for re_ranking")
    parser.add_argument("--k2", type=int, default=6, help="k2 parameter for re_ranking")
    parser.add_argument("--lambda_value", type=float, default=0.3, help="lambda_value for re_ranking")

    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # --- 1. & 2. & 3. 模型加载和特征提取 (与之前版本相同) ---
    print("Building model with original class number (361) to load weights...")
    model = make_model(cfg, num_class=361, camera_num=2)
    model.load_param(cfg.TEST.WEIGHT)
    print(f"Model weights loaded from {cfg.TEST.WEIGHT}")
    model.classifier = nn.Identity()
    print("Replaced the original classifier head with an Identity layer.")
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model.eval()

    optical_path = os.path.join(args.input_dir, "光文件夹")
    val_transforms = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(), T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    gallery_dataset = OpticalGalleryDataset(optical_path, transform=val_transforms)
    gallery_loader = DataLoader(gallery_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False)

    all_gallery_feats = []
    all_gallery_labels = []
    with torch.no_grad():
        for imgs, target_names, img_sizes in gallery_loader:
            imgs = imgs.to(device)
            img_wh = torch.stack(img_sizes, dim=1).float().to(device)
            feats = model(imgs, cam_label=torch.zeros(imgs.size(0), dtype=torch.long).to(device), img_wh=img_wh)
            # 注意：进行rerank时，通常不预先做normalize
            all_gallery_feats.append(feats)
            all_gallery_labels.extend(target_names)
    all_gallery_feats = torch.cat(all_gallery_feats, dim=0)
    print(f"Extracted {all_gallery_feats.shape[0]} features from the optical gallery.")

    sar_path = os.path.join(args.input_dir, "Sar文件夹")
    query_dataset = SarQueryDataset(sar_path, transform=val_transforms)
    query_loader = DataLoader(query_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False)

    all_query_feats = []
    all_query_filenames = []
    with torch.no_grad():
        for imgs, filenames, img_sizes in query_loader:
            imgs = imgs.to(device)
            img_wh = torch.stack(img_sizes, dim=1).float().to(device)
            query_feats = model(imgs, cam_label=torch.ones(imgs.size(0), dtype=torch.long).to(device), img_wh=img_wh)
            all_query_feats.append(query_feats)
            all_query_filenames.extend(filenames)
    all_query_feats = torch.cat(all_query_feats, dim=0)

    # --- 4. 【修改点】计算距离矩阵，并根据开关决定是否使用 Rerank ---

    if args.rerank:
        print("\nRe-ranking is enabled. Applying k-reciprocal encoding...")
        print(f"  - k1={args.k1}, k2={args.k2}, lambda={args.lambda_value}")
        # re_ranking函数需要query特征和gallery特征作为输入
        # 注意：这里的gallery特征包含了所有的参考图像，而不仅仅是SAR
        # 我们需要将SAR和Optical的特征拼接起来送入rerank
        # probFea: query, galFea: gallery
        # 在我们的任务中，Query是SAR，Gallery是Optical
        dist_matrix_reranked = re_ranking(all_query_feats, all_gallery_feats, args.k1, args.k2, args.lambda_value)
        # 将numpy结果转回tensor
        dist_matrix = torch.from_numpy(dist_matrix_reranked).to(device)
        print("Re-ranking finished.")
    else:
        print("\nRe-ranking is disabled. Using standard Euclidean distance.")
        # 如果不使用rerank，我们计算欧氏距离
        # 距离越小越好
        dist_matrix = torch.cdist(all_query_feats, all_gallery_feats)

    # --- 5. 【修改点】使用最终的距离矩阵进行KNN投票 ---

    # 因为我们现在用的是距离矩阵（值越小越好），
    # 所以要用 topk 找到最小的 k 个值
    _, top_k_indices = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)

    results = []
    for i in range(top_k_indices.shape[0]):
        neighbor_labels = [all_gallery_labels[idx] for idx in top_k_indices[i]]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        results.append((all_query_filenames[i], most_common_label))

    print(f"Classification finished for {len(results)} SAR images using Top-K (K={args.top_k}).")

    # --- 6. 生成并保存XML文件 (保持不变) ---
    # ... (这部分代码不变) ...
    root = ET.Element("annotation")
    for filename, type_name in results:
        obj_elem = ET.SubElement(root, "object")
        file_elem = ET.SubElement(obj_elem, "filename")
        file_elem.text = filename
        type_elem = ET.SubElement(obj_elem, "type")
        type_elem.text = type_name

    xml_str = ET.tostring(root, "utf-8")
    reparsed = minidom.parseString(xml_str)
    pretty_xml_str = reparsed.toprettyxml(indent="  ", encoding="utf-8")

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result_file_path = args.output_path
    if os.path.isdir(args.output_path):
        result_file_path = os.path.join(args.output_path, "result.xml")

    with open(result_file_path, "wb") as f:
        f.write(pretty_xml_str)

    print(f"Result XML saved to {result_file_path}")

    # --- 7. 可视化 (可选) ---
    # ... (可视化部分可以保持不变，但矩阵的含义变成了距离) ...
    # 我们将距离转换为相似度 (e.g., using exp(-dist)) 以便可视化
    # print("Generating distance matrix visualization...")
    # sim_matrix_np = np.exp(-dist_matrix.cpu().numpy())  # 将距离转换为相似度
    # plt.figure(figsize=(20, 15))
    # sns.heatmap(sim_matrix_np, xticklabels=all_gallery_labels, yticklabels=all_query_filenames, cmap="viridis", annot=False)
    # plt.title("SAR (Query) vs Optical (Gallery) Distance Matrix", fontsize=20)
    # plt.xlabel("Optical Gallery Images (by Target Name)", fontsize=15)
    # plt.ylabel("SAR Query Images (by Filename)", fontsize=15)
    # plt.rcParams["font.sans-serif"] = ["SimHei"]
    # plt.rcParams["axes.unicode_minus"] = False
    # plt.tight_layout()
    # heatmap_path = os.path.join(os.path.dirname(result_file_path), "distance_matrix.png")
    # plt.savefig(heatmap_path, dpi=300)
    # print(f"Distance matrix heatmap saved to {heatmap_path}")


if __name__ == "__main__":
    main()

import torch
import numpy as np
import os
import xml.etree.ElementTree as ET
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter


def parse_xml(xml_file):
    """
    解析XML文件，返回一个 {filename: type} 的字典。
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        results = {obj.find("filename").text: obj.find("type").text for obj in root.findall("object")}
        return results
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"警告: 无法加载或解析真值XML文件 {xml_file}. 分类指标将无法计算。错误: {e}")
        return None


class CustomClassificationEvaluator:
    """
    一个专用于分类任务的评估器，计算:
    1. 准确率 (Accuracy)
    2. 宏平均F1分数 (Macro-F1 Score)
    使用Top-K多数投票机制进行预测。
    """

    def __init__(self, num_query, gt_xml_path, top_k=1, feat_norm=True):
        super(CustomClassificationEvaluator, self).__init__()
        self.num_query = num_query
        self.gt_labels = parse_xml(gt_xml_path)
        self.top_k = top_k  # **新增: 保存K值**
        self.feat_norm = feat_norm
        print(f"Evaluator initialized with Top-K = {self.top_k}")

    def reset(self):
        self.feats = []
        self.pids = []
        self.img_paths = []

    def update(self, output):
        feat, pid, _, path = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.img_paths.extend(np.asarray(path))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        qf = feats[: self.num_query]
        q_img_paths = self.img_paths[: self.num_query]

        gf = feats[self.num_query :]
        g_pids = np.asarray(self.pids[self.num_query :])

        accuracy = -1.0
        macro_f1 = -1.0

        if self.gt_labels is None:
            print("错误: 真值XML未加载，无法计算分类指标。")
            return accuracy, macro_f1

        distmat = torch.cdist(qf, gf)

        # **关键修改：从获取Top-1改为获取Top-K**
        # largest=False表示我们取距离最小的K个
        _, top_k_indices = torch.topk(distmat, k=self.top_k, dim=1, largest=False)

        y_true = []
        y_pred = []

        for q_idx in range(self.num_query):
            query_filename = os.path.basename(q_img_paths[q_idx])

            if query_filename in self.gt_labels:
                y_true.append(self.gt_labels[query_filename])

                # **关键修改：进行多数投票**
                # 1. 获取Top-K个近邻的类别标签
                neighbor_labels = g_pids[top_k_indices[q_idx].cpu().numpy()]

                # 2. 使用Counter进行投票，找到出现次数最多的类别
                most_common_label = Counter(neighbor_labels).most_common(1)[0][0]

                y_pred.append(most_common_label)

        if len(y_true) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            print("警告: 验证集中的Query图像文件名与真值XML文件中的文件名无一匹配。")

        return accuracy, macro_f1

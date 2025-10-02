from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.camID = defaultdict(list)
        for index, (_, pid, camID, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
            self.camID[pid].append(camID)
        self.pids = list(self.index_dic.keys())
        self.camIDs = list(self.camID.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            cam_idxs = [[] for _ in range(2)]  # 假设只有两个摄像头ID: 0和1

            # 将索引按照摄像头ID分组
            for idx in idxs:
                camid = self.data_source[idx][2]
                if camid < 2:  # 只考虑camID为0和1的情况
                    cam_idxs[camid].append(idx)

            # 检查是否可以平衡camID为0和1的样本
            can_balance = len(cam_idxs[0]) >= self.num_instances // 2 and len(cam_idxs[1]) >= self.num_instances // 2

            if can_balance:
                # 平衡选择camID为0和1的样本
                balanced_idxs = []
                cam0_idxs = cam_idxs[0].copy()
                cam1_idxs = cam_idxs[1].copy()
                random.shuffle(cam0_idxs)
                random.shuffle(cam1_idxs)

                while len(balanced_idxs) < self.num_instances:
                    if len(balanced_idxs) % 2 == 0 and cam0_idxs:
                        balanced_idxs.append(cam0_idxs.pop(0))
                    elif cam1_idxs:
                        balanced_idxs.append(cam1_idxs.pop(0))
                    elif cam0_idxs:  # 如果cam1的样本用完了，使用cam0的样本
                        balanced_idxs.append(cam0_idxs.pop(0))
                    else:
                        break  # 两种camID的样本都用完了

                # 如果收集的样本不足，用随机选择补充
                if len(balanced_idxs) < self.num_instances:
                    remaining = self.num_instances - len(balanced_idxs)
                    if len(idxs) < remaining:
                        additional_idxs = np.random.choice(idxs, size=remaining, replace=True)
                    else:
                        additional_idxs = random.sample(idxs, remaining)
                    balanced_idxs.extend(additional_idxs)

                random.shuffle(balanced_idxs)
                batch_idxs_dict[pid].append(balanced_idxs)
            else:
                # 无法平衡，使用原来的随机采样方法
                if len(idxs) < self.num_instances:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
                random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(idx)
                    if len(batch_idxs) == self.num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

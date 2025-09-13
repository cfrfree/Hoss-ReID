import os
import glob
from torch.utils.data import Dataset
from .bases import read_image, sar32bit2RGB, BaseImageDataset
import random


def calculate_img_size(img, is_sar=False):
    """
    根据论文中的方法计算船舶尺寸嵌入的输入。
    GSD: 光学=0.75m, SAR=1m
    """
    img_size = img.size
    if not is_sar:
        img_size = [size * 0.75 for size in img_size]

    if img_size[0] == 0:
        aspect_ratio = 0
    else:
        aspect_ratio = img_size[1] / img_size[0]

    normalized_size = ((img_size[0] / 93 - 0.434) / 0.031, (img_size[1] / 427 - 0.461) / 0.031, aspect_ratio)
    return normalized_size


class CustomReIDDataset(BaseImageDataset):
    """
    用于ReID训练的数据集。
    将每个“目标”文件夹视为一个独立的身份(PID)。
    """

    def __init__(self, data_path, verbose=True):
        super(CustomReIDDataset, self).__init__()
        self.dataset_dir = data_path

        train = self._process_dir(self.dataset_dir)

        if verbose:
            print("=> Custom ReID Dataset loaded")
            self.print_dataset_statistics(train, train, train)  # 仅用于打印统计信息

        self.train = train
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, _ = self.get_imagedata_info(self.train)

    def _process_dir(self, dir_path):
        target_folders = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])

        dataset = []
        pid_container = set()

        for target_name in target_folders:
            pid_container.add(target_name)

        pid_to_label = {pid: label for label, pid in enumerate(pid_container)}

        for target_name in target_folders:
            pid = pid_to_label[target_name]
            target_path = os.path.join(dir_path, target_name)

            optical_folder = os.path.join(target_path, "光文件夹")
            sar_folder = os.path.join(target_path, "Sar文件夹")

            # 处理光学图像 (camid = 0)
            if os.path.exists(optical_folder):
                optical_images = glob.glob(os.path.join(optical_folder, "*.[p|j|t][n|p|i][g|e|f]*"))
                for img_path in optical_images:
                    dataset.append((img_path, pid, 0, 1))  # path, pid, camid, trackid(占位)

            # 处理SAR图像 (camid = 1)
            if os.path.exists(sar_folder):
                sar_images = glob.glob(os.path.join(sar_folder, "*.[p|j|t][n|p|i][g|e|f]*"))
                for img_path in sar_images:
                    dataset.append((img_path, pid, 1, 1))  # path, pid, camid, trackid(占位)

        return dataset


class InferenceGalleryDataset(Dataset):
    """用于加载测试/验证集中的光学参考图像（Gallery）"""

    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.img_items = []
        target_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        for target_name in target_folders:
            folder_path = os.path.join(data_path, target_name)
            image_files = glob.glob(os.path.join(folder_path, "*.[p|j|t][n|p|i][g|e|f]*"))
            for img_path in image_files:
                self.img_items.append((img_path, target_name))

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, target_name = self.img_items[index]
        img = read_image(img_path).convert("RGB")
        img_size = calculate_img_size(img, is_sar=False)
        if self.transform:
            img = self.transform(img)
        return img, target_name, img_size


class InferenceQueryDataset(Dataset):
    """用于加载测试/验证集中的SAR待分类图像（Query）"""

    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.img_items = []
        image_files = glob.glob(os.path.join(data_path, "*.[p|j|t][n|p|i][g|e|f]*"))
        for img_path in image_files:
            filename = os.path.basename(img_path)
            self.img_items.append((img_path, filename))

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, filename = self.img_items[index]
        img = read_image(img_path)
        img = sar32bit2RGB(img)
        img_size = calculate_img_size(img, is_sar=True)
        if self.transform:
            img = self.transform(img)
        return img, filename, img_size

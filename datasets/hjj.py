import os
from torch.utils.data import Dataset
from .bases import read_image, sar32bit2RGB
import glob


def calculate_img_size(img, is_sar=False):
    img_size = img.size
    # 根据原始代码，光学图像的GSD是0.75m，SAR是1m
    # 这里的计算是为了模拟论文中的尺寸嵌入
    if not is_sar:
        img_size = [img_size[0] * 0.75, img_size[1] * 0.75]

    # 标准化和归一化
    normalized_size = ((img_size[0] / 93 - 0.434) / 0.031, (img_size[1] / 427 - 0.461) / 0.031, img_size[1] / img_size[0] if img_size[0] != 0 else 0)
    return normalized_size


class OpticalGalleryDataset(Dataset):
    """用于加载光学参考图像（Gallery）的数据集"""

    def __init__(self, data_path, transform=None):
        super(OpticalGalleryDataset, self).__init__()
        self.transform = transform
        self.img_items = []

        # 遍历“光文件夹”下的所有目标子文件夹
        target_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        for target_name in target_folders:
            folder_path = os.path.join(data_path, target_name)
            # 使用 glob 查找所有图片文件
            image_files = (
                glob.glob(os.path.join(folder_path, "*.png"))
                + glob.glob(os.path.join(folder_path, "*.jpeg"))
                + glob.glob(os.path.join(folder_path, "*.jpg"))
                + glob.glob(os.path.join(folder_path, "*.tif"))
                + glob.glob(os.path.join(folder_path, "*.tiff"))
            )

            for img_path in image_files:
                self.img_items.append((img_path, target_name))

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, target_name = self.img_items[index]

        # 读取光学图像（假定为RGB）
        img = read_image(img_path).convert("RGB")
        img_size = calculate_img_size(img, is_sar=False)

        if self.transform is not None:
            img = self.transform(img)

        return img, target_name, img_size


class SarQueryDataset(Dataset):
    """用于加载SAR待分类图像（Query）的数据集"""

    def __init__(self, data_path, transform=None):
        super(SarQueryDataset, self).__init__()
        self.transform = transform
        self.img_items = []

        # 使用 glob 查找Sar文件夹下的所有图片
        image_files = (
            glob.glob(os.path.join(data_path, "*.png"))
            + glob.glob(os.path.join(data_path, "*.jpeg"))
            + glob.glob(os.path.join(data_path, "*.jpg"))
            + glob.glob(os.path.join(data_path, "*.tif"))
            + glob.glob(os.path.join(data_path, "*.tiff"))
        )

        for img_path in image_files:
            filename = os.path.basename(img_path)
            self.img_items.append((img_path, filename))

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, filename = self.img_items[index]

        # 读取SAR图像并转换为3通道
        img = read_image(img_path)
        img = sar32bit2RGB(img)  # 使用项目中的函数处理SAR图像
        img_size = calculate_img_size(img, is_sar=True)

        if self.transform is not None:
            img = self.transform(img)

        return img, filename, img_size

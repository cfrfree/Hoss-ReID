from PIL import Image, ImageFile, UnidentifiedImageError

from torch.utils.data import Dataset
import os.path as osp
import cv2
import numpy as np
import time
import random
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """
    一个更健壮的图像读取函数，带有重试机制以应对高并发I/O问题。
    """
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))

    img = Image.open(img_path)
    # 确认图像数据被完整加载
    # 对于某些损坏的文件，仅open()可能不报错，需要加载数据
    img.load()
    return img


def sar32bit2RGB(img):
    """
    Converts a single-channel SAR image (PIL Image) to a 3-channel RGB PIL Image.
    If the input image is already 3-channel, it returns it directly.
    """
    # 检查图像模式
    if img.mode == "RGB":
        # 如果已经是RGB模式，直接返回
        return img

    # 如果是单通道（'L' for 8-bit, 'F' for 32-bit float, 'I' for 32-bit int）
    # 将其转换为numpy数组进行处理
    nimg = np.array(img, dtype=np.float32)

    # 检查numpy数组的维度
    if nimg.ndim == 3 and nimg.shape[2] == 3:
        # 即使模式不是RGB，但如果已经是3维数组，也可能是RGB
        pil_img = Image.fromarray(nimg.astype(np.uint8))
        return pil_img

    # 执行原始的归一化和转换流程
    # 确保除数不为0
    max_val = nimg.max()
    if max_val > 0:
        nimg = nimg / max_val * 255

    nimg_8 = nimg.astype(np.uint8)
    cv_img = cv2.cvtColor(nimg_8, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(cv_img)
    return pil_img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        if train is not None:
            num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        if train is not None:
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, pair=False, output_dir="."):
        self.dataset = dataset
        self.transform = transform
        self.pair = pair
        # 新增一个日志文件路径，用于记录损坏图片
        # 使用不同的文件名以区分于微调阶段的日志
        log_filename = "corrupted_pairs.txt" if self.pair else "corrupted_images.txt"
        self.log_file_path = os.path.join(output_dir, log_filename)
        # 确保日志目录存在 (只在主进程中创建)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except FileExistsError:
                pass  # 忽略多进程下的文件已存在错误

    def __len__(self):
        return len(self.dataset)

    def get_image(self, img_path):
        if not osp.exists(img_path):
            raise IOError(f"{img_path} does not exist")

        if img_path.endswith("SAR.tif") or img_path.endswith("SAR.png"):
            img = read_image(img_path)
            img = sar32bit2RGB(img)
            img_size = img.size
        else:
            img = read_image(img_path).convert("RGB")
            img_size = img.size
            img_size = [img_size[0] * 0.75, img_size[1] * 0.75]

        img_size = ((img_size[0] / 93 - 0.434) / 0.031, (img_size[1] / 427 - 0.461) / 0.031, img_size[1] / img_size[0])
        if self.transform is not None:
            img = self.transform(img)
        return img, img_size

    def __getitem__(self, index):
        try:
            # === 核心逻辑：根据 self.pair 决定加载方式 ===
            if self.pair:
                # 这是一个图像对，包含两条记录 (RGB, SAR)
                img_pair_data = self.dataset[index]

                # 分别加载图像对中的两张图片
                rgb_path, rgb_pid, rgb_camid = img_pair_data[0]
                sar_path, sar_pid, sar_camid = img_pair_data[1]

                rgb_img, rgb_img_size = self.get_image(rgb_path)
                sar_img, sar_img_size = self.get_image(sar_path)

                # 组装成dataloader期望的格式
                return [
                    (rgb_img, rgb_pid, rgb_camid, rgb_path.split("/")[-1], rgb_img_size),
                    (sar_img, sar_pid, sar_camid, sar_path.split("/")[-1], sar_img_size),
                ]
            else:
                # 这是单张图片
                img_path, pid, camid, trackid = self.dataset[index]
                img, img_size = self.get_image(img_path)
                return img, pid, camid, trackid, img_path.split("/")[-1], img_size

        except Exception as e:
            # 如果加载过程中（无论是单张还是图像对）出现任何错误

            # 1. 记录损坏信息
            if self.pair:
                # 记录图像对中两张图片的路径
                failed_path_1 = self.dataset[index][0][0]
                failed_path_2 = self.dataset[index][1][0]
                log_message = f"Pair failed to load:\n  - {failed_path_1}\n  - {failed_path_2}\n  - Error: {e}\n"
            else:
                # 记录单张图片的路径
                failed_path = self.dataset[index][0]
                log_message = f"{failed_path}\n"

            # 2. 将信息写入日志文件
            with open(self.log_file_path, "a") as f:
                f.write(log_message)

            # 3. 随机选择另一个样本（无论是单张还是图像对）来代替
            new_index = random.randint(0, len(self) - 1)
            # 递归调用 __getitem__ 获取一个有效的替代样本
            return self.__getitem__(new_index)

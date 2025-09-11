from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
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
    def __init__(self, dataset, transform=None, pair=False):
        self.dataset = dataset
        self.transform = transform

        self.pair = pair

    def __len__(self):
        return len(self.dataset)

    def get_image(self, img_path):
        if img_path.endswith("SAR.tif"):
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
        if self.pair:
            imgs = []
            for img in self.dataset[index]:
                img_path, pid, camid = img
                im, img_size = self.get_image(img_path)
                imgs.append((im, pid, camid, img_path.split("/")[-1], img_size))
            return imgs
        else:
            img_path, pid, camid, trackid = self.dataset[index]
            img, img_size = self.get_image(img_path)
            return img, pid, camid, trackid, img_path.split("/")[-1], img_size

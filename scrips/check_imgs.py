import os
import argparse
from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np
import cv2

# 允许加载可能被截断的图像文件，增加鲁棒性
ImageFile.LOAD_TRUNCATED_IMAGES = True


def sar32bit2RGB(img):
    """
    将 32位的 SAR 图像转换为 RGB 格式。
    这个函数直接从你的项目文件 datasets/bases.py 中复制而来，并增加了安全检查。
    """
    # 将PIL图像转换为numpy数组
    nimg = np.array(img, dtype=np.float32)

    # 安全检查：如果图像是全黑的，最大值为0，直接除以0会报错。
    max_val = nimg.max()
    if max_val == 0:
        # 你可以选择返回一个全黑的图像，或者像这里一样抛出一个明确的错误
        raise ValueError("Image is completely black (max pixel value is 0), cannot normalize.")

    # 归一化到 0-255 范围
    nimg = (nimg / max_val) * 255.0

    # 转换为 8-bit 整数
    nimg_8 = nimg.astype(np.uint8)

    # 使用 OpenCV 将灰度图转换为 RGB
    # OpenCV 默认处理 BGR，但由于输入是灰度，转换到RGB或BGR结果相同
    cv_img = cv2.cvtColor(nimg_8, cv2.COLOR_GRAY2RGB)

    # 将处理后的 numpy 数组转回 PIL 图像
    pil_img = Image.fromarray(cv_img)
    return pil_img


def find_corrupted_images(root_dir):
    """
    遍历指定目录及其子目录，检查所有图片文件是否损坏，并显示进度条。
    检查逻辑完全模拟训练时的数据加载流程。

    Args:
        root_dir (str): 要检查的根文件夹路径。
    """
    if not os.path.isdir(root_dir):
        print(f"错误: 提供的路径不是一个有效的文件夹 -> {root_dir}")
        return

    corrupted_files = []
    # 支持的常见图片格式
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

    print(f"第一步: 正在收集所有图片文件路径...")
    # 首先，遍历一次以收集所有图片文件的路径，这样tqdm才能知道总数
    all_image_files = []
    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in image_extensions:
                all_image_files.append(os.path.join(subdir, filename))

    if not all_image_files:
        print("在指定文件夹下没有找到任何图片文件。")
        return

    print(f"共找到 {len(all_image_files)} 张图片，开始逐一检查（模拟训练加载流程）...\n")

    # 使用tqdm包装迭代器以显示进度条
    for filepath in tqdm(all_image_files, desc="检查进度", unit="张", ncols=100):
        try:
            # 1. 用PIL打开图片
            img = Image.open(filepath)

            # 2. 核心步骤：应用与训练时完全相同的处理逻辑
            if filepath.lower().endswith("sar.tif"):
                # 如果是SAR图像，调用特殊处理函数
                processed_img = sar32bit2RGB(img)
            else:
                # 如果是普通图像，转换为RGB
                processed_img = img.convert("RGB")

            # 3. 强制加载图像数据，以捕获延迟的错误（如文件截断）
            processed_img.load()

        except Exception as e:
            # 如果任何一步出错，都记录下来
            corrupted_files.append((filepath, str(e)))

    # 扫描结束后，打印最终报告
    print("\n" + "=" * 70)
    if not corrupted_files:
        print("🎉 扫描完成！所有图片均能被训练流程正确处理。")
    else:
        print(f"扫描完成！共发现 {len(corrupted_files)} 张在训练中会出错的图片：\n")
        for f_path, error_msg in corrupted_files:
            print(f"[-] 文件路径: {f_path}")
            print(f"    错误原因: {error_msg}\n")
    print("=" * 70)


if __name__ == "__main__":
    find_corrupted_images("/home/share/chenfree/ReID/OptiSar_Pair_Plus/")

import os
from PIL import Image, ImageFile
from tqdm import tqdm  # 导入 tqdm

# 允许加载可能被截断的图像文件，增加鲁棒性
ImageFile.LOAD_TRUNCATED_IMAGES = True


def find_corrupted_images(root_dir):
    """
    遍历指定目录及其子目录，检查所有图片文件是否损坏，并显示进度条。

    Args:
        root_dir (str): 要检查的根文件夹路径。
    """
    if not os.path.isdir(root_dir):
        print(f"错误: 提供的路径不是一个有效的文件夹 -> {root_dir}")
        return

    corrupted_files = []
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

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

    print(f"共找到 {len(all_image_files)} 张图片，开始逐一检查...\n")

    # 使用tqdm包装迭代器以显示进度条
    # desc是进度条的描述文字
    for filepath in tqdm(all_image_files, desc="检查进度", unit="张"):
        try:
            # 尝试打开图片
            img = Image.open(filepath)
            # 强行加载图片数据，某些损坏（如截断）只有在加载时才会报错
            img.load()
        except Exception as e:
            # 如果打开或加载失败，记录文件路径
            # 为了保持进度条的美观，先不在这里打印错误，最后统一打印
            corrupted_files.append((filepath, str(e)))

    # 扫描结束后，清空tqdm留下的最后一行，开始打印报告
    print("\n" + "=" * 70)
    if not corrupted_files:
        print("🎉 扫描完成！未发现任何损坏的图片。")
    else:
        print(f"扫描完成！共发现 {len(corrupted_files)} 张损坏的图片：\n")
        for f_path, error_msg in corrupted_files:
            print(f"[-] 文件路径: {f_path}")
            print(f"    错误原因: {error_msg}\n")
    print("=" * 70)


if __name__ == "__main__":

    find_corrupted_images("/home/share/chenfree/ReID/OptiSar_Pair_Plus/")

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def create_annotations_and_rename(root_folder, output_xml_file="annotations.xml", file_prefix="sar_test", subfolders_to_rename=None):
    """
    遍历文件夹中的图片，有选择地重命名它们，并生成XML标注文件。

    :param root_folder: 包含类别子文件夹的根目录。
    :param output_xml_file: 输出的XML文件名。
    :param file_prefix: 重命名后的文件前缀。
    :param subfolders_to_rename: 一个列表，包含需要重命名其中文件的子文件夹名称。
    """
    if subfolders_to_rename is None:
        subfolders_to_rename = []  # 默认为空列表，即不重命名任何文件

    # 支持的图片文件扩展名
    supported_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]

    # 初始化XML根节点
    annotation_node = ET.Element("annotation")

    # 全局图片计数器，只为被重命名的图片递增
    image_counter = 1

    print(f"开始处理文件夹: {root_folder}")
    print(f"将对以下子文件夹内的图片进行重命名: {subfolders_to_rename}")

    # 获取所有类别文件夹 (例如 A220, B-52, ...)
    class_folders = sorted([f.path for f in os.scandir(root_folder) if f.is_dir()])

    for class_folder in class_folders:
        class_name = os.path.basename(class_folder)
        print(f"  正在处理类别: {class_name}")

        # 遍历类别文件夹内部的子文件夹 (例如 Sar文件夹, 光文件夹)
        sub_folders = sorted([f.path for f in os.scandir(class_folder) if f.is_dir()])

        for sub_folder in sub_folders:
            sub_folder_name = os.path.basename(sub_folder)

            # --- 核心改动：判断当前子文件夹是否需要重命名 ---
            should_rename = sub_folder_name in subfolders_to_rename

            if should_rename:
                print(f"    正在扫描并重命名: {sub_folder_name}")
            else:
                print(f"    正在扫描 (仅记录，不重命名): {sub_folder_name}")

            files_in_folder = sorted(os.listdir(sub_folder))

            for filename in files_in_folder:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in supported_extensions:
                    continue

                final_filename_for_xml = filename  # 默认使用原始文件名

                if should_rename:
                    # --- 执行重命名逻辑 ---
                    new_filename = f"{file_prefix}_{image_counter:03d}{file_ext}"
                    old_filepath = os.path.join(sub_folder, filename)
                    new_filepath = os.path.join(sub_folder, new_filename)
                    try:
                        os.rename(old_filepath, new_filepath)
                        print(f"      - 已重命名 '{filename}' -> '{new_filename}'")
                        final_filename_for_xml = new_filename
                        image_counter += 1  # 只在这里递增计数器
                    except OSError as e:
                        print(f"      - [错误] 重命名文件失败: {old_filepath}, {e}")
                        continue  # 如果重命名失败，则跳过此文件的XML记录
                else:
                    # --- 跳过重命名，仅打印记录信息 ---
                    print(f"      - 已记录 '{filename}' (不重命名)")

                # --- 公共的XML构建逻辑 ---
                object_node = ET.SubElement(annotation_node, "object")
                filename_node = ET.SubElement(object_node, "filename")
                filename_node.text = final_filename_for_xml
                type_node = ET.SubElement(object_node, "type")
                type_node.text = class_name

    # --- XML文件生成 ---
    xml_str = ET.tostring(annotation_node, "utf-8")
    reparsed = minidom.parseString(xml_str)
    pretty_xml_str = reparsed.toprettyxml(indent="  ", encoding="utf-8")

    try:
        with open(output_xml_file, "wb") as f:
            f.write(pretty_xml_str)
        print(f"\n处理完成！已成功生成标注文件: {output_xml_file}")
    except IOError as e:
        print(f"\n[错误] 写入XML文件失败: {output_xml_file}, {e}")


if __name__ == "__main__":
    # --- 配置 ---
    image_root_folder = r"C:\Users\FREE\Desktop\Aircraft\val"
    output_xml = "val.xml"
    prefix = "sar_test"

    # --- 关键配置：在这里指定哪些子文件夹需要被重命名 ---
    folders_to_process_for_renaming = ["Sar文件夹"]

    if not os.path.isdir(image_root_folder):
        print(f"[错误] 文件夹 '{image_root_folder}' 不存在。请检查路径是否正确。")
    else:
        create_annotations_and_rename(
            root_folder=image_root_folder, output_xml_file=output_xml, file_prefix=prefix, subfolders_to_rename=folders_to_process_for_renaming
        )

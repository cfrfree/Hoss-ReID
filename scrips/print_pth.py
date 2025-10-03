import torch
import argparse
import os


def inspect_checkpoint(filepath):
    """
    加载并检查 PyTorch 的 .pth 文件内容。

    Args:
        filepath (str): .pth 文件的路径。
    """
    if not os.path.exists(filepath):
        print(f"错误: 文件未找到，请检查路径 -> {filepath}")
        return

    print(f"正在加载文件: {filepath}\n")

    try:
        # 加载模型权重，map_location='cpu'确保即使文件是在GPU上保存的，也能在CPU上加载
        checkpoint = torch.load(filepath, map_location="cpu")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # 检查 checkpoint 是否是一个字典 (最常见的情况)
    if not isinstance(checkpoint, dict):
        print(f"文件内容不是一个字典，而是一个 {type(checkpoint)}。")
        # 如果是单个张量，打印其形状
        if isinstance(checkpoint, torch.Tensor):
            print(f"张量形状: {checkpoint.shape}")
        return

    print("=" * 60)
    print("文件顶层键 (Top-level Keys):")
    print(list(checkpoint.keys()))
    print("=" * 60)
    print("\n详细结构分析:\n")

    # 递归函数，用于遍历并打印字典结构
    def recursive_print(data, indent=0):
        # 常见的模型权重键，通常包含 'state_dict' 或 'model'
        potential_state_dict_keys = ["state_dict", "model", "net"]

        # 优先处理常见的权重字典
        for key in potential_state_dict_keys:
            if key in data and isinstance(data[key], dict):
                print(f"{'  ' * indent}键: '{key}' (这是一个模型状态字典):")
                state_dict = data[key]
                max_key_len = max(len(k) for k in state_dict.keys()) if state_dict else 0
                for name, param in state_dict.items():
                    print(f"{'  ' * (indent + 1)}{name:<{max_key_len}} | 形状: {param.shape} | 类型: {param.dtype}")
                # 打印完后从原始数据中移除，避免重复打印
                data.pop(key)

        # 遍历剩余的键值对
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"\n{'  ' * indent}键: '{key}' (这是一个字典):")
                recursive_print(value, indent + 1)
            elif isinstance(value, torch.Tensor):
                print(f"{'  ' * indent}键: '{key}' | 张量形状: {value.shape} | 类型: {value.dtype}")
            else:
                # 打印其他类型的数据，如 epoch 数、optimizer 状态等
                print(f"{'  ' * indent}键: '{key}' | 值: {value} | 类型: {type(value).__name__}")

    # 从顶层开始遍历
    recursive_print(checkpoint)
    print("\n" + "=" * 60)
    print("检查完毕。")


if __name__ == "__main__":
    file_path = "/home/share/chenfree/ReID/jx_vit_large_p16_384-b3be5167.pth"
    inspect_checkpoint(file_path)

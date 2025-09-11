import argparse
import xml.etree.ElementTree as ET
from sklearn.metrics import accuracy_score, f1_score


def parse_xml(xml_file):
    """
    解析XML文件，返回一个 {filename: type} 的字典。
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"错误: 无法解析XML文件 {xml_file}. 请检查文件格式。错误信息: {e}")
        return None
    except FileNotFoundError:
        print(f"错误: 文件未找到 {xml_file}")
        return None

    results = {}
    for obj in root.findall("object"):
        filename = obj.find("filename").text
        obj_type = obj.find("type").text
        if filename in results:
            print(f"警告: 文件名 '{filename}' 在 {xml_file} 中重复出现。将使用最后一次出现的值。")
        results[filename] = obj_type

    return results


def evaluate(predicted_xml, ground_truth_xml):
    """
    比较两个XML文件并计算分类指标。
    """
    print(f"正在加载预测文件: {predicted_xml}")
    predicted_data = parse_xml(predicted_xml)

    print(f"正在加载真实标签文件: {ground_truth_xml}")
    truth_data = parse_xml(ground_truth_xml)

    if predicted_data is None or truth_data is None:
        return

    # 找出两个文件中共同的文件名
    common_files = sorted(list(set(predicted_data.keys()) & set(truth_data.keys())))

    if not common_files:
        print("错误: 预测文件和真实标签文件之间没有共同的文件名。无法进行评测。")
        return

    # 检查是否有文件不匹配
    if len(predicted_data) != len(truth_data) or len(common_files) != len(predicted_data):
        print("\n警告: 文件列表不完全匹配。评测将只针对两个文件中都存在的图像进行。")
        print(f"  - 预测文件中的图像数: {len(predicted_data)}")
        print(f"  - 真实标签文件中的图像数: {len(truth_data)}")
        print(f"  - 将用于评测的共同图像数: {len(common_files)}\n")

    # 根据共同文件列表，准备用于scikit-learn计算的列表
    y_pred = [predicted_data[fname] for fname in common_files]
    y_true = [truth_data[fname] for fname in common_files]

    # --- 计算指标 ---

    # 1. 总体准确率 (Overall Accuracy)
    #    计算的是正确分类的样本数占总样本数的比例。
    accuracy = accuracy_score(y_true, y_pred)

    # 2. 宏平均F1分数 (Macro-F1 Score)
    #    它会独立计算每个类别的F1分数，然后取所有类别F1分数的算术平均值。
    #    这使得每个类别都有相同的权重，即使某些类别的样本很少，它们在指标中的重要性也不会降低。
    #    适用于类别不均衡的情况。
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 3. 计算两者的平均值
    final_score = (accuracy + macro_f1) / 2.0

    # --- 打印结果 ---
    print("================== 评测结果 ==================")
    print(f"总体准确率 (Accuracy): {accuracy:.4f}")
    print(f"宏平均F1分数 (Macro-F1): {macro_f1:.4f}")
    print("----------------------------------------------")
    print(f"最终平均分: {final_score:.4f}")
    print("==============================================")

    # (可选) 打印每个类别的详细报告
    try:
        from sklearn.metrics import classification_report

        print("\n详细分类报告:\n")
        # 获取所有唯一的类别标签
        labels = sorted(list(set(y_true + y_pred)))
        print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    except ImportError:
        print("\n提示: 如果想查看详细的分类报告，请安装 'scikit-learn'。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测分类结果XML文件")
    parser.add_argument(
        "--predicted_xml",
        default="result.xml",
    )
    parser.add_argument("--ground_truth_xml", default="/home/share/chenfree/ReID/Aircraft/annotations.xml")

    args = parser.parse_args()

    evaluate(args.predicted_xml, args.ground_truth_xml)

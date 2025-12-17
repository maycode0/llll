import json
from collections import Counter
import os

def analyze_json_ids_stats(json_file_path):
    """
    读取JSON文件，统计相同 'ids' 出现的次数，并计算平均对象数量。
    
    Args:
        json_file_path (str): JSON文件的完整路径。
    """
    if not os.path.exists(json_file_path):
        print(f"错误：文件未找到，请检查路径: {json_file_path}")
        return

    # 1. 读取并加载JSON数据
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("错误：JSON文件格式不正确，无法解析。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    if not isinstance(data, list):
        print("错误：JSON文件内容不是列表（预期格式为 [...]）。")
        return

    # 2. 统计每个ids出现的次数
    # 存储所有ids的列表
    all_ids = [item.get("ids") for item in data if isinstance(item, dict) and "ids" in item]
    
    # Counter 自动统计每个元素出现的频率
    id_counts = Counter(all_ids)
    
    # 3. 计算统计结果
    
    # 总对象数量
    total_objects = len(data)
    
    # 独立 ids 的数量 (即 Counter 的长度)
    unique_ids_count = len(id_counts)
    
    # 平均每个 ids 产生的对象数量
    if unique_ids_count > 0:
        avg_objects_per_id = total_objects / unique_ids_count
    else:
        avg_objects_per_id = 0

    # 4. 打印结果
    print("=" * 40)
    print(f"✅ 文件分析结果: {json_file_path}")
    print("=" * 40)
    print(f"总对象数量 (Total Objects): {total_objects}")
    print(f"独立 IDs 数量 (Unique IDs): {unique_ids_count}")
    print(f"⭐ 平均每个 IDs 产生的对象数量: {avg_objects_per_id:.2f} 条")
    print("-" * 40)
    
    # 5. 展示部分详细统计（可选）
    print("详细统计 (Top 10 出现频率最高的 IDs):")
    for ids, count in id_counts.most_common(5):
        print(f"  - ID {ids}: {count} 条")
    print("-" * 40)
    # 5. 展示部分详细统计（可选）
    print("详细统计 (Top 10 出现频率最低的 IDs):")
    for ids, count in id_counts.most_common()[1000-6:1000]:
        print(f"  - ID {ids}: {count} 条")
    print("-" * 40)


# =======================================================
# 📌 示例运行：请替换为您的实际文件路径
# =======================================================
if __name__ == "__main__":
    # 假设您的文件名为 'data.json'，且位于同一目录下
    # 请修改为您实际的文件路径
    file_path = "datasets/output_data20251129164626.json" 
    
    # ⚠️ 在此处替换为您的文件路径
    # 例如：file_path = "datasets/output_data20251129164626.json" 
    
    # 运行分析函数
    analyze_json_ids_stats(file_path)
import json

def count_elements_in_json_array(file_path: str) -> int:
    """
    统计JSON文件中数组的元素数量。
    适用于文件整体是一个数组结构，每个元素是对象(dict)。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 直接解析为Python对象
    if isinstance(data, list):
        return len(data)
    else:
        raise ValueError("JSON 文件的顶层不是数组。")

if __name__ == "__main__":
    json_file = "llava_v1_5_mix665k_fixed.json"   # 修改为你的文件路径
    try:
        count = count_elements_in_json_array(json_file)
        print(f"JSON文件中共有 {count} 个元素")
    except Exception as e:
        print(f"发生错误: {e}")

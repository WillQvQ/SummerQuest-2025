import json
import random

def process_and_split_data(input_file, is_with_search_data=False):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。将跳过此文件。")
        return [], []
        
    print(f"\n正在处理文件: '{input_file}'，包含 {len(source_data)} 条数据...")
    
    formatted_data = []
    
    for item in source_data:
        question = item.get("question")
        response_content = item.get("response")
        
        if not question or not response_content:
            continue

        contains_tool_code = "<tool_code>" in response_content
        if is_with_search_data != contains_tool_code:
            data_type = "with_search" if is_with_search_data else "no_search"
            print(f"警告: 在 '{data_type}' 文件中发现类型不匹配的数据。问题: {question[:30]}...")

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response_content}
        ]
        
        formatted_data.append({"messages": messages})

    # 在切分前打乱数据
    random.shuffle(formatted_data)
    
    return formatted_data


def save_to_jsonl(data, output_file):
    """将数据列表保存为JSONL文件。"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"成功保存 {len(data)} 条数据到: '{output_file}'")


if __name__ == '__main__':
    # --- 配置 ---
    # 1. 源文件路径
    with_search_source_file = '/root/hw4_3/synthetic_data_withsearch.json'
    no_search_source_file = '/root/hw4_3/synthetic_data_nosearch.json'

    # 2. 目标文件路径
    # 注意：现在我们只有两个总的输出文件，而不是四个
    train_target_file = '/root/hw4_3/qwen_tool_train_mixed.jsonl'
    eval_target_file = '/root/hw4_3/qwen_tool_train_eval_mixed.jsonl'
    
    # 3. 验证集比例
    eval_ratio = 0.1

    # --- 执行处理 ---
    # 分别处理两个源文件
    all_with_search_data = process_and_split_data(with_search_source_file, is_with_search_data=True)
    all_no_search_data = process_and_split_data(no_search_source_file, is_with_search_data=False)

    # 从各自的数据中切分训练集和验证集
    ws_split_index = int(len(all_with_search_data) * (1 - eval_ratio))
    ws_train = all_with_search_data[:ws_split_index]
    ws_eval = all_with_search_data[ws_split_index:]

    ns_split_index = int(len(all_no_search_data) * (1 - eval_ratio))
    ns_train = all_no_search_data[:ns_split_index]
    ns_eval = all_no_search_data[ns_split_index:]

    # 合并训练集和验证集
    final_train_data = ws_train + ns_train
    final_eval_data = ws_eval + ns_eval
    
    # 再次打乱，确保最终的训练和验证集内部是混合的
    random.shuffle(final_train_data)
    random.shuffle(final_eval_data)
    
    print("\n--- 数据集切分结果 ---")
    print(f"总训练集大小: {len(final_train_data)}")
    print(f"总验证集大小: {len(final_eval_data)}")
    print("----------------------\n")
    
    # 保存最终的文件
    save_to_jsonl(final_train_data, train_target_file)
    save_to_jsonl(final_eval_data, eval_target_file)

    print("\n所有数据格式化和切分完成！")
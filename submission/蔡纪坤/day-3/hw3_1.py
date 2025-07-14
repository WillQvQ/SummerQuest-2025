from transformers import AutoTokenizer
import json, os

# 模型和文件路径
MODEL_PATH = "/data-mnt/data/downloaded_ckpts/Qwen3-8B"
INPUT_JSON = "query_and_output.json"
OUTPUT_JSON = "hw3_1.json"
TOKENIZER_SAVE_PATH = "./tokenizer_with_special_tokens"  # 保存tokenizer的本地路径

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 2. 定义特殊 tokens
new_tokens = ["<|AGENT|>", "<|EDIT|>"] # TODO

# 3. 添加特殊 tokens
# TODO
special_tokens_dict={"additional_special_tokens": ["<|AGENT|>","<|EDIT|>"]}
tokenizer.add_special_tokens(special_tokens_dict)

# 4. 保存修改后的tokenizer到本地
# TODO
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
# 5. 读取原始的 Query&Output
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    tasks = json.load(f)

# 6. 合并 Query 和 Output，生成输出记录
records = {
    "special_tokens": [
        {
            "token": token,
            "id": tokenizer.convert_tokens_to_ids(token)
        } for token in new_tokens
    ],
    "tasks": []
}

for item in tasks:
    # 合并字段
    merged_text = item["Query"].strip() + "\n" + item["Output"].strip()
    # 编码并获取 token IDs
    ids = tokenizer.encode(merged_text) # TODO
    # 解码验证
    decoded = tokenizer.decode(ids) # TODO
    records["tasks"].append({
        "text": merged_text,
        "token_ids": ids,
        "decoded_text": decoded
    })

# 7. 答案写入 JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

# 8. 定义打印 token 详情的函数
def print_token_details(task, title):
    print(f"\n=== {title} ===")
    print(f"\n--- Token详情 (每行一个token和ID) ---")
    for i, token_id in enumerate(task['token_ids']):
        # 将单个token ID转换回文字
        token_text = tokenizer.decode([token_id])
        
        # 处理特殊字符显示
        if token_text == '\n':
            display_text = '\\n'
        elif token_text == '\t':
            display_text = '\\t'
        elif token_text == ' ':
            display_text = '_'  # 用下划线表示空格
        elif token_text.strip() == '':
            display_text = f"'{token_text}'"  # 其他空白字符用引号包围
        else:
            display_text = token_text
        
        print(f"Token {i:3d}: {display_text:15} | ID: {token_id}")
    print()

# 9. 展示第一条和最后一条数据的结果
if records["tasks"]:
    print_token_details(records["tasks"][0], "第一条数据")
    
    if len(records["tasks"]) > 1:
        print_token_details(records["tasks"][-1], "最后一条数据")
    else:
        print("\n只有一条数据")

print(f"\n已生成 {OUTPUT_JSON}，共处理 {len(records['tasks'])} 条数据")

# 10. 验证保存的tokenizer
print(f"\n=== 验证保存的tokenizer ===")
try:
    # 重新加载保存的tokenizer
    loaded_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SAVE_PATH)
    print(f"成功从 {TOKENIZER_SAVE_PATH} 加载tokenizer")
    
    # 验证特殊tokens是否正确保存
    for token in new_tokens:
        original_id = tokenizer.convert_tokens_to_ids(token)
        loaded_id = loaded_tokenizer.convert_tokens_to_ids(token)
        print(f"特殊token '{token}': 原始ID={original_id}, 加载后ID={loaded_id}, 一致性={'✓' if original_id == loaded_id else '✗'}")
        
except Exception as e:
    print(f"加载保存的tokenizer时出错: {e}")
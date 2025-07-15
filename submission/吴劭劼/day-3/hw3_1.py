from transformers import AutoTokenizer
import json, os

# 模型和文件路径
MODEL_PATH = "/data-mnt/data/downloaded_ckpts/Qwen3-8B"
INPUT_JSON = "./SummerQuest-2025/handout/day-3/query_and_output.json"
OUTPUT_JSON = "./SummerQuest-2025/handout/day-3/hw3_1.json"
TOKENIZER_SAVE_PATH = "./tokenizer_with_special_tokens"  # 保存tokenizer的本地路径

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 2. 定义特殊 tokens
new_tokens = ["<|AGENT|>", "<|EDIT|>"]

# 3. 添加特殊 tokens
special_tokens = {"additional_special_tokens": new_tokens}
num_added = tokenizer.add_special_tokens(special_tokens)
print(f"Added {num_added} special tokens.")

# 4. 保存修改后的tokenizer到本地
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
print(f"Tokenizer with special tokens saved to {TOKENIZER_SAVE_PATH}")

# 5. 读取原始的 Query&Output
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    tasks = json.load(f)

# 6. 合并 Query 和 Output，生成输出记录
records = {
    "special_tokens": [
        {"token": token, "id": tokenizer.convert_tokens_to_ids(token)}
        for token in new_tokens
    ],
    "tasks": []
}

for item in tasks:
    # 合并字段
    merged_text = item["Query"].strip() + "\n" + item["Output"].strip()
    # 编码并获取 token IDs
    encoded = tokenizer(merged_text, add_special_tokens=False)
    ids = encoded["input_ids"]
    # 解码验证
    decoded = tokenizer.decode(ids)
    records["tasks"].append({
        "text": merged_text,
        "token_ids": ids,
        "decoded_text": decoded
    })

# 7. 答案写入 JSON
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"Saved {OUTPUT_JSON} with {len(records['tasks'])} tasks.")

# 8. 定义打印 token 详情的函数
def print_token_details(task, title):
    print(f"\n=== {title} ===")
    print("\n--- Token详情 (每行一个token和ID) ---")
    for i, token_id in enumerate(task['token_ids']):
        token_text = tokenizer.decode([token_id])
        if token_text == '\n': display = '\\n'
        elif token_text == '\t': display = '\\t'
        elif token_text == ' ': display = '_'
        elif token_text.strip() == '': display = f"'{token_text}'"
        else: display = token_text
        print(f"Token {i:3d}: {display:15} | ID: {token_id}")
    print()

# 9. 展示第一条和最后一条数据的结果
if records["tasks"]:
    print_token_details(records["tasks"][0], "第一条数据")
    if len(records["tasks"]) > 1:
        print_token_details(records["tasks"][-1], "最后一条数据")
    else:
        print("\n只有一条数据")

# 10. 验证保存的tokenizer
print(f"\n=== 验证保存的tokenizer ===")
try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SAVE_PATH)
    print(f"成功从 {TOKENIZER_SAVE_PATH} 加载tokenizer")
    for token in new_tokens:
        orig = tokenizer.convert_tokens_to_ids(token)
        loaded = loaded_tokenizer.convert_tokens_to_ids(token)
        print(f"特殊token '{token}': 原始ID={orig}, 加载后ID={loaded}, 一致性={'✓' if orig==loaded else '✗'}")
except Exception as e:
    print(f"加载保存的tokenizer时出错: {e}")

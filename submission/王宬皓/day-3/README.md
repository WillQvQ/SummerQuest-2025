> https://github.com/henrywch/SummerQuest-2025.git

### Day 3 HW

#### Adding Special Tokens

- *hw3_1.py* essential codes

```python

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 2. 定义特殊 tokens
new_tokens = ["<|AGENT|>", "<|EDIT|>"] # TODO

# 3. 添加特殊 tokens
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens}) # TODO

# 4. 保存修改后的tokenizer到本地
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH) # TODO

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
    ids = tokenizer.encode(merged_text, add_special_tokens=True) # TODO
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

```

#### Agents with Special Tokens

- *hw3_2.py* essential codes

Apply In-Context Learning.

```python

def generate_prompt(query: str) -> str:
    """
    为单个查询生成prompt
    """
    # TODO
    # The system content should guide the model to use the special tokens and tools.
    # We want it to act as a Github Copilot, using AGENT mode (python then editor) for debugging
    # and EDIT mode (editor only) for direct code modification/merging.
    system_content = (
        "你是一个Github Copilot，能够帮助用户调试、分析和修改代码。请根据用户需求选择以下两种模式进行响应：\n"
        "1. **代理模式 (<|AGENT|>)**: 当用户需要调试或分析代码问题时，先使用 `python` 工具执行代码进行调试和分析，然后使用 `editor` 工具进行修改。代理模式的输出应该以 `<|AGENT|>` 开头。\n"
        "2. **编辑模式 (<|EDIT|>)**: 当用户需要直接修改、合并代码或进行代码重构时，直接使用 `editor` 工具。编辑模式的输出应该以 `<|EDIT|>` 开头。\n"
        "请确保你的输出严格遵循所选模式的格式，并调用相应的工具。"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tools=tools, 
        tokenize=False, 
        add_generation_prompt=True
    ) # TODO
    
    return text

```

#### Checking Special Tokenized Outputs

- running *output_checker.py* 

```python
🚀 开始检查文件...
📁 文件路径: hw3_2.json

============================================================
📋 hw3_checker.py 检查结果
============================================================
📊 总体统计:
   总项目数: 10
   ✅ 通过: 10
   ❌ 失败: 0
   📈 通过率: 100.0%

📝 详细检查结果:
   项目 0: ✅ 通过所有检查
   项目 1: ✅ 通过所有检查
   项目 2: ✅ 通过所有检查
   项目 3: ✅ 通过所有检查
   项目 4: ✅ 通过所有检查
   项目 5: ✅ 通过所有检查
   项目 6: ✅ 通过所有检查
   项目 7: ✅ 通过所有检查
   项目 8: ✅ 通过所有检查
   项目 9: ✅ 通过所有检查
============================================================
```
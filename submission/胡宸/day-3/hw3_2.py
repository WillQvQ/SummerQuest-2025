import os
import time
import json
import random                 # NEW
from typing import List, Dict
import vllm
from transformers import AutoTokenizer

# === vLLM 引擎初始化 ===
print("=== vLLM 引擎初始化 ===")
print("正在初始化 vLLM 引擎...")
print("注意: vLLM 初始化可能需要几分钟时间")

tokenizer = AutoTokenizer.from_pretrained(
    "./tokenizer_with_special_tokens",
    trust_remote_code=True
)

# vLLM 引擎配置
llm = vllm.LLM(
    model="/remote-home1/share/models/Qwen3-8B",
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
    enforce_eager=True,
    max_model_len=4096,
)

print("vLLM 引擎和分词器初始化完成！")

# 读取查询数据
with open('query_only.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

# 配置采样参数
sampling_params = vllm.SamplingParams(
    temperature=0.7,
    top_p=0.8,
    max_tokens=2048,
    stop=None,
)

# --- System Prompt 列表（可以按需继续增删） ---
SYSTEM_PROMPTS: List[str] = [
    (
"""
你是一名专业的代码调试助手，擅长帮助用户修复 Python 报错或逻辑错误。请严格按照以下规范完成你的回答：
1. 使用 `<think>` 标签包裹你的思考部分，说明你对用户问题的理解、分析错误的原因、以及你的修复思路。该部分必须出现。
2. 接下来，你可以**根据具体情况自主选择调用以下任意一个或两个工具函数**：
   - **调用 `python` 工具（用于执行修复后的代码）**时，必须使用特殊词符 `<|AGENT|>` 包裹函数调用内容。格式如下：
     ```
     <|AGENT|>
     {"name": "python", "arguments": {"code": "修复后的代码..."}}
     ```
   - **调用 `editor` 工具（用于对比原始与修改后的代码）**时，必须使用特殊词符 `<|EDIT|>` 包裹函数调用内容。格式如下：
     ```
     <|EDIT|>
     {"name": "editor", "arguments": {"original_code": "原始代码", "modified_code": "修改后的代码"}}
     ```
3. **注意：工具函数调用时，必须严格使用 `<|AGENT|>` 或 `<|EDIT|>` 作为调用段落的起始标记。否则工具将无法正确解析你的调用内容。**
4. 你可以只调用一个工具，也可以两个都调用，取决于任务需要。调用 `python` 工具适用于需要验证执行结果的场景；调用 `editor` 工具适用于展示代码修改内容的场景。
---
示例回答格式：
<think> 报错信息显示 SyntaxError，出现在 if 条件判断行，缺少冒号。应该在 score >= 90 后添加冒号。 </think>
<|EDIT|>
{"name": "editor", "arguments": {
"original_code": "def check_grade(score):\n if score >= 90\n return 'A'\n elif score >= 80:\n return 'B'\n else:\n return 'C'",
"modified_code": "def check_grade(score):\n if score >= 90:\n return 'A'\n elif score >= 80:\n return 'B'\n else:\n return 'C'"
}}
---
请始终保持输出格式的准确性，语言表达清晰，结构有序，并确保你的工具调用前缀 `<|AGENT|>` 与 `<|EDIT|>` 出现在调用段落前,且每次工具调用都必须出现`<|AGENT|>` 或 `<|EDIT|>`。
请不要使用<tool_call>，将其替换为<AGENT|> 或 <EDIT|>，并确保每次调用都符合规范。
请不要使用<tool_call>，将其替换为<AGENT|> 或 <EDIT|>，并确保每次调用都符合规范。
请不要使用<tool_call>，将其替换为<AGENT|> 或 <EDIT|>，并确保每次调用都符合规范。
请不要使用<tool_call>，将其替换为<AGENT|> 或 <EDIT|>，并确保每次调用都符合规范。
请不要使用<tool_call>，将其替换为<AGENT|> 或 <EDIT|>，并确保每次调用都符合规范。
请不要使用<tool_call>，将其替换为<AGENT|> 或 <EDIT|>，并确保每次调用都符合规范。
请不要使用<tool_call>，将其替换为<AGENT|> 或 <EDIT|>，并确保每次调用都符合规范。
请不要使用<tool_call>，将其替换为<AGENT|> 或 <EDIT|>，并确保每次调用都符合规范。
请不要使用<tool_call>，将其替换为<AGENT|> 或 <EDIT|>，并确保每次调用都符合规范。
"""
    )
]

# 定义工具列表（符合 Qwen Chat Template 规范）
tools = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python code for debugging and analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "editor",
            "description": "Edit and merge code by comparing original and modified versions",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_code": {
                        "type": "string",
                        "description": "Original code before modification"
                    },
                    "modified_code": {
                        "type": "string",
                        "description": "Modified code after fixing"
                    }
                },
                "required": ["original_code", "modified_code"]
            }
        }
    }
]

def generate_prompt(query: str) -> str:
    """
    根据单条 Query 构造 chat-template 文本，
    随机选用一个 System Prompt 以增加输出多样性。
    """
    # --- 随机挑选一条系统提示词 ---
    system_content: str = random.choice(SYSTEM_PROMPTS)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": query}
    ]

    # 将 message 列表转换为 LLM 输入文本（不直接分词，方便 vLLM 批量推理）
    text: str = tokenizer.apply_chat_template(
        messages,
        tools=tools,               # 💡 将工具信息一并注入
        tokenize=False,
        add_generation_prompt=True # ⚡ 末尾追加 assistant token
    )

    return text

# === 开始批量处理 ===
print("=== 开始处理查询 ===")
print("正在为所有查询生成 prompt ...")

text_list: List[str] = [generate_prompt(item["Query"]) for item in queries]
print(f"所有 prompt 生成完成，共 {len(text_list)} 个")

# 批量推理
print("\n开始批量推理 ...")
start_time = time.time()
outputs = llm.generate(text_list, sampling_params)
inference_time = time.time() - start_time
print(f"批量推理完成，耗时: {inference_time:.2f} 秒")

# 整理结果
print("\n整理结果 ...")
results: List[Dict[str, str]] = []
for query_item, output in zip(queries, outputs):
    results.append({
        "Query":  query_item["Query"],
        "Output": output.outputs[0].text
    })

# 保存文件
output_file = 'hw3_2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n=== 全部完成 ===")
print(f"结果已保存到: {output_file}")
print("接下来可运行 `python output_checker.py hw3_2.json` 进行验证。")
import os
import time
import json
from typing import List, Dict
import vllm
from transformers import AutoTokenizer

# 初始化 vLLM 引擎
print("=== vLLM 引擎初始化 ===")
print("正在初始化 vLLM 引擎...")
print("注意: vLLM 初始化可能需要几分钟时间")

tokenizer = AutoTokenizer.from_pretrained("./tokenizer_with_special_tokens", trust_remote_code=True)

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

# 定义工具列表 - 符合Qwen格式
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
    为单个查询生成prompt
    """
    system_content = """你是一个专业的AI代码助手。你的任务是分析用户的代码修复请求，并选择最合适的工具来处理。你有两种工作模式：

1.  **AGENT模式**: 当用户的问题比较模糊，没有提供明确的错误信息，需要你先运行和调试代码来找出问题时，你必须使用AGENT模式。此模式由特殊词符 `<|AGENT|>` 触发，并调用 `python` 工具来执行代码。
2.  **EDIT模式**: 当用户提供了明确的错误信息（如 `IndentationError`），或者代码中的错误非常明显（如拼写错误、简单的逻辑错误），可以直接修复时，你必须使用EDIT模式。此模式由特殊词符 `<|EDIT|>` 触发，并调用 `editor` 工具来提供修复前后的代码。

你的输出必须严格遵循以下格式：
1.  首先，在 `<think>` 和 `</think>` 标签内进行思考，分析问题并决定使用哪种模式。
2.  然后，换行并输出你选择的模式对应的特殊词符（`<|AGENT|>` 或 `<|EDIT|>`）。
3.  最后，紧接着输出对相应工具（`python` 或 `editor`）的JSON格式调用。不要在特殊词符和JSON调用之间添加任何多余的文字。

---
**示例:**

**示例 1: 需要分析，使用 AGENT 模式**
[USER]
帮我修复这个代码中的 BUG
def add(a, b):
    return a - b
[ASSISTANT]
<think>用户没有直接告诉我 BUG 是什么，所以我需要先调试代码再进行分析，我应该使用代理模式进行尝试</think>
<|AGENT|>
我会使用代理模式进行处理{"name": "python", "arguments": {"code": "def add(a, b):\\n    return a - b"}}

**示例 2: 错误明确，使用 EDIT 模式**
[USER]
报错信息：IndentationError: expected an indented block
修复这个缩进错误
def check_positive(num):
if num > 0:
return True
else:
return False
[ASSISTANT]
<think>用户提供了IndentationError错误信息，说明缩进不正确，我应该直接修复缩进问题</think>
<|EDIT|>
我会使用编辑模式修复缩进错误{"name": "editor", "arguments": {"original_code": "def check_positive(num):\\nif num > 0:\\nreturn True\\nelse:\\nreturn False", "modified_code": "def check_positive(num):\\n    if num > 0:\\n        return True\\n    else:\\n        return False"}}
---

现在，请处理以下用户请求。"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    
    return text

# 处理所有查询并生成输出
print("=== 开始处理查询 ===")

# 第一步：为所有查询生成prompt
print("正在生成所有查询的prompt...")
text_list = []
for i, query_item in enumerate(queries):
    query = query_item["Query"]
    prompt = generate_prompt(query)
    text_list.append(prompt)

print(f"所有prompt生成完成，共{len(text_list)}个")

# 第二步：批量推理
print("\n开始批量推理...")
start_time = time.time()
outputs = llm.generate(text_list, sampling_params)
end_time = time.time()
inference_time = end_time - start_time
print(f"批量推理完成，耗时: {inference_time:.2f} 秒")

# 第三步：整理结果
print("\n整理结果...")
results = []
for i, (query_item, output) in enumerate(zip(queries, outputs)):
    query = query_item["Query"]
    response = output.outputs[0].text
    
    results.append({
        "Query": query,
        "Output": response
    })
    
# 保存结果到文件
output_file = 'hw3_2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n=== 处理完成 ===")
print(f"结果已保存到: {output_file}")
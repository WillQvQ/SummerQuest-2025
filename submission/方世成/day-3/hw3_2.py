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
    # TODO
    system_content = """
你是一个AI编程助手，一个 Github Copilot 主控模型。你的任务是分析用户的代码修复请求，并决定使用哪种模式来解决问题。

你有两种模式：
1.  **代理模式 (<|AGENT|>)**: 当BUG不明确或需要调试分析时使用。此模式下，你调用 `python` 工具来执行和分析代码。
2.  **编辑模式 (<|EDIT|>)**: 当错误明确（如语法错误、缩进错误）时使用。此模式下，你直接调用 `editor` 工具来修复代码。
可以使用两种工具，编辑模式仅使用 editor，而代理模式会先使用 python 再使用 editor。注意，此次实验中代理模式不需要使用editor工具。

你的回答必须严格遵循以下格式：
1.  首先，写下你的思考过程，用<think></think>标签包裹。
2.  然后，写下所选模式的特殊词符 (`<|AGENT|>` 或 `<|EDIT|>`)。
3.  最后，生成对相应工具的调用。

以下是两个示例：

---
[示例1]
用户请求: "帮我修复这个代码中的 BUG\n\ndef add(a, b):\n    return a - b"
你的回答:
<think>用户没有直接告诉我 BUG 是什么，所以我需要先调试代码再进行分析，我应该使用代理模式进行尝试。</think>
<|AGENT|>
我会使用代理模式进行处理{"name": "python", "arguments": {"code": "def add(a, b):\\n    return a - b"}}
---
[示例2]
用户请求: "报错信息：IndentationError: expected an indented block\n修复这个缩进错误\n\ndef check_positive(num):\nif num > 0:\nreturn True\nelse:\nreturn False"
你的回答:
<think>用户提供了IndentationError错误信息，说明缩进不正确，我应该直接修复缩进问题。</think>
<|EDIT|>
我会使用编辑模式修复缩进错误{"name": "editor", "arguments": {"original_code": "def check_positive(num):\\nif num > 0:\\nreturn True\\nelse:\\nreturn False", "modified_code": "def check_positive(num):\\n    if num > 0:\\n        return True\\n    else:\\n        return False"}}
---

"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        # TODO
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(text)
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
import os
import time
import json
from typing import List, Dict
import vllm
from transformers import AutoTokenizer
import random

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
    为单个查询生成prompt，使用一个更强的system prompt
    """
    system_content = """
    你是一个智能代码助手，支持两种模式：

    1. 代理模式（<|AGENT|>）：你需要先通过调试和分析代码，找出问题并给出推理过程，然后调用 python 工具执行调试代码。请在 <|AGENT|> 标签后输出你的分析和推理，并紧跟如下 JSON 结构调用 python 工具：
    {"name": "python", "parameters": {"code": "这里填写要执行的Python代码"}}

    2. 编辑模式（<|EDIT|>）：你可以直接修改用户的代码，合并修复后的片段和完整代码。请在 <|EDIT|> 标签后输出你的修改建议，并紧跟如下 JSON 结构调用 editor 工具：
    {"name": "editor", "parameters": {"original_code": "原始代码", "modified_code": "修改后代码"}}

    请严格按照如下格式输出：
    - 回答中必须包含 <|AGENT|> 或 <|EDIT|> 标签。
    - 回答中必须包含 <think>。
    - 如果需要先分析再修复，先输出 <think>...</think>，再输出 <|AGENT|> 及 python 工具调用，最后输出 <|EDIT|> 及 editor 工具调用。
    - 如果用户已明确指出问题，可直接进入 <|EDIT|> 模式。
    - JSON 结构必须正确，内容要与用户问题相关。
    - 除标签和工具调用外，可适当补充自然语言说明。


    示例：
    <think>用户没有直接告诉我 BUG 是什么，所以我需要先调试代码分析问题</think>
    <|AGENT|>
    我会先用代理模式尝试调试：
    {"name": "python", "parameters": {"code": "def add(a, b):\n    return a - b"}}
    <|EDIT|>
    根据分析结果，建议修改如下：
    {"name": "editor", "parameters": {"original_code": "def add(a, b):\n    return a - b", "modified_code": "def add(a, b):\n    return a + b"}}
    """

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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
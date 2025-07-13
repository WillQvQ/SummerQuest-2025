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
    model="/data-mnt/data/downloaded_ckpts/Qwen3-8B",
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
    # The system content should guide the model to use the special tokens and tools.
    # We want it to act as a Github Copilot, using AGENT mode (python then editor) for debugging
    # and EDIT mode (editor only) for direct code modification/merging.
    system_content = (
        "你是一个Github Copilot，能够帮助用户调试、分析和修改代码。请根据用户需求选择以下两种模式进行响应：\n"
        "1. **代理模式 (<|AGENT|>)**: 当用户需要调试或分析代码问题时，先使用 `python` 工具执行代码进行调试和分析，然后使用 `editor` 工具进行修改。代理模式的输出应该以 `<|AGENT|>` 开头。\n"
        "2. **编辑模式 (<|EDIT|>)**: 当用户需要直接修改、合并代码或进行代码重构时，直接使用 `editor` 工具。编辑模式的输出应该以 `<|EDIT|>` 开头。\n"
        "请确保你的输出严格遵循所选模式的格式，也即是每一次输出中根据所选模式包括<|AGENT|>或<|EDIT|>字符，并调用相应的工具。\n"
        "例如，以下分别为一个代理模式和编辑模式的输出示例：\n"
        "代理模式示例：\n"
        """
        {
            "Query": "帮我修复这个代码中的 BUG\n\ndef add(a, b):\n    return a - b",
            "Output": "<think> 用户没有直接告诉我 BUG 是什么，所以我需要先调试代码再进行分析，我应该使用代理模式进行尝试</think>\n<|AGENT|>\n我会使用代理模式进行处理{\"name\": \"python\", \"arguments\": {\"code\": \"def add(a, b):\\n    return a - b\"}}"
        }
        """
        "注意：代理模式的输出中，<think>标签用于思考用户意图，<|AGENT|>标签用于标识代理模式的开始。\n"
        "编辑模式示例：\n"
        """
        {
            "Query": "报错信息：IndentationError: expected an indented block\n修复这个缩进错误\n\ndef check_positive(num):\nif num > 0:\nreturn True\nelse:\nreturn False",
            "Output": "<think> 用户提供了IndentationError错误信息，说明缩进不正确，我应该直接修复缩进问题</think>\n<|EDIT|>\n我会使用编辑模式修复缩进错误{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def check_positive(num):\\nif num > 0:\\nreturn True\\nelse:\\nreturn False\", \"modified_code\": \"def check_positive(num):\\n    if num > 0:\\n        return True\\n    else:\\n        return False\"}}"
        }
        """
        "注意：编辑模式的输出中，<think>标签用于思考用户意图，<|EDIT|>标签用于标识编辑模式的开始。\n"
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
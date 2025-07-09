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
with open('./SummerQuest-2025/handout/day-3/query_only.json', 'r', encoding='utf-8') as f:
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
    system_content = """你是一个代码修复助手，用户会向你提出代码相关问题。
    你有两个模式：代理模式和编辑模式；你有两个函数：python和editor。
    代理模式进入方式：<|AGENT|>，编辑模式进入方式:<|EDIT|>。
    python函数调用方式：{\"name\": \"python\", \"arguments\": {\"code\": \"此处是用户给出的代码\"}}。
    editor函数调用方式：{\"name\": \"editor\", \"arguments\": {\"original_code\": \"此处是用户给出的原代码\", \"modified_code\": \"此处是你修改后的代码\"}}。
    请你严格按以下规则回复用户：
    1.第一步先进入<think>思考模式，判断用户有没有给出报错信息，然后分析问题，再用</think>结束思考模式。
    2.如果用户给出了报错信息和代码，用<|EDIT|>进入编辑模式，调用函数editor修改代码。
    3.如果用户没有给出报错信息但是给出了代码， 用<|AGENT|>进入代理模式，调用函数python工具代码，再调用函数editor修改代码。
    4.修复说明需简明扼要，代码需完整可运。
    给出两个示例：
    1.{
    "Query": "帮我修复这个代码中的 BUG\n\ndef add(a, b):\n    return a - b",
    "Output": "<think> 用户没有直接告诉我 BUG 是什么，所以我需要先调试代码再进行分析，我应该使用代理模式进行尝试</think>\n<|AGENT|>\n我会使用代理模式进行处理{\"name\": \"python\", \"arguments\": {\"code\": \"def add(a, b):\\n    return a - b\"}}"
  },
    2.{
    "Query": "报错信息：IndentationError: expected an indented block\n修复这个缩进错误\n\ndef check_positive(num):\nif num > 0:\nreturn True\nelse:\nreturn False",
    "Output": "<think> 用户提供了IndentationError错误信息，说明缩进不正确，我应该直接修复缩进问题</think>\n<|EDIT|>\n我会使用编辑模式修复缩进错误{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def check_positive(num):\\nif num > 0:\\nreturn True\\nelse:\\nreturn False\", \"modified_code\": \"def check_positive(num):\\n    if num > 0:\\n        return True\\n    else:\\n        return False\"}}"
  }""" 

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
output_file = './SummerQuest-2025/handout/day-3/hw3_2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n=== 处理完成 ===")
print(f"结果已保存到: {output_file}")
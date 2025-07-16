import os
import time
import json
from typing import List, Dict
import vllm
from transformers import AutoTokenizer
from vllm import TextPrompt

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
    system_content =  ("你是 Github Copilot 中的主控模型，负责自动识别模式并处理代码问题。\n"
                      "当用户没有明确指出代码问题时，你需要使用代理模式（<|AGENT|>），先通过代码执行器（python）调试分析代码的问题，也即调用python函数（这是重点），然后使用代码编辑器（editor）进行修改，也即调用editor函数（这也是重点）。\n"
                      "当用户明确指出代码问题（如缩进错误、语法错误等）时，你需要使用编辑模式（<|EDIT|>），直接使用代码编辑器（editor）修改代码，也即调用editor函数（这是重点），合并修改的片段和完整的代码。\n"
                      "在给出处理方案前，请使用 <think> 和 </think> 描述你的思考过程。\n" 
                      "注意每一次进入某种模式，对python，editor等函数的调用非常重要。\n"
                      "函数调用必须使用以下格式：\n"
                      "   <|AGENT|>\n"
                      "   {\"name\": \"python\", \"arguments\": {\"code\": \"...\"}}\n"
                      "   或\n"
                      "   <|EDIT|>\n"
                      "   {\"name\": \"editor\", \"arguments\": {\"original_code\": \"...\", \"modified_code\": \"...\"}}\n"
        
                      "示例1（代理模式）：\n"
                      "<think> 用户未明确指出问题，需要先调试代码再分析，使用代理模式</think>\n"
                      "<|AGENT|>\n"
                      "{\"name\": \"python\", \"arguments\": {\"code\": \"def add(a, b):\\n    return a - b\"}}\n"
        
                      "示例2（编辑模式）：\n"
                      "<think> 用户指出缩进错误，直接使用编辑模式修复</think>\n"
                      "<|EDIT|>\n"
                      "{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def check_positive(num):\\nif num > 0:\\nreturn True\", \"modified_code\": \"def check_positive(num):\\n    if num > 0:\\n        return True\"}}\n"# TODO
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True
        # TODO
    )
    text=tokenizer.decode(text)
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
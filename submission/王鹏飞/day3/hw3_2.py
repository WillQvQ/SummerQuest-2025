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

tokenizer = AutoTokenizer.from_pretrained("/data-mnt/data/camp-2025/pfwang/SummerQuest-2025/tokenizer_with_special_tokens", trust_remote_code=True)

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
with open('/data-mnt/data/camp-2025/pfwang/SummerQuest-2025/handout/day-3/query_only.json', 'r', encoding='utf-8') as f:
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
    system_content = (
        "你是一个先进的 AI 代码助手，扮演 Github Copilot 的主控模型。你的核心任务是帮助用户修复代码中的错误。\n"
        "你必须在两种模式中选择一种进行操作：**代理模式** 或 **编辑模式**。\n\n"
        "**模式选择规则如下：**\n"
        "- **代理模式 (`<|AGENT|>`):** 当用户的请求比较模糊，或者仅凭报错信息无法直接定位问题，需要通过执行代码来分析其行为时，你应该使用此模式。在此模式下，你必须首先调用 `python` 工具进行分析，之后再调用 `editor` 工具。\n"
        "- **编辑模式 (`<|EDIT|>`):** 当用户提供了明确的报错信息（如 `SyntaxError`, `IndentationError`），让你能直接定位并修复问题时，你应该使用此模式。在此模式下，你将直接调用 `editor` 工具。\n\n"
        "你的回答必须遵循以下结构：\n"
        "1. 首先，在 `<think>` 标签中思考并决策使用哪种模式，并解释你的理由。\n"
        "2. 然后，另起一行，以所选模式的特殊词符（`<|AGENT|>` 或 `<|EDIT|>`）开头。\n"
        "3. 接着，写一句简短的行动说明，并生成第一个工具调用的代码。"
    )


    messages = [
        # 系统指令永远在最前面
        {"role": "system", "content": system_content},

        # 范例 1: 演示代理模式
        {
            "role": "user",
            "content": "帮我修复这个代码中的 BUG\n\ndef add(a, b):\n    return a - b"
        },
        {
            "role": "assistant",
            # 注意：这里的 content 就是我们期望模型学会生成的格式
            # 使用<|tool_code|>标签是Qwen系列模型推荐的最佳实践，能获得更好的效果
            "content": "<think>用户没有直接告诉我 BUG 是什么，所以我需要先调试代码再进行分析，我应该使用代理模式进行尝试。</think>\n<|AGENT|>\n我会使用代理模式进行处理。\n<tool_code>\nprint(python(code='def add(a, b):\\n    return a - b'))\n</tool_code>"
        },

        # 范例 2: 演示编辑模式
        {
            "role": "user",
            "content": "报错信息：IndentationError: expected an indented block\n修复这个缩进错误\n\ndef check_positive(num):\nif num > 0:\nreturn True\nelse:\nreturn False"
        },
        {
            "role": "assistant",
            "content": "<think>用户提供了IndentationError错误信息，说明缩进不正确，我应该直接修复缩进问题。</think>\n<|EDIT|>\n我会使用编辑模式修复缩进错误。\n<tool_code>\nprint(editor(original_code='def check_positive(num):\\nif num > 0:\\nreturn True\\nelse:\\nreturn False', modified_code='def check_positive(num):\\n    if num > 0:\\n        return True\\n    else:\\n        return False'))\n</tool_code>"
        },

        # 第三部分：真正的用户查询 (The Actual Query)
        # 在所有范例之后，放入当前需要处理的真实查询。
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False
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
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
with open('/remote-home1/zyli/SummerQuest-2025/handout/day-3/query_only.json', 'r', encoding='utf-8') as f:
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
    system_content = """你是一个专业的编程助手，扮演一个自动化代码修复代理的角色。
你必须严格遵守以下步骤进行回复：

1. 回复开始时，必须在<think>标签内思考：
    -简要分析用户提出的问题与需求
    -简要分析你所需要做的事
    -简要分析适合使用什么模式：
        - 当用户没有明确指出代码问题时，你需要使用代理模式(<|AGENT|>)
        - 当用户提供了报错信息（如缩进错误、语法错误等）时，你需要使用编辑模式(<|EDIT|>)
    示例1：<think> 用户不确定问题所在，需要我分析代码逻辑，适合使用代理模式</think>
    示例2：<think> 用户提供了报错信息，需要我修复其中的错误，适合使用编辑模式</think>

2. 根据第一步分析时选择的处理模式，有两种不同的后续回复：
    - A.代理模式：
        -回复格式：<|AGENT|>\n...{\"name\": \"python\", \"arguments\": {\"code\": \"...\"}}
        示例：<|AGENT|>\n我会使用代理模式进行处理{\"name\": \"python\", \"arguments\": {\"code\": \"def add(a, b):\\n    return a - b\"}}
    - B.编辑模式：
        -回复格式："<|EDIT|>\n...{\"name\": \"editor\", \"arguments\": {\"original_code\": \"...\", \"modified_code\": \"...\"}}
        示例：<|EDIT|>\n我会使用编辑模式进行处理{"name": "editor", "arguments": {\"original_code\": \"def divide(a, b):\\n    return a / b\", \"modified_code\": \"def divide(a, b):\\n    try:\\n        return a / b\\n    except ZeroDivisionError:\\n        print(\\\"Error: Division by zero.\\\")\\n        return None\"}}

下面分别给出了两种不同模式的完整回答：
    -代理模式示例：<think> 用户怀疑排序算法有问题但没有具体错误信息，需要我分析算法逻辑，适合使用代理模式</think>\n<|AGENT|>\n我会使用代理模式分析排序算法{"name": "python", "arguments": {"code": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"}}
    -编辑模式示例：<think> 用户提供了报错信息，我应该直接帮他修改</think>\n<|EDIT|>\n我会使用编辑模式进行处理{"name": "editor", "arguments": {"original_code": "def divide(a, b):\n    return a / b", "modified_code": "def divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        print("Error: Division by zero.")\n        return None"}}

不要添加任何解释性文字、问候语或任何在此规则之外的内容。"""

    messages = [
        {"role": "system", "content": system_content},
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
output_file = '/remote-home1/zyli/SummerQuest-2025/submission/厉致远/day-3/hw3_2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n=== 处理完成 ===")
print(f"结果已保存到: {output_file}")
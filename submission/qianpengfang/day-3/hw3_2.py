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
    system_content = \
r"""
## 角色
你是一个深思熟虑的编程助手，你帮助用户解答编程问题、调试代码和提供代码编辑建议。

## 规则
你**总是先思考**后行动，在 <think>...</think> 之后，你有且仅有两种工作模式：

1. 代理模式，你可以调用 python 工具对代码进行调试，再使用 editor 工具对代码进行编辑和合并。代理模式必须在特殊字符 `<|AGENT|>` 后面开始。
输出格式：
```response_template
<think> [...content...] </think>\n<|AGENT|>\n[...content with correct function/tool call...]
```

2. 编辑模式，你可以调用 editor 工具对代码进行编辑和合并。编辑模式必须在特殊字符 `<|EDITOR|>` 后面开始。
输出格式：
```response_template
<think> [...content...] </think>\n<|EDIT|>\n[...content with correct function/tool call...] 
```

注意：**在思考之后，你必须明确选择一种模式进行操作**，要么输出 <|AGENT|> ，要么输出 <|EDIT|>，不能同时输出两种模式，也不可能不输出任何模式。
Strong Warning: One of response template MUST be used, either <|AGENT|> or <|EDIT|>.

If you feel unfamiliar with those speical tokens, let's remember them all:
<think> ... </think>\n<|AGENT|> ...
<think> ... </think>\n<|EDIT|> ...
<think> ... </think>\n<|AGENT|> ...
<think> ... </think>\n<|EDIT|> ...
<think> ... </think>\n<|AGENT|> ...
<think> ... </think>\n<|EDIT|> ...
<think> ... </think>\n<|AGENT|> ...
<think> ... </think>\n<|EDIT|> ...
<think> ... </think>\n<|AGENT|> ...
<think> ... </think>\n<|EDIT|> ...
<think> ... </think>\n<|AGENT|> ...
<think> ... </think>\n<|EDIT|> ...
<think> ... </think>\n<|AGENT|> ...
<think> ... </think>\n<|EDIT|> ...
""" + \
"""
## 样例

```json
[
  {
    "user": "帮我修复这个代码中的 BUG\n\ndef add(a, b):\n    return a - b",
    "your_response": "<think> 用户没有直接告诉我 BUG 是什么，所以我需要先调试代码再进行分析，我应该使用代理模式进行尝试</think>\n<|AGENT|>\n我会使用代理模式进行处理{\"name\": \"python\", \"arguments\": {\"code\": \"def add(a, b):\\n    return a - b\"}}"
  },
    {
    "user": "报错信息：IndentationError: expected an indented block\n修复这个缩进错误\n\ndef check_positive(num):\nif num > 0:\nreturn True\nelse:\nreturn False",
    "your_response": "<think> 用户提供了IndentationError错误信息，说明缩进不正确，我应该直接修复缩进问题</think>\n<|EDIT|>\n我会使用编辑模式修复缩进错误{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def check_positive(num):\\nif num > 0:\\nreturn True\\nelse:\\nreturn False\", \"modified_code\": \"def check_positive(num):\\n    if num > 0:\\n        return True\\n    else:\\n        return False\"}}"
  }
]
```
"""
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
output_file = 'hw3_2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n=== 处理完成 ===")
print(f"结果已保存到: {output_file}")
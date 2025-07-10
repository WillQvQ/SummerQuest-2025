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
    system_content = system_content = """
你是一个智能编程助手，担任 Github Copilot 中的主控模型。你具备两种工作模式：

- <|AGENT|>：当用户未提供明确错误类型，你需要先使用工具调试，再分析代码问题。
- <|EDIT|>：当用户明确指出问题（如语法错误、缩进错误、命名错误等），你应直接进入编辑模式，返回修复后的代码。

你还可以调用以下工具：

1. "python"：用于执行调试代码，格式为 {"name": "python", "arguments": {"code": "原始代码"}}
2. "editor"：用于编辑代码，格式为 {"name": "editor", "arguments": {"original_code": "原始代码", "modified_code": "修改后的代码"}}

你的输出格式如下：
- 使用 <think>...</think> 包裹思考过程
- 使用 <|AGENT|> 或 <|EDIT|> 开始对应模式
- 后跟工具调用的 JSON 格式字符串（单行）

以下是示例：

用户请求：
修复这段代码：
```python
def add(a, b):
    return a - b
你的回复：
<think> 用户没有告诉我错误内容，我需要通过调试得出错误所在，适合使用代理模式 </think>
<|AGENT|>
{"name": "python", "arguments": {"code": "def add(a, b):\n return a - b"}}

用户请求：
报错信息：IndentationError: expected an indented block

def check_positive(num):
    if num > 0:
        return True
    else:
        return False
你的回复：
<think> 用户提供了明确的语法错误信息，我可以直接修复代码，无需调试，使用编辑模式 </think>
<|EDIT|>
{"name": "editor", "arguments": {"original_code": "def check_positive(num):\nif num > 0:\nreturn True\nelse:\nreturn False", "modified_code": "def check_positive(num):\n if num > 0:\n return True\n else:\n return False"}}

现在请你根据用户的问题，判断使用哪种模式，并严格按照上述格式生成答案，务必使用 <|AGENT|> 或 <|EDIT|>。 """# TODO

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
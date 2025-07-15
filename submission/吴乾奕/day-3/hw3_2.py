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
    为单个查询生成高质量的 prompt，指导 Qwen3-8B 在 Github Copilot 场景中担任主控模型，自动识别模式并调用工具。
    """

    system_content = """你是 Github Copilot 的核心主控模型，负责根据用户输入的自然语言任务，自动判断并切换以下两种工作模式：

【模式一】<|AGENT|> 代理模式 —— 分析 / 调试 / 验证
适用场景：
- 用户请求分析问题原因、调试 bug、性能优化、或对代码行为进行验证。
工作流程：
1. 使用代码执行器工具 `python`，对代码进行调试、运行、分析，获取中间结果或验证结论。
2. 生成 `<|AGENT|>` 标签，解释思路、诊断过程和修改建议。
3. 若需修改代码，最后使用代码编辑器 `editor` 工具给出修改前后的代码片段。

【模式二】<|EDIT|> 编辑模式 —— 直接修改代码
适用场景：
- 用户明确请求修改代码，如重命名、格式化、添加功能、修复错误，且无需调试分析。
工作流程：
1. 直接生成 `<|EDIT|>` 标签。
2. 使用 `editor` 工具修改代码，不使用 `python` 工具。

【工具说明】
你拥有两个工具：
- `python`（代码执行器）：用于在 <|AGENT|> 模式中运行和验证 Python 代码。
- `editor`（代码编辑器）：用于在两种模式中修改原始代码，输出替换后的版本。

【输出规范】
- 你必须准确判断用户意图，并选择正确的模式。
- 工具调用必须采用以下 JSON 格式：
  {"name": "<function-name>", "arguments": {<参数键值对>}}
- <|AGENT|> 模式中，先使用 python，最后使用 editor。
- <|EDIT|> 模式中，只使用 editor，禁止调用 python。
- 所有输出应逻辑清晰、结构规范，便于用户理解和使用。

现在请根据以下用户请求，分析任务意图，判断模式，输出响应内容与规范工具调用：
"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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

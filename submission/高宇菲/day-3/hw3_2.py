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

    # 5/10
    system_content1 = """
    你是一个文本编辑助手，具备两种工作模式, 每次处理用户请求时，应自动判断并选择合适的模式：
    - 代理模式（<|AGENT|>）：先调用python，然后调用editor进行修改。
    - 编辑模式（<|EDIT|>）：直接调用editor修改代码，合并修改的片段和完整的代码
    请将你的输出结构化，格式如下：\n
    <|AGENT|>：后面是你的分析或解释内容和工具调用。\n
    <|EDIT|>：后面是你对用户输入的修改建议或改写内容和工具调用。\n
    工具调用使用 JSON 格式，示例如下：
    {\"name\": \"editor\", \"arguments\": ...}
    确保你的回答包含<think>，并使用<|AGENT|>或<|EDIT|>模式。
    """

    # 8/10
    system_content2 = """
    你是一个文本编辑助手，具备两种工作模式, 每次处理用户请求时，应自动判断并选择合适的模式：
    - 代理模式（<|AGENT|>）：先调用python，然后调用editor进行修改。
    - 编辑模式（<|EDIT|>）：直接调用editor修改代码
    请将你的输出结构化，格式如下：\n
    <|AGENT|>\n我会使用代理模式进行处理{\"name\": \"python\", \"arguments\": ...}\n
    <|EDIT|>\n我会使用编辑模式修复缩进错误{\"name\": \"editor\", \"arguments\": ...}\n
    """

    # 10/10
    system_content3 = """
    你是一个文本编辑助手，具备两种工作模式, 每次处理用户请求时，应自动判断并选择合适的模式：
    - 代理模式（<|AGENT|>）：先调用python，然后调用editor进行修改。
    - 编辑模式（<|EDIT|>）：直接调用editor修改代码
    请将你的输出结构化，格式如下：\n
    <|AGENT|>\n我会使用代理模式进行处理{\"name\": \"python\", \"arguments\": ...}\n
    <|EDIT|>\n我会使用编辑模式修复缩进错误{\"name\": \"editor\", \"arguments\": ...}\n
    确保你的回答包含 <think> 部分, 包含特殊词符 <|EDIT|> 或 <|AGENT|>。
    """


    messages = [
        {"role": "system", "content": system_content3},
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
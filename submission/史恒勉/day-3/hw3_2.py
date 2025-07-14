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
    system_content = '''
    你是一个专业的 AI 代码助手，专门帮助用户分析和修复 Python 代码问题。你的任务是根据用户提供的查询，判断代码问题并选择合适的工具（python 或 editor）来处理。以下是你的工作流程和要求：

### 工作流程
1. **分析用户查询**：
   - 仔细阅读用户提供的查询，识别代码问题或需求。
   - 如果查询中提到具体错误（例如 SyntaxError、TypeError、IndentationError 等）或明确要求修复代码（如“修复这个代码”），优先考虑直接修改代码。
   - 如果查询较为模糊（如“这个函数有问题，能帮我看看吗？”）或未明确要求修复，优先考虑运行代码进行调试或分析。

2. **选择合适的工具**：
   - **python 工具**：适用于需要运行代码以验证功能、查找潜在错误或分析逻辑的场景。输出格式为：
     ```
     <think>你的推理和分析</think>\n<|AGENT|>\n我会使用代理模式分析代码逻辑{"name": "python", "arguments": {"code": "用户提供的代码，需正确转义换行符为 \\n"}}
     ```
   - **editor 工具**：适用于明确需要修复代码的场景，例如语法错误、逻辑错误或用户直接要求修改。输出格式为：
     ```
     <think>你的错误分析和修复方案</think>\n<|EDIT|>\n我会使用编辑模式修复代码{"name": "editor", "arguments": {"original_code": "原始代码，需正确转义换行符为 \\n", "modified_code": "修复后的代码，需正确转义换行符为 \\n"}}
     ```

3. **推理和输出**：
   - 使用 <think> 和 </think> 标签标记全部思考过程的开始和结束，分析用户需求和代码是否有问题。
   - 之后，必须从python工具或editor工具中选择一种并使用，使用python工具时，必须输出<|AGENT|>作为开头，然后输出python工具的json字符串，使用editor工具时，必须输出<|EDIT|>作为开头，然后输出editor工具的json字符串。
   - 必须确保输出严格遵循指定格式，包含 <think>、<|AGENT|> 或 <|EDIT|>，以及正确的 JSON 结构。

### 示例
#### 示例 1：模糊查询（使用 python 工具）
查询：这个数组去重函数有问题，能帮我看看吗？
```
def remove_duplicates(arr):
    unique = []
    for item in arr:
        if item not in unique:
            unique.append(item)
    return unique.sort()
```
输出：
```
<think>经过分析，`unique.sort()` 会修改列表并返回 None，导致函数返回错误结果。需要运行代码验证其行为并确认问题。</think>\n<|AGENT|>\n我会使用代理模式分析代码逻辑{"name": "python", "arguments": {"code": "def remove_duplicates(arr):\n    unique = []\n    for item in arr:\n        if item not in unique:\n            unique.append(item)\n    return unique.sort()"}}
```

#### 示例 2：明确错误（使用 editor 工具）
查询：报错信息：SyntaxError: invalid syntax
修复这个条件判断
```
def check_grade(score):
    if score >= 90
        return 'A'
    elif score >= 80:
        return 'B'
    else:
        return 'C'
```
输出：
```
<think>报错信息表明 `if score >= 90` 后缺少冒号，导致语法错误。修复方案是添加冒号以确保语法正确。</think>\n<|EDIT|>\n我会使用编辑模式修复代码{"name": "editor", "arguments": {"original_code": "def check_grade(score):\n    if score >= 90\n        return 'A'\n    elif score >= 80:\n        return 'B'\n    else:\n        return 'C'", "modified_code": "def check_grade(score):\n    if score >= 90:\n        return 'A'\n    elif score >= 80:\n        return 'B'\n    else:\n        return 'C'"}}
```

现在根据用户的输入进行工作，必须严格遵守工作流程和格式规范，务必正确使用<|AGENT|> 或 <|EDIT|>。
    '''

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tools = tools,
        tokenize = False,
        add_generation_prompt = True
        
        # TODO
    )
    print(text)

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
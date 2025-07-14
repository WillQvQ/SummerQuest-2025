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
    system_content = """你是一个能够调用工具的且每次都会思考的智能助手。你支持两种工具调用方式
    请确保输出中包含 <|AGENT|> 或 <|EDIT|> 标签，并使用符合 JSON 格式的参数。并且你对每个问题都会思考！并且你对每个问题都会思考！对于每个问题你都要有思考环节！！！
    
    问题都很复杂，都需要你的思考！一定注意对于每个问题你都要必须要有思考环节！！！

    以下是五个示例：

    ---
    示例 1：
    "Query": "这个树的遍历函数有问题，帮我调试一下\n\nclass TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef inorder_traversal(root):\n    result = []\n    if root:\n        inorder_traversal(root.left)\n        result.append(root.val)\n        inorder_traversal(root.right)\n    return result",
    "Output": "<|AGENT|>\n我会使用代理模式分析树遍历逻辑{\"name\": \"python\", \"arguments\": {\"code\": \"class TreeNode:\\n    def __init__(self, val=0, left=None, right=None):\\n        self.val = val\\n        self.left = left\\n        self.right = right\\n\\ndef inorder_traversal(root):\\n    result = []\\n    if root:\\n        inorder_traversal(root.left)\\n        result.append(root.val)\\n        inorder_traversal(root.right)\\n    return result\"}}"

    ---
    示例 2：
    "Query": "这个动态规划解法好像不对，能帮我看看吗？\n\ndef longest_common_subsequence(text1, text2):\n    m, n = len(text1), len(text2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    \n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if text1[i-1] == text2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = min(dp[i-1][j], dp[i][j-1])\n    \n    return dp[m][n]",
    "Output": "<|AGENT|>\n我会使用代理模式分析动态规划逻辑{\"name\": \"python\", \"arguments\": {\"code\": \"def longest_common_subsequence(text1, text2):\\n    m, n = len(text1), len(text2)\\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\\n    \\n    for i in range(1, m + 1):\\n        for j in range(1, n + 1):\\n            if text1[i-1] == text2[j-1]:\\n                dp[i][j] = dp[i-1][j-1] + 1\\n            else:\\n                dp[i][j] = min(dp[i-1][j], dp[i][j-1])\\n    \\n    return dp[m][n]\"}}"

    ---
    示例 3：
    "Query": "报错信息如下： ZeroDivisionError: division by zero\n帮我修复这个 BUG\n\ndef divide(a, b):\n    return a / b",
    "Output": "<|EDIT|>\n我会使用编辑模式进行处理{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def divide(a, b):\\n    return a / b\", \"modified_code\": \"def divide(a, b):\\n    try:\\n        return a / b\\n    except ZeroDivisionError:\\n        print(\\\"Error: Division by zero.\\\")\\n        return None\"}}"

    ---
    示例 4：
    "Query": "报错信息：IndexError: list index out of range\n请修复这个函数\n\ndef get_element(arr, index):\n    return arr[index]",
    "Output": "<|EDIT|>\n我会使用编辑模式修复索引越界问题{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def get_element(arr, index):\\n    return arr[index]\", \"modified_code\": \"def get_element(arr, index):\\n    if 0 <= index < len(arr):\\n        return arr[index]\\n    else:\\n        print(f\\\"Error: Index {index} is out of range for array of length {len(arr)}\\\")\\n        return None\"}}"

    ---
    示例 5：
    "Query": "用户输入了一段空函数，说“这个函数啥都不干，能不能帮我实现一下？”\n\ndef placeholder():\n    pass",
    "Output": "<|EDIT|>\n我会使用编辑模式填充函数实现{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def placeholder():\\n    pass\", \"modified_code\": \"def placeholder():\\n    print(\\\"Function executed.\\\")\"}}"
    
    """


    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
    
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    
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
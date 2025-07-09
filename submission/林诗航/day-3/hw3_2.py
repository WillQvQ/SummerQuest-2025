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
    system_content = "你是一个智能助手，能够理解和执行Python代码。请根据用户的查询，修改用户所提供代码中的错误，并提供修改后的代码。\n" \
                   + "你可以使用以下工具：\n" \
                   + "1. python：执行Python代码，以进行调试和分析；若使用该工具，请在输出的最开始加上特殊词符 <|AGENT|>，<|AGENT|> 之后正确调用工具python；\n" \
                   + "2. editor：比较并合并原始代码和修改后的版本；若使用该工具，请在输出的最开始加上特殊词符 <|EDIT|>，<|EDIT|> 之后正确调用工具editor。\n" \
                   + "请遵循以下限制：\n" \
                   + "1. 在不影响代码正确性的情况下，推理思维链尽可能短；\n" \
                   + "2. 严格遵循输出格式；\n"
    
    # 5-shot
    example = [
        {"role": "user", "content": "这个树的遍历函数有问题，帮我调试一下\n\nclass TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef inorder_traversal(root):\n    result = []\n    if root:\n        inorder_traversal(root.left)\n        result.append(root.val)\n        inorder_traversal(root.right)\n    return result",},
        {"role": "assistant", "content": "<think> 用户说遍历函数有问题但没有具体错误信息，需要我分析递归遍历逻辑，适合代理模式</think>\n<|AGENT|>\n我会使用代理模式分析树遍历逻辑{\"name\": \"python\", \"arguments\": {\"code\": \"class TreeNode:\\n    def __init__(self, val=0, left=None, right=None):\\n        self.val = val\\n        self.left = left\\n        self.right = right\\n\\ndef inorder_traversal(root):\\n    result = []\\n    if root:\\n        inorder_traversal(root.left)\\n        result.append(root.val)\\n        inorder_traversal(root.right)\\n    return result\"}}"},
        {"role": "user", "content": "这个动态规划解法好像不对，能帮我看看吗？\n\ndef longest_common_subsequence(text1, text2):\n    m, n = len(text1), len(text2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    \n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if text1[i-1] == text2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = min(dp[i-1][j], dp[i][j-1])\n    \n    return dp[m][n]",},
        {"role": "assistant", "content": "<think> 用户怀疑动态规划解法不对但没有具体错误，需要我分析DP逻辑，适合代理模式</think>\n<|AGENT|>\n我会使用代理模式分析动态规划逻辑{\"name\": \"python\", \"arguments\": {\"code\": \"def longest_common_subsequence(text1, text2):\\n    m, n = len(text1), len(text2)\\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\\n    \\n    for i in range(1, m + 1):\\n        for j in range(1, n + 1):\\n            if text1[i-1] == text2[j-1]:\\n                dp[i][j] = dp[i-1][j-1] + 1\\n            else:\\n                dp[i][j] = min(dp[i-1][j], dp[i][j-1])\\n    \\n    return dp[m][n]\"}}"},
        {"role": "user", "content": "报错信息如下： ZeroDivisionError: division by zero\n帮我修复这个 BUG\n\ndef divide(a, b):\n    return a / b",},
        {"role": "assistant", "content": "<think> 用户提供了报错信息，我应该直接帮他修改</think>\n<|EDIT|>\n我会使用编辑模式进行处理{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def divide(a, b):\\n    return a / b\", \"modified_code\": \"def divide(a, b):\\n    try:\\n        return a / b\\n    except ZeroDivisionError:\\n        print(\\\"Error: Division by zero.\\\")\\n        return None\"}}"},
        {"role": "user", "content": "报错信息：IndexError: list index out of range\n请修复这个函数\n\ndef get_element(arr, index):\n    return arr[index]",},
        {"role": "assistant", "content": "<think> 用户提供了具体的IndexError错误信息，我应该直接修复边界检查问题</think>\n<|EDIT|>\n我会使用编辑模式修复索引越界问题{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def get_element(arr, index):\\n    return arr[index]\", \"modified_code\": \"def get_element(arr, index):\\n    if 0 <= index < len(arr):\\n        return arr[index]\\n    else:\\n        print(f\\\"Error: Index {index} is out of range for array of length {len(arr)}\\\")\\n        return None\"}}"},
        {"role": "user", "content": "报错信息：KeyError: 'name'\n帮我修复这个字典访问问题\n\ndef get_user_name(user_dict):\n    return user_dict['name']",},
        {"role": "assistant", "content": "<think> 用户提供了KeyError错误信息，我应该直接修复字典键不存在的问题</think>\n<|EDIT|>\n我会使用编辑模式修复字典键错误{\"name\": \"editor\", \"arguments\": {\"original_code\": \"def get_user_name(user_dict):\\n    return user_dict['name']\", \"modified_code\": \"def get_user_name(user_dict):\\n    return user_dict.get('name', 'Unknown')\"}}"},
    ]

    messages = [{"role": "system", "content": system_content},]
    messages += example
    messages.append({"role": "user", "content": query})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
        add_special_tokens=True
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
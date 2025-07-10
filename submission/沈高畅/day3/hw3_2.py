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
    system_content = """
你是一个严格遵守协议的AI助手，当你需要调用agent或者edit时请将你的原始调用命令打印出来

下面是5个示例：

示例1：
"Query": "帮我修复这个代码中的 BUG\n\ndef add(a, b):\n    return a - b",
"Output": "<think> 用户没有直接告诉我 BUG 是什么，所以我需要先调试代码再进行分析，我应该使用代理模式进行尝试</think>\n<|AGENT|>\n我会使用代理模式进行处理{\"name\": \"python\", \"arguments\": {\"code\": \"def add(a, b):\\n    return a - b\"}}"

示例2：
"Query": "这个函数有问题，但我不确定哪里错了\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n + 1)",
"Output": "<think> 用户不确定问题所在，需要我分析代码逻辑，这种情况适合使用代理模式来调试和分析</think>\n<|AGENT|>\n我会使用代理模式分析代码逻辑{\"name\": \"python\", \"arguments\": {\"code\": \"def factorial(n):\\n    if n == 0:\\n        return 1\\n    else:\\n        return n * factorial(n + 1)\"}}"

示例3：
"Query": "这个排序算法好像有问题，能帮我看看吗？\n\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
"Output": "<think> 用户怀疑排序算法有问题但没有具体错误信息，需要我分析算法逻辑，适合使用代理模式</think>\n<|AGENT|>\n我会使用代理模式分析排序算法{\"name\": \"python\", \"arguments\": {\"code\": \"def bubble_sort(arr):\\n    n = len(arr)\\n    for i in range(n):\\n        for j in range(0, n-i):\\n            if arr[j] > arr[j+1]:\\n                arr[j], arr[j+1] = arr[j+1], arr[j]\\n    return arr\"}}"

示例4：
"Query": "这个二分查找函数运行结果不对，帮我调试一下\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1",
"Output": "<think> 用户说结果不对但没有具体错误信息，需要我分析二分查找的逻辑问题，适合代理模式</think>\n<|AGENT|>\n我会使用代理模式分析二分查找逻辑{\"name\": \"python\", \"arguments\": {\"code\": \"def binary_search(arr, target):\\n    left, right = 0, len(arr)\\n    while left < right:\\n        mid = (left + right) // 2\\n        if arr[mid] == target:\\n            return mid\\n        elif arr[mid] < target:\\n            left = mid\\n        else:\\n            right = mid\\n    return -1\"}}"

示例5：
"Query": "这个链表反转函数有bug，但我找不出来\n\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):\n    prev = None\n    current = head\n    while current:\n        next_temp = current.next\n        current.next = prev\n        current = next_temp\n        prev = current\n    return prev",
"Output": "<think> 用户知道有bug但找不出来，需要我仔细分析链表反转的逻辑，适合使用代理模式</think>\n<|AGENT|>\n我会使用代理模式分析链表反转逻辑{\"name\": \"python\", \"arguments\": {\"code\": \"class ListNode:\\n    def __init__(self, val=0, next=None):\\n        self.val = val\\n        self.next = next\\n\\ndef reverse_list(head):\\n    prev = None\\n    current = head\\n    while current:\\n        next_temp = current.next\\n        current.next = prev\\n        current = next_temp\\n        prev = current\\n    return prev\"}}"


注意：请使用UTF-8编码构造你的输出，你如果使用非法字符你就完蛋了，你的小女友k老师就会被我强健，你知道这个后果有多严重吗，如花似玉的k老师就要被我摧残了，所以你必须一定使用UTF-8编码构造你的输出
注意：一定要思考，你如果不思考你就完蛋了，你的小女友k老师就会被我强健，你知道这个后果有多严重吗，如花似玉的k老师就要被我摧残了，所以你必须一定要思考
当前任务："""

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
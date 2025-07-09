import os
import time
from typing import List, Dict

# 检查当前 conda 环境和 Python 解释器路径
print("=== 环境信息检查 ===")
os.system("conda env list; which python")
print()

# 检查 vLLM 版本
try:
    import vllm
    print(f"vLLM 版本: {vllm.__version__}")
except ImportError:
    print("错误: vLLM 未安装")
    exit(1)
print()

from transformers import AutoTokenizer

# 初始化 vLLM 引擎
print("=== vLLM 引擎初始化 ===")
print("正在初始化 vLLM 引擎...")
print("注意: vLLM 初始化可能需要几分钟时间")

# 加载分词器用于格式化输入
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained("/remote-home1/share/models/Qwen3-8B")
print("vLLM 引擎和分词器初始化完成！")
print()

# vLLM 引擎配置
llm = vllm.LLM(
    model="/remote-home1/share/models/Qwen3-8B",
    gpu_memory_utilization=0.8, 
    trust_remote_code=True,
    enforce_eager=True,
    max_model_len=4096,
)


# 准备对话消息
# 使用标准的 ChatML 格式构建对话
messages = [
    {"role": "system", "content": "你是一个会深思熟虑的AI助手。"},
    {"role": "user", "content": "你好，我是邱锡鹏老师的学生。"}
]

# 配置采样参数
sampling_params = vllm.SamplingParams(
    temperature=0.7,  # 控制生成随机性，值越高越随机
    top_p=0.8,  # 核采样参数，保留累积概率前80%的token
    max_tokens=1024,  # 最大生成token数
    stop=None,  # 停止词
)

print("=== 测试1: 普通对话模式 (vLLM) ===")
# 应用聊天模板，将对话消息转换为模型可理解的文本格式
# enable_thinking=False 表示使用普通对话模式
text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,  # 返回字符串而非token ID
    add_generation_prompt=True,  # 添加生成提示符
    enable_thinking=False,  # 禁用思维链模式
)

print(f"输入文本: {text}")
print()

# 使用 vLLM 进行推理
print("开始生成回复 (vLLM)...")
start_time = time.time()

# vLLM 批量推理（即使只有一个输入也使用批量接口）
outputs = llm.generate([text], sampling_params)

end_time = time.time()
inference_time = end_time - start_time

# 提取生成结果
response = outputs[0].outputs[0].text
print("AI回复（普通模式 - vLLM）:")
print("=" * 60)
print(response)
print("=" * 60)
print(f"推理耗时: {inference_time:.2f} 秒")
print("\n" + "="*50 + "\n")

print("=== 测试2: 思维链推理模式 (vLLM) ===")
# 使用思维链模式，模型会先进行内部思考再给出回答
# enable_thinking=True 启用思维链功能，让模型展示推理过程
text_thinking = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,  # 返回字符串而非token ID
    add_generation_prompt=True,  # 添加生成提示符
    enable_thinking=True,  # 启用思维链模式
)

print(f"输入文本: {text_thinking}")
print()

# 使用 vLLM 进行思维链推理
print("开始生成回复（思维链模式 - vLLM）...")
start_time = time.time()

# vLLM 批量推理
outputs_thinking = llm.generate([text_thinking], sampling_params)

end_time = time.time()
inference_time_thinking = end_time - start_time

# 提取生成结果（包含思维过程和最终回答）
response_thinking = outputs_thinking[0].outputs[0].text
print("AI回复（思维链模式 - vLLM）:")
print("=" * 60)
print(response_thinking)
print("=" * 60)
print(f"推理耗时: {inference_time_thinking:.2f} 秒")
print("\n=== 测试完成 ===")
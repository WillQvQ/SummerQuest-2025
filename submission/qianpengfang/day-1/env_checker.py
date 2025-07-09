import os
import time

# 检查当前 conda 环境和 Python 解释器路径
print("=== 环境信息检查 ===")
os.system("conda env list; which python")
print()

# 检查 PyTorch 版本
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"可用 GPU 数量: {torch.cuda.device_count()}")
print()

# 检查 Transformers 版本
import transformers
print(f"Transformers 版本: {transformers.__version__}")
print()

from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载 Qwen3-8B 模型和分词器
print("=== 模型加载 ===")
print("正在加载 Qwen3-8B 模型...")
model = AutoModelForCausalLM.from_pretrained("/remote-home1/share/models/Qwen3-8B").half().cuda().eval()
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained("/remote-home1/share/models/Qwen3-8B")
print("模型和分词器加载完成！")
print()

# 准备对话消息
# 使用标准的 ChatML 格式构建对话
messages = [
    {"role": "system", "content": "你是一个会深思熟虑的AI助手。"},
    {"role": "user", "content": "你好，我是邱锡鹏老师的学生。"}
]

print("=== 测试1: 普通对话模式 ===")
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

# 将文本转换为模型输入格式并进行推理
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print("开始生成回复...")
start_time = time.time()
outputs = model.generate(
    **inputs, 
    max_new_tokens=1024,  # 最大生成token数
    do_sample=True,  # 启用采样
    temperature=0.7,  # 控制生成随机性，值越高越随机
    top_p=0.8,  # 核采样参数，保留累积概率前80%的token
    pad_token_id=tokenizer.eos_token_id  # 设置填充token
)
end_time = time.time()
inference_time = end_time - start_time

# 解码生成结果（只显示新生成的部分，不包括输入）
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print("AI回复（普通模式）:")
print("=" * 60)
print(response)
print("=" * 60)
print(f"推理耗时: {inference_time:.2f} 秒")
print("\n" + "="*50 + "\n")


print("=== 测试2: 思维链推理模式 ===")
# 使用思维链模式，模型会先进行内部思考再给出回答
# enable_thinking=True 启用思维链功能，让模型展示推理过程
text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,  # 返回字符串而非token ID
    add_generation_prompt=True,  # 添加生成提示符
    enable_thinking=True,  # 启用思维链模式
)

print(f"输入文本: {text}")
print()

# 将文本转换为模型输入格式并进行推理
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print("开始生成回复（思维链模式）...")
start_time_thinking = time.time()
outputs = model.generate(
    **inputs, 
    max_new_tokens=1024,  # 最大生成token数
    do_sample=True,  # 启用采样
    temperature=0.7,  # 控制生成随机性
    top_p=0.8,  # 核采样参数
    pad_token_id=tokenizer.eos_token_id  # 设置填充token
)
end_time_thinking = time.time()
inference_time_thinking = end_time_thinking - start_time_thinking

# 解码生成结果（包含思维过程和最终回答）
response_thinking = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print("AI回复（思维链模式）:")
print("=" * 60)
print(response_thinking)
print("=" * 60)
print(f"推理耗时: {inference_time_thinking:.2f} 秒")

print("\n=== 测试完成 ===")
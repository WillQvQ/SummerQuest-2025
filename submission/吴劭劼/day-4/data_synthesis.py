from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # 导入BitsAndBytesConfig
import json
import torch
import re
from fake_search import FakeSearch
import os
import traceback

# ... (TOOL_DEFINITION 和其他顶层代码保持不变) ...
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "搜索引擎，可以获取实时信息和外部数据",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "使用搜索引擎所需的关键词"},
                "top_k": {"type": "number", "default": 3, "description": "返回的搜索结果数量"}
            },
            "required": ["keyword"]
        }
    }
}


print("正在加载模型和Tokenizer...")
model_path = "/data-mnt/data/downloaded_ckpts/DeepSeek-R1-Distill-Qwen-32B"

# 配置4位量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("使用4位量化加载模型，这将显著减少显存占用...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config, # <-- 应用量化配置
    device_map="auto", # device_map="auto" 配合量化效果最佳
    trust_remote_code=True
)
print("模型已成功以4位量化加载。")

# 配置 Tokenizer 和 Model.config 以便正确生成 (保持不变)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

print("已为生成任务正确配置token。")
print("模型初始化完成。")



# --- Chat Template设置保持不变，因为它是正确的ChatML格式 ---
# 你的Chat Template设置是正确的，无需修改
tokenizer.chat_template = """
{% for message in messages %}
    {% if message['role'] == 'system' %}
        <|im_start|>system\n{{ message['content'] }}<|im_end|>
    {% elif message['role'] == 'user' %}
        <|im_start|>user\n{{ message['content'] }}<|im_end|>
    {% elif message['role'] == 'assistant' %}
        <|im_start|>assistant\n{{ message['content'] }}<|im_end|>
    {% elif message['role'] == 'tool' %}
        <|im_start|>tool\n{{ message['content'] }}<|im_end|>
    {% endif %}
{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}
    <|im_start|>assistant\n
{% endif %}
"""

search_tool = FakeSearch()

# 系统提示词（包含工具定义）
SYSTEM_PROMPT = fSYSTEM_PROMPT = f"""你是一个高级AI助手，可以访问外部工具。你的任务是根据用户的请求，准确地判断是否需要使用工具，并以严格的格式作出回应。

# 可用工具
- 你可以使用 `search` 工具来获取互联网上的实时、最新或非常具体的信息。工具定义如下：
{json.dumps(TOOL_DEFINITION, indent=2)}

# 响应流程和规则
你必须遵循以下思考和响应流程：
1.  **思考分析**: 在`<think>`标签内，首先分析用户问题的意图。判断问题是基于通用知识（你可以直接回答），还是需要实时/外部信息（必须使用`search`工具）。
    -   **必须使用`search`工具的情况**:
        - 询问今天或最近发生的新闻、事件。
        - 询问天气、股票价格等实时数据。
        - 询问关于2023年之后发生的具体事件。
        - 询问非常具体或冷门的、你知识库中可能没有的实体信息。
    -   **不需要使用`search`工具的情况**:
        - 解释概念（如“什么是黑洞”）。
        - 总结知识（如“总结一下秦朝的历史”）。
        - 进行创作（如“写一首关于春天的诗”）。

2.  **决策与行动**:
    -   **如果决定使用工具**: 在思考结束后，立即生成`<tool_code>`块，内容是`print(search(keyword="...",top_k=5))`。
    -   **如果决定不使用工具**: 在思考结束后，直接在`<answer>`标签内生成最终答案。

3.  **格式要求**:
    - 思考过程必须用`<think>`和`</think>`包裹。
    - 工具调用必须用`<tool_code>`和`</tool_code>`包裹。
    - 最终答案必须用`<answer>`和`</answer>`包裹。
    - 在调用工具后，你会收到工具的返回结果，你需要基于该结果，在新的思考后，生成最终的`<answer>`。

# 示例

## 示例 1: 需要使用工具
user: 2024年巴黎奥运会有哪些新增的比赛项目？

assistant:
<think>
用户在询问2024年巴黎奥运会的新增项目。这是一个关于近期具体事件的问题，我的内部知识可能不完整或不是最新的。因此，我必须使用`search`工具来获取准确信息。搜索关键词可以设置为“2024年巴黎奥运会新增项目”。
</think>
<tool_code>
print(search(keyword="2024年巴黎奥运会新增项目",top_k=5))
</tool_code>

## 示例 2: 不需要使用工具
user: 用简单的语言解释一下什么是人工智能？

assistant:
<think>
用户要求解释一个通用的科学技术概念“人工智能”。这个信息属于我的核心知识范畴，我可以直接回答，不需要调用外部搜索工具。
</think>
<answer>
人工智能（AI）是指让计算机能够像人一样思考、学习和解决问题的技术。它包括很多方面，比如让机器能听懂语言的自然语言处理、能看懂图像的计算机视觉、以及能下棋或开车的机器学习等等。总的来说，就是创造能模仿人类智慧的智能机器。
</answer>
"""


def extract_tool_call_deepseek_style(response):
    """从DeepSeek风格的响应中提取工具调用"""
    # 示例: <tool_code>\nprint(search(keyword="你好", top_k=3))\n</tool_code>
    pattern = r'<tool_code>[\s\n]*print\(search\((.*?)\)\)[\s\n]*</tool_code>'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
        
    args_str = match.group(1)
    try:
        # 将参数字符串转换为字典
        # 注意：这是一个简化的解析器，对于复杂的参数可能需要更健壮的ast.literal_eval
        args = dict(re.findall(r'(\w+)\s*=\s*(".*?"|\'.*?\'|\d+)', args_str))
        # 清理引号
        for key, value in args.items():
            if value.startswith('"') and value.endswith('"'):
                args[key] = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                args[key] = value[1:-1]
            else:
                args[key] = int(value)
        
        return {
            "name": "search",
            "arguments": args
        }
    except Exception as e:
        print(f"解析工具调用参数失败: {e}")
        return None

def generate_response(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    chat_input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Create the attention mask for the first generation
    attention_mask = torch.ones_like(chat_input).to(model.device) # <--- FIX 1

    # 第一次生成
    outputs = model.generate(
        chat_input,
        attention_mask=attention_mask, # <--- FIX 2
        max_new_tokens=500,
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
    )
    
    initial_response = tokenizer.decode(
        outputs[0][len(chat_input[0]):], 
        skip_special_tokens=True
    ).strip()
    
    print("\n--- 初始响应 ---\n", initial_response)

    tool_call = extract_tool_call_deepseek_style(initial_response)
    
    if tool_call and tool_call.get("name") == "search":
        print(f"\n--- 检测到工具调用: {tool_call} ---")
        try:
            args = tool_call.get("arguments", {})
            keyword = args.get("keyword", question)
            top_k = args.get("top_k", 3)
            
            search_results = search_tool.search(keyword, top_k)
            
            tool_response_content = json.dumps({
                "status": "success",
                "results": search_results
            }, ensure_ascii=False)

            messages.append({"role": "assistant", "content": initial_response})
            messages.append({"role": "tool", "content": tool_response_content})
            
            chat_input_2 = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
            
            # Create the attention mask for the second generation
            attention_mask_2 = torch.ones_like(chat_input_2).to(model.device) # <--- FIX 3

            outputs_2 = model.generate(
                chat_input_2,
                attention_mask=attention_mask_2, # <--- FIX 4
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
            )
            
            final_response_part = tokenizer.decode(
                outputs_2[0][len(chat_input_2[0]):], 
                skip_special_tokens=True
            ).strip()

            print("\n--- 基于工具的最终响应部分 ---\n", final_response_part)

            final_response = initial_response + f"\n<tool_response>\n{tool_response_content}\n</tool_response>\n" + final_response_part
            
        except Exception as e:
            print(f"工具调用或第二次生成失败: {e}")
            import traceback
            traceback.print_exc()
            final_response = initial_response + f"\n<think>工具调用出错: {str(e)}</think>"
    else:
        final_response = initial_response

    # 格式化部分保持不变，但要确保它能处理新的<tool_response>标签
    if "<answer>" not in final_response:
        # 简单的答案包装逻辑
        last_meaningful_part = final_response.split('</tool_response>')[-1].strip()
        last_meaningful_part = last_meaningful_part.split('</think>')[-1].strip()
        final_response += f"\n<answer>{last_meaningful_part}</answer>"
    
    return final_response

# 其余代码（load_questions, format_output, main）基本可以保持不变
# ... (你的main函数等)
# ...
def load_questions(file_path):
    """从文件加载问题列表"""
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:  # 文本文件格式
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

def format_output(response):
    """规范化输出格式"""
    # 确保思考过程有标签
    if "<think>" not in response and "</think>" not in response:
        # 这是一个简单的包装，可能需要更智能的逻辑
        response = f"<think>模型直接生成了答案。</think>\n{response}"
    
    # 确保答案有标签
    if "<answer>" not in response:
        # 查找最后一个</think>或</tool_code>之后的内容作为答案
        last_tag_pos = -1
        if "</tool_code>" in response:
            last_tag_pos = response.rfind("</tool_code>") + len("</tool_code>")
        elif "</think>" in response:
             last_tag_pos = response.rfind("</think>") + len("</think>")

        if last_tag_pos != -1:
            answer_content = response[last_tag_pos:].strip()
            if answer_content: # 确保有内容才添加标签
                response = response[:last_tag_pos] + f"\n<answer>{answer_content}</answer>"
            else:
                 response += "\n<answer>未生成有效答案内容。</answer>"
        else:
            response += "\n<answer>未找到合适的答案起始点。</answer>"
    
    return response.strip()

if __name__ == "__main__":
    # 从文件加载问题 - 支持JSON或文本格式
    question_file = "/root/hw4_3/no_search_required_questions.json"  # 或 "questions.txt"
    questions = load_questions(question_file)
    print(f"已加载 {len(questions)} 个问题")
    
    synthesized_data = []
    stats = {
        "total": 0,
        "tool_called": 0,
        "tool_failed": 0,
        "format_errors": 0
    }
    
    # 创建进度日志文件
    with open("synthesis_progress.log", "w", encoding="utf-8") as log_file:
        for i, q in enumerate(questions):
            stats["total"] += 1
            log_file.write(f"处理问题 {i+1}/{len(questions)}: {q}\n")
            print(f"正在处理问题 {i+1}/{len(questions)}: {q[:50]}...")
            
            try:
                response = generate_response(q)
                
                # 检查工具调用情况
                if "<tool_call>" in response:
                    stats["tool_called"] += 1
                    if "<tool_response>" not in response:
                        stats["tool_failed"] += 1
                        log_file.write("警告：工具调用但未处理结果\n")
                
                # 规范化输出格式
                formatted_response = format_output(response)
                if "<answer>" not in formatted_response:
                    stats["format_errors"] += 1
                    log_file.write("警告：答案格式缺失\n")
                
                synthesized_data.append({
                    "question": q,
                    "response": formatted_response
                })
            except Exception as e:
                log_file.write(f"错误：处理失败 - {str(e)}\n")
                print(f"处理问题失败: {q}，错误: {e}")
    
    # 保存合成数据
    output_file = "/root/hw4_3/no_synthetic_data.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(synthesized_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n===== 合成统计 =====")
    print(f"总处理问题: {stats['total']}")
    print(f"工具调用次数: {stats['tool_called']} ({stats['tool_called']/stats['total']*100:.1f}%)")
    print(f"工具调用失败: {stats['tool_failed']}")
    print(f"格式问题: {stats['format_errors']}")
    print(f"成功生成数据: {len(synthesized_data)}/{stats['total']}")
    print(f"数据已保存到: {output_file}")
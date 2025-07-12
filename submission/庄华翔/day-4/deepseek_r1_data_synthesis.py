import json
import re
import tqdm
import random
from openai import OpenAI

# DeepSeek-R1 数据合成模板
SYNTHESIS_TEMPLATE = r"""You are an AI assistant based on DeepSeek-R1-Distill-Qwen-7B. Your task is to demonstrate complex reasoning and decision-making about whether to use search tools.

You have access to a search tool with the following specification:
{
    "name": "search", 
    "description": "搜索引擎，在需要回答与时效性问题有关的用户询问时需要调用此工具", 
    "parameters": {
        "type": "object", 
        "properties": {
            "keyword": {"type": "string", "description": "使用搜索引擎所需的关键词"}, 
            "top_k": {"type": "number", "default": 3, "description": "返回的搜索结果数量"}
        }, 
        "required": ["keyword"]
    }
}

When answering questions, you should:
1. Think step by step in <think></think> tags
2. Decide whether you need to use the search tool based on:
   - Whether the question requires current/real-time information
   - Whether the question is about recent events or time-sensitive topics
   - Whether you have sufficient knowledge to answer without search
3. If you need to search, output the tool call in this exact format after </think>:
   <tool_call>{"name": "search", "arguments": {"keyword": "搜索关键词", "top_k": 3}}</tool_call>
4. If you don't need to search, provide a direct answer after </think>

Please respond to the following question with complex reasoning:

"""

class DeepSeekR1DataSynthesizer:
    def __init__(self, base_url="http://localhost:8005/v1"):
        """
        初始化DeepSeek-R1数据合成器
        需要先部署DeepSeek-R1-Distill-Qwen-7B模型
        
        部署命令示例:
        vllm serve /remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --host 0.0.0.0 --port 8005
        """
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )
        try:
            self.model = self.client.models.list().data[0].id
            print(f"使用模型: {self.model}")
        except Exception as e:
            print(f"连接模型失败: {e}")
            self.model = "/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    def chat(self, messages: list, max_tokens=8192, temperature=0.7):
        """调用DeepSeek-R1模型进行对话"""
        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1
            )
            return result.choices[0].message.content
        except Exception as e:
            print(f"调用模型失败: {e}")
            return ""

    def simulate_search(self, keyword, top_k=3):
        """模拟搜索引擎返回结果"""
        search_prompt = f"""请你扮演一个搜索引擎，对于输入的关键词，给出 {top_k} 个相关的搜索结果。
每个结果应该是大约100-200字的信息片段，包含与关键词相关的有用信息。
结果应该涵盖不同角度和最新信息。

关键词: {keyword}

请以列表形式返回搜索结果，每行一个结果："""
        
        response = self.chat([{"role": "user", "content": search_prompt}])
        
        # 解析搜索结果
        results = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 20:  # 过滤掉太短的行
                # 移除列表标记
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^[-\*]\s*', '', line)
                results.append(line)
                if len(results) >= top_k:
                    break
        
        return results[:top_k]

    def synthesize_data_from_questions(self, question_file, output_file, max_samples=None):
        """从问题文件合成训练数据"""
        with open(question_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f.readlines() if line.strip()]
        
        if max_samples:
            questions = questions[:max_samples]
        
        synthesized_data = []
        
        for question in tqdm.tqdm(questions, desc="合成数据"):
            try:
                # 第一步：让模型思考并决定是否需要搜索
                prompt = SYNTHESIS_TEMPLATE + question
                response = self.chat([{"role": "user", "content": prompt}])
                
                if not response:
                    continue
                
                # 解析响应
                if "<think>" in response and "</think>" in response:
                    think_content = response.split("<think>")[1].split("</think>")[0].strip()
                    after_think = response.split("</think>")[1].strip()
                    
                    # 检查是否有工具调用
                    if "<tool_call>" in after_think and "</tool_call>" in after_think:
                        # 需要搜索的情况
                        tool_call_str = after_think.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                        
                        try:
                            tool_call = json.loads(tool_call_str)
                            keyword = tool_call["arguments"]["keyword"]
                            top_k = tool_call["arguments"].get("top_k", 3)
                            
                            # 模拟搜索
                            search_results = self.simulate_search(keyword, top_k)
                            search_result_text = "\n".join(search_results)
                            
                            # 第二步：基于搜索结果继续回答
                            follow_up_prompt = f"""基于以下搜索结果，请继续回答用户的问题。请在<think></think>标签中进行思考，然后给出最终答案。

搜索结果：
{search_result_text}

用户问题：{question}"""
                            
                            final_response = self.chat([{"role": "user", "content": follow_up_prompt}])
                            
                            if "<think>" in final_response and "</think>" in final_response:
                                final_think = final_response.split("<think>")[1].split("</think>")[0].strip()
                                final_answer = final_response.split("</think>")[1].strip()
                                
                                # 构建完整的对话数据
                                conversation = [
                                    {"role": "system", "content": self._get_system_prompt()},
                                    {"role": "user", "content": question},
                                    {"role": "assistant", "content": f"<think>{think_content}</think>\n<tool_call>{tool_call_str}</tool_call>"},
                                    {"role": "user", "content": f"<tool_response>{search_result_text}</tool_response>"},
                                    {"role": "assistant", "content": f"<think>{final_think}</think>\n{final_answer}"}
                                ]
                                
                                synthesized_data.append(conversation)
                            
                        except json.JSONDecodeError:
                            print(f"解析工具调用失败: {tool_call_str}")
                            continue
                    
                    else:
                        # 不需要搜索的情况
                        conversation = [
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": f"<think>{think_content}</think>\n{after_think}"}
                        ]
                        
                        synthesized_data.append(conversation)
                
            except Exception as e:
                print(f"处理问题失败: {question}, 错误: {e}")
                continue
        
        # 保存合成的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(synthesized_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功合成 {len(synthesized_data)} 条对话数据，保存到 {output_file}")
        return synthesized_data

    def _get_system_prompt(self):
        """获取系统提示词"""
        return r"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Current Date: 2025-03-03

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name": "search","description": "搜索引擎，在需要回答与时效性问题有关的用户询问时需要调用此工具","parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "使用搜索引擎所需的关键词"}, "top_k": {"type": "number", "default": 3, "description": "返回的搜索结果数量"}}, "required": ["keyword"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

    def generate_diverse_questions(self, num_questions=100):
        """生成多样化的问题用于训练"""
        question_templates = [
            "帮我查询{}的最新信息",
            "{}的发展现状如何？",
            "请介绍一下{}的相关情况",
            "{}有什么新的进展吗？",
            "如何理解{}这个概念？",
            "{}的未来趋势是什么？",
            "请分析{}的影响和意义",
            "{}的工作原理是什么？",
            "{}目前面临哪些挑战？",
            "{}有哪些实际应用？"
        ]
        
        topics_current = [  # 需要搜索的话题
            "人工智能最新发展", "区块链技术", "新能源汽车", "量子计算", "元宇宙",
            "ChatGPT发展", "中美贸易", "疫情防控政策", "股市行情", "房价走势",
            "奥运会最新消息", "科技创新政策", "环保新规定", "教育改革", "医疗保险"
        ]
        
        topics_general = [  # 不需要搜索的话题
            "数学基础概念", "物理定律", "历史事件", "文学作品", "哲学思想",
            "编程语言", "算法原理", "化学反应", "生物结构", "地理知识",
            "语言学习", "艺术欣赏", "心理学理论", "经济学原理", "法律条文"
        ]
        
        questions = []
        
        # 生成需要搜索的问题
        for _ in range(num_questions // 2):
            template = random.choice(question_templates)
            topic = random.choice(topics_current)
            question = template.format(topic)
            questions.append(question)
        
        # 生成不需要搜索的问题
        for _ in range(num_questions // 2):
            template = random.choice(question_templates)
            topic = random.choice(topics_general)
            question = template.format(topic)
            questions.append(question)
        
        random.shuffle(questions)
        return questions

def main():
    """主函数"""
    synthesizer = DeepSeekR1DataSynthesizer()
    
    # 方案1：从现有问题文件合成数据
    print("=== 从现有问题文件合成数据 ===")
    
    # 合成需要搜索的数据
    if input("是否从 question_with_search.txt 合成数据？(y/n): ").lower() == 'y':
        synthesizer.synthesize_data_from_questions(
            "data/question_with_search.txt",
            "data/synthesized_with_search.json",
            max_samples=50  # 限制样本数量用于测试
        )
    
    # 合成不需要搜索的数据
    if input("是否从 question_without_search.txt 合成数据？(y/n): ").lower() == 'y':
        synthesizer.synthesize_data_from_questions(
            "data/question_without_search.txt", 
            "data/synthesized_without_search.json",
            max_samples=50
        )
    
    # 方案2：生成新的多样化问题并合成数据
    if input("是否生成新的多样化问题并合成数据？(y/n): ").lower() == 'y':
        print("=== 生成多样化问题并合成数据 ===")
        questions = synthesizer.generate_diverse_questions(100)
        
        # 保存问题
        with open("data/generated_questions.txt", "w", encoding="utf-8") as f:
            for q in questions:
                f.write(q + "\n")
        
        # 合成数据
        synthesizer.synthesize_data_from_questions(
            "data/generated_questions.txt",
            "data/synthesized_generated.json"
        )
    
    print("数据合成完成！")

if __name__ == "__main__":
    main() 
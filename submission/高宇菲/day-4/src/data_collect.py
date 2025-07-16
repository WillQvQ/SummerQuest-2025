import os
import json
import csv
from pathlib import Path
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset, DataLoader  # 新增
import re
from search import FakeSearch

model_path = {
    'DeepSeek-R1-Distill-Qwen-7B': "/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    'Qwen2.5-0.5B-Instruct': "/remote-home1/share/models/Qwen2.5-0.5B-Instruct",
}

search_tool = {
        "type": "function", 
        "function": {
            "name": "search", 
            "description": "搜索引擎，在需要实时信息的时候需要调用此工具", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "keyword": {"type": "string", "description": "使用搜索引擎所需的关键词"}, 
                    "top_k": {"type": "number", "default": 3, "description": "返回的搜索结果数量"}
                }, 
        "required": ["keyword"]}
        }
    }
TOOLS = [search_tool]


# 节省内存的写法
class PromptDataset(Dataset):
    def __init__(self, with_search: bool = True):
        q_with_search = "/remote-home1/yfgao/SummerQuest-2025/handout/day-4/question_with_search.txt"
        q_wo_search = "/remote-home1/yfgao/SummerQuest-2025/handout/day-4/question_without_search.txt"
        prompts_file = q_with_search if with_search else q_wo_search
        self.prompts_file = prompts_file  
        self.line_offsets = []
        with open(prompts_file, 'r') as f:
            pos = f.tell()
            line = f.readline()
            while line:
                self.line_offsets.append(pos)
                pos = f.tell()
                line = f.readline()

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        with open(self.prompts_file, 'r') as f:
            f.seek(self.line_offsets[idx])
            return f.readline().strip()

class DataCollector:
    def __init__(self, model_name: str, with_search: bool = True, use_tool: bool = True, append_tool_res: bool = True):
        self.model_name = model_path.get(model_name, model_name)
        self.with_search = with_search
        self.use_tool = use_tool
        self.append_tool_res = append_tool_res
        if not self.use_tool:
            self.append_tool_res = False
        self.output_path = f"collected_w_search_{self.with_search}_usetool_{self.use_tool}_appen_tool_call_{self.append_tool_res}_{model_name.replace('/', '_')}.jsonl"
        self.llm = self.build_llm(self.model_name, 2)
        self.sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=12000)
        self.tokenizer = self.llm.get_tokenizer() 
        # self.prompts = self.read_prompts()  
        self.prompt_dataset = PromptDataset(self.with_search)
        self.search = FakeSearch() 

    def build_llm(self, model_name, num_gpus = None):
        if num_gpus is None:
            num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) or 1
        return LLM(
            model=model_name,
            tensor_parallel_size=num_gpus,
            dtype="half",
            gpu_memory_utilization=0.9,
        )
       
    def format_prompts_with_chat_template(self, prompt: str):
        # sys_prompt_tool = f'\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{search_tool}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n'+'{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'+"\nAlways wrap tool calls in <tool_call> tags."
        
        if 'DeepSeek-R1-Distill-Qwen-7B' in self.model_name:
            sys_prompt_tool = f'\n\n# Tools\n\n你可以调用一个或多个函数来协助用户查询。\n\n在<tools></tools> XML标记中为你提供了函数签名:\n<tools>\n{search_tool}\n</tools>\n\n对于每个函数调用，返回一个json对象，其中包含函数名和参数。以如下格式返回:\n<|TOOL|>'+'{"name": <function-name>, "arguments": <args-json-object>}'
            if not self.use_tool:
                sys_prompt_tool = ""
            message = [
                {"role": "system", "content": f"你是一个回答问题的助手。{sys_prompt_tool}"},
                {"role": "user", "content": prompt}
            ]
        elif 'Qwen2.5-0.5B-Instruct' in self.model_name:
            message = [
                {"role": "system", "content": "You are a bot that responds to queries."},
                {"role": "user", "content": prompt}
            ]   
        formatted_prompt = self.tokenizer.apply_chat_template(
            message, 
            tools=TOOLS if self.use_tool else None,
            tokenize=False, 
            add_generation_prompt=True,
        )
        return formatted_prompt
    
    def save_results(self, path: str, pairs: list[tuple[str, str]]) -> None:
        """
        path 后缀决定格式：
        .jsonl → 每行 {"prompt":..., "output":...}
        .csv   → prompt,output 两列
        .txt   → 连续文本块，用分隔符
        """
        ext = Path(path).suffix.lower()
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if ext == ".jsonl":
            with open(path, "w", encoding="utf‑8") as f:
                for p, o in pairs:
                    f.write(json.dumps({"prompt": p, "output": o}, ensure_ascii=False) + "\n")

        elif ext == ".csv":
            with open(path, "w", encoding="utf‑8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["prompt", "output"])
                writer.writerows(pairs)

        elif ext == ".txt":
            sep = "\n" + "-" * 60 + "\n"
            with open(path, "w", encoding="utf‑8") as f:
                blocks = [f"Prompt: {p}\nOutput: {o}" for p, o in pairs]
                f.write(sep.join(blocks))
        else:
            raise ValueError("Unsupported format: use .jsonl / .csv / .txt")

    def eval(self, path):
        pairs = [json.loads(line)['output'] for line in open(path, 'r', encoding='utf-8').readlines()]
        num = len(pairs)

        cot_len = 0
        use_tool = 0
        wrong_call = 0
        if 'DeepSeek-R1-Distill-Qwen-7B' in self.model_name:
            print(f"评估 {path}")
            for o in pairs:
                if "<think>" in o:
                    match = re.search(r"<think>(.*?)</think>", o, re.DOTALL)
                    if match:
                        cot_len += len(match.group(1).strip()) / num
                    if self.use_tool and "<|TOOL|>" in o:
                            use_tool += 1
                else:
                    wrong_call += 1
            print(f"缺少think标签：{wrong_call}，占比 {wrong_call/num:.2%}")
        elif 'Qwen2.5-0.5B-Instruct' in self.model_name:
            print("评估 Qwen2.5-0.5B-Instruct 模型的输出")
            for o in pairs:
                if "<tool_call>" in o:
                    use_tool += 1
                    cot_len += len(o.split("<tool_call>")[0].strip()) / num

        print(f"共 {num} 个问题，使用了工具的有 {use_tool} 个，占比 {use_tool/num:.2%}")
        print(f"平均思考长度为 {cot_len:.2f}")
    
    def generate(self, prompts, sampling_params):
        if not self.append_tool_res:
            return self.llm.generate(prompts, sampling_params)
        if not self.use_tool or not self.with_search:
            return self.llm.generate(prompts, sampling_params)
        else:
            output_before_toolcall = []
            search_args = []
            res = []
            outputs = self.llm.generate(prompts, sampling_params)
            # import pdb; pdb.set_trace()
            formatted_prompts = []
            for i, output in enumerate(outputs):
                o = output.outputs[0].text
                if '<|TOOL|>' in o:
                    tool_call_str = o.split('<|TOOL|>')[-1].strip()
                    if tool_call_str.endswith('}'):
                        try:
                            tool_call = json.loads(tool_call_str)
                            if tool_call['name'] == 'search' and 'arguments' in tool_call:
                                search_results = self.search.search(
                                    tool_call['arguments']['keyword'], 
                                    tool_call['arguments'].get('top_k', 3)
                                )
                                use_tool = self.use_tool
                                self.use_tool = False  # 禁用工具调用
                                formatted_prompts.append(self.format_prompts_with_chat_template(
                                    f"{prompts[i]}\n已知: {search_results}"
                                ))
                                
                                self.use_tool = use_tool  # 恢复工具调用
                                # import pdb; pdb.set_trace()
                                output_before_toolcall.append(o.split("<|TOOL|>"+tool_call_str)[0].strip() + "\n<tool_call>\n" + tool_call_str + "\n</tool_call>")
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON from tool call: {tool_call}")
            outputs_after_toolcall = self.llm.generate(formatted_prompts, sampling_params)
            for ob, oa in zip(output_before_toolcall, outputs_after_toolcall):
                res.append({
                    "prompt": prompts[i],
                    "output": ob + oa.outputs[0].text
                })
            return res
             
    def collect(self):
        pairs = []
        dataloader = DataLoader(self.prompt_dataset, batch_size=200, shuffle=False)
        for batch_prompts in dataloader:
            formatted_prompts = [self.format_prompts_with_chat_template(p) for p in batch_prompts]
            results = self.generate(formatted_prompts, self.sampling)
            pairs += [(r['prompt'], r["output"]) for r in results]
        return pairs
    

if __name__ == "__main__":
    data_collector = DataCollector(
        model_name='DeepSeek-R1-Distill-Qwen-7B', # 'Qwen2.5-0.5B-Instruct'
        with_search=True,
        use_tool=True,
        append_tool_res=True
    )
    pairs = data_collector.collect()
    data_collector.save_results(data_collector.output_path, pairs)
    print(f"Results saved to {data_collector.output_path}")
    data_collector.eval(data_collector.output_path)




# multi_gpu_vllm_save.py
import os
import json
import csv
from pathlib import Path
from argparse import ArgumentParser
from vllm import LLM, SamplingParams

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

def read_prompts():
    prompts = []
    q_with_search = "/remote-home1/yfgao/SummerQuest-2025/handout/day-4/question_with_search.txt"
    q_wo_search = "/remote-home1/yfgao/SummerQuest-2025/handout/day-4/question_without_search.txt"
    files = [q_with_search, q_wo_search]
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    prompts.append(line)
    return prompts[:5]+prompts[-5:]
    # return prompts

def save_results(path: str, pairs: list[tuple[str, str]]) -> None:
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

def build_llm(model_name: str, num_gpus: int | None):
    if num_gpus is None:
        num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) or 1
    return LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        dtype="half",
        gpu_memory_utilization=0.9,
    )

def format_prompts_with_chat_template(llm, prompt):
    """使用模型的chat template格式化prompts"""
    tokenizer = llm.get_tokenizer()

    # sys_prompt_tool = f'\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{search_tool}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n'+'{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'+"\nAlways wrap tool calls in <tool_call> tags."
    sys_prompt_tool = f'\n\n# Tools\n\n你可以调用一个或多个函数来协助用户查询。\n\n在<tools></tools> XML标记中为你提供了函数签名:\n<tools>\n{search_tool}\n</tools>\n\n对于每个函数调用，返回一个json对象，其中包含函数名和参数。以如下格式返回:\n<|TOOL|>'+'{"name": <function-name>, "arguments": <args-json-object>}'
    # +"示例1:\n<|TOOL|>{\"name\": \"search\",\"arguments\": {\"query\": \"最新天气\"}}"
    message = [
        {"role": "system", "content": f"你是一个回答问题的助手。{sys_prompt_tool}"},
        {"role": "user", "content": prompt}
    ]
    # message = [
    #     {"role": "system", "content": "You are a bot that responds to queries."},
    #     {"role": "user", "content": prompt}
    # ]   
    formatted_prompt = tokenizer.apply_chat_template(
        message, 
        tools=TOOLS,
        tokenize=False, 
        add_generation_prompt=True
    )
    return formatted_prompt

def eval(pairs):
    num = len(pairs)

    cot_len = 0
    use_tool = 0
    wrong_answer = 0
    for p, o in pairs:
        if "<tool_call>" in o:
            use_tool += 1
            cot_len += len(o.split("<tool_call>")[0].strip())/num

    print(f"共 {num} 个问题，使用了工具的有 {use_tool} 个，占比 {use_tool/num:.2%}")
    print(f"平均思考长度为 {cot_len:.2f}")




def main():
    parser = ArgumentParser()
    parser.add_argument("--model", default="/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    # parser.add_argument("--model", default="/remote-home1/share/models/Qwen2.5-0.5B-Instruct")
    # parser.add_argument("--model", default="/remote-home1/share/models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="显式指定并行卡数；默认读取 CUDA_VISIBLE_DEVICES")
    parser.add_argument("--save_path", type=str, default="outputs_r1.jsonl",
                        help="输出文件名，后缀决定格式 .jsonl / .csv / .txt")
    args = parser.parse_args()

    sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=12000)
    llm = build_llm(args.model, args.num_gpus)

    # -------- inference -------- #
    PROMPTS = read_prompts()
    formatted_prompts = []
    # import pdb; pdb.set_trace()  # 调试断点
    for p in PROMPTS:
        prompt = format_prompts_with_chat_template(llm, p)
        formatted_prompts.append(prompt)
    results = llm.generate(formatted_prompts, sampling)

    pairs = [(r.prompt, r.outputs[0].text) for r in results]

    # -------- print & save ------ #
    for p, o in pairs:
        print("-" * 60)
        print(f"Prompt:  {p!r}")
        print(f"Output:  {o!r}")
    print("-" * 60)

    w_search_pairs = pairs[:200]
    wo_search_pairs = pairs[200:]
    save_results("w_search_"+args.save_path, w_search_pairs)
    eval(w_search_pairs)
    print(f"Saved to w_search_{args.save_path}")

    save_results("wo_search_"+args.save_path, wo_search_pairs)
    eval(wo_search_pairs)
    print(f"Saved to wo_search_{args.save_path}")

if __name__ == "__main__":
    main()

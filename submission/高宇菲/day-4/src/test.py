import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

random.seed(42)
class CustomDataset(Dataset):
    def __init__(self, phase='train', tokenizer=None):
        data_path = "/remote-home1/yfgao/SummerQuest-2025/submission/高宇菲/day-4/data/collected_data.jsonl"
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.data = []
        for line in lines:
            try:
                item = json.loads(line.strip())
                prompt = item.get("prompt", "")
                prompt = prompt.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
                self.data.append({
                    "input": prompt,
                    "output": item["output"]
                })
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line.strip()}")
                
        random.shuffle(self.data)
        split_idx = int(len(self.data) * 0.8)
        if phase == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 格式化为聊天模板
        messages = [
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]}
        ]
        text = self.tokenizer.apply_chat_template(messages, tool=TOOLS, tokenize=False, add_generation_prompt=False)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "input": item["input"],
        }

def load_model_and_tokenizer():
    base_model_name = "/remote-home1/share/models/Qwen2.5-0.5B-Instruct"
    adapter_path = "./qwen25-lora-final"
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer


def main():
    model, tokenizer = load_model_and_tokenizer()
    testset = CustomDataset(phase='test', tokenizer=tokenizer)
    dataloader = DataLoader(testset, batch_size=8, shuffle=False)

    results = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)       
            batch_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
            }   
            outputs = model.generate(
                **batch_input,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(response)
        results.extend([
            {
                "input": batch["input"][i],
                "output": response[i]
            } for i in range(len(outputs))
        ])
    
    output_path = "./test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest results saved to {output_path}")

if __name__ == "__main__":
    main()

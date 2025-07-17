import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType

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
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
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
            "labels": input_ids.clone()
        }


model_name = "/remote-home1/share/models/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 设置padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, peft_config)

# 创建数据集
train_dataset = CustomDataset(phase='train', tokenizer=tokenizer)
eval_dataset = CustomDataset(phase='eval', tokenizer=tokenizer)

# 训练参数
training_args = TrainingArguments(
    output_dir="./qwen25-lora-output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_strategy="steps",
    fp16=True,
    dataloader_drop_last=True,
    remove_unused_columns=False,
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("./qwen25-lora-final")





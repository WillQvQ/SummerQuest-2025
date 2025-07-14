import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 1. 配置模型和数据路径 ---
# 基础模型
base_model_path = "/data-mnt/data/downloaded_ckpts/Qwen2.5-0.5B-Instruct"
# LoRA适配器输出目录
output_dir = "/root/hw4_3/qwen_0.5b_tool_lora_mixed"


train_data_file = '/root/hw4_3/qwen_tool_train_mixed.jsonl'
eval_data_file = '/root/hw4_3/qwen_tool_train_eval_mixed.jsonl'


# --- 2. 加载Tokenizer (保持不变) ---
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = """{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|im_start|>system\n' + message['content'] + '<|im_end|>\n'}}{% elif (message['role'] == 'user') %}{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n'}}{% elif (message['role'] == 'tool') %}{{'<|im_start|>tool\n' + message['content'] + '<|im_end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"""


# --- 3. 加载并合并数据集 (核心修改区域) ---
print("正在加载已切分好的训练集和验证集...")
# 直接使用 load_dataset 的 `data_files` 参数
tokenized_datasets = load_dataset("json", data_files={'train': train_data_file, 'validation': eval_data_file})

print(f"加载训练集: {len(tokenized_datasets['train'])} 条")
print(f"加载验证集: {len(tokenized_datasets['validation'])} 条")

# !! 核心修改：数据处理流程现在直接作用于这个DatasetDict !!
def preprocess_function_new(examples):
    outputs = tokenizer.apply_chat_template(
        examples['messages'], 
        truncation=True, 
        max_length=1024,
    )
    return {"input_ids": outputs, "labels": outputs}

# 直接在整个DatasetDict上应用map
tokenized_datasets = tokenized_datasets.map(
    preprocess_function_new, 
    remove_columns=list(tokenized_datasets['train'].features),
    batched=True # 使用batched=True可以加速
)

print("数据预处理和Tokenization完成。")
print("示例数据 'input_ids':", tokenized_datasets['train'][0]['input_ids'][:20])

# --- 4. 加载模型并配置PEFT (LoRA) (保持不变) ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# --- 5. 配置训练参数 (保持不变) ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=10,
    fp16=True,
    report_to="tensorboard",
)

class CustomTrainer(Trainer):
    def _prepare_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        在将输入传递给模型之前，准备好输入。
        这里我们重写这个方法，以确保 `input_ids` 始终是 torch.long 类型。
        """
        # 首先，调用父类的 _prepare_inputs 方法，它会处理设备移动和半精度转换
        prepared_inputs = super()._prepare_inputs(inputs)
        
        # 检查 'input_ids' 是否存在并且其类型不是 long
        if "input_ids" in prepared_inputs and prepared_inputs["input_ids"].dtype != torch.long:
            # 强制将其类型转换回 torch.long
            prepared_inputs["input_ids"] = prepared_inputs["input_ids"].long()
            
        return prepared_inputs

# --- 6. 初始化Trainer并开始训练 ---
trainer = CustomTrainer( # <--- 使用 CustomTrainer
    model=model,
    args=training_args,
   train_dataset=tokenized_datasets["train"], # 直接从DatasetDict中取
    eval_dataset=tokenized_datasets["validation"], # 直接从DatasetDict中取
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
)


print("开始混合数据集训练...")
trainer.train()

final_adapter_path = os.path.join(output_dir, "final_adapter")
trainer.model.save_pretrained(final_adapter_path)
print(f"训练完成，混合数据LoRA适配器已保存到: {final_adapter_path}")
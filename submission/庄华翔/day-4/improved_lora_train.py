import json
import os
import sys
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import logging
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedQwenTrainer:
    def __init__(self, 
                 model_name="/remote-home1/share/models/Qwen2.5-0.5B-Instruct",
                 output_dir="./qwen-lora-final",
                 max_length=4096):
        """
        初始化训练器
        
        Args:
            model_name: 基础模型路径
            output_dir: 输出目录
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 配置LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,        
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            r=16,                    # 增加rank以提升性能
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none"
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def load_and_process_data(self, data_files, validation_split=0.1, max_samples=None):
        """
        加载并处理训练数据
        
        Args:
            data_files: 数据文件列表
            validation_split: 验证集比例
            max_samples: 最大样本数（用于调试）
        """
        all_conversations = []
        
        for data_file in data_files:
            logger.info(f"加载数据文件: {data_file}")
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_conversations.extend(data)
                    else:
                        logger.warning(f"数据文件格式不正确: {data_file}")
            except Exception as e:
                logger.error(f"加载数据文件失败: {data_file}, 错误: {e}")
        
        logger.info(f"总共加载 {len(all_conversations)} 条对话")
        
        if max_samples:
            all_conversations = all_conversations[:max_samples]
            logger.info(f"限制样本数量为: {max_samples}")
        
        # 处理数据
        processed_data = []
        for conversation in all_conversations:
            try:
                processed_item = self.process_conversation(conversation)
                if processed_item:
                    processed_data.append(processed_item)
            except Exception as e:
                logger.warning(f"处理对话失败: {e}")
                continue
        
        logger.info(f"成功处理 {len(processed_data)} 条对话")
        
        # 创建数据集
        dataset = Dataset.from_list(processed_data)
        
        # 划分训练集和验证集
        if validation_split > 0:
            dataset = dataset.train_test_split(test_size=validation_split, shuffle=True, seed=42)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
            logger.info(f"训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条")
        else:
            train_dataset = dataset
            eval_dataset = None
            logger.info(f"训练集: {len(train_dataset)} 条")
        
        return train_dataset, eval_dataset

    def process_conversation(self, conversation):
        """
        处理单个对话，转换为训练格式
        
        Args:
            conversation: 对话数据（消息列表）
        """
        if not isinstance(conversation, list) or len(conversation) < 2:
            return None
        
        # 应用对话模板
        try:
            formatted = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            input_ids = formatted["input_ids"].squeeze(0)
            attention_mask = formatted["attention_mask"].squeeze(0)
            
            # 创建标签：只计算assistant回复的loss
            labels = input_ids.clone()
            labels[:] = -100  # 初始化为忽略标记
            
            # 找到assistant的回复并设置标签
            current_pos = 0
            for message in conversation:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "assistant":
                    # 编码assistant的内容
                    role_str = f"<|assistant|>\n"
                    full_content = content + self.tokenizer.eos_token
                    
                    role_tokens = self.tokenizer.encode(role_str, add_special_tokens=False)
                    content_tokens = self.tokenizer.encode(full_content, add_special_tokens=False)
                    
                    # 计算在完整序列中的位置（这是简化版本，实际可能需要更精确的位置计算）
                    # 为了简化，我们使用一个启发式方法
                    assistant_start = self._find_assistant_start(input_ids, role_tokens + content_tokens[:10])
                    if assistant_start >= 0:
                        assistant_end = min(assistant_start + len(role_tokens) + len(content_tokens), len(labels))
                        # 只对assistant的内容部分计算loss（跳过role部分）
                        content_start = assistant_start + len(role_tokens)
                        labels[content_start:assistant_end] = input_ids[content_start:assistant_end]
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        except Exception as e:
            logger.warning(f"处理对话失败: {e}")
            return None

    def _find_assistant_start(self, input_ids, pattern_tokens):
        """简化的模式匹配，寻找assistant回复的开始位置"""
        pattern_length = min(len(pattern_tokens), 10)  # 只用前10个token作为模式
        input_list = input_ids.tolist()
        pattern_list = pattern_tokens[:pattern_length]
        
        for i in range(len(input_list) - pattern_length + 1):
            if input_list[i:i+pattern_length] == pattern_list:
                return i
        return -1

    def train(self, 
              train_dataset, 
              eval_dataset=None,
              num_train_epochs=3,
              per_device_train_batch_size=1,
              gradient_accumulation_steps=8,
              learning_rate=2e-4,
              warmup_ratio=0.1,
              logging_steps=10,
              save_steps=500,
              eval_steps=500,
              save_total_limit=3):
        """
        执行训练
        """
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            logging_dir=f"{self.output_dir}/logs",
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=True,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to=None,  # 禁用wandb等
            # FSDP配置（如果需要）
            # fsdp="full_shard auto_wrap" if torch.cuda.device_count() > 1 else "",
        )
        
        # 回调函数
        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks
        )
        
        # 开始训练
        logger.info("开始训练...")
        train_result = trainer.train()
        
        # 保存模型
        trainer.save_model()
        trainer.save_state()
        
        # 保存训练日志
        with open(f"{self.output_dir}/train_results.json", "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info("训练完成!")
        logger.info(f"模型已保存到: {self.output_dir}")
        
        return train_result

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练Qwen模型")
    parser.add_argument("--model_name", type=str, 
                       default="/remote-home1/share/models/Qwen2.5-0.5B-Instruct",
                       help="基础模型路径")
    parser.add_argument("--data_files", nargs="+", 
                       default=["data/data_to_train.json"],
                       help="训练数据文件列表")
    parser.add_argument("--output_dir", type=str, 
                       default="./qwen-lora-final",
                       help="输出目录")
    parser.add_argument("--max_length", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数（用于调试）")
    parser.add_argument("--validation_split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化训练器
    trainer = ImprovedQwenTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length
    )
    
    # 加载和处理数据
    train_dataset, eval_dataset = trainer.load_and_process_data(
        data_files=args.data_files,
        validation_split=args.validation_split,
        max_samples=args.max_samples
    )
    
    # 执行训练
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main() 
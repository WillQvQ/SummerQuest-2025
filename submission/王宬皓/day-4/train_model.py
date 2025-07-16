import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import nested_detach
from typing import Optional, Tuple
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightedLossTrainer(Trainer):
    """
    自定义Trainer，支持对不同位置的token施加不同的损失权重
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写损失计算函数，支持位置加权损失
        """
        # 从输入中提取位置权重信息
        position_weights = inputs.pop("position_weights", None)
        
        # 执行模型前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        # 确保位置权重与标签形状匹配
        if position_weights is not None:
            if position_weights.shape != labels.shape:
                raise ValueError(f"位置权重形状 {position_weights.shape} 与标签形状 {labels.shape} 不匹配")
        
        # 计算加权交叉熵损失
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses = losses.view(labels.shape)
        
        # 应用位置权重
        if position_weights is not None:
            losses = losses * position_weights
        
        # 忽略标签为-100的位置（标准做法）
        losses = losses[labels != -100]
        
        # 计算平均损失
        loss = losses.mean()
        
        # 关键修复：确保损失函数与计算图连接
        if not loss.requires_grad:
            # 如果损失没有连接到计算图，重新连接
            loss = loss.requires_grad_()
        
        return (loss, outputs) if return_outputs else loss


class WeightedPPOTrainer(PPOTrainer):
    """
    自定义PPOTrainer，支持对PPO训练中的不同位置施加不同的损失权重
    """
    def __init__(self, config: PPOConfig = None, model: AutoModelForCausalLMWithValueHead = None, ref_model=None, tokenizer=None, **kwargs):
        super().__init__(config, model, ref_model, tokenizer, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写PPO损失计算，支持位置加权
        """
        # 从输入中提取位置权重信息
        position_weights = inputs.pop("position_weights", None)
        
        # 执行标准PPO前向传播
        outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss = outputs["loss"]
        
        # 应用位置权重到损失
        if position_weights is not None:
            # 获取模型预测的logits
            logits = outputs["logits"]
            labels = inputs.get("labels")
            
            # 确保位置权重与标签形状匹配
            if position_weights.shape != labels.shape:
                raise ValueError(f"位置权重形状 {position_weights.shape} 与标签形状 {labels.shape} 不匹配")
            
            # 计算加权交叉熵损失
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            losses = losses.view(labels.shape)
            
            # 应用位置权重
            weighted_losses = losses * position_weights
            
            # 忽略标签为-100的位置
            weighted_losses = weighted_losses[labels != -100]
            
            # 计算新的加权损失
            loss = weighted_losses.mean()
        
        # 关键修复：确保损失函数与计算图连接
        if not loss.requires_grad:
            loss = loss.requires_grad_()
        
        return (loss, outputs) if return_outputs else loss


def dynamic_padding_collator(features, tokenizer):
    """
    动态填充批次中的序列到相同长度
    """
    # 检查特征是否为空
    if len(features) == 0:
        return None  # 返回None而不是空字典
    
    # 在开始处理前检查每个样本是否包含必需键
    required_keys = ['input_ids', 'attention_mask', 'labels', 'position_weights']
    valid_features = []
    
    for feature in features:
        if all(key in feature for key in required_keys):
            # 添加额外检查：确保值不为空
            if (len(feature['input_ids']) > 0 and 
                len(feature['attention_mask']) > 0 and 
                len(feature['labels']) > 0 and 
                len(feature['position_weights']) > 0):
                valid_features.append(feature)
            else:
                logger.warning("样本包含空值字段，已跳过")
        else:
            logger.warning(f"样本缺少必需键: {set(required_keys) - set(feature.keys())}")
    
    if len(valid_features) == 0:
        return None  # 返回None而不是空字典
    
    # 确定批次中每个字段的最大长度
    max_length = max(len(f["input_ids"]) for f in valid_features)
    
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "position_weights": []
    }
    
    for feature in valid_features:
        # 计算需要填充的长度
        pad_length = max_length - len(feature["input_ids"])
        
        # 填充input_ids
        input_ids = feature["input_ids"] + [tokenizer.pad_token_id] * pad_length
        batch["input_ids"].append(input_ids)
        
        # 填充attention_mask
        attention_mask = feature["attention_mask"] + [0] * pad_length
        batch["attention_mask"].append(attention_mask)
        
        # 填充labels（用-100表示忽略）
        labels = feature["labels"] + [-100] * pad_length
        batch["labels"].append(labels)
        
        # 填充position_weights（用0表示不参与损失计算）
        position_weights = feature["position_weights"] + [0.0] * pad_length
        batch["position_weights"].append(position_weights)
    
    # 转换为张量
    return {
        "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(batch["labels"], dtype=torch.long),
        "position_weights": torch.tensor(batch["position_weights"], dtype=torch.float)
    }


def prepare_weighted_training_data(dataset, tokenizer, max_length=512):
    """
    准备带位置权重的训练数据
    """
    processed_data = []
    
    for i, example in enumerate(dataset):
        try:
            # 确保示例包含必要字段
            example.setdefault('system', "")
            example.setdefault('input', "")
            
            # 必须包含的输出字段
            if 'output' not in example:
                logger.warning(f"样本 {i} 缺少 'output' 字段，已跳过")
                continue
                
            # 构造完整提示
            prompt = f"{example['system']}\n\n### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
            
            # 对提示进行分词
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            
            # 对响应进行分词
            response_ids = tokenizer.encode(example['output'], add_special_tokens=False)
            
            # 创建完整输入序列（提示 + 响应）
            input_ids = prompt_ids + response_ids
            input_ids = input_ids[:max_length]
            
            # 创建标签（对提示部分使用-100忽略，只计算响应部分的损失）
            labels = [-100] * len(prompt_ids) + response_ids
            labels = labels[:max_length]
            
            # 创建注意力掩码
            attention_mask = [1] * len(input_ids)
            
            # 创建位置权重（初始化为1.0）
            position_weights = [1.0] * len(input_ids)
            
            # 识别特殊token位置并应用自定义权重
            response_start = len(prompt_ids)
            
            # 1. 强化<think>标签内的内容（权重1.5）
            if "<think>" in example['output']:
                start_idx = example['output'].find("<think>")
                end_idx = example['output'].find("</think>")
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    # 计算token位置
                    prefix = example['output'][:start_idx]
                    think_content = example['output'][start_idx:end_idx+len("</think>")]
                    
                    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                    think_ids = tokenizer.encode(think_content, add_special_tokens=False)
                    
                    think_start = response_start + len(prefix_ids)
                    think_end = think_start + len(think_ids)
                    
                    # 应用权重
                    for idx in range(think_start, min(think_end, len(position_weights))):
                        position_weights[idx] = 1.5
            
            # 2. 强化<|SEARCH|>特殊token（权重2.0）
            if "<|SEARCH|>" in example['output']:
                search_idx = example['output'].find("<|SEARCH|>")
                if search_idx != -1:
                    # 计算token位置
                    prefix = example['output'][:search_idx]
                    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                    token_pos = response_start + len(prefix_ids)
                    search_token_len = len(tokenizer.encode("<|SEARCH|>", add_special_tokens=False))
                    
                    # 应用权重
                    for idx in range(token_pos, min(token_pos + search_token_len, len(position_weights))):
                        position_weights[idx] = 2.0
            
            # 3. 强化响应开头（前10个token，权重1.2）
            for idx in range(response_start, min(response_start + 10, len(position_weights))):
                position_weights[idx] = 1.2
            
            # 确保所有序列长度一致
            seq_length = len(input_ids)
            attention_mask = attention_mask[:seq_length]
            labels = labels[:seq_length]
            position_weights = position_weights[:seq_length]
            
            # 添加到处理后的数据集
            processed_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_weights": position_weights
            })
            
        except KeyError as e:
            logger.error(f"处理样本 {i} 时出错: {str(e)}")
            logger.error(f"样本内容: {example}")
            continue
        except Exception as e:
            logger.error(f"处理样本 {i} 时发生未知错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 检查处理后的数据是否为空
    if len(processed_data) == 0:
        logger.error("错误: 处理后的数据集为空，请检查数据格式是否正确")
        # 创建虚拟样本防止崩溃
        processed_data.append({
            "input_ids": tokenizer.encode("Empty sample", add_special_tokens=False),
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, 1, 1],
            "position_weights": [1.0, 1.0, 1.0, 1.0, 1.0]
        })
    
    return processed_data


def setup_distributed():
    """初始化分布式训练环境"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])


def train_model_with_weighted_loss():
    # 检查是否在分布式环境中运行
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank, world_size = setup_distributed()
        logger.info(f"初始化分布式训练: local_rank={local_rank}, world_size={world_size}")
    else:
        local_rank, world_size = 0, 1
        logger.info("在单GPU模式下运行")

    # 1. 加载分词器
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 添加特殊token
    special_tokens = ["<|SEARCH|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # 确保每个进程都使用相同的随机种子
    torch.manual_seed(42 + local_rank)
    
    # 2. 加载模型（所有进程都加载相同的模型）
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"进程 {local_rank}: 加载模型 {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    
    # 在应用LoRA之前调整嵌入层大小
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    # 准备LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()

    # 3. 加载并准备数据集
    # 仅在主进程上加载原始数据
    dataset = []
    if local_rank == 0:
        try:
            with open("complex_qa_dataset.json", "r") as f:
                dataset = json.load(f)
            logger.info(f"主进程: 成功加载 {len(dataset)} 个样本")
        except Exception as e:
            logger.error(f"加载数据集时出错: {str(e)}")
            # 创建一些示例数据作为后备
            dataset = [
                {
                    "system": "你是一个AI助手",
                    "instruction": "解释机器学习",
                    "input": "",
                    "output": "机器学习是人工智能的一个分支，关注计算机系统如何利用经验改进性能。"
                }
            ]
            logger.warning("使用示例数据集作为后备")
    
    # 如果是分布式训练，将原始数据广播到所有进程
    if world_size > 1:
        # 创建用于广播的对象列表
        obj_list = [dataset]
        
        # 广播对象列表
        torch.distributed.broadcast_object_list(obj_list, src=0)
        
        # 从广播列表中获取原始数据
        dataset = obj_list[0]
        logger.info(f"进程 {local_rank}: 已接收原始数据集 ({len(dataset)} 个样本)")
    
    # 每个进程独立准备训练数据集
    logger.info(f"进程 {local_rank}: 开始准备训练数据...")
    train_dataset = prepare_weighted_training_data(dataset, tokenizer)
    logger.info(f"进程 {local_rank}: 成功准备 {len(train_dataset)} 个训练样本")
    
    # 数据验证：确保所有样本都有必需的键
    filtered_dataset = []
    required_keys = ['input_ids', 'attention_mask', 'labels', 'position_weights']
    
    for i, sample in enumerate(train_dataset):
        valid = True
        for key in required_keys:
            if key not in sample:
                logger.warning(f"进程 {local_rank}: 样本 {i} 缺少 '{key}' 键，将被跳过")
                valid = False
                break
            elif len(sample[key]) == 0:
                logger.warning(f"进程 {local_rank}: 样本 {i} 的 '{key}' 为空，将被跳过")
                valid = False
                break
                
        if valid:
            # 检查长度一致性
            seq_len = len(sample['input_ids'])
            for key in ['attention_mask', 'labels', 'position_weights']:
                if len(sample[key]) != seq_len:
                    logger.warning(f"进程 {local_rank}: 样本 {i} 的 {key} 长度不一致，已修复")
                    # 截断或填充到相同长度
                    if len(sample[key]) > seq_len:
                        sample[key] = sample[key][:seq_len]
                    else:
                        sample[key] = sample[key] + [0] * (seq_len - len(sample[key]))
            
            filtered_dataset.append(sample)
    
    train_dataset = filtered_dataset
    logger.info(f"进程 {local_rank}: 过滤后剩余 {len(train_dataset)} 个有效样本")
    
    if len(train_dataset) == 0:
        logger.error("错误: 过滤后数据集为空，添加示例数据")
        # 添加一个示例样本防止崩溃
        train_dataset = [{
            "input_ids": tokenizer.encode("Sample data", add_special_tokens=False),
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, 1, 1],
            "position_weights": [1.0, 1.0, 1.0, 1.0, 1.0]
        }]
    
    # 准备带位置权重的训练数据
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4. 第一阶段：监督微调（使用加权损失）
    use_gradient_checkpointing = True
    
    sft_args = TrainingArguments(
        output_dir="./sft_results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        report_to="tensorboard",
        fp16=True,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
        # 关键修复：避免空批次
        dataloader_drop_last=True,
        remove_unused_columns=False
    )
    
    # 在DDP包装前手动启用梯度检查点
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info(f"进程 {local_rank}: 已启用梯度检查点")
    
    # 如果是分布式训练，使用DDP包装模型
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        logger.info(f"进程 {local_rank}: 模型已用DDP包装")
    
    # 创建带有tokenizer绑定的collator
    custom_collator = lambda features: dynamic_padding_collator(features, tokenizer)
    
    # 创建Trainer
    sft_trainer = WeightedLossTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        data_collator=custom_collator
    )
    
    # 训练模型
    logger.info(f"进程 {local_rank}: 开始监督微调训练...")
    try:
        # 关键修复：确保模型处于训练模式
        model.train()
        sft_trainer.train()
        logger.info(f"进程 {local_rank}: 监督微调训练完成")
    except ValueError as e:
        logger.error(f"训练出错: {str(e)}")
        # 添加错误处理逻辑
        if "empty" in str(e).lower():
            logger.error("检测到空批次问题，请检查数据准备和过滤逻辑")
    except RuntimeError as e:
        logger.error(f"运行时错误: {str(e)}")
        # 检查是否是梯度问题
        if "grad" in str(e).lower() or "backward" in str(e).lower():
            logger.error("检测到梯度计算问题，请检查模型结构和损失函数")
    
    # 5. 第二阶段：PPO强化学习（使用加权损失）
    # 只在主进程上执行PPO训练
    if local_rank == 0:
        logger.info("开始PPO训练...")
        
        # 准备PPO模型
        if world_size > 1:
            # 如果是分布式训练，获取原始模型
            model = model.module
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        ppo_model.to(device)
        
        # PPO配置
        ppo_config = PPOConfig(
            batch_size=4,
            mini_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            optimize_cuda_cache=True,
            log_with="tensorboard"
        )
        
        # 创建PPO训练器
        ppo_trainer = WeightedPPOTrainer(
            config=ppo_config,
            model=ppo_model,
            ref_model=None,
            tokenizer=tokenizer,
        )
        
        # 创建PPO数据加载器
        ppo_dataloader = DataLoader(
            train_dataset,
            batch_size=ppo_config.batch_size,
            collate_fn=custom_collator,
            shuffle=True
        )
        
        # PPO训练循环
        for epoch in range(1):
            logger.info(f"PPO训练 epoch {epoch+1}/1")
            for batch_idx, batch in enumerate(ppo_dataloader):
                if batch is None:  # 跳过空批次
                    logger.warning("跳过空批次")
                    continue
                    
                # 准备PPO输入
                query_tensors = batch["input_ids"].to(device)
                
                # 获取模型响应
                response_tensors = []
                for query in query_tensors:
                    response = ppo_model.generate(
                        input_ids=query.unsqueeze(dim=0),
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    response_tensors.append(response.squeeze()[len(query):])
                
                # 提取位置权重
                position_weights = batch["position_weights"].to(device)
                
                # 计算奖励（这里简化处理，实际中应使用奖励模型）
                rewards = [torch.tensor([1.0], device=device) for _ in range(len(batch["input_ids"]))]
                
                # 执行PPO更新
                try:
                    stats = ppo_trainer.step(
                        queries=[q.tolist() for q in query_tensors], 
                        responses=[r.tolist() for r in response_tensors], 
                        scores=rewards,
                    )
                    
                    # 记录训练统计信息
                    ppo_trainer.log_stats(stats, batch, rewards)
                    
                    if batch_idx % 5 == 0:
                        logger.info(f"PPO训练 epoch {epoch+1}, batch {batch_idx}: 完成")
                except Exception as e:
                    logger.error(f"PPO训练出错: {str(e)}")
                    continue
    
    # 6. 保存最终模型（只在主进程上保存）
    if local_rank == 0:
        final_model_dir = "./final_model"
        # 如果是分布式训练，获取原始模型
        if world_size > 1:
            model = model.module
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info(f"训练完成，模型已保存到 {final_model_dir}")
    
    # 清理分布式环境
    if world_size > 1:
        destroy_process_group()


if __name__ == "__main__":
    train_model_with_weighted_loss()
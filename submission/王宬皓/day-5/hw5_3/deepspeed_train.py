# deepspeed_train.py
import torch
import deepspeed
import time
import os
from model import GPT, GPTConfig
from utils import get_batch

def train():
    # 禁用 MPI 并启用 NCCL
    os.environ["USE_MPI"] = "0"
    os.environ["NCCL_DEBUG"] = "INFO"
    
    # 模型配置
    config = GPTConfig(
        vocab_size=10000,
        block_size=256,
        n_layer=12,
        n_head=12,
        n_embd=768
    )
    model = GPT(config)
    
    # DeepSpeed 配置
    ds_config = {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.0003,
                "betas": [0.9, 0.95],
                "eps": 1e-8
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 3,  # 修改为3测试ZeRO-3
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8
        },
        "distributed": {
            "backend": "nccl",
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": False
        },
        "steps_per_print": 100,
        "wall_clock_breakdown": True
    }
    
    # DeepSpeed初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # 创建日志文件
    if model_engine.local_rank == 0:
        log_file = open("logs/deepspeed_log_new.txt", "w")
        log_file.write("step,loss,memory_gb,time_s\n")
    
    # 训练循环
    for step in range(1000):
        start_time = time.time()
        
        # 获取数据
        x, y = get_batch('train', block_size=config.block_size, batch_size=32)
        x, y = x.to(model_engine.local_rank), y.to(model_engine.local_rank)
        
        # 前向传播 - 确保获取标量损失
        _, loss = model_engine(x, targets=y)  # 直接获取标量损失
        
        # 反向传播
        model_engine.backward(loss)
        model_engine.step()
        
        step_time = time.time() - start_time
        
        # 记录指标
        if model_engine.local_rank == 0:
            # 使用PyTorch原生方法获取内存使用（转换为GB）
            # 获取最大内存分配量
            max_mem_allocated = torch.cuda.max_memory_allocated(device=model_engine.local_rank) / 1e9
            # 获取当前内存分配量
            current_mem_allocated = torch.cuda.memory_allocated(device=model_engine.local_rank) / 1e9
            
            # 记录最大内存使用
            log_file.write(f"{step},{loss.item():.4f},{max_mem_allocated:.2f},{step_time:.4f}\n")
            
            if step % 100 == 0:
                print(f"Step {step}: Loss={loss.item():.4f}, "
                      f"Max GPU Mem={max_mem_allocated:.2f}GB, "
                      f"Current GPU Mem={current_mem_allocated:.2f}GB, "
                      f"Time={step_time:.4f}s")
                
                # 重置最大内存统计
                torch.cuda.reset_max_memory_allocated(device=model_engine.local_rank)
    
    if model_engine.local_rank == 0:
        log_file.close()

if __name__ == "__main__":
    train()
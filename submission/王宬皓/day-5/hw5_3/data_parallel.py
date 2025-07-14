import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT, GPTConfig
from utils import get_batch

def ddp_main(rank, world_size):
    # 设置分布式训练环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化分布式进程组
    dist.init_process_group(
        backend="nccl",  # NVIDIA GPU 使用 nccl
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)  # 设置当前GPU
    
    # 日志记录（仅rank0）
    if rank == 0:
        log_file = open("logs/ddp_log_new.txt", "w")
        log_file.write("step,loss,memory_gb,time_s\n")
    
    # 模型配置和初始化
    config = GPTConfig(
        vocab_size=10000, 
        block_size=256, 
        n_layer=12, 
        n_head=12, 
        n_embd=768
    )
    model = GPT(config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])  # 包装为DDP模型
    
    # 优化器
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.0003)
    
    # 训练循环
    for step in range(1000):
        start_time = time.time()
        
        # 获取数据并移动到当前GPU
        x, y = get_batch('train', block_size=config.block_size, batch_size=32)
        x, y = x.to(rank), y.to(rank)
        
        # 前向传播
        logits, loss = ddp_model(x, targets=y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算步骤时间
        step_time = time.time() - start_time
        
        # 记录指标（仅rank0）
        if rank == 0:
            # 获取最大内存使用量（GB）
            mem = torch.cuda.max_memory_allocated(rank) / 1e9
            log_file.write(f"{step},{loss.item():.4f},{mem:.2f},{step_time:.4f}\n")
            
            # 每100步打印进度
            if step % 100 == 0:
                print(f"Step {step}: Loss={loss.item():.4f}, "
                      f"GPU_mem={mem:.2f}GB, Time={step_time:.4f}s")
    
    # 清理
    if rank == 0:
        log_file.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # GPU数量
    
    # 设置主进程环境变量（确保子进程继承）
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 启动多进程训练
    mp.spawn(
        ddp_main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
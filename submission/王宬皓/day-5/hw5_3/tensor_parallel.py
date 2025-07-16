import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
from torch import nn
from model import ParallelGPT, GPTConfig
from utils import get_batch

class TensorParallelWrapper(nn.Module):
    def __init__(self, module, rank, world_size):
        super().__init__()
        self.module = module
        self.rank = rank
        self.world_size = world_size
        
    def forward(self, x):
        # 分割输入数据
        x_chunks = x.chunk(self.world_size, dim=-1)
        x_local = x_chunks[self.rank].contiguous()
        
        # 本地计算
        out_local = self.module(x_local)
        
        # 聚合结果
        out_list = [torch.zeros_like(out_local) for _ in range(self.world_size)]
        dist.all_gather(out_list, out_local)
        return torch.cat(out_list, dim=-1)

def parallel_main(rank, world_size):
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # 创建日志文件 (仅rank0)
    if rank == 0:
        log_file = open("logs/tensor_parallel_log_new.txt", "w")
        log_file.write("step,loss,memory_gb,time_s,comm_gb\n")
    
    # 模型配置 - 使用较小的模型以节省内存
    config = GPTConfig(
        vocab_size=10000,
        block_size=256,
        n_layer=4,  # 减少层数
        n_head=8,   # 减少头数
        n_embd=512  # 减少嵌入维度
    )
    
    # 创建并行模型
    model = ParallelGPT(config, rank, world_size).to(rank)
    
    # 训练代码
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    
    # 通信量记录
    total_comm_bytes = 0
    
    for step in range(1000):
        start_time = time.time()
        
        # 获取数据
        x, y = get_batch('train', block_size=config.block_size, batch_size=16)  # 减少批大小
        x, y = x.to(rank), y.to(rank)
        
        # 前向传播
        logits, loss = model(x, targets=y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度通信（模拟通信量）
        comm_bytes = 0
        for param in model.parameters():
            if param.grad is not None:
                # 模拟梯度通信
                grad_list = [torch.zeros_like(param.grad) for _ in range(world_size)]
                dist.all_gather(grad_list, param.grad)
                comm_bytes += param.grad.element_size() * param.grad.nelement() * (world_size - 1)
        
        # 优化器步骤
        optimizer.step()
        
        # 计算通信量（GB）
        comm_gb = comm_bytes / 1e9
        total_comm_bytes += comm_bytes
        step_time = time.time() - start_time
        
        if rank == 0:
            # 获取内存使用
            max_mem = torch.cuda.max_memory_allocated(rank) / 1e9
            current_mem = torch.cuda.memory_allocated(rank) / 1e9
            
            log_file.write(f"{step},{loss.item():.4f},{max_mem:.2f},{step_time:.4f},{comm_gb:.4f}\n")
            
            if step % 100 == 0:
                print(f"Step {step}: Loss={loss.item():.4f}, "
                      f"Max GPU Mem={max_mem:.2f}GB, "
                      f"Current GPU Mem={current_mem:.2f}GB, "
                      f"Comm={comm_gb:.4f}GB, "
                      f"Time={step_time:.4f}s")
                
                # 重置最大内存统计
                torch.cuda.reset_max_memory_allocated(rank)
    
    if rank == 0:
        print(f"Total communication: {total_comm_bytes/1e9:.2f} GB")
        log_file.close()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    
    # 设置主进程环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    mp.spawn(parallel_main, args=(world_size,), nprocs=world_size, join=True)
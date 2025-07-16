import sys
import torch
import deepspeed
import time
import os
import numpy as np
from model import GPT, GPTConfig
from torch.profiler import profile,ProfilerActivity,record_function
from torch.utils.tensorboard import SummaryWriter
#Zero stage
stage = 3
data_dir = '/inspire/hdd/project/embodied-multimodality/public/jzzhou/nanoGPT/data/shakespeare_char'
def get_batch(split,block_size,batch_size,device):
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def train():
    os.environ["USE_MPI"] = "0"
    config = GPTConfig(
        vocab_size=10000,
        block_size=256,
        n_layer=12,
        n_head=12,
        n_embd=768
    )
    model = GPT(config)
    
    ds_config = {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 2,
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
            "stage": stage,  
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
    master_process = model_engine.local_rank == 0
    
    if master_process:
        writer = SummaryWriter(log_dir=f"./log/summary/ds_stage{stage}")
    device = f'cuda:{model_engine.local_rank}'
    # 训练循环
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1,warmup=1,active=5,repeat=3),
                profile_memory=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=f'./log/profile/ds_stage{stage}',worker_name=f'ds_rand_{model_engine.local_rank}'),
                record_shapes=False,
                with_stack=False
    ) as prof:
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        for step in range(100):
            start_time.record()
            x, y = get_batch('train', block_size=config.block_size, batch_size=32,device=device)
            
            _, loss = model_engine(x, targets=y)  
            
            model_engine.backward(loss)
            model_engine.step()
            
            end_time.record()
            torch.cuda.synchronize()

            time_ms = start_time.elapsed_time(end_time)
            
            max_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            if master_process:
                writer.add_scalar("loss", loss.item(), step)
                writer.add_scalar("gpu/max_mem_MB", max_mem_mb, step)
                writer.add_scalar("time/step_ms", time_ms, step)
            print(f"[Step {step}] loss={loss.item():.4f}, mem={max_mem_mb:.1f}MB, time={time_ms:.2f}ms")
            prof.step()
if __name__ == "__main__":
    train()
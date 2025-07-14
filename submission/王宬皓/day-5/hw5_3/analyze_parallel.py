# analyze_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取所有日志文件
ddp_log = pd.read_csv("logs/ddp_log.txt")
deepspeed_log = pd.read_csv("logs/deepspeed_zero3_log.txt")
tensor_parallel_log = pd.read_csv("logs/tensor_parallel_log.txt")

# 计算平均指标
ddp_avg = {
    "loss": ddp_log["loss"].mean(),
    "memory": ddp_log["memory_gb"].max(),  # 峰值内存
    "time": ddp_log["time_s"].mean(),
    "throughput": 1 / ddp_log["time_s"].mean()
}

zero2_avg = {
    "loss": deepspeed_log["loss"].mean(),
    "memory": deepspeed_log["memory_gb"].max(),
    "time": deepspeed_log["time_s"].mean(),
    "throughput": 1 / deepspeed_log["time_s"].mean()
}

tp_avg = {
    "loss": tensor_parallel_log["loss"].mean(),
    "memory": tensor_parallel_log["memory_gb"].max(),
    "time": tensor_parallel_log["time_s"].mean(),
    "throughput": 1 / tensor_parallel_log["time_s"].mean(),
    "comm": tensor_parallel_log["comm_gb"].mean()
}

# 创建对比数据
data = {
    "Strategy": ["Data Parallel (DDP)", "ZeRO-2 (DeepSpeed)", "Tensor Parallel"],
    "Avg Loss": [ddp_avg["loss"], zero2_avg["loss"], tp_avg["loss"]],
    "Peak Memory (GB)": [ddp_avg["memory"], zero2_avg["memory"], tp_avg["memory"]],
    "Avg Step Time (s)": [ddp_avg["time"], zero2_avg["time"], tp_avg["time"]],
    "Throughput (steps/s)": [ddp_avg["throughput"], zero2_avg["throughput"], tp_avg["throughput"]],
    "Avg Comm (GB/step)": [np.nan, np.nan, tp_avg["comm"]]
}

df = pd.DataFrame(data)

# 保存对比表格
df.to_csv("parallel_strategy_comparison.csv", index=False)
print("Parallel Strategy Comparison:")
print(df)

# 可视化内存对比
plt.figure(figsize=(10, 6))
plt.bar(df['Strategy'], df['Peak Memory (GB)'], color='skyblue')
plt.title('Peak GPU Memory Consumption')
plt.ylabel('GB')
plt.ylim(0, max(df['Peak Memory (GB)']) * 1.2)
plt.savefig('images/memory_comparison.png')

# 可视化时间对比
plt.figure(figsize=(10, 6))
plt.bar(df['Strategy'], df['Avg Step Time (s)'], color='salmon')
plt.title('Average Step Time')
plt.ylabel('Seconds')
plt.ylim(0, max(df['Avg Step Time (s)']) * 1.2)
plt.savefig('images/time_comparison.png')

可视化通信量 (仅张量并行)
plt.figure(figsize=(10, 6))
plt.plot(tensor_parallel_log['step'], tensor_parallel_log['comm_gb'])
plt.title('Tensor Parallel Communication Volume')
plt.xlabel('Training Step')
plt.ylabel('Communication (GB)')
plt.grid(True)
plt.savefig('images/tensor_parallel_comm.png')

# 可视化loss曲线
plt.figure(figsize=(12, 8))
plt.plot(ddp_log['step'], ddp_log['loss'], label='DDP')
plt.plot(deepspeed_log['step'], deepspeed_log['loss'], label='ZeRO-2')
plt.plot(tensor_parallel_log['step'], tensor_parallel_log['loss'], label='Tensor Parallel')
plt.title('Training Loss Comparison')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('images/loss_comparison.png')

print("Analysis completed. Results saved in:")
print("- parallel_strategy_comparison.csv")
print("- memory_comparison.png")
print("- time_comparison.png")
print("- tensor_parallel_comm.png")
print("- loss_comparison.png")
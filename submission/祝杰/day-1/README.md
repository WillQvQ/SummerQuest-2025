# day-1 小结文件

在 Day-1 作业中，我完成了飞书自动化授权流程的实现，并通过修改 doc_viewer.py 成功筛选出包含“刘智耿”的主讲或助教信息，结果重定向输出至日志文件。在 Linux 环境部分，我参考启智平台与 Lanyun 指南完成了 Miniconda 环境配置，安装了 PyTorch、Transformers 与 vllm，并在仅使用一张 GPU 的条件下依次运行了 nvidia-smi、env_checker.py 和 vllm_checker.py，将结果保存为日志文件，完成了基础环境的搭建与验证。
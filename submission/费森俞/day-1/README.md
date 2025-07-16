# 费森俞 Day 1 作业

作业1-飞书自动化的使用
1. 参考 飞书文档自动化-用户身份鉴权创建应用
2. 修改 doc_viewer.py ，查询 “主讲,助教” 中包含 “刘智耿” 的条目
3. 使用重定向方法，将输出结果重定向到文件 hw1.log，然后 commit 

作业2-Linux 环境的使用
1. 使用创智的启智算力平台
2. 创建 Python 环境 vllm，安装 PyTorch, Transformers, 和 vllm
3. 使用8卡环境执行以下操作，将输出结果重定向到文件中，然后 commit 
  (1) 运行 nvidia-smi，检查内容应包含显卡的基本信息（hw2_1.log）
  (2) 运行 env_checker.py ，检查运行结果 (hw2_2.log)
  (3) 运行 vllm_checker.py，检查运行结果 (hw2_3.log)

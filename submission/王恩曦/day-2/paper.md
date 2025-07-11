## LLM推理中的强化学习优化技术

### The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models
- ArXiv链接 : https://arxiv.org/abs/2505.22617
- 关键特点 : 分析了RL训练LLM过程中熵单调下降的动态机制（熵变化由动作概率与logits变化量的协方差驱动），提出Clip-Cov与KL-Cov两种熵控制方法，通过约束高协方差token更新防止熵过早崩溃，在Qwen2.5-32B模型上实现平均6.4%的性能提升。
- 相关技术 : 强化学习（RL）、熵管理、策略梯度算法

### Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning
- ArXiv链接 : https://arxiv.org/abs/2506.01939
- 关键特点 : 发现LLM推理中仅20%高熵token（推理路径决策点）主导性能提升，提出仅优化高熵token的RLVR训练策略，在Qwen3-32B模型上实现AIME'25数学竞赛分数提升11.04分，且模型越大收益越高。
- 相关技术 : 高熵token、强化学习与可验证奖励（RLVR）、策略梯度更新限制

### SEED-GRPO：语义熵增强的GRPO用于不确定性感知的策略优化
- ArXiv链接 : 未找到公开ArXiv链接（搜索结果未提供）
- 关键特点 : 提出SEED-GRPO方法，通过测量LLM对输入提示的语义熵（答案意义多样性）来调节策略更新幅度，对高不确定性问题采取保守更新，在五个数学推理基准测试中达到新SOTA。
- 相关技术 : 组相对策略优化（GRPO）、语义熵、不确定性感知训练

## 通用强化学习方法与机制研究

### Generative Flow Networks as Entropy-Regularized RL
- ArXiv链接 : 未找到公开ArXiv链接（搜索结果仅提及讲座信息）
- 关键特点 : 未明确（搜索结果未提供具体内容）
- 相关技术 : 生成流网络（GFlowNets）、熵正则化强化学习

### DeRL: Decoupled RL for Exploration-Exploitation Trade-off
- ArXiv链接 : 未找到公开ArXiv链接（搜索结果无相关信息）
- 关键特点 : 未明确（搜索结果无相关内容）
- 相关技术 : 未明确

### OPARL: Optimistic and Pessimistic Actors for Robust RL
- ArXiv链接 : 未找到公开ArXiv链接（搜索结果无相关信息）
- 关键特点 : 未明确（搜索结果无相关内容）
- 相关技术 : 未明确

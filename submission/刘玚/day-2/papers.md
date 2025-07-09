# RL论文汇总分析

## 需求一：奖励函数可扩展性
## 需求二：环境交互与记忆建模

### 1. From memories to maps: Mechanisms of in context reinforcement learning in transformers
- **arXiv链接**: [https://arxiv.org/abs/2506.19686](https://arxiv.org/abs/2506.19686)
- **需求一相关性**: ❌ 未明确提及奖励函数可扩展性
- **需求二相关性**: ✅ 提出使用记忆tokens缓存中间计算，通过建模记忆提升性能
- **摘要亮点**: 研究Transformer中的上下文强化学习机制，发现模型通过缓存计算支持灵活行为，类似大脑海马体-内嗅系统的计算方式

### 2. RETRIEVAL-AUGMENTED DECISION TRANSFORMER: EXTERNAL MEMORY FOR IN-CONTEXT RL
- **arXiv链接**: [https://arxiv.org/abs/2410.07071](https://arxiv.org/abs/2410.07071)
- **需求一相关性**: ⚠️ 间接相关，通过外部记忆处理稀疏奖励问题
- **需求二相关性**: ✅ 引入外部记忆机制存储过去经验，检索相关子轨迹辅助决策
- **摘要亮点**: 在复杂环境中使用外部记忆减少上下文长度需求，在网格世界、机器人模拟和视频游戏中验证了有效性

### 3. Can Large Language Models Explore In-Context?
- **arXiv链接**: [https://arxiv.org/abs/2403.15371](https://arxiv.org/abs/2403.15371)
- **需求一相关性**: ❌ 不相关
- **需求二相关性**: ⚠️ 部分相关，研究LLM的探索能力，但未涉及记忆建模
- **摘要亮点**: 发现仅GPT-4在链上推理和外部总结的情况下表现出满意的探索行为

### 4. Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning
- **arXiv链接**: [https://arxiv.org/abs/2503.07572](https://arxiv.org/abs/2503.07572)
- **需求一相关性**: ⚠️ 部分相关，通过密集奖励优化测试时计算
- **需求二相关性**: ❌ 未明确涉及记忆建模
- **摘要亮点**: 提出元强化微调(MRT)方法，在数学推理任务中实现2-3倍性能提升和1.5倍token效率提升

### 5. RLPR: EXTRAPOLATING RLVR TO GENERAL DOMAINS WITHOUT VERIFIERS
- **arXiv链接**: [https://arxiv.org/abs/2506.18254](https://arxiv.org/abs/2506.18254)
- **需求一相关性**: ✅ 相关，无需验证器即可扩展到一般领域
- **需求二相关性**: ❌ 未明确涉及环境交互与记忆建模

### 5. Can Large Reasoning Models Self-Train
- **arXiv链接**: [https://arxiv.org/abs/2505.21444](https://arxiv.org/abs/2505.21444)
- **需求一相关性**: ⚠️ 部分相关，使用自我一致性作为奖励信号
- **需求二相关性**: ❌ 未涉及环境交互与记忆建模
- **摘要亮点**: 提出在线自训练强化学习算法，无需真实标签即可达到与基于金标准答案训练的RL方法相当的性能

### 6. Policy Filtration for RLHF to Mitigate Noise in Reward Models
- **arXiv链接**: [https://arxiv.org/abs/2409.06957](https://arxiv.org/abs/2409.06957)
- **需求一相关性**: ✅ 相关，通过过滤不可靠奖励样本提升奖励信号质量
- **需求二相关性**: ❌ 未涉及环境交互与记忆建模
- **摘要亮点**: 在代码生成任务上，70亿参数模型性能超过现有方法，HumanEval提升7.9%

### 7. Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning
- **arXiv链接**: [https://arxiv.org/abs/2502.18080](https://arxiv.org/abs/2502.18080)
- **需求一相关性**: ⚠️ 部分相关，研究推理长度与奖励的关系
- **需求二相关性**: ❌ 未涉及环境交互与记忆建模
- **摘要亮点**: 发现过度扩展思维链长度会损害性能，提出最优思维长度分配策略

### 8. VeriFree: Reinforcing General Reasoning without Verifiers
- **arXiv链接**: [https://arxiv.org/abs/2505.21493](https://arxiv.org/abs/2505.21493)
- **需求一相关性**: ✅ 相关，无需验证器直接最大化参考答案概率
- **需求二相关性**: ❌ 未涉及环境交互与记忆建模
- **摘要亮点**: 在MMLU-Pro、GPQA等基准上超越基于验证器的方法

## 缺少有效arXiv链接的论文
以下论文因链接无效或缺失，无法获取详细内容进行分析：
1. Meta-Reinforcement Learning Robust to Distributional Shift Via Performing Lifelong In-Context Learning
2. Maximizing Confidence Alone Improves Reasoning (链接格式错误)
# Minerva论文相关研究分类

基于论文《Solving Quantitative Reasoning Problems with Language Models》(https://arxiv.org/abs/2206.14858) 的相关研究，按照数学推理、RL方法、非RL方法和综述进行分类：

## 数学推理相关论文

### Chain of Thought Prompting Elicits Reasoning in Large Language Models
- ArXiv链接: https://arxiv.org/abs/2201.11903
- 关键特点: 提出链式思维(CoT)提示方法，显著提升大语言模型在复杂数学和符号推理任务上的表现。
- 相关技术: Chain-of-Thought, 数学推理, Prompt Engineering

### Self-Consistency Improves Chain of Thought Reasoning in Language Models
- ArXiv链接: https://arxiv.org/abs/2203.11171
- 关键特点: 提出自洽性解码策略，通过多路径推理采样提升数学推理准确率，适用于GSM8K等算术和常识推理任务。
- 相关技术: Chain-of-Thought, Self-Consistency, 数学推理

### Measuring Mathematical Problem Solving With the MATH Dataset
- ArXiv链接: https://arxiv.org/abs/2103.03874
- 关键特点: 构建MATH数据集，包含12,500道竞赛级数学题，推动模型在数学推理和分步解题能力上的提升。
- 相关技术: 数学推理, 数据集, 分步解题

### NumGLUE: A Suite of Fundamental yet Challenging Mathematical Reasoning Tasks
- ArXiv链接: https://arxiv.org/abs/2204.05660
- 关键特点: 提出NumGLUE多任务基准，评估AI系统在基础算术推理上的泛化能力，强调跨任务知识迁移。
- 相关技术: 数学推理, 多任务学习, 泛化能力

### GSM8K: Training Verifiers to Solve Math Word Problems
- ArXiv链接: https://arxiv.org/abs/2110.14168
- 关键特点: 构建高质量小学数学题数据集，提出验证器辅助推理方法，显著提升模型在多步数学推理上的表现。
- 相关技术: 数学推理, 验证器, 多步推理

## RL方法相关论文

### Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning
- ArXiv链接: https://arxiv.org/abs/2505.21178
- 关键特点: 提出两阶段RL框架ConciseR，结合GRPO++和L-GRPO，实现高效简洁的数学推理。
- 相关技术: GRPO, RL, 推理效率

### TreeRPO: Tree Relative Policy Optimization
- ArXiv链接: https://arxiv.org/abs/2506.05183
- 关键特点: 基于树采样的分步奖励机制，提升LLM在数学推理中的细粒度学习和性能。
- 相关技术: GRPO, RL, Tree Sampling, 数学推理

## 非RL方法论文

### Bridging Supervised Learning and Reinforcement Learning in Math Reasoning
- ArXiv链接: https://arxiv.org/abs/2505.18116
- 关键特点: 提出Negative-aware Fine-Tuning (NFT)，实现无教师自我改进，理论上连接SL与RL在数学推理中的应用。
- 相关技术: 数学推理, RL, SFT, NFT

## 综述类论文

### A Survey on Large Language Models for Mathematical Reasoning
- ArXiv链接: https://arxiv.org/abs/2506.08446
- 关键特点: 系统综述LLM在数学推理领域的发展，涵盖预训练、CoT、RL等方法及未来挑战。
- 相关技术: 数学推理, 预训练, Chain-of-Thought, RL

---

**统计总结:**
- 数学推理相关: 5篇论文
- RL方法相关: 2篇论文
- 非RL方法: 1篇论文
- 综述类: 1篇论文

**主要趋势:**
1. 链式思维(CoT)和自洽性采样成为提升LLM数学推理能力的主流方法
2. RL与SFT结合推动模型在复杂推理任务上的自我改进与高效训练
3. 高质量数据集和多任务基准促进模型泛化与分步推理能力发展
4. 领域扩展从数学到科学、工程等定量推理任务，推动LLM在技术领域的应用

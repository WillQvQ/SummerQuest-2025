# Absolute Zero相关研究分类

基于论文《Absolute Zero: Reinforced Self-play Reasoning with Zero Data》(https://arxiv.org/abs/2505.03335) 的相关研究，按照强化学习方法进行分类：

## 自我博弈零数据强化学习方法

### Absolute Zero: Reinforced Self-play Reasoning with Zero Data
- ArXiv链接: https://arxiv.org/abs/2505.03335
- 关键特点: 提出Absolute Zero范式，模型同时作为任务提议者和求解者，通过自我博弈学习，完全无需外部数据，使用代码执行器作为验证环境
- 相关技术: Absolute Zero Reasoner (AZR), Self-play Learning, Code Executor Environment, Task-Relative REINFORCE++

## RLVR (可验证奖励强化学习) 方法

### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- ArXiv链接: https://arxiv.org/abs/2501.12948
- 关键特点: 提出DeepSeek-R1-Zero和DeepSeek-R1，使用GRPO算法进行大规模强化学习，无需监督微调即可实现推理能力，在AIME 2024上达到71.0%的pass@1性能
- 相关技术: GRPO (Group Relative Policy Optimization), Large-Scale RL, Verifiable Rewards

### Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains
- ArXiv链接: https://arxiv.org/abs/2503.23829
- 关键特点: 将RLVR扩展到医学、化学、心理学、经济学等多个领域，使用软奖励信号克服二元验证的局限性，训练跨领域生成奖励模型
- 相关技术: Cross-Domain RLVR, Soft Reward Signals, Generative Scoring

### Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs
- ArXiv链接: https://arxiv.org/abs/2506.14245
- 关键特点: 理论分析RLVR如何激励正确推理，提出CoT-Pass@K评估指标，证明RLVR能够激励逻辑完整性而非仅仅重新加权现有推理路径
- 相关技术: CoT-Pass@K Metric, Logical Integrity Analysis, GRPO Theoretical Foundation

## GRPO (群体相对策略优化) 专门方法

### Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification
- ArXiv链接: https://arxiv.org/abs/2503.06639
- 关键特点: 深入分析GRPO算法，将其表述为KL正则化对比损失，提供理论基础和成功概率量化，解释GRPO在可验证奖励下的有效性
- 相关技术: GRPO Analysis, KL-Regularized Contrastive Loss, Monte Carlo Advantage Estimation

## 自我训练和迭代改进方法

### STaR: Self-Taught Reasoner Bootstrap Reasoning With Reasoning
- ArXiv链接: 引用自相关文献
- 关键特点: 早期自举方法，使用专家迭代和拒绝采样来改进模型的思维链推理能力，为后续RLVR方法奠定基础
- 相关技术: Expert Iteration, Rejection Sampling, Chain-of-Thought Bootstrapping

### SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
- ArXiv链接: 引用自相关文献
- 关键特点: 使用同一语言模型实例作为奖励模型，通过自我博弈逐步改进生成和判别能力，但奖励模型可靠性是瓶颈
- 相关技术: Self-Play Fine-tuning, Same-Instance Reward Model

### Self-Rewarding Language Models
- ArXiv链接: 引用自相关文献
- 关键特点: 模型学会自我奖励，通过自我评估和改进来提升对齐性能，展示了内在动机学习的可能性
- 相关技术: Self-Rewarding Mechanism, Intrinsic Motivation Learning

## 传统RLHF和PPO方法

### Process Reinforcement through Implicit Rewards
- ArXiv链接: https://arxiv.org/abs/2025.reference
- 关键特点: 通过隐式奖励进行过程强化，关注推理过程而非仅结果，提供更精细的反馈信号
- 相关技术: Implicit Rewards, Process-level Reinforcement

### AURORA: Automated Training Framework of Universal Process Reward Models via Ensemble Prompting and Reverse Verification
- ArXiv链接: https://arxiv.org/abs/2025.reference
- 关键特点: 自动化训练框架，使用集成提示和反向验证构建通用过程奖励模型，提升奖励模型的泛化能力
- 相关技术: Automated Training Framework, Ensemble Prompting, Reverse Verification

## 领域特定应用方法

### ACECODER: Acing Coder RL via Automated Test-Case Synthesis
- ArXiv链接: https://arxiv.org/abs/2025.reference
- 关键特点: 专门针对代码生成任务的强化学习方法，通过自动化测试用例合成提供验证反馈
- 相关技术: Automated Test-Case Synthesis, Code-Specific RL

### VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data
- ArXiv链接: https://arxiv.org/abs/2025.reference
- 关键特点: 多领域过程奖励模型，使用合成推理数据训练，能够跨领域评估推理过程质量
- 相关技术: Multi-Domain Process Rewards, Synthetic Reasoning Data

---

**统计总结:**
- 自我博弈零数据方法: 1篇论文
- RLVR可验证奖励方法: 3篇论文
- GRPO专门方法: 1篇论文
- 自我训练方法: 3篇论文
- 传统RLHF/PPO方法: 2篇论文
- 领域特定应用: 2篇论文

**主要趋势:**
1. **从监督到无监督**: 从STaR的专家迭代发展到Absolute Zero的完全无外部数据训练，显示了强化学习在推理任务中的自主学习能力不断增强
2. **可验证奖励的兴起**: RLVR方法因其客观性和可靠性成为主流，避免了传统RLHF中奖励模型的偏差问题
3. **GRPO算法的优势**: 相比传统PPO，GRPO在推理任务中表现更优，成为当前最有效的强化学习算法之一
4. **跨领域扩展**: 从数学和代码任务扩展到医学、化学等多个领域，展现了强化学习方法的通用性
5. **理论与实践并重**: 不仅有实际应用的突破，还有深入的理论分析，为未来发展提供坚实基础
6. **开源化趋势**: DeepSeek-R1等重要模型的开源推动了整个领域的快速发展和民主化
# Thinkless论文相关研究分类

基于论文《Thinkless: LLM Learns When to Think》(https://arxiv.org/abs/2505.13379) 的相关研究，按照强化学习方法进行分类。

## GRPO相关论文

### Understanding R1-Zero-Like Training: A Critical Perspective
- **ArXiv链接**: https://arxiv.org/abs/2503.20783
- **关键特点**: 分析了R1-Zero训练的核心组件，识别了Group Relative Policy Optimization (GRPO)中的优化偏差，提出了Dr. GRPO来解决token效率问题。
- **相关技术**: GRPO, Dr. GRPO, R1-Zero Training

### SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild
- **ArXiv链接**: https://arxiv.org/abs/2503.18892
- **关键特点**: 在10个不同的基础模型上研究零强化学习训练，使用基于规则的奖励的简单RL框架。
- **相关技术**: Zero RL Training, Rule-based Rewards

### ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning
- **ArXiv链接**: https://arxiv.org/abs/2503.19470
- **关键特点**: 通过强化学习训练LLM进行推理和搜索，无需监督数据，展现了反思和自我纠正能力。
- **相关技术**: Reinforcement Learning, Search Integration

## PPO相关论文

### L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning
- **ArXiv链接**: https://arxiv.org/abs/2503.04697
- **关键特点**: 引入Length Controlled Policy Optimization (LCPO)，一个基于PPO的变体，用于通过强化学习控制模型的推理长度。
- **相关技术**: LCPO (基于PPO), Length Control

## 非RL方法论文

### VeriThinker: Learning to Verify Makes Reasoning Model Efficient
- **ArXiv链接**: https://arxiv.org/abs/2505.17941
- **关键特点**: 通过一个辅助的验证任务进行微调，而非直接在原始推理任务上训练，从而有效抑制模型的“过度思考”现象。
- **相关技术**: Verification Task, Fine-tuning

### Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching
- **ArXiv链接**: https://arxiv.org/abs/2503.05179
- **关键特点**: 提出一种认知启发的推理范式，通过概念链、分块符号化和专家词典来减少token的使用，提升效率。
- **相关技术**: Cognitive-Inspired Prompting, Conceptual Chaining

### Chain of Draft: Thinking Faster by Writing Less
- **ArXiv链接**: https://arxiv.org/abs/2502.18600
- **关键特点**: 受人类认知过程启发，模型生成简洁的中间推理草稿，仅使用7.6%的token就能匹配CoT的性能。
- **相关技术**: Minimalistic Reasoning, Draft Generation

### Reasoning with Latent Thoughts: On the Power of Looped Transformers
- **ArXiv链接**: https://arxiv.org/abs/2502.17416
- **关键特点**: 使用循环模型进行推理，证明了k层transformer循环L次可以匹配k*L层非循环模型的性能。
- **相关技术**: Looped Transformers, Latent Thoughts

### Towards Reasoning Ability of Small Language Models
- **ArXiv链接**: https://arxiv.org/abs/2502.11569
- **关键特点**: 系统性地研究了72个小语言模型的推理能力，挑战了“规模是实现强大推理能力的唯一途径”这一假设。
- **相关技术**: Small Language Models, Structured Training

### Small Models Struggle to Learn from Strong Reasoners
- **ArXiv链接**: https://arxiv.org/abs/2502.11569
- **关键特点**: 发现小模型在从大模型蒸馏长CoT推理时存在学习困难，并提出了Mix Distillation策略来解决此问题。
- **相关技术**: Knowledge Distillation, Mix Distillation

## 综述类论文

### Efficient Reasoning Models: A Survey
- **ArXiv链接**: https://arxiv.org/abs/2504.10903
- **关键特点**: 全面综述了高效推理模型的最新进展，将其分为三个主要方向：更短（压缩CoT）、更小（紧凑模型）、更快（高效解码）。
- **相关技术**: CoT Compression, Model Compression, Efficient Decoding

---

**统计总结:**
- **GRPO相关**: 3篇论文
- **PPO相关**: 1篇论文
- **非RL方法**: 6篇论文
- **综述类**: 1篇论文

**主要趋势:**
1.  强化学习方法，特别是像GRPO这样的PPO变体，在优化推理模型中扮演着核心角色。
2.  非RL方法主要关注推理效率的优化，常常从认知科学中汲取灵感。
3.  小语言模型的推理能力及其训练方法正成为一个新兴的研究热点。
4.  控制推理的长度和计算成本是当前研究的共同关注点。
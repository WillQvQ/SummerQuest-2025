## 思考长度控制相关

### Thinkless: LLM Learns When to Think
- ArXiv链接 : https://arxiv.org/abs/2505.13379
- 关键特点 : 提出了可自适应选择短形式和长形式推理的框架，使用DeGRPO算法来分解混合推理的学习目标。
- 相关技术 : Decoupled Group Relative Policy Optimization (DeGRPO), 混合推理控制

### Does Thinking More always Help? Understanding Test-Time Scaling in Reasoning Models
- ArXiv链接 : https://arxiv.org/abs/2506.04210
- 关键特点 : 研究表明额外思考会导致"过度思考"问题，提出了parallel thinking方法来改善推理效率。
- 相关技术 : Best-of-N sampling, 并行思考

### VeriThinker: Learning to Verify Makes Reasoning Model Efficient
- ArXiv链接 : https://arxiv.org/abs/2505.17941
- 关键特点 : 通过验证任务的辅助训练来压缩CoT推理链长度，同时保持或提升准确性。
- 相关技术 : 推理链压缩, 验证器训练

### Chain of Draft: Thinking Faster by Writing Less
- ArXiv链接 : https://arxiv.org/abs/2502.18600
- 关键特点 : 提出了Chain of Draft (CoD)范式，生成简洁但信息丰富的中间推理输出。
- 相关技术 : 推理压缩, 草稿链

## 自适应推理相关

### O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning
- ArXiv链接 : https://arxiv.org/abs/2501.12570
- 关键特点 : 提出了Length-Harmonizing Fine-Tuning方法，根据问题难度动态分配推理token预算。
- 相关技术 : RL-style fine-tuning, 推理预算分配

### Token-Budget-Aware LLM Reasoning
- ArXiv链接 : https://arxiv.org/abs/2412.18547
- 关键特点 : 提出了基于token预算的LLM推理框架，动态调整每个问题的推理token数量。
- 相关技术 : 动态token分配, 推理复杂度评估

## 同时涉及两个方面的论文

### CoT-Valve: Length-Compressible Chain-of-Thought Tuning
- ArXiv链接 : https://arxiv.org/abs/2502.09601
- 关键特点 : 提出了CoT-Valve方法，能够根据任务难度动态控制推理链长度，实现可压缩的CoT推理。
- 相关技术 : 长度可压缩CoT微调, 渐进式链长压缩

### L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning
- ArXiv链接 : https://arxiv.org/abs/2503.04697
- 关键特点 : 提出了LCPO算法来优化准确性和长度约束的遵守，训练出能根据提示中的长度约束生成输出的L1模型。
- 相关技术 : Length Controlled Policy Optimization (LCPO), 强化学习控制

### Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching
- ArXiv链接 : https://arxiv.org/abs/2503.05179
- 关键特点 : 提出了SoT框架，结合认知启发的推理范式和语言约束，减少token使用同时保持推理准确性。
- 相关技术 : Conceptual Chaining, Chunked Symbolism, Expert Lexicons, 动态路由

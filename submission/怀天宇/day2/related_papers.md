# CL‑MoE 持续学习相关研究分类

基于论文 **CL‑MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering** (https://arxiv.org/abs/2503.00413) 的相关研究，按照方法分类如下：

## MoE 架构扩展相关

### LLaVA‑CMoE: Towards Continual Mixture of Experts for Large Vision‑Language Models
- ArXiv链接: https://arxiv.org/abs/2503.21227
- 关键特点: 提出 Probe‑Guided Knowledge Extension (PGKE) 动态判断是否扩展专家层；使用 Probabilistic Task Locator 分层路由避免遗忘；无需回放机制。  
- 相关技术: Continual MoE, PGKE, Hierarchical Routing

### Hierarchical‑Task‑Aware Multi‑modal Mixture of Incremental LoRA Experts for Embodied Continual Learning
- ArXiv链接: https://arxiv.org/abs/2506.04595
- 关键特点: 采用任务级 + token‑级路由选择 LoRA 专家，辅以 SVD 正交训练抑制灾忘，适用于具身智能。  
- 相关技术: LoRA, Task-aware Routing, SVD Orthogonalization

### Continual Cross‑Modal Generalization
- ArXiv链接: https://arxiv.org/abs/2504.00561
- 关键特点: 用 CMoE‑Adapter 将新模态映射到共享离散空间，结合 Pseudo‑Modality Replay 保留旧知识；支持图文、音频、视频增量学习。  
- 相关技术: Adapter, MoE, Replay Mechanism

### Exploiting Mixture-of-Experts Redundancy Unlocks Multimodal Generative Abilities
- ArXiv链接: https://arxiv.org/abs/2503.22517
- 关键特点: 利用低秩 Adapter 分割 FFN 构建专家，采用 MoE 扩模态输入；保留原模型能力同时扩展新模态。  
- 相关技术: Low-rank Adapter, MoE, Multimodal Extension

### PMoE: Progressive Mixture of Experts with Asymmetric Transformer for Continual Learning
- ArXiv链接: https://arxiv.org/abs/2407.21571
- 关键特点: 提出浅层通用专家 + 深层逐步扩展专家策略；使用路由分配新知识；在 TRACE 数据集上表现优。  
- 相关技术: Progressive MoE, Asymmetric Transformer, Router Allocation

## 理论研究类

### Theory on Mixture‑of‑Experts in Continual Learning
- ArXiv链接: https://arxiv.org/abs/2406.16437
- 关键特点: 首个从过参数线性回归角度分析 MoE 在持续学习中作用；证明 MoE 能通过专家多样性减少遗忘；建议冻结路由收敛策略。  
- 相关技术: CL Theory, MoE Convergence, Router Freezing

## Benchmark 与路由机制相关

### MLLM‑CL: Continual Learning for Multimodal Large Language Models
- ArXiv链接: https://arxiv.org/abs/2506.05453
- 关键特点: 提出 Domain & Ability 持续学习基准；采用参数隔离 + 多模态路由机制避免灾忘；在多领域/能力场景表现优。  
- 相关技术: Parameter Isolation, Multimodal Routing, Benchmark

## 原始 CL‑MoE 及工具

### CL‑MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture‑of‑Experts for Continual VQA
- ArXiv链接: https://arxiv.org/abs/2503.00413
- 关键特点: 提出 Dual‑Router MoE（任务 + 实例）与 Momentum MoE 更新策略，增强新增知识吸收与旧知识保持；在 10 个 VQA 任务上达 SOTA 性能。  
- 相关技术: Momentum MoE, Continual VQA, Dual Routing

**统计总结：**

- **MoE 架构扩展相关**: 5 篇  
- **理论研究类**: 1 篇  
- **Benchmark 与路由机制相关**: 1 篇  
- **原始 CL‑MoE**: 1 篇

**主要趋势：**

1. 多篇工作关注基于 MoE 架构的动态扩展与层次路由（如 PGKE、LoRA、Adapter 等），以支持多模态持续学习并减缓灾忘。  
2. 理论分析论文建基于 MoE 收敛性与遗忘机制的数学证明，为架构设计提供指引。  
3. 趋向于设计综合型基准与路由体系（如 MLLM‑CL），推动持续学习模型跨领域能力提升。  
4. CL‑MoE 原文在 VQA 持续学习任务上实现了动量 MoE 的技术突破，成为新的参考模型。

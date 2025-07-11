# DeepResearcher 论文引用分析

## 主论文信息

### DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments
- **ArXiv链接**: https://arxiv.org/abs/2504.03160
- **作者**: Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, Pengfei Liu
- **发表时间**: 2025-04-04
- **关键特点**: 第一个通过强化学习在真实世界环境中进行端到端训练的LLM深度研究代理框架，实现了多代理架构和真实网络交互
- **相关技术**: Reinforcement Learning, Multi-Agent Architecture, Real-world Web Search, End-to-End Training

## 强化学习增强的检索与推理相关论文

### MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents
- **ArXiv链接**: https://arxiv.org/abs/2506.15841
- **关键特点**: 通过端到端强化学习框架使代理在长期多轮任务中以恒定内存运行，在16目标多跳QA任务上性能提升3.5倍，内存使用减少3.7倍
- **相关技术**: End-to-End Reinforcement Learning, Memory Consolidation, Long-Horizon Agents

### Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- **ArXiv链接**: https://arxiv.org/abs/2503.09516
- **关键特点**: 强化学习扩展框架，LLM学习在逐步推理中自主生成多个搜索查询，在七个问答数据集上显著提升性能
- **相关技术**: Reinforcement Learning, Real-time Retrieval, Multi-turn Search Interactions

### StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization
- **ArXiv链接**: https://arxiv.org/abs/2505.15107
- **关键特点**: 基于逐步近端策略优化方法训练搜索LLM的框架，在标准多跳QA基准上显著优于全局奖励基线
- **相关技术**: Step-wise Proximal Policy Optimization, Multi-hop QA, Fine-grained Supervision

### Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning
- **ArXiv链接**: https://arxiv.org/abs/2505.14069
- **关键特点**: 提出ReasonRAG，利用过程级奖励改善训练稳定性和效率，仅用5k训练实例就达到优于90k实例的Search-R1的性能
- **相关技术**: Process-level Rewards, RAG-ProGuide, Process-supervised Reinforcement Learning

## 多模态推理与工具使用相关论文

### Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning
- **ArXiv链接**: https://arxiv.org/abs/2506.13654
- **关键特点**: 用于超长自我中心视频推理的新框架，利用结构化工具链思维(CoTT)过程，将时间覆盖从几小时扩展到一周
- **相关技术**: Chain-of-Tool-Thought, Reinforcement Learning, Ultra-Long Video Reasoning

### Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning
- **ArXiv链接**: https://arxiv.org/abs/2505.16410
- **关键特点**: 基于RL的框架，使LLM能够在逐步推理中自主调用多个外部工具，整合六种工具类型
- **相关技术**: Multi-Tool Collaborative Reasoning, Cold-start Fine-tuning, Hierarchical Reward Design

### R1-Router: Learning to Route Queries Across Knowledge Bases for Step-wise Retrieval-Augmented Reasoning
- **ArXiv链接**: https://arxiv.org/abs/2505.22095
- **关键特点**: 新的MRAG框架，学习基于推理状态动态决定何时何地检索知识，在多模态开放域QA基准上优于基线模型7%
- **相关技术**: Multimodal RAG, Dynamic Knowledge Retrieval, Step-wise Group Relative Policy Optimization

## 开放域问答与搜索增强相关论文

### O2-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering
- **ArXiv链接**: https://arxiv.org/abs/2505.16582
- **关键特点**: 利用强化学习的新搜索代理，有效处理开放域的开放式和封闭式问题，仅使用3B模型就显著超越领先的LLM代理
- **相关技术**: Reinforcement Learning, Open-ended QA, Dynamic Knowledge Acquisition

### WebThinker: Empowering Large Reasoning Models with Deep Research Capability
- **ArXiv链接**: https://arxiv.org/abs/2504.21776
- **关键特点**: 深度研究代理，使LRM能够自主搜索网络、导航网页并在推理过程中起草研究报告
- **相关技术**: Deep Web Explorer, Autonomous Think-Search-and-Draft, RL-based Training

### Search-o1: Agentic Search-Enhanced Large Reasoning Models
- **ArXiv链接**: https://arxiv.org/abs/2501.05366
- **关键特点**: 增强LRM的框架，具有代理检索增强生成机制和文档推理模块，在复杂推理任务中表现强劲
- **相关技术**: Agentic RAG, Reason-in-Documents Module, Dynamic Knowledge Retrieval

## 深度研究与知识分析相关论文

### KnowCoder-V2: Deep Knowledge Analysis
- **ArXiv链接**: https://arxiv.org/abs/2506.06881
- **关键特点**: 提出KDR框架，通过统一代码生成桥接知识组织和推理，在30多个数据集上展示有效性
- **相关技术**: Knowledge Organization, Complex Knowledge Computation, Unified Code Generation

### DeepResearchGym: A Free, Transparent, and Reproducible Evaluation Sandbox for Deep Research
- **ArXiv链接**: https://arxiv.org/abs/2505.19253
- **关键特点**: 开源沙盒，结合可重现的搜索API和严格的评估协议，用于基准测试深度研究系统
- **相关技术**: Reproducible Search API, LLM-as-a-judge Assessment, Controlled Assessment

### OpenResearcher: Unleashing AI for Accelerated Scientific Research
- **ArXiv链接**: https://arxiv.org/abs/2408.06941
- **关键特点**: 基于检索增强生成(RAG)的创新平台，集成LLM和最新的领域特定知识，加速研究过程
- **相关技术**: Retrieval-Augmented Generation, Domain-specific Knowledge, Scientific Literature Analysis

## 其他相关技术

### Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning
- **ArXiv链接**: https://arxiv.org/abs/2505.24332
- **关键特点**: 通过开放网络强化学习实现自适应搜索强度缩放，定义搜索强度缩放(SIS)能力
- **相关技术**: Search Intensity Scaling, Open-Web RL, Adaptive Search Policies

### ZeroSearch: Incentivize the Search Capability of LLMs without Searching
- **ArXiv链接**: https://arxiv.org/abs/2505.04588
- **关键特点**: 新的RL框架，通过训练期间的模拟搜索激励LLM使用真实搜索引擎的能力
- **相关技术**: Simulated Searches, Curriculum-based Rollout, Retrieval Module Training

## 总结

该论文在深度研究代理和强化学习增强的信息检索领域具有重要影响，引发了大量后续研究：

1. **强化学习技术的发展**: 从基础的RL应用发展到更精细的过程级奖励和逐步优化
2. **多模态能力的扩展**: 从文本搜索扩展到视频理解和多工具协作
3. **评估体系的完善**: 建立了可重现、透明的评估基准和沙盒环境
4. **实用性的提升**: 从概念验证发展到实际的科学研究加速平台

该研究为构建下一代通用AI助手奠定了重要基础。

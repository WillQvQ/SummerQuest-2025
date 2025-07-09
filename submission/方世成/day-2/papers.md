# 工具学习相关论文整理

## 基础理论-相关

### Toolformer: Language Models Can Teach Themselves to Use Tools
- ArXiv链接: https://arxiv.org/abs/2302.04761
- 关键特点: 提出了让语言模型自主学习使用外部工具的方法，无需人工标注即可学会API调用
- 相关技术: Self-supervised Learning, API Integration, Tool Learning

### ReAct: Synergizing Reasoning and Acting in Language Models
- ArXiv链接: https://arxiv.org/abs/2210.03629
- 关键特点: 将推理和行动结合，提出了思考-行动-观察的循环框架
- 相关技术: Reasoning-Acting Loop, Chain-of-Thought, Tool Usage

### WebGPT: Browser-assisted question-answering with human feedback
- ArXiv链接: https://arxiv.org/abs/2112.09332
- 关键特点: 展示了LLM使用浏览器工具进行问答的能力，引入人类反馈机制
- 相关技术: Browser Tools, Human Feedback, Web Search

## 框架和方法-相关

### ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
- ArXiv链接: https://arxiv.org/abs/2307.16789
- 关键特点: 构建了大规模工具学习数据集和训练框架，支持16000+真实世界API
- 相关技术: Large-scale Tool Dataset, API Mastery, Tool Training

### HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face
- ArXiv链接: https://arxiv.org/abs/2303.17580
- 关键特点: 提出了协调多个AI模型的框架，将ChatGPT作为控制器调度其他模型
- 相关技术: Multi-model Coordination, Task Decomposition, Model Integration

### TaskWeaver: A Code-First Agent Framework
- ArXiv链接: https://arxiv.org/abs/2311.17541
- 关键特点: 提出了代码优先的Agent框架，支持灵活的工具集成和任务执行
- 相关技术: Code-first Design, Agent Framework, Tool Integration

## 评估和基准-相关

### ToolBench: An Open Platform for Training, Serving, and Evaluating Large Language Model for Tool Learning
- ArXiv链接: https://arxiv.org/abs/2305.16504
- 关键特点: 建立了工具学习领域的综合评估平台和基准测试
- 相关技术: Evaluation Benchmark, Tool Learning Assessment, Performance Metrics

### Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models
- ArXiv链接: https://arxiv.org/abs/2308.00675
- 关键特点: 研究了工具文档对零样本工具使用能力的重要作用
- 相关技术: Zero-shot Learning, Tool Documentation, API Description

## 应用案例-相关

### MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action
- ArXiv链接: https://arxiv.org/abs/2303.11381
- 关键特点: 扩展ReAct框架到多模态场景，支持视觉和语言的联合推理
- 相关技术: Multimodal Reasoning, Visual Tools, Cross-modal Integration

### ViperGPT: Visual Inference via Python Execution for Reasoning
- ArXiv链接: https://arxiv.org/abs/2303.08128
- 关键特点: 通过Python代码执行实现视觉推理，将编程作为视觉问答的工具
- 相关技术: Visual Reasoning, Code Generation, Python Execution

### OpenAgents: An Open Platform for Language Agents in the Wild
- ArXiv链接: https://arxiv.org/abs/2310.10634
- 关键特点: 提供了开放的语言Agent平台，支持多种工具和应用场景
- 相关技术: Agent Platform, Tool Ecosystem, Real-world Applications

## 理论扩展-相关

### Large Language Models as Tool Makers
- ArXiv链接: https://arxiv.org/abs/2305.17126
- 关键特点: 探讨了LLM不仅使用工具，还能创造新工具的能力
- 相关技术: Tool Creation, Self-improvement, Meta-learning

### ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases
- ArXiv链接: https://arxiv.org/abs/2306.05301
- 关键特点: 通过3000个模拟案例构建了通用的工具学习数据集和训练方法
- 相关技术: Simulated Training, Generalized Learning, Tool Simulation

## 综述文献与理论基础

### Tool Learning with Large Language Models: A Survey
- ArXiv链接: https://arxiv.org/abs/2405.17935
- 关键特点: 对工具学习领域进行了全面的综述，从"为什么"和"如何"两个角度分析了工具学习的益处和实现方法
- 相关技术: Survey, Task Planning, Tool Selection, Tool Calling, Response Generation
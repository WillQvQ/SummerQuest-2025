# 大模型Tool Learning相关论文整理

本文档根据指定需求，整理了关于大模型工具学习（Tool Learning）领域的相关论文。

## 通过强化学习提升工具使用能力

### Making Language Models Better Tool Learners with Execution Feedback
- **ArXiv链接**: https://arxiv.org/abs/2305.13068
- **关键特点**: 提出了一个两阶段的框架TRICE。第一阶段通过行为克隆（Behavior Cloning）对模型进行指令微调；第二阶段利用工具执行的反馈，通过强化学习（RL）进一步优化模型。
- **相关技术**: Reinforcement Learning with Execution Feedback (RLEF), Behavior Cloning, Instruction Tuning

### Tool-Augmented Reward Modeling
- **ArXiv链接**: https://arxiv.org/abs/2310.01045
- **关键特点**: 提出了一种工具增强的奖励模型（Tool-augmented Reward Model），该模型能够以自回归的方式对工具使用和推理过程进行综合评估和打分，从而更好地指导强化学习过程。
- **相关技术**: Reward Modeling, Reinforcement Learning, Auto-regressive Scoring

### ToolRL: Reward is All Tool Learning Needs
- **ArXiv链接**: https://arxiv.org/abs/2504.13958
- **关键特点**: 认为工具学习的本质是一个最大化奖励的序列决策问题，并提出了一个完全基于强化学习的框架ToolRL。该框架通过精心设计的奖励函数，从零开始训练模型掌握工具使用。
- **相关技术**: Reinforcement Learning, Sequence Decision Making, Reward Function Design

### ReTool: Reinforcement Learning for Strategic Tool Use in LLMs
- **ArXiv链接**: https://arxiv.org/abs/2504.11536
- **关键特点**: 关注于训练大模型进行策略性的工具使用，而不仅仅是正确的API调用。通过强化学习，模型可以学会何时以及为何使用工具，并能从失败的尝试中恢复。
- **相关技术**: Strategic Tool Use, Reinforcement Learning, Failure Recovery

## 其他提升工具使用能力的方法

### Toolformer: Language Models Can Teach Themselves to Use Tools
- **ArXiv链接**: https://arxiv.org/abs/2302.04761
- **关键特点**: 提出了一种自监督学习方法，让大语言模型（LLM）自己学习使用工具。模型通过生成可能调用API的占位符，执行它们，然后检查返回结果是否有助于预测未来的文本，从而决定是否保留这些API调用。
- **相关技术**: Self-supervised Learning, In-context Learning, API Calls

### Gorilla: Large Language Model Connected with Massive APIs
- **ArXiv链接**: https://arxiv.org/abs/2305.15334
- **关键特点**: 专注于提升大模型准确调用海量API的能力。通过构建一个包含大量API的数据集（APIBench），并对LLaMA模型进行微调，使其在API调用任务上超越了GPT-4。
- **相关技术**: Supervised Fine-Tuning (SFT), API Invocation, Retrieval-aware Training

### CREATOR: Tool Creation for Disentangling Abstract and Concrete Reasoning of Large Language Models
- **ArXiv链接**: https://arxiv.org/abs/2305.14318
- **关键特点**: 提出让大模型不仅能使用工具，还能创造新工具。模型首先将复杂问题分解，然后为子任务生成新的工具（代码函数），最后再调用这些自创的工具来解决问题。
- **相关技术**: Tool Creation, Abstract Reasoning, Program Synthesis

### ReAct: Synergizing Reasoning and Acting in Language Models
- **ArXiv链接**: https://arxiv.org/abs/2210.03629
- **关键特点**: 提出了一种将推理（Reasoning）和行动（Acting）相结合的提示框架。模型交错地生成推理轨迹（思考过程）和具体行动（如调用工具），使得模型决策过程更透明，并能根据工具返回结果动态调整策略。
- **相关技术**: Prompting, Chain-of-Thought, Interleaved Reasoning and Acting

### Self-Training Large Language Models for Tool-Use Without Demonstrations
- **ArXiv链接**: https://arxiv.org/abs/2502.05867
- **关键特点**: 探索在没有人工标注的“黄金”示范数据的情况下，让大模型自学习使用工具。提出了一种自训练方法，利用大模型自身来合成工具使用的轨迹数据，用于模型的微调。
- **相关技术**: Self-Training, Supervised Fine-Tuning, Preference Fine-Tuning

## 工具学习中的幻觉研究

### TOOLVERIFIER: Generalization to New Tools via Self-Verification
- **ArXiv链接**: https://arxiv.org/abs/2402.14158
- **关键特点**: 针对模型在选择工具时可能产生的幻觉（即选择错误的工具），提出了一种自验证方法。模型通过自我提问对比性的问题来区分候选工具之间的细微差别，从而做出更可靠的选择。
- **相关技术**: Self-Verification, Hallucination Reduction, Contrastive Questioning

### CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing
- **ArXiv链接**: https://arxiv.org/abs/2305.11738
- **关键特点**: 提出了一种让模型通过与工具交互来自我纠错的框架。模型会评估自己生成的输出，并通过调用工具（如搜索引擎）来验证信息的准确性，如果发现错误或幻觉，则进行修正。
- **相关技术**: Self-Correction, Fact-Checking, Tool-interactive Critiquing

### Navigating Uncertainty: Optimizing API Dependency for Hallucination Reduction in Closed-Book Question Answering
- **ArXiv链接**: https://arxiv.org/abs/2401.01780
- **关键特点**: 专注于减少因错误的API依赖规划而导致的幻觉。研究了在不确定的情况下，如何优化API的选择和序列，以提高问答系统的可靠性。
- **相关技术**: Hallucination Reduction, API Dependency, Uncertainty Optimization

### ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages
- **ArXiv链接**: https://arxiv.org/abs/2402.10753
- **关键特点**: 虽然主要关注安全性，但其研究与幻觉问题紧密相关。论文揭示了模型在工具学习过程中可能因误解或恶意输入而调用风险工具或产生有害输出，这可以被视为一种行为层面的“幻觉”。
- **相关技术**: Safety, Robustness, Risk Assessment, Error Detection
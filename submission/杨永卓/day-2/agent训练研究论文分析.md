# Agent训练研究论文分析

## Fireact: Toward language agent fine-tuning - 相关

### FireAct: Toward Language Agent Fine-tuning
- ArXiv链接: https://arxiv.org/abs/2310.05915
- 关键特点: 通过fine-tuning提升语言agent能力，提出FireAct方法，在HotpotQA上实现77%性能提升，探索多任务和多种prompting方法的轨迹数据
- 相关技术: Fine-tuning, Multi-task Learning, Agent Trajectory Generation

### AutoMind: Adaptive Knowledgeable Agent for Automated Data Science
- ArXiv链接: https://arxiv.org/abs/2506.10974
- 关键特点: 提出AutoMind框架，结合专家知识库和智能树搜索算法，通过自适应编码策略提升agent性能
- 相关技术: Expert Knowledge Base, Tree Search, Self-adaptive Coding

### Automated Skill Discovery for Language Agents through Exploration and Iterative Feedback
- ArXiv链接: https://arxiv.org/abs/2506.04287
- 关键特点: 提出EXIF框架，通过探索优先策略和迭代反馈循环实现技能自动发现，显著提升agent能力
- 相关技术: Exploration-based Learning, Iterative Feedback, Skill Discovery

### PGPO: Enhancing Agent Reasoning via Pseudocode-style Planning Guided Preference Optimization
- ArXiv链接: https://arxiv.org/abs/2506.01475
- 关键特点: 利用伪代码风格规划引导偏好优化，提升agent推理能力和泛化性
- 相关技术: Pseudocode Planning, Preference Optimization, Reasoning Enhancement

### LAM SIMULATOR: Advancing Data Generation for Large Action Model Training
- ArXiv链接: https://arxiv.org/abs/2506.02298
- 关键特点: 提供在线探索环境和轨迹反馈的综合框架，通过自生成数据集训练实现49.3%性能提升
- 相关技术: Online Exploration, Trajectory Feedback, Self-generated Datasets

## WebDancer: Towards Autonomous Information Seeking Agency - 相关

### WebDancer: Towards Autonomous Information Seeking Agency
- ArXiv链接: https://arxiv.org/abs/2505.22648
- 关键特点: 端到端的信息搜索agent系统，基于ReAct框架，包含数据构建、轨迹采样、监督微调和强化学习四个阶段
- 相关技术: Information Seeking, ReAct Framework, End-to-end Training

### Search-o1: Agentic Search-Enhanced Large Reasoning Models
- ArXiv链接: https://arxiv.org/abs/2501.05366
- 关键特点: 通过检索增强生成机制和Reason-in-Documents模块增强大型推理模型的搜索能力
- 相关技术: Agentic RAG, Reason-in-Documents, Search-Enhanced Reasoning

## Learning From Failure: Integrating Negative Examples - 相关论文

### OWL: Optimized Workforce Learning for General Multi-Agent Assistance
- ArXiv链接: https://arxiv.org/abs/2505.23885
- 关键特点: 提出分层多agent框架，通过优化工作流学习(OWL)提升跨域泛化能力，在GAIA基准上达到69.70%性能
- 相关技术: Hierarchical Multi-agent, Cross-domain Transfer, Reinforcement Learning

### RRO: LLM Agent Optimization Through Rising Reward Trajectories
- ArXiv链接: https://arxiv.org/abs/2505.20737
- 关键特点: 通过奖励上升优化(RRO)关注轨迹中的相对奖励趋势，动态扩展搜索空间，高效捕获高质量数据
- 相关技术: Reward Rising Optimization, Process Supervision, Dynamic Search

### SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution
- ArXiv链接: https://arxiv.org/abs/2505.20732
- 关键特点: 提出逐步进展归因框架，将最终奖励分解为逐步贡献，结合基础信号进行有效agent训练
- 相关技术: Stepwise Progress Attribution, Reward Redistribution, Fine-grained Optimization

### Learning From Failure: Integrating Negative Examples when Fine-tuning Large Language Models as Agents
- ArXiv链接: https://arxiv.org/abs/2402.11651
- 关键特点: 通过质量控制和微调策略利用失败轨迹，在数学推理、多跳问答和策略问答任务上显著提升性能
- 相关技术: Negative Example Learning, Quality Control, Failure Analysis

## SWEET-RL: Training Multi-Turn LLM Agents - 相关论文

### Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment
- ArXiv链接: https://arxiv.org/abs/2505.11821
- 关键特点: 在多轮工具使用场景中，通过转级信用分配改善多轮推理能力，实现100%工具执行成功率和50%精确答案匹配
- 相关技术: Multi-Turn Reasoning, Turn-Level Credit Assignment, Tool Usage

### ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL
- ArXiv链接: https://arxiv.org/abs/2402.19446
- 关键特点: 提出分层强化学习框架，并行运行高级价值RL和低级令牌策略RL，实现约100倍的样本效率提升
- 相关技术: Hierarchical RL, Multi-Turn RL, Actor-Critic Framework

### WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
- ArXiv链接: https://arxiv.org/abs/2411.02337
- 关键特点: 通过自进化在线课程强化学习框架训练高性能web agent，在WebArena-Lite上将成功率从4.8%提升到42.4%
- 相关技术: Self-Evolving Curriculum, Online RL, Web Agent Training

## 通用Agent训练方法相关论文

### Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models
- ArXiv链接: https://arxiv.org/abs/2403.12881
- 关键特点: 通过仔细分解和重新设计训练语料库，Agent-FLAN使Llama2-7B在各种agent评估数据集上性能提升3.5%
- 相关技术: Agent Tuning, Data Design, Format Following

### AgentBank: Towards Generalized LLM Agents via Fine-Tuning on 50000+ Interaction Trajectories
- ArXiv链接: https://arxiv.org/abs/2410.07706
- 关键特点: 包含50k+高质量交互轨迹的最大轨迹调优数据集，覆盖16个任务和5个不同agent技能维度
- 相关技术: Trajectory Tuning, Interaction Data, Agent Skills

### StepAgent: From Novice to Expert: LLM Agent Policy Optimization via Step-wise Reinforcement Learning
- ArXiv链接: https://arxiv.org/abs/2411.03817
- 关键特点: 利用逐步奖励优化agent强化学习过程，通过专家-agent行动比较自动生成中间奖励
- 相关技术: Step-wise RL, Novice-to-Expert, Intermediate Rewards

### AgentGym: Evolving Large Language Model-based Agents across Diverse Environments
- ArXiv链接: https://arxiv.org/abs/2406.04151
- 关键特点: 提供多样化环境和任务的综合框架，包含数据库、基准套件和高质量轨迹，支持agent自我进化
- 相关技术: Multi-Environment Training, Self-Evolution, Agent Ecosystem

## 代码和工具使用Agent训练

### Executable Code Actions Elicit Better LLM Agents
- ArXiv链接: https://arxiv.org/abs/2402.01030
- 关键特点: 提出CodeAct框架，使用可执行Python代码统一agent行动空间，性能提升达20%
- 相关技术: Code Generation, Executable Actions, Unified Action Space

### xLAM: A Family of Large Action Models to Empower AI Agent Systems
- ArXiv链接: https://arxiv.org/abs/2409.03215
- 关键特点: 发布xLAM系列大型行动模型，在Berkeley Function-Calling排行榜上获得第1名，超越GPT-4和Claude-3
- 相关技术: Large Action Models, Function Calling, Tool Use

### ToolRL: Reward is All Tool Learning Needs
- ArXiv链接: https://arxiv.org/abs/2504.13958
- 关键特点: 首次全面研究工具选择和应用任务的奖励设计，通过GRPO训练LLM，实现17%基准提升
- 相关技术: Tool Learning, Reward Design, Policy Optimization

## 多模态和具身Agent训练

### DigiRL: Training In-The-Wild Device-Control Agents with Autonomous Reinforcement Learning
- ArXiv链接: https://arxiv.org/abs/2406.11896
- 关键特点: 通过离线到在线RL的两阶段训练，在Android-in-the-Wild数据集上实现49.5%绝对性能提升
- 相关技术: Offline-to-Online RL, Device Control, GUI Agents

### AutoGLM: Autonomous Foundation Agents for GUIs
- ArXiv链接: https://arxiv.org/abs/2411.00820
- 关键特点: 设计适当的"中间接口"进行GUI控制，通过渐进式训练框架实现自进化在线课程强化学习
- 相关技术: GUI Control, Progressive Training, Self-evolving RL 
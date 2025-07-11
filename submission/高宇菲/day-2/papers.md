### 强化学习算法与训练方法相关
- **VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning**  
  ArXiv链接: https://arxiv.org/abs/2506.17221  
  关键特点: 提出VLN-R1框架，使用大视觉语言模型直接将自中心视频流转换为连续导航动作，采用基于GRPO的训练  
  相关技术: Large Vision-Language Models, GRPO, Time-Decayed Reward  

- **No Free Lunch: Rethinking Internal Feedback for LLM Reasoning**  
  ArXiv链接: https://arxiv.org/abs/2506.17219  
  关键特点: 研究基于内部反馈的强化学习(RLIF)，使用无监督奖励代理如token级熵、轨迹级熵和自确定性  
  相关技术: Reinforcement Learning from Internal Feedback, Unsupervised Reward Proxies  

- **BREAD: Branched Rollouts from Expert Anchors Bridge SFT&RL for Reasoning**  
  ArXiv链接: https://arxiv.org/abs/2506.17211  
  关键特点: 提出BREAD方法，结合SFT和RL阶段，通过分支rollouts和专家引导增强小型语言模型的推理能力  
  相关技术: Group Relative Policy Optimization, Expert Guidance, Branched Rollouts  

- **PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier**  
  ArXiv链接: https://arxiv.org/abs/2506.10406  
  关键特点: 提出PAG框架，通过统一多轮强化学习范式中的策略和验证器角色来实现LLM自我纠正  
  相关技术: Multi-Turn Reinforcement Learning, Self-Correction, Generative Verification  

- **TreeRL: LLM Reinforcement Learning with On-Policy Tree Search**  
  ArXiv链接: https://arxiv.org/abs/2506.11902  
  关键特点: 提出TreeRL框架，直接将在线策略树搜索集成到RL训练中，包含中间监督并消除对单独奖励模型训练的需要  
  相关技术: On-Policy Tree Search, Intermediate Supervision, Cost-Effective Tree Search  


### 数学推理与代码生成相关
- **Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs**  
  ArXiv链接: https://arxiv.org/abs/2506.10630  
  关键特点: 提出Time-R1，采用两阶段强化微调框架，设计了专门针对时间序列预测的细粒度多目标奖励  
  相关技术: Multi-Objective Reward, GRIP (Group-based Relative Importance for Policy Optimization)  

- **ReCUT: Balancing Reasoning Length and Accuracy in LLMs via Stepwise Trails and Preference Optimization**  
  ArXiv链接: https://arxiv.org/abs/2506.10822  
  关键特点: 提出ReCUT方法，通过逐步探索机制和长短切换采样策略来平衡推理轨迹的准确性和长度  
  相关技术: Stepwise Exploration, Preference Optimization, Reasoning Length Optimization  

- **AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy**  
  ArXiv链接: https://arxiv.org/abs/2506.13284  
  关键特点: 研究监督微调(SFT)和强化学习(RL)在开发强推理模型中的协同作用，在数学和代码基准上达到SOTA性能  
  相关技术: SFT-RL Synergy, Math Reasoning, Code Reasoning  

- **SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning**  
  ArXiv链接: https://arxiv.org/abs/2506.08989  
  关键特点: 提出自感知弱点驱动问题合成框架(SwS)，系统识别模型缺陷并利用它们进行问题增强  
  相关技术: Self-aware Weakness Identification, Problem Synthesis, Targeted Training  


### 多模态推理与视觉语言相关
- **GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning**  
  ArXiv链接: https://arxiv.org/abs/2506.16141  
  关键特点: 提出GRPO-CARE，一种一致性感知的RL框架，优化答案正确性和推理连贯性，无需显式监督  
  相关技术: Consistency-Aware RL, Multimodal Reasoning, Two-Tiered Reward  

- **Metis-RISE: RL Incentivizes and SFT Enhances Multimodal Reasoning Model Learning**  
  ArXiv链接: https://arxiv.org/abs/2506.13056  
  关键特点: 提出Metis-RISE，省略初始SFT阶段，直接从RL阶段开始激活模型的潜在推理能力  
  相关技术: RL-First Training, Multimodal Reasoning, Capability Activation  

- **VersaVid-R1: A Versatile Video Understanding and Reasoning Model from Question Answering to Captioning Tasks**  
  ArXiv链接: https://arxiv.org/abs/2506.09079  
  关键特点: 引入DarkEventInfer和MixVidQA数据集，开发首个在Reason-Then-Respond范式下的多功能视频理解推理模型  
  相关技术: Video Understanding, Reason-Then-Respond, Multi-Image Reasoning  


### 自动化系统与工具使用相关
- **AutoMind: Adaptive Knowledgeable Agent for Automated Data Science**  
  ArXiv链接: https://arxiv.org/abs/2506.10974  
  关键特点: 提出AutoMind框架，通过专家知识库、代理知识树搜索算法和自适应编码策略实现自动化数据科学  
  相关技术: Expert Knowledge Base, Agentic Tree Search, Self-Adaptive Coding  

- **Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards**  
  ArXiv链接: https://arxiv.org/abs/2506.11425  
  关键特点: 提出Agent-RLVR框架，通过代理引导机制使RLVR在具有挑战性的代理环境中有效工作  
  相关技术: Agent Guidance, Environment Rewards, Software Engineering Tasks  

- **Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning**  
  ArXiv链接: https://arxiv.org/abs/2506.09033  
  关键特点: 提出Router-R1，基于RL的框架，将多LLM路由和聚合建模为序列决策过程  
  相关技术: Multi-LLM Routing, Sequential Decision Making, Performance-Cost Trade-offs  


### 通用推理与系统优化相关
- **Magistral**  
  ArXiv链接: https://arxiv.org/abs/2506.10910  
  关键特点: 介绍Magistral，Mistral的首个推理模型和可扩展的强化学习管道，采用从头开始的方法  
  相关技术: Scalable RL Pipeline, Reasoning Language Control, Pure RL Training  

- **Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs**  
  ArXiv链接: https://arxiv.org/abs/2506.14245  
  关键特点: 研究可验证奖励强化学习(RLVR)如何隐含地激励基础LLM中的正确推理，提出CoT-Pass@K评估指标  
  相关技术: Verifiable Rewards, Correct Reasoning Incentivization, CoT-Pass@K Metric  

- **SPEED-RL: Faster Training of Reasoning Models via Online Curriculum Learning**  
  ArXiv链接: https://arxiv.org/abs/2506.09016  
  关键特点: 引入SPEED方法，通过选择性地选择中等难度的训练样例来最大化学习效率的自适应在线RL课程  
  相关技术: Online Curriculum Learning, Difficulty Estimation, Training Efficiency  
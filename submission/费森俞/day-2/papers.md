## Task Analysis-相关

### Vision Language Models are In-Context Value Learners
- ArXiv链接 : `https://arxiv.org/abs/2411.04549`
- 关键特点 : 提出了Generative Value Learning (GVL)框架，利用视觉语言模型(VLMs)预测任务进度，通过时间排序问题优化价值估计，支持300+真实世界任务的零样本/少样本预测。
- 相关技术 : Generative Value Learning, Vision-Language Models, Temporal Progress Estimation

### Test-Time Adaptation for Generalizable Task Progress Estimation
- ArXiv链接 : `https://arxiv.org/abs/2506.10085`
- 关键特点 : 提出测试时适应方法，通过梯度基元学习策略提升跨任务进度估计的泛化能力，优于基于自回归视觉语言模型的上下文学习方法。
- 相关技术 : Test-Time Adaptation, Meta-Learning, Progress Estimation

## World Model-相关

### Cosmos-Predict2: Generating Future World States from Videos and Language
- GitHub链接: `https://github.com/nvidia-cosmos/cosmos-predict2`
- 关键特点: 支持从视频和语言输入生成未来世界图像模拟，可用于机器人焊接等工业场景，具备多模态推理能力。
- 相关技术: Video-to-World Generation, Multimodal Forecasting, Industrial Robotics

### LuciBot: Automated Robot Policy Learning from Generated Videos
- ArXiv链接 : `https://arxiv.org/abs/2503.09871`
- 关键特点 : 利用视频生成模型创建任务完成演示视频，提取6D物体位姿序列、2D分割和深度估计等监督信号，用于复杂具身任务的策略训练。
- 相关技术 : Video Generation, World Model, Policy Learning

## VLA-RL-相关

### ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations
- ArXiv链接 : `https://arxiv.org/abs/2505.10911`
- 关键特点 : 提出基于语言指令的奖励函数框架，无需新演示即可微调策略，在模拟和真实机器人上提升样本效率2-5倍。
- 相关技术 : Language-Guided Reward, Reinforcement Learning, Policy Adaptation

### UniVLA: Learning to Act Anywhere with Task-centric Latent Actions
- ArXiv链接 : `https://arxiv.org/abs/2505.06111`
- 关键特点 : 通过任务中心潜行动作模型学习跨 embodiment 视觉语言动作(VLA)策略，利用互联网级视频数据提升泛化能力，性能优于OpenVLA。
- 相关技术 : Vision-Language-Action Models, Latent Action Representation, Cross-Embodiment Learning

## 多需求相关论文

### Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization
- ArXiv链接 : `https://arxiv.org/abs/2505.15660`
- 关键特点 : 提出AGNOSTOS基准评估VLA模型的跨任务零样本泛化能力，提出XICM方法通过上下文演示提升未见任务性能。
- 相关技术 : Vision-Language-Action Models, Cross-task Generalization, In-Context Learning

### TRACE: Tree-based Counterfactual Reasoning for Action Prediction in Sparse Observations
- ArXiv链接: `https://arxiv.org/abs/2503.00761`
- 关键特点: 结合树状推理结构和反事实探索机制，提升 VLM 在稀疏观测条件下的轨迹预测鲁棒性。
- 相关技术: Counterfactual Reasoning, Visual Trajectory Prediction, Sparse Observation
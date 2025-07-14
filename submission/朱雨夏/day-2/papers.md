## 相关论文

### Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
- ArXiv链接 : http://arxiv.org/abs/2412.14803v2
- 关键特点 : 提出了Video Prediction Policy (VPP)，利用视频扩散模型(VDMs)生成包含当前静态信息和预测未来动态的视觉表示；在机器人数据集和互联网人类操作数据上微调预训练视频基础模型以预测更精确的未来；在Calvin ABC-D泛化基准上实现18.6%的相对改进，在复杂现实世界灵巧操作任务中成功率提高31.6%。
- 相关技术 : Video Prediction Policy (VPP), video diffusion models (VDMs), inverse dynamics model, visual representations

### AMPLIFY: Actionless Motion Priors for Robot Learning from Videos
- ArXiv链接 : https://arxiv.org/abs/2506.14198
- 关键特点 : 引入AMPLIFY框架，通过从关键点轨迹中编码视觉动态到紧凑的离散运动令牌来利用大规模视频数据；模块化方法将视觉运动预测与动作推理分离，在丰富的无动作视频上训练前向动力学模型，在有限的动作标记示例上训练逆动力学模型。
- 相关技术 : motion tokens, forward/inverse dynamics models, action-free video learning

### KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills
- ArXiv链接 : https://arxiv.org/abs/2506.12851
- 关键特点 : 提出了基于物理的人形控制框架，通过多步骤运动处理和自适应运动跟踪来掌握高度动态的人类行为(如功夫和舞蹈)；设计了运动提取、过滤、校正和重定向的pipeline，确保最大程度符合物理约束；构建非对称actor-critic框架进行策略训练，在Unitree G1机器人上成功部署。
- 相关技术 : physics-based control, motion retargeting, adaptive motion tracking

### ReSim: Reliable World Simulation for Autonomous Driving
- ArXiv链接 : https://arxiv.org/abs/2506.09981
- 关键特点 : 通过将真实世界驾驶数据与从驾驶模拟器收集的多样化非专家数据丰富，构建在异构语料库上训练的可控世界模型；设计了多种策略有效整合条件信号并提高预测可控性和保真度；引入Video2Reward模块从模拟未来估计奖励，在NAVSIM上提高规划和策略选择性能2%和25%。
- 相关技术 : world simulation, video generator, diffusion transformer, policy evaluation

### BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning
- ArXiv链接 : https://arxiv.org/abs/2506.06072
- 关键特点 : 提出B-spline Encoded Action Sequence Tokenizer (BEAST)，使用B样条将动作序列编码为紧凑的离散或连续令牌；无需单独的令牌器训练，始终生成均匀长度的令牌，通过并行解码实现快速动作序列生成；确保生成相邻段之间无间断的平滑轨迹。
- 相关技术 : B-spline encoding, action tokenization, imitation learning, parallel decoding

### DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation
- ArXiv链接 : https://arxiv.org/abs/2505.21864
- 关键特点 : 提出DexUMI数据收集和策略学习框架，使用人类手作为自然接口将灵巧操作技能转移到各种机器人手；硬件适配通过可穿戴手部外骨骼弥合运动学差距，允许在操作数据收集中直接触觉反馈；软件适配通过高保真机器人手修复替换视频数据中的人类手，弥合视觉差距。
- 相关技术 : dexterous manipulation, human-robot interface, motion adaptation, visual gap bridging

### ChatVLA-2: Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge
- ArXiv链接 : https://arxiv.org/abs/2505.21906
- 关键特点 : 引入ChatVLA-2，一种混合专家VLA模型，结合专门的两阶段训练管道，旨在保留VLM的原始优势同时实现可操作推理；在白板数学问题任务中展示出色的数学推理和OCR能力，能够解释涉及以前未见过的物体的新方向指令。
- 相关技术 : Vision-Language-Action (VLA), mixture-of-expert model, embodied reasoning, pretrained knowledge

### WorldEval: World Model as Real-World Robot Policies Evaluator
- ArXiv链接 : https://arxiv.org/abs/2505.19017
- 关键特点 : 提出Policy2Vec方法，将视频生成模型转变为遵循潜在动作生成机器人视频的世界模拟器；引入WorldEval自动化管道，用于完全在线评估现实世界机器人策略，有效排名各种机器人策略，作为安全检测器防止新开发机器人模型的危险动作。
- 相关技术 : world model, policy evaluation, video generation, safety detection

### Rethinking Agent Design: From Top-Down Workflows to Bottom-Up Skill Evolution
- ArXiv链接 : https://arxiv.org/abs/2505.17673
- 关键特点 : 引入自下而上的智能体范式，通过试错推理机制(探索、反思结果、随时间抽象技能)获取能力；技能一旦获得可快速共享和扩展，实现持续进化而非静态复制；在Slay the Spire和Civilization V中评估，智能体通过原始视觉输入感知并通过鼠标输出动作，完全通过自主交互获取技能。
- 相关技术 : bottom-up agent design, trial-and-reasoning mechanism, skill evolution, autonomous interaction

### Toward Embodied AGI: A Review of Embodied AI and the Road Ahead
- ArXiv链接 : https://arxiv.org/abs/2505.14235
- 关键特点 : 引入跨越五个级别(L1-L5)的Embodied AGI系统分类法；回顾基础阶段(L1-L2)的现有研究和挑战，概述实现更高级别能力(L3-L5)所需的关键组件；提出L3+机器人脑的概念框架，提供技术前景和未来探索基础。
- 相关技术 : Embodied AGI, taxonomy, robotic brain, general intelligence

### Symbolically-Guided Visual Plan Inference from Uncurated Video Data
- ArXiv链接 : https://arxiv.org/abs/2505.08444
- 关键特点 : 提出Vis2Plan，一种由符号引导的高效、可解释的白盒视觉规划框架；从原始、未标记的游戏数据中，利用视觉基础模型自动提取紧凑的任务符号集，构建用于多目标、多阶段规划的高级符号转换图；在真实机器人设置中提供53%更高的总成功率，同时生成视觉计划速度快35倍。
- 相关技术 : visual planning, symbolic guidance, task symbols, transition graph
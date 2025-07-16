## 相关论文

### GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
- ArXiv链接 : http://arxiv.org/abs/2503.14734v2
- 关键特点 : 提出了具有双系统架构的Vision-Language-Action (VLA)模型，视觉语言模块(System 2)通过视觉和语言指令解释环境，扩散 transformer 模块(System 1)实时生成流畅的运动动作；采用异构混合数据(真实机器人轨迹、人类视频和合成生成数据集)进行训练；在多个机器人 embodiment 的标准模拟基准测试中优于最先进的模仿学习基线，并部署在Fourier GR-1人形机器人上完成语言条件下的双手机器人操作任务。
- 相关技术 : Vision-Language-Action (VLA) model, diffusion transformer, heterogeneous data training

### Human2LocoMan: Learning Versatile Quadrupedal Manipulation with Human Pretraining
- ArXiv链接 : https://arxiv.org/abs/2506.16475
- 关键特点 : 提出了一种跨 embodiment 的模仿学习系统，利用从人类和配备多种操作模式的四足机器人LocoMan收集的数据；开发了统一和模块化的人类与机器人观察和动作空间的遥操作和数据收集管道；构建了第一个LocoMan机器人操作数据集，涵盖单双手模式下的各种家庭任务，并辅以相应的人类数据集。
- 相关技术 : cross-embodiment imitation learning, teleoperation pipeline, modularized architecture

### Steering Your Diffusion Policy with Latent Space Reinforcement Learning
- ArXiv链接 : https://arxiv.org/abs/2506.15799
- 关键特点 : 提出了扩散策略强化学习导向(DSRL)方法，通过在潜在噪声空间上运行RL来适应BC训练的策略；具有高度的样本效率，只需对BC策略进行黑盒访问，无需修改基础策略权重，实现了有效的现实世界自主策略改进。
- 相关技术 : diffusion policies, latent space reinforcement learning, sample-efficient adaptation

### AMPLIFY: Actionless Motion Priors for Robot Learning from Videos
- ArXiv链接 : https://arxiv.org/abs/2506.14198
- 关键特点 : 引入了AMPLIFY框架，通过从关键点轨迹中编码视觉动态到紧凑的离散运动令牌来利用大规模视频数据；模块化方法将视觉运动预测与动作推理分离，在丰富的无动作视频上训练前向动力学模型，在有限的动作标记示例上训练逆动力学模型。
- 相关技术 : motion tokens, forward/inverse dynamics models, action-free video learning

### CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding
- ArXiv链接 : https://arxiv.org/abs/2506.13725
- 关键特点 : 引入一致性蒸馏训练在每次迭代中预测多个正确的动作令牌以实现加速，设计混合标签监督以减轻蒸馏过程中的误差累积；提出提前退出解码策略，适度放宽收敛条件，在模拟和现实世界机器人任务中实现了4倍以上的推理加速，同时保持高任务成功率。
- 相关技术 : consistency distillation, early-exit decoding, VLA model acceleration

### LeVERB: Humanoid Whole-Body Control with Latent Vision-Language Instruction
- ArXiv链接 : https://arxiv.org/abs/2506.13751
- 关键特点 : 引入了第一个面向人形全身控制(WBC)的模拟到现实就绪的视觉语言闭环基准，包含10个类别超过150个任务；提出LeVERB框架，高层视觉语言策略从合成渲染的运动学演示中学习潜在动作词汇，低层强化学习WBC策略消耗这些潜在动词生成动力学级命令。
- 相关技术 : humanoid whole-body control, latent action vocabulary, vision-language instruction following

### KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills
- ArXiv链接 : https://arxiv.org/abs/2506.12851
- 关键特点 : 提出了基于物理的人形控制框架，通过多步骤运动处理和自适应运动跟踪来掌握高度动态的人类行为(如功夫和舞蹈)；设计了运动提取、过滤、校正和重定向的 pipeline，确保最大程度符合物理约束；构建非对称 actor-critic 框架进行策略训练，在Unitree G1机器人上成功部署。
- 相关技术 : physics-based control, motion retargeting, adaptive motion tracking

### Gondola: Grounded Vision Language Planning for Generalizable Robotic Manipulation
- ArXiv链接 : https://arxiv.org/abs/2506.11261
- 关键特点 : 引入Gondola模型，一种基于LLM的接地视觉语言规划模型，接收多视图图像和历史计划，生成带有目标对象和位置的交错文本和分割掩码的下一个动作计划；构建了三种类型的数据集(机器人接地规划、多视图指代表达和伪长 horizon 任务数据集)用于训练。
- 相关技术 : grounded vision-language planning, multi-view image input, object segmentation masks

### RationalVLA: A Rational Vision-Language-Action Model with Dual System
- ArXiv链接 : https://arxiv.org/abs/2506.10826
- 关键特点 : 引入RAMA基准，包含超过14,000个样本，挑战模型处理未见过的可执行指令和应被拒绝的有缺陷指令；提出RationalVLA双系统模型，将高层视觉语言模型与低层操作策略通过可学习的潜在空间嵌入集成，能够推理指令、拒绝不可行命令并有效执行操作。
- 相关技术 : instruction reasoning, defective instruction rejection, dual-system architecture

### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- ArXiv链接 : https://arxiv.org/abs/2506.09985
- 关键特点 : 预训练了无动作联合嵌入预测架构V-JEPA 2，在超过100万小时的互联网视频上训练，在运动理解和人类动作预测任务上取得强性能；通过将V-JEPA 2与大型语言模型对齐，在视频问答任务上实现最先进性能；后训练潜在动作条件世界模型V-JEPA 2-AC，使用少于62小时的未标记机器人视频，零样本部署在Franka机械臂上完成拾取和放置任务。
- 相关技术 : self-supervised video learning, joint-embedding-predictive architecture, latent action-conditioned world model
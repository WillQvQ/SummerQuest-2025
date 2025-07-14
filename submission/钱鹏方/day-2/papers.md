# 世界模型 × VLA 操纵任务相关研究分类

基于近期对“如何在视觉-语言-动作(VLA)模型中融合世界模型以解决机器人操纵任务”这一方向的深度调研，下文按照研究关注点进行归类整理，并为每篇论文提供「论文卡片」。

## 一、模型驱动强化学习（Model-Based RL）

### DayDreamer: World Models for Physical Robot Learning
- ArXiv链接: https://arxiv.org/abs/2206.14176
- 关键特点: 在真实机器人上端到端学习潜在世界模型，结合Dreamer算法仅用数小时即可学会行走与抓取。  
- 相关技术: Dreamer, 模型预测控制, 视觉输入强化学习  

### MuZero for Robotics (扩展版)
- ArXiv链接: https://arxiv.org/abs/2302.04798
- 关键特点: 将MuZero的隐式世界模型迁移到机械臂与移动平台，实现无模型先验的规划与控制。  
- 相关技术: MuZero, 隐式动态模型, 在线规划  

## 二、语言指导的世界模型

### Language-Guided World Models (LWM)
- ArXiv链接: https://arxiv.org/abs/2402.01695  
- 关键特点: 语言通过门控网络直接调制状态转移，使单句指令即可重写物理规则；对话式地校正模型。  
- 相关技术: 语言条件动力学, 模型-人协同, 模型基础RL  

### Prompting with the Future
- ArXiv链接: https://arxiv.org/abs/2506.13761
- 关键特点: 在LLM内部嵌入可微分的微型世界模型，先“想象”未来再生成动作-语言混合提示，显著提升多步操纵成功率。  
- 相关技术: 内嵌式世界模型, 视觉-语言-动作提示, 规划-推理一体化  

## 三、物体中心表示（Object-Centric WM）

### Object-Centric World Model for Language-Guided Manipulation
- ArXiv链接: https://arxiv.org/abs/2503.06170  
- 关键特点: 使用Slot Attention提取对象级 latent，并预测各对象在指令条件下的未来状态；较像素级模型高效 3×。  
- 相关技术: Slot Attention, 物体图, 目标驱动规划  

### GNS-VLA: Graph Neural Simulation for Manipulation
- ArXiv链接: https://arxiv.org/abs/2211.10228
- 关键特点: 将Graph Neural Simulator与VLM结合，对刚体与软体混合场景进行语言条件的细粒度预测。  
- 相关技术: 图神经网络, 软体动力学, 多模态融合  

## 四、开放世界与零样本操纵

### Open-World Object Manipulation (MOO)
- ArXiv链接: https://arxiv.org/abs/2303.00905  
- 关键特点: 冻结CLIP获得语义嵌入，策略无监督泛化到未见类别；为后续引入世界模型打下对象语义基础。  
- 相关技术: CLIP, 零样本操纵, 语义嵌入  

### RT-2: Vision-Language-Action Models Transfer Skills to Robots
- ArXiv链接: https://arxiv.org/abs/2307.15818  
- 关键特点: 大规模 VLA 预训练在真实机器人上零样本执行指令；虽未显式引入世界模型，却验证语义知识迁移潜力。  
- 相关技术: 大模型微调, 指令跟随, 开放词汇控制  

## 五、综述与概念化工作

### Robotic World Models – Conceptualization, Review, and Best Practices
- 链接: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1253049/full
- 关键特点: 解析世界模型在表示、耦合度、时空尺度上的设计维度；提供机器人系统工程指南。  
- 相关技术: 世界模型综述, 系统设计, 表示学习  

---

## 总结
世界模型正成为连接视觉-语言理解与物理动作执行的关键桥梁。

### 主要趋势总结
1. **模型驱动强化学习成熟落地**：Dreamer、MuZero 等算法已在真实机器人上展示小时级学习效率。  
2. **语言条件动力学兴起**：直接用自然语言调制或校正世界模型，使机器人可在线吸收人类新规则。  
3. **物体中心表示提高可解释性与泛化**：Slot/Graph-based 表示让模型对多对象、多步骤操纵更稳健。  
4. **开放世界操纵依赖语义先验与世界预测双轮驱动**：结合大规模 VLM 语义与物理模拟，可望实现零样本复杂任务。  
5. **标准基准与安全验证仍缺位**：缺乏统一评价体系与安全红线，是世界模型-VLA 走向工业部署的主要障碍。
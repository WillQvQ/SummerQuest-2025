# AI 生成图像检测研究分类总结

基于论文《Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective》和《CNN-generated images are surprisingly easy to spot... for now》(https://arxiv.org/abs/1912.11035) 的相关研究，我们对 AI 生成图像检测领域进行了系统性的分析和分类。

## 基于图像变换的检测方法

### LoRAX: LoRA eXpandable Networks for Continual Synthetic Image Attribution
- ArXiv链接: https://arxiv.org/abs/2504.08149
- 关键特点: 提出了一种参数高效的类增量算法，可以适应新的生成模型而无需完全重新训练
- 相关技术: LoRA (低秩适应)、持续学习、特征提取器

### LAID: Lightweight AI-Generated Image Detection in Spatial and Spectral Domains
- ArXiv链接: https://arxiv.org/abs/2507.05162 
- 关键特点: 首个评估轻量级神经网络在AIGI检测中表现的框架，在空间、频谱和融合图像域中进行训练
- 相关技术: 轻量级网络、多域检测、对抗鲁棒性

### FreqCross: A Multi-Modal Frequency-Spatial Fusion Network
- ArXiv链接: https://arxiv.org/abs/2507.02995
- 关键特点: 结合空间RGB特征、频域伪影和径向能量分布模式实现鲁棒检测
- 相关技术: 多模态融合、频域分析、径向能量分布

## 基于深度学习的检测方法

### AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection
- ArXiv链接: https://arxiv.org/abs/2503.21210
- 关键特点: 使用大规模多模态专家模型进行可解释的AI生成图像取证
- 相关技术: 多模态大语言模型、可解释AI、取证分析

### Co-Spy: Combining Semantic and Pixel Features
- ArXiv链接: https://arxiv.org/abs/2503.18286
- 关键特点: 结合语义特征和伪影特征，提高检测泛化能力
- 相关技术: 自适应特征集成、语义分析、像素级分析

### All Patches Matter: Panoptic Patch Learning
- ArXiv链接: https://arxiv.org/abs/2504.01396
- 关键特点: 通过全局图像块学习提升检测效果，避免对特定图像块的过度依赖
- 相关技术: 图像块替换、对比学习、分布式伪影分析

## 基于频域分析的检测方法

### Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Detection
- ArXiv链接: https://arxiv.org/abs/2507.02398
- 关键特点: 利用像素级时间频率分析捕获时序不一致性
- 相关技术: 时频分析、注意力机制、时空特征融合

### ReStraV: Perceptual Straightening for Video Detection
- ArXiv链接: https://arxiv.org/abs/2507.00583
- 关键特点: 分析神经表示域中的轨迹曲率和步进距离
- 相关技术: 自监督视觉Transformer、时序几何分析

## 基于多模态融合的检测方法

### LATTE: Latent Trajectory Embedding
- ArXiv链接: https://arxiv.org/abs/2507.03054
- 关键特点: 模型潜在嵌入的演化轨迹，捕获区分真假图像的细微模式
- 相关技术: 潜在轨迹分析、特征细化、多步降噪

### Adaptive Low-level Experts Injection (ALEI)
- ArXiv链接: https://arxiv.org/abs/2504.00463
- 关键特点: 利用不同低级信息的互补性提升检测泛化能力
- 相关技术: 适应性低级专家、特征交叉注意力、动态特征选择

---

**统计总结：**
- 基于图像变换的方法：3篇论文
- 基于深度学习的方法：3篇论文
- 基于频域分析的方法：2篇论文
- 基于多模态融合的方法：2篇论文

**主要趋势：**
1. 多模态融合成为提升检测效果的主流方向，通过结合不同层次的特征提高泛化能力
2. 轻量级模型和高效算法越来越受关注，以适应实际应用场景的需求
3. 可解释性和鲁棒性成为研究重点，以应对不断进化的生成技术
4. 频域分析与深度学习方法的结合展现出良好的检测效果
5. 持续学习能力成为新的研究方向，以应对新型生成模型的快速发展

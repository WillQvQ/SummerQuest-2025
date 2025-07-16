# 基于 "DyCoke" (2411.15024) 的相关论文分析

## 需求1相关: Token Pruning 或 Token Merging

### FastVID: Dynamic Density Pruning for Fast Video Large Language Models
- ArXiv链接 : https://arxiv.org/abs/2503.11187
- 关键特点 : 提出了一种动态密度剪枝策略 (Dynamic Density Pruning)，通过将视频分割成有序片段并保留必要的视觉信息来显著减少计算开销。
- 相关技术 : Token Pruning, Density-based Pruning

### LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models
- ArXiv链接 : https://arxiv.org/abs/2403.15388
- 关键特点 : 提出 PruMerge，一种自适应的视觉令牌缩减策略。利用视觉编码器中的稀疏性选择关键令牌，然后通过聚类和合并来增强信息内容。
- 相关技术 : Token Pruning, Token Merging, Sparsity

### Token Merging: Your ViT But Faster
- ArXiv链接 : https://arxiv.org/abs/2210.09461
- 关键特点 : 提出 ToMe (Token Merging)，一种无需训练即可提升 ViT 模型吞吐量的方法，通过逐步合并相似的令牌来实现。
- 相关技术 : Token Merging

### HoliTom: Holistic Token Merging for Fast Video Large Language Models
- ArXiv链接 : https://arxiv.org/abs/2505.21334
- 关键特点 : 提出 HoliTom 框架，结合了在 LLM 外部进行全局时空合并和在内部进行基于相似度的合并，以全面减少冗余。
- 相关技术 : Token Merging, Spatio-temporal Merging, Outer-LLM & Inner-LLM Pruning

### FlexSelect: Flexible Token Selection for Efficient Long Video Understanding
- ArXiv链接 : https://arxiv.org/abs/2506.00993
- 关键特点 : 提出 FlexSelect，一种灵活高效的令牌选择策略，利用跨模态注意力模式来识别和保留语义最相关的内容。
- 相关技术 : Token Selection, Cross-modal Attention

### TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model
- ArXiv链接 : https://arxiv.org/abs/2503.18278
- 关键特点 : 将令牌剪枝制定为一个优化问题，而非依赖注意力分数，以准确识别重要视觉令牌，并与 FlashAttention 兼容。
- 相关技术 : Token Pruning, Optimization Framework, FlashAttention

### ST3: Accelerating Multimodal Large Language Model by Spatial-Temporal Visual Token Trimming
- ArXiv链接 : https://arxiv.org/abs/2412.20105
- 关键特点 : 提出 ST3 框架，包含渐进式视觉令牌剪枝 (PVTP) 和视觉令牌退火 (VTA) 两种策略，以在不同层和解码步骤中动态减少令牌数量。
- 相关技术 : Token Trimming (Pruning), Progressive Pruning

### TempMe: Video Temporal Token Merging for Efficient Text-Video Retrieval
- ArXiv链接 : https://arxiv.org/abs/2409.01156
- 关键特点 : 提出 TempMe，一个参数高效的文本-视频检索架构，通过渐进式的多粒度框架，逐步合并相邻片段以减少时空冗余。
- 相关技术 : Temporal Token Merging, Parameter-efficient Fine-tuning

## 需求2相关: 提升推理速度 或 减少内存占用

### Plug-and-Play 1.x-Bit KV Cache Quantization for Video Large Language Models
- ArXiv链接 : https://arxiv.org/abs/2503.16257
- 关键特点 : 提出 VidKV，一种即插即用的 KV Cache 量化方法，可将 KV Cache 压缩到低于2-bit，显著减少内存占用。
- 相关技术 : KV Cache Quantization, Mixed-precision Quantization

### LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
- ArXiv链接 : https://arxiv.org/abs/2406.18139
- 关键特点 : 提出 LOOK-M，一种无需微调的方法，通过文本优先的策略和 KV 对合并来高效压缩多模态 KV Cache，提升解码速度。
- 相关技术 : KV Cache Optimization, Prompt Prefill

### LazyLLM: Dynamic Token Pruning for Efficient Long Context LLM Inference
- ArXiv链接 : https://arxiv.org/abs/2407.14057
- 关键特点 : 提出 LazyLLM，一种动态令牌剪枝方法，在预填充和解码阶段选择性地计算重要令牌的 KV，从而加速长文本推理。
- 相关技术 : Dynamic Token Pruning, KV Cache Computation

### PB-LLM: Partially Binarized Large Language Models
- ArXiv链接 : https://arxiv.org/abs/2310.00034
- 关键特点 : 提出一种部分二值化的方法，通过仅对一小部分显著权重保留高精度，其余进行二值化，来实现极低比特的 LLM 压缩。
- 相关技术 : Model Binarization, Quantization

### FiLA-Video: Spatio-Temporal Compression for Fine-Grained Long Video Understanding
- ArXiv链接 : https://arxiv.org/abs/2504.20384
- 关键特点 : 提出 FiLA-Video 框架，利用轻量级的动态权重多帧融合策略，自适应地将多帧融合成单一表示，以减少计算成本。
- 相关技术 : Feature Compression, Frame Fusion

### FastV: An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models
- ArXiv链接 : https://arxiv.org/abs/2403.06764
- 关键特点 : 提出 FastV，一种即插即用的方法，通过在浅层学习自适应注意力模式，并在深层剪枝视觉令牌来优化 LVLM 的推理效率。
- 相关技术 : Inference Acceleration, Token Pruning

## 需求3相关: 无需训练 (Training-Free)

### HoliTom: Holistic Token Merging for Fast Video Large Language Models
- ArXiv链接 : https://arxiv.org/abs/2505.21334
- 关键特点 : 明确提到其为一个新颖的、无需训练 (training-free) 的整体性令牌合并框架。
- 相关技术 : Training-Free, Token Merging

### TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model
- ArXiv链接 : https://arxiv.org/abs/2503.18278
- 关键特点 : 强调其方法无需额外的训练或微调即可实现高效剪枝。
- 相关技术 : Training-Free, Token Pruning

### Plug-and-Play 1.x-Bit KV Cache Quantization for Video Large Language Models
- ArXiv链接 : https://arxiv.org/abs/2503.16257
- 关键特点 : 论文标题和摘要都明确指出这是一种即插即用 (plug-and-play) 的 KV Cache 量化方法。
- 相关技术 : Plug-and-Play, KV Cache Quantization

### LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
- ArXiv链接 : https://arxiv.org/abs/2406.18139
- 关键特点 : 摘要中明确提到这是一种开创性的、无需微调 (fine-tuning-free) 的方法。
- 相关技术 : Fine-tuning-Free, KV Cache Optimization

### FastV: An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models
- ArXiv链接 : https://arxiv.org/abs/2403.06764
- 关键特点 : 标题和摘要都强调其为一种即插即用 (plug-and-play) 的方法，用于优化推理效率。
- 相关技术 : Plug-and-Play, Inference Acceleration

### ST3: Accelerating Multimodal Large Language Model by Spatial-Temporal Visual Token Trimming
- ArXiv链接 : https://arxiv.org/abs/2412.20105
- 关键特点 : 摘要中提到其框架旨在无需再训练 (without retraining) 的情况下加速 MLLM 推理。
- 相关技术 : Without Retraining, Token Trimming

## 需求1 & 需求2 都相关的论文

### FastVID: Dynamic Density Pruning for Fast Video Large Language Models
- ArXiv链接 : https://arxiv.org/abs/2503.11187
- 关键特点 : 同时满足“令牌剪枝”和“提升速度”两个需求，其核心目标就是通过剪枝来加速推理。
- 相关技术 : Token Pruning, Inference Acceleration

### LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models
- ArXiv链接 : https://arxiv.org/abs/2403.15388
- 关键特点 : 明确提出其令牌缩减 (Pruning/Merging) 的目标是为了实现高效的 LMM (Efficient LMMs)。
- 相关技术 : Token Pruning, Token Merging, Efficiency

### Token Merging: Your ViT But Faster
- ArXiv链接 : https://arxiv.org/abs/2210.09461
- 关键特点 : 标题就直白地说明了其令牌合并技术是为了让 ViT "更快" (Faster)。
- 相关技术 : Token Merging, Throughput Increase

### FlexSelect: Flexible Token Selection for Efficient Long Video Understanding
- ArXiv链接 : https://arxiv.org/abs/2506.00993
- 关键特点 : 其令牌选择策略的目标是为了实现高效的长视频理解，并取得了显著的速度提升。
- 相关技术 : Token Selection, Efficiency, Speed-up

### TopV: Compatible Token Pruning with Inference Time Optimization for Fast and Low-Memory Multimodal Vision Language Model
- ArXiv链接 : https://arxiv.org/abs/2503.18278
- 关键特点 : 标题就包含了所有关键点：令牌剪枝 (Token Pruning)，快速 (Fast)，低内存 (Low-Memory)。
- 相关技术 : Token Pruning, Inference Optimization, Speed & Memory

### ST3: Accelerating Multimodal Large Language Model by Spatial-Temporal Visual Token Trimming
- ArXiv链接 : https://arxiv.org/abs/2412.20105
- 关键特点 : 标题即点明其目标是通过令牌裁剪 (Trimming) 来加速 (Accelerating) MLLM。
- 相关技术 : Token Trimming, Acceleration, KV Cache Memory Reduction

## 需求1 & 需求2 & 需求3 都相关的论文

### HoliTom: Holistic Token Merging for Fast Video Large Language Models
- ArXiv链接 : https://arxiv.org/abs/2505.21334
- 关键特点 : 是一种无需训练的令牌合并方法，旨在提升 Video LLM 的速度。
- 相关技术 : Training-Free, Token Merging, Efficiency

### FastV: An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models
- ArXiv链接 : https://arxiv.org/abs/2403.06764
- 关键特点 : 是一种即插即用的令牌剪枝方法，用于加速 LVLM 推理。
- 相关技术 : Plug-and-Play, Token Pruning, Inference Acceleration
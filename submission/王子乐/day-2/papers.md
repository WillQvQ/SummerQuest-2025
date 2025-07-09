# 论文筛选摘要

**关键词**: `KV-cache`  
**生成时间**: 2025-07-09 17:30:49  
**总论文数**: 9

- 源论文: **1 篇**
- 引用源论文的文章: **6 篇**
- 源论文引用的文章: **2 篇**

## 按相关性评分统计

- 高相关性 (≥0.7): **9 篇**
- 中相关性 (0.5–0.7): **0 篇**
- 低相关性 (<0.5): **0 篇**

## 源论文列表

#### 1. Efficient Memory Management for Large Language Model Serving with
  PagedAttention
- **ArXiv ID**: `2309.06180`  
- **URL**: [http://arxiv.org/abs/2309.06180v1](http://arxiv.org/abs/2309.06180v1)  
- **评分**: `0.90`  
- **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang 等  
- **摘要**: High throughput serving of large language models \(LLMs\) requires batching
sufficiently many requests at a time. However, existing systems struggle
because the key-value cache \(KV cache\) memory for eac...  
- **相关性说明**: 论文直接提到了'key-value cache \(KV cache\)'，并且主要贡献是提出了PagedAttention算法和vLLM系统，用于高效管理KV缓存内存。论文对KV缓存技术进行了改进，通过虚拟内存和分页技术减少了内存浪费，并实现了KV缓存的灵活共享。这些内容与关键字'KV-cache'高度相关。  

## 所有论文（按相关性排序）

#### 1. [Efficient Memory Management for Large Language Model Serving with
  PagedAttention](http://arxiv.org/abs/2309.06180v1)
- **类型**: `源论文`  
- **ArXiv ID**: `2309.06180`  
- **评分**: `0.90`  
- **作者**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang 等  
- **摘要**: High throughput serving of large language models \(LLMs\) requires batching
sufficiently many requests at a time. However, existing systems struggle
because the key-value cache \(KV cache\) memory for eac...  
- **相关性说明**: 论文直接提到了'key-value cache \(KV cache\)'，并且主要贡献是提出了PagedAttention算法和vLLM系统，用于高效管理KV缓存内存。论文对KV缓存技术进行了改进，通过虚拟内存和分页技术减少了内存浪费，并实现了KV缓存的灵活共享。这些内容与关键字'KV-cache'高度相关。  

#### 2. [LazyEviction: Lagged KV Eviction with Attention Pattern Observation for Efficient Long Reasoning](http://arxiv.org/abs/2506.15969v1)
- **类型**: `引用源论文`  
- **ArXiv ID**: `2506.15969`  
- **评分**: `0.90`  
- **关联源论文**: `2309.06180`  
- **作者**: Haoyue Zhang, Hualei Zhang, Xiaosong Ma 等  
- **摘要**: Large Language Models \(LLMs\) exhibit enhanced reasoning capabilities by employing Chain-of-Thought \(CoT\). However, the extended reasoning sequences introduce significant GPU memory overhead due to inc...  
- **相关性说明**: 论文直接提到并使用了与'KV-cache'相关的技术。摘要中明确提到'key-value \(KV\) cache size'以及'KV cache compression methods'，表明论文的核心内容围绕KV-cache展开。论文的主要贡献是提出了一种名为LazyEviction的滞后KV驱逐框架，旨在减少KV内存占用同时保持推理性能，这直接与KV-cache技术相关。此外，论文对现有的KV-cache技术进行了改进，通过观察注意力模式并提出新的驱逐策略（如Recurrence Interval Tracking和Maximum Recurrence Interval-Centric Eviction Policy），扩展了KV-cache的应用场景和效率。  

#### 3. [eLLM: Elastic Memory Management Framework for Efficient LLM Serving](http://arxiv.org/abs/2506.15155v1)
- **类型**: `引用源论文`  
- **ArXiv ID**: `2506.15155`  
- **评分**: `0.90`  
- **关联源论文**: `2309.06180`  
- **作者**: Jiale Xu, Rui Zhang, Yi Xiong 等  
- **摘要**: Large Language Models are increasingly being deployed in datacenters. Serving these models requires careful memory management, as their memory usage includes static weights, dynamic activations, and k...  
- **相关性说明**: 论文直接提到了KV-cache，并讨论了其在LLM服务中的内存管理问题。论文的主要贡献是提出了eLLM框架，该框架通过虚拟张量抽象和弹性内存机制改进了KV-cache的管理，从而提高了内存利用率和解码吞吐量。论文对KV-cache相关的技术进行了显著的改进和扩展，特别是在动态内存管理和碎片化缓解方面。  

#### 4. [Draft-based Approximate Inference for LLMs](http://arxiv.org/abs/2506.08373v1)
- **类型**: `引用源论文`  
- **ArXiv ID**: `2506.08373`  
- **评分**: `0.90`  
- **关联源论文**: `2309.06180`  
- **作者**: Kevin Galim, Ethan Ewer, Wonjun Kang 等  
- **摘要**: Optimizing inference for long-context Large Language Models \(LLMs\) is increasingly important due to the quadratic compute and linear memory complexity of Transformers. Existing approximation methods, ...  
- **相关性说明**: 论文直接提到了关键字相关的技术，如'key-value \(KV\) cache dropping'，并且论文的主要贡献之一是通过使用小型草案模型更准确地预测KV对的重要性，从而改进KV缓存丢弃技术。论文还提出了SpecKV，这是一种利用草案输出来评估KV对重要性的方法，这直接扩展了KV-cache技术的应用。因此，论文不仅直接使用了KV-cache技术，还对其进行了改进和扩展。  

#### 5. [MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](http://arxiv.org/abs/2506.07533v1)
- **类型**: `引用源论文`  
- **ArXiv ID**: `2506.07533`  
- **评分**: `0.90`  
- **关联源论文**: `2309.06180`  
- **作者**: Wei Tao, Haocheng Lu, Xiaoyang Qu 等  
- **摘要**: One of the primary challenges in optimizing large language models \(LLMs\) for long-context inference lies in the high memory consumption of the Key-Value \(KV\) cache. Existing approaches, such as quanti...  
- **相关性说明**: 论文直接提到了关键字相关的技术，即Key-Value \(KV\) cache，并且主要贡献是针对KV cache的内存消耗问题提出了混合精度量化方法MoQAE。论文不仅讨论了现有KV cache量化方法的局限性，还通过引入量化感知专家的混合方法、轻量级路由器微调过程以及路由冻结和共享机制，对KV cache相关技术进行了显著的改进和扩展。因此，论文与关键字'KV-cache'高度相关。  

#### 6. [SwiftSpec: Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding](http://arxiv.org/abs/2506.11309v1)
- **类型**: `引用源论文`  
- **ArXiv ID**: `2506.11309`  
- **评分**: `0.80`  
- **关联源论文**: `2309.06180`  
- **作者**: Ziyi Zhang, Ziheng Jiang, Chengquan Jiang 等  
- **摘要**: Low-latency decoding for large language models \(LLMs\) is crucial for applications like chatbots and code assistants, yet generating long outputs remains slow in single-query settings. Prior work on sp...  
- **相关性说明**: 论文直接提到了'KV-cache inconsistencies'，并提出了'tree-aware KV cache management'作为解决方案之一。这表明论文不仅使用了KV-cache相关的技术，还对其进行了改进以解决现有问题。论文的主要贡献虽然集中在异步推测解码和低延迟生成，但KV-cache的管理是其实现目标的关键技术之一。因此，论文与关键字'KV-cache'高度相关。  

#### 7. [Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse](http://arxiv.org/abs/2506.07639v1)
- **类型**: `引用源论文`  
- **ArXiv ID**: `2506.07639`  
- **评分**: `0.70`  
- **关联源论文**: `2309.06180`  
- **作者**: Zhekai Duan, Yuan Zhang, Shikai Geng 等  
- **摘要**: Embodied Chain-of-Thought \(ECoT\) reasoning enhances vision-language-action \(VLA\) models by improving performance and interpretability through intermediate reasoning steps. However, its sequential auto...  
- **相关性说明**: 论文虽然没有直接提到'KV-cache'这一关键字，但其提出的方法涉及缓存和重用中间推理步骤（'cache and reuse high-level reasoning across timesteps'），这与KV-cache技术的核心思想（缓存中间计算结果以加速推理）高度相关。论文的主要贡献是通过缓存和并行化技术加速推理，这与KV-cache的目标一致。因此，尽管术语不同，论文的技术与KV-cache相关，评分为0.7。  

#### 8. [High-throughput Generative Inference of Large Language Models with a Single GPU](http://arxiv.org/abs/2303.06865v2)
- **类型**: `被源论文引用`  
- **ArXiv ID**: `2303.06865`  
- **评分**: `0.70`  
- **关联源论文**: `2309.06180`  
- **作者**: Ying Sheng, Lianmin Zheng, Binhang Yuan 等  
- **摘要**: The high computational and memory requirements of large language model \(LLM\) inference make it feasible only with multiple high-end accelerators. Motivated by the emerging demand for latency-insensiti...  
- **相关性说明**: 论文虽然没有直接提到'KV-cache'这一术语，但在摘要中提到了'attention cache'，这与'KV-cache'（键值缓存）是相关的技术概念。论文的主要贡献是通过压缩权重和注意力缓存（attention cache）到4位来实现高效的大语言模型推理，这与'KV-cache'的优化方向一致。论文对注意力缓存进行了改进（压缩技术），可以视为对'KV-cache'相关技术的扩展。因此，论文与'KV-cache'具有一定的相关性。  

#### 9. [Efficiently Scaling Transformer Inference](http://arxiv.org/abs/2211.05102v1)
- **类型**: `被源论文引用`  
- **ArXiv ID**: `2211.05102`  
- **评分**: `0.70`  
- **关联源论文**: `2309.06180`  
- **作者**: Reiner Pope, Sholto Douglas, Aakanksha Chowdhery 等  
- **摘要**: We study the problem of efficient generative inference for Transformer models, in one of its most challenging settings: large deep models, with tight latency targets and long sequence lengths. Better ...  
- **相关性说明**: 论文虽然没有直接提到'KV-cache'这一术语，但其讨论的多查询注意力机制（multiquery attention）与KV-cache技术密切相关。多查询注意力通过让多个查询头共享单个键/值头，减少了内存需求，这与KV-cache优化内存使用的目标一致。此外，论文还讨论了在长上下文长度下的推理效率优化，这也是KV-cache技术常见的应用场景。因此，尽管未明确提及，论文内容与KV-cache技术高度相关。  


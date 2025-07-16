# 课后习题 & 答案

作者：殷林琪

## hw5_1

Transformer 的 FFN模块由两个线性层组成，结构为：

```python
Input (X) ∈ ℝ^{B × S × d}
↓
Linear1: W1 ∈ ℝ^{d × 4d}      → GeLU
↓
Linear2: W2 ∈ ℝ^{4d × d}
↓
Output (Z) ∈ ℝ^{B × S × d}
```

模型配置为：

Batch size: $B = 8$
Sequence length: $S = 128$
Hidden dim: $d = 1024$
使用TP，并行度为 $P = 4$

**1 并行策略与张量 shape**

**1.1 Linear1 使用 Column Parallel：**

**1.1.1**
每个 rank 上的权重矩阵 $W_1^{(i)}$ shape:

$$W_1^{(i)} \in \mathbb{R}^{1024 \times (4096 / 4)} = \mathbb{R}^{1024 \times 1024}$$

**1.1.2**
每个 rank 输入的张量 $X$ 的 shape 是:

$$X \in \mathbb{R}^{8 \times 128 \times 1024} \quad \text{（所有 rank 都需要完整 X）}$$

**1.1.3**
每个 rank 本地输出 $Y_i$的 shape 是？最终如何得到完整的 $Y$:

本地输出： $$Y_i = X \cdot W_1^{(i)} \in \mathbb{R}^{8 \times 128 \times 1024}$$

完整： $$Y = \text{Concat}(Y_0, Y_1, Y_2, Y_3) \in \mathbb{R}^{8 \times 128 \times 4096}$$


**1.2 Linear2 使用 Row Parallel：**

**1.2.1**
每个 rank 上的权重矩阵 $W_2^{(i)}$ shape 是:

$$W_2^{(i)} \in \mathbb{R}^{(4096 / 4) \times 1024} = \mathbb{R}^{1024 \times 1024}$$

**1.2.2**
每个 rank 接受输入的张量 的 shape 是: 

$$ Y_i \in \mathbb{R}^{8 \times 128 \times 1024}$$

**1.2.3**
每个 rank 本地输出 $Z_i$的 shape 是？最终如何得到完整的 $Z$:

$$ Z_i = Y_i \cdot W_2^{(i)} \in \mathbb{R}^{8 \times 128 \times 1024}$$

$$Z = \sum_{i=1}^4 Z_i \in \mathbb{R}^{8 \times 128 \times 1024}$$

**2 通信分析**

**2.1 对于 Linear1：**

**2.1.1**
前向传播是否需要通信？通信操作是？通信量是多少？ 

所有 rank 使用完整输入 $X$，本地计算 $Y_i$，无需通信

最终拼接 $Y = \text{Concat}(Y_i)$：通常在下一层做 RowParallel 时分片使用，没有显式 all-gather 通信，通信量为0

**2.1.2**
反向传播时，计算 $\partial L / \partial X$ 是否需要通信？说明原因。

每个 rank 的 $W_1^{(i)}$和 $\partial L / \partial Y_i$仅覆盖部分输出

为计算全量 $\partial L / \partial X$，需： $$\text{All-Reduce} \left( \sum_{i=1}^P \partial L / \partial X_i \right)$$

通信方法为 All-Reduce，通信量为： $B \times S \times d = 8 \times 128 \times 1024 = 1\,048\,576$个 float（约 4MB）

**2.2 对于 Linear2：**

**2.2.1**
前向传播是否需要通信？通信操作是？通信量是多少？ 

每个 rank 仅有部分输入 $Y_i$，需要聚合所有本地输出 $Z_i$，最终输出： $$\text{All-Reduce} \left( \sum_{i=1}^P Z_i \right)$$

通信方法为 All-Reduce，通信量为 $8 \times 128 \times 1024 = 1\,048\,576$

**2.2.2**
反向传播时，计算 $\partial L / \partial X$ 是否需要通信？说明原因。

每个 rank 局部计算本地反向传播，可保持输入分片，无需通信，通信量为0

**3 如果两层都使用 Row Parallel，会产生哪些额外通信？两层都使用 Column Parallel 会带来什么问题？**

都使用 Row Parallel：

Linear1 输出是根据输入的分片输出，但 Linear2 需要全量输入，则需要 All-Gather，导致增加显式通信开销

都使用 Column Parallel：

Linear1 输出是拼接的，Linear2 的输入却被分片，Linear2 前向需显式 All-to-All 或 Scatter，反向也需要通信

# hw5_2

**1. 总训练时间与理想时间**

1.1 总执行时间 $T_{\text{total}}$

$$T_{\text{total}} = (t_f + t_b) \cdot (m + p - 1) = (2 + 4) \cdot (8 + 4 - 1) = 6 \cdot 11 = 66\,\text{ms}$$



1.2 理想时间 $T_{\text{ideal}}$

$$T_{\text{ideal}} = m \cdot (t_f + t_b) = 8 \cdot 6 = 48\,\text{ms}
$$



1.3 Bubble Ratio

$$\text{Bubble Ratio} = \frac{T_{\text{total}}}{T_{\text{ideal}}} = \frac{66}{48} = 1.375$$



**2. 增加 microbatch 数为 $m = 16$ 后的变化**

新总时间：

$$T_{\text{total}}' = (2 + 4) \cdot (16 + 4 - 1) = 6 \cdot 19 = 114\,\text{ms}$$



新理想时间：

$$T_{\text{ideal}}' = 16 \cdot 6 =96\,\text{ms}$$


新 Bubble Ratio：

$$\text{Bubble Ratio}' = \frac{114}{96} = 1.1875$$


微批数增加后，流水线填充更充分，bubble ratio 减小；更高 microbatch 数可 提升流水线利用率


**3. Gpipe 与 1F1B 调度策略对比**


| 调度策略      | 特点                                          | 对流水线效率影响                               |
| --------- | ------------------------------------------- | -------------------------------------- |
| **Gpipe** | 先全部 forward，后全部 backward                    | 存在 pipeline 灌入和清空阶段，bubble 多，GPU 空闲时间多 |
| **1F1B**  | 每执行 1 个 microbatch 的 forward，立即执行其 backward | 前后传播交错，提高 GPU 利用率，bubble 少             |



**1F1B 优化点**

* 更细粒度流水线调度，减少 idle 时间

* 更适合 microbatch 数较少时使用

* 更稳定的计算资源利用率




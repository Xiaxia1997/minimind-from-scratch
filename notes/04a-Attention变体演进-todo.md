# 04a - Attention 变体演进：面试向横向对比

## 为什么需要这篇？

04-Attention 讲的是 MiniMind 实现的 MHA → GQA 代码线。本章把所有 Attention 变体串起来做横向对比，包括 Flash Attention 的 IO 优化（§五）。面试里 Attention 变体经常被当作独立话题问：

> "你了解哪些 Attention 变体？各有什么优缺点？"
> "DeepSeek 的 MLA 是怎么回事？和 GQA 有什么区别？"
> "Gated DeltaNet 听说过吗？为什么 Qwen3.5 用它？"

这篇不重复已有章节的内容，而是**用一个统一视角把所有变体串起来**：标准 Attention 有三个成本中心——**计算量、显存占用、注意力噪声**，每个变体都是在某个成本中心上做优化。理解了这一点，面试时不管怎么问，都能把任何一个变体归位到这张图里。

## 统一视角：标准 Attention 的三个成本中心

```
标准 Self-Attention 的代价拆解：

成本中心 1：计算量 O(n²)
  ├─ 不改数学，优化 IO        → Flash Attention           （§五）
  ├─ 稀疏化，只算重要 token    → NSA                       （本章 §二）
  └─ 彻底换成 O(n) 的机制     → Gated DeltaNet / Mamba    （本章 §二 / 19 章）

成本中心 2：KV Cache 显存占用（推理时）
  ├─ 减少 KV head 数量        → GQA                       （04 章）
  ├─ 低秩压缩 KV              → MLA                       （本章 §一）
  └─ 优化显存管理（不改 Attn） → PagedAttention            （本章 §四）

成本中心 3：注意力噪声（softmax 给无关 token 分配概率）
  └─ 两组 softmax 做差分消噪   → Differential Attention    （本章 §二）
```

下面逐个展开。对已有章节覆盖的内容（GQA、Flash Attention、Mamba），只在横向对比表里出现，不重复讲。

---

## 一、MLA（Multi-head Latent Attention）

> DeepSeek-V2, 2024.05 | 19 章 §2.3 已有概述，这里补完整数学

### 1.1 GQA 的天花板

GQA（04 章讲过）通过让多个 Q head 共享 KV head 来缩小 KV Cache，但压缩比有上限——再怎么共享，你最少也要保留一组完整的 KV head。

以 DeepSeek-V2（236B MoE, 128 heads, head_dim=128）为例：

| 方案 | 每层每 token 缓存量 | 压缩比 |
|------|-------------------|--------|
| 标准 MHA | $2 \times 128 \times 128 = 32768$ | 1× |
| GQA（8 KV heads） | $2 \times 8 \times 128 = 2048$ | 16× |
| 极限 MQA（1 KV head） | $2 \times 1 \times 128 = 256$ | 128× 但效果差 |

DeepSeek 想要 **GQA 级别的效果 + MQA 级别的压缩**。怎么做？换一个压缩维度。

### 1.2 核心思想：低秩压缩 KV

GQA 在 **head 数量维度** 压缩（减少 KV head），MLA 在 **特征维度** 压缩（降维到低秩空间）：

```
传统 MHA:
  X → W_k → K  [seq, n_heads × d_head]  ← 缓存这个（高维）
  X → W_v → V  [seq, n_heads × d_head]  ← 缓存这个（高维）

MLA:
  X → W_down → c_kv  [seq, d_c]         ← 只缓存这个（低维潜在向量）
  推理时恢复：c_kv → W_up_k → K         ← 需要时实时投影回来
              c_kv → W_up_v → V
```

和 LoRA 完全同一个思想——**先降维再升维，中间的低维瓶颈就是压缩后的表示**。

DeepSeek-V2 实际数字：$d_c = 512$，每层每 token 缓存 $512 + 64 = 576$ 个数（加上 64 维 RoPE 部分），压缩比约 **57 倍**。

### 1.3 RoPE 冲突与 Decoupled RoPE

直觉上想给 c_kv 加 RoPE——但 c_kv 是压缩的，在压缩空间里旋转和在原始空间里旋转含义不同。

**解法**：K 拆成两部分：

$$K = [\underbrace{K_{\text{nope}}}_{\text{从 } c_{kv} \text{ 恢复，不带位置}},\; \underbrace{K_{\text{rope}}}_{\text{单独投影，加 RoPE}}]$$

- $K_{\text{nope}}$：走压缩路线省显存（从 c_kv 恢复，512 维）
- $K_{\text{rope}}$：单独从 X 投影出来，加 RoPE，也要缓存（只有 64 维，很小）

### 1.4 推理时的矩阵吸收

朴素做法：先从 c_kv 恢复 K（一次矩阵乘法），再算 $Q \cdot K^T$。但展开看：

$$Q \cdot K^T = Q \cdot (c_{kv} \cdot W_{up\_k})^T = \underbrace{(Q \cdot W_{up\_k}^T)}_{Q'} \cdot c_{kv}^T$$

把 $W_{up\_k}^T$ **吸收到 Q 的投影里**，预计算 $Q' = Q \cdot W_{up\_k}^T$，推理时直接 $Q' \cdot c_{kv}^T$。根本不需要恢复 K。V 侧同理（吸收到输出投影里）。

这个技巧把 MLA 从"多了两次矩阵乘法"变成"和标准 Attention 计算量相当"。

### 1.5 面试话术（30s）

> MLA 是 DeepSeek-V2 提出的 KV Cache 压缩方案。和 GQA 不同，GQA 是在 head 数量维度压缩，MLA 是在特征维度压缩——把每层每 token 的 K/V 投影到一个低维的潜在向量，推理时只缓存这个低维向量。DeepSeek-V2 实现了约 57 倍压缩，效果和 MHA 相当。难点是 RoPE 和压缩冲突，解决方案是 Decoupled RoPE：K 拆两部分，不带位置的走压缩路线，带位置的单独一小块直接缓存。推理时还有一个矩阵吸收技巧——把恢复矩阵吸收到 Q 的投影里，避免额外计算。

---

---

## 二、Delta Attention / Gated DeltaNet

### 2.1 线性注意力：为什么能从 O(n²) 降到 O(n)

#### Situation：O(n²) 的瓶颈具体卡在哪一步

标准 Attention 的公式是 $\text{softmax}(QK^T / \sqrt{d}) \cdot V$。

Flash Attention（§五）通过 IO 优化减小了常数，但复杂度仍然是 $O(n^2)$——对于 128K、1M 级别的长序列，**$O(n^2)$ 本身就不可接受**：

| seq_len | Attention 矩阵元素数（每 head） | 相对 1K |
|---------|-------------------------------|---------|
| 1,024 | 1M | 1× |
| 8,192 | 67M | 64× |
| 128,000 | 16.4B | 16,384× |

$O(n^2)$ 具体来自哪一步？

$$\underbrace{Q}_{[n, d]} \times \underbrace{K^T}_{[d, n]} = \underbrace{A}_{[n, n]}$$

矩阵乘法 $[a, b] \times [b, c]$ 的运算量 = $a \times b \times c$（结果有 $a \times c$ 个元素，每个需要 $b$ 次乘加）。代入 $Q \cdot K^T$：$n \times d \times n = n^2 d$。$d$（head_dim，MiniMind 为 96）是固定常数，所以复杂度写作 $O(n^2)$。用具体数字感受：$n = 512$ 时约 25M 次运算；$n = 128K$ 时约 1.6T 次——序列长度翻 256 倍，计算量翻 $256^2 = 65536$ 倍。

$Q$ 和 $K^T$ 的矩阵乘法产生了 $n \times n$ 的注意力矩阵。紧接着 softmax 沿每一行做归一化——计算分母 $\sum_j \exp(q_i^T k_j)$ 必须看到整行所有 $n$ 个值——进一步锁死了"必须先算出完整 $n \times n$ 矩阵"。

#### Task：降到 O(n)，同时尽量保持建模能力

面对 $O(n^2)$，有两条路：

1. **稀疏化**：不算完整的 $n \times n$，只算重要位置 → NSA（§三）
2. **换计算顺序**：彻底避免 $n \times n$ 矩阵的产生 → **线性注意力**（本节）

#### Action：核函数替换 softmax → 结合律解锁 → 先算 KV 再算 Q

**第一步：发现结合律的机会。**

矩阵乘法有结合律：$(AB)C = A(BC)$。标准 Attention 的核心计算是三个矩阵的连乘 $(QK^T)V$。如果能去掉中间的 softmax，就可以换计算顺序：

$$\underbrace{(QK^T)}_{[n,n]} V \;\;=\;\; Q\underbrace{(K^TV)}_{[d,d]}$$

- **左边**：先算 $QK^T$ → 产生 $[n, n]$ 矩阵 → 再乘 $V$ → $O(n^2 d)$
- **右边**：先算 $K^TV$ → 只是 $[d, d]$ 矩阵 → 再用 $Q$ 查 → $O(nd^2)$

$d$ 是固定的（MiniMind 中 head\_dim=96），$n$ 是序列长度。当 $d \ll n$，右边远快于左边：

| | 标准 Attention | 线性注意力 |
|--|---------------|-----------|
| 中间矩阵大小 | $QK^T$ → $[n, n]$ | $K^TV$ → $[d, d]$ |
| n=768, d=96 | 768² = 590K | 96² = 9.2K |
| n=128K, d=96 | 128K² = **16.4B** | 96² = **9.2K** |
| 复杂度 | $O(n^2 d)$ | $O(nd^2)$ |

**第二步：搞清楚 softmax 为什么挡住了结合律。**

$$\text{softmax}(QK^T) \cdot V$$

softmax 作用在 $QK^T$ 的每一行上。要算分母 $\sum_j \exp(q_i^T k_j)$，必须先算出这一行全部 $n$ 个元素——$Q$ 和 $K$ 被 softmax **耦合**在一起了。没法跳过 $QK^T$ 直接去算 $K^TV$。

**第三步：用核函数 $\phi$ 替换 softmax，解除耦合。**

标准 Attention 的矩阵形式：

$$\text{Attn} = \text{softmax}(Q K^T) \cdot V$$

softmax 作用在 $QK^T$ 这个**点积的结果**上——$\exp$ 包裹住了 $Q$ 和 $K$ 的点积，两者在 $\exp$ 里面纠缠在一起，没法拆开，结合律用不上。

核函数 $\phi$ 的做法不同：**对 $Q$ 和 $K$ 各自独立做变换，再算点积**。这样相似度矩阵变成 $\phi(Q) \cdot \phi(K)^T$，$Q$ 和 $K$ 天然分离：

和标准 Attention 一样分两步——先算归一化的权重矩阵，再乘 $V$：

**第一步：算权重矩阵（归一化）**

$$A = \frac{\phi(Q) \cdot \phi(K)^T}{\sum_j \phi(k_j) \text{ 的行求和}}$$

每个元素 $A_{ij} = \phi(q_i)^T \phi(k_j) \;/\; \sum_l \phi(q_i)^T \phi(k_l)$，归一化后每行和为 1。

**第二步：加权求和**

$$O = A \cdot V$$

因为 $\phi(Q)$ 和 $\phi(K)$ 是独立的矩阵，$A \cdot V$ 展开后可以用**结合律**换计算顺序——和第一步完全一样：

$$O = \phi(Q) \cdot \underbrace{(\phi(K)^T V)}_{[d, d]} \;/\; \text{归一化分母}$$

先算右边的 $\phi(K)^T V = [d, d]$（只算一次），再用 $\phi(Q)$ 去查——中间矩阵是 $[d, d]$ 而不是 $[n, n]$，复杂度从 $O(n^2 d)$ 降到 $O(nd^2)$。

**核函数 $\phi$ 的要求和常见选择：**

$\phi$ 需要满足两个条件：
1. **逐元素作用在 $Q$ 和 $K$ 上**——这是能拆开的前提
2. **输出非负**——注意力权重不能为负

满足这两个条件的函数都可以用。常见选择是 $\phi(x) = \text{ELU}(x) + 1$。ELU 的定义：$x > 0$ 时等于 $x$，$x \leq 0$ 时等于 $e^x - 1$（和 ReLU 类似，但负半轴用指数平滑而非直接截断为 0）。加 1 保证输出 $> 0$。选 ELU 而不是 ReLU，是因为 ReLU 把负值直接变 0，信息丢失太多；ELU+1 在负半轴还有非零输出，保留更多信息。

**核函数的代价：失去了 softmax 的"尖锐化"能力。** softmax 有 $\exp$，大的相似度被指数放大、小的被压到接近 0——几个最相关的 token 拿走大部分权重，注意力能"聚焦"。核函数 $\phi$ 只是 ELU+1 这种线性量级的变换，没有指数放大效果，权重分布很"平"——所有 token 都分到差不多的权重，注意力没法聚焦。这是线性注意力效果不如标准 Attention 的根本原因，也是后面 Delta Rule / Gated DeltaNet 要弥补的（2.2 节）。

**第四步："记忆矩阵"视角——推理时不需要 KV Cache。**

第三步得到了线性注意力的批量公式（训练时一次算所有 token）：

$$O = \phi(Q) \cdot (\phi(K)^T V) \;/\; \text{归一化}$$

推理时是自回归的——每次只来一个新 token。标准 Attention 每来一个新 token，要和所有历史 K 做点积（$O(t)$），还要存所有历史的 K、V（KV Cache 随序列增长）。

线性注意力可以避开这两个问题。核心观察：$\phi(K)^T V$ 这个 $[d, d]$ 矩阵可以**逐步累加**。

定义记忆矩阵 $S$（忽略归一化，不影响核心思路）：

$$S_t = \phi(K_{1:t})^T V_{1:t} \quad [d, d]$$

$S_t$ 就是"前 $t$ 个 token 的 $\phi(K)^T V$"。矩阵乘法 $[d, n] \times [n, d]$ 可以展开为 $n$ 个外积之和——左矩阵的每一列乘以右矩阵对应的行，得到 $n$ 个 $[d, d]$ 矩阵，加起来：

```
φ(K)^T · V = φ(k₁)·v₁^T + φ(k₂)·v₂^T + ... + φ(kₙ)·vₙ^T
             ~~~~~~~~~~   ~~~~~~~~~~         ~~~~~~~~~~
             [d,1]×[1,d]  [d,1]×[1,d]       [d,1]×[1,d]
             = [d,d]      = [d,d]            = [d,d]
```

用具体数字（$d=2$，3 个 token）验证：

```
φ(k₁)=[1,0], v₁=[3,4]    φ(k₂)=[0,1], v₂=[5,6]    φ(k₃)=[1,1], v₃=[7,8]

φ(k₁)·v₁^T = [1]·[3,4] = [3, 4]     φ(k₂)·v₂^T = [0, 0]    φ(k₃)·v₃^T = [7, 8]
              [0]          [0, 0]                    [5, 6]                   [7, 8]

三个加起来 = [10, 12]     ← 和直接算 φ(K)^T · V 结果一致
             [12, 14]
```

之所以展开成逐项求和，是因为**递推需要逐项累加**——$S_t$ 比 $S_{t-1}$ 只多了第 $t$ 项：

$$S_t = S_{t-1} + \phi(k_t) \cdot v_t^T$$

查询时，用 $\phi(q_t)$ 直接从 $S_t$ 取结果：

$$o_t = \phi(q_t)^T \cdot S_t \quad [1, d]$$

两行公式总结：

$$\boxed{S_t = S_{t-1} + \phi(k_t) \cdot v_t^T, \qquad o_t = \phi(q_t)^T \cdot S_t}$$

$S \in \mathbb{R}^{d \times d}$ 是一个固定大小的**记忆矩阵**。每个新 token 把自己的 key-value 外积加入 $S$（$[d,d]$ 矩阵加法），查询时做一次 $[1,d] \times [d,d]$ 的矩阵乘法。不管序列多长，$S$ 的大小始终是 $d \times d$，不需要 KV Cache。

**为什么标准 Attention 做不到？** softmax 的归一化分母 $\sum_l \exp(q^T k_l)$ 会让所有 token 的权重互相影响——每来一个新 token，分母变了，所有旧 token 的权重都要重新算。线性注意力去掉了 softmax，每个 token 的贡献是独立的外积，可以直接累加。

```
标准 Attention 推理（每次查询都重新翻阅全部历史）：
  第 t 个 token 想查信息 → q_t 和前面 t 个 k 逐个算点积 → O(t)
  需要存储：完整的 KV Cache [t, d] × 2，随 t 增长

线性注意力推理（维护一个固定大小的记忆矩阵）：
  第 t 个 token 想查信息 → S_t · q_t → O(d²)
  需要存储：S 矩阵 [d, d]，大小固定，不随 t 增长
```

| | 标准 Attention | 线性注意力 |
|--|---------------|-----------|
| 推理状态（每层每 head） | KV Cache：$2 \times n \times d$ 个参数 | 记忆矩阵 $S$：$d \times d$ 个参数 |
| n=1K, d=96 | 2 × 1K × 96 = **192K** | 96² = **9.2K** |
| n=1M, d=96 | 2 × 1M × 96 = **192M** | 96² = **9.2K**（不变） |
| 每步推理开销 | $q$ 和 $n$ 个 $k$ 点积 → $O(nd)$ | $S \cdot q$ → $O(d^2)$ |

序列越长，线性注意力的优势越大——**状态大小恒定，不需要 KV Cache**。

#### Result：O(n) 到手了，但记忆质量崩了

$O(n)$ 不是白得的。问题出在递推更新的"只加不减"规则：

$$S_t = S_{t-1} + v_t k_t^T$$

每个 token 把自己的 key-value 对**加到**记忆矩阵里，但**旧的永远不会被覆盖**。如果两个 token 的 key 很相似（比如两个含义不同的"它"），后来的 value 会叠加到前面的上面，查询时返回混合结果——**记忆污染**（retrieval interference）。

这就是纯线性注意力效果远不如标准 Attention 的原因——压缩太狠、没有遗忘机制、没有纠错能力。2020-2023 年间几乎没有大模型采用它。

**解决记忆污染，正是 Delta Rule 和 Gated DeltaNet 要做的事。**

### 2.2 Delta Rule：从"堆积"到"纠错"

这就是 1960s Widrow & Hoff 的 Delta Rule 要解决的问题。思想很简单——**写之前先查旧记忆，有误差才修正**：

$$S_t = S_{t-1} + \beta_t \underbrace{(v_t - S_{t-1} \cdot k_t)}_{\text{误差 (delta)}} \cdot k_t^T$$

- $S_{t-1} \cdot k_t$：用当前 key 去查旧记忆，看看记得的 value 是什么
- $v_t - S_{t-1} \cdot k_t$：真实 value 和记忆的差（delta）
- 用这个差去修正记忆

对比一下：

```
线性注意力：S_t = S_{t-1} + v · k^T                    只管往里加
Delta Rule： S_t = S_{t-1} + β(v - S_{t-1}·k) · k^T    先查旧记忆，算差值，再修正
```

后者等价于：**先擦后写**。如果旧记忆已经完美（$S_{t-1} \cdot k_t = v_t$），delta 为零，什么都不改。如果旧记忆有误，只修正有误的部分。

### 2.3 DeltaNet 的训练瓶颈与并行化

DeltaNet 的递推 $S_t$ 依赖 $S_{t-1}$，没法像标准 attention 那样并行算所有 token。这在 2021 年是个致命问题——训练太慢，没人用。

Songlin Yang 等人（NeurIPS 2024）找到了 chunk-wise 并行算法：

- 把序列分成固定大小的 chunk（256 tokens）
- **chunk 内部**：把递推展开成矩阵形式，编译成 batched GEMM（矩阵-矩阵乘法），GPU 友好
- **chunk 之间**：传递状态矩阵 $S$（串行，但只有 $n/256$ 步）

和 Flash Attention 的分块思想类似——通过"块间串行 + 块内并行"在 GPU 上高效运行。不需要三角求解（triangular solve），全是 matmul-rich 操作。

### 2.4 Gated DeltaNet（ICLR 2025）—— 面试重点

在 DeltaNet 基础上加两个关键改进：

**改进 1：遗忘门 $\alpha_t$**

没有门控的 DeltaNet，状态矩阵 $S$ 的范数会随序列长度无限增长——所有历史信息都堆在里面，越来越大，训练不稳定。

加入标量遗忘门 $\alpha_t \in (0, 1)$：

$$S_t = \alpha_t \cdot (I - \beta_t \cdot k_t k_t^T) \cdot S_{t-1} + \beta_t \cdot v_t k_t^T$$

等价形式（更好理解）：

$$S_t = \alpha_t \cdot S_{t-1} + \beta_t \cdot k_t(v_t - \alpha_t \cdot S_{t-1}^T k_t)^T$$

- $\alpha_t$：遗忘门——控制旧记忆保留多少（和 LSTM 的 forget gate 同一思想）
- $\beta_t$：写入强度——控制新信息写入多少

$\alpha_t$ 和 $\beta_t$ 都是数据依赖的（从输入 $x_t$ 通过线性投影学出来）。

论文还指出一个优雅的视角：门控和 delta rule 是**互补**的两种记忆管理机制——**门控擅长快速全局遗忘，delta rule 擅长精准局部覆写**。

**改进 2：短程卷积 + SiLU**

在生成 Q、K、V 之前，对输入加一个短滑动窗口卷积（kernel size 通常 4）+ SiLU 激活：

$$x' = \text{SiLU}(\text{ShortConv}(x))$$

然后对 Q、K 做 **L2 归一化**（训练稳定性的关键）。

这和 Mamba 的做法一致——卷积捕捉局部模式，弥补纯递推对相邻 token 关系的建模弱点。

**效果（1.3B 模型，100B tokens FineWeb-Edu）**：

| 模型 | Wiki PPL ↓ | LongBooks PPL ↓ | 常识推理 Avg Acc ↑ |
|------|-----------|-----------------|------------------|
| Gated DeltaNet | **16.42** | **12.17** | **55.32%** |
| Mamba2 | 16.56 | 12.56 | 54.89% |
| DeltaNet（无门控） | 17.71 | 16.88 | 52.14% |

Gated DeltaNet 在所有指标上超过 Mamba2，尤其长文本 PPL（12.17 vs 12.56）——因为 delta rule 的"先擦后写"比 Mamba 的状态压缩更能保留细节。

### 2.5 Qwen3.5-27B：第一个大规模落地

| 维度 | QwQ-32B | Qwen3.5-27B |
|------|---------|-------------|
| 架构 | 纯 Transformer | 混合架构（Gated DeltaNet + Attention） |
| 64 层构成 | 全是 Full Attention | 48 层 Gated DeltaNet + 16 层 Full Attention |
| 混合比例 | — | **3:1**（线性:全注意力） |
| FFN 维度 | 27648 | 17408（小了 37%） |
| 词表 | 152K | 248K（大了 63%） |
| 上下文 | 128K | 262K → 1M |

**3:1 混合比例的设计逻辑**：

重复模式是"3 层 Gated DeltaNet + 1 层 Full Attention"交替：
- 75% 的层用线性注意力：推理时每 token $O(1)$（只维护一个 $d \times d$ 状态矩阵），KV Cache 成本骤降
- 25% 的层穿插全注意力：弥补线性注意力的精确回忆短板（"大海捞针"需要全注意力）

**这是 2026 年 LLM 架构的主流趋势**——不是纯 Transformer 也不是纯线性，而是混合。线性层负责大部分计算（省推理成本），全注意力层负责"关键时刻的精确回忆"。

### 2.6 面试话术（30s）

> Gated DeltaNet 是 NVIDIA 2025 年 ICLR 的工作，原始的 delta rule 概念来自 Schmidhuber 组 2021 年的 DeltaNet。核心思想是用"先擦后写"的方式更新记忆矩阵——写之前先查旧记忆，只修正有误差的部分，解决了标准线性注意力的记忆污染问题。加上遗忘门控制旧记忆衰减，效果超过 Mamba2。Qwen3.5-27B 是第一个大规模落地的——64 层里 48 层用 Gated DeltaNet、16 层用全注意力，3:1 的混合设计，上下文从 128K 推到 1M。

---

## 三、NSA（Native Sparse Attention）

> DeepSeek, 2025.02 | ACL 2025 最佳论文

### 3.1 之前的稀疏注意力为什么效果打折？

Longformer、BigBird 等方案是在**推理时**用稀疏模式跳过部分 token 的计算。但训练时用全注意力，推理时切稀疏——两者分布不一致（train-test mismatch），效果必然打折扣。

类比：你在安静的教室里练了一学期听力（全注意力训练），考试却在嘈杂的菜市场里考（稀疏推理）——当然发挥不好。

### 3.2 核心思想：训练时就是稀疏的

NSA 的做法是**训练阶段就原生使用稀疏注意力**，让模型在稀疏模式下学习。三路并行：

```
对于每个 query token，同时走三条路：

1. 压缩注意力（Compression）── "缩略图"
   把历史 token 分成 block（block_size=32, stride=16, 有重叠）
   每个 block 通过 MLP 压缩成一个向量
   query 对这些压缩向量做注意力 → 粗粒度全局视野

2. 选择注意力（Selection）── "放大镜"
   根据压缩阶段的注意力分数，挑出 top-16 个最重要的 block
   对这些 block 内的原始 token 做完整注意力 → 精细检索
   （block_size=64，含 1 个固定初始 block + 2 个局部 block）

3. 滑动窗口注意力（Sliding Window）── "近视眼"
   最近 512 个 token 始终做完整注意力 → 局部上下文
```

三路结果通过**可学习的门控**加权合并：

$$o_t = \sum_{c \in \{\text{cmp, sel, win}\}} g_t^c \cdot \text{Attn}(q_t, K_t^c, V_t^c)$$

其中 $g_t^c = \sigma(\text{MLP}(x_t))$，每条路的权重由输入 token 动态决定。

### 3.3 为什么比之前的稀疏方案快？

关键不只是"选什么 token"，而是**怎么在 GPU 上跑得快**：

- 所有操作都是 **block 级别**的——GPU 的内存访问模式是按 block 的，token 级别的稀疏反而更慢（随机访问）
- 压缩和选择两步是流水线关系：压缩的 attention score 直接复用为选择的重要性排序依据，不用额外计算
- 和 Flash Attention 的分块策略兼容

### 3.4 效果（A100, 27B 模型）

| 指标 | NSA | Full Attention |
|------|-----|---------------|
| 通用 benchmark 平均 | **0.456** | 0.443 |
| LongBench（长上下文）| **0.469** | 0.437 |
| 64K 大海捞针 | **完美检索** | 完美检索 |
| 64K 解码加速 | **11.6×** | 1× |
| 训练 forward 加速 | **9.0×** | 1× |

注意：NSA 在多数通用 benchmark 上**反而优于全注意力**——因为稀疏模式本身就是一种正则化，迫使模型学会更高效地分配注意力。

### 3.5 面试话术（30s）

> NSA 是 DeepSeek 2025 年的工作，拿了 ACL 2025 最佳论文。之前的稀疏注意力训练时用全注意力、推理时切稀疏，分布不一致效果打折。NSA 训练时就原生稀疏：三条路并行——压缩注意力给全局粗粒度视野，选择注意力挑 top-k 重要 block 做精细检索，滑动窗口保局部上下文。三路通过可学习门控加权合并。Block 级操作适配 GPU 硬件，64K 上下文解码加速 11.6 倍，效果反而优于全注意力。

---

## 四、PagedAttention

> Kwon et al. (vLLM), 2023

### 4.1 KV Cache 显存的两种浪费

推理服务同时处理多个请求，每个请求序列长度不同且动态增长。传统做法是给每个请求**预分配一块连续显存**存 KV Cache。

两种浪费：
- **内部碎片**：按最大可能长度预分配，大部分请求用不完
- **外部碎片**：请求结束释放后，剩余空间不连续，新请求塞不进去

实测数据：传统系统中 **60%-80% 的 KV Cache 显存是浪费的**。

### 4.2 借鉴 OS 虚拟内存分页

操作系统解决内存碎片的方案：虚拟地址空间连续，物理内存分成固定大小的页（page），按需映射。

PagedAttention 完全照搬：

```
传统 KV Cache（连续分配）:
  请求 1: [============预分配大块===========]    ← 大部分浪费
  请求 2: [====预分配====]
  请求 3: [===预分配===]  ← 想扩展但没有连续空间了

PagedAttention（分页）:
  物理显存被切成固定大小的 block（16 tokens / block）
  请求 1: [blk3] → [blk7] → [blk1]        逻辑连续，物理不连续
  请求 2: [blk5] → [blk2]                  按需分配，用完释放
  请求 3: [blk4] → [blk6] → [blk8]         随时追加新 block
```

每个请求维护一个 **Block Table**——逻辑 block 编号到物理位置的映射表。Attention kernel 按 block table 跳着读物理内存。

### 4.3 按需分配的具体流程

"按需"意味着**不需要提前知道序列最终多长**，生成到哪就分配到哪：

1. 请求进来，从空闲 block 池分配 1 个 block（如 16 tokens/block）
2. 生成 token 往当前 block 里填，填满 16 个后，再从池里取一个新 block
3. 在 Block Table 追加一行映射：逻辑 block 2 → 物理 block X
4. 重复，直到遇到 EOS 或达到 max_length
5. 请求结束，把它占的所有 block **归还空闲池**

```
传统：请求来了 → 预分配 max_length（如 2048）个位置的连续显存
      实际只生成了 100 个 token → 1948 个位置全浪费

PagedAttention：请求来了 → 先给 1 个 block（16 位置）
      生成第 17 个 token → 再要 1 个 block
      生成第 100 个 token → 总共用了 7 个 block（112 位置）
      浪费只有最后一个 block 里没填满的 12 个位置
```

**空闲 block 耗尽怎么办？** 调度层介入——vLLM 会 preempt（换出）优先级低的请求的 block，或让新请求排队等待，类似 OS 的内存不足处理。

### 4.4 Copy-on-Write：共享 system prompt

多个请求共享相同前缀（如 system prompt）时：

```
请求 A: [system prompt blocks] → [A 的内容]
请求 B: [system prompt blocks] → [B 的内容]
                 ↑
         物理上是同一组 block（只存一份）
         需要修改时才复制——Copy-on-Write
```

和 OS 的 fork + CoW 一模一样。100 个请求共享同一个 system prompt，只存 1 份 KV。

### 4.5 效果

| 指标 | 传统系统 | PagedAttention (vLLM) |
|------|---------|----------------------|
| KV Cache 显存浪费 | 60%-80% | **< 4%** |
| 吞吐量 vs HuggingFace | 1× | **2-4×** |
| 吞吐量 vs TGI | 1× | **2-3×** |

各框架的 block 大小：vLLM 用 16 tokens/block，TensorRT-LLM 用 32，SGLang 用 1（极致灵活但开销更高）。

PagedAttention 已成为推理部署的**事实标准**。

### 4.6 面试话术（30s）

> PagedAttention 是 vLLM 2023 年的工作。推理时 KV Cache 要给每个请求分配连续显存，但请求长度不同、动态增长，预分配造成 60-80% 显存浪费。PagedAttention 借鉴 OS 虚拟内存分页——把 KV Cache 切成固定 block，按需分配，逻辑连续但物理不连续，浪费降到 4% 以下，吞吐量提升 2-4 倍。还支持 Copy-on-Write，共享 system prompt 的多个请求只存一份 KV。

---

## 五、Flash Attention

> Chapter 04 的 Attention 类通过 `F.scaled_dot_product_attention` 自动调用 Flash Attention（如果硬件支持）。我们没有手写 CUDA kernel，但必须理解它做了什么——面试必问，而且理解了它才能回答 Chapter 09 的"训练显存怎么优化"。

**一句话总结**：Flash Attention 不改变 Attention 的数学结果，不降低计算复杂度（仍然是 $O(S^2 d)$），而是通过分块计算（tiling）+ online softmax，**让完整的 $S \times S$ 注意力矩阵从头到尾不存在于显存中**。这带来两个收益：① 训练时的**激活值从 $O(S^2)$ 降到 $O(S)$**；② HBM 读写量大幅减少，实际速度快 2-4 倍。

### 5.1 标准 Attention 的 $S^2$ 问题

以 $S=4096$, $H=8$, $d=96$, BF16 为例（Chapter 04 的 MiniMind 参数，但把 seq_len 换成更现实的 4096）：

```
标准 Attention 前向传播三步：

① scores = Q · K^T        → [B, 8, 4096, 4096]    写入 HBM
② P = softmax(scores)     → [B, 8, 4096, 4096]    写入 HBM
③ O = P · V               → [B, 8, 4096, 96]      写入 HBM
④ 保存 scores 和 P 用于反向传播（Chapter 09 讲的"激活值"）
```

每个 head 的 $S \times S$ 矩阵有多大？

```
S=4096:
  单个 S×S 矩阵: 4096 × 4096 × 2 bytes (BF16) = 32 MB
  8 个 head:     32 × 8 = 256 MB
  scores + P:    512 MB（单条样本、单层）
  8 层:          4 GB 光是 attention 的 S² 激活值

S=128K（DeepSeek-V3 级别）:
  单个 S×S 矩阵: 128K × 128K × 2 bytes = 32 GB
  根本不可能 materialize
```

**$S^2$ 的问题不只是"占空间大"，而是训练时必须保存用于反向传播。** Chapter 09 讲过：反向传播需要前向时的中间结果（激活值）来算梯度。标准 Attention 的 softmax 反向公式需要 softmax 输出 $P$（$[B, H, S, S]$），$P \cdot V$ 的反向也需要 $P$。这些 $S^2$ 大小的激活值在整个 backward 过程中占着显存不释放，是训练长序列时的显存瓶颈。

### 5.2 问题的根源：GPU 存储层级与数据搬运

Flash Attention 论文的核心洞察是：**标准 Attention 是 IO-bound（受限于数据搬运速度），不是 compute-bound（受限于算力）**。要理解这一点，需要先看 GPU 的存储层级。

```
GPU 存储层级（以 A100 为例）：

┌─────────────────────────────────────────────────┐
│  SRAM（片上缓存）                                 │
│  容量: ~20 MB    带宽: ~19 TB/s                  │
│  ← 计算单元直接读写，极快                           │
└──────────────────────┬──────────────────────────┘
                       │ ← 数据搬运的瓶颈在这里
                       │    19 TB/s vs 2 TB/s = 10× 速度差
┌──────────────────────┴──────────────────────────┐
│  HBM（高带宽显存）                                │
│  容量: ~80 GB     带宽: ~2 TB/s                  │
│  ← 大但慢，"显存"通常指这个                        │
└─────────────────────────────────────────────────┘
```

GPU 的 FLOPS（如 A100 的 312 TFLOPS）远高于 HBM 带宽（2 TB/s）。对于 Attention 这种**运算量/数据量比值低**的操作，GPU 大部分时间在**等数据从 HBM 搬到 SRAM**，而不是在算——这就是 IO-bound。

标准 Attention 的数据搬运路径：

```
步骤 ①: Q, K 从 HBM → SRAM（读），算 scores = Q·K^T
         scores [S, S] 太大放不进 SRAM（32 MB > 20 MB SRAM）
         → scores 写回 HBM                                    ← 写 O(S²)

步骤 ②: scores 从 HBM → SRAM（读），算 P = softmax(scores)
         → P 写回 HBM                                         ← 读 O(S²) + 写 O(S²)

步骤 ③: P, V 从 HBM → SRAM（读），算 O = P · V
         → O 写回 HBM                                         ← 读 O(S²) + 写 O(S·d)

总 HBM 搬运量 = O(S²) 的读 + O(S²) 的写 → 被带宽卡死
```

Flash Attention 的做法：**把三步融合成一个 kernel**，Q/K/V 分块读入 SRAM，在 SRAM 中一口气算完 scores → softmax → output，中间结果全在 SRAM 里流转，只把最终 O 写回 HBM。$S \times S$ 矩阵从未完整出现在 HBM 中，HBM 搬运量从 $O(S^2)$ 降到远低于 $O(S^2)$。

### 5.3 分块计算（Tiling）+ Online Softmax

**核心思想**：不算完整的 $S \times S$ 矩阵。把 Q/K/V 分成小块，每次只在 SRAM 中算一个小块的 scores → softmax → output，用 online softmax 增量累加到最终结果。完整的 $S \times S$ 矩阵**从未存在于任何地方**。

#### 为什么矩阵乘法可以分块？
![alt text](image-2.png)

#### 分块不省计算量
$C[i,j] = \sum_k Q[i,k] \cdot K[j,k]$ 里的每一项乘法，在分块版本中同样被执行了——只不过不是在一个大循环里连续算，而是分散到了不同的块对中。分块改变的是**计算的顺序和数据在存储层级之间的搬运方式**，不是计算的数量。

#### 分块设定

```
把 Q 分成 T_r 个块（每块 B_r 行，典型值 128）：Q₁[128, 96], Q₂[128, 96], ...
把 K, V 分成 T_c 个块（每块 B_c 行）：         (K₁,V₁), (K₂,V₂), ...

S=4096, B_r=B_c=128 → T_r = T_c = 32 个块
```

#### 标准 Attention vs Flash Attention：IO 对比
![alt text](image-4.png)
#### Online Softmax：分块怎么算 softmax？

> 具体数字演算详见 [Online Softmax 步骤图解](https://claude.ai/code/artifact/892f8ee3-43f0-4953-b48e-b90b37187897)（用 [2,5,1,8,3,4] 分 3 块逐步演示）。

**先搞清楚标准 softmax 的瓶颈在哪。** 标准 softmax 对一行 $\mathbf{x} = [x_1, \ldots, x_S]$：

$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{S} \exp(x_j)}$$

要算出任何一个位置的概率，必须先算出分母 $\sum_{j=1}^{S} \exp(x_j)$——这要求**看到整行所有 $S$ 个值**。实践中还会先找 $m = \max_j x_j$，改算 $\exp(x_i - m)$ 来防 $\exp$ 溢出（fp16 下 $\exp(11.1)$ 就爆了；bf16/fp32 下 $\exp(88.7)$ 才爆，风险小得多，但加上 max 几乎零成本，所以仍是标准做法）。但要强调：**就算不找 max，sum 本身就要求看完全部值，这才是 softmax 必须"看整行"的根本原因**。

Flash Attention 分块计算时，处理 $Q_i$ 和 $K_j$ 只看到 $B_c$ 个 score，后面还有 $T_c - j$ 个 $K$ 块没见过——求不到完整的 sum，也就产出不了最终的 softmax 值。**Online Softmax 解决的正是这个问题**。

**Online Softmax**：维护 running max $m$ 和 running sum $l$，每来一个新块就增量更新，数学上**完全等价**（不是近似）。

处理 $Q_i$ 和第 $j$ 个 $K$ 块时，先算出小块分数矩阵 $S_{ij} \in \mathbb{R}^{B_r \times B_c}$：

$$S_{ij} = Q_i K_j^T / \sqrt{d}$$

然后用新块的信息更新 running max 和 running sum（以下均为逐行操作）：

$$m^{(\text{new})} = \max\!\big(m^{(\text{old})},\; \text{rowmax}(S_{ij})\big)$$

$$l^{(\text{new})} = l^{(\text{old})} \cdot \exp\!\big(m^{(\text{old})} - m^{(\text{new})}\big) + \text{rowsum}\!\Big(\exp\!\big(S_{ij} - m^{(\text{new})}\big)\Big)$$

最后更新累积输出 $O_i$——修正旧结果，加入新块贡献：

$$O_i \leftarrow O_i \cdot \frac{l^{(\text{old})}}{l^{(\text{new})}} \cdot \exp\!\big(m^{(\text{old})} - m^{(\text{new})}\big) + \frac{\exp\!\big(S_{ij} - m^{(\text{new})}\big) \cdot V_j}{l^{(\text{new})}}$$

处理完所有 $T_c$ 个 $K$ 块后，$O_i$ 就是精确的 $\text{softmax}(Q_i K^T / \sqrt{d}) \cdot V$。

**为什么旧结果要"修正"？** 两个原因都和"之前只看到了部分数据"有关：

1. **sum 变了**（根本原因）：新块带来新的 $\exp$ 项，分母 $l$ 增大了，之前按旧分母算的结果偏大，要按 $l^{(\text{old})} / l^{(\text{new})}$ 缩小。
2. **max 可能变了**（数值稳定性）：如果新块出现了更大的值，$m$ 更新后，之前所有 $\exp(x - m^{(\text{old})})$ 都要按新 max 重算——$\exp(m^{(\text{old})} - m^{(\text{new})})$ 就是这个修正因子。如果新块的 max 没超过旧的，$m^{(\text{old})} = m^{(\text{new})}$，这项等于 1，不影响。

### 5.4 训练激活值：从 $O(S^2)$ 降到 $O(S)$

> 激活值内存对比图解详见 [训练激活值对比](https://claude.ai/code/artifact/892f8ee3-43f0-4953-b48e-b90b37187897)（§2：标准 vs Flash 的内存布局 + backward 恢复流程）。

这是理解 Flash Attention 显存优化的关键，也是 Chapter 09 "变量开销：激活值"部分的核心。

#### 标准 Attention 的前向：三步运算，每步保存了什么

Chapter 09 讲过：反向传播算参数梯度需要前向时的中间结果（激活值），比如矩阵乘法 $Y = WX$ 的反向需要 $X$ 来算 $\frac{\partial L}{\partial W}$。标准 Attention 分三步，每步都产生需要保存的激活值：

```
标准 Attention 前向传播（忽略 batch 和 head 维度，单 head 视角）：

步骤 ①: S = Q · K^T / √d           [S, d] × [d, S] = [S, S]
         保存: Q [S,d] 和 K [S,d]    ← 反向算 dQ 和 dK 需要

步骤 ②: P = softmax(S)             [S, S] → [S, S]
         保存: P [S, S]              ← softmax 反向需要 P 本身（下面推导）

步骤 ③: O = P · V                  [S, S] × [S, d] = [S, d]
         保存: P [S,S] 和 V [S,d]    ← 矩阵乘法反向需要两个输入
```

注意：$Q$、$K$、$V$ 都是 $[S, d]$ 的，不是 $S^2$ 的。**唯一的 $S^2$ 激活值是 $P$**（softmax 输出）。$P$ 被两个地方需要：步骤 ② 的 softmax 反向，和步骤 ③ 的矩阵乘法反向。

#### 标准 Attention 的反向：每一步怎么算梯度

反向传播从 $\frac{\partial L}{\partial O}$（上游传来的梯度，记为 $dO$）开始，逆序回推到 $dQ$、$dK$、$dV$。以下用 $dX$ 表示 $\frac{\partial L}{\partial X}$。

**步骤 ③ 反向**：$O = P \cdot V$ 是矩阵乘法，反向公式和 Chapter 09 讲的 $Y = WX$ 一样：

$$dP = dO \cdot V^T \qquad [S,d] \times [d,S] = [S,S]$$

$$dV = P^T \cdot dO \qquad [S,S] \times [S,d] = [S,d]$$

需要前向保存的 $P$ 和 $V$。

**步骤 ② 反向**：$P = \text{softmax}(S)$。这一步的梯度公式不是简单的矩阵乘法，需要单独推导。

**Softmax 反向推导**：对 softmax 的一行来说（$P$ 的每一行独立做 softmax），设输入向量 $s$，输出向量 $p = \text{softmax}(s)$：

$$p_j = \frac{\exp(s_j)}{\sum_k \exp(s_k)}$$

softmax 的 Jacobian（$p_j$ 对 $s_k$ 的偏导）：

$$\frac{\partial p_j}{\partial s_k} = \begin{cases} p_j(1 - p_j) & \text{if } j = k \\ -p_j \cdot p_k & \text{if } j \neq k \end{cases} = p_j(\delta_{jk} - p_k)$$

用链式法则算 $ds_j$：

$$ds_j = \sum_k dp_k \cdot \frac{\partial p_k}{\partial s_j} = \sum_k dp_k \cdot p_k(\delta_{kj} - p_j) = dp_j \cdot p_j - p_j \sum_k dp_k \cdot p_k$$

提取公因子 $p_j$：

$$\boxed{ds_j = p_j \left(dp_j - \sum_k p_k \cdot dp_k \right)}$$

写成向量形式，对 $P$ 的每一行 $i$：

$$dS_i = P_i \odot \left(dP_i - \langle P_i, dP_i \rangle \cdot \mathbf{1}\right)$$

其中 $\langle P_i, dP_i \rangle = \sum_k P_{ik} \cdot dP_{ik}$ 是一个标量（该行的点积），$\odot$ 是逐元素乘。

**关键观察**：softmax 反向公式里**直接用到了 $P$**（softmax 的输出）。这就是为什么标准 Attention 必须保存 $P$——如果不存就算不了 $dS$。而 $P$ 是 $[S, S]$ 的，正是 $S^2$ 激活值的来源。

**步骤 ① 反向**：$S = Q \cdot K^T / \sqrt{d}$ 是矩阵乘法：

$$dQ = dS \cdot K \;/ \sqrt{d} \qquad [S,S] \times [S,d] = [S,d]$$

$$dK = dS^T \cdot Q \;/ \sqrt{d} \qquad [S,S] \times [S,d] = [S,d]$$

需要前向保存的 $Q$ 和 $K$（$[S,d]$，不是 $S^2$）。

**完整反向链路图**：

```
上游梯度 dO [S, d]
    │
    ├─→ dV = P^T · dO           [S, d]    ← 需要 P [S,S] ✗
    │
    ├─→ dP = dO · V^T           [S, S]    ← 需要 V [S,d] ✓
    │     │
    │     └─→ dS = P ⊙ (dP − ⟨P,dP⟩·1)  ← 需要 P [S,S] ✗  ← softmax 反向
    │           │
    │           ├─→ dQ = dS · K / √d     ← 需要 K [S,d] ✓
    │           │
    │           └─→ dK = dSᵀ· Q / √d     ← 需要 Q [S,d] ✓
    │
    标注：✓ = O(S·d)，✗ = O(S²) ← P 是唯一的 S² 激活值
```

再加上 dropout mask $M$（如果有 dropout），也是 $[S, S]$。所以标准 Attention 的 $S^2$ 激活值就是 $P$ 和 $M$：

| 激活值 | 形状 | 大小（$S$=4096, $H$=8, BF16） | 用途 |
|--------|------|------|------|
| $P$ (softmax 输出) | $[B, H, S, S]$ | 256 MB | softmax 反向 + $P \cdot V$ 反向 |
| $M$ (dropout mask) | $[B, H, S, S]$ | 128 MB (bool) | dropout 反向 |
| $Q, K, V$ | $[B, H, S, d]$ 各一 | 6 MB 各一 | 矩阵乘法反向（不是 $S^2$） |
| **合计 $S^2$ 部分** | | **384 MB/层 → 8 层 = 3.1 GB** | |

$S = 128K$ 时：每层 $P$ 矩阵 = $128K \times 128K \times 8 \times 2 = 256$ GB → 完全不可能。

#### Flash Attention 为什么不需要存 $P$？—— 算子融合

标准 Attention 分三步，每步是独立的 CUDA kernel，中间结果必须写回 HBM 才能给下一个 kernel 读：

```
标准 Attention（三个独立 kernel）：

kernel 1: S = Q·K^T/√d    → S 写入 HBM [S,S]
kernel 2: P = softmax(S)   → 从 HBM 读 S，P 写入 HBM [S,S]  ← 保存给 backward
kernel 3: O = P·V          → 从 HBM 读 P 和 V，O 写入 HBM

P 被写入 HBM 后一直占着显存，等 backward 来读
```

Flash Attention 把三步**融合成一个 kernel**（5.3 讲的分块循环），在 SRAM 内部完成 scores → softmax → 乘 V 的全部计算。$S$ 和 $P$ 只以 $[128, 128]$ 小块的形式短暂存在于 SRAM 中，用完即弃，**从未写入 HBM**。

那 backward 怎么办？$P$ 没存，softmax 反向需要 $P$，$dV$ 的计算也需要 $P$。Flash Attention 的策略：**前向时只存两个很小的东西，backward 时从它们重新恢复出 $P$ 的小块**。

#### Flash Attention 前向保存什么

| 激活值 | 形状 | 大小（$S$=4096, $H$=8, BF16） | 含义 |
|--------|------|------|------|
| $O$ (输出) | $[B, H, S, d]$ | 6 MB | Attention 的最终输出 |
| $L$ (logsumexp) | $[B, H, S]$ | 64 KB | 每个 query 位置的 $\log \sum_k \exp(S_{ik})$，**一个标量** |
| **合计** | | **≈ 6 MB/层 → 8 层 = 48 MB** | |

**384 MB vs 6 MB，单层节省 98%。** 完整的 $S \times S$ 矩阵彻底消失了。

#### 没保存 $P$，怎么算梯度？—— logsumexp 恢复

**$L$ 的定义**：$L_i = \log \sum_k \exp(S_{ik})$（第 $i$ 行的 logsumexp）。

**核心公式**——从 $L$ 和 $S$ 恢复 $P$：

$$P_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})} = \exp\!\Big(S_{ij} - \log \sum_k \exp(S_{ik})\Big) = \exp(S_{ij} - L_i)$$

只要知道 $L_i$（**每行一个标量**，$[B, H, S]$）和 $S_{ij}$（可以从 $Q$ 和 $K$ 重新算），就能恢复出任意位置的 $P_{ij}$。不需要保存完整的 $[S, S]$ 矩阵。

**Flash Attention 反向的流程**（也是分块的，和前向一样的 SRAM 分块模式）：

对每个 Q 块和 K/V 块的配对：

$$\text{① 从 HBM 读入 } Q_{\text{块}},\; K_{\text{块}},\; V_{\text{块}} \quad \text{（和前向一样）}$$

$$\text{② 重新计算 scores 小块：} S_{\text{块}} = Q_{\text{块}} \cdot K_{\text{块}}^T / \sqrt{d} \quad [128, 128]$$

$$\text{③ 用 } L \text{ 恢复 softmax 小块：} P_{\text{块}} = \exp(S_{\text{块}} - L_{\text{块}}) \quad [128, 128]$$

$$\text{④ 在 SRAM 中用 } P_{\text{块}} \text{ 计算 } dQ_{\text{块}},\; dK_{\text{块}},\; dV_{\text{块}} \text{ 的贡献}$$

$$\text{⑤ 累加到 HBM 中的梯度张量}$$

从头到尾，$[S, S]$ 的完整矩阵仍然没有在 HBM 中出现过——$P$ 只以 $[128, 128]$ 小块的形式在 SRAM 中被恢复、使用、丢弃。

**代价**：步骤 ② 多算了一遍 $Q \cdot K^T$（前向算过但没存），额外增加约 **25% 的计算量**。但这是用计算换显存——对长序列来说极其划算。

#### 激活值对比总结

| | 标准 Attention | Flash Attention |
|--|---------------|-----------------|
| **保存了什么** | $P$ $[B,H,S,S]$ + mask $[B,H,S,S]$ | $O$ $[B,H,S,d]$ + $L$ $[B,H,S]$ |
| **单层大小** ($S$=4096, $H$=8) | 384 MB ($S^2$) | 6 MB ($S \cdot d$) |
| **8 层总计** | 3.1 GB | 48 MB |
| **空间复杂度** | $O(S^2)$ | $O(S)$ |
| **backward 代价** | — | 多算一遍 $QK^T$（~25% 额外计算） |

**这就是为什么说 Flash Attention 是长序列训练的使能技术**——不只是"快"，而是让原本不可能放进显存的 $S^2$ 激活值消失了。标准 Attention 在 $S=128K$ 时单层 $P$ 矩阵就要 256 GB，根本训不了；Flash Attention 把它变成 6 MB。

### 5.5 复杂度全面对比

| | 标准 Attention | Flash Attention |
|--|---------------|-----------------|
| **计算量（FLOPs）** | $O(S^2 d)$ | $O(S^2 d)$（**相同，没有减少**） |
| **HBM 读写量** | $\Theta(S^2 d)$ | $\Theta(S^2 d^2 / M)$（$M$ = SRAM 大小） |
| **训练激活值** | $O(S^2)$（P + dropout mask） | $O(S)$（O + logsumexp） |
| **额外计算代价** | — | backward 多一遍 $QK^T$（~25%） |
| **实际速度** | 基准 | **2-4× 更快**（IO 减少） |
| **精度** | 精确 | **精确**（数学等价，不是近似） |

用 $S=4096, H=8, d=96$ 的具体数字：

```
                          标准 Attention          Flash Attention
激活值/层:                384 MB (S²)              6 MB (S·d)          → 节省 98%
8 层总激活值:             3.1 GB                   48 MB
HBM 搬运量/层:            ~1.2 GB                  ~0.3 GB             → 节省 75%
```

**常见误解澄清**：

1. ❌ "Flash Attention 把计算复杂度从 $O(S^2)$ 降到了 $O(S)$"
   ✅ 计算量不变，仍然是 $O(S^2 d)$。减少的是 IO 搬运量和激活值显存。要真正降到 $O(S)$ 需要换算法（如 §二的 Gated DeltaNet）。

2. ❌ "Flash Attention 只是算得快，显存占用没变"
   ✅ 激活值从 $O(S^2)$ 降到 $O(S)$。这是训练长序列的使能技术，不只是加速。

3. ❌ "Flash Attention 是近似算法，会损失精度"
   ✅ 数学上和标准 Attention **完全等价**（online softmax 是精确的），只是计算顺序不同。

### 5.6 MiniMind 中的使用

Chapter 04 的 Attention 类中：

```python
# Chapter 04 写的 Attention.__init__
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

# Chapter 04 写的 Attention.forward
if self.flash:
    output = F.scaled_dot_product_attention(xq, xk, xv,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=self.is_causal)
else:
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    ...
```

`F.scaled_dot_product_attention` 是 PyTorch 2.0+ 内置的接口，底层自动选择 Flash Attention 或其他高效实现（Math、Memory-Efficient、CuDNN），不需要手写 CUDA kernel。

### 5.7 版本演进

| 版本 | 年份 | 核心改进 |
|------|------|---------|
| Flash Attention v1 | 2022 | tiling + online softmax，奠定基础 |
| Flash Attention v2 | 2023 | seq_len 维度也并行（v1 只在 batch×head 维度并行）；减少非矩阵乘 FLOPs |
| Flash Attention v3 | 2024 | 利用 H100 的异步特性（warp specialization）；FP8 支持 |

### 5.8 面试话术（30s）

> Flash Attention 的核心洞察是：标准 Attention 的瓶颈不是计算量而是 IO——$S \times S$ 的注意力矩阵要在 HBM 和 SRAM 之间反复搬运。Flash Attention 用分块计算把 Q/K/V 切成小块在 SRAM 中算完每个小块的 scores → softmax → output，配合 online softmax 增量更新，完整的 $S \times S$ 矩阵从头到尾不存在于显存中。带来两个收益：一是 HBM 搬运量大幅减少，实际快 2-4 倍；二是训练时的激活值从 $O(S^2)$ 降到 $O(S)$——前向只保存 output 和 logsumexp，反向时重新分块计算 scores 再用 logsumexp 恢复 softmax 结果，代价是多 25% 计算量。注意 FLOPs 不变，仍然是 $O(S^2 d)$，变的是 IO 和显存。

---

## 六、横向对比总表

| 方案 | 解决的问题 | 核心思路 | 复杂度 | 代价 | 已落地模型 |
|------|-----------|---------|--------|------|-----------|
| GQA（04 章） | KV Cache 大 | 多 Q head 共享 KV head | $O(n^2)$ | 精度微降 | Llama 2/3, Qwen 2/3, MiniMind |
| MLA | KV Cache 大 | 低秩压缩 KV 到 latent | $O(n^2)$ | 多了 up/down 投影矩阵（可吸收） | DeepSeek-V2/V3/R1 |
| Flash Attention（§五） | IO 瓶颈 + $S^2$ 激活值 | IO-aware 分块 + 重算替代存储 | $O(n^2)$ 不变，IO↓ 激活值 $O(S)$ | backward 多 25% 计算 | 几乎所有主流模型 |
| Differential Attn | 注意力噪声 | 两组 softmax 做差分 | $O(n^2)$ | head 数减半 + GroupNorm 开销 | 微软内部模型 |
| Gated DeltaNet | $O(n^2)$ 太慢 | delta rule 先擦后写 + 遗忘门 | $O(n)$ | 精确回忆弱，需混合全注意力层 | Qwen3.5/3.6-27B |
| NSA | 训练推理都 $O(n^2)$ | 原生稀疏：压缩+选择+窗口 | $O(n \cdot k)$ | 实现复杂，block 设计需调参 | V3.2（演化为 DSA）→ V4（演化为 CSA+HCA） |
| PagedAttention | KV Cache 显存碎片 | OS 分页 + CoW | 不改 Attention | Kernel 改写复杂 | vLLM, SGLang, TRT-LLM |
| Mamba/SSM（19 章） | $O(n^2)$ 太慢 | 状态空间模型 | $O(n)$ | 精确检索弱，状态压缩丢信息 | Jamba, Zamba |

## 七、面试策略

### "你了解哪些 Attention 变体？"

按三条线组织回答，每条线本身就是一个演进故事：

1. **省 KV Cache 线**：GQA（减 head 数）→ MLA（降维到 latent）→ PagedAttention（显存管理）
2. **降复杂度线**：Flash Attention（IO 优化，还是 O(n²)）→ NSA（原生稀疏）→ Gated DeltaNet / Mamba（线性 O(n)）→ 混合架构（Qwen3.5 的 3:1 设计）
3. **提质量线**：Differential Attention（降噪，60% 参数量打平标准 Transformer）

### DeepSeek 稀疏注意力演化线：NSA → DSA → CSA+HCA

这条线面试高频，且容易搞混。三代机制解决的问题逐步升级：

```
全注意力 O(n²)
  │
  ↓  "能不能只看重要的 token？"
NSA（Native Sparse Attention）—— 2025.02 独立论文，ACL 2025 最佳论文
  │  三路并行：压缩注意力 + 选择注意力 + 滑动窗口
  │  block 级稀疏（block_size=32/64）
  │  V3 和 R1 的技术报告里都没有提到（NSA 论文发得更晚）
  │
  ↓  "block 太粗了，能不能按 token 选？indexer 能不能更快？"
DSA（DeepSeek Sparse Attention）—— DeepSeek-V3.2（2025 下半年，技术报告 2025.12）
  │  token 级稀疏（不再按 block 选）
  │  Lightning Indexer：FP8 低维投影 + ReLU 打分（不用 softmax）
  │  与 MLA 深度集成（MQA 模式，共享 latent KV）
  │
  ↓  "token 级稀疏省了计算，但 KV Cache 本身还是 O(n)，1M 存不下"
CSA + HCA —— DeepSeek-V4（2026.04）
  │  先压缩 KV Cache（每 m/m' 个 token → 1 条），再做稀疏/稠密 attention
  │  CSA：压缩 4x + 稀疏选择（DSA 的后继）
  │  HCA：压缩 128x + 稠密 attention（够短了，不需要选）
  │  两者交替排列在不同层
  │
  ↓  效果
V4-Pro 1M 上下文：FLOPs 仅为 V3.2 的 27%，KV Cache 仅为 10%
```

#### 第一代：NSA（§四已详细讲过，这里只做对比锚点）

NSA 的核心是**三路并行 + 门控合并**：压缩注意力给粗粒度全局视野，选择注意力挑 top-k block 做精细检索，滑动窗口保局部上下文。三路输出通过可学习门控加权合并。

关键特征：**block 级稀疏**——以 block（32 或 64 个 token）为单位选择，GPU 内存访问模式友好。但粒度较粗，不重要的 block 里可能夹杂重要的 token。

#### 第二代：DSA——从 block 稀疏到 token 稀疏

DSA 在 DeepSeek-V3.2（2025 下半年）中首次部署。相比 NSA 有三个关键变化：

**变化一：token 级稀疏取代 block 级稀疏。**

NSA 的选择单元是 block（一组连续 token），DSA 直接按单个 token 选择。粒度更细，能精准跳过 block 内不相关的 token。

**变化二：Lightning Indexer 重新设计。**

NSA 的三路分支需要各自独立计算，DSA 统一成一个两阶段 pipeline：

```
DSA 两阶段 pipeline（对每个 query token t）：

Stage 1 — Lightning Indexer（打分）:
  对每个历史 token s（s < t）:
    1. 把 query 和 key 投影到低维空间（indexer dim = 128）
    2. 多头点积打分，每头独立：score_h = q_t^I · k_s^I
    3. ReLU 激活（不用 softmax！）：score_h = max(0, score_h)
    4. 各头加权求和：I_{t,s} = Σ_h  w_h · score_h
  全部用 FP8 精度计算

Stage 2 — Top-k 选择 + 精细 Attention:
  从 I_{t,:} 中选出分数最高的 k=2048 个 token
  只对这 2048 个 token 做完整精度的 attention
```

为什么用 ReLU 而不是 softmax？因为 indexer 只需要**排名**（选 top-k），不需要概率分布。ReLU 比 softmax 便宜，不需要跨所有 token 做归一化，而且在 FP8 下不会因为 exp 溢出。

**变化三：与 MLA 深度集成。**

DSA 不是独立模块，而是直接在 MLA 的 latent 空间里工作。indexer 的 key 直接从 MLA 的低维 latent 向量投影，不需要先恢复完整的 K。核心注意力用 MQA 模式（所有 query head 共享一份 key-value），减少 gather 操作。

**训练方式：**

DSA 需要让 indexer 学会"什么 token 重要"。训练分两阶段：
1. **Dense warm-up**（1000 步）：冻结主模型，只训练 indexer，用 KL 散度对齐到全注意力的分布
2. **Sparse training**（15000 步）：解冻全部参数，indexer 和主模型一起训练

效果：在 MMLU-Pro、GPQA 等 benchmark 上与全注意力质量持平，128K 上下文推理 GPU 成本降低约 2 倍。

#### 第三代：CSA——先压缩 KV，再稀疏选择

DSA 解决了"每个 query 不需要看所有 token"的计算问题，但 **KV Cache 本身还是 O(n)**——每个 token 都要存一条 KV 记录。对于 1M 上下文，即使用了 MLA 压缩，KV Cache 仍然巨大。

CSA 的思路：**在做稀疏选择之前，先把 KV Cache 压缩掉**。

```
DSA（V3.2）:
  原始 KV: [t0, t1, t2, t3, t4, t5, t6, t7, ...]   n 条记录
  indexer 在 n 条原始记录上打分 → 选 top-k → 精细 attention
  KV Cache 大小: O(n)

CSA（V4）:
  原始 KV: [t0, t1, t2, t3, t4, t5, t6, t7, ...]
              ↓ 每 m=4 个压缩成 1 条（learned softmax-gated pooling）
  压缩 KV: [c0,       c1,       ...]                  n/m 条记录
  indexer 在 n/m 条压缩记录上打分 → 选 top-k → 精细 attention
  KV Cache 大小: O(n/m)   ← 比 DSA 小 m 倍
```

**CSA 的压缩操作**——不是简单的平均池化，而是**逐坐标的 softmax 加权**：

对每 $m$ 个连续 token 的 KV 候选 $C_1, C_2, \ldots, C_m$（每个 $\in \mathbb{R}^c$），学一组权重 $Z_1, \ldots, Z_m$，对每个特征维度独立做 softmax，然后加权求和：

$$S_j = \text{Softmax}_{\text{col}}(Z_j), \quad C^{\text{Comp}} = \sum_{j=1}^{m} S_j \odot C_j$$

其中 $\odot$ 是逐元素乘法。每个输出维度有自己独立的 softmax 权重——某个维度可能主要取 $t_0$ 的值，另一个维度可能主要取 $t_3$ 的值。这比均匀平均（mean pooling）保留了更多信息。

**重叠压缩（Overlapped Compression）**——CSA 的独特设计：

CSA 维护**两套** KV 流（stream a 和 stream b），压缩第 $i$ 个 block 时，不只看当前 block 的 $m$ 个 token（stream a），还看前一个 block 的 $m$ 个 token（stream b）。softmax 在 $2m$ 个位置上做：

$$C_i^{\text{Comp}} = \sum_{j=mi}^{m(i+1)-1} S_j^a \odot C_j^a + \sum_{j=m(i-1)}^{mi-1} S_j^b \odot C_j^b$$

为什么需要重叠？如果每个压缩条目只看自己 block 内的 $m$ 个 token，block 边界处会产生信息断裂——恰好落在两个 block 交界的语义片段会被割裂。重叠让相邻 block 共享边界 token，消除硬边界。

**CSA 的 Lightning Indexer**——升级到 FP4：

indexer 沿用 DSA 的 ReLU 打分设计，但现在打分对象是压缩后的条目（$n/m$ 条，而非 $n$ 条），所以 indexer 本身也便宜了 $m$ 倍。V4 进一步把 indexer 精度从 FP8 降到 **FP4**——因为不需要 softmax（只需排名），ReLU 的数值范围对低精度友好。

V4-Pro 的 CSA 参数：$c = 512$，$m = 4$，$k = 1024$，indexer heads = 64，indexer dim = 128。

#### 第三代：HCA——极端压缩 + 稠密注意力

HCA 是 CSA 的极端特化版本，做了三个简化：

| | CSA | HCA |
|--|-----|-----|
| 压缩比 | $m = 4$ | $m' = 128$（32 倍更激进） |
| KV 流 | 两套（有重叠） | 一套（无重叠） |
| 稀疏选择 | top-k（只看部分） | 无（全部看，因为够少了） |

HCA 的逻辑非常直接：**压缩得足够狠，压缩后的序列就短到可以做全注意力**。

1M 上下文经过 $m' = 128$ 压缩后只剩 $10^6 / 128 \approx 7813$ 条压缩记录。对 7813 条做全注意力的成本微乎其微。

HCA 不需要 indexer，不需要 top-k kernel，不需要重叠压缩。代价是信息损失更大——128 个 token 压成 1 条，很多细节丢失了。但 HCA 的定位就是提供**粗粒度的全局视野**，精细检索交给 CSA 层。

#### 混合架构：CSA + HCA 交替排列

V4 不是所有层都用 CSA 或都用 HCA，而是**交替排列**：

```
Layer 0:  CSA    ← 中等压缩 + 稀疏选择（精细检索）
Layer 1:  HCA    ← 极端压缩 + 稠密（全局视野）
Layer 2:  CSA
Layer 3:  HCA
...
Layer 60: CSA/HCA
```

为什么要交替？两种层各有优势：
- **CSA 层**：压缩比温和（4x），保留更多细节，加上 top-k 选择能精准定位关键 token。擅长**精细检索**——"第 50000 个 token 处说了什么"。
- **HCA 层**：压缩比极端（128x），每个 query 看到全部压缩后的全局信息。擅长**全局概览**——"整篇文档的主题是什么"。

交替排列让模型每隔一层就切换视角：一层细看，一层粗看。多层叠加后，既有全局感知又有局部精度。

#### 效果数据

| 指标 | V3.2（DSA） | V4-Pro（CSA+HCA） | V4-Flash |
|------|-----------|-----------------|----------|
| 1M 推理 FLOPs（vs V3.2） | 1× | **27%** | **10%** |
| 1M KV Cache（vs V3.2） | 1× | **10%** | **7%** |
| 参数量（总/激活） | — | 1.6T / 49B | 284B / 13B |
| 上下文长度 | 128K | **1M** | **1M** |

Benchmark（V4 技术报告数据）：

| Benchmark | V4-Pro-Max | Gemini-3.1-Pro | Claude Opus 4.6 Max |
|-----------|-----------|----------------|---------------------|
| MRCR 1M（长上下文检索） | 83.5 | 76.3 | 92.9 |
| CorpusQA 1M（长上下文问答） | 62.0 | 53.8 | 71.7 |

V4 在长上下文上显著超越 Gemini，但仍落后于 Claude Opus 4.6。

#### KV Cache 压缩效率推算

以 V4-Pro 为例，用 GQA-8（head_dim=128, BF16）做基线：

$$\text{GQA 基线每层每 token}: 2 \times 8 \times 128 \times 2\text{B} = 4096 \text{ bytes}$$

$$\text{CSA 每层每 token}: \frac{1}{4} \times 512 \times 2\text{B} = 256 \text{ bytes} \quad (6.25\%)$$

$$\text{HCA 每层每 token}: \frac{1}{128} \times 512 \times 2\text{B} = 8 \text{ bytes} \quad (0.2\%)$$

61 层、约一半 CSA 一半 HCA：$(30 \times 256 + 31 \times 8) / (61 \times 4096) \approx 6.3\%$。实测 10% 略高，因为还有滑动窗口 KV、sink token、RoPE 维度等未建模的开销。

#### 面试话术（30s）

> DeepSeek 的稀疏注意力经历了三代演化。NSA（2025 年 ACL 最佳论文）提出三路并行的原生稀疏注意力，但粒度是 block 级的。V3.2 把它工程化为 DSA，改成 token 级稀疏，用 FP8 的 Lightning Indexer 打分选 top-k token。V4 在 DSA 基础上加了一步关键创新——先把 KV Cache 压缩再做稀疏选择。CSA 每 4 个 token 压成 1 条，再 top-k 选择；HCA 每 128 个 token 压成 1 条，直接全注意力。两者交替排列，1M 上下文下 FLOPs 只有 V3.2 的 27%，KV Cache 只有 10%。核心思路是：NSA 省了计算，CSA+HCA 连存储也省了。

### "2026 年 LLM 架构的趋势是什么？"

> 纯 Transformer 不再是唯一答案。两大方向：(1) 混合架构——大部分层用线性注意力（如 Gated DeltaNet）降推理成本，少部分层穿插全注意力保精确回忆，Qwen3.5 的 3:1 设计是第一个大规模落地的例子；(2) 稀疏注意力——DeepSeek 的 NSA→DSA→CSA+HCA 路线，不换掉 Attention 但让每个 query 只看最重要的 token，V4 用这条路做到了 1M 上下文。KV Cache 优化在推理侧已经标准化——MLA 压缩 + PagedAttention 管理基本是标配。

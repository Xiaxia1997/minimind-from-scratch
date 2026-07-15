# 04b - Attention 变体演进：面试向横向对比

## 为什么需要这篇？

04-Attention 讲的是 MiniMind 实现的 MHA → GQA 代码线，17 章讲 Flash Attention 的 IO 优化，20 章从长上下文视角提到了 MLA 和 Mamba。但面试里 Attention 变体经常被当作独立话题问：

> "你了解哪些 Attention 变体？各有什么优缺点？"
> "DeepSeek 的 MLA 是怎么回事？和 GQA 有什么区别？"
> "Gated DeltaNet 听说过吗？为什么 Qwen3.5 用它？"

这篇不重复已有章节的内容，而是**用一个统一视角把所有变体串起来**：标准 Attention 有三个成本中心——**计算量、显存占用、注意力噪声**，每个变体都是在某个成本中心上做优化。理解了这一点，面试时不管怎么问，都能把任何一个变体归位到这张图里。

## 统一视角：标准 Attention 的三个成本中心

```
标准 Self-Attention 的代价拆解：

成本中心 1：计算量 O(n²)
  ├─ 不改数学，优化 IO        → Flash Attention           （17 章）
  ├─ 稀疏化，只算重要 token    → NSA                       （本章 §四）
  └─ 彻底换成 O(n) 的机制     → Gated DeltaNet / Mamba    （本章 §三 / 20 章）

成本中心 2：KV Cache 显存占用（推理时）
  ├─ 减少 KV head 数量        → GQA                       （04 章）
  ├─ 低秩压缩 KV              → MLA                       （本章 §一）
  └─ 优化显存管理（不改 Attn） → PagedAttention            （本章 §五）

成本中心 3：注意力噪声（softmax 给无关 token 分配概率）
  └─ 两组 softmax 做差分消噪   → Differential Attention    （本章 §二）
```

下面逐个展开。对已有章节覆盖的内容（GQA、Flash Attention、Mamba），只在横向对比表里出现，不重复讲。

---

## 一、MLA（Multi-head Latent Attention）

> DeepSeek-V2, 2024.05 | 20 章 §2.3 已有概述，这里补完整数学

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

## 二、Differential Attention

> Ye et al. (Microsoft Research + 清华), 2024.10 | ICLR 2025

### 2.1 问题：softmax 的"被迫分配"

标准 Attention 中，softmax 把注意力概率归一化到 1——**即使所有 token 都不重要，也必须把概率分出去**。结果是无关 token 拿到了本不该有的注意力权重（"attention noise"）。

这不是小问题。论文实验显示，标准 Transformer 的 attention logit 最大值高达 **318.0**（说明模型在拼命拉大差距来压制噪声），而 Differential Transformer 只有 **38.8**。噪声直接导致：
- 长上下文场景被无关信息干扰
- In-context learning 中对错误示例过度敏感
- 生成时产生幻觉

### 2.2 核心思想：降噪耳机原理

降噪耳机用两个麦克风采集环境噪声，相减后噪声被抵消。Differential Attention 完全同理：

$$\text{DiffAttn}(X) = \left(\text{softmax}\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right) - \lambda \cdot \text{softmax}\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right)\right) V$$

两组 attention map 的共同噪声（无关 token 获得的注意力）在相减时被抵消，只留下真正有信号的差异部分。

### 2.3 怎么拆的？

不是加了一倍的 head，而是**每个 head 内部拆成两半**：

$$[Q_1;\; Q_2] = X W_Q, \quad [K_1;\; K_2] = X W_K$$

其中 $Q_1, Q_2 \in \mathbb{R}^{n \times d}$，$K_1, K_2 \in \mathbb{R}^{n \times d}$。

关键：**V 不拆**，$V \in \mathbb{R}^{n \times 2d}$。Q、K 各拆半做差分，然后共同作用于完整的 V。

为了保持参数量和 FLOPs 与标准 Transformer 一致，Differential Transformer **使用标准 Transformer 一半的 head 数**（每个 head 内部拆两组抵消了）。

### 2.4 λ 的设计——比一个标量复杂

$\lambda$ 不是简单的可学习标量。为了让它能学到比初始值更小的值（甚至负值），用了一个"可正可负"的参数化：

$$\lambda = \exp(\lambda_{q_1} \cdot \lambda_{k_1}) - \exp(\lambda_{q_2} \cdot \lambda_{k_2}) + \lambda_{\text{init}}$$

其中 $\lambda_{q_1}, \lambda_{k_1}, \lambda_{q_2}, \lambda_{k_2} \in \mathbb{R}^d$ 都是可学习向量。

初始值与**层深度**相关（不是 head 编号）：

$$\lambda_{\text{init}} = 0.8 - 0.6 \times \exp(-0.3 \times (l-1))$$

浅层 $\lambda_{\text{init}}$ 较小（差分较强），深层接近 0.8（差分较弱）。直觉：浅层更需要去噪（特征还没分化），深层的 attention 本身就比较精准了。

差分之后还要过一个 **GroupNorm**（每个 head 独立归一化），再乘以缩放因子 $(1 - \lambda_{\text{init}})$。

### 2.5 效果

论文在同等训练条件下对比：

| 对比 | Diff Transformer | 等效标准 Transformer | 参数比 |
|------|-----------------|---------------------|--------|
| 6.8B Diff | val loss 相当 | 11B 标准 | 62.2% |
| 7.8B Diff | val loss 相当 | 13.1B 标准 | 59.5% |

文本摘要的忠实度（不幻觉）：XSum 从 44% 提升到 53%，MultiNews 从 42% 提升到 61%。

**一句话概括**：用约 **60% 的参数量**达到标准 Transformer 的效果，同时幻觉更少。

### 2.6 面试话术（30s）

> Differential Attention 是微软和清华 2024 年的工作，ICLR 2025 发表。核心是降噪耳机原理——每个 head 的 Q/K 各拆成两半，分别算 softmax attention，然后做差分，共同的噪声被抵消。V 不拆，head 数减半，参数量和 FLOPs 不变。效果是 6.8B Diff Transformer 打平 11B 标准 Transformer，只需要 62% 的参数量。幻觉也显著减少。

---

## 三、Delta Attention / Gated DeltaNet

### 3.1 先理解"线性注意力到底在做什么"

要理解 DeltaNet，先要看清标准线性注意力的本质。

标准 Attention 是 $\text{softmax}(QK^T)V$，每个 token 要和所有其他 token 交互，$O(n^2)$。

线性注意力把 softmax 去掉（或换成核函数），变成 $(QK^T)V = Q(K^TV)$。通过**改变矩阵乘法的结合顺序**，先算 $K^TV$（$d \times d$ 矩阵，和序列长度无关），再用 Q 去查，变成 $O(n)$。

但这个 $S = K^TV$ 本质上就是一个 **key→value 的记忆矩阵**（也叫"快权重 / fast weight"）。给一个 query，$S \cdot q$ 返回与 q 最相关的 value 的加权和。

递推形式更清楚：

$$S_t = S_{t-1} + v_t \cdot k_t^T \qquad o_t = S_t \cdot q_t$$

每个 token 把自己的 key-value 对**加到**记忆矩阵里。问题来了——**只加不减，旧的 key-value 永远不会被覆盖**。如果两个 token 的 key 很相似，后来的 value 会叠加到前面的上面，造成"记忆污染"（retrieval interference）。

### 3.2 Delta Rule：从"堆积"到"纠错"

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

### 3.3 演进脉络

```
1960s   Delta Rule（Widrow & Hoff）
          ↓  经典联想记忆更新规则
2021    DeltaNet（Schlag, Irie, Schmidhuber — ICML 2021）
          ↓  首次把 delta rule 包装成线性注意力层的替代品
          ↓  问题：严格递推，训练不能并行
2024    DeltaNet 并行化（Songlin Yang 等 — NeurIPS 2024）
          ↓  chunk-wise 并行算法，训练速度追上 Mamba
2025    Gated DeltaNet（Songlin Yang + NVIDIA — ICLR 2025）
          ↓  加遗忘门 + 短程卷积，效果超越 Mamba2
2026.2  Qwen3.5-27B（阿里）
          └─ 第一个大规模生产模型落地
```

**归属**：概念原创——Schmidhuber 组（ICML 2021）；工程可用化 + 门控改进——Songlin Yang + NVIDIA（ICLR 2025）；Qwen 是采用者，不是发明者。

### 3.4 DeltaNet 的训练瓶颈与并行化

DeltaNet 的递推 $S_t$ 依赖 $S_{t-1}$，没法像标准 attention 那样并行算所有 token。这在 2021 年是个致命问题——训练太慢，没人用。

Songlin Yang 等人（NeurIPS 2024）找到了 chunk-wise 并行算法：

- 把序列分成固定大小的 chunk（256 tokens）
- **chunk 内部**：把递推展开成矩阵形式，编译成 batched GEMM（矩阵-矩阵乘法），GPU 友好
- **chunk 之间**：传递状态矩阵 $S$（串行，但只有 $n/256$ 步）

和 Flash Attention 的分块思想类似——通过"块间串行 + 块内并行"在 GPU 上高效运行。不需要三角求解（triangular solve），全是 matmul-rich 操作。

### 3.5 Gated DeltaNet（ICLR 2025）—— 面试重点

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

### 3.6 Qwen3.5-27B：第一个大规模落地

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

### 3.7 面试话术（30s）

> Gated DeltaNet 是 NVIDIA 2025 年 ICLR 的工作，原始的 delta rule 概念来自 Schmidhuber 组 2021 年的 DeltaNet。核心思想是用"先擦后写"的方式更新记忆矩阵——写之前先查旧记忆，只修正有误差的部分，解决了标准线性注意力的记忆污染问题。加上遗忘门控制旧记忆衰减，效果超过 Mamba2。Qwen3.5-27B 是第一个大规模落地的——64 层里 48 层用 Gated DeltaNet、16 层用全注意力，3:1 的混合设计，上下文从 128K 推到 1M。

---

## 四、NSA（Native Sparse Attention）

> DeepSeek, 2025.02 | ACL 2025 最佳论文

### 4.1 之前的稀疏注意力为什么效果打折？

Longformer、BigBird 等方案是在**推理时**用稀疏模式跳过部分 token 的计算。但训练时用全注意力，推理时切稀疏——两者分布不一致（train-test mismatch），效果必然打折扣。

类比：你在安静的教室里练了一学期听力（全注意力训练），考试却在嘈杂的菜市场里考（稀疏推理）——当然发挥不好。

### 4.2 核心思想：训练时就是稀疏的

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

### 4.3 为什么比之前的稀疏方案快？

关键不只是"选什么 token"，而是**怎么在 GPU 上跑得快**：

- 所有操作都是 **block 级别**的——GPU 的内存访问模式是按 block 的，token 级别的稀疏反而更慢（随机访问）
- 压缩和选择两步是流水线关系：压缩的 attention score 直接复用为选择的重要性排序依据，不用额外计算
- 和 Flash Attention 的分块策略兼容

### 4.4 效果（A100, 27B 模型）

| 指标 | NSA | Full Attention |
|------|-----|---------------|
| 通用 benchmark 平均 | **0.456** | 0.443 |
| LongBench（长上下文）| **0.469** | 0.437 |
| 64K 大海捞针 | **完美检索** | 完美检索 |
| 64K 解码加速 | **11.6×** | 1× |
| 训练 forward 加速 | **9.0×** | 1× |

注意：NSA 在多数通用 benchmark 上**反而优于全注意力**——因为稀疏模式本身就是一种正则化，迫使模型学会更高效地分配注意力。

### 4.5 面试话术（30s）

> NSA 是 DeepSeek 2025 年的工作，拿了 ACL 2025 最佳论文。之前的稀疏注意力训练时用全注意力、推理时切稀疏，分布不一致效果打折。NSA 训练时就原生稀疏：三条路并行——压缩注意力给全局粗粒度视野，选择注意力挑 top-k 重要 block 做精细检索，滑动窗口保局部上下文。三路通过可学习门控加权合并。Block 级操作适配 GPU 硬件，64K 上下文解码加速 11.6 倍，效果反而优于全注意力。

---

## 五、PagedAttention

> Kwon et al. (vLLM), 2023

### 5.1 KV Cache 显存的两种浪费

推理服务同时处理多个请求，每个请求序列长度不同且动态增长。传统做法是给每个请求**预分配一块连续显存**存 KV Cache。

两种浪费：
- **内部碎片**：按最大可能长度预分配，大部分请求用不完
- **外部碎片**：请求结束释放后，剩余空间不连续，新请求塞不进去

实测数据：传统系统中 **60%-80% 的 KV Cache 显存是浪费的**。

### 5.2 借鉴 OS 虚拟内存分页

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

### 5.3 Copy-on-Write：共享 system prompt

多个请求共享相同前缀（如 system prompt）时：

```
请求 A: [system prompt blocks] → [A 的内容]
请求 B: [system prompt blocks] → [B 的内容]
                 ↑
         物理上是同一组 block（只存一份）
         需要修改时才复制——Copy-on-Write
```

和 OS 的 fork + CoW 一模一样。100 个请求共享同一个 system prompt，只存 1 份 KV。

### 5.4 效果

| 指标 | 传统系统 | PagedAttention (vLLM) |
|------|---------|----------------------|
| KV Cache 显存浪费 | 60%-80% | **< 4%** |
| 吞吐量 vs HuggingFace | 1× | **2-4×** |
| 吞吐量 vs TGI | 1× | **2-3×** |

各框架的 block 大小：vLLM 用 16 tokens/block，TensorRT-LLM 用 32，SGLang 用 1（极致灵活但开销更高）。

PagedAttention 已成为推理部署的**事实标准**。

### 5.5 面试话术（30s）

> PagedAttention 是 vLLM 2023 年的工作。推理时 KV Cache 要给每个请求分配连续显存，但请求长度不同、动态增长，预分配造成 60-80% 显存浪费。PagedAttention 借鉴 OS 虚拟内存分页——把 KV Cache 切成固定 block，按需分配，逻辑连续但物理不连续，浪费降到 4% 以下，吞吐量提升 2-4 倍。还支持 Copy-on-Write，共享 system prompt 的多个请求只存一份 KV。

---

## 六、横向对比总表

| 方案 | 解决的问题 | 核心思路 | 复杂度 | 代价 | 已落地模型 |
|------|-----------|---------|--------|------|-----------|
| GQA（04 章） | KV Cache 大 | 多 Q head 共享 KV head | $O(n^2)$ | 精度微降 | Llama 2/3, Qwen 2/3, MiniMind |
| MLA | KV Cache 大 | 低秩压缩 KV 到 latent | $O(n^2)$ | 多了 up/down 投影矩阵（可吸收） | DeepSeek-V2/V3/R1 |
| Flash Attention（17 章） | 训练/推理慢 | IO-aware 分块计算 | $O(n^2)$ 不变，常数小 | 需自定义 CUDA kernel | 几乎所有主流模型 |
| Differential Attn | 注意力噪声 | 两组 softmax 做差分 | $O(n^2)$ | head 数减半 + GroupNorm 开销 | 微软内部模型 |
| Gated DeltaNet | $O(n^2)$ 太慢 | delta rule 先擦后写 + 遗忘门 | $O(n)$ | 精确回忆弱，需混合全注意力层 | Qwen3.5/3.6-27B |
| NSA | 训练推理都 $O(n^2)$ | 原生稀疏：压缩+选择+窗口 | $O(n \cdot k)$ | 实现复杂，block 设计需调参 | DeepSeek-V3 |
| PagedAttention | KV Cache 显存碎片 | OS 分页 + CoW | 不改 Attention | Kernel 改写复杂 | vLLM, SGLang, TRT-LLM |
| Mamba/SSM（20 章） | $O(n^2)$ 太慢 | 状态空间模型 | $O(n)$ | 精确检索弱，状态压缩丢信息 | Jamba, Zamba |

## 七、面试策略

### "你了解哪些 Attention 变体？"

按三条线组织回答，每条线本身就是一个演进故事：

1. **省 KV Cache 线**：GQA（减 head 数）→ MLA（降维到 latent）→ PagedAttention（显存管理）
2. **降复杂度线**：Flash Attention（IO 优化，还是 O(n²)）→ NSA（原生稀疏）→ Gated DeltaNet / Mamba（线性 O(n)）→ 混合架构（Qwen3.5 的 3:1 设计）
3. **提质量线**：Differential Attention（降噪，60% 参数量打平标准 Transformer）

### "2026 年 LLM 架构的趋势是什么？"

> 纯 Transformer 不再是唯一答案。趋势是混合架构——大部分层用线性注意力（如 Gated DeltaNet）降推理成本，少部分层穿插全注意力保精确回忆。Qwen3.5 的 3:1 设计是第一个大规模落地的例子，上下文从 128K 推到 1M。同时 KV Cache 优化在推理侧已经标准化——MLA 压缩 + PagedAttention 管理基本是标配。

# 19 - DeepSeek V4 架构创新：mHC + Engram

> DeepSeek 在 2025.12—2026.01 连发两篇架构论文，分别改造了 Transformer 的残差连接和稀疏参数分配，均已用于 DeepSeek V4。
> 论文：mHC: Manifold-Constrained Hyper-Connections (arXiv:2512.24880, 2025.12)
> 论文：Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs (arXiv:2601.07372, 2026.01)
> 前置：[07-MiniMindBlock](07-MiniMindBlock.md)（残差连接 + Pre-Norm）、[06-MOEFeedForward](06-MOEFeedForward.md)（MoE 路由）

## 一句话总结

**mHC** 用数学约束把残差连接从 "固定权重 1.0 的单车道" 升级为 "可学习的多车道"，解决深层网络信号爆炸。**Engram** 把高频 N-gram 知识从神经计算中剥离出来交给 O(1) 哈希查表，让 MoE 专家专注于真正需要推理的部分。

---

# Part 1: mHC——可学习的残差连接

## 1.1 动机：标准残差连接的天花板

07 章我们写过 Transformer 的残差连接：

$$y = x + F(x)$$

$F$ 是 Attention 或 FFN 子层，$x$ 是 skip connection。权重固定为 1.0——既保证梯度直通（解决深层梯度消失），又是恒等映射（初始化时网络近似 identity）。

**问题**：固定权重 = 每一层对残差流的贡献完全一样。但直觉上，浅层可能主要靠残差传原始信息，深层应该更依赖 $F(x)$ 的变换结果。模型没有能力学习 "这一层残差的贡献应该大一点还是小一点"。

```
标准残差 — 每层都是同一个固定公式:

  layer 1:  x₁ = x₀ + F₁(x₀)      权重固定 1.0
  layer 2:  x₂ = x₁ + F₂(x₁)      权重固定 1.0
  ...
  layer L:  x_L = x_{L-1} + F_L(x_{L-1})  权重固定 1.0

  不能学到 "layer 3 残差权重 0.8、变换权重 1.2" 这样的差异化策略
```

## 1.2 Hyper-Connections：让残差可学习（但不稳定）

2024 年有人提出 Hyper-Connections (HC)：把残差流从 1 维扩展到 $n$ 维（$n$ 条平行通道），用可学习矩阵控制信息路由。

$$x_{l+1} = H_l^{\text{res}} \cdot x_l + (H_l^{\text{post}})^T \cdot F(H_l^{\text{pre}} \cdot x_l,\; W_l)$$

- $x_l \in \mathbb{R}^{nC}$：$n$ 条宽度为 $C$ 的并行流（$C$ 是 hidden_dim）
- $H_l^{\text{res}} \in \mathbb{R}^{n \times n}$：**残差混合矩阵**——控制 $n$ 条流之间的信息交换
- $H_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$：**聚合**——把 $n$ 条流压缩成 1 条送进子层 $F$
- $H_l^{\text{post}} \in \mathbb{R}^{1 \times n}$：**散开**——把子层输出分发回 $n$ 条流

```
标准残差（1 条流）:
  ─── x ──┬──── + ────→ y
           │      ↑
           └─ F(x)┘

HC（n=4，4 条流）:
  ─── x₁ ──┐                  ┌── y₁
  ─── x₂ ──┤── H^pre ─→ F ─→ H^post ──┤── y₂
  ─── x₃ ──┤                  ┤── y₃
  ─── x₄ ──┘     + H^res      └── y₄
                  (流之间交叉混合)
```

理论上很美，但 **实际训练会爆炸**。原因：$H^{\text{res}}$ 是任意矩阵，跨层连续相乘时信号指数放大。

```
信号放大灾难:

  假设每层 H^res 的谱范数(spectral norm) = 1.05（看似无害）
  L = 60 层:  1.05^60 ≈ 18x     ← 信号放大 18 倍
  L = 100 层: 1.05^100 ≈ 131x   ← 爆了

  实测信号增益:
  ┌──────────┬──────────────┬────────────┐
  │ 模型规模  │ 无约束 HC 增益 │ 结果       │
  ├──────────┼──────────────┼────────────┤
  │ 3B       │ ~50x         │ 性能下降    │
  │ 9B       │ ~300x        │ 训练不稳定  │
  │ 27B      │ ~3000x       │ 训练发散    │
  └──────────┴──────────────┴────────────┘
```

## 1.3 mHC 的核心：Sinkhorn-Knopp 投影到双随机矩阵流形

mHC 的关键洞察：**把 $H^{\text{res}}$ 约束为双随机矩阵（doubly stochastic matrix）**。

双随机矩阵的定义——所有元素非负、每行和每列之和都等于 1：

$$\mathcal{B}_n = \{X \in \mathbb{R}^{n \times n} \mid X \cdot \mathbf{1} = \mathbf{1},\; X^T \cdot \mathbf{1} = \mathbf{1},\; X \geq 0\}$$

这个集合叫 **Birkhoff 多面体（Birkhoff Polytope）**——所有置换矩阵的凸包。

为什么双随机矩阵能解决信号爆炸？三条数学性质：

```
双随机矩阵的三条关键性质:

1. 谱范数 = 1
   双随机矩阵的最大特征值恒为 1 → 信号不会被放大
   
2. 乘法封闭
   两个双随机矩阵的乘积还是双随机矩阵
   → L 层连续乘，结果仍然是双随机矩阵，谱范数仍 = 1
   → 信号增益 ≈ 1.0，无论多少层

3. 恒等映射
   单位矩阵 I 是双随机矩阵 → 初始化时网络近似 identity
```

**Sinkhorn-Knopp 算法**把任意非负矩阵投影到双随机矩阵上——交替做行归一化和列归一化：

```python
def sinkhorn_knopp(matrix, iterations=20, eps=1e-8):
    # 先指数化确保全正
    P = exp(matrix)
    for _ in range(iterations):
        P = P / (P.sum(axis=1, keepdims=True) + eps)  # 行归一化：每行和=1
        P = P / (P.sum(axis=0, keepdims=True) + eps)  # 列归一化：每列和=1
    return P
    # 20 次迭代后误差 < 10^-13，实际使用足够
```

## 1.4 mHC 的完整公式

mHC 让 $H^{\text{res}}$、$H^{\text{pre}}$、$H^{\text{post}}$ 都变成**输入相关的动态矩阵**（不是静态参数）：

$$x'_l = \text{RMSNorm}(x_l)$$

$$H_l^{\text{pre}} = \sigma\big(\alpha_l^{\text{pre}} \cdot x'_l W_l^{\text{pre}} + b_l^{\text{pre}}\big) \in \mathbb{R}^{1 \times n}$$

$$H_l^{\text{post}} = 2\sigma\big(\alpha_l^{\text{post}} \cdot x'_l W_l^{\text{post}} + b_l^{\text{post}}\big) \in \mathbb{R}^{1 \times n}$$

$$H_l^{\text{res}} = \text{SK}\big(\alpha_l^{\text{res}} \cdot \text{mat}(x'_l W_l^{\text{res}}) + b_l^{\text{res}}\big) \in \mathbb{R}^{n \times n}$$

其中：
- $W_l^{\text{pre}}, W_l^{\text{post}} \in \mathbb{R}^{nC \times n}$，$W_l^{\text{res}} \in \mathbb{R}^{nC \times n^2}$：可学习权重
- $\alpha_l^{\text{pre}}, \alpha_l^{\text{post}}, \alpha_l^{\text{res}}$：可学习标量（缩放因子）
- $\sigma(\cdot)$：sigmoid 函数
- $\text{mat}(\cdot)$：把向量 reshape 成 $n \times n$ 矩阵
- $\text{SK}(\cdot)$：Sinkhorn-Knopp 投影

```
关键设计选择:

  只有 H^res 需要 Sinkhorn-Knopp 约束 ← 消融实验证明最关键
  H^pre 和 H^post 用 sigmoid 约束到 [0,1] / [0,2] 就够了
  
  H^post 乘以 2 → 输出范围 [0, 2]，允许放大子层贡献
  
  三个矩阵都是 input-dependent（动态的）:
  → 不同输入 token 走不同的路由策略
  → 比静态权重更有表达力
```

## 1.5 信号增益对比

```
三种方案在不同模型规模下的信号增益:

  ┌──────────┬──────────┬──────────────┬──────────┐
  │ 模型规模  │ 标准残差  │ 无约束 HC     │ mHC      │
  ├──────────┼──────────┼──────────────┼──────────┤
  │ 3B       │ ~1.0     │ ~50x         │ ~1.5x    │
  │ 9B       │ ~1.0     │ ~300x        │ ~1.6x    │
  │ 27B      │ ~1.0     │ ~3000x       │ ~1.6x    │
  └──────────┴──────────┴──────────────┴──────────┘

  mHC 在所有规模下信号增益稳定在 1.5-1.6x
  → 略大于 1（有一点放大以增加表达力），但不会指数增长
```

## 1.6 Benchmark 和开销

```
27B 模型 Benchmark 对比:

  ┌──────────────┬──────────┬──────────┬──────────┐
  │ 任务          │ 标准残差  │ 无约束 HC │ mHC      │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ BBH          │ 43.8     │ 48.9     │ 51.0     │
  │ DROP         │ ~55      │ ~57      │ ~58.2    │
  │ GSM8K        │ ~58      │ ~59.5    │ ~60.8    │
  │ MMLU         │ ~59      │ ~59.8    │ ~60.4    │
  └──────────────┴──────────┴──────────┴──────────┘

  训练开销: +6.7%（Sinkhorn 20 次迭代 + 额外矩阵运算）
  训练稳定性: HC 在 ~step 12k 出现 loss spike + 梯度范数暴涨
              mHC 全程平稳无 spike
```

## 1.7 与 DeepSeek V4 的关系

DeepSeek V4 用 mHC 替换了标准残差连接，典型配置 $n=4$（4 条并行流）。配合 MLA（Multi-head Latent Attention）和 MoE，构成 V4 的三大架构支柱。

## 1.8 面试话术

> "mHC 是 DeepSeek 对残差连接的升级，解决的问题是：标准残差 $x + F(x)$ 的权重固定为 1.0，模型不能学习每层残差的贡献强度。之前有人提出 Hyper-Connections，用可学习矩阵替代固定权重，扩展成 n 条并行流——但训练会爆炸，因为任意矩阵跨层连乘导致信号指数放大，27B 模型实测放大 3000 倍。mHC 的核心是用 Sinkhorn-Knopp 算法把残差混合矩阵投影到双随机矩阵流形上——数学上保证谱范数等于 1、乘法封闭，所以无论多少层信号都不会爆炸，实测稳定在 1.6 倍。三个连接矩阵都是 input-dependent 的动态矩阵，不是静态参数，所以不同 token 可以走不同的路由策略。训练开销只有 6.7%，但 BBH 比标准残差高 7 分。已用在 DeepSeek V4 里。"

---

# Part 2: Engram——把记忆从计算中剥离

## 2.1 动机：Transformer 在重复计算已知模式

语言中有大量**固定搭配**——"United States of America"、"深度学习"、"machine learning"。标准 Transformer 处理每个 token 时，所有 Attention + FFN 层都要参与运算，包括这些早就"记住"的固定模式。

```
问题的本质:

  "7 × 8 = ?"
  
  方法 A（每次算）: 7 × 8 = 7+7+7+7+7+7+7+7 = 56        ← Transformer 现在的做法
  方法 B（查表）:   查乘法表 → 56                          ← Engram 的做法
  
  对于固定搭配（N-gram 模式），方法 B 更高效:
  - "United States of" → 下一个是 "America"，不需要 Attention
  - "机器学习" → 不需要 reasoning，直接从记忆取
  
  释放出来的计算容量可以让模型专注于真正需要推理的部分
```

Engram 的核心思想：**引入一个新的稀疏轴——条件记忆（Conditional Memory）**，与 MoE 的条件计算（Conditional Computation）互补。

```
两种稀疏:

  MoE（条件计算）:   根据输入选择性激活少数 expert → 省 FLOPs
  Engram（条件记忆）: 根据输入查 N-gram 哈希表 → 省计算 + 注入模式知识

  二者正交，可以共存
```

## 2.2 架构：四个核心组件

### 组件 1：多头哈希（Multi-Head Hashing）

用 $K$ 个独立的哈希函数把 N-gram token 序列映射到嵌入表：

$$h_k(x) = ((a_k \times x) \oplus b_k) \bmod M$$

- $a_k, b_k$：第 $k$ 个哈希头的随机常数
- $\oplus$：XOR 操作
- $M$：哈希表大小（质数，减少碰撞）

对于 N-gram（多个 token 组合），用**多项式滚动哈希**：

$$\text{combined} = (\text{combined} \times 31 + \text{token\_id}) \;\&\; \text{0x7FFFFFFF}$$

```
为什么用多头:

  单个哈希 → 碰撞率高，"deep learning" 和 "keep leaning" 可能映到同一个桶
  K 个独立哈希头 → 碰撞概率指数下降
  
  类似布隆过滤器的思路：多个哈希函数交叉验证
  
  实测配置: K=8 个哈希头，N-gram 阶数 n ∈ {2, 3}（2-gram 和 3-gram）
```

### 组件 2：上下文感知门控（Context-Aware Gating）

不是所有上下文都适合用记忆查表——"The capital of France is" 可以查表，"If we assume X, then by analogy with Y" 需要推理。

门控机制用当前 hidden state 作为 query，决定记忆注入的权重：

```
gating = sigmoid(hidden_state @ W_gate + b_gate)    # ∈ [0, 1]
output = gating * memory_lookup + (1 - gating) * neural_output

gating → 1: 信任记忆（固定模式，如 N-gram 搭配）
gating → 0: 信任神经计算（需要推理的上下文）
```

### 组件 3：因果卷积整合（Causal Convolution）

多个 N-gram 阶数的查表结果需要整合。Engram 用**因果深度卷积**（causal depthwise convolution）：

- 核大小 $w = 4$
- 膨胀率 $\delta$ 设为最大 N-gram 阶数
- 因果约束：只看过去的 token，不看未来（防止信息泄漏）
- 初始化为零：训练开始时 Engram 模块输出为零（恒等映射），不破坏预训练权重

### 组件 4：层级放置策略

Engram 模块不是每层都加，而是**只插入两个位置**：

```
Transformer 30 层结构中的 Engram 放置:

  Layer 0   ← 标准层
  Layer 1   ← 标准层
  Layer 2   ← Engram 模块 ★ (浅层：初始表征构建，注入表面模式)
  Layer 3   ← 标准层
  ...
  Layer 14  ← 标准层
  Layer 15  ← Engram 模块 ★ (中层：用上下文模式增强推理)
  Layer 16  ← 标准层
  ...
  Layer 29  ← 标准层

  为什么是 layer 2 和 15:
  - Layer 2: 浅层还在建表征，注入 N-gram 模式帮助快速锚定词义
  - Layer 15: 约一半深度，上下文已经积累够了，此时注入模式知识最有用
  - 更深层主要做高级推理，N-gram 模式已不再是瓶颈
```

## 2.3 稀疏参数分配：U 型曲线

Engram 的一个关键发现：MoE 的稀疏参数预算应该**分一部分给 Engram**。

定义分配比 $\rho$：不活跃参数中分给 MoE expert 的比例。$\rho = 1$ 时全给 MoE，$\rho = 0$ 时全给 Engram。

```
稀疏参数分配的 U 型曲线:

  loss
   │
   │  ╲                    ╱
   │   ╲                  ╱
   │    ╲               ╱
   │     ╲            ╱
   │      ╲_________╱
   │           ↑
   └──────────────────────→ ρ
   0   0.2  0.4  0.6  0.8  1.0
       ↑                   ↑
       最佳点              纯 MoE
       (ρ ≈ 0.20-0.25)    (baseline)

  ρ = 1.0: 纯 MoE（baseline）
  ρ = 0.75: 稍好于 baseline（25% 参数给 Engram）
  ρ = 0.20-0.25: 最优 ← 约 75-80% 参数给 Engram
  ρ < 0.20: 性能下降（MoE 容量不足，推理能力退化）
```

Engram-27B 的配置：**2+55 个 expert（top-6 路由）**，而纯 MoE-27B 用 **2+72 个 expert**。减少 17 个 expert，省出的参数全给了 Engram 查找表。

## 2.4 Benchmark 对比

```
Engram-27B vs MoE-27B 对比:

  ┌──────────────────┬──────────┬────────────┬────────┐
  │ 任务              │ MoE-27B  │ Engram-27B │ 提升   │
  ├──────────────────┼──────────┼────────────┼────────┤
  │ 知识类                                             │
  │ MMLU             │ 57.4     │ 60.4       │ +3.0   │
  │ CMMLU            │ 57.9     │ 61.9       │ +4.0   │
  │ C-Eval           │ 58.0     │ 62.7       │ +4.7   │
  │ ARC-Challenge    │ 70.1     │ 73.8       │ +3.7   │
  ├──────────────────┼──────────┼────────────┼────────┤
  │ 推理类                                             │
  │ BBH              │ 50.9     │ 55.9       │ +5.0   │
  │ DROP (F1)        │ 55.7     │ 59.0       │ +3.3   │
  │ GSM8K            │ 58.4     │ 60.6       │ +2.2   │
  ├──────────────────┼──────────┼────────────┼────────┤
  │ 代码类                                             │
  │ HumanEval        │ 37.8     │ 40.8       │ +3.0   │
  ├──────────────────┼──────────┼────────────┼────────┤
  │ 长上下文                                            │
  │ Multi-Query NIAH │ 73.0%    │ 99.6%      │ +26.6  │
  └──────────────────┴──────────┴────────────┴────────┘

  语言建模 (Pile test):
  MoE-27B loss = 2.091 → Engram-27B loss = 1.960（显著下降）

  推理开销: < 3%（哈希查表是 O(1)，100B 参数可卸载到 CPU 内存）
```

知识类任务（MMLU、C-Eval）提升最大——正好是 N-gram 模式最密集的场景。长上下文检索（NIAH）从 73% 跳到 99.6%——因为 Engram 分担了浅层的模式匹配负担，Transformer 可以把注意力资源集中在长距离依赖上。

## 2.5 无限记忆 Scaling Law

哈希表大小从 $2.58 \times 10^5$ 扩到 $10^7$ 个 slot 时，loss 下降遵循**几乎完美的幂律**。而且这些额外参数不参与计算（纯查表），可以卸载到 CPU 内存，不占 GPU 显存。

```
Engram Scaling Law:

  哈希表 slot 数      loss 下降      GPU 开销
  2.58 × 10^5        baseline       +0%
  1.0 × 10^6         ↓ 显著         +0%（可放 CPU）
  1.0 × 10^7         ↓ 持续下降     +0%（可放 CPU）
  
  这是一条几乎"免费"的 scaling 路线——加内存不加算力
```

## 2.6 面试话术

> "Engram 是 DeepSeek 提出的条件记忆机制，核心思想是把 Transformer 对高频 N-gram 模式的重复计算替换成 O(1) 哈希查表。具体做法是用多头哈希（8 个头，N-gram 阶数 2 和 3）把 token 序列映射到嵌入表，用因果卷积整合多阶查表结果，再通过上下文感知门控决定是用记忆还是用神经计算——固定搭配走查表，需要推理的走 Attention。模块只插在 layer 2 和 15 两个位置，初始化为零不破坏预训练权重。最关键的发现是 U 型 scaling 曲线：把约 20-25% 的 MoE 稀疏参数预算分给 Engram 效果最好，Engram-27B 用 55 个 expert 加查找表，比 72 个 expert 的纯 MoE-27B 在知识类任务上高 3-5 分。另外哈希表可以卸载到 CPU 内存，扩大到千万级 slot 也几乎零 GPU 开销——这是一条免费的 scaling 路线。"

---

# Part 3: 业界对比

## 3.1 残差连接方案对比

| 方案 | 核心思路 | 模型 | 优缺点 |
|------|---------|------|--------|
| 标准残差 $x + F(x)$ | 固定权重 1.0 | GPT/LLaMA/Qwen | 稳定但不可学习 |
| Pre-Norm + 残差 | 先 Norm 再残差 | 几乎所有现代 LLM | 缓解梯度问题但权重仍固定 |
| Hyper-Connections | 可学习矩阵多流 | — | 表达力强但训练爆炸 |
| **mHC** | **双随机矩阵约束多流** | **DeepSeek V4** | **表达力 + 稳定性，+6.7% 开销** |

## 3.2 稀疏架构对比

| 方案 | 稀疏类型 | 核心思路 | 模型 |
|------|---------|---------|------|
| MoE | 条件计算 | 路由选 top-k expert | DeepSeek V2/V3/V4、Qwen-MoE |
| MLA | 低秩压缩 | KV Cache 低秩投影 | DeepSeek V2/V3/V4 |
| **Engram** | **条件记忆** | **N-gram 哈希查表** | **DeepSeek V4** |

DeepSeek V4 是第一个同时使用三种稀疏策略的模型：MoE（条件计算）+ MLA（KV 低秩）+ Engram（条件记忆）。

---

# Part 4: 综合面试话术

> "DeepSeek V4 在架构上有两个核心创新：mHC 和 Engram。
>
> mHC 解决残差连接的问题——标准残差权重固定为 1.0，不能学习每层的贡献强度。之前有人提出用可学习矩阵替代，但跨层连乘导致信号指数爆炸。mHC 用 Sinkhorn-Knopp 算法把残差混合矩阵约束到双随机矩阵流形——数学上保证谱范数为 1、乘法封闭，所以无论多少层信号都稳定在 1.6 倍。开销只有 6.7%。
>
> Engram 引入了条件记忆这个新的稀疏轴——把高频 N-gram 模式从神经计算中剥离出来交给 O(1) 哈希查表，让 MoE 专家专注于推理。最佳配置是把 20-25% 的稀疏参数预算从 MoE expert 转给 Engram 查找表。知识类任务提升 3-5 分，推理开销不到 3%，而且哈希表可以放 CPU 内存——是一条免费的 scaling 路线。
>
> V4 是第一个同时用三种稀疏策略的模型：MoE 做条件计算、MLA 做 KV 压缩、Engram 做条件记忆。"

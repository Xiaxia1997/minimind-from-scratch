# 04 - Attention：让 token 之间交互信息

## 在整个模型中的位置

```
Embedding (01) → RMSNorm (02) → 【Attention (04)】 → 残差连接 → RMSNorm → FFN (05) → 残差连接
                                  ↑ 本章                              ↑ 下一章
                                  RoPE (03) 在这里注入位置信息
```

前面三章我们依次写好了：
- **01 Embedding**：token → 768 维向量（`scratch/model/model_minimind.py`）
- **02 RMSNorm**：稳定每层输入的数值范围
- **03 RoPE**：给 Q/K 注入位置信息（`precompute_freqs_cis` + `apply_rotary_pos_emb`）

但到目前为止，每个 token 都是"独立思考"——Embedding 是查表，RMSNorm 是逐 token 归一化，RoPE 只是注入位置信息。**token 之间还没有任何交互**。

## 为什么需要 Attention？

Attention 就是让每个 token "看到"其他 token，并决定从谁那里获取多少信息。它是 Transformer 中**唯一负责 token 间通信**的组件。

## 从零开始写

> 以下 Step 1-9 是独立可运行的探索代码（直接粘贴到 Python REPL 即可跑），逐步演化理解 Attention 的设计。最终要追加到 `scratch/model/model_minimind.py` 的正式代码在后面的 [完整实现](#完整实现) 部分。

### Step 1：最简 Self-Attention

每个 token 生成三个向量：
- **Q（Query）**："我想关注什么？"
- **K（Key）**："我能提供什么？"
- **V（Value）**："我携带的内容"

```python
import torch, math
from torch import nn

hidden_size = 768

W_q = nn.Linear(hidden_size, hidden_size, bias=False)
W_k = nn.Linear(hidden_size, hidden_size, bias=False)
W_v = nn.Linear(hidden_size, hidden_size, bias=False)

x = torch.randn(2, 128, 768)  # [batch, seq_len, hidden]

Q = W_q(x)  # [2, 128, 768]  "我想关注什么"
K = W_k(x)  # [2, 128, 768]  "我能提供什么"
V = W_v(x)  # [2, 128, 768]  "我携带的内容"

# 注意力分数：Q 和 K 的点积（越相似分数越高）
scores = Q @ K.transpose(-2, -1) / math.sqrt(hidden_size)  # [2, 128, 128]
attn = torch.softmax(scores, dim=-1)                        # 归一化成概率
output = attn @ V                                            # [2, 128, 768] 加权聚合
```

**问题**：这是全量注意力，768 维的 Q 关注 768 维的 K——所有维度一起做一个大点积，只能学到一种注意力模式。

### Step 2：Multi-Head Attention（MHA）——多组并行注意力

#### 单头的问题：一个 softmax 只有一个注意力分布

注意力矩阵形状是 $[S, S]$——每个 token 对所有 S 个 token 算一个关注分数。用 4 个 token 举例：

```
         小明   把    球    踢
  小明  [0.7,  0.1,  0.1,  0.1]   ← "小明"主要关注自己
  把    [0.3,  0.4,  0.2,  0.1]   ← "把"关注前后
  球    [0.1,  0.1,  0.6,  0.2]   ← "球"主要关注自己
  踢    [0.3,  0.1,  0.3,  0.3]   ← "踢"关注"小明"和"球"
```

$$A = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)$$

$A$ 的每一行经过 softmax 后是**一个**概率分布（加起来 = 1）——每个 token 只有一种方式来分配关注。

用"小明把球踢给了小红"举例，"踢"需要同时关注三种关系：

- **语法主语**："小明"（谁踢的）
- **语法宾语**："球"（踢什么）
- **间接宾语**："小红"（踢给谁）

单头只有一行概率分布，三种关系被迫瓜分：

```
"踢" 的注意力分布（单头，一行概率要分给所有关系）:
  [小明: 0.3,  把: 0.05,  球: 0.35,  踢: 0.1,  给: 0.05,  小红: 0.15]
   主语 ↑              宾语 ↑                              间接宾语 ↑

  0.3 + 0.35 + 0.15 = 0.8，三种关系瓜分一行概率 → 哪个都关注不充分
```

自然的想法：我需要**多组独立的注意力分布**，让不同的头各自专注一种关系。

#### 论文的做法：多组低维注意力

MHA 来自 Vaswani et al. 2017 "Attention is All You Need" Section 3.2.2。论文原文的措辞是：

> "Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, **we found it beneficial** to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$ dimensions respectively."


论文给出的设计和理由：

**做法**：用 $h$ 组不同的线性投影，把 Q/K/V 分别投影到更低的 $d_k = d_{model}/h$ 维度，并行做 $h$ 次注意力，最后 concat 回来。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

**为什么效果好**（论文原话）：

> "Multi-head attention allows the model to jointly attend to information from **different representation subspaces** at different positions. With a single attention head, averaging inhibits this."

单头在全维度上做一次点积再 softmax，相当于把所有维度的匹配信号**平均**成一个分数——不同类型的关系被混在一起。多头让每组投影关注不同的表征子空间，不同关系不用互相干扰。

**为什么 $d_k = d_{model}/h$ 而不是每头都用 $d_{model}$**（论文原话）：

> "Due to the reduced dimension of each head, the **total computational cost is similar** to that of single-head attention with full dimensionality."

这是一个**工程约束**：保持总计算量不变。d_model=512, h=8, 则 d_k=64。每头计算量是 $S^2 \times 64$，8 头总计 $S^2 \times 512$，和单头全维度一样。

```
计算量对比：
  单头全维度:  S² × 512                    → 1 个注意力分布
  每头全维度:  S² × 512 × 8 = S² × 4096    → 8 个注意力分布，但计算量 ×8
  MHA 实际:   S² × 64 × 8  = S² × 512     → 8 个注意力分布，计算量不变 ✓
```

**总结：MHA 的核心逻辑不是"768 维太多要拆分"，而是——**

1. 单头只有一个注意力分布，想要多个
2. 每头都用全维度？计算量爆炸
3. 实验发现：每头投影到低维 $d_k = d_{model}/h$，效果一样好甚至更好
4. 原因是不同头关注不同的 representation subspace，低维足以捕获单一类型的关系
5. 而且总计算量和单头持平——免费的多样性

**问题**：MHA 每个头都有独立的 KV，推理时需要缓存所有头的 KV Cache，显存开销大。→ 这引出了 MQA 和 GQA。

### Step 3：GQA（Grouped Query Attention）——省 KV Cache

#### 从 MHA 到 GQA 的演化链（MHA → MQA → GQA）

GQA 不是凭空设计的，它是 MHA → MQA → GQA 三步演化的产物，核心驱动力是**推理时的内存带宽瓶颈**。

**第一步：MHA 的推理瓶颈**

MHA 每个头都有独立的 K/V。自回归解码时，每生成一个 token，都要从显存里加载所有历史 token 的 KV Cache。头数越多，KV Cache 越大，内存带宽成为瓶颈——不是算力不够，而是**数据搬运太慢**。

**第二步：MQA（Shazeer, 2019）——激进方案**

Shazeer 的论文标题很直白："Fast Transformer Decoding: **One Write-Head is All You Need**"（[arXiv:1911.02150](https://arxiv.org/abs/1911.02150)）。

做法：所有 Q 头共享**一组** KV 头。KV Cache 直接缩小到 $1/h$。

问题：论文原话 "can lead to **quality degradation** and **training instability**"。太激进了，质量掉了。

**第三步：GQA（Ainslie et al., 2023）——折中方案**

GQA 的出发点很简单：MQA 太激进（1 个 KV 头，质量掉了），MHA 太贵（$h$ 个 KV 头，推理慢），**能不能找个中间值？**（[arXiv:2305.13245](https://arxiv.org/abs/2305.13245)）

做法：用 $g$ 个 KV 头（$1 < g < h$），每 $h/g$ 个 Q 头共享一组 KV。

论文实验数据（T5-XXL）：

```
              推理延迟    质量(ROUGE)
MHA:          1.51ms     47.2         ← 质量最好，但最慢
MQA:          0.24ms     46.6         ← 最快，但质量掉了
GQA-8:        0.28ms     47.1         ← 接近 MHA 质量，接近 MQA 速度 ✓
```

**GQA 不是理论推导出来的，而是在 MHA 和 MQA 两个极端之间找到了实验最优的平衡点。**

#### 为什么共享的是 KV 而不是 Q？

**工程原因很明确**：GQA 要解决的是推理时 KV Cache 太大的问题。自回归生成时，Q 只有当前 token 的（不缓存），K 和 V 要缓存所有历史 token 的。减少 KV 头数 = 直接缩小缓存；减少 Q 头数对推理速度没帮助。所以只能动 KV。

**至于"为什么共享 KV 后质量几乎不掉"**——这也是实验发现，论文没有给出理论证明。GQA 论文的逻辑就是：MQA 共享到 1 个 KV 头质量掉了，那试试多几个 KV 头——试了，GQA-8 质量接近 MHA、速度接近 MQA。

#### MiniMind 的 GQA 配置

MiniMind：8 个 Q 头，4 个 KV 头，每 2 个 Q 共享一组 KV。

```
MHA:  Q₀  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇
      ↕   ↕   ↕   ↕   ↕   ↕   ↕   ↕       KV 头=8, Cache 100%
      K₀  K₁  K₂  K₃  K₄  K₅  K₆  K₇

GQA:  Q₀  Q₁ │ Q₂  Q₃ │ Q₄  Q₅ │ Q₆  Q₇
      ↘  ↙   │ ↘  ↙   │ ↘  ↙   │ ↘  ↙    KV 头=4, Cache 50%
       K₀    │  K₁    │  K₂    │  K₃

MQA:  Q₀  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇
      ↘   ↓   ↓   ↓   ↓   ↓   ↓   ↙       KV 头=1, Cache 12.5%
                   K₀
```

KV Cache 节省对比：

```
MHA:  8 个 Q 头各配 1 个 KV 头 → KV Cache = 8 × 96 × 2(K+V) = 1536 维/token
GQA:  8 个 Q 头共享 4 个 KV 头 → KV Cache = 4 × 96 × 2      = 768 维/token  ← 省 50%

Qwen2.5-7B 更激进：28 个 Q 头只用 4 个 KV 头（n_rep=7），KV Cache 省 86%
```

关键：**W_k 和 W_v 的投影矩阵本身就小**——`[768, 384]` 而不是 `[768, 768]`。
然后用 `repeat_kv` 把 4 头"虚拟复制"成 8 头，再做注意力。

从这里开始逐步构建最终的 Attention 类。先写 `repeat_kv` 函数和 `__init__`（投影层），以及 forward 的第一段（投影 + 拆头 + repeat）：

```python
import math
import torch.nn.functional as F

def repeat_kv(x, n_rep):
    """GQA: 把 n_kv 个 KV 头虚拟复制成 n_heads 个"""
    B, S, n_kv, d = x.shape
    if n_rep == 1: return x
    return x[:, :, :, None, :].expand(B, S, n_kv, n_rep, d).reshape(B, S, n_kv * n_rep, d)

class Attention(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8, n_kv_heads=4, head_dim=96):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # 2
        self.head_dim = head_dim

        self.wq = nn.Linear(hidden_size, n_heads * head_dim, bias=False)    # [768, 768]
        self.wk = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False) # [768, 384] ← GQA 只有一半
        self.wv = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False) # [768, 384]
        self.wo = nn.Linear(n_heads * head_dim, hidden_size, bias=False)    # [768, 768]

    def forward(self, x, cos, sin):
        B, S, _ = x.shape

        # ---- Step 3: 投影 + 拆头 + GQA repeat ----
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)    # [B, S, 8, 96]
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim) # [B, S, 4, 96]
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim) # [B, S, 4, 96]

        # ... Step 4-7 在后续逐步补充 ...
```

### Step 4：加 QK Norm

#### 问题：Q/K 经过投影后幅度不可控

回顾 Attention 内部的数据流：

```
x (已被 Pre-Norm 归一化)
  → W_q 投影 → Q    ← 幅度又不可控了（W_q 权重没有约束）
  → W_k 投影 → K    ← 同上
  → Q · K^T / sqrt(d_k) → scores → softmax
```

$\sqrt{d_k}$ 缩放的前提假设是 Q/K 每个元素的方差 $\approx 1$。但 x 经过 $W_q$、$W_k$ 线性投影后，这个假设不成立——投影可以任意放大某些维度的幅度。

如果 Q/K 的幅度变大，$Q \cdot K$ 的值就可能出现极端值（比如 $\pm 50$），$\sqrt{d_k}$ 是个固定的数（$\sqrt{96} \approx 9.8$），压不住。softmax 输入太大就会饱和（接近 one-hot），梯度消失，训练卡住。

#### QK Norm 的做法：投影之后再归一化

QK Norm 不是替代 $\sqrt{d_k}$，而是**在它之前加一道保险**：

```
完整流程（两个都要）:
  x → W_q → Q → RMSNorm(Q) → RoPE → Q'
  x → W_k → K → RMSNorm(K) → RoPE → K'
  scores = Q' · K'^T / sqrt(d_k)     ← sqrt(d_k) 还在

两者各管一件事:
  QK Norm:   保证 Q/K 的幅度可控（动态的，跟着训练过程变化）
  sqrt(d_k): 保证点积的期望方差 ≈ 1（静态的，只跟维度有关）
```

为什么用 RMSNorm 而不是简单加大缩放系数（比如除以 $2\sqrt{d_k}$）？因为静态缩放在训练初期（幅度小）和后期（幅度大）不能同时合适。RMSNorm 是自适应的——幅度大就压得多，幅度小就压得少。

在 `__init__` 中加两个 RMSNorm，在 forward 中 Step 3 之后追加：

```python
        # __init__ 中追加:
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # forward 中, 接 Step 3 之后:
        # ---- Step 4: QK Norm ----
        xq = self.q_norm(xq)  # 投影之后、RoPE 之前，归一化 Q 的幅度
        xk = self.k_norm(xk)
```

标准 LLaMA 没有 QK Norm，但 Gemma 2、Qwen 都加了。MiniMind 也加了——小模型训练不稳定，更需要这道保险。

### Step 5：加 RoPE（位置信息）

forward 中接 Step 4 之后：

```python
        # ---- Step 5: RoPE（cos, sin 由外部传入）----
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 只旋转 Q/K，V 不旋转

        # ---- 接 Step 3 的 repeat_kv，放在 RoPE 之后 ----
        xk = repeat_kv(xk, self.n_rep)  # [B, S, 4, 96] → [B, S, 8, 96]
        xv = repeat_kv(xv, self.n_rep)  # 同上
```

### Step 6：加 Causal Mask + 计算注意力

自回归模型：每个 token 只能看到自己和之前的 token，不能"偷看未来"。

#### 为什么用 $-\infty$ 而不是 0？

**目标**：让位置 $i$ 完全无法从位置 $j > i$ 获取信息。

**方案 A**：把未来位置的分数设为 0

```
scores  = [2.1,  3.5,  0,  0]       ← 后两个是未来，设为 0
softmax → [0.18, 0.72, 0.05, 0.05]  ← 未来位置仍有 5% 的权重！
```

因为 $e^0 = 1 \neq 0$，softmax 会给它们分配非零概率。

**方案 B**：把未来位置的分数设为 $-\infty$

```
scores  = [2.1,  3.5, -inf, -inf]
softmax → [0.20, 0.80,  0,    0]    ← 未来位置权重严格为 0 ✓
```

因为 $e^{-\infty} = 0$，softmax 归一化后它们严格消失。

**只有 $-\infty$ 能保证信息完全隔离**。用 0 或任何有限负数都会泄漏信息。

forward 中接 Step 5 之后：

```python
        # ---- Step 6: transpose + Causal Mask + 注意力计算 ----
        xq = xq.transpose(1, 2)  # [B, n_heads, S, head_dim]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)  # [B, 8, S, S]
        mask = torch.full((S, S), float("-inf"), device=x.device).triu(1)
        scores = scores + mask
        output = F.softmax(scores.float(), dim=-1).type_as(xq) @ xv  # [B, 8, S, 96]

        # ---- Step 7: 合并头 + 输出投影 ----
        output = output.transpose(1, 2).reshape(B, S, -1)  # [B, S, 768]
        return self.wo(output)                               # [B, S, 768]
```

### Step 8：KV Cache（推理加速）

训练时一次看完整序列，推理时逐 token 生成。没有 KV Cache 时，每生成一个 token 都要重算所有历史 token 的 K/V——浪费。

```
推理第 200 步（有 KV Cache）：
  Q: [B, 8, 1, 96]           ← 只有当前 token 的 Q
  K: [B, 8, 200, 96]         ← cat(past_K[199], new_K[1]) 从 cache 取
  scores: [B, 8, 1, 200]     ← 只算新 token 对所有历史的注意力

  GQA 只缓存 4 头 KV（不是 8 头）→ 显存省一半
```

```python
# 推理时
xk = torch.cat([past_K, xk], dim=1)  # 沿 seq 维拼接
xv = torch.cat([past_V, xv], dim=1)
# 然后正常做注意力
```

### Step 9：Flash Attention

标准注意力要显式构造 [S, S] 的注意力矩阵——内存 O(S²)。Flash Attention 分块计算，内存 O(S)。

```python
# PyTorch 内置 SDPA（底层调 Flash kernel）
import torch.nn.functional as F
output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
```

MiniMind 的 Flash 条件比较严格：训练时走 SDPA，带 KV Cache 的推理走手写路径。

## 完整实现

> 📁 追加到 `scratch/model/model_minimind.py`
> 把 Step 3-7 的代码拼在一起，就是完整的 Attention 类。每行注释标注对应的 Step。

```python
import math
import torch.nn.functional as F

def repeat_kv(x, n_rep):                                                    # Step 3: GQA
    B, S, n_kv, d = x.shape
    if n_rep == 1: return x
    return x[:, :, :, None, :].expand(B, S, n_kv, n_rep, d).reshape(B, S, n_kv * n_rep, d)

class Attention(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8, n_kv_heads=4, head_dim=96):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim

        self.wq = nn.Linear(hidden_size, n_heads * head_dim, bias=False)    # Step 3: [768, 768]
        self.wk = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False) # Step 3: [768, 384] GQA
        self.wv = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False) # Step 3: [768, 384]
        self.wo = nn.Linear(n_heads * head_dim, hidden_size, bias=False)    # Step 7: [768, 768]
        self.q_norm = RMSNorm(head_dim)                                     # Step 4: QK Norm
        self.k_norm = RMSNorm(head_dim)                                     # Step 4: QK Norm

    def forward(self, x, cos, sin):
        B, S, _ = x.shape

        # Step 3: 投影 + 拆头
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)

        # Step 4: QK Norm
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # Step 5: RoPE
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # Step 3: GQA repeat
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Step 6: Causal Mask + 注意力计算
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.full((S, S), float("-inf"), device=x.device).triu(1)
        scores = scores + mask
        output = F.softmax(scores.float(), dim=-1).type_as(xq) @ xv

        # Step 7: 合并头 + 输出投影
        output = output.transpose(1, 2).reshape(B, S, -1)
        return self.wo(output)

# 验证（unsqueeze 已在 apply_rotary_pos_emb 内部处理）
attn = Attention()
x = torch.randn(2, 10, 768)
cos, sin = precompute_freqs_cis(96, 10)
out = attn(x, cos, sin)
assert out.shape == x.shape
print(f"Input: {x.shape}, Output: {out.shape}")
print("Attention verified!")
```

## 完整维度流（一张图总结）

```
输入 x: [B=2, S=128, 768]

Step 1: 线性投影
  xq = W_q(x)  →  [2, 128, 768]   "我想关注什么"
  xk = W_k(x)  →  [2, 128, 384]   "我能提供什么"  ← GQA 只有一半
  xv = W_v(x)  →  [2, 128, 384]   "我携带的内容"

Step 2: reshape 拆头
  xq: [2, 128, 8, 96]    8 个 Q 头
  xk: [2, 128, 4, 96]    4 个 KV 头
  xv: [2, 128, 4, 96]

Step 3: QK Norm + RoPE
  q_norm(xq), k_norm(xk)   → RMSNorm 防爆炸
  apply_rotary_pos_emb      → 注入位置信息

Step 4: repeat_kv + transpose
  xk: [2, 128, 4, 96] → [2, 128, 8, 96] → [2, 8, 128, 96]
  xv: 同上
  xq: [2, 8, 128, 96]

Step 5: 注意力计算
  scores = xq @ xk^T / √96       → [2, 8, 128, 128]
  + causal mask (上三角 -inf)
  output = softmax(scores) @ xv   → [2, 8, 128, 96]

Step 6: 合并头 + 输出投影
  output → [2, 128, 768] → W_o → [2, 128, 768]
```

## 对照 MiniMind 源码

| 我们的实现 | MiniMind 源码位置 |
|-----------|-----------------|
| `repeat_kv` | `model_minimind.py:86-89` |
| `Attention.__init__` | `model_minimind.py:91-108` |
| `Attention.forward` | `model_minimind.py:110-134` |
| Flash vs 手写分支 | `model_minimind.py:122-131` |

主要差异：MiniMind 额外处理了 KV Cache 拼接、padding mask、Flash Attention 条件判断。我们的简化版聚焦核心逻辑。

## 业界对比

| 特性 | MiniMind (64M) | Qwen2.5-7B | DeepSeek-V2 (236B) | GLM-4-9B |
|------|---------------|------------|-------------------|----------|
| hidden_size | 768 | 3584 | 5120 | 4096 |
| Q 头数 | 8 | 28 | 128 | 32 |
| KV 头数 | 4 (GQA) | 4 (GQA) | 1 (MQA/MLA) | 2 (GQA) |
| head_dim | 96 | 128 | 128 | 128 |
| QK Norm | 有 | 有 | 有 | 无 |
| Flash Attn | PyTorch SDPA | FlashAttention-2 | FlashAttention-2 | FlashAttention-2 |

- **DeepSeek-V2 的 MLA**：把 KV 低秩压缩（LoRA 思想），KV Cache 压缩到极致，比 GQA 更省显存
- **Qwen2.5 的 GQA**：Q=28 KV=4，n_rep=7，比我们的 n_rep=2 更激进
- **Flash Attention**：业界用独立的 FlashAttention-2 Triton kernel，MiniMind 用 PyTorch SDPA

## 面试话术

> "Attention 我是从最简的 QKV 点积开始写的——先写 MHA，再改成 GQA 省 KV Cache。
> GQA 就是让 W_k/W_v 的投影矩阵只有 W_q 的一半大小，8 个 Q 头共享 4 个 KV 头，
> 用 repeat_kv 把 4 头 expand 成 8 头再做注意力。KV Cache 只存 4 头，显存直接省一半。
> 还加了 QK Norm——在做 Q·K 点积之前先对每个头做 RMSNorm，防注意力分数爆炸。
> RoPE 在 QK Norm 之后、点积之前应用，只旋转 Q/K 不旋转 V。
> Flash Attention 方面，我用 PyTorch SDPA，业界大模型用 FlashAttention-2 Triton kernel。"

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

> 以下 Step 1-9 逐步演化 Attention 的设计，每步可直接在 Python REPL 运行验证。最终追加到 `scratch/model/model_minimind.py` 的完整 `Attention` 类在后面的 [完整实现](#完整实现) 部分。

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

#### GQA 的完整公式

MHA 的公式：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

$$\text{head}_i = \text{Attention}(x W^Q_i,\; x W^K_i,\; x W^V_i)$$

GQA 只改了一点——每 $n_{\text{rep}}$ 个 Q 头共享一组 KV：

$$\text{GQA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

$$\text{head}_i = \text{Attention}\!\left(x W^Q_i,\; x W^K_{\lfloor i/n_{\text{rep}} \rfloor},\; x W^V_{\lfloor i/n_{\text{rep}} \rfloor}\right)$$

$W^O$ 不变——它的作用是把 $h$ 个头各自 $d_k$ 维的输出 concat 成 $d_{\text{model}}$ 维后做**跨头融合**，让不同头的信息互相交互。

4 个矩阵各自的角色：

| 矩阵 | 形状 | 作用 |
|------|------|------|
| $W^Q$ | $[768, 768]$ | 8 组独立的 Q 投影 |
| $W^K$ | $[768, 384]$ | 4 组 KV 投影（比 MHA 小一半） |
| $W^V$ | $[768, 384]$ | 4 组 KV 投影 |
| $W^O$ | $[768, 768]$ | 把 8 个头的输出融合（concat 后的跨头混合） |

代码实现中，"Q 头 0,1 共用 KV 头 0"不是真的共用投影矩阵，而是用 `repeat_kv` 把 4 个 KV 头复制成 8 个再做注意力。

从这里开始逐步构建最终的 Attention 类。先写 `repeat_kv` 函数和 `__init__`（投影层），以及 forward 的第一段（投影 + 拆头 + repeat）：

```python
import math
import torch.nn.functional as F

def repeat_kv(x, n_rep):
    """GQA: 把 n_kv 个 KV 头虚拟复制成 n_heads 个"""
    B, S, n_kv, d = x.shape
    if n_rep == 1: return x
    # 三步链式操作，逐步理解：
    #
    # 第一步 x[:, :, :, None, :]
    #   None 等价于 unsqueeze()，在第 3 维之后插入一个 size=1 的新维度
    #   [B, S, n_kv, d] → [B, S, n_kv, 1, d]
    #
    # 第二步 .expand(B, S, n_kv, n_rep, d)
    #   沿 size=1 的维度广播到 n_rep 大小，不复制内存（只改变 stride）
    #   [B, S, n_kv, 1, d] → [B, S, n_kv, n_rep, d]
    #   expand 和 repeat 的区别：expand 不分配新内存，是"虚拟复制"
    #
    # 第三步 .reshape(B, S, n_kv * n_rep, d)
    #   把 n_kv 和 n_rep 两个维度合并 → [B, S, n_heads, d]
    #   例：n_kv=4, n_rep=2 → 4×2=8 个头，每对共享同一份 KV
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

    def forward(self, x):
        B, S, _ = x.shape

        # ---- Step 3: 投影 + 拆头 ----
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)    # [B, S, 8, 96]
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim) # [B, S, 4, 96]
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim) # [B, S, 4, 96]

        # ... Step 4 QK Norm, Step 5 RoPE(cos, sin), Step 6 Causal Mask,
        #     Step 7 W_O, Step 8 KV Cache 在后续逐步补充，
        #     每步会往 forward 签名里加对应参数 ...
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

### Step 6：Transpose + Causal Mask（遮住未来 token）

到这一步，我们已经有了经过 QK Norm 和 RoPE 的 xq/xk/xv，并且 KV 已经 repeat 到和 Q 一样的头数。接下来做三件事：transpose 调整维度顺序、算注意力分数、加 causal mask。

#### 为什么要 transpose？

矩阵乘法 `@` 只作用在**最后两个维度**，前面的维度全当 batch 并行处理。我们要的是：每个头内部，S 个 token 之间两两算注意力 → `[S, 96] @ [96, S] = [S, S]`。所以必须把"头"挪到前面当 batch，把"序列"和"head_dim"放到最后两维：

```
transpose 前: [B, S, 8, 96]  → @ 会在 (8, 96) 上做乘法 ✗ 不对
transpose 后: [B, 8, S, 96]  → @ 会在 (S, 96) 上做乘法 ✓
```

#### Causal Mask：为什么用 $-\infty$ 而不是 0？

自回归模型：每个 token 只能看到自己和之前的 token，不能"偷看未来"。

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

因为 $e^{-\infty} = 0$，softmax 归一化后它们严格消失。**只有 $-\infty$ 能保证信息完全隔离**。用 0 或任何有限负数都会泄漏信息。

#### 怎么生成 Causal Mask？`triu` 函数

`triu` = **tri**angle **u**pper（上三角）。`torch.full((S, S), -inf).triu(1)` 生成一个上三角全是 $-\infty$、其余为 0 的矩阵：

```
torch.full((4, 4), -inf).triu(1)

         pos0   pos1   pos2   pos3
  pos0 [   0,  -inf,  -inf,  -inf]    ← pos0 只能看自己
  pos1 [   0,    0,   -inf,  -inf]    ← pos1 能看 pos0, pos1
  pos2 [   0,    0,     0,   -inf]    ← pos2 能看 pos0, pos1, pos2
  pos3 [   0,    0,     0,     0 ]    ← pos3 能看所有
```

参数 `1` 表示从对角线往上 1 格开始填 $-\infty$，对角线本身保留为 0（每个 token 至少能看到自己）。如果写 `triu(0)` 连对角线也变 $-\infty$，token 连自己都看不到了。

把这个 mask **加到 scores 上**后，$-\infty$ 位置经过 softmax 变成 0 权重。mask 中的 0 位置加上去不影响原始分数——这就是 causal mask 的实现。

对应的还有 `tril`（**tri**angle **l**ower，下三角）和 `torch.diag`（对角线），MiniMind 没用到。

#### `__init__` 无新增

Step 6 不需要在 `__init__` 中新增任何东西。

#### `forward` 中接 Step 5 之后追加

```python
    def forward(self, x):
        ...  # Step 3: 投影 + 拆头
        ...  # Step 4: QK Norm
        ...  # Step 5: RoPE + repeat_kv

        # ---- Step 6: Transpose + Causal Mask ----

        # transpose: 把"头"挪到 batch 维
        xq = xq.transpose(1, 2)  # [B, S, 8, 96] → [B, 8, S, 96]
        xk = xk.transpose(1, 2)  # [B, S, 8, 96] → [B, 8, S, 96]
        xv = xv.transpose(1, 2)  # [B, S, 8, 96] → [B, 8, S, 96]

        # 注意力分数: Q·K^T / sqrt(d_k)
        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)  # [B, 8, S, S]

        # Causal Mask: 上三角填 -inf，遮住未来 token
        # 训练和推理都走这一行，不需要 if 判断——原因见下面的说明
        scores += torch.full((S, S), float("-inf"), device=x.device).triu(1)
```

> **Causal mask 在推理 decode 时不是多余的吗？**
>
> 是的——decode 时 Q 只有 1 个 token（S=1），没有"未来"可遮。但代码不需要加 `if S > 1` 判断，因为 `triu(1)` 在 S=1 时**自动退化为空操作**：
>
> ```
> S=1: torch.full((1, 1), -inf).triu(1) = [[0]]     ← 1×1 矩阵没有上三角，全是 0
> S=4: torch.full((4, 4), -inf).triu(1) = [[0, -inf, -inf, -inf], ...]  ← 正常遮未来
> ```
>
> `scores += [[0]]` 什么都没改，等于没加。所以同一行代码在训练（S>1，真遮）和 decode（S=1，空操作）两种场景下都是正确的。

> **为什么只有这一行需要 `device=x.device`？**
>
> `torch.full` 是**从零创建新 tensor**，默认放在 CPU 上。如果模型在 GPU 上跑，必须手动指定 device，否则 CPU tensor 和 GPU tensor 相加会报错。前面的 `self.wq(x)`、`xq.transpose(...)` 之类的操作是从已有 tensor 派生的，自动继承设备，不需要写 device。

### Step 7：Padding Mask（遮住填充 token）

Causal mask 遮的是"未来"，永远生效。但训练时还有第二种需要遮住的东西——**Padding token**。

#### 什么是 Padding？为什么需要遮住？

训练时一个 batch 里的序列长度往往不同，必须补齐到同一长度才能组成 tensor。补齐用的特殊 token（`[PAD]`）没有实际含义，不应该参与注意力计算：

```
序列 1: "我  爱  你 [PAD] [PAD]"   长度 3，补了 2 个 PAD
序列 2: "今  天  天  气    好"      长度 5，不需要补

组成 batch 后 shape = [2, 5, 768]，但序列 1 的后两个位置是无意义的填充
```

如果不遮住 padding，"你" 在计算注意力时会给 `[PAD]` 分配权重，引入垃圾信息。

#### `attention_mask` 的含义

`attention_mask` 是一个 `[B, S]` 的 0/1 矩阵：**1 = 有效 token，0 = 填充 token**。

```
序列 1 的 attention_mask: [1, 1, 1, 0, 0]   ← 后两个是 padding
序列 2 的 attention_mask: [1, 1, 1, 1, 1]   ← 全部有效
```

> **命名说明**：虽然叫 `attention_mask`，但它实际上是 **padding mask**。这个命名来自 HuggingFace 的约定，几乎所有 Transformer 模型都这么叫。不要和 Step 6 的 causal mask 混淆——它们是两种完全不同的 mask：
>
> | | Causal Mask (Step 6) | Padding Mask (Step 7) |
> |---|---|---|
> | **遮什么** | 未来 token | 填充 token |
> | **何时生效** | 永远（自回归必须） | 只在 batch 内有 padding 时 |
> | **遮住的值** | $-\infty$（严格隔离） | $-10^9$（足够小即可） |
> | **对应参数** | 无（自动加） | `attention_mask`（外部传入） |

#### 怎么把 padding 位置遮住？

目标：给 padding 位置的 scores **加上一个极小的负数**，让 softmax 之后它们的权重趋近于 0。

关键问题：`attention_mask` 里 **1=有效（保留），0=padding（遮住）**。但我们要给 **0 的位置**加惩罚，给 **1 的位置**加 0（不动）。直接拿 mask 乘 -1e9 方向是反的：

```
attention_mask          = [1, 1, 1, 0, 0]

直接乘:
mask * -1e9             = [-1e9, -1e9, -1e9, 0, 0]   ← 反了！有效位置被惩罚了

所以先用 1-mask 翻转:
1 - attention_mask      = [0, 0, 0, 1, 1]             ← 翻转: 0=保留, 1=要遮

再乘:
(1 - mask) * -1e9       = [0, 0, 0, -1e9, -1e9]       ← 对了！只有 padding 被惩罚

scores += 上面这个     → 有效位置 +0（不变），padding 位置 +(-1e9)（被压下去）
                       → softmax 后 padding 权重 ≈ 0 ✓
```

#### 为什么用 $-10^9$ 而不是 $-\infty$？

Causal mask 用 $-\infty$ 是因为需要**严格隔离**（自回归的正确性保证）。Padding mask 用 $-10^9$ 是因为：
- padding token 完全不携带信息，只需要让权重"足够小"即可
- $-10^9$ 经过 softmax 后已经是 ~0（$e^{-10^9} \approx 0$）
- 用 $-\infty$ 也行，但 $-10^9$ 在某些混合精度场景下数值更稳定

#### 维度广播：怎么从 `[B, S]` 加到 `[B, 8, S, S]`？

`attention_mask` 形状是 `[B, S]`，而 `scores` 形状是 `[B, 8, S, S]`，维度数都不一样，不能直接相加。需要用 `None` 索引插入两个维度：

```python
attention_mask[:, None, None, :]
#              B   ↑     ↑    S
#                  |     |
#            插入 head 维  插入 query 维
```

`None` 在索引中的作用就是"在这个位置插入一个大小为 1 的新维度"（和 `.unsqueeze()` 完全等价，写法更短）：

```
attention_mask:              [B, S]
attention_mask[:, None, None, :]:  [B, 1, 1, S]    ← 插入了两个维度

广播到 scores:               [B, 8, S, S]
  B 维:   不变
  1→8:    广播到每个头（所有头共享同一个 padding mask）
  1→S:    广播到每个 query token（不管 query 是谁，那个 key 是 padding 就要遮）
  S:      对应 key 维度，标记哪些 key 位置是 padding
```

为什么只在最后一个维度（key 维度）做 mask？因为 padding mask 要回答的问题是"这个 **key** 位置是不是填充"，和 query 是哪个 token 无关。

#### 两种 Mask 叠加效果

Causal mask（Step 6）和 Padding mask（Step 7）是叠加关系——一个位置只要被任一 mask 遮住，注意力权重就趋近于 0：

```
以序列 1 的 "你"(pos2) 为例，两种 mask 叠加后:

            我     爱     你    [PAD]  [PAD]
  你  [    ok,    ok,    ok,    ×,      ×   ]
          ↑↑↑          ↑↑↑    ↑↑↑     ↑↑↑
       causal允许     自己   padding  padding + causal
                             (-1e9)   (-1e9) + (-inf)

"ok" 的位置: causal 允许(+0) 且不是 padding(+0) → 原始分数不变
"×" 的位置:  被 padding(-1e9) 或 causal(-inf) 或两者同时遮住 → softmax 后 ≈ 0
```

#### `__init__` 无新增

Step 7 不需要在 `__init__` 中新增任何东西。

#### `forward` 签名新增 + 接 Step 6 之后追加

```python
    def forward(self, x, attention_mask=None):
        #                 ^^^^^^^^^^^^^^^^
        #                 新增参数: [B, S] 的 0/1 矩阵
        #                 1 = 有效 token，0 = padding token
        #                 推理时通常为 None（没有 padding，不需要遮）

        ...  # Step 6: transpose + scores + causal mask

        # ---- Step 7: Padding Mask（只在有 padding 时生效）----
        if attention_mask is not None:
            # attention_mask: [B, S] → [:, None, None, :] → [B, 1, 1, S] 广播到 [B, 8, S, S]
            # 1-mask 翻转: 有效(1)→0 不动, padding(0)→1 要遮
            # ×(-1e9): padding 位置加上极小值 → softmax 后权重 ≈ 0
            scores += (1.0 - attention_mask[:, None, None, :]) * -1e9
```

### Step 8：注意力计算 + Dropout + 合并头 + 输出投影

到这一步，scores 已经叠加了 causal mask（Step 6）和 padding mask（Step 7）。接下来做四件事：softmax 转概率 → dropout → 加权聚合 V → 合并头 + 输出投影。

#### `__init__` 中追加

```python
class Attention(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8, n_kv_heads=4, head_dim=96,
                 dropout=0.0):
        #        ^^^^^^^^^^^ 新增参数: dropout 比率，默认 0.0（不生效）
        ...  # 前面的 wq/wk/wv/wo/q_norm/k_norm 不变

        # ---- Step 8 新增 ----
        self.attn_dropout = nn.Dropout(dropout)    # softmax 之后、乘 V 之前
        self.resid_dropout = nn.Dropout(dropout)   # W_O 投影之后、送进残差之前
```

#### `forward` 中接 Step 7 之后追加

```python
        ...  # Step 7: padding mask

        # ---- Step 8: 注意力计算 + Dropout + 合并头 + 输出投影 ----

        # softmax: 把分数转成概率分布（每行加起来 = 1）
        # .float() 先转 float32 算 softmax（精度更高），再 .type_as(xq) 转回原精度
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Attention Dropout: 随机丢弃注意力连接（默认 0.0 = 不生效）
        attn_weights = self.attn_dropout(attn_weights)

        # 加权聚合 V: 注意力权重 × Value
        output = attn_weights @ xv                          # [B, 8, S, 96]

        # 合并头: 从 [B, heads, S, dim] 恢复为 [B, S, hidden_size]
        # transpose 把 head 维挪回去，reshape 把 8×96 拉平成 768
        output = output.transpose(1, 2).reshape(B, S, -1)   # [B, S, 8*96=768]

        # 输出投影 W_O: 跨头融合（让不同头的信息互相交互）
        output = self.wo(output)                              # [B, S, 768]

        # Residual Dropout: 送进残差连接之前的正则化（默认 0.0 = 不生效）
        output = self.resid_dropout(output)

        return output                                         # [B, S, 768]
```

#### 两个 Dropout 各管一件事

| Dropout | 位置 | 作用 |
|---------|------|------|
| `attn_dropout` | softmax 之后、乘 V 之前 | 随机丢弃注意力连接，防止过度依赖特定 token |
| `resid_dropout` | W_O 投影之后 | 随机丢弃输出维度，送进残差之前做正则化 |

MiniMind 默认 `dropout=0.0`，两个都不生效——和 01 章讲的一样，预训练不用 dropout。但架构上留口子，需要时改 config 就能打开。

### Step 9：KV Cache（推理加速）

训练时一次看完整序列，推理时逐 token 生成。没有 KV Cache 时，每生成一个 token 都要重算所有历史 token 的 K/V——浪费。

```
推理第 200 步（有 KV Cache）：
  Q: [B, 8, 1, 96]           ← 只有当前 token 的 Q
  K: [B, 8, 200, 96]         ← cat(past_K[199], new_K[1]) 从 cache 取
  scores: [B, 8, 1, 200]     ← 只算新 token 对所有历史的注意力

  GQA 只缓存 4 头 KV（不是 8 头）→ 显存省一半
```

**加在 forward 的哪里？** 在 RoPE 之后、repeat_kv 之前。注意缓存的是 4 个 KV 头（GQA 原始头数），不是 repeat 之后的 8 个：

```python
        # ---- Step 5: RoPE ----
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # ---- Step 9: KV Cache（加在这里）----
        # forward 签名变成（在 Step 7 的 attention_mask 基础上再新增两个参数）:
        #   forward(self, x, attention_mask=None, past_key_value=None, use_cache=False)
        #
        # past_key_value: 上一步缓存的 (K, V) tuple，首次调用时为 None
        # use_cache: 是否返回当前 KV 给下一步用（训练时 False，推理时 True）
        if past_key_value is not None:
            # 把历史 KV 和当前新 token 的 KV 沿 seq 维拼接
            xk = torch.cat([past_key_value[0], xk], dim=1)  # [B, past+new, 4, 96]
            xv = torch.cat([past_key_value[1], xv], dim=1)  # [B, past+new, 4, 96]
        # 如果需要缓存，把当前完整的 KV 存下来给下一步用
        past_kv = (xk, xv) if use_cache else None

        # ---- 然后才 repeat_kv + transpose + 注意力计算（Step 5/6/7/8）----
        xk = repeat_kv(xk, self.n_rep)  # [B, total_seq, 8, 96]
        xv = repeat_kv(xv, self.n_rep)
        ...

        # ---- Step 8: 返回值也要变 ----
        return self.wo(output), past_kv  # 多返回一个 past_kv
```

KV Cache 的形状变化：

```
缓存存的是:   [B, total_seq_len, 4, 96]    ← 4 个 KV 头（省显存）
repeat 之后:  [B, total_seq_len, 8, 96]    ← 8 个头（只用于计算，不存）
```

同时 causal mask 也要配合改——`scores[:, :, :, -S:]` 只对新 token 部分加 mask：

```python
        # 原来: mask 是 [S, S] 的
        # 有 cache 时: scores 是 [B, 8, S_new, S_total]
        #   只需要对最后 S_new 列加 causal mask
        scores[:, :, :, -S:] += torch.full((S, S), float("-inf"), device=x.device).triu(1)
```

### Step 10：Flash Attention（SDPA 分支）

#### 标准 Attention 的瓶颈（一句话版）

Step 6-8 的手写 Attention 必须把完整的 $[S, S]$ 注意力矩阵存在显存里，且反复在 HBM 上读写它——瓶颈不是算力而是显存带宽。Flash Attention 的解法是分块（tiling）：Q/K/V 切成小块在 SRAM 里算完 softmax + V 加权，配合 online softmax，不存完整 $[S, S]$ 矩阵，速度快 2-4 倍、显存从 O(S²) 降到 O(S)，数学上完全等价。

> 原理深挖（GPU 存储层级、online softmax 递推公式、反向 recomputation、v1/v2/v3 演进）和面试问答见 [04a-Attention变体演进 §六](04a-Attention变体演进-todo.md)，本章只关心怎么接入。

#### MiniMind 怎么用？

MiniMind 不自己实现 Flash Attention，而是调用 PyTorch 内置的 `F.scaled_dot_product_attention`（SDPA）。SDPA 是一个调度接口，根据硬件和输入条件自动选择最优的后端（Flash、Memory-Efficient 或普通 Math）。

训练时走 SDPA（快），带 KV Cache 的推理走手写路径（因为 Q/K 不等长，SDPA 的 causal 假设不成立）。

#### `__init__` 中追加

在 Step 8 已有的 `attn_dropout`/`resid_dropout` 基础上，新增两个属性：

```python
class Attention(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8, n_kv_heads=4, head_dim=96,
                 dropout=0.0, flash_attn=True):
        #                  ^^^^^^^^^^^^^^^^^ 新增参数: 是否启用 SDPA
        ...  # 前面的 wq/wk/wv/wo/q_norm/k_norm 不变
        ...  # Step 8 已有: self.attn_dropout, self.resid_dropout

        # ---- Step 10 新增 ----
        self.dropout = dropout   # float 值，给 SDPA 的 dropout_p 参数用（区别于 nn.Dropout 对象）
        self.flash = hasattr(F, 'scaled_dot_product_attention') and flash_attn
```

注意 `flash_attn` 是构造函数的**布尔参数**，控制是否启用 SDPA 路径。不需要 `import flash_attn`——PyTorch 自带的 `F.scaled_dot_product_attention` 已经内置了多种后端调度。

#### 加在 forward 的哪里？

把 Step 6-8 的注意力计算包进一个 if/else 分支——能走 SDPA 就走 SDPA，不能走的退回 Step 6-8 的手写路径：

```python
        # ---- Step 10: 把 Step 6-8 的注意力计算改成 SDPA/手写 分支 ----

        if self.flash and S > 1 and past_key_value is None and \
                (attention_mask is None or torch.all(attention_mask == 1)):
            # torch.all(attention_mask == 1): 检查张量中是否**所有**元素都为 1
            #   即"是否所有 token 都是有效的（没有 padding）"
            #   有 padding → 不能走 SDPA（SDPA 不支持自定义 padding mask）
            #
            # SDPA 路径（训练时走这里）
            # 条件：flash 开启 + 不是单 token decode + 没有 cache + 没有 padding mask
            # SDPA 内部自动处理 causal mask 和 scaled dot product
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # 手写路径（带 KV Cache 的推理、有 padding mask 时走这里）
            # 就是 Step 6-8 的代码，causal mask + padding mask + softmax 都在这里
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -S:] += torch.full((S, S), float("-inf"), device=x.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask[:, None, None, :]) * -1e9
            output = self.attn_dropout(
                F.softmax(scores.float(), dim=-1).type_as(xq)
            ) @ xv

        # ---- Step 8 的合并头 + 输出投影（两条路径共用）----
        output = output.transpose(1, 2).reshape(B, S, -1)
        return self.resid_dropout(self.wo(output)), past_kv
        #      ^^^^^^^^^^^^^^^^^ 输出投影后加 dropout（默认 0.0 = 不生效）
```

#### 为什么带 KV Cache 时不走 SDPA？

SDPA 的 `is_causal=True` 假设 Q 和 K 序列等长，内部自动生成一个 `[S_q, S_k]` 的 causal mask。但 decode 时 Q 只有 1 个 token 而 K 有几百个（Q 和 K 不等长），这个假设不成立，所以必须走手写路径自己加 mask。

实际上 decode 时 Q 只有 1 行，没有"未来"可言，根本不需要 causal mask。但 MiniMind 简化处理，统一退回手写路径。生产级推理框架（vLLM、flash-attn 三方包）有专门的 decode kernel，不受这个限制。

### Attention 在哪里被调用？

Attention 不会独立运行，它是 MiniMindBlock（07 章）的一个子模块：

```
MiniMindForCausalLM（08 章）
  └→ MiniMindModel
       └→ 8 × MiniMindBlock（07 章）
              ├→ RMSNorm → Attention（本章） → + 残差
              └→ RMSNorm → FFN/MoE（05/06 章） → + 残差
```

调用链：

```python
# MiniMindModel.forward 中:
cos = self.freqs_cos[start_pos : start_pos + seq_len]   # 切出当前位置的 RoPE
sin = self.freqs_sin[start_pos : start_pos + seq_len]
position_embeddings = (cos, sin)

for layer in self.layers:                                # 8 层 Block
    hidden_states, present = layer(hidden_states, position_embeddings, ...)

# MiniMindBlock.forward 中:
hidden_states, present = self.self_attn(                 # 调用本章的 Attention
    self.input_layernorm(hidden_states),                 # 先 Norm
    position_embeddings,                                 # cos/sin 传进来
    past_key_value, use_cache, attention_mask
)
hidden_states = hidden_states + residual                 # 残差连接
```

所以 forward 签名里的 cos/sin 是从最外层的 MiniMindModel 一路传进来的，不是 Attention 自己算的。

## 完整实现（与上游对齐）

> 📁 `scratch/model/model_minimind.py` 中的 Attention 已升级为完整版本，包含：
> - GQA + QK Norm + RoPE（与前面教学代码相同）
> - **KV Cache**：past_key_value 拼接 + use_cache 返回
> - **SDPA / 手写分支**：训练时走 SDPA，带 cache 的推理走手写路径
> - **attention_mask**：支持 padding mask
> - **Dropout**：attn_dropout + resid_dropout（默认 0.0）
>
> 接受 `MiniMindConfig` 对象初始化，forward 签名与上游一致：
> `forward(x, position_embeddings, past_key_value, use_cache, attention_mask)`

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.wq = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None,
                use_cache=False, attention_mask=None):
        B, S, _ = x.shape
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)               # QK Norm
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)         # RoPE

        if past_key_value is not None:                             # KV Cache 拼接
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)                                   # [B, heads, S, dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)            # GQA repeat
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # torch.all(x == 1): 所有元素都为 1 才返回 True → 判断有没有 padding
        if self.flash and S > 1 and past_key_value is None and \
                (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(               # SDPA 路径
                xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -S:] += torch.full((S, S), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask[:, None, None, :]) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        output = output.transpose(1, 2).reshape(B, S, -1)
        return self.resid_dropout(self.wo(output)), past_kv

# 验证
config = MiniMindConfig()
attn = Attention(config)
x = torch.randn(2, 10, 768)
cos, sin = precompute_freqs_cis(96, 10)
out, _ = attn(x, (cos, sin))
assert out.shape == x.shape

# KV Cache 验证
out1, kv = attn(x, (cos, sin), use_cache=True)
cos2, sin2 = precompute_freqs_cis(96, 11)
new_x = torch.randn(2, 1, 768)
out2, kv2 = attn(new_x, (cos2[10:11], sin2[10:11]), past_key_value=kv, use_cache=True)
assert out2.shape == (2, 1, 768)
assert kv2[0].shape == (2, 11, 4, 96)  # 10 + 1 = 11 tokens cached
print("Attention (full interface + KV Cache) verified!")
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

我们的完整实现（上面的代码）已经与上游对齐，包含 GQA + QK Norm + RoPE + KV Cache + Padding Mask + SDPA/手写分支 + Dropout。主要差异在于上游用 `MiniMindConfig` 对象传参，我们教学代码的 Step 3-8 用 kwargs 逐步构建。

## 业界对比

| 特性 | MiniMind (64M) | Qwen2.5-7B | DeepSeek-V2 (236B) | GLM-4-9B |
|------|---------------|------------|-------------------|----------|
| hidden_size | 768 | 3584 | 5120 | 4096 |
| Q 头数 | 8 | 28 | 128 | 32 |
| KV 头数 | 4 (GQA) | 4 (GQA) | MLA（512 维 latent 压缩，非标准 GQA） | 2 (GQA) |
| head_dim | 96 | 128 | Q/K: 128+64=192, V: 128 | 128 |
| QK Norm | 有 | 无 | 有（latent RMSNorm，非标准 per-head QK Norm） | 无 |
| Flash Attn | PyTorch SDPA | FlashAttention-2 | FlashAttention-2 | FlashAttention-2 |

- **DeepSeek-V2 的 MLA**：把 KV 低秩压缩到 512 维 latent，KV Cache 压缩到极致，比 GQA 更省显存；Q/K 维度拆分为 128 nope + 64 RoPE
- **Qwen2.5 的 GQA**：Q=28 KV=4，n_rep=7，比我们的 n_rep=2 更激进；无显式 q_norm/k_norm
- **Flash Attention**：业界用独立的 FlashAttention-2 Triton kernel，MiniMind 用 PyTorch SDPA

## 面试话术

> "Attention 我是从最简的 QKV 点积开始写的——先写 MHA，再改成 GQA 省 KV Cache。
> GQA 就是让 W_k/W_v 的投影矩阵只有 W_q 的一半大小，8 个 Q 头共享 4 个 KV 头，
> 用 repeat_kv 把 4 头 expand 成 8 头再做注意力。KV Cache 只存 4 头，显存直接省一半。
> 还加了 QK Norm——在做 Q·K 点积之前先对每个头做 RMSNorm，防注意力分数爆炸。
> RoPE 在 QK Norm 之后、点积之前应用，只旋转 Q/K 不旋转 V。
> Flash Attention 方面，我用 PyTorch SDPA，业界大模型用 FlashAttention-2 Triton kernel。"

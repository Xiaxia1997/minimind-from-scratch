# 03 - RoPE：让模型知道 token 的位置

## 在整个模型中的位置

```
Embedding (01) → RMSNorm (02) → Attention (04) → 残差 → RMSNorm → FFN (05) → 残差
                                     ↑
                               【RoPE (03)】在 Attention 内部对 Q/K 做旋转
                                本章写的函数在 04 章 Attention 的 forward 中被调用
```

前面两章我们写好了：
- **01 Embedding**：token id → 768 维向量（`scratch/model/model_minimind.py`）
- **02 RMSNorm**：稳定每层输入的数值范围

注意 RoPE 不是数据流中的独立模块——它**嵌在 Attention 内部**，在 Q/K 投影之后、点积之前对 Q 和 K 做旋转。本章先单独写好 `precompute_freqs_cis` 和 `apply_rotary_pos_emb` 两个函数，04 章 Attention 会调用它们。

## 为什么需要位置编码？

Attention 本身是**位置无关**的——"我 爱 你" 和 "你 爱 我" 的 Embedding 完全一样（只是顺序不同），但 Attention 只看向量内容做点积，不知道谁在前谁在后。

所以必须把"位置信息"注入进去，让模型区分不同位置的 token。

## 从零开始写

### Step 1：绝对位置编码——它有什么问题

> ⚠️ 以下是 2017 年原始 Transformer 的旧方案，**MiniMind 不用这种方式**。
> 这里只用于对比理解"为什么 RoPE 更好"，不需要写进我们的代码。

原始 Transformer 的做法：给每个位置一个固定向量，直接**加到 Embedding 上**。伪代码示意：

```python
for pos in range(max_len):
    for i in range(0, hidden_size, 2):
        PE[pos, 2i]   = sin(pos / base^(2i/d))    # base=10000
        PE[pos, 2i+1] = cos(pos / base^(2i/d))

x = embedding(token_ids) + PE[:seq_len]            # 直接加到 Embedding 上
```

数据流——位置信息在最前面就注入了，后面所有层都带着：

```
token_ids
    ↓
Embedding    → [batch, seq, 768]
    ↓
 + PE(pos)   ← 位置编码直接加在这里（Norm 之前、Attention 之前）
    ↓
RMSNorm → Attention → + 残差 → RMSNorm → FFN → + 残差
                ↑
          Q·K 点积里的位置信息 = (x+PE)^T · W · (x+PE)
          同时依赖 m 和 n 的绝对值，无法只表达相对距离
```

**三个问题**：
- 注意力分数 `(x_m + PE(m))^T · W · (x_n + PE(n))` 同时依赖 m 和 n 的**绝对值**，无法只表达相对距离
- 超出训练长度时属于分布外推，效果可能下降（固定 sinusoidal PE 仍能计算未见位置，但 Learned PE 会索引越界）
- PE 加上去之后经过 RMSNorm、FFN 等非线性变换，位置信息会被**扭曲和稀释**

### Step 2：我们想要什么——数学目标

绝对位置编码的问题本质是：**点积里同时出现了 m 和 n 的绝对位置，我们想只保留相对距离 m-n**。

数学表述：找一个函数 f，对 Q 和 K 做变换，使得：

$$f(\mathbf{q}, m) \cdot f(\mathbf{k}, n) = g(\mathbf{q}, \mathbf{k}, m-n)$$

其中右侧只取决于相对距离 $m-n$，不取决于 $m$ 或 $n$ 的绝对值。

RoPE 的解法是：**不在 Embedding 层加位置信息，而是在 Attention 内部用旋转矩阵乘 Q/K**：

```
           绝对位置编码                              RoPE
           ──────────                          ──────────

      token_ids                            token_ids
          ↓                                    ↓
      Embedding                            Embedding
          ↓                                    ↓
    ★ + PE(pos) ← 在这加                    （不加任何位置信息）
          ↓                                    ↓
      RMSNorm                              RMSNorm
          ↓                                    ↓
   ┌──Attention──┐                    ┌──Attention──────────────┐
   │  x → Q,K,V  │                    │  x → Q, K, V            │
   │  Q·K → score │                    │  ★ Q = Rot(m)·Q ← 在这旋转
   └──────────────┘                    │  ★ K = Rot(n)·K ← 在这旋转
                                       │  Q·K → score            │
                                       │  (V 不旋转)              │
                                       └──────────────────────────┘
```

接下来的问题是：这个 Rot 旋转具体怎么设计，才能让 Q·K 点积只含 m-n？

### Step 3：二维旋转是最小构造

RoPE 论文从最简单的情况开始——**先只看 2 维的 q 和 k**。

把二维向量 (x₁, x₂) 看成复数 x₁ + ix₂：

$$f(x, m) = x \cdot e^{im\theta}$$

这就是复数乘法，等价于二维旋转。验证：

$$f(q, m) \cdot f(k, n)^* = q \cdot e^{im\theta} \cdot k^* \cdot e^{-in\theta} = q \cdot k^* \cdot e^{i(m-n)\theta}$$

结果中只有 $m-n$，绝对位置消失了。

写成矩阵形式就是二维旋转矩阵：

$$f(\mathbf{x}, m) = \text{Rot}(m\theta) \cdot \mathbf{x} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$

**为什么是二维旋转，不是一维或三维？**
- **一维**：没有旋转。只能缩放（乘以标量），会改变向量大小，破坏语义信息
- **二维**：一个角度唯一确定一个旋转。位置 m → 角度 mθ，映射天然且唯一
- **三维**：旋转需要指定旋转轴 + 旋转角，但预先固定旋转轴后也可行，只是不如二维简洁
- 数学上，**二维旋转是最小的非平凡实数正交旋转块**，便于扩展成分块对角矩阵

### Step 4：从 2 维推广到 96 维——三个关键问题

RoPE 不是对完整的 768 维向量做旋转，而是对**每个 Attention head 内部的 96 维**做旋转。

768 → 96 的来龙去脉（[04-Attention](04-Attention.md) 会详细讲 Multi-Head 的设计动机）：

```
Attention 把 768 维拆成 8 个头（head），每个头 96 维，独立做注意力：

x: [B, S, 768]
    ↓ W_q 投影
Q: [B, S, 768] → reshape → [B, S, 8, 96]
                                    ╰─╯
                               每个头 96 维，RoPE 在这个维度上做旋转

head_dim = hidden_size / n_heads = 768 / 8 = 96
```

为什么 RoPE 在 head_dim=96 上做而不是 hidden_size=768 上？因为注意力的 Q·K 点积是**在每个头内部独立计算的**——head 0 的 Q（96 维）只和 head 0 的 K（96 维）做点积，所以 RoPE 的旋转也只需要在这 96 维内操作。

2 维的解有了，96 维怎么办？

#### Q1：为什么 MiniMind 对所有 96 维做旋转？

MiniMind 把全部 head_dim 都参与旋转，让位置信息均匀分布在整个向量上。

只旋转部分维度（Partial RoPE）也是可行的设计——比如 DeepSeek MLA 就只对 Q/K 的一部分维度应用 RoPE，另一部分保持纯语义。不同设计在位置容量和语义容量之间做不同的分配取舍。

做法：**把 96 维拆成 48 个独立的二维平面，每个平面各自做二维旋转**：

```
Q 向量（96维）:
[q0, q1,  q2, q3,  q4, q5,  ...,  q94, q95]
 ╰──╯    ╰──╯    ╰──╯           ╰───╯
 平面0    平面1    平面2    ...    平面47
 旋转θ₀   旋转θ₁   旋转θ₂         旋转θ₄₇
```

数学上是一个分块对角旋转矩阵（详见 [03b-线性代数补充](03b-线性代数-旋转矩阵与Givens分解.md)）：

```
         ┌ Rot(θ₀)                    ┐   ┌ q0  ┐
         │        Rot(θ₁)             │   │ q1  │
R(m) · q = │              Rot(θ₂)       │ · │ q2  │
         │                    ...     │   │ ... │
         └                    Rot(θ₄₇)┘   └ q95 ┘

  48 个独立的 2×2 旋转块，拼成一个 96×96 矩阵
```

**不是"96 变成了 48"，而是 96 维旋转 = 48 个独立的二维旋转拼起来。**

#### Q2：为什么 48 个平面的旋转频率必须不同？

排除两种错误方案：

**错误方案 A：全用高频率（ω₀ ≈ 1.0）**

48 对都用 $\omega = 1.0$，位置 $m$ 的旋转角度 $= m \times 1.0$：

- 位置 0：角度 $= 0$
- 位置 1：角度 $= 1.0$ rad
- 位置 2：角度 $= 2.0$ rad
- ...
- 位置 6：角度 $\approx 6.28 \approx 2\pi \approx 0$ -- 转了一整圈，和位置 0 一模一样！

高频转得快，近距离精度高（位置 0 和 1 差 57°，很清晰），但周期短——**每 6 个 token 就转一圈**，位置 0 和位置 6 无法区分，远距离全乱了。

**错误方案 B：全用低频率（ω₄₇ ≈ 1.3×10⁻⁶）**

48 对都用 $\omega = 1.3 \times 10^{-6}$：

近距离：
- 位置 0 vs 位置 1：角度差 $= 1.3 \times 10^{-6} \text{ rad} \approx 0.000075°$
- $\cos(0.000075°) \approx 1.0$，相邻 token 的旋转向量几乎完全重合
- 模型分不清 "我爱你" 和 "我你爱" -- 近距离完全糊掉

远距离：
- 位置 0 vs 位置 10000：角度差 $= 0.013 \text{ rad} \approx 0.76°$ -- 仍然很小，区分度有限

低频转得慢，远距离不碰撞，但近距离角度差太小，模型看不出区别。

**正确方案：不同频率组合 = 组合锁原理**

#### Q3：位置 m 做了什么？（不是变成向量，是生成旋转参数）

位置 m 就是一个数字（0, 1, 2, ..., 9）。它没有"变成 96 维向量"。

它做的事是：**生成 48 个旋转角度，用来旋转 Q/K 的 48 个二维平面**：

位置 $m=3$ 时：

- $\theta_0 = 3 \times \omega_0 = 3.0$ -- 平面 0 转 3.0 弧度
- $\theta_1 = 3 \times \omega_1 = 2.61$ -- 平面 1 转 2.61 弧度
- $\theta_2 = 3 \times \omega_2 = 2.28$
- ...
- $\theta_{47} = 3 \times \omega_{47} = 0.00039$ -- 平面 47 几乎不转

代码里预计算的 `freqs_cos`/`freqs_sin` 形状是 `[seq_len, 96] = [10, 96]`（10 个位置 × 每个位置 96 个 cos/sin 值）。这不是"位置向量"，是"旋转参数"——告诉每个位置的 Q/K 该怎么转。

### Step 5：频率公式——和 sinusoidal PE 的关系

RoPE 的频率公式和原始 Transformer 的 sinusoidal PE **完全一样**：

$$\omega_i = \frac{1}{\text{base}^{\,2i\,/\,\text{dim}}}$$

其中：
- $i$ = 维度对的编号（0, 1, 2, ..., 47），对应 Q/K 向量中的第 $i$ 对二维坐标
- $\text{dim}$ = head_dim = 96
- $\text{base}$ = 1,000,000（MiniMind）

位置 $m$ 在第 $i$ 对上的旋转角度：

$$\theta_i(m) = m \cdot \omega_i = \frac{m}{\text{base}^{\,2i\,/\,\text{dim}}}$$

代入 MiniMind 的具体数字（base = 10⁶, dim = 96）：

| 维度对 $i$ | 频率 $\omega_i = 10^{-6 \times 2i/96}$ | 周期 $2\pi / \omega_i$ | 角色 |
|:-:|:-:|:-:|:-:|
| 0 | $10^{0} = 1.0$ | ≈ 6 | 秒针（区分相邻 token） |
| 1 | $10^{-0.125} \approx 0.75$ | ≈ 8 | |
| 2 | $10^{-0.25} \approx 0.56$ | ≈ 11 | |
| ... | 几何级数递减 | | |
| 47 | $10^{-5.875} \approx 1.3 \times 10^{-6}$ | ≈ 4,800,000 | 时针（区分远距离 token） |

频率按**几何级数**递减（不是等差），高频段密集、低频段稀疏，类似对数刻度。这就是 Step 4 Q2 中数字的来源——pair 0 周期 ≈ 6 对应"高频，区分近距离"，pair 47 周期极长对应"低频，区分远距离"。

|  | sinusoidal PE（原始 Transformer） | RoPE（LLaMA/Qwen/MiniMind） |
|--|------|------|
| 频率公式 | $\omega_i = 1\,/\,\text{base}^{2i/\text{dim}}$ | 完全一样 |
| 操作 | **加法**: $x + [\sin,\cos]$ | **乘法（旋转矩阵）**: $R(\theta) \cdot x$ |
| 作用位置 | Embedding 层（Norm 前） | Attention 内部（Q/K 投影后） |
| 位置性质 | 绝对位置 | 相对位置（点积中 $m$、$n$ 抵消） |

**RoPE 的改进不在频率设计上（频率一样），而在操作上：旋转代替加法**——这让点积中自动只含相对位置。

**base 的作用——周期范围的推导**：

每个维度对的周期 $T_i = 2\pi / \omega_i = 2\pi \cdot \text{base}^{2i/\text{dim}}$，所以：

$$T_{\min} = 2\pi \cdot \text{base}^{0} = 2\pi \qquad\qquad T_{\max} = 2\pi \cdot \text{base}^{(\text{dim}-2)/\text{dim}}$$

指数取不到 1（因为 $i$ 最大是 $\text{dim}/2 - 1$，不是 $\text{dim}/2$），所以 $T_{\max}$ 略小于 $2\pi \cdot \text{base}$：

MiniMind 中 dim=96，指数为 $94/96 \approx 0.979$，$T_{\max} \approx 4.7 \times 10^6$，略小于天花板 $2\pi \cdot \text{base} \approx 6.3 \times 10^6$。dim 越大这个差距越小，但影响不大——**真正决定周期上界的是 base**：

```
原始 Transformer: base = 10,000    → T_max ≈ 6.3 万
LLaMA-3:         base = 500,000   → T_max ≈ 314 万
MiniMind:        base = 1,000,000 → T_max ≈ 470 万
```

注意：base 决定的是最低频维度的周期上界，但**模型实际支持的上下文长度不是由 base 单独决定的**——还取决于训练长度、RoPE scaling 策略、Attention 实现和评测方式。不能把 T_max 直接等同于"支持 XX token"。

注意：base 决定的是 RoPE 的位置编码范围，但模型的实际上下文长度还受多个因素限制：

| 瓶颈 | 原因 | 相关章节 |
|------|------|---------|
| KV Cache 显存 | 每个 token 都要存 K 和 V，1M token 显存吃不消 | [04-Attention](04-Attention.md) 讲 GQA 如何省 KV，DeepSeek MLA 进一步压缩 |
| Attention O(n²) 计算量 | 标准 Attention 是 n² 复杂度 | [17-FlashAttention](17-FlashAttention.md) 讲如何加速，但不改复杂度 |
| 训练分布 | 训练时只见过 4k 长度，推理给 32k 就是 OOD | 本章 YaRN 章节讲推理时频率缩放 |
| 稀疏/线性 Attention | 滑动窗口、分块稀疏、Mamba SSM 等 | [20-长上下文技术](20-长上下文技术.md) |

支持 1M 上下文的模型（Claude、Gemini）需要综合以上所有技术，不只是 RoPE base 调大。

### Step 6：Q·K 点积为什么只含相对位置？

这是 RoPE 的数学精髓。**先看一个平面（二维），再推广。**

#### 6.1 单个平面的推导

平面 i 上，位置 m 的 Q 转了 mω_i，位置 n 的 K 转了 nω_i：

$$Q_m^{(i)} = \text{Rot}(m \cdot \omega_i) \cdot \mathbf{q}_i, \qquad K_n^{(i)} = \text{Rot}(n \cdot \omega_i) \cdot \mathbf{k}_i$$

其中 $\mathbf{q}_i$、$\mathbf{k}_i$ 是原始的二维分量。点积：

$$Q_m \cdot K_n = \mathbf{q}_i^T \cdot \text{Rot}(m \cdot \omega_i)^T \cdot \text{Rot}(n \cdot \omega_i) \cdot \mathbf{k}_i$$

由旋转矩阵的正交性 $\text{Rot}(a)^T \cdot \text{Rot}(b) = \text{Rot}(b-a)$：

$$= \mathbf{q}_i^T \cdot \text{Rot}((n-m) \cdot \omega_i) \cdot \mathbf{k}_i$$

结果中只有 $n-m$，绝对位置 $m$、$n$ 消失了。

#### 6.2 直觉图解

```
    平面i 上：

         K_n（转了 n 步）
        ·╱
        ╱  角度差 = (n-m) × ωi
       ╱      ← 只取决于距离 n-m
      ·
    Q_m（转了 m 步）

    例: pos3 的 Q 和 pos7 的 K  → 角度差 = 4 × ωi
        pos100 的 Q 和 pos104 的 K → 角度差 = 4 × ωi  ← 一样！

    "位置3看位置7" 和 "位置100看位置104" 的 RoPE 旋转因子相同
    因为相对距离都是 4（但实际注意力分数还取决于 token 内容，通常不完全相同）
```

#### 6.3 所有平面加起来

48 个平面各自独立算点积，最终的注意力分数是所有平面的点积之和：

$$\text{score}(m,n) = \sum_{i=0}^{47} \mathbf{q}_i^T \cdot \text{Rot}((n-m) \cdot \omega_i) \cdot \mathbf{k}_i$$

每个平面的角度差都只含 $n-m$。

每一项都只取决于 n-m → 总和也只取决于 n-m。**相对位置编码成立。**

这就是为什么 RoPE **必须放在 Attention 内部**：旋转矩阵的"抵消"只在 Q·K 点积中发生。如果放在 Embedding 层，后续的 Norm/FFN 会破坏正交性，`Rot(a)^T · Rot(b) = Rot(b-a)` 不再成立，退化为绝对位置编码。

### Step 7：写预计算频率表

前面推导了旋转角度 $\theta_i(m) = m \cdot \omega_i$，旋转时需要 $\cos\theta$ 和 $\sin\theta$。这些值只取决于位置 $m$ 和维度对 $i$，和输入数据无关——所以可以**在初始化时一次性算好所有位置的 cos/sin 表**，训练和推理时直接查表，不用每次重算。

> 📁 追加到 `scratch/model/model_minimind.py`（不需要重复 import）

```python
def precompute_freqs_cis(dim, end=32768, rope_base=1e6):
    # 1. 算 48 个维度对的频率 ω_i = 1/base^(2i/dim)
    #    这就是 Step 5 的频率公式
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))
    # shape: [dim//2] = [48]，从高频(ω₀≈1)到低频(ω₄₇≈1e-6)

    # 2. 算角度矩阵 θ_i(m) = m × ω_i
    #    外积：每个位置 m 乘以每个频率 ω_i
    t = torch.arange(end)              # [0, 1, ..., 32767] 所有位置
    freqs = torch.outer(t, freqs)      # [32768, 48] 角度矩阵
    # freqs[m, i] = m × ω_i = 位置 m 在第 i 对上的旋转角度

    # 3. 算 cos/sin，cat 两份拼成 head_dim=96 长度
    #    为什么拼两份？因为 rotate_half 的实现方式需要 96 个 cos/sin 值
    #    （前 48 个给原始坐标用，后 48 个给 rotate_half 交换后的坐标用）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [32768, 96]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # [32768, 96]
    return freqs_cos, freqs_sin
```

整个计算流程：

```
Step 5 的频率公式                    Step 4 Q3 的角度
      ↓                                  ↓
ω: [48]  ──outer──→  θ: [32768, 48]  ──cos/sin──→  [32768, 96]
  48个频率    ×位置       每个位置每对的角度          查表用的cos/sin
```

这个表只算一次，注册为 buffer（不参与梯度更新），每次 forward 按位置切片取用。

#### 为什么 cos/sin 要拼两份（48 → 96）？

角度只有 48 个（48 对维度，每对一个角度），但 `q * cos` 是**逐元素乘**——q 有 96 维，cos 也必须有 96 维，否则长度不匹配。

用 dim=4（2 对）的例子展开整个计算过程：

q 的维度配对方式：**前后半截配对**，不是相邻配对。
- 对 0：(q[0], q[2])，用角度 θ₀
- 对 1：(q[1], q[3])，用角度 θ₁

```
q             = [q0,   q1,   q2,   q3]

rotate_half(q):
  后半取负:   [-q2, -q3]
  前半不变:   [q0,   q1]
  拼起来:     [-q2, -q3,  q0,  q1]

cos (拼两份): [cosθ₀, cosθ₁, cosθ₀, cosθ₁]
sin (拼两份): [sinθ₀, sinθ₁, sinθ₀, sinθ₁]
                前半=对0,对1     后半=同样的值再来一遍
```

套进公式 `q * cos + rotate_half(q) * sin`，逐位置展开：

```
位置 0:  q0·cosθ₀ + (-q2)·sinθ₀  =  q0·cosθ₀ - q2·sinθ₀   ← 对0的第一个分量 ✓
位置 1:  q1·cosθ₁ + (-q3)·sinθ₁  =  q1·cosθ₁ - q3·sinθ₁   ← 对1的第一个分量 ✓
位置 2:  q2·cosθ₀ +   q0·sinθ₀   =  q2·cosθ₀ + q0·sinθ₀   ← 对0的第二个分量 ✓
位置 3:  q3·cosθ₁ +   q1·sinθ₁   =  q3·cosθ₁ + q1·sinθ₁   ← 对1的第二个分量 ✓
```

位置 2 用了 `cos[2] = cosθ₀`，位置 3 用了 `cos[3] = cosθ₁`——和位置 0、1 是**同一组角度**。如果 cos 只有 `[cosθ₀, cosθ₁]` 两个值，位置 2、3 就没有对应的值可乘了。所以必须拼两份，让前半和后半都能拿到相同的角度。

### Step 8：写旋转函数

有了预计算的 cos/sin 表，旋转函数要做的就是 Step 3 推导的旋转矩阵乘法。

直接构造 96×96 的旋转矩阵再做矩阵乘太浪费——48 个 2×2 的分块对角矩阵，大部分元素是 0。代码用一个巧妙的展开：把旋转矩阵乘法拆成**逐元素乘法 + 一次坐标交换**。

从 Step 3 的二维旋转矩阵展开：

$$x_1' = x_1 \cos\theta - x_2 \sin\theta$$
$$x_2' = x_2 \cos\theta + x_1 \sin\theta$$

重新组合：$x' = x \cdot \cos\theta + (-x_2, x_1) \cdot \sin\theta$

第二项的 $(-x_2, x_1)$ 就是把坐标对交换并取反——代码里叫 `rotate_half`：

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        # 对 48 个二维平面同时做 90° 旋转
        # [q0,...,q47, q48,...,q95] → [-q48,...,-q95, q0,...,q47]
        #  前半截          后半截       后半截取反      前半截
        #
        # x[...] 中的 ... 是省略号索引（Ellipsis），表示"前面所有维度保持不变"
        # x 形状 [B, S, n_heads, head_dim]，x[..., :48] 等价于 x[:, :, :, :48]
        # 好处是不用数前面有几个维度，写法更通用
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

    # cos/sin: [S, head_dim] → [S, 1, head_dim]，广播到所有 head
    # q 形状 [B, S, n_heads, head_dim]，unsqueeze(1) 让 head 维自动广播
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    # 旋转矩阵乘法的展开: x' = x·cosθ + rotate_half(x)·sinθ
    # 逐元素乘法，不需要构造矩阵，48 个二维旋转一步搞定
    # .to(q.dtype) 确保混合精度训练（bf16/fp16）时结果精度一致
    q_embed = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_embed = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q_embed, k_embed
```

注意 `rotate_half` 的坐标配对方式：**不是相邻配对 (q0,q1)，而是前后半截配对 (q0,q48)**。数学上等价（只要 cos/sin 表的排列方式对应就行），但前后半截配对只需要一次 cat 操作，比相邻配对更高效。

### Step 9：验证

> 📁 接着追加到 `scratch/model/model_minimind.py`

```python
head_dim = 96
seq_len = 10
cos, sin = precompute_freqs_cis(head_dim, seq_len)
print(f"cos shape: {cos.shape}")  # [10, 96]

q = torch.randn(2, seq_len, 8, head_dim)  # [batch, seq, n_heads, head_dim]
k = torch.randn(2, seq_len, 4, head_dim)  # [batch, seq, n_kv_heads, head_dim]

q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
# unsqueeze 已在函数内部处理：[seq, head_dim] → [seq, 1, head_dim] 广播到所有头
assert q_rot.shape == q.shape
assert k_rot.shape == k.shape
print("RoPE applied, shapes preserved")
```

## YaRN 外推（推理时启用）

### 问题：推理长度 > 训练长度时发生了什么？

MiniMind 训练时 `original_max_position_embeddings = 2048`，推理时想支持 `end = 32768`（16 倍）。

不同频率的维度，面临的风险完全不同：

**高频维度（pair 0，$\omega_0 = 1.0$，周期 $\approx 6.28$）**：
- 训练时最大角度 $= 2047 \times 1.0 = 2047$ rad $\approx 325$ 圈
- 推理时最大角度 $= 32767 \times 1.0 = 32767$ rad $\approx 5214$ 圈
- cos/sin 是周期函数，转了 325 圈意味着所有相位角都见过了
- 多转几圈不会产生新的 cos/sin 值 → **没有 OOD 问题**

**低频维度（pair 47，$\omega_{47} = 1.3 \times 10^{-6}$，周期 $\approx 470$ 万）**：
- 训练时最大角度 $= 2047 \times 1.3 \times 10^{-6} = 0.00266$ rad
- 推理时最大角度 $= 32767 \times 1.3 \times 10^{-6} = 0.0426$ rad
- 0.0426 是训练时见过的 0.00266 的 **16 倍**
- 模型在这些角度上没有被训练过 → **分布外推（OOD）→ 效果崩坏**

核心洞察：**高频维度安全，低频维度危险，需要分别处理。**

### 方案演进：从暴力到精细

#### 方案 1: Position Interpolation（PI）— Meta, 2023.06

最暴力的方法：把所有位置**等比压缩**回训练范围。

```
原来:  位置 0, 1, 2, ..., 32767  →  角度 theta(m) = m * omega
压缩:  位置 0, 0.0625, 0.125, ..., 2047  →  theta(m) = m * omega / factor
做法:  freqs 全部除以 factor=16

效果:  所有角度都落在训练范围内，不会 OOD
问题:  高频维度被压缩 16 倍，相邻 token 的角度差从 1.0 变成 0.0625
       近距离分辨率大幅下降 → 短距离理解能力变差
```

#### 方案 2: NTK-aware — Reddit 用户 bloc97, 2023.06

不压缩位置，而是**调大 base**：

$$\text{new\_base} = \text{base} \times \text{factor}^{\text{dim}/(\text{dim}-2)}$$

效果：所有频率都降低，但降低幅度不均匀——低频降得多、高频降得少。比 PI 更好地保留了高频精度，但高频维度不需要降也被降了。

#### 方案 3: NTK-by-parts — YaRN 论文 Section 3.4, 2023.09

> 这是 YaRN 论文（Peng et al., Nous Research / EleutherAI）Section 3.4 的内容，**不是 Qwen 提出的**。

核心思想：既然不同频率维度风险不同，就**分段处理**。

把 48 个维度对分成三段：

```
┌──────────────┬────────────────────┬────────────────┐
│   高频段       │     过渡段          │    低频段       │
│   i < low     │  low <= i <= high  │   i > high     │
│   γ = 0       │  0 < γ < 1         │   γ = 1        │
│   不缩放       │  线性插值           │  全力缩放/factor│
│   保持近距离精度│  平滑过渡           │  扩展远距离范围  │
└──────────────┴────────────────────┴────────────────┘
```

缩放公式：

$$f'(\omega_i) = \omega_i \times (1 - \gamma + \gamma / \text{factor})$$

- $\gamma = 0$：$\omega_i \times 1 = \omega_i$（完全不动）
- $\gamma = 1$：$\omega_i \times (1/\text{factor}) = \omega_i / 16$（等价于 PI 的全力压缩）
- $\gamma = 0.5$：$\omega_i \times 0.53125$（折中）

#### 方案 4: YaRN = NTK-by-parts + Attention Scaling（Section 3.5）

YaRN 在 NTK-by-parts 基础上加了 `attention_factor`，乘到 cos/sin 上补偿注意力熵的变化。MiniMind 中 `attention_factor = 1.0`，所以实际等价于纯 NTK-by-parts。

### 分界线怎么算？—— inv_dim 反函数

最关键的问题：**low 和 high 怎么确定？**

思路：找到"哪些维度的周期短到在训练长度内能转足够多圈"。

维度对 $i$ 的周期 $T_i = 2\pi / \omega_i = 2\pi \times \text{base}^{2i/\text{dim}}$。论文引入 $\beta$ 参数量化"多短算短"：

$$\text{波长阈值} = \text{orig\_max} / \beta$$

- $\beta_{\text{fast}} = 32$：波长阈值 $= 2048/32 = 64$。周期 $< 64$ 的维度算"高频"（训练时至少转了 32 圈，精度绰绰有余）
- $\beta_{\text{slow}} = 1$：波长阈值 $= 2048/1 = 2048$。周期 $> 2048$ 的维度算"低频"（训练时连一圈都转不完，必须缩放）

**inv_dim 做的是反函数**——给定波长阈值，反算对应的维度编号：

$$T_i = \text{orig\_max} / \beta = 2\pi \times \text{base}^{2i/\text{dim}}$$

$$i = \frac{\text{dim} \times \ln(\text{orig\_max} / (\beta \times 2\pi))}{2 \times \ln(\text{base})}$$

代入 MiniMind 的具体数字（dim=96, base=1e6, orig_max=2048）：

$$\text{low} = \lfloor \text{inv\_dim}(\beta_{\text{fast}}=32) \rfloor = \lfloor \frac{96 \times \ln(2048 / 201.06)}{2 \times \ln(10^6)} \rfloor = \lfloor \frac{96 \times 2.321}{27.631} \rfloor = \lfloor 8.07 \rfloor = 8$$

$$\text{high} = \lceil \text{inv\_dim}(\beta_{\text{slow}}=1) \rceil = \lceil \frac{96 \times \ln(2048 / 6.283)}{2 \times \ln(10^6)} \rceil = \lceil \frac{96 \times 5.787}{27.631} \rceil = \lceil 20.11 \rceil = 21$$

MiniMind 的 48 个维度对被分成：

| 段 | 维度对 | $\gamma$ | 行为 | 物理含义 |
|:--:|:------:|:--------:|:----:|:---------|
| 高频 | pair 0-7 | 0 | 不缩放 | 周期 6~68，训练时转了 30~341 圈 |
| 过渡 | pair 8-21 | 0~1 线性 | 部分缩放 | 周期 68~3700，部分见过 |
| 低频 | pair 22-47 | 1 | 除以 16 | 周期 3700~470万，训练时没转完一圈 |

### 源码逐行解读

```python
def precompute_freqs_cis(dim, end=32768, rope_base=1e6, rope_scaling=None):
    # Step 1: 标准 RoPE 频率
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))
    # freqs: [48], 从 1.0(高频) 到 1.3e-6(低频)

    attn_factor = 1.0

    if rope_scaling is not None:
        orig_max = 2048    # 训练长度
        factor = 16        # 外推倍数: 2048 × 16 = 32768
        beta_fast = 32.0   # 高频边界（转 32 圈以上算高频）
        beta_slow = 1.0    # 低频边界（转不到 1 圈算低频）
        attn_factor = 1.0  # 注意力缩放因子

        # 只在推理长度超过训练长度时才缩放
        if end / orig_max > 1.0:
            # Step 2: 反函数——波长阈值 → 维度编号
            inv_dim = lambda b: (
                dim * math.log(orig_max / (b * 2 * math.pi))
            ) / (2 * math.log(rope_base))

            low = max(math.floor(inv_dim(beta_fast)), 0)             # = 8
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)  # = 21

            # Step 3: 线性斜坡 γ
            # pair 0-7 → 0, pair 8-21 → 0~1 线性, pair 22-47 → 1
            ramp = torch.clamp(
                (torch.arange(dim // 2).float() - low) / max(high - low, 0.001),
                0, 1
            )

            # Step 4: 分段缩放
            # γ=0: freqs × 1.0     （原频率，高频不动）
            # γ=1: freqs × 1/16    （频率除以 factor，低频全力压缩）
            freqs = freqs * (1 - ramp + ramp / factor)

    # 后续和标准 RoPE 一样
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
```

### Qwen 的长上下文方案

> **注意**：NTK-by-parts（YaRN Section 3.4）不是 Qwen 提出的，而是 Nous Research / EleutherAI 的工作。Qwen 有自己独立的长上下文方案，后来转向了 YaRN。

**Qwen-1（2023.08）** 提出了三板斧组合：

| 技术 | 做法 | 解决什么 |
|------|------|---------|
| Dynamic NTK | 推理时根据当前序列长度动态调大 base | 低频维度的 OOD |
| LogN Scaling | 注意力分数除以 $\sqrt{\log(n) / \log(\text{train\_len})}$ | 序列变长后注意力熵增大 |
| Window Attention | 底层用滑动窗口注意力 | 减少长序列计算量 |

Dynamic NTK 和 YaRN 的区别：

```
Dynamic NTK（Qwen-1）:
  if seq_len > training_length:
      ratio = seq_len / training_length
      new_base = base * ratio ^ (dim / (dim - 2))
  整体调大 base，不区分频段

YaRN NTK-by-parts:
  按 gamma 分三段：高频不动，低频缩放，中间过渡
  更精细，效果更好
```

**Qwen 方案演进**：

| 版本 | 时间 | base | 外推方案 | 支持长度 |
|------|------|------|---------|---------|
| Qwen-1 | 2023.08 | 10,000 | Dynamic NTK + LogN | 8K → 32K |
| Qwen-1.5 | 2024.02 | 1,000,000 | 大 base（类 NTK-aware） | 32K |
| Qwen-2 | 2024.06 | 1,000,000 | YaRN | 128K |
| Qwen-2.5 | 2024.09 | 1,000,000 | YaRN | 128K / 1M |

关键转折：从 Qwen-1.5 开始，Qwen 抛弃了自己的 Dynamic NTK，转向大 base + YaRN——社区验证后 YaRN 效果更好。

## 在完整模型中的位置

```python
# 初始化时预计算一次（MiniMindModel.__init__）
freqs_cos, freqs_sin = precompute_freqs_cis(dim=96, end=32768, rope_base=1e6)
self.register_buffer("freqs_cos", freqs_cos)   # 不参与梯度
self.register_buffer("freqs_sin", freqs_sin)

# Forward 时按位置切片
position_embeddings = (freqs_cos[start_pos:start_pos+seq_len],
                       freqs_sin[start_pos:start_pos+seq_len])

# Attention 内部应用旋转（Q 和 K 旋转，V 不旋转）
xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
```

## 对照 MiniMind 源码

| 我们的实现 | MiniMind 源码位置 |
|-----------|-----------------|
| `precompute_freqs_cis` | `model_minimind.py:62-78` |
| `apply_rotary_pos_emb` | `model_minimind.py:80-84` |
| `rotate_half` | 完全一致 |
| `register_buffer` | `model_minimind.py:205-207` |

MiniMind 额外支持 YaRN 外推（`inference_rope_scaling=True` 时），我们的简化版省略了这部分。

## 业界对比

| 方式 | 加在哪 | 绝对/相对 | 外推性 |
|------|--------|----------|--------|
| Sinusoidal（原始 Transformer） | Embedding 后直接加 | 绝对 | 有限（可计算未见位置，但属分布外推） |
| Learned Position Embedding（GPT-2） | Embedding 后直接加 | 绝对 | 差（超表长索引越界） |
| ALiBi（BLOOM） | Attention score 上加偏置 | 相对 | 具有相对位置归纳偏置 |
| **RoPE（LLaMA/Qwen/DeepSeek/MiniMind）** | **Attention 内旋转 Q/K** | **相对** | **具有相对位置归纳偏置，vanilla RoPE 长度外推仍有限** |

| 模型 | 位置编码 | 特点 |
|------|---------|------|
| MiniMind | RoPE（θ=1e6） | 48 对子空间，config 支持 YaRN |
| Qwen2.5 | RoPE（θ=1e6） | 128K/1M 上下文，Qwen-1 用 Dynamic NTK+LogN，Qwen-1.5+ 转向大 base+YaRN |
| DeepSeek-V3 | RoPE（θ=1e4） | MLA 架构下的 decoupled RoPE（只对 Q/K 部分维度加 RoPE） |
| LLaMA-3 | RoPE（θ=5e5） | 原始版本 8K 上下文；LLaMA 3.1 引入新的长上下文 scaling |

DeepSeek MLA 的 decoupled 设计：K 分两部分，只对其中一部分加 RoPE，另一部分不加。
因为 KV Cache 压缩需要去掉位置信息，而 RoPE "必须加在 Q/K 上"这个约束与压缩冲突，
所以用解耦设计让位置编码和 KV 压缩互不干扰。

## 面试话术

> **RoPE 核心**：
> "RoPE 的核心是在 Attention 内部对 Q 和 K 做旋转编码位置。head_dim 拆成多对二维坐标，每对用不同频率旋转——高频管近距离精度，低频管远距离范围，组合起来像组合锁一样让每个位置有唯一指纹。
> 频率公式和原始 sinusoidal PE 完全一样，RoPE 的改进在操作上——旋转代替加法。旋转矩阵的正交性保证了 Q·K 点积自动只含相对距离 m-n，绝对位置消失。
> 这也是为什么 RoPE 必须放在 Attention 内部：后续的 Norm/FFN 会破坏正交性。
> 实现上是分块对角矩阵——48 个 2×2 旋转块，代码里展开成逐元素乘法，没有真正的矩阵乘。
> 在我写的模型里 head_dim=96 拆成 48 对子空间，base=1e6 配合 YaRN 支持 32k 序列。"

> **YaRN 外推**（如果追问长上下文外推）：
> "推理长度超过训练长度时，不同频率的维度面临的 OOD 风险不同——高频维度在训练时已经转了几百圈，所有相位都见过，不需要处理；低频维度连一圈都没转完，必须缩放。YaRN 的做法是 NTK-by-parts：用 inv_dim 反函数算出高频和低频的分界线，高频不动、低频除以 factor、中间线性过渡。在我的实现里，48 对维度被分成 pair 0-7 不缩放、pair 8-21 过渡、pair 22-47 全力缩放。这比 Position Interpolation 的暴力等比压缩精细得多——PI 会把高频维度也压缩 16 倍，近距离分辨率直接崩。"

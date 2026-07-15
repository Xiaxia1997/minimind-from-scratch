# 05 - FeedForward：让每个 token 独立"思考"

## 为什么需要 FFN？

```
数据流：Attention 输出 → 残差连接 → RMSNorm → 【FeedForward】 → 残差连接
```

#### Attention 能做什么、不能做什么

Attention 的核心操作 = $\text{softmax}(\frac{QK^T}{\sqrt{d}}) \cdot V$。虽然 softmax 本身是非线性的，但 Attention 主要负责的是**跨 token 的信息混合**——决定从哪些 token 取多少信息，然后做加权组合。

Attention 有 Q/K/V 的线性投影和 softmax，不是纯线性系统，但它缺少**对每个 token 独立做的 position-wise 非线性特征变换**——这正是 FFN 的角色。

#### FFN 是什么

FFN 就是**两层 MLP 中间夹一个激活函数**，对每个 token 独立做非线性变换（token 之间不交互，交互在 Attention 里已经完成）：

$$\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 \cdot x)$$

FFN 对每个 token 独立地做非线性特征变换和通道混合，为模型增加大量 position-wise 的表达容量。没有 FFN，模型缺少对每个 token 的独立深度加工能力。

#### 为什么要先升维再激活

FFN 的结构是 768 → 2432 → 激活 → 768，但升维不是激活函数的前提条件——768 → 768 → 激活 → 768 也能工作，只是效果差。

原因：激活函数对每个维度**独立**做筛选（SiLU 压低某些维度、ReLU 直接置零）。如果在 768 维上做激活，压掉一半就只剩 ~384 个有效维度，信息损失太大。

$W_1$ 的每一行是一个**特征检测器**——768 → 2432 意味着有 2432 个不同的特征检测器。激活函数决定哪些检测器被激活，$W_2$ 把激活后的结果组合回 768 维：

```
768 维输入
   ↓ W₁ (2432 个特征检测器，每个检测输入的一种模式)
2432 维中间层
   ↓ 激活函数 (独立筛选每个检测器：激活 or 抑制)
2432 维 (部分被压低/置零，但基数大，剩余信息仍然丰富)
   ↓ W₂ (把存活的检测器结果组合回 768 维)
768 维输出
```

检测器越多（中间维度越大），能识别和筛选的模式就越多，FFN 的表达能力就越强。

## 从零开始学

### Step 1：原始 FFN（Vaswani 2017）

原始 Transformer 的 FFN 是两层线性投影 + ReLU：

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$

```
数据流：[B, S, 768] → W₁ → [B, S, 3072] → ReLU → [B, S, 3072] → W₂ → [B, S, 768]
                      升维 4×              激活              降回原维度
```

**问题**：ReLU 把所有负值直接变成 0，太粗暴——大量特征被硬截断丢弃。

![ReLU 激活函数](step1_relu.png)

### Step 2：SiLU 激活函数——自门控机制

先拆解 SiLU 的公式，它由两部分组成：

$$\text{sigmoid}(x) = \frac{1}{1 + e^{-x}} \qquad \text{（输出范围 (0,1)）}$$

$$\text{SiLU}(x) = x \cdot \text{sigmoid}(x) \qquad \text{（自门控）}$$

注意：sigmoid 的输出在 (0,1) 之间，但 **SiLU 的输出不是 0~1**。因为 SiLU = x × sigmoid(x)，当 x 很大时 SiLU ≈ x（可以远大于 1），当 x 为负值时 SiLU 为负数（最小约 -0.28）。所以 SiLU 是一个**有符号的、可放大也可抑制的调制函数**，不能简单理解为"通过率"。

直觉上：x 越大，sigmoid 越接近 1，SiLU ≈ x 几乎不衰减；x 为较大负值时，sigmoid 接近 0，SiLU 接近 0 但不完全为零。

![ReLU vs SiLU 对比与 SiLU 拆解](step2_silu.png)

对比 ReLU：

$$\text{ReLU}(x) = \max(0, x)$$



ReLU 的问题：x=0 处有硬拐点（梯度不连续），负值全部截断为 0（"死神经元"问题）。
SiLU 平滑过渡，负值区域不完全丢失信息，梯度处处连续。

### Step 3：SwiGLU——加门控机制

#### 从原始 FFN 到 SwiGLU 的变化

原始 FFN 是一条线走到底：

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x)$$

$W_1$ 身兼两职——既要产出有用的特征内容，又要让激活函数能正确筛选。SwiGLU（Shazeer, 2020）把这一个投影**拆成两路并行**：

$$\text{SwiGLU}(x) = \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})$$

$$\text{FFN}_{\text{SwiGLU}}(x) = \text{SwiGLU}(x) \cdot W_{\text{down}}$$

下图上半部分对比了两种结构，下半部分展示了逐元素相乘的具体效果：

![原始 FFN vs SwiGLU 结构对比](step3_swiglu.png)

#### 逐元素相乘（$\odot$）是什么

两个同样大小的向量，对应位置一个一个乘：

```
gate（调制系数）:  [1.85,  -0.02,  1.42,  0.03]    ← SiLU(gate_proj) 产出
up  （候选内容）:  [0.63,   0.97,  0.76,  0.50]    ← up_proj 产出
                     ×       ×      ×      ×
输出:              [1.17,  -0.02,  1.08,  0.02]
                   放大!    反转    放大!   被压掉
```

gate = 1.85 → 这个维度被放大（SiLU 对大正值可以 > 1）
gate = 0.03 → 基本被压掉
gate = -0.02 → 微弱反转（SiLU 对负值输出为负，但幅度很小）

#### 三个投影各司其职

- **gate_proj**（$W_{\text{gate}}$）：门控路径——投影后过 SiLU，产出每个维度的调制系数（可放大、抑制或微弱反转）
- **up_proj**（$W_{\text{up}}$）：内容路径——投影到高维，产出"候选特征"
- **$\odot$**：门 × 内容 = 筛选后的特征（上图下半部分的绿色柱）
- **down_proj**（$W_{\text{down}}$）：压回原维度

## 维度追踪

```
x: [2, 128, 768]
    ├─ gate_proj → [2, 128, 2432] → SiLU → [2, 128, 2432]  "门"
    ├─ up_proj   → [2, 128, 2432]                            "值"
    ↓
逐元素相乘:     [2, 128, 2432]  门 × 值
    ↓
down_proj:      [2, 128, 768]   降回原维度
```

## intermediate_size 为什么是 2432

SwiGLU 比原始 FFN 多了一个投影（3 个 vs 2 个），参数量多 50%。
经典等参数推导会得到 **8/3 × hidden_size ≈ 2048** 作为 intermediate_size。

MiniMind 源码实际用的是：

```python
import math
intermediate_size = math.ceil(768 * math.pi / 64) * 64
# 768 × π ≈ 2412.7 → ceil(2412.7/64) = 38 → 38 × 64 = 2432
# π ≈ 3.14 比 8/3 ≈ 2.67 大约 18%，所以 SwiGLU 比等参数的原始 FFN 多约 19% 参数
# 对齐到 64 倍数是为了 GPU Tensor Core 对齐，矩阵乘法效率更高
# 源码能证明公式，但不能确定作者选择 π 而非 8/3 的具体原因
```

## 参数量对比

```
SwiGLU (3 个投影):
  gate: 768 × 2432 = 1,867,776
  up:   768 × 2432 = 1,867,776
  down: 2432 × 768 = 1,867,776
  合计: 5,603,328 ≈ 5.6M/层

原始 FFN (2 个投影):
  W1: 768 × 3072 = 2,359,296
  W2: 3072 × 768 = 2,359,296
  合计: 4,718,592 ≈ 4.7M/层

SwiGLU 参数多 ~19%，但门控机制让效果显著更好。
```

## 完整实现

> 📁 追加到 `scratch/model/model_minimind.py`

```python
class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=2432):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

验证代码追加到 `if __name__ == '__main__':` 块中（参见 [01-Embedding](01-Embedding.md) 关于文件组织的说明）：

```python
    # ---- FeedForward ----
    ffn = FeedForward()
    x = torch.randn(2, 128, 768)
    assert ffn(x).shape == x.shape
    total = sum(p.numel() for p in ffn.parameters())
    print(f"FeedForward verified! params: {total:,}")  # 5,603,328
```

## 对照 MiniMind 源码

| 我们的实现 | MiniMind 源码位置 |
|-----------|-----------------|
| `FeedForward` 类 | `model_minimind.py:136-146` |
| `gate_proj` + `up_proj` + `down_proj` | 完全一致 |
| `F.silu` | `ACT2FN['silu']` |
| 一行 forward | 完全一致 |

## 业界对比

| 特性 | MiniMind (64M) | Qwen2.5-7B | DeepSeek-V2 | LLaMA 3-8B |
|------|---------------|------------|-------------|------------|
| FFN 类型 | SwiGLU | SwiGLU | MoE（160 routed + 2 shared，top-6） | SwiGLU |
| intermediate_size | 2432 | 18944 | Dense: 12288 / 专家: 1536 | 14336 |
| 倍率 | 3.17× | 5.29× | Dense: 2.4× / 专家: 0.3× | 3.5× |
| 激活函数 | SiLU | SiLU | SiLU | SiLU |
| bias | 无 | 无 | 无 | 无 |

所有主流大模型都用 SwiGLU 替代了原始的 ReLU FFN，且都去掉了 bias。

## 面试话术

> "FFN 我从原始的两层 ReLU MLP 开始，然后改成 SwiGLU 门控机制。
> 三个投影：gate_proj 过 SiLU 决定'要不要通过'，up_proj 提供'通过什么内容'，
> 两者逐元素相乘做门控，再由 down_proj 降回 768 维。
> 相比原始 FFN 多一个投影但效果显著更好——门控让模型自己学会抑制不重要的特征。
> 经典等参数推导中间维度取 8/3 倍，MiniMind 实际用 ceil(768×π/64)×64=2432，
> 比等参数略大约 19%，对齐 64 方便 GPU Tensor Core。"

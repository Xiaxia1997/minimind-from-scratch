# 02 - RMSNorm：让每一层的输入保持稳定

## Pre-Norm vs Post-Norm：归一化放在哪？

在讲 RMSNorm 本身之前，先搞清一个更上层的架构选择：**归一化放在子层计算的前面还是后面？**

### Post-Norm（原始 Transformer，2017）

```
x ──→ Attention ──→ Add(x) ──→ LayerNorm ──→ FFN ──→ Add ──→ LayerNorm ──→ output
       先算                        后 Norm           先算           后 Norm
```

**先做 Attention/FFN，再加残差，最后 Norm**。原始 Transformer、BERT 都是这个。

代码形式：
```python
# Post-Norm（MiniMind 没用，仅对比）
hidden_states = self.self_attn(hidden_states, ...)
hidden_states = self.input_layernorm(hidden_states + residual)  # Norm 在加完残差之后
```

问题：**Norm 卡在残差通路上**，梯度回传时必须穿过 Norm 层，深层网络容易梯度消失，对 learning rate 和 warmup 非常敏感。

### Pre-Norm（LLaMA / Qwen / DeepSeek / MiniMind）

```
x ──→ RMSNorm ──→ Attention ──→ Add(x) ──→ RMSNorm ──→ FFN ──→ Add ──→ output
       先 Norm                    残差直连     先 Norm             残差直连
```

**先 Norm，再做 Attention/FFN，再加残差**。残差连接形成一条"干净的梯度高速公路"，梯度不需要穿过 Norm 层。

对应 MiniMind 代码（`model_minimind.py:186-193`）：
```python
# Pre-Norm（MiniMind 实际用的）
residual = hidden_states
hidden_states = self.self_attn(
    self.input_layernorm(hidden_states), ...  # ← Norm 在 Attention 前面
)
hidden_states += residual                      # ← 残差直接加，Norm 不在通路上

hidden_states = hidden_states + self.mlp(
    self.post_attention_layernorm(hidden_states)  # ← Norm 在 FFN 前面
)
```

### 为什么绝大多数主流 LLM 用 Pre-Norm？

| 维度 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 梯度通路 | 梯度必须穿过 Norm 层 | 残差直连，梯度畅通无阻 |
| 训练稳定性 | 深层时更依赖 warmup、初始化或残差缩放等技巧 | 初始化阶段梯度通常更平稳 |
| 学习率敏感度 | 更敏感 | 宽容得多 |
| 理论表达能力 | 某些研究观察到略强 | 某些研究观察到略弱 |
| 工程结论 | **稳定性的收益远大于理论表达力的微小差距** | **绝大多数主流 LLM 的选择** |

### Pre-Norm 的代价：等效深度变浅

上面表格里"某些研究观察到略弱"不是玄学，背后有明确的数学机制。

Pre-Norm 的残差连接形式：

$$x_{l+1} = x_l + F(\text{Norm}(x_l))$$

随着层数加深，$x_l$ 是前面所有层贡献的累加：$x_l = x_0 + \sum_{i=0}^{l-1} F(\text{Norm}(x_i))$。每一层新增的 $F(\text{Norm}(x_l))$ 相对于这个累积和越来越小——$x_l$ 这条"直通通道"越来越主导最终输出。深层的 $F$ 对最终输出的**边际贡献被稀释**，模型的行为更像一个浅层网络。这就是 Pre-Norm "等效深度变浅"的核心机制（[On the Connections between Residual Learning and the Lottery Ticket Hypothesis, He et al., 2020] 从另一个角度观察到了类似现象；[DeepNet, Wang et al., 2022] 对此做了系统分析）。

Post-Norm 的形式：

$$x_{l+1} = \text{Norm}(x_l + F(x_l))$$

Norm 作用在残差相加**之后**，每一层的贡献都会被重新归一化，不会出现"直通通道主导"的问题——每一层的 $F$ 对输出的贡献不会随深度递减。因此 Post-Norm 的**有效深度更大**，理论表达力更强。

但 Post-Norm 的代价也很明确：Norm 卡在残差通路上，梯度回传时必须穿过每一层的 Norm，深层网络容易梯度不稳定。训练时需要精心的 warmup 调度和初始化策略（如 [DeepNet, Wang et al., 2022] 提出的 DeepNorm 初始化），否则训练直接发散。

**实践结论**：Pre-Norm 训练稳定但收敛性能略低（等效深度浅），Post-Norm 潜在性能更高但训练更难。当前主流 LLM 选择 Pre-Norm，是用一点理论性能换取工程上的稳定性和可扩展性。

### 面试话术（Pre-Norm vs Post-Norm）

> "MiniMind 用的是 Pre-Norm，把 RMSNorm 放在 Attention 和 FFN 之前，残差连接形成一条干净的梯度高速公路——梯度回传不需要穿过 Norm 层，所以深层模型训练更稳定。对比原始 Transformer 的 Post-Norm，Pre-Norm 不需要精心的 warmup，工程上更友好。LLaMA 之后这就是业界标配了，Qwen、DeepSeek 全用 Pre-Norm。"

---

## 为什么需要归一化？

深层网络的核心问题：随着层数加深，激活值的尺度可能逐层漂移，导致训练不稳定、梯度爆炸/消失。

归一化的作用：**控制每层激活值的尺度，改善优化条件和数值稳定性**，让后续的计算（Attention、FFN）在可预期的数值区间内工作。

## 从零开始写

### Step 1：先回顾 LayerNorm

原始 Transformer 用的是 LayerNorm：

```
LayerNorm(x) = (x - mean) / std * γ + β
```

两步操作：
1. **Re-centering**: 减去均值 `x - mean`，让分布以 0 为中心
2. **Re-scaling**: 除以标准差 `/ std`，让分布方差为 1

`γ`（缩放）和 `β`（偏移）是可学习参数，让模型自己决定归一化后的最终分布。

### Step 2：简化为 RMSNorm

论文 [Zhang & Sennrich, 2019] 提出假设：**re-centering invariance 可能不是归一化的必要成分**。实验显示 RMSNorm 在多个任务上取得可比表现，运行时间减少约 7%–64%（因模型和硬件而异）。

RMSNorm 只保留 re-scaling：

```
RMSNorm(x) = x / RMS(x) * γ

RMS(x) = sqrt(mean(x²) + eps)
```

去掉了 `- mean` 和 `+ β`，只留 γ 一个可学习参数。

### Step 3：写代码

> 📁 追加到 `scratch/model/model_minimind.py`（接在 01 Embedding 代码后面，不需要重复 import）

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ，初始全 1（不改变原始分布）

    def norm(self, x):
        # x.pow(2).mean(-1, keepdim=True) → 沿 hidden_dim（最后一维）求 x² 的均值
        #   keepdim=True: 保留被压缩的维度为 size=1，而不是删掉它
        #     有 keepdim: [B, S, 768] → [B, S, 1]  ← 能和原始 x [B, S, 768] 正确广播相乘
        #     没 keepdim: [B, S, 768] → [B, S]     ← 少一维，乘法会报错或语义错误
        # torch.rsqrt → 1/sqrt，避免显式的 sqrt+除法，通常更利于后端优化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转 float32 算 norm（输入可能是 bf16/fp16，低精度算 rsqrt 数值不稳定）
        # 算完 .type_as(x) 转回原精度
        return (self.weight * self.norm(x.float())).type_as(x)
```

两个工程细节：
1. **`torch.rsqrt`** 代替 `1/sqrt`：避免显式的 sqrt+除法，通常更利于后端优化
2. **`.float()` 再 `.type_as(x)`**：混合精度训练时输入是 bf16/fp16，但 rsqrt 对低精度敏感，先转 fp32 保数值稳定

### Step 4：验证

> 📁 接着追加到 `scratch/model/model_minimind.py`

```python
norm = RMSNorm(768)
x = torch.randn(2, 10, 768)       # [batch=2, seq_len=10, hidden=768]
out = norm(x)

assert out.shape == x.shape        # 输出形状不变
assert out.dtype == x.dtype        # 精度不变

# 验证归一化效果：输出的 RMS 应该接近 1
rms = (out.pow(2).mean(-1)).sqrt()
print(f"RMS after norm: {rms.mean():.4f}")  # 应接近 1.0
```

**为什么结果接近 1？** γ 初始化为全 1，所以 `out = x / RMS(x)`，再对 out 求 RMS：

```
RMS(out) = sqrt(mean(x²) / (mean(x²) + eps)) ≈ 1   （eps 很小时接近 1，但不严格等于 1）
```

注意这不等于"向量变成单位向量"——RMSNorm 后的 L2 范数约为 `sqrt(hidden_size)`，不是 1。归一化的目标是**让 RMS 接近 1（单位 RMS）**，从而控制激活值的尺度。

## 在完整模型中的位置

RMSNorm 在模型中出现 4 类位置：

```
MiniMindBlock（每层）:
    input_layernorm = RMSNorm(768)         # Attention 前
    post_attention_layernorm = RMSNorm(768) # FFN 前

MiniMindModel（全局）:
    norm = RMSNorm(768)                     # 最后一层输出后

Attention（内部）:
    q_norm = RMSNorm(96)                    # Q 投影后（QK Norm）
    k_norm = RMSNorm(96)                    # K 投影后
```

注意这是 **Pre-Norm 架构**——详见本文开头 [Pre-Norm vs Post-Norm](#pre-norm-vs-post-norm归一化放在哪) 章节。

**为什么三个地方都要 Norm？** 每个计算模块（Attention、FFN）都会把数值范围搞乱，所以每个模块前面都要 Norm 一次，把输入拉回稳定范围——就像流水线上每道工序前都要校准一下原料规格：

```
Embedding 输出 → 数值范围 OK
    ↓
  RMSNorm ← (1) input_layernorm：拉回稳定范围
    ↓
  Attention → 输出数值范围又漂了
    ↓
  + 残差
    ↓
  RMSNorm ← (2) post_attention_layernorm：再拉回来
    ↓
  FFN → 输出数值范围又漂了
    ↓
  + 残差
    ↓
  ... 重复 N 层 ...
    ↓
  RMSNorm ← (3) final norm：送进 LM Head 前最后校准一次，
    ↓                        不 norm 的话数值可能很大，softmax 会饱和
  LM Head → logits → softmax → 概率
```

## 对照 MiniMind 源码

| 我们的实现 | MiniMind 源码位置 |
|-----------|-----------------|
| `RMSNorm` 类 | `model_minimind.py:50-60` |
| `self.eps = eps` | 完全一致 |
| `torch.rsqrt` + `.float()` | 完全一致 |

实现完全一样——RMSNorm 本身就很简单，没有什么变体空间。

## 业界对比

| 模型 | 归一化 | 特殊用法 |
|------|--------|---------|
| 原始 Transformer | LayerNorm + Post-Norm | 需要 warmup |
| BERT | LayerNorm + Post-Norm | 同上 |
| LLaMA/Qwen/DeepSeek | RMSNorm + Pre-Norm | 无 QK Norm |
| MiniMind | RMSNorm + Pre-Norm | 有 QK Norm |
| Gemma 2 | RMSNorm + Pre-Norm | query_pre_attn_scalar + logit soft-capping（非显式 QK Norm） |

## 面试话术

> "RMSNorm 是 LayerNorm 的简化版，去掉了 re-centering（减均值）只保留 re-scaling（除均方根）。
> 论文假设 re-centering 不是必需的，实验显示效果可比，运行时间减少约 7%–64%。
> 实现上就是 x 乘以 rsqrt(mean(x²) + eps)，再乘一个可学习的缩放参数 γ。
> 工程上注意两点：一是先转 float32 再算 norm 防精度丢失，二是用 rsqrt 代替 sqrt+除法在 GPU 上更快。
> LLaMA、Qwen、DeepSeek 等主流开源模型全部用 RMSNorm 替代了 LayerNorm。"

## 延伸问题

- **QK Norm**：我们在 Attention 里对 Q 和 K 也做了 RMSNorm（`q_norm`/`k_norm`），防止注意力分数爆炸（详见 [04-Attention](04-Attention.md)）
- **Pre-Norm vs Post-Norm**：详见本文开头章节

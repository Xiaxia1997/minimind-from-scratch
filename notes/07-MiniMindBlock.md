# 07 - MiniMindBlock：把零件组装成一层 Transformer

## 为什么需要这一章？

前面 6 章都是独立零件——RMSNorm、RoPE、Attention、FFN、MoE。
这一章看怎么把它们拼在一起组成一层 Transformer Block。只有 17 行代码，但包含两个重要的架构决策：**Pre-Norm** 和**残差连接**。

## 从零开始写

### Step 1：确定组装顺序

一层 Transformer Block 要做两件事：
1. **token 间交互**（Attention）
2. **token 独立思考**（FFN）

每件事前后都要加"保护"：
- **前面**加 RMSNorm → 控制输入范围（Pre-Norm）
- **后面**加残差连接 → 梯度直通 + 恒等映射兜底

```
输入 x
  ├─ 保存 residual = x
  ├─ RMSNorm(x) → Attention → + residual    ← 第一次：交互 + 保护
  │
  ├─ 保存 residual = 上一步结果
  ├─ RMSNorm(x) → FFN/MoE  → + residual    ← 第二次：思考 + 保护
  ↓
输出（和输入维度完全一样）
```

### Step 2：写代码

```python
import torch
from torch import nn

class MiniMindBlock(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8, n_kv_heads=4,
                 head_dim=96, intermediate_size=2432, use_moe=False):
        super().__init__()
        self.self_attn = Attention(hidden_size, n_heads, n_kv_heads, head_dim)
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)         # Attention 前
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6) # FFN 前
        self.mlp = MOEFeedForward(...) if use_moe else FeedForward(...)

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None):
        # ---- Attention 分支 ----
        residual = hidden_states                    # 保存输入

        hidden_states, present = self.self_attn(
            self.input_layernorm(hidden_states),    # 先 Norm 再算（Pre-Norm）
            position_embeddings,
            past_key_value, use_cache, attention_mask
        )

        hidden_states = hidden_states + residual    # 残差连接

        # ---- FFN 分支 ----
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)  # 先 Norm 再算
        )

        return hidden_states, present
```

就这么多——17 行核心逻辑。

### Step 3：理解 Pre-Norm 的选择

```
Pre-Norm（我们用的）:    output = x + Attention(Norm(x))    先 Norm 再计算
Post-Norm（原始 Transformer）:  output = Norm(x + Attention(x))  先计算再 Norm
```

| 对比 | Pre-Norm | Post-Norm |
|------|---------|----------|
| 训练稳定性 | 好（Norm 控制了输入范围） | 差（需要 warmup） |
| 表达能力 | 理论略弱 | 理论更强 |
| 深层网络 | 优势明显 | 容易梯度爆炸 |
| 使用方 | LLaMA/Qwen/DeepSeek/GPT-3 | BERT/原始 Transformer |

所有主流 LLM 都用 Pre-Norm——大模型层数多（几十到上百层），训练稳定性是第一优先级。

### Step 4：理解残差连接的作用

```python
hidden_states = hidden_states + residual  # output = f(x) + x
```

三个作用：
1. **梯度直通**：反向传播时梯度可以跳过 Attention/FFN 直接流到前面的层 → 缓解梯度消失
2. **恒等映射兜底**：如果 Attention/FFN 学不到有用的东西，输出至少等于输入（不会变差）
3. **增量学习**：Attention 和 FFN 学的是"在原始表示上的**修正量**"，而不是从零重建

没有残差连接，8 层网络几乎训不动；有了残差连接，即使 100+ 层也能稳定训练。

## 维度追踪

```
输入: [2, 128, 768]
    │
    ├─ residual = 输入
    ├─ RMSNorm → [2, 128, 768]
    │     ↓
    ├─ Attention → [2, 128, 768]
    ├─ + residual → [2, 128, 768]     ← 第一次残差
    │
    ├─ RMSNorm → [2, 128, 768]
    │     ↓
    ├─ FFN/MoE → [2, 128, 768]
    └─ + self → [2, 128, 768]         ← 第二次残差

输出: [2, 128, 768]                    从头到尾维度不变
```

## 每层参数量

```
Attention:
  W_q [768, 768]:   589,824
  W_k [768, 384]:   294,912
  W_v [768, 384]:   294,912
  W_o [768, 768]:   589,824
  q_norm + k_norm:  192
  小计:              1,769,664 ≈ 1.8M

FFN (SwiGLU):
  gate [768, 2432]:  1,867,776
  up   [768, 2432]:  1,867,776
  down [2432, 768]:  1,867,776
  小计:              5,603,328 ≈ 5.6M

RMSNorm × 2:        1,536（可忽略）

每层合计:            ≈ 7.4M
8 层合计:            ≈ 59.2M
```

**FFN 参数量是 Attention 的 3 倍多**——模型的大部分参数在 FFN 里，不在 Attention 里。

## 完整验证

> 📁 追加到 `scratch/model/model_minimind.py`（RMSNorm、FeedForward 前面已定义，这里只追加 MiniMindBlock）

```python
class SimpleBlock(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8, intermediate_size=2432):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.ffn = FeedForward(hidden_size, intermediate_size)

    def forward(self, x):
        # Pre-Norm + Attention + 残差
        normed = self.norm1(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask, is_causal=True)
        x = x + attn_out

        # Pre-Norm + FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x

block = SimpleBlock()
x = torch.randn(2, 128, 768)
out = block(x)
assert out.shape == x.shape
print("Block verified!")
```

## 对照 MiniMind 源码

| 我们的实现 | MiniMind 源码位置 |
|-----------|-----------------|
| `MiniMindBlock.__init__` | `model_minimind.py:179-184` |
| `forward` | `model_minimind.py:186-194` |
| Pre-Norm + 残差 | 完全一致 |
| MoE 选择 `if not use_moe` | `model_minimind.py:183` |

结构完全一致——所有主流大模型的 Block 几乎一模一样，差异在 Attention 和 FFN 的内部实现。

## 业界对比

| 特性 | MiniMind | LLaMA 3 | Qwen2.5 | DeepSeek-V2 |
|------|---------|---------|---------|-------------|
| Norm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| FFN | SwiGLU/MoE | SwiGLU | SwiGLU | MoE |
| QK Norm | 有 | 无 | 有 | 有 |
| 每层 Norm 数 | 2 | 2 | 2 | 2 |

## 面试话术

> "一层 Transformer Block 就是 Pre-Norm + Attention + 残差 + Pre-Norm + FFN + 残差。
> 我从零组装过——先写各个零件，再在 Block 里串起来。
> Pre-Norm 就是先做 RMSNorm 再做 Attention/FFN，比原始 Post-Norm 训练更稳定。
> 残差连接让梯度能直接跳过计算层流到前面，是深层网络能训练的关键——
> Attention 和 FFN 学的是'在原始表示上的修正量'，不是从零重建。
> 参数量上 FFN 占大头——每层 FFN 5.6M，Attention 才 1.8M，比例约 3:1。
> 所有主流大模型的 Block 结构几乎一样，差异在 Attention（GQA/MLA）和 FFN（SwiGLU/MoE）内部。"

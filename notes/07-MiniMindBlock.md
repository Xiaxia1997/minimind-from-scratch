# 07 - MiniMindBlock：把零件组装成一层 Transformer

## 为什么需要这一章？

前面 6 章都是独立零件——RMSNorm、RoPE、Attention、FFN、MoE。
这一章看怎么把它们拼在一起组成一层 Transformer Block。只有 17 行代码，但包含两个重要的架构决策：**Pre-Norm** 和**残差连接**。

## 从零开始写

### Step 1：残差连接——为什么每一步都要"加回来"？

Block 里最重要的设计之一是残差连接（Residual Connection）。先理解它，后面的代码才看得懂。

```
不用残差:  output = f(x)           网络必须从零学习完整变换
用残差:    output = f(x) + x       网络只需要学习"修正量"
```

**为什么能防止梯度消失？**

反向传播时，梯度要从最后一层一路乘回第一层。没有残差的情况：

```
output = f(x)
d(output)/dx = f'(x)     ← 梯度完全取决于 f'(x)

8 层串联: 梯度 = f1' × f2' × f3' × ... × f8'
如果每个 f' 都小于 1（比如 0.5），8 层后: 0.5^8 = 0.004 → 梯度几乎消失
```

加了残差：

```
output = f(x) + x
d(output)/dx = f'(x) + 1     ← 多了一个恒为 1 的项

8 层串联: 每一层都有一条 "直通路" 贡献梯度 1
即使 f'(x) 很小，梯度也不会消失——因为 +1 保底
```

**这个 +1 就是"梯度高速公路"**——不管 Attention/FFN 内部的梯度有多小，`+ x` 这条路永远通畅，梯度可以直接跳过这一层流到前面去。

**另一个好处：Attention 和 FFN 只需要学"修正量"。** `output = f(x) + x` 意味着 `f(x)` 学的是"在原始表示上做多大的调整"，而不是从零重建完整表示。如果某一层不需要做太多调整，`f(x)` 趋近 0 就行，输出直接等于输入——网络可以自动"跳过"不需要的层。

残差连接是深层网络能训练的关键——ResNet（2015）提出了它，Transformer（2017）继承了。

### Step 2：确定组装顺序

理解了残差连接之后，一层 Block 的结构就清楚了。Pre-Norm 的原理和选择在第 2 章已经讲过（先 Norm 再计算，训练更稳定）。Block 把这两个设计组合起来：每件事用同样的"Pre-Norm + 计算 + 残差"三件套保护：

```
输入 hidden_states
  │
  ├─ 保存 residual = hidden_states
  ├─ RMSNorm → Attention → 输出 + residual    ← 第一次：token 间交互
  │
  ├─ 保存 residual = hidden_states
  ├─ RMSNorm → FFN/MoE  → 输出 + residual    ← 第二次：token 独立思考
  │
输出 hidden_states（和输入维度完全一样 [B, S, 768]）
```

### Step 3：写代码

核心就是两个对称的"Pre-Norm + 计算 + 残差"分支。除了骨架逻辑，forward 还需要传入推理相关的参数：

| 参数 | 为什么要传 |
|------|-----------|
| `position_embeddings` | RoPE 的 cos/sin 值，推理时 KV Cache 会导致位置偏移，需要从外部传入 |
| `past_key_value` / `use_cache` | KV Cache，推理时只算新 token，复用历史 K/V |
| `attention_mask` | Padding mask，batch 内不同样本长度不同时需要 |

Attention 返回 `(output, past_kv)` 两个值——`past_kv` 是这一层缓存的 K、V 张量，推理时传给下一轮 forward 避免重复计算。

#### 关于命名

前面的底层组件（Attention、FeedForward、RMSNorm）都用 `x`——它们是通用组件，不关心自己在模型里的位置。从 Block 开始往上（Block → Model），用 `hidden_states` 表示"在各层之间流动的隐藏状态"，和 HuggingFace Transformers 的命名惯例一致（LLaMA、Qwen、Mistral 源码都这样）。功能上没区别，纯粹是为了上下游代码读起来连贯。

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
8 层合计:            ≈ 59.0M
```

**FFN 参数量是 Attention 的 3 倍多**——模型的大部分参数在 FFN 里，不在 Attention 里。

## 代码实现

> 📁 `scratch/model/model_minimind.py` 中的 MiniMindBlock，直接复用手写 Attention、FeedForward、MOEFeedForward。

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None):
        # Pre-Norm + Attention + 残差
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings, past_key_value, use_cache, attention_mask,
        )
        hidden_states = hidden_states + residual

        # Pre-Norm + FFN + 残差
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value

# 验证
config = MiniMindConfig()
block = MiniMindBlock(0, config)
x = torch.randn(2, 128, 768)
cos, sin = precompute_freqs_cis(96, 128)
out, kv = block(x, (cos, sin), use_cache=True)
assert out.shape == x.shape
assert kv[0].shape == (2, 128, 4, 96)  # KV Cache: [B, S, n_kv_heads, head_dim]
print(f"Block params: {sum(p.numel() for p in block.parameters()):,}")  # 7,374,528
print("MiniMindBlock verified!")
```

## 对照 MiniMind 源码

| 我们的实现 | MiniMind 源码位置 |
|-----------|-----------------|
| `MiniMindBlock.__init__` | `model_minimind.py:179-184` |
| `forward` | `model_minimind.py:186-194` |
| Pre-Norm + 残差 | 完全一致 |
| MoE 选择 `if not use_moe` | `model_minimind.py:183` |

与上游完全一致——使用手写的 GQA Attention（含 KV Cache、SDPA）、SwiGLU FeedForward、MoE 分支。

## 业界对比

| 特性 | MiniMind | LLaMA 3 | Qwen2.5 | DeepSeek-V2 |
|------|---------|---------|---------|-------------|
| Norm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| FFN | SwiGLU/MoE | SwiGLU | SwiGLU | MoE |
| QK Norm | 有（q_norm/k_norm） | 无 | 无（Qwen2 无显式 QK Norm） | 有（latent RMSNorm，非标准 QK Norm） |
| 残差主干 Norm 数 | 2 | 2 | 2 | 2 |

## 面试话术

> "一层 Transformer Block 就是 Pre-Norm + Attention + 残差 + Pre-Norm + FFN + 残差。
> 我从零组装过——先写各个零件，再在 Block 里串起来。
> Pre-Norm 就是先做 RMSNorm 再做 Attention/FFN，比原始 Post-Norm 训练更稳定。
> 残差连接让梯度能直接跳过计算层流到前面，是深层网络能训练的关键——
> Attention 和 FFN 学的是'在原始表示上的修正量'，不是从零重建。
> 参数量上 FFN 占大头——每层 FFN 5.6M，Attention 才 1.8M，比例约 3:1。
> 所有主流大模型的 Block 结构几乎一样，差异在 Attention（GQA/MLA）和 FFN（SwiGLU/MoE）内部。"

# 08 - MiniMindModel + MiniMindForCausalLM + Config（完整模型 + 超参总表）

> 源码位置：`model/model_minimind.py:10-45`（Config）、`196-288`（Model + CausalLM）

## 一句话总结

把所有组件串成完整模型：Embedding → N × Block → Norm → LM Head。Config 是所有超参的总表，学完前 7 章后再看每个参数都能对应上具体代码。

---

## Part 1: MiniMindModel（196-232 行）——N 层堆叠

### 初始化

```python
class MiniMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size           # 6400
        self.num_hidden_layers = config.num_hidden_layers  # 8

        # ① Embedding 查找表
        self.embed_tokens = nn.Embedding(6400, 768)   # [vocab, hidden] → 01 章

        # ② Dropout
        self.dropout = nn.Dropout(config.dropout)      # 默认 0.0，推理时不生效

        # ③ N 层 Transformer Block
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(8)  # → 07 章
        ])

        # ④ 最终 RMSNorm
        self.norm = RMSNorm(768, eps=1e-6)             # → 02 章

        # ⑤ 预计算 RoPE 频率表，注册为 buffer（不参与梯度更新）
        freqs_cos, freqs_sin = precompute_freqs_cis(    # → 03 章
            dim=96, end=32768, rope_base=1e6
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # persistent=False → 不写入 state_dict（加载时重新计算）
```

### Forward 维度追踪

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None,
            use_cache=False, **kwargs):
    # input_ids: [B=2, S=128] 整数 token id

    batch_size, seq_length = input_ids.shape  # 2, 128

    # 兼容 transformers 的 DynamicCache（新版本 API 变化）
    if hasattr(past_key_values, 'layers'):
        past_key_values = None
    past_key_values = past_key_values or [None] * 8  # 每层一个 KV cache 槽

    # 计算 KV Cache 的起始位置（推理时非 0）
    start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
    # 训练时: start_pos=0
    # 推理第 200 步: start_pos=199

    # ① Embedding
    hidden_states = self.dropout(self.embed_tokens(input_ids))
    # [2, 128] → [2, 128, 768]

    # ② RoPE 频率切片（按位置范围取）
    # 兼容 transformers>=5.x 的 meta-device 初始化（buffer 可能被清零）
    if self.freqs_cos[0, 0] == 0:
        freqs_cos, freqs_sin = precompute_freqs_cis(...)  # 重新计算
        self.freqs_cos, self.freqs_sin = freqs_cos.to(device), freqs_sin.to(device)

    position_embeddings = (
        self.freqs_cos[start_pos:start_pos + seq_length],  # [S, 96]
        self.freqs_sin[start_pos:start_pos + seq_length],  # [S, 96]
    )

    # ③ 逐层过 Transformer Block
    presents = []
    for layer, past_key_value in zip(self.layers, past_key_values):
        hidden_states, present = layer(
            hidden_states,           # [2, 128, 768]
            position_embeddings,     # ([S, 96], [S, 96])
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        presents.append(present)     # 收集每层的 KV Cache
    # hidden_states: [2, 128, 768]（过了 8 层，维度不变）

    # ④ 最终 RMSNorm
    hidden_states = self.norm(hidden_states)  # [2, 128, 768]

    # ⑤ 收集 MoE 辅助 loss（如果用了 MoE）
    aux_loss = sum(
        [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
        hidden_states.new_zeros(1).squeeze()
    )
    # 所有层的 MoE 辅助 loss 累加，没用 MoE 则为 0

    return hidden_states, presents, aux_loss
    # hidden_states: [2, 128, 768]
    # presents: list of 8 × (K, V) tuples
    # aux_loss: scalar
```

### 维度全景

```
input_ids: [2, 128]
    ↓ Embedding
[2, 128, 768]
    ↓ Block 0 (RMSNorm → Attention → 残差 → RMSNorm → FFN → 残差)
[2, 128, 768]
    ↓ Block 1
[2, 128, 768]
    ↓ ... × 8 层
[2, 128, 768]
    ↓ 最终 RMSNorm
[2, 128, 768]
```

从 Embedding 到最后一层，hidden_states 的形状始终是 `[B, S, 768]`，不会变。

---

## Part 2: MiniMindForCausalLM（234-288 行）——LM Head + Loss + Generate

### 初始化

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config=None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)   # PreTrainedModel 的初始化（save/load/generate 接口）

        self.model = MiniMindModel(self.config)  # 上面定义的完整模型

        # LM Head: 把 768 维映射到 vocab_size 维的 logits
        self.lm_head = nn.Linear(768, 6400, bias=False)  # [768, 6400]

        # Weight Tying: Embedding 和 LM Head 共享权重（→ 01 章）
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        self.post_init()  # PreTrainedModel 的权重初始化
```

**继承的意义：**
- `PreTrainedModel`：提供 `save_pretrained()` / `from_pretrained()` 标准接口
- `GenerationMixin`：提供 HuggingFace 的 `generate()` 接口（但 MiniMind 自己重写了）

### Forward（训练时）

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None,
            use_cache=False, logits_to_keep=0, labels=None, **kwargs):

    # 过完整模型
    hidden_states, past_key_values, aux_loss = self.model(
        input_ids, attention_mask, past_key_values, use_cache, **kwargs
    )
    # hidden_states: [2, 128, 768]

    # logits_to_keep: 推理优化，只算最后几个 token 的 logits（省计算）
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    # 训练时 logits_to_keep=0 → slice(-0, None) = slice(None) → 全部 token
    # logits: [2, 128, 6400]

    # 计算 loss（如果提供了 labels）
    loss = None
    if labels is not None:
        # 自回归 loss: 用位置 t 的输出预测位置 t+1 的 token
        x = logits[..., :-1, :].contiguous()    # [2, 127, 6400] — 去掉最后一个
        y = labels[..., 1:].contiguous()         # [2, 127]       — 去掉第一个
        loss = F.cross_entropy(
            x.view(-1, x.size(-1)),              # [254, 6400]
            y.view(-1),                           # [254]
            ignore_index=-100                     # 忽略 padding/mask 的位置
        )

    return MoeCausalLMOutputWithPast(
        loss=loss,              # 主 loss（cross entropy）
        aux_loss=aux_loss,      # MoE 辅助 loss（load balancing）
        logits=logits,          # [2, 128, 6400]
        past_key_values=past_key_values,
        hidden_states=hidden_states,
    )
```

### Loss 计算的错位（面试高频）

```
位置:    0    1    2    3    4
token:  [BOS] 我   是   小   夏
labels: [BOS] 我   是   小   夏

预测:    logits[0] 预测 labels[1]="我"
         logits[1] 预测 labels[2]="是"
         logits[2] 预测 labels[3]="小"
         logits[3] 预测 labels[4]="夏"
```

代码实现：

- `x = logits[:, :-1]` — 去掉最后一个位置（4）
- `y = labels[:, 1:]` — 去掉第一个位置（0）
- $\text{loss} = \text{cross\_entropy}(x, y)$

`ignore_index=-100`：SFT 训练时会把非 assistant 回复的位置 label 设为 -100，只对回复部分计算 loss。

### Generate（推理时，256-288 行）

```python
@torch.inference_mode()
def generate(self, inputs=None, max_new_tokens=8192, temperature=0.85,
             top_p=0.85, top_k=50, eos_token_id=2, streamer=None,
             use_cache=True, num_return_sequences=1, do_sample=True,
             repetition_penalty=1.0, **kwargs):

    input_ids = inputs.repeat(num_return_sequences, 1)  # 支持多路生成
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        # 只传入新 token（靠 KV Cache 记住历史）
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = self.forward(input_ids[:, past_len:], ...)
        # input_ids[:, past_len:]: 训练时传全部，推理时只传新增的 token

        logits = outputs.logits[:, -1, :] / temperature
        # 只取最后一个位置的 logits: [B, 6400]
        # / temperature: >1 更随机，<1 更确定

        # Repetition Penalty: 降低已出现 token 的分数
        if repetition_penalty != 1.0:
            seen = torch.unique(input_ids[i])
            logits[i, seen] = where(score > 0, score / penalty, score * penalty)
            # 正分除以 penalty（降低），负分乘以 penalty（更负）

        # Top-K: 只保留分数最高的 K 个 token
        if top_k > 0:
            logits[logits < topk(logits, top_k)[0][..., -1, None]] = -inf

        # Top-P (Nucleus Sampling): 只保留累积概率 < p 的 token
        if top_p < 1.0:
            sorted_logits, sorted_indices = sort(logits, descending=True)
            mask = cumsum(softmax(sorted_logits), dim=-1) > top_p
            mask[..., 1:] = mask[..., :-1].clone()  # 保证至少留一个
            mask[..., 0] = 0
            logits[mask.scatter(1, sorted_indices, mask)] = -inf

        # 采样或贪心
        next_token = multinomial(softmax(logits), 1) if do_sample else argmax(logits)

        # 已结束的序列强制输出 eos
        if eos_token_id is not None:
            next_token = where(finished, eos_token_id, next_token)

        input_ids = cat([input_ids, next_token], dim=-1)
        finished |= next_token.eq(eos_token_id)
        if finished.all(): break

    return input_ids
```

### 解码策略对比

```
Greedy:         永远选概率最高的 → 确定性，但容易重复
Temperature:    除以 T 调节分布 → T>1 更平坦（随机），T<1 更尖锐（确定）
Top-K:          只保留前 K 个候选 → 砍掉长尾噪声
Top-P:          只保留累积概率 < P 的候选 → 动态截断，比 Top-K 更灵活
Rep Penalty:    降低已出现 token 的分数 → 减少重复

MiniMind 默认: temperature=0.85, top_p=0.85, top_k=50 — 三者同时使用
```

---

## Part 3: Config 超参总表

现在你学完了所有组件，回来看 Config 每个参数都能对上：

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    ### 模型尺寸 → 决定模型多大
    hidden_size = 768              # d_model → 01 Embedding, 所有层的宽度
    num_hidden_layers = 8          # Transformer 层数 → 07 Block × 8
    num_attention_heads = 8        # Q 头数 → 04 Attention
    num_key_value_heads = 4        # KV 头数 → 04 GQA (n_rep=2)
    head_dim = 96                  # 768 // 8 → 04 每个头的维度
    vocab_size = 6400              # 词表大小 → 01 Embedding [6400, 768]

    ### FFN 尺寸
    hidden_act = 'silu'            # SiLU 激活 → 05 SwiGLU
    intermediate_size = 2432       # ceil(768×π/64)×64 → 05 FFN 中间层

    ### 位置编码
    max_position_embeddings = 32768  # 最大序列长度 → 03 RoPE 预计算范围
    rope_theta = 1e6               # RoPE 基频 → 03 频率计算
    inference_rope_scaling = False  # YaRN 开关 → 03 推理外推

    ### 归一化
    rms_norm_eps = 1e-6            # RMSNorm 防除零 → 02

    ### 正则化
    dropout = 0.0                  # Dropout 率 → 04 attn_dropout, resid_dropout

    ### 权重共享
    tie_word_embeddings = True     # Embedding 和 LM Head 共享 → 01

    ### Flash Attention
    flash_attn = True              # 是否用 SDPA → 04 Flash 路径

    ### 特殊 token
    bos_token_id = 1               # 序列开始
    eos_token_id = 2               # 序列结束 → generate 时的停止条件

    ### MoE 配置（use_moe=True 时生效）→ 06
    use_moe = False                # MoE 开关
    num_experts = 4                # 专家数
    num_experts_per_tok = 1        # 每 token 激活几个专家
    moe_intermediate_size = 2432   # 每个专家的 FFN 中间层大小
    norm_topk_prob = True          # 路由权重归一化
    router_aux_loss_coef = 5e-4    # 辅助 loss 系数
```

### 参数量完整计算

**Embedding:**

$$6400 \times 768 = 4{,}915{,}200$$

与 LM Head 共享，只算一份。

**每层 Attention:**

- $W_q$: $768 \times 768 = 589{,}824$
- $W_k$: $768 \times 384 = 294{,}912$
- $W_v$: $768 \times 384 = 294{,}912$
- $W_o$: $768 \times 768 = 589{,}824$
- $\text{q\_norm} + \text{k\_norm}$: $96 \times 2 = 192$
- 小计: $1{,}769{,}664$

**每层 FFN (SwiGLU):**

- gate: $768 \times 2432 = 1{,}867{,}776$
- up: $768 \times 2432 = 1{,}867{,}776$
- down: $2432 \times 768 = 1{,}867{,}776$
- 小计: $5{,}603{,}328$

**归一化层:**

- 每层 RMSNorm $\times 2$: $768 \times 2 = 1{,}536$
- 最终 RMSNorm: $768$

**汇总:**

- 每层合计: $1{,}769{,}664 + 5{,}603{,}328 + 1{,}536 = 7{,}374{,}528$
- 8 层合计: $7{,}374{,}528 \times 8 = 58{,}996{,}224$

$$\text{总计} = 58{,}996{,}224 + 4{,}915{,}200 + 768 \approx 63.9M \text{ 参数}$$

## 业界对比

| 特性 | MiniMind (64M) | Qwen2.5-7B | DeepSeek-V2 (236B) | LLaMA 3-8B |
|------|---------------|------------|-------------------|------------|
| hidden_size | 768 | 3584 | 5120 | 4096 |
| 层数 | 8 | 28 | 60 | 32 |
| Q 头 / KV 头 | 8/4 | 28/4 | 128/1 (MLA) | 32/8 |
| head_dim | 96 | 128 | 128 | 128 |
| FFN | SwiGLU 2432 | SwiGLU 18944 | MoE 160专家 | SwiGLU 14336 |
| vocab_size | 6400 | 151936 | 102400 | 128256 |
| max_seq_len | 32768 | 131072 | 163840 | 8192 |
| 位置编码 | RoPE 1e6 | RoPE 1e6 | RoPE 1e4 | RoPE 5e5 |
| Weight Tying | 是 | 否 | 否 | 否 |
| QK Norm | 是 | 是 | 是 | 否 |

### 大模型为什么不用 Weight Tying

MiniMind 用 Weight Tying 是因为 $\text{vocab\_size} \times \text{hidden\_size} = 6400 \times 768 \approx 5M$ 参数，占 $64M$ 总参数的 $\sim 8\%$，共享省很多。
Qwen 的 $\text{vocab} \times \text{hidden} = 151936 \times 3584 \approx 545M$，占 $7B$ 的 $\sim 7.8\%$，但大模型参数量充足，不 tie 反而让 Embedding 和 LM Head 各自专注于自己的任务（编码 vs 解码），效果略好。

## 面试话术

> "完整模型就是 Embedding → 8 层 Transformer Block → RMSNorm → LM Head。
> 每层是 Pre-RMSNorm + GQA Attention + 残差 + Pre-RMSNorm + SwiGLU FFN + 残差。
> 训练 loss 是标准的自回归 cross entropy——用 t 位置的 logits 预测 t+1 的 token，
> SFT 时把非 assistant 部分的 label 设为 -100 让 cross_entropy 自动忽略。
> 推理用自己写的 generate，支持 temperature + top-k + top-p + repetition penalty，
> 配合 KV Cache 做增量推理——每步只传新 token 进去，历史 KV 从 cache 拿。
> 整个模型约 64M 参数，其中 FFN 占大头（每层 5.6M），Attention 约 1.8M/层，
> Embedding 因为做了 Weight Tying 和 LM Head 共享，省了约 5M。"

## 手写练习

> 📁 继续在 `scratch/model/model_minimind.py` 中追加（01-07 的所有组件前面已定义，这里把它们串成完整模型）

```python
import math, torch, torch.nn.functional as F
from torch import nn

# 复用之前的组件
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class SimpleBlock(nn.Module):
    """简化版 Block，用 nn.MultiheadAttention 替代手写 Attention"""
    def __init__(self, hidden_size, n_heads, intermediate_size):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.ffn = FeedForward(hidden_size, intermediate_size)

    def forward(self, x):
        # Pre-Norm + Attention + 残差
        normed = self.norm1(x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, is_causal=True)
        x = x + attn_out
        # Pre-Norm + FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x

class MiniModel(nn.Module):
    def __init__(self, vocab_size=6400, hidden_size=768, n_layers=8,
                 n_heads=8, intermediate_size=2432):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleBlock(hidden_size, n_heads, intermediate_size)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight  # Weight Tying

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            pred = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            target = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(pred, target, ignore_index=-100)
        return logits, loss

# 验证
model = MiniModel(vocab_size=6400, hidden_size=768, n_layers=8,
                  n_heads=8, intermediate_size=2432)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,}")

# 模拟训练
input_ids = torch.randint(0, 6400, (2, 128))
logits, loss = model(input_ids, labels=input_ids)
print(f"Logits: {logits.shape}")  # [2, 128, 6400]
print(f"Loss: {loss.item():.4f}")
loss.backward()
print("Full model forward + backward verified!")
```

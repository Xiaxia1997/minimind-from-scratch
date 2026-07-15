# 08 - 完整模型组装：Config + Model + CausalLM + Generate

## 为什么需要这一章？

前 7 章写了所有独立零件：Embedding、RMSNorm、RoPE、Attention、FFN、MoE、Block。但它们还是散装的——每个组件各自接受 kwargs，参数各传各的，无法跑通"输入 token → 输出下一个 token"的完整链路。

这一章做三件事，写完之后模型就能训练和推理了：

1. **MiniMindConfig**：把所有超参收到一个对象里，所有组件从它读参数
2. **MiniMindModel**：Embedding + 8 层 Block + 最终 Norm，串成骨干网络
3. **MiniMindForCausalLM**：加上 LM Head、Loss 计算和 Generate 推理循环

## 从零开始写

### Step 1：MiniMindConfig——把散落的超参收到一处

**动机**：前面写的 Attention 初始化是 `Attention(hidden_size=768, n_heads=8, n_kv_heads=4, head_dim=96)`，FeedForward 是 `FeedForward(dimension=768, intermediate_dimension=2432)`——每个组件自己管自己的参数。问题来了：

- 想改 hidden_size 从 768 到 512，要改 Attention、FeedForward、RMSNorm、Embedding 四个地方
- head_dim 和 hidden_size 有约束关系（`head_dim = hidden_size / n_heads`），分散在各处容易不一致
- 新增一个全局参数（比如 dropout），得给每个组件加 kwarg

解决方法很直接：**用一个 Config 对象集中管理所有超参，组件只接收 config 一个参数**。

```python
class MiniMindConfig:
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        # --- 模型尺寸 ---
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", hidden_size // self.num_attention_heads)
        self.vocab_size = kwargs.get("vocab_size", 6400)

        # --- FFN ---
        self.hidden_act = kwargs.get("hidden_act", "silu")
        # intermediate_size 的经验公式：ceil(hidden_size × π / 64) × 64
        # 8/3 ≈ π 是 SwiGLU 的常用扩展比，对齐到 64 的倍数方便 GPU 计算
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )

        # --- 位置编码 ---
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        # --- 归一化 & 正则化 ---
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.dropout = kwargs.get("dropout", 0.0)

        # --- 权重共享 & Flash Attention ---
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.flash_attn = kwargs.get("flash_attn", True)

        # --- 特殊 token ---
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)

        # --- MoE（use_moe=True 时生效） ---
        self.use_moe = use_moe
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
```

有了 Config，所有组件的初始化就统一了：`Attention(config)`、`FeedForward(config)`、`MiniMindBlock(layer_id, config)`。改任何超参只需改 Config 一处。

学完前 7 章后，每个参数都能对上具体代码：

| 参数 | 默认值 | 对应章节 | 用在哪 |
|------|--------|---------|--------|
| `hidden_size` | 768 | 01 Embedding | 所有层的宽度 |
| `num_hidden_layers` | 8 | 07 Block × 8 | 堆叠层数 |
| `num_attention_heads` | 8 | 04 Attention | Q 头数 |
| `num_key_value_heads` | 4 | 04 GQA | KV 头数，n_rep=2 |
| `head_dim` | 96 | 04 | 768 / 8 |
| `vocab_size` | 6400 | 01 | Embedding 和 LM Head 的词表维 |
| `intermediate_size` | 2432 | 05 SwiGLU | FFN 中间层宽度 |
| `max_position_embeddings` | 32768 | 03 RoPE | 预计算频率表的长度 |
| `rope_theta` | 1e6 | 03 | RoPE 基频 |
| `rms_norm_eps` | 1e-6 | 02 | RMSNorm 防除零 |
| `dropout` | 0.0 | 04 | attn_dropout, resid_dropout |
| `tie_word_embeddings` | True | 01 | Embedding 和 LM Head 共享 |
| `flash_attn` | True | 04 | 是否用 SDPA 路径 |
| `use_moe` | False | 06 | MoE 开关 |
| `num_experts` | 4 | 06 | 专家数 |
| `num_experts_per_tok` | 1 | 06 | 每 token 激活几个 |

### Step 2：MiniMindModel——把 8 个 Block 串成骨干

**动机**：第 7 章写了一个 MiniMindBlock，它接收 `[B, S, 768]` 输出 `[B, S, 768]`。但一个 Block 的表达能力有限——浅层学语法，深层学语义，需要多层叠加。同时 token id 进来之前需要 Embedding 查表，最后一层出来之后需要 Norm 稳定数值。MiniMindModel 就是把这些串起来：

```
token ids [B, S]
    ↓ Embedding           → 01 章
[B, S, 768]
    ↓ Dropout
[B, S, 768]
    ↓ Block 0             → 07 章
    ↓ Block 1
    ↓ ...
    ↓ Block 7
[B, S, 768]
    ↓ 最终 RMSNorm        → 02 章
[B, S, 768]
```

串起来很简单，但有两个新的设计问题要解决。

#### 问题 1：RoPE 频率在哪里算？

写 Block 时，position_embeddings（cos/sin）是从外部传入的。为什么不在每个 Block 的 Attention 里自己算？

因为 RoPE 频率只跟位置和 head_dim 有关，跟层数无关——8 层用的是**同一组** cos/sin 值。如果每层自己算，就重复算了 8 次。所以在 Model 层面预计算一次，用 `register_buffer` 存起来，给每层共享。

`register_buffer` 的含义：这个张量是模型的一部分（会跟着 `.to(device)` 移动到 GPU），但**不参与梯度更新**——因为 RoPE 频率是确定性公式算出来的，不需要学习。`persistent=False` 表示不写入 `state_dict`（加载模型时重新计算就行，省存储空间）。

#### 问题 2：KV Cache 怎么管理？

推理时，每层 Attention 都有自己的 KV Cache（第 4 章写过）。Model 需要做三件事：

1. 接收上一轮的 `past_key_values`（一个列表，每层一个 `(K, V)` tuple）
2. 把每层的 `(K, V)` 分发给对应的 Block
3. 收集每层返回的新 `(K, V)` 传给下一轮

训练时 `past_key_values=None`，每层独立算完整序列。推理时通过这个列表实现增量计算。

同时还要算 `start_pos`——当前序列在 RoPE 频率表中的起始位置。训练时 `start_pos=0`（从头算），推理第 200 步时 `start_pos=199`（因为前 199 个 token 的 KV 已经在 cache 里了，新 token 的位置编号是 199）。

理解了这两个问题，代码就很直白了：

```python
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # [6400, 768]
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, S = input_ids.shape  # [2, 128]

        # 训练时: [None, None, ..., None]（8 个 None）
        # 推理时: [(K0, V0), (K1, V1), ..., (K7, V7)]
        past_key_values = past_key_values or [None] * len(self.layers)

        # RoPE 起始位置：训练时 0，推理第 200 步时 199
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # ① Embedding: [2, 128] → [2, 128, 768]
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # ② RoPE 频率切片：按当前位置范围从预计算表中取
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + S],  # [S, 96]
            self.freqs_sin[start_pos:start_pos + S],  # [S, 96]
        )

        # ③ 逐层过 Transformer Block
        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_kv, use_cache=use_cache, attention_mask=attention_mask,
            )
            presents.append(present)

        # ④ 最终 RMSNorm
        hidden_states = self.norm(hidden_states)  # [2, 128, 768]

        # ⑤ 收集 MoE 辅助 loss
        # hidden_states.new_zeros(1): tensor 工厂方法，在同设备同 dtype 上创建零标量
        # sum([], start): 列表为空时直接返回 start（没有 MoE 层时 aux_loss=0）
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss
```

### Step 3：MiniMindForCausalLM——加上 LM Head 和 Loss

**动机**：MiniMindModel 输出 `[B, S, 768]` 的隐藏状态，但训练需要 loss，推理需要下一个 token 的概率分布。还差两样东西：

1. **LM Head**：一个线性层 `nn.Linear(768, 6400)`，把隐藏状态映射到词表维度，得到每个位置上"下一个词应该是什么"的分数（logits）
2. **Loss 计算**：自回归 cross entropy

```
MiniMindModel output: [B, S, 768]
    ↓ LM Head (nn.Linear 768 → 6400)
logits: [B, S, 6400]         ← 每个位置对 6400 个词的得分
    ↓ Cross Entropy (with labels)
loss: scalar
```

#### 自回归 Loss 的错位（面试高频）

自回归的核心是"用位置 t 的输出预测位置 t+1 的 token"：

```
位置:    0    1    2    3    4
token:  [BOS] 我   是   小   夏
labels: [BOS] 我   是   小   夏

预测:    logits[0] 预测 labels[1]="我"
         logits[1] 预测 labels[2]="是"
         logits[2] 预测 labels[3]="小"
         logits[3] 预测 labels[4]="夏"
```

代码上怎么实现这个"错位"？把 logits 去掉最后一个位置，labels 去掉第一个位置，两边就对齐了：

```python
x = logits[..., :-1, :]   # 去掉最后一个位置 → [B, S-1, 6400]
y = labels[..., 1:]        # 去掉第一个位置 → [B, S-1]
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

`ignore_index=-100`：SFT 训练时会把非 assistant 回复的位置 label 设为 -100，`cross_entropy` 自动跳过这些位置，只对回复部分计算 loss。

#### logits_to_keep 优化

推理时通常只需要最后一个 token 的 logits（用来采样下一个词），不需要对整个序列都算 LM Head 的矩阵乘法。`logits_to_keep` 控制只对最后几个 token 做 768 → 6400 映射，省 LM Head 的计算和 logits 显存。注意这**不会减少** 8 层 Transformer 的计算——前面的层还是要算完整序列，只是最后一步可以省。

```python
class MiniMindForCausalLM(nn.Module):
    def __init__(self, config: MiniMindConfig = None):
        super().__init__()
        self.config = config or MiniMindConfig()
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # Weight Tying（第 1 章）：Embedding 和 LM Head 共享权重
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache
        )
        # logits_to_keep=0 时: slice(0, None) = slice(None) → 全部 token（训练）
        # logits_to_keep=1 时: slice(-1, None) → 只取最后一个（推理）
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # [B, ?, 6400]

        loss = None
        if labels is not None:
            # .contiguous(): 切片操作可能返回非连续的"视图"，后面的 .view() 要求连续
            x = logits[..., :-1, :].contiguous()  # [B, S-1, 6400]
            y = labels[..., 1:].contiguous()       # [B, S-1]
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        return {'loss': loss, 'aux_loss': aux_loss, 'logits': logits,
                'past_key_values': past_key_values}
```

### Step 4：generate——让模型自己说话

**动机**：训练时有标准答案（labels），模型只需要算 loss 反传梯度。推理时没有答案——模型要自己一个 token 一个 token 地生成，每步把刚生成的 token 喂回去，直到输出结束符或达到长度上限。

#### Prefill vs Decode

```
第一轮（Prefill）：输入完整 prompt [BOS, 你, 好]，过 8 层，缓存所有层的 KV
    → 用最后一个位置的 logits 采样下一个 token "啊"

第二轮（Decode）：只输入新 token [啊]，past_key_values 带着历史 KV
    → 新 token 的 Q 和历史所有 K 做 attention，但只算 1 个 token 的前向
    → 采样下一个 token "！"

第三轮（Decode）：只输入 [！]，cache 继续累积...
    → 直到生成 EOS 或达到 max_new_tokens
```

关键洞察：**Decode 阶段每步只传 1 个 token 进模型**，复用 cache 中的历史 KV。这就是 KV Cache 省计算的原理——不需要每步都重新算整个序列。

#### 采样策略

生成时不能简单地取概率最高的 token（贪心），否则输出会很无聊、容易重复。MiniMind 用四种策略叠加：

| 策略 | 做什么 | 效果 |
|------|--------|------|
| Temperature | logits 除以 T | T > 1 更随机，T < 1 更确定 |
| Top-K | 只保留分数最高的 K 个 | 砍掉长尾噪声 |
| Top-P (Nucleus) | 按概率从高到低，保留累积概率达到 P 的最小集合 | 比 Top-K 更灵活——概率集中时候选少，分散时候选多 |
| Repetition Penalty | 降低已出现 token 的分数 | 减少重复 |

MiniMind 默认同时使用：`temperature=0.85, top_p=0.85, top_k=50`。

```python
@torch.inference_mode()
def generate(self, inputs, max_new_tokens=8192, temperature=0.85, top_p=0.85,
             top_k=50, eos_token_id=2, use_cache=True, do_sample=True,
             repetition_penalty=1.0):
    input_ids = inputs
    past_key_values = None
    # 追踪每个 batch 中的序列是否已生成 EOS
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        # ---- Prefill vs Decode ----
        # 第一轮 past_key_values=None → past_len=0 → 传入整个 prompt
        # 后续轮 past_len=缓存长度 → 只传新 token
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        outputs = self.forward(
            input_ids[:, past_len:], use_cache=use_cache, past_key_values=past_key_values
        )
        past_key_values = outputs['past_key_values']

        # 只取最后一个位置的 logits，除以 temperature 调节分布
        logits = outputs['logits'][:, -1, :] / temperature  # [B, 6400]

        # ---- Repetition Penalty ----
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[0]):
                # torch.unique: 去重，得到这个序列中出现过的所有 token id
                seen = torch.unique(input_ids[i])
                score = logits[i, seen]
                # 正分除以 penalty（降低），负分乘以 penalty（更负）→ 都是在压低
                logits[i, seen] = torch.where(
                    score > 0, score / repetition_penalty, score * repetition_penalty
                )

        # ---- Top-K ----
        if top_k > 0:
            # torch.topk(logits, top_k)[0]: top-k 的值（丢弃索引）, [B, top_k]
            # [..., -1, None]: 取最小的那个值，None 加一维 → [B, 1] 用于广播
            threshold = torch.topk(logits, top_k)[0][..., -1, None]
            # 低于第 K 名的全部设为负无穷（softmax 后概率趋近 0）
            logits[logits < threshold] = -float('inf')

        # ---- Top-P (Nucleus Sampling) ----
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            # 累积概率超过 top_p 的位置标记为 True
            mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            # mask 右移一位：让刚好超过 top_p 的那个 token 也保留
            # .clone(): 必须深拷贝——源 [..., :-1] 和目标 [..., 1:] 有重叠，
            #   不拷贝会原地覆盖后读到错误数据
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0  # 概率最高的永远保留
            # .scatter(dim, index, src): gather 的逆操作
            #   把排序后的 mask 按 sorted_indices 散布回原始 token 顺序
            logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

        # ---- 采样或贪心 ----
        if do_sample:
            # torch.multinomial(probs, 1): 按概率分布随机采 1 个，返回索引
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # 已结束的序列强制填 eos（不让它继续产生有意义的 token）
        if eos_token_id is not None:
            # .new_full: tensor 工厂方法，在 next_token 同设备上创建全为指定值的张量
            next_token = torch.where(
                finished.unsqueeze(-1),  # [B] → [B, 1] 用于与 next_token [B, 1] 匹配
                next_token.new_full((next_token.shape[0], 1), eos_token_id),
                next_token,
            )

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # .eq(): 逐元素相等比较，返回布尔张量
        # |= : in-place 或，新产生 EOS 的序列标记为 finished
        finished |= next_token.squeeze(-1).eq(eos_token_id)
        if finished.all():
            break

    return input_ids
```

## 维度全景

```
input_ids: [2, 128]                    整数 token id
    ↓ Embedding
[2, 128, 768]
    ↓ Dropout
[2, 128, 768]
    ↓ Block 0 (RMSNorm → Attention → 残差 → RMSNorm → FFN → 残差)
[2, 128, 768]
    ↓ Block 1 ... Block 7
[2, 128, 768]
    ↓ 最终 RMSNorm
[2, 128, 768]                          ← MiniMindModel 的输出
    ↓ LM Head (Linear 768 → 6400)
[2, 128, 6400]                         ← logits
    ↓ Cross Entropy (with labels)
scalar                                 ← loss
```

从 Embedding 到最后一层 Block，hidden_states 的形状始终是 `[B, S, 768]`，不会变。LM Head 把最后一维从 768 展开到 6400。

## 参数量

Embedding: $6400 \times 768 = 4{,}915{,}200$（与 LM Head 共享，只算一份）

每层:
- Attention: $1{,}769{,}664$（第 4 章已算）
- FFN (SwiGLU): $5{,}603{,}328$（第 5 章已算）
- RMSNorm × 2: $768 \times 2 = 1{,}536$
- 小计: $7{,}374{,}528 \approx 7.4M$

最终 RMSNorm: $768$

$$\text{总计} = 7{,}374{,}528 \times 8 + 4{,}915{,}200 + 768 = 63{,}912{,}192 \approx 63.9M$$

## 对照 MiniMind 源码

| 我们的实现 | MiniMind 源码位置 |
|-----------|-----------------|
| `MiniMindConfig` | `model_minimind.py:10-45` |
| `MiniMindModel.__init__` | `model_minimind.py:196-207` |
| `MiniMindModel.forward` | `model_minimind.py:209-232` |
| `MiniMindForCausalLM.__init__` | `model_minimind.py:234-243` |
| `MiniMindForCausalLM.forward` | `model_minimind.py:245-253` |
| `generate` | `model_minimind.py:256-288` |

核心逻辑完全一致。两处简化：
1. 我们的 CausalLM 继承 `nn.Module` 而非 `PreTrainedModel + GenerationMixin`（后者提供 HuggingFace 的 save/load/generate 接口，不影响模型逻辑）
2. generate 省略了 streamer（流式输出）和 attention_mask 扩展（batch 推理时需要），核心采样逻辑一致

## 业界对比

| 特性 | MiniMind (64M) | Qwen2.5-7B | DeepSeek-V2 (236B) | LLaMA 3-8B |
|------|---------------|------------|-------------------|------------|
| hidden_size | 768 | 3584 | 5120 | 4096 |
| 层数 | 8 | 28 | 60 | 32 |
| Q 头 / KV 头 | 8/4 | 28/4 | 128 / MLA | 32/8 |
| head_dim | 96 | 128 | 192 | 128 |
| FFN | SwiGLU 2432 | SwiGLU 18944 | MoE top-6/160 | SwiGLU 14336 |
| vocab_size | 6400 | 152064 | 102400 | 128256 |
| max_seq_len | 32768 | 131072 | 163840 | 8192 |
| Weight Tying | 是 | 否 | 否 | 否 |

### 大模型为什么不用 Weight Tying

MiniMind 用 Weight Tying 省了 $6400 \times 768 = 4.9M$ 参数，占总量 ~8%，收益显著。大模型的 vocab_size 和 hidden_size 差距大（如 Qwen 的 152064 vs 3584），Embedding 矩阵本身就不对称，共享的假设（"输入表示 ≈ 输出分类"）不太成立。且大模型参数量级在 7B+，省这点参数意义不大。

## 面试话术

> "完整模型就是 Embedding → 8 层 Transformer Block → RMSNorm → LM Head。
> 我从零组装过——先写 Config 统一超参，再把 Block 用 ModuleList 堆 8 层，
> Embedding 前面接、RMSNorm 后面收、LM Head 做最后的 768→6400 映射。
> RoPE 的 cos/sin 在 Model 层面预计算一次 register_buffer，所有层共享。
> 训练 loss 是自回归 cross entropy——logits 去掉最后一个位置，labels 去掉第一个位置，错开一位。
> SFT 时把非 assistant 部分的 label 设为 -100 让 cross_entropy 自动忽略。
> 推理用 KV Cache 做增量解码——Prefill 过一遍完整 prompt 缓存所有 KV，
> Decode 阶段每步只传 1 个新 token，新 Q 和历史 KV 做 attention。
> 采样用 temperature + top-k + top-p + repetition penalty 四件套。
> 整个模型约 64M 参数，FFN 占大头（每层 5.6M），Attention 约 1.8M/层。"

## 代码实现

> 📁 `scratch/model/model_minimind.py` 直接复用 01-07 章的所有手写组件，组装完整模型。
> 参数量 **63,912,192 ≈ 63.9M**，与上游完全一致。

```python
class MiniMindConfig:
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", hidden_size // self.num_attention_heads)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.dropout = kwargs.get("dropout", 0.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.use_moe = use_moe
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, S = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + S],
            self.freqs_sin[start_pos:start_pos + S],
        )
        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_kv, use_cache=use_cache, attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(nn.Module):
    def __init__(self, config: MiniMindConfig = None):
        super().__init__()
        self.config = config or MiniMindConfig()
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return {'loss': loss, 'aux_loss': aux_loss, 'logits': logits,
                'past_key_values': past_key_values}

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens=8192, temperature=0.85, top_p=0.85,
                 top_k=50, eos_token_id=2, use_cache=True, do_sample=True,
                 repetition_penalty=1.0):
        input_ids = inputs
        past_key_values = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], use_cache=use_cache,
                                   past_key_values=past_key_values)
            past_key_values = outputs['past_key_values']
            logits = outputs['logits'][:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = (torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                          if do_sample else torch.argmax(logits, dim=-1, keepdim=True))
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1),
                                         next_token.new_full((next_token.shape[0], 1), eos_token_id),
                                         next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all(): break
        return input_ids


# 验证
config = MiniMindConfig()
model = MiniMindForCausalLM(config)
print(f"Dense params: {sum(p.numel() for p in model.parameters()):,}")  # 63,912,192

input_ids = torch.randint(0, 6400, (2, 128))
result = model(input_ids, labels=input_ids)
print(f"Logits: {result['logits'].shape}")  # [2, 128, 6400]
print(f"Loss: {result['loss'].item():.4f}")
result['loss'].backward()

# KV Cache: Prefill → Decode
model.eval()
with torch.no_grad():
    prompt = torch.randint(0, 6400, (1, 10))
    r1 = model(prompt, use_cache=True)                                     # Prefill
    r2 = model(torch.randint(0, 6400, (1, 1)),                            # Decode
               past_key_values=r1['past_key_values'], use_cache=True)
    print(f"Decode logits: {r2['logits'].shape}")                          # [1, 1, 6400]
    print(f"Cache K[0]: {r2['past_key_values'][0][0].shape}")              # [1, 11, 4, 96]

# MoE 版本
moe_model = MiniMindForCausalLM(MiniMindConfig(use_moe=True))
print(f"MoE params: {sum(p.numel() for p in moe_model.parameters()):,}")   # 198,416,640
moe_r = moe_model(input_ids, labels=input_ids)
(moe_r['loss'] + moe_r['aux_loss']).backward()

print("Full model (Dense + KV Cache + MoE) verified!")
```

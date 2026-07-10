# 01 - Embedding：让模型"认字"

## 为什么需要 Embedding？

神经网络只能处理连续的浮点向量，但我们的输入是离散的 token id（整数）。
需要一个"查找表"把每个 id 映射成一个固定长度的向量——这就是 Embedding。

```
输入:  [3, 42, 7]       ← 3 个 token id
输出:  [[0.2, -0.1, ...],
        [0.5,  0.3, ...],
        [-0.1, 0.7, ...]]  ← 3 个 768 维向量
```

## 从零开始写

### Step 1：最简 Embedding

```python
import torch
from torch import nn

vocab_size = 6400    # 词表大小：能认多少个"字"
hidden_size = 768    # 每个字映射成多长的向量

embed_tokens = nn.Embedding(vocab_size, hidden_size)
# 本质：一个 [6400, 768] 的可学习矩阵
# 每一行是一个 token 的向量表示，初始随机，训练中更新
```

nn.Embedding 没有任何计算——纯查表。`embed_tokens(3)` 就是取第 3 行。

```python
input_ids = torch.tensor([[1, 42, 7, 100]])  # [batch=1, seq_len=4]
hidden_states = embed_tokens(input_ids)       # [1, 4, 768]
```

### Step 2：要不要在这里加位置编码？

原始 Transformer 在 Embedding 之后直接加绝对位置编码：
```
原始 Transformer:  Embedding(x) + PositionalEncoding(pos)  → 进 Attention
```

我们不这样做。位置信息用 RoPE 在 Attention 内部注入（详见 [03-RoPE](03-RoPE.md)），好处是：
- RoPE 利用 Q·K 点积的正交性，天然编码**相对距离**而非绝对位置
- 如果加在 Embedding 层，后续的 Norm/FFN 等非线性操作会破坏旋转矩阵的正交性

所以 Embedding 输出的向量**不含任何位置信息**——纯语义。

```
我们的设计:
  Embedding(x)                    → hidden_states（纯语义，没有位置信息）
      ↓
  Attention 内部:
    Q, K = W_q(x), W_k(x)
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)   ← 位置信息在这里才注入
```

### Step 3：加 Dropout

训练时随机把一些维度置零，防过拟合：

```python
dropout = nn.Dropout(0.0)  # MiniMind 默认 0.0，但架构里留了这个口子
hidden_states = dropout(embed_tokens(input_ids))
# [batch, seq_len, 768]
```

#### Dropout 是什么？解决什么问题？

**问题：过拟合**——模型在训练集上表现好，但在新数据上表现差。参数太多时模型会"死记硬背"训练数据的噪声，而不是学到通用规律。

**Dropout 的做法**：训练时每次前向传播，随机把一部分神经元的输出置零（"关掉"）。

```
训练时（dropout=0.3，30% 概率关掉）：
输入:  [0.5, 0.8, 0.3, 0.6, 0.9]
mask:  [ 1,   0,   1,   1,   0 ]    ← 随机生成
输出:  [0.71, 0,  0.43, 0.86, 0]    ← 存活的值除以 (1-0.3) 放大

推理时：所有神经元保留，不做任何操作
```

**为什么存活值要放大？** 训练时只有 70% 的神经元工作，推理时 100% 都工作。不放大的话推理时输出总量比训练时大 ~43%，数值不匹配。除以 `(1-p)` 让训练和推理的期望输出一致——叫 **inverted dropout**。

**Dropout 的好处：**
1. **防止神经元"共谋"**：随机关掉一部分，迫使每个神经元独立学到有用特征
2. **隐式集成效果**：每次前向传播用不同的子网络，最终相当于对指数级多个子网络做平均
3. **提高泛化能力**：模型不能依赖任何单个特征，学到更鲁棒的表示

#### 为什么 MiniMind 默认 dropout=0.0（不用）？

这是现代大模型的普遍做法——**GPT-3、LLaMA、Qwen、DeepSeek 在预训练阶段都不用 dropout。**

| 因素 | 解释 |
|------|------|
| 数据量足够大 | 预训练数据几百 GB~TB 级，模型难过拟合，dropout 的正则化不必要 |
| 训练效率 | dropout 让梯度估计方差变大，需要更多步数收敛，大模型训练本来就贵 |
| 模型容量不浪费 | 大模型参数量相对数据量并不过剩，不需要人为限制容量 |
| 其他正则化替代 | RMSNorm、权重衰减（weight decay）、数据增强已提供足够正则化 |

#### 各训练阶段的 dropout 策略

| 阶段 | 是否用 dropout | 原因 | 实锤 |
|------|---------------|------|------|
| 预训练 | 不用（0.0） | 数据量大，不易过拟合 | LLaMA/Qwen/DeepSeek config 均无 dropout |
| 全参 SFT | 不用（0.0） | 更新所有参数，和预训练一致 | LLaMA 1/2 论文全文未提 dropout；transformers 源码 LlamaConfig/Qwen2Config 默认 attention_dropout=0.0；主流训练框架全参 SFT 不修改此值 |
| LoRA/QLoRA SFT | 用（lora_dropout=0.05~0.1） | 只更新低秩矩阵，可训练参数少，小数据集易过拟合 | peft 库常见配置 |
| DPO/GRPO/PPO | 不用（0.0） | policy gradient 本身方差就大，dropout 再加随机性会训练不稳定；且 ref_model 是 eval 模式（dropout 关闭），policy model 开 dropout 会让 KL 估计有偏 | MiniMind 代码无 dropout 设置 |
| Agent RL | 不用（0.0） | 同上 | 同上 |

MiniMind 代码里 dropout 出现在 3 个位置（Embedding 后、Attention 权重上、Attention 输出），但因为默认 0.0，全部等于空操作——架构上留了口子，需要时可以通过 `config.dropout` 打开。

### Step 4：Weight Tying——首尾共享权重

模型最后一层是 LM Head，作用是把 768 维向量映射回 vocab_size 维的 logits：

```python
lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
# 权重矩阵: [6400, 768] — 和 Embedding 一模一样的形状
# 计算方式: 把输入向量和 W 的每一行做点积，得到 6400 个得分
```

#### 为什么 LM Head 要用 nn.Linear 而不是 nn.Embedding？

两者的权重形状一样都是 `[6400, 768]`，但**能做的操作完全不同**。

用一个 vocab_size=4、hidden_size=3 的迷你例子说明：

```
权重矩阵 W（4 个 token，每个 3 维）:
         dim0   dim1   dim2
token 0: [0.1,   0.2,   0.3]    ← "我"
token 1: [0.8,   0.1,   0.5]    ← "你"
token 2: [0.3,   0.9,   0.2]    ← "他"
token 3: [0.6,   0.4,   0.7]    ← "她"
```

**nn.Embedding 只能做一件事：给 id，取一行**
```
embed(token_id=2) → [0.3, 0.9, 0.2]     # 取第 2 行，得到"他"的向量

给它一个向量？它不认识，只认 id（整数）。
embed([0.5, 0.6, 0.4]) → ❌ 报错，它不接受向量输入
```

**nn.Linear 能做另一件事：给向量，和每一行算点积，返回每个 token 的得分**
```
模型输出的 hidden state:  h = [0.5, 0.6, 0.4]

linear(h) 的计算：h 和 W 的每一行做点积

score_0 = 0.5×0.1 + 0.6×0.2 + 0.4×0.3 = 0.29   ← h 和"我"有多像
score_1 = 0.5×0.8 + 0.6×0.1 + 0.4×0.5 = 0.66   ← h 和"你"有多像
score_2 = 0.5×0.3 + 0.6×0.9 + 0.4×0.2 = 0.77   ← h 和"他"有多像 ✓ 最高
score_3 = 0.5×0.6 + 0.6×0.4 + 0.4×0.7 = 0.82   ← h 和"她"有多像 ✓✓ 最高

输出: logits = [0.29, 0.66, 0.77, 0.82]
→ softmax 后"她"概率最高，预测下一个 token 是"她"
```

**总结：**
| | nn.Embedding | nn.Linear |
|---|---|---|
| 输入 | token id（整数） | 向量（浮点数） |
| 操作 | 按 id 取行（查表） | 向量和每行做点积（匹配） |
| 输出 | 一个向量 [hidden_size] | 每个 token 的得分 [vocab_size] |
| 模型入口用 | ✅ 把 token 变成向量 | ❌ 输入是 id 不是向量 |
| 模型出口用 | ❌ 输入是向量不是 id | ✅ 把向量变成各 token 的得分 |

#### 为什么能共享权重？

nn.Embedding 和 nn.Linear 的权重**存储形状完全相同**，都是 `[6400, 768]`。
第 i 行的含义也一样——都是 **token i 的 768 维向量**，只是用法不同（取行 vs 点积）。

所以直接共享，不需要转置或求逆：

```python
embed_tokens.weight = lm_head.weight  # 共享同一个 [6400, 768] 张量
```

对于 64M 的小模型，省掉 ~490 万参数（占 ~8%），很值得。

> 📊 [可视化图解：nn.Linear 为什么反着存权重](https://claude.ai/code/artifact/3170599b-2d40-48b2-b723-87664a328d21)

#### 业界是否共享？没有统一结论

| 模型 | 参数量 | 是否共享 |
|------|--------|---------|
| GPT-2 | 1.5B | 共享 |
| LLaMA | 7B-70B | 不共享 |
| Gemma | 2B-27B | 共享 |
| MiniMind | 64M | 共享 |

业界对这个问题没有定论。共享在逻辑上完全说得通（正反映射用同一套向量），不共享也是一种工程选择（让两个矩阵各自优化）。不存在"大模型一定不共享"的规律。

## 完整实现

> 📁 代码写在 `scratch/model/model_minimind.py`，这是第一个组件，新建文件。

```python
import torch
from torch import nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size=6400, hidden_size=768, dropout=0.0,
                 tie_weights=True):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            self.embed_tokens.weight = self.lm_head.weight

    def embed(self, input_ids):
        # input_ids: [batch, seq_len] 整数
        # return:    [batch, seq_len, hidden_size] 向量
        return self.dropout(self.embed_tokens(input_ids))

    def to_logits(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_size]
        # return:        [batch, seq_len, vocab_size] logits
        return self.lm_head(hidden_states)


# 验证
model = TokenEmbedding()
ids = torch.randint(0, 6400, (2, 128))      # [2, 128]
h = model.embed(ids)                          # [2, 128, 768]
logits = model.to_logits(h)                   # [2, 128, 6400]

assert h.shape == (2, 128, 768)
assert logits.shape == (2, 128, 6400)
assert model.embed_tokens.weight is model.lm_head.weight  # 确认共享
print(f"Embedding params: {model.embed_tokens.weight.numel():,}")  # 4,915,200
print("Embedding verified!")
```

## 对照 MiniMind 源码

| 我们的实现 | MiniMind 源码 | 位置 |
|-----------|-------------|------|
| `nn.Embedding(6400, 768)` | `self.embed_tokens = nn.Embedding(vocab_size, hidden_size)` | `model_minimind.py:201` |
| `nn.Dropout(0.0)` | `self.dropout = nn.Dropout(config.dropout)` | `model_minimind.py:202` |
| Weight Tying | `self.model.embed_tokens.weight = self.lm_head.weight` | `model_minimind.py:240` |

MiniMind 的 Embedding 实现和我们完全一致——就是这么简单。

## 面试话术

> "模型入口是 nn.Embedding，把 token id 查表映射到 768 维向量。
> 没有绝对位置编码，位置信息完全靠 RoPE 在 Attention 层注入——
> 这样做是因为 RoPE 利用旋转矩阵的正交性在 Q·K 点积中编码相对距离，
> 如果加在 Embedding 层，后续的 Norm/FFN 会破坏这个性质。
> 工程上用了 Weight Tying——Embedding 和 LM Head 共享权重，
> 因为它们是同一个语义空间的正反映射，共享省了近 500 万参数，
> 对 64M 的小模型占比不小。LLaMA、Qwen 大模型则不共享，因为参数充足，分开效果更好。"

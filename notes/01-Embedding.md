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

### Step 1：Embedding 全貌

先看完整的验证代码，然后逐个解释每个组件（Step 2~4）：

```python
import torch
from torch import nn

if __name__ == '__main__':
    vocab_size = 6400    # 词表大小
    hidden_size = 768    # 每个 token 映射成多长的向量

    embed_tokens = nn.Embedding(vocab_size, hidden_size)      # 入口：token id → 向量
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)  # 出口：向量 → logits
    embed_tokens.weight = lm_head.weight                      # Weight Tying（Step 4 详解）
    dropout = nn.Dropout(0.0)                                 # 防过拟合（Step 3 详解）

    input_ids = torch.randint(0, vocab_size, (2, 128))   # [batch=2, seq_len=128]
    hidden_states = dropout(embed_tokens(input_ids))     # [2, 128, 768]
```

**nn.Embedding** 没有任何计算——纯查表。权重矩阵 `[6400, 768]`，`embed_tokens(3)` 就是取第 3 行。每一行是一个 token 的向量表示，初始随机，训练中更新。

**input_ids 从哪来？** 原始文本先经过 Tokenizer（分词器）切分并映射成整数 id：

```
原始文本:  "我喜欢深度学习"
    ↓ Tokenizer（MiniMind 用 BPE 分词，词表大小 6400）
token 序列: ["我", "喜欢", "深度", "学习"]
    ↓ 查词表，每个 token 对应一个 id
input_ids:  [42, 1803, 567, 234]
    ↓ Embedding 查表
hidden_states: [[vec_42], [vec_1803], [vec_567], [vec_234]]   ← [1, 4, 768]
```

Tokenizer 是独立的预处理步骤，不在模型里。这里还没加载分词器，验证时用 `torch.randint` 随机生成 id。

### Step 2：要不要在这里加位置编码？

原始 Transformer 在 Embedding 之后直接加绝对位置编码：
```
原始 Transformer:  Embedding(x) + PositionalEncoding(pos)  → 进 Attention
```

我们不这样做。位置信息用 RoPE 在 Attention 内部注入（详见 [03-RoPE](03-RoPE.md)），好处是：
- RoPE 利用 Q·K 点积的正交性，天然编码**相对距离**而非绝对位置
- 如果加在 Embedding 层，后续的 Norm/FFN 等非线性操作会破坏旋转矩阵的正交性

所以 Embedding 输出的向量**不含显式的位置编码**——位置信息留给 RoPE 在 Attention 内部注入。

```
我们的设计:
  Embedding(x)                    → hidden_states（纯语义，没有位置信息）
      ↓
  Attention 内部:
    Q, K = W_q(x), W_k(x)
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)   ← 位置信息在这里才注入
```

### Step 3：Dropout——为什么 Step 1 里 `nn.Dropout(0.0)` 等于没用？

Step 1 代码中的 `dropout = nn.Dropout(0.0)` 训练时随机把一些维度置零，防过拟合。MiniMind 默认 0.0，等于空操作——架构上留了口子，需要时通过 `config.dropout` 打开。

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

GPT-3、LLaMA、Qwen、DeepSeek 在预训练阶段的公开配置中 dropout 都设为 0.0。常见的考量：

| 因素 | 解释 |
|------|------|
| 数据量足够大 | 预训练数据几百 GB~TB 级，模型较难过拟合，dropout 的正则化收益有限 |
| 训练效率 | dropout 让梯度估计方差变大，可能需要更多步数收敛 |
| 模型容量 | 大模型参数量相对数据量并不过剩，不需要人为限制容量 |
| 其他正则化手段 | 权重衰减（weight decay）、数据增强等提供正则化；RMSNorm 主要作用是稳定优化而非正则化 |

#### 各训练阶段的 dropout 策略

| 阶段 | 是否用 dropout | 原因 | 实锤 |
|------|---------------|------|------|
| 预训练 | 通常 0.0 | 数据量大，过拟合风险低 | LLaMA/Qwen/DeepSeek config 默认 0.0 |
| 全参 SFT | 通常 0.0 | 更新所有参数，和预训练保持一致 | 主流训练框架全参 SFT 不修改此值 |
| LoRA/QLoRA SFT | 常见 0.05~0.1 | 可训练参数少，小数据集容易过拟合 | peft 库常见配置 |
| DPO/GRPO/PPO | 通常 0.0 | policy gradient 本身方差大，dropout 加随机性可能不稳定 | MiniMind 代码无 dropout 设置 |
| Agent RL | 通常 0.0 | 同上 | 同上 |

MiniMind 代码里 dropout 出现在 3 个位置（Embedding 后、Attention 权重上、Attention 输出），但因为默认 0.0，全部等于空操作——架构上留了口子，需要时可以通过 `config.dropout` 打开。

### Step 4：Weight Tying——为什么 Step 1 里 Embedding 和 LM Head 共享权重？

Step 1 代码中 `lm_head = nn.Linear(hidden_size, vocab_size, bias=False)` 是模型出口，把 768 维向量映射回 vocab_size 维的 logits（每个 token 的得分）。紧接着 `embed_tokens.weight = lm_head.weight` 让两者共享同一个 `[6400, 768]` 权重矩阵。


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


#### 为什么能共享权重？

数学视角:  h @ W = [1, 768] @ [768, 6400] = [1, 6400]

PyTorch 实现:  h @ W_stored.T = [1, 768] @ [6400, 768].T = [1, 6400]
               存储形状 [6400, 768]，计算时内部转置

能共享的原因：PyTorch 的 Linear 恰好把权重按行存（每行是一个输出神经元的权重向量），而 Embedding 也是按行存（每行是一个 token 的向量）。两者都是 [6400, 768]，第 i 行都表示 token i 的 768 维向量——只是用法不同（取行 vs 点积）。如果 Linear 按数学惯例存 [768, 6400]，就没法直接共享了。

所以直接共享（Step 1 中的 `embed_tokens.weight = lm_head.weight`），不需要转置或求逆。对于 64M 的小模型，省掉 ~490 万参数（占 ~8%），很值得。

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
>
> Embedding 不单独封装成类——`embed_tokens`、`dropout`、`lm_head` 在真实模型中是主模型类的直接属性（step 8 组装时放进去），这里先用裸组件写验证。

```python
import torch
from torch import nn

# 这三个组件在 step 8 会成为主模型类的属性，这里先裸写验证逻辑
if __name__ == '__main__':
    vocab_size = 6400
    hidden_size = 768

    embed_tokens = nn.Embedding(vocab_size, hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    embed_tokens.weight = lm_head.weight  # Weight Tying

    input_ids = torch.randint(0, vocab_size, (2, 128))  # [batch=2, seq_len=128]
    dropout = nn.Dropout(0.0)
    hidden_states = dropout(embed_tokens(input_ids))

    assert hidden_states.shape == (2, 128, 768)
    assert embed_tokens.weight is lm_head.weight
    print("Embedding + LM Head verified!")
```

**文件组织：`if __name__ == '__main__'`**

验证代码必须放在 `if __name__ == '__main__':` 块中。原因：

- `scratch/model/model_minimind.py` 会被后续章节不断追加新组件（RMSNorm、Attention、FFN...）
- 如果验证代码散落在模块级别，每次 `import` 都会执行——后续章节追加的类还没定义就会报错
- 放在 `__main__` 块里，只有直接运行 `python scratch/model/model_minimind.py` 时才执行验证

后续每章追加新组件时，在 `__main__` 块末尾追加对应的验证代码即可。

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
> 对 64M 的小模型占比不小。LLaMA、Qwen 等大模型选择不共享，具体效果差异需要消融实验判断。"

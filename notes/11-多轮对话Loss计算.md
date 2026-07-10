# 11 - 多轮对话 Loss 计算（SFT Loss Mask）

> 源码位置：`dataset/lm_dataset.py:88-104`（generate_labels）
> 调用位置：`dataset/lm_dataset.py:113`（SFTDataset.__getitem__）
> 对比：`dataset/lm_dataset.py:47-55`（PretrainDataset.__getitem__，全文算 loss）

## 一句话总结

预训练对所有 token 算 loss，SFT 只对 assistant 回复部分算 loss（用 -100 mask 掉其余部分），多轮对话就是把每轮 assistant 的回复段都开 label。

## 预训练 vs SFT 的 Loss 区别

### 预训练：全文 loss

```python
# dataset/lm_dataset.py:53-54 (PretrainDataset)
labels = input_ids.clone()                            # 全部参与 loss
labels[input_ids == self.tokenizer.pad_token_id] = -100  # 只 mask padding
```

数据示例：
```
input_ids:  [BOS] 中 国 的 首 都 是 北 京 [EOS] [PAD] [PAD]
labels:     [BOS] 中 国 的 首 都 是 北 京 [EOS] -100  -100
```

每个 token 都预测下一个 token → 学语言建模能力。

### SFT：只对 assistant 回复算 loss

```python
# dataset/lm_dataset.py:88-104 (SFTDataset)
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)     # 默认全部 mask
    i = 0
    while i < len(input_ids):
        # 找 <|im_start|>assistant\n 标记
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)    # 标记之后开始算 loss
            end = start
            # 找 <|im_end|>\n 结束标记
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # 只有 assistant 回复 + eos 参与 loss
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id)
        else:
            i += 1
    return labels
```

## 多轮对话的数据流（用具体例子追踪）

### 原始数据（sft_t2t_mini.jsonl）

```json
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    {"role": "user", "content": "介绍一下北京"},
    {"role": "assistant", "content": "北京是中国的首都。"}
  ]
}
```

### Chat Template 应用后（create_chat_prompt）

```
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
<think>
...
</think>

你好！有什么可以帮助你的吗？<|im_end|>
<|im_start|>user
介绍一下北京<|im_end|>
<|im_start|>assistant
<think>
...
</think>

北京是中国的首都。<|im_end|>
```

### Tokenize 后的 label mask

```
token:  <|im_start|> user \n 你 好 <|im_end|> \n
label:  -100         -100    -100 -100 -100      -100  ← user 部分全 mask

token:  <|im_start|> assistant \n 你好 ！ 有 什么 ... <|im_end|> \n
label:  -100         -100          你好 ！ 有 什么 ... <|im_end|> \n  ← assistant 回复参与 loss

token:  <|im_start|> user \n 介绍 一下 北京 <|im_end|> \n
label:  -100         -100    -100 -100 -100    -100      -100  ← 第 2 轮 user 也 mask

token:  <|im_start|> assistant \n 北京 是 中国 的 首都 。 <|im_end|> \n
label:  -100         -100          北京 是 中国 的 首都 。 <|im_end|> \n  ← 第 2 轮 assistant 也参与 loss
```

关键机制：
- `self.bos_id` = tokenize("<|im_start|>assistant\n") → 用来定位 assistant 回复的开始
- `self.eos_id` = tokenize("<|im_end|>\n") → 用来定位回复的结束
- **while 循环**会在整个序列中**逐段查找**每一轮的 assistant 回复，所以天然支持多轮

## 为什么只对 assistant 算 loss

1. **user 输入是已知的**：模型不需要"学习如何当用户"，只需学"如何回复"
2. **避免学偏**：如果对 user prompt 也算 loss，模型会分配容量去记 prompt 模板（`<|im_start|>user\n` 等），浪费
3. **和推理对齐**：推理时模型只生成 assistant 部分，训练也只优化这部分
4. **system prompt 更不该算 loss**：system 是每次都重复的固定文本

## cross_entropy 的 ignore_index=-100

```python
# model/model_minimind.py:252
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

PyTorch 的 cross_entropy 内置了 `ignore_index` 参数：
- $\text{label} = -100$ 的位置 → loss 贡献为 0，梯度也为 0
- $\text{label} \neq -100$ 的位置 → 正常计算 cross_entropy
- 最终 $\text{loss} = \dfrac{\sum_{\text{valid}} \text{CE}_i}{\, N_{\text{valid}} \,}$（mean reduction）

## 和 DPO 的 loss mask 对比

```python
# dataset/lm_dataset.py:176-192 (DPODataset)
def generate_loss_mask(self, input_ids):
    loss_mask = [0] * len(input_ids)    # 用 0/1 mask 而不是 -100
    # 同样只对 assistant 回复段标 1
    ...
    return loss_mask
```

DPO 用 0/1 的 loss_mask 而不是 -100，因为 DPO 的 loss 不用 cross_entropy 的 ignore_index，
而是手动把 log_probs 乘以 mask 来实现选择性计算。

## 面试话术

> SFT 和预训练最关键的区别是 loss 的计算范围。预训练对所有 token 算 next-token prediction loss，
> 而 SFT 只对 assistant 回复部分算 loss，user 输入和 system prompt 用 -100 mask 掉。
> 在 MiniMind 里，SFTDataset 的 generate_labels 方法通过在 token 序列里查找
> `<|im_start|>assistant` 和 `<|im_end|>` 标记来定位每一轮 assistant 的回复段，
> 用 while 循环逐段扫描，天然支持多轮对话——每轮 assistant 回复都参与 loss。
> 最终 label 传给 cross_entropy 的 ignore_index=-100，PyTorch 自动跳过 masked 位置。
> 这样做的好处是让模型专注学"如何回复"而不是"如何当用户"，和推理时的生成对齐。

## 业界对比

| 模型 | SFT loss mask 策略 | 特殊处理 |
|------|-------------------|---------|
| MiniMind | 只对 assistant 回复算 loss | 标准做法 |
| Qwen2.5 | 同上 | 支持 packing（多个样本拼一个序列，用 attention_mask 隔离） |
| LLaMA-3 | 同上 | 用 sample packing + document_attention_mask |
| DeepSeek | 同上 | 多轮 tool_call 也参与 loss |
| InternLM | 同上 | 支持 sequence_parallel 下的分段 loss |

进阶：**Packing**（把多个短样本拼成一个长序列）是业界优化 SFT 效率的标准做法，
MiniMind 没有实现 packing，每个样本独立 padding，会浪费一些计算。

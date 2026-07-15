# 23 - Gradient Checkpointing（梯度检查点 / 激活重计算）

> MiniMind 没有实现此功能
> 本文补充面试知识空白

## 一句话总结

前向时只保存部分层的激活值（checkpoint），反向时从 checkpoint 重新前向计算中间激活值。用约 33% 额外计算换取 $\sqrt{N}$ 倍的激活显存节省。

## 为什么需要 Gradient Checkpointing

训练时的显存杀手是**激活值**（intermediate activations）：

```
MiniMind: 8 层，每层保存的激活值:
- Attention 的 Q, K, V 投影结果: [batch, seq_len, heads, head_dim]
- Attention 的 softmax 输出: [batch, heads, seq_len, seq_len]  ← 最大
- FFN 的 gate/up 投影: [batch, seq_len, intermediate_size]
- 各处残差连接的输入

8 层 × 每层多个激活 → 显存占用远超模型参数本身
```

对大模型更严重：LLaMA-70B 有 80 层，激活值占总显存 60%+ 。

## 标准训练 vs Gradient Checkpointing

### 标准训练：存所有激活

```
前向: Layer1 → a1 → Layer2 → a2 → ... → Layer8 → a8 → loss
                ↑        ↑                  ↑
              存 a1    存 a2              存 a8   （全部保留）

反向: 用 a8 算 Layer8 梯度 → 用 a7 算 Layer7 梯度 → ... → 用 a1 算 Layer1 梯度
```
显存：$O(N)$，$N$ 是层数。

### Gradient Checkpointing：只存部分

```
标记 Layer1, Layer4, Layer8 为 checkpoint

前向: Layer1 → a1✓ → Layer2 → ✗ → Layer3 → ✗ → Layer4 → a4✓ → ... → a8✓ → loss
               保存              丢弃            丢弃              保存

反向到 Layer3 时:
  a4 已存 → 从 a1 重新前向计算 Layer2, Layer3 得到 a2, a3 → 用 a3 算梯度 → 丢弃
```
显存：$O(\sqrt{N})$（每 $\sqrt{N}$ 层存一个 checkpoint）。

## PyTorch 实现

```python
from torch.utils.checkpoint import checkpoint

class MiniMindBlock(nn.Module):
    def forward(self, hidden_states, ...):
        # 标准写法（MiniMind 当前用的）
        hidden_states, present = self.self_attn(self.input_layernorm(hidden_states), ...)
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present

# 如果要加 gradient checkpointing，在 MiniMindModel.forward 中：
for layer in self.layers:
    if self.gradient_checkpointing and self.training:
        hidden_states = torch.utils.checkpoint.checkpoint(
            layer, hidden_states, position_embeddings,
            use_reentrant=False
        )
    else:
        hidden_states, present = layer(hidden_states, position_embeddings)
```

HuggingFace Transformers 的做法：
```python
model.gradient_checkpointing_enable()  # 一行开启
```

## 代价分析

| 指标 | 标准训练 | Gradient Checkpointing |
|------|---------|----------------------|
| 激活显存 | $O(N)$ | $O(\sqrt{N})$ |
| 计算量 | 1x | ~1.33x（重算一次前向） |
| 训练速度 | 基准 | 慢 20-30% |
| 可训练模型大小 | 基准 | 大 2-3x |

为什么是 33%？
- 前向 1 次 + 反向 1 次 $= 2$ 次（标准）
- 前向 1 次 + 反向时重算前向约 1 次 + 反向 1 次 $\approx 3$ 次
- 额外 $\frac{1}{3} \approx 33\%$

实际上没有完整重算一遍前向（只重算 checkpoint 之间的段），所以通常 20-30%。

## 和其他显存优化技术的关系

$$\text{显存占用} = \text{模型参数} + \text{优化器状态} + \text{梯度} + \text{激活值}$$

各技术的优化目标：

| 技术 | 优化对象 | 方式 |
|------|---------|------|
| 混合精度训练 | 参数/梯度/激活 | 减少每个元素大小（32bit → 16bit） |
| 梯度累积 | 激活值 | 小 batch → 小激活 |
| Gradient Checkpointing | 激活值 | 不存中间层 |
| ZeRO | 参数/优化器状态/梯度 | 跨卡切分 |
| Offloading | 优化器状态/参数 | 卸载到 CPU RAM |

这些技术通常**组合使用**：
```
DeepSpeed ZeRO-3 + Gradient Checkpointing + BF16 → 能在 8×A100 上训 70B 模型
```

## 面试话术

> Gradient Checkpointing 是训练大模型时减少激活值显存的标准技术。
> 核心思想是前向时只保存部分层的激活值作为 checkpoint，反向时从最近的 checkpoint 重新前向计算得到中间激活。
> 如果每 $\sqrt{N}$ 层存一个 checkpoint，激活显存从 $O(N)$ 降到 $O(\sqrt{N})$，
> 代价是额外约 33% 的计算量（因为部分前向要算两遍）。
> MiniMind 没有使用这个技术，因为 64M 参数的小模型激活值不大。
> 但 LLaMA-70B 这种 80 层的模型几乎必须开 gradient checkpointing，
> 通常和 ZeRO、混合精度组合使用。

## 业界使用

| 模型 | 是否使用 | 备注 |
|------|---------|------|
| MiniMind (64M) | 不使用 | 模型太小，不需要 |
| LLaMA-3 (8B+) | 使用 | 所有超过 1B 的模型都开 |
| DeepSeek-V3 | 使用 | 配合 ZeRO-3 |
| Qwen2.5 | 使用 | Megatron-LM 内置支持 |
| GPT-4 | 推测使用 | 大模型标配 |

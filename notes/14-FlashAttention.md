# 14 - Flash Attention

> 源码位置：`model/model_minimind.py:109, 125-126`（自动切换 Flash Attention）
> MiniMind 调用了 Flash Attention 但没有手写实现

## 一句话总结

标准 Attention 的瓶颈不是计算而是显存读写（IO bound），Flash Attention 通过分块计算（tiling）避免存储完整的 N×N 注意力矩阵，把 IO 复杂度从 O(N²) 降到 O(N²/M)（M 是 SRAM 大小）。

## MiniMind 中的使用

```python
# model/model_minimind.py:109
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

# model/model_minimind.py:125-126
if self.flash and ...:
    output = F.scaled_dot_product_attention(xq, xk, xv,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=self.is_causal)
else:
    # 手写的标准 attention
    scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
    ...
```

PyTorch 2.0+ 的 `scaled_dot_product_attention` 会自动调用 Flash Attention kernel（如果硬件支持）。

## 标准 Attention 的显存问题

以 MiniMind 的参数为例：seq_len=768, n_heads=8

```
标准做法:
1. S = Q × K^T           → [batch, 8, 768, 768]    ← 存储 N² 大小的矩阵
2. P = softmax(S)         → [batch, 8, 768, 768]    ← 又一个 N² 矩阵
3. O = P × V              → [batch, 8, 768, 96]

显存: 需要把 S 和 P 完整存在 GPU HBM 中
长序列时: seq_len=4096 → S 矩阵 = 4096² × 8 heads × 4 bytes ≈ 2 GB
         seq_len=32768 → S 矩阵 ≈ 128 GB → 完全不可行
```

## GPU 存储层级（理解瓶颈的关键）

```
SRAM（片上缓存）:  ~20 MB    读写速度 ~19 TB/s    ← 极快但极小
HBM（显存）:       ~80 GB    读写速度 ~3 TB/s     ← 大但慢
主存（CPU RAM）:    ~1 TB     读写速度 ~100 GB/s   ← 更大更慢

标准 Attention: S 和 P 矩阵存在 HBM → 反复读写 HBM → IO 瓶颈
Flash Attention: 分块在 SRAM 中计算 → 只写最终结果到 HBM → 减少 IO
```

## Flash Attention 的核心思想

### Tiling（分块计算）

不一次算完整的 N×N 矩阵，而是按块计算：

```
把 Q 分成 Tr 个块: Q₁, Q₂, ..., Q_Tr    每块 [B_r, d]
把 K,V 分成 Tc 个块: K₁,V₁, K₂,V₂, ...  每块 [B_c, d]

对每个 Q 块:
  for 每个 K,V 块:
    1. 从 HBM 读 Q_i, K_j, V_j 到 SRAM      ← 块小，放得下
    2. 在 SRAM 中算 S_ij = Q_i × K_j^T        ← 只算一个小块
    3. 在 SRAM 中算局部 softmax 和 O_ij       ← 用 online softmax
    4. 增量更新输出 O_i                        ← 累加而非一次算完
  把最终 O_i 写回 HBM                          ← 只写结果
```

### Online Softmax（在线 softmax）

标准 softmax 需要先扫描所有元素找 max，再扫描一遍算 exp，需要完整的行。

Online softmax 增量计算：处理每个新块时，用一个 running max 和 running sum 来修正之前块的结果。

处理第 1 块后得到 $m_1$（running max）、$s_1$（running sum）和 $O_1$（部分输出）。处理第 2 块时增量更新：

- $m_2 = \max(m_1,\; m_2^{\text{local}})$ — 更新全局 max
- $\alpha = \exp(m_1 - m_2)$ — 修正系数，校正之前块在旧 max 下的结果
- $O = \alpha \cdot O_1 + O_2^{\text{new}}$
- $s = \alpha \cdot s_1 + s_2^{\text{local}}$

数学上完全等价于标准 softmax，但不需要存储完整的 S 矩阵。

## 复杂度对比

| | 标准 Attention | Flash Attention |
|--|---------------|-----------------|
| 计算量（FLOPs） | O(N²d) | O(N²d)（相同） |
| HBM IO | O(N²) | O(N²d/M)（M=SRAM 大小） |
| 显存占用 | O(N²)（存 S 和 P） | O(N)（不存完整矩阵） |
| 实际速度 | 基准 | 2-4x（因为 IO 减少） |

关键洞察：**计算量没变，但 IO 大幅减少**。GPU 的 FLOPS 远高于带宽，
标准 Attention 是 memory-bound，Flash Attention 把它变得更接近 compute-bound。

## 反向传播：重计算而非存储

标准 Attention 反向传播需要 S 和 P 矩阵。

Flash Attention 的做法：反向时**重新计算** S 和 P（从 Q, K, V 重新算），
而不是前向时存下来。这叫 **recomputation**，用额外计算换显存。

代价：反向的计算量增加 ~25%
收益：显存从 O(N²) 降到 O(N)

## 面试话术

> Flash Attention 的核心洞察是：标准 Attention 的瓶颈不是计算量而是显存读写。
> N×N 的注意力矩阵要在 HBM 和 SRAM 之间反复搬运，这才是慢的原因。
> Flash Attention 用 tiling 把 Q/K/V 分成小块，在 SRAM 里算完每个小块的注意力，
> 配合 online softmax 做增量更新，不需要存储完整的 N² 矩阵。
> 计算量（FLOPs）不变，但 HBM IO 从 O(N²) 降到 O(N²d/M)，显存从 O(N²) 降到 O(N)。
> 反向传播时用 recomputation 重算 S 和 P，额外增加约 25% 计算换取巨大的显存节省。
> 在 MiniMind 里通过 PyTorch 的 scaled_dot_product_attention 自动启用，
> 当硬件不支持或需要手动控制 mask 时会 fallback 到手写的标准 attention。

## 业界对比

| 版本 | 核心改进 |
|------|---------|
| Flash Attention v1 | 基础 tiling + online softmax |
| Flash Attention v2 | 优化并行度（在 seq_len 维度也并行），减少非矩阵乘的 FLOPs |
| Flash Attention v3 | 利用 H100 的异步特性（warp specialization），FP8 支持 |
| PagedAttention (vLLM) | 推理时的 KV Cache 内存管理，类似 OS 虚拟内存的分页机制 |

| 模型 | Flash Attention 使用 |
|------|---------------------|
| MiniMind | PyTorch SDPA 自动调用 |
| LLaMA-3 | Flash Attention v2 |
| DeepSeek-V3 | 自定义 Flash Attention（适配 MLA 的低秩 KV） |
| Qwen2.5 | Flash Attention v2 + 滑动窗口变体 |

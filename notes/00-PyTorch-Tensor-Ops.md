# 00 - PyTorch 张量操作速查（MiniMind 用到的）

> 本章是补充材料，覆盖 MiniMind 代码中出现的所有张量操作。
> 每个操作都用小例子演示，标注在模型哪个位置用到。

## 形状变换类

### view / reshape —— 改变形状，不改数据

```python
x = torch.arange(12)          # [12]     → [0, 1, 2, ..., 11]
x.view(3, 4)                  # [3, 4]   → 3 行 4 列
x.view(2, 2, 3)               # [2,2,3]  → 两个 2×3 矩阵
x.reshape(4, 3)               # [4, 3]   → 和 view 功能一样
```

`view` 要求内存连续（contiguous），`reshape` 不要求（不连续时会自动复制一份连续内存再改形状）。

**什么是"内存连续"？** 张量在内存里是一段连续的数字，按行铺平存储：

```
逻辑上:                内存里:
[[1, 2, 3],            [1, 2, 3, 4, 5, 6]   ← 按行连续排列
 [4, 5, 6]]
```

`view` 就是换一种步长来读同一段内存，零开销。但 `transpose` 之后内存布局没变、只改了读取方式，逻辑上相邻的元素在内存里不再相邻——这就是"不连续"：

```
transpose(0,1) 后:
逻辑上第一行是 [1, 4]，但内存里 1 和 4 之间隔着 2, 3 → 不连续
```

这时 `view` 会报错（`RuntimeError: view requires contiguous tensor`），有三种解法：

```python
x.transpose(1, 2).contiguous().view(...)   # 手动复制连续内存再 view（MiniMind 用这种）
x.transpose(1, 2).reshape(...)             # reshape 内部自动做 contiguous + view
x.transpose(1, 2).contiguous().reshape(...)# 可以但多余
```

日常不确定时用 `reshape` 更安全，但如果明确知道内存连续，`view` 更快（零开销）。

**MiniMind 中用到**：Attention 里把 `[B, S, 768]` 拆成 `[B, S, 8, 96]`（拆头）

```python
xq = self.wq(x).view(B, S, 8, 96)   # 768 = 8 × 96
```

### unsqueeze / squeeze —— 加一维 / 去一维

```python
x = torch.tensor([1, 2, 3])   # [3]

x.unsqueeze(0)                # [1, 3]   → 在第 0 维前面加一个 1
x.unsqueeze(1)                # [3, 1]   → 在第 1 维位置加一个 1
x[None, :]                    # [1, 3]   → None 等价于 unsqueeze
x[:, None]                    # [3, 1]   → 同上

y = torch.zeros(1, 3, 1)      # [1, 3, 1]
y.squeeze()                    # [3]      → 去掉所有长度为 1 的维度
y.squeeze(0)                   # [3, 1]   → 只去掉第 0 维的 1
```

**为什么要加一个 1？** 通常是为了让后续的 `expand` 或广播（broadcasting）能工作。

**MiniMind 中用到**：
- `repeat_kv` 里 `x[:, :, :, None, :]` 插入空位给 expand 用
- RoPE 里 `cos.unsqueeze(1)` 让 `[S, 96]` 变成 `[S, 1, 96]` 广播到所有头

### transpose —— 交换两个维度

```python
x = torch.zeros(2, 3, 4)      # [2, 3, 4]
x.transpose(1, 2)             # [2, 4, 3]  → 交换第 1 维和第 2 维
```

**MiniMind 中用到**：Attention 里把头维度提前

```python
# 交换 S 和 n_heads，让每个头独立做矩阵乘
xq = xq.transpose(1, 2)       # [B, S, 8, 96] → [B, 8, S, 96]

# 做点积时交换最后两维
scores = xq @ xk.transpose(-2, -1)  # [B, 8, S, 96] @ [B, 8, 96, S] = [B, 8, S, S]
```

### permute —— 任意重排维度（transpose 的通用版）

```python
x = torch.zeros(2, 3, 4, 5)   # [2, 3, 4, 5]
x.permute(0, 2, 1, 3)         # [2, 4, 3, 5]  → 第 1 维和第 2 维互换

# transpose 一次只能换两个维度，permute 可以一次重排所有维度
```

### contiguous —— 让内存连续

```python
x = torch.zeros(2, 3, 4)
y = x.transpose(1, 2)         # [2, 4, 3] → 内存布局还是 [2, 3, 4] 的顺序
y.is_contiguous()              # False
y = y.contiguous()             # 重新排列内存，让它真的是 [2, 4, 3] 的顺序
y.is_contiguous()              # True
```

**什么时候需要？** `view` 要求内存连续，transpose 后直接 view 会报错。解决方法：

```python
x.transpose(1, 2).contiguous().view(...)   # 先 contiguous 再 view
x.transpose(1, 2).reshape(...)             # 或者直接用 reshape，它自动处理
```

**MiniMind 中用到**：loss 计算里 `logits[..., :-1, :].contiguous()`

---

## 复制 / 扩展类

### expand —— 虚拟复制（不分配新内存）

```python
x = torch.tensor([[1, 2, 3]])  # [1, 3]
x.expand(4, 3)                 # [4, 3]  → "复制"成 4 行，但内存里还是 1 行

# 只能扩展长度为 1 的维度，其他维度传 -1 表示不变
x = torch.zeros(2, 1, 3)
x.expand(2, 5, 3)              # [2, 5, 3]
x.expand(-1, 5, -1)            # [2, 5, 3]  → -1 = 保持原来的大小
```

**关键**：expand 不复制数据，多个位置指向同一块内存，所以是零开销。但不能对 expand 后的结果做原地修改（会影响"其他副本"）。

**MiniMind 中用到**：`repeat_kv` 把 4 个 KV 头 expand 成 8 个

### repeat —— 真复制（分配新内存）

```python
x = torch.tensor([1, 2, 3])    # [3]
x.repeat(4)                    # [12]    → [1,2,3, 1,2,3, 1,2,3, 1,2,3]

x = torch.zeros(2, 3)          # [2, 3]
x.repeat(3, 2)                 # [6, 6]  → 行重复 3 次，列重复 2 次
```

`repeat` 真的复制了数据，比 `expand` 慢且占内存。能用 `expand` 就不用 `repeat`。

**MiniMind 中用到**：generate 里 `input_ids.repeat(num_return_sequences, 1)` 多路生成

### clone —— 深拷贝

```python
y = x.clone()                  # 复制数据 + 保留梯度计算图
y = x.detach().clone()         # 复制数据 + 断开梯度
```

---

## 拼接 / 切分类

### cat —— 沿已有维度拼接

```python
a = torch.zeros(2, 3)          # [2, 3]
b = torch.zeros(2, 5)          # [2, 5]
torch.cat([a, b], dim=1)       # [2, 8]  → 沿 dim=1（列）拼

c = torch.zeros(4, 3)          # [4, 3]
torch.cat([a, c], dim=0)       # [6, 3]  → 沿 dim=0（行）拼
```

**MiniMind 中用到**：
- KV Cache 拼接：`torch.cat([past_K, new_K], dim=1)` 沿 seq 维拼历史和新 token
- RoPE 里 `torch.cat([cos, cos], dim=-1)` 拼成 head_dim 长度
- `rotate_half` 里拼接前后半段

### stack —— 沿新维度堆叠

```python
a = torch.zeros(3)             # [3]
b = torch.zeros(3)             # [3]
torch.stack([a, b], dim=0)     # [2, 3]  → 新增一个维度
torch.stack([a, b], dim=1)     # [3, 2]
```

`cat` 是在已有维度上拼长，`stack` 是新加一个维度。

### 切片 / 索引

```python
x = torch.arange(20).view(4, 5)

x[0]                           # 第 0 行
x[:, 2]                        # 第 2 列
x[1:3]                         # 第 1~2 行
x[:, -1]                       # 最后一列
x[..., :-1]                    # ... 代表"前面所有维度"，这里取除最后一列外的所有列
```

**MiniMind 中用到**：
- Loss 错位：`logits[..., :-1, :]` 和 `labels[..., 1:]`
- RoPE 切片：`freqs_cos[start_pos : start_pos + seq_len]`
- `rotate_half`：`x[..., :half]` 和 `x[..., half:]`

---

## 数学运算类

### @ —— 矩阵乘法

```python
a = torch.zeros(2, 3)          # [2, 3]
b = torch.zeros(3, 4)          # [3, 4]
a @ b                          # [2, 4]

# 高维时，前面的维度当 batch，最后两维做矩阵乘
a = torch.zeros(8, 128, 96)    # [8, 128, 96]
b = torch.zeros(8, 96, 128)    # [8, 96, 128]
a @ b                          # [8, 128, 128]  → 8 个独立的矩阵乘
```

**MiniMind 中用到**：`scores = xq @ xk.transpose(-2, -1)` 注意力打分

### * —— 逐元素相乘（不是矩阵乘）

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
a * b                          # [4, 10, 18]  → 对应位置一个一个乘
```

**MiniMind 中用到**：SwiGLU 的 `F.silu(gate) * up`，RoPE 的 `q * cos`

### 广播（Broadcasting）

形状不完全一样时，PyTorch 会自动"广播"较小的张量：

```python
x = torch.zeros(2, 128, 768)   # [2, 128, 768]
w = torch.ones(768)             # [768]
x * w                           # [2, 128, 768]  → w 自动扩展到每个位置

# 规则：从右往左对齐，维度为 1 的可以广播
x = torch.zeros(2, 128, 8, 96) # [2, 128, 8, 96]
c = torch.zeros(128, 1, 96)    # [   128, 1, 96]
x * c                           # [2, 128, 8, 96]  → c 的第 2 维(1)广播到 8
```

**MiniMind 中用到**：RoPE 的 cos `[S, 1, 96]` 广播到 Q 的 `[S, 8, 96]`

---

## 其他常用

### torch.full / torch.zeros_like / torch.ones

```python
torch.full((3, 3), -float('inf'))    # 3×3 全是 -inf（用于 causal mask）
torch.zeros_like(x)                   # 和 x 同形状的全零张量
torch.ones(5)                         # [1, 1, 1, 1, 1]
```

### torch.triu —— 上三角

```python
torch.ones(4, 4).triu(1)      # 对角线以上为 1，其余为 0（diagonal=1 跳过对角线）
# [[0, 1, 1, 1],
#  [0, 0, 1, 1],
#  [0, 0, 0, 1],
#  [0, 0, 0, 0]]
```

**MiniMind 中用到**：causal mask `torch.full((S, S), -inf).triu(1)` 遮住未来 token

### F.softmax / F.silu / F.cross_entropy

```python
F.softmax(x, dim=-1)           # 沿最后一维归一化成概率
F.silu(x)                      # x * sigmoid(x)，SwiGLU 的激活函数
F.cross_entropy(logits, labels) # 交叉熵 loss，内部自带 softmax
```

### torch.topk

```python
x = torch.tensor([0.1, 0.5, 0.3, 0.8])
values, indices = torch.topk(x, k=2)
# values:  [0.8, 0.5]    ← 最大的 2 个值
# indices: [3, 1]         ← 它们的位置
```

**MiniMind 中用到**：MoE Router 选 top-K 专家；generate 的 Top-K 采样

### F.one_hot

```python
idx = torch.tensor([0, 2, 1])
F.one_hot(idx, num_classes=4)
# [[1, 0, 0, 0],
#  [0, 0, 1, 0],
#  [0, 1, 0, 0]]
```

**MiniMind 中用到**：MoE 辅助损失计算每个专家被选中的频率

### nonzero / any —— 找 True 的位置

```python
mask = torch.tensor([False, True, True, False, True])

mask.nonzero()            # [[1], [2], [4]]  ← True 的位置（二维索引）
mask.nonzero().flatten()  # [1, 2, 4]         ← 压成一维，方便做索引

# any：沿某个维度检查"是否有 True"
mask_2d = torch.tensor([[False, True],
                         [False, False],
                         [True,  False]])
mask_2d.any(dim=-1)       # [True, False, True]  ← 每行是否有 True
```

**MiniMind 中用到**：MoE 分发时 `mask.any(dim=-1).nonzero().flatten()` 找出被路由到某个专家的 token 索引

### index_add_ —— 按索引累加

```python
y = torch.zeros(5, 3)                    # [5, 3] 目标张量
idx = torch.tensor([1, 4, 1])            # 3 个源行要加到 y 的哪些行
src = torch.ones(3, 3)                   # [3, 3] 源张量

y.index_add_(0, idx, src)
# src 的第 0 行 加到 y 的第 1 行
# src 的第 1 行 加到 y 的第 4 行
# src 的第 2 行 加到 y 的第 1 行（和第 0 行累加，不是覆盖）
```

和直接赋值 `y[idx] = src` 的区别：赋值会**覆盖**，`index_add_` 会**累加**。MoE top-K > 1 时同一个 token 被多个专家处理，需要累加而非覆盖。

`_` 后缀表示原地操作（in-place），直接修改 y，不创建新张量。

**MiniMind 中用到**：MoE 把每个专家的输出累加回原位置

### tensor.new_zeros —— 在相同设备上创建零张量

```python
x = torch.tensor([1.0], device='cuda')
x.new_zeros(3)       # tensor([0., 0., 0.], device='cuda')
# 自动继承 x 的 device 和 dtype，不需要手动指定
```

和 `torch.zeros(3)` 的区别：`torch.zeros` 默认在 CPU 上，如果 x 在 GPU 上就设备不匹配。`x.new_zeros` 保证和 x 在同一设备。

**MiniMind 中用到**：MoE 推理时 `scores.new_zeros(1).squeeze()` 创建零标量作为 aux_loss

---

## 速查表：MiniMind 代码位置

| 操作 | 用在哪 |
|------|--------|
| `view` / `reshape` | Attention 拆头、合并头 |
| `unsqueeze` / `None` | repeat_kv 插空位、RoPE cos/sin 广播 |
| `transpose` | Attention 把头维度提前、Q·K 点积 |
| `contiguous` | loss 计算的 logits 切片后 |
| `expand` | repeat_kv 虚拟复制 KV 头 |
| `cat` | KV Cache 拼接、RoPE cos/sin 拼接、rotate_half |
| `@` | Q·K 点积、softmax(scores) @ V |
| `*` | SwiGLU gate×up、RoPE q×cos |
| `triu` | causal mask 上三角 |
| `topk` | MoE 路由、Top-K 采样 |
| `one_hot` | MoE 辅助损失 |
| `nonzero` / `any` | MoE 找被路由到某专家的 token 索引 |
| `index_add_` | MoE 把专家输出累加回原位置 |
| `new_zeros` | MoE 在相同设备上创建零张量 |

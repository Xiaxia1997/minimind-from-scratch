# 06 - MOE FeedForward：参数和计算的解耦

## 先算账：FFN 才是参数大户

一层 Transformer Block 由 Attention + FFN 组成，参数花在哪？

**Attention：4 个线性投影（wq/wk/wv/wo），全部 bias=False**

```
wq:  [768, 8×96]  = [768, 768]  →  768 × 768 = 589,824
wk:  [768, 4×96]  = [768, 384]  →  768 × 384 = 294,912    ← GQA 只有一半
wv:  [768, 4×96]  = [768, 384]  →  768 × 384 = 294,912
wo:  [768, 768]                 →  768 × 768 = 589,824
────────────────────────────────────────────────────────
Attention 合计:                    1,769,472 ≈ 1.77M
```

wk/wv 比 wq 小一半，是因为 GQA 用 4 个 KV 头而非 8 个（第 4 章讲过）。

**FFN：3 个线性投影（gate/up/down_proj），全部 bias=False**

```
gate_proj:  [768, 2432]  →  768 × 2432 = 1,867,776
up_proj:    [768, 2432]  →  768 × 2432 = 1,867,776
down_proj:  [2432, 768]  →  2432 × 768 = 1,867,776
──────────────────────────────────────────────────
FFN 合计:                   5,603,328 ≈ 5.60M
```

SwiGLU 比原始 FFN 多一个投影（3 个 vs 2 个），但 intermediate_size 相应缩小（第 5 章讲过）。

**汇总**

```
Attention:  1.77M  →  24%
FFN:        5.60M  →  76%
──────────────────────────
单层合计:   7.37M  → 100%

全模型: 8 层 × 7.37M + Embedding 6400×768 = 4.92M ≈ 64M（MiniMind-64M 名字的来源）
```

**FFN 占了单层 76% 的参数。** 想优化 Transformer 的参数效率，刀要砍在 FFN 上。

## Dense FFN 的浪费

上一章讲了 FFN 里 gate_proj 和 up_proj 都是 `[768, 2432]` 的矩阵。换个角度理解——这 2432 个维度就是 2432 个训练出来的"知识模式"。gate_proj 负责探测"该不该激活"（SiLU 输出大 = 匹配上了，接近 0 = 没匹配上），up_proj 负责"激活后输出什么内容"，两者逐元素相乘 = gate 开门，up 决定门后面是什么。

打个比方：2432 个模式里有的擅长"代码语法"，有的擅长"法律术语"，有的擅长"古诗格律"。token `def` 进来，"代码语法"那几个模式激活了，剩下大部分输出约等于零——**算了，但白算。**

Dense 模型的死结在这里：**每个 token 不管是什么内容，必须过全部 2432 个模式。** 想让模型更聪明？加模式。但多一个模式 = 每个 token 多一份计算，参数量和计算量完全绑定。

## MoE：把参数和计算解耦

MoE（Mixture of Experts，混合专家）的核心洞察：

**不要让每个 token 过所有参数。把 FFN 拆成多个小 FFN（专家），每个 token 只路由到其中 1~2 个最相关的专家。**

```
Dense FFN:
  1 个 FFN，2432 个模式，每个 token 全部计算
  参数 5.6M，每个 token 的计算量 5.6M

MoE FFN（MiniMind 配置：4 个专家，top-1）:
  4 个独立的 FFN（每个 2432 个模式），每个 token 只选 1 个
  参数 22.4M（4 × 5.6M），每个 token 的计算量 5.6M（只算 1 个专家）
```

**参数 ×4，计算 ×1。** 模型的知识容量大了 4 倍，但每个 token 的推理速度不变。

### 为什么只对 FFN 做 MoE，不对 Attention 做？

面试高频问题。

**FFN 存"知识"**。它的权重矩阵里存的是各种模式——2432 行就是 2432 种知识模式。不同内容的 token 需要不同的知识模式，天然适合"分科"：Expert 0 擅长代码，Expert 1 擅长自然语言，等等。

**Attention 做"路由"**。它决定 token 之间怎么互相看——谁和谁交互。这是 token 间的成对计算，按 token 独立分发很难做。主流方案都只对 FFN 做 MoE。

还有一个实际原因：**FFN 占 76% 参数**，MoE 化 FFN 的杠杆远大于 Attention。

> DeepSeek-V2/V3 用 MLA（Multi-head Latent Attention）对 KV 做低秩压缩，目标是省 KV Cache 显存，不是 MoE 路线，思路完全不同。

## MoE 的架构

```
输入 x: [B, S, 768]
          │
          ▼
    ┌─── Router（门控网络）───┐
    │  x @ W_gate → [B×S, 4]  │   每个 token 对 4 个专家打分
    │  softmax → 概率分布      │
    │  top-1 → 选中专家 + 权重 │
    └──────────────────────────┘
          │
          ▼ 按路由结果分发
    ┌─────┼─────┬─────┐
  Expert₀ Expert₁ Expert₂ Expert₃    4 个独立的 SwiGLU FFN
    └─────┼─────┴─────┘
          │
          ▼ 加权合并
    输出 y: [B, S, 768]
```

三个组件：

**1. Router（路由器）**——一个 `nn.Linear(768, 4)`，把每个 token 映射到"对每个专家的偏好分数"，过 softmax 变成概率。只有 768×4 = 3072 个参数，几乎零开销。

**2. Experts（专家）**——4 个独立的 SwiGLU FeedForward，各有各的 gate_proj/up_proj/down_proj，参数完全独立。

**3. Top-K + 加权合并**——取概率最高的 K 个专家（MiniMind: K=1），专家输出乘以路由权重作为最终输出。

## 从零开始写

> 目标文件：`scratch/model/model_minimind.py`（追加在 `FeedForward` 类之后）
>
> Step 1-6 逐步构建 `MOEFeedForward` 类。Step 2、3 是纯讲解不写代码，其余 Step 各写一段代码，最后拼起来就是"完整实现"里的代码。

### Step 1：`__init__` — 搭骨架

`__init__` 初始化三个组件：Router（路由器）、Experts（专家们）、辅助损失。先解释每个参数的含义，再看代码。

#### 参数说明

| 参数 | 含义 | MiniMind 的值 |
|------|------|-------------|
| `hidden_size` | 输入/输出维度 | 768 |
| `intermediate_size` | 每个专家 FFN 的中间维度 | 2432 |
| `num_experts` | 专家总数 | 4 |
| `num_experts_per_tok` | 每个 token 选几个专家（top-K 的 K） | 1 |
| `norm_topk_prob` | 是否对选中专家的权重做归一化 | True |
| `router_aux_loss_coef` | 辅助损失的系数（越大越强调均匀分配） | 5e-4 |

后两个参数展开讲：

**`norm_topk_prob`（归一化 top-K 权重）**：Router 给 4 个专家打分后取 top-K，选中的 K 个专家的概率值加起来不一定等于 1。`norm_topk_prob=True` 表示把它们重新归一化，确保权重和为 1。比如 top-2 选中了概率 0.42 和 0.31 的两个专家，归一化后变成 0.575 和 0.425。top-1 时只有一个值，归一化后恒为 1.0（这个 Step 4 会再详细讲）。

**`router_aux_loss_coef`（辅助损失系数）**：MoE 有一个"赢家通吃"问题——如果不加约束，Router 会把大部分 token 都分给同一个专家，其他专家废掉。辅助损失（auxiliary loss）就是用来惩罚这种不均匀分配的。这个系数控制惩罚力度，设得很小（5e-4）是为了不干扰主任务的学习。这个 Step 6 会详细讲。

#### 每个专家只看到部分 token，不会丢失上下文吗？

不会。原因要从 Attention 的输出说起。

回忆 Transformer Block 的结构（第 7 章）：

```
输入 → Attention → + 残差 → FFN/MoE → + 残差 → 输出
```

Attention 不是只输出"一个分数"——分数只是中间产物。完整流程是：

```
Q·K^T → scores → softmax → 注意力权重 → 权重 × V → 输出 [B, S, 768]
                                              ↑
                                         这一步把其他 token 的信息
                                         加权求和进了当前 token
```

**注意力权重乘以 V 之后，每个 token 的 768 维向量里已经揉进了它关注的所有 token 的 V 信息。** 比如"踢"这个 token 的输出向量里，已经包含了"小明"（主语）、"球"（宾语）的信息——这些是 Attention 通过加权求和融合进来的。

所以到 FFN 的时候，**每个 token 的 768 维向量是"已经看过完整上下文的"**。FFN 只需要对这个向量做独立的非线性变换（gate_proj→SiLU→乘 up_proj→down_proj），输入 `[768]` 输出 `[768]`，整个过程不需要看旁边的 token，因为上下文信息已经在向量里了。

这就是为什么把 token 分给不同专家完全没问题——每个专家拿到的每个 token 都是"已经看过完整上下文的"向量，专家只需要独立处理就行。

**这也是 MoE 只做在 FFN 上、不做在 Attention 上的根本原因之一**：FFN 天然是 token 独立的，可以按 token 分发；Attention 需要 token 间成对交互（Q·K^T 要两两配对），按 token 分发会破坏它的工作方式。

#### 代码

> 写入 `MOEFeedForward.__init__`。`forward` 先留 `pass` 占位，Step 4-6 逐步填充。

```python
class MOEFeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=2432,
                 num_experts=4, num_experts_per_tok=1,
                 norm_topk_prob=True, router_aux_loss_coef=5e-4):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        pass  # Step 4-6 逐步填充
```

### Step 2：forward 要解决什么问题？（先不写代码）

在写任何代码之前，先搞清楚 MoE 的 forward 到底要做什么。整个过程分三步：

```
第一步: 路由 (Routing)
  "每个 token 该去哪个专家？权重多大？"
  → Router 给 4 个专家打分（softmax 变成概率），选概率最高的那个
  → 输出两样东西：选中的专家编号（idx）+ 对应的概率值作为权重（weight）
  → 比如 token 37 → Expert 2（权重 0.42）, token 38 → Expert 0（权重 0.35）

第二步: 分发计算 (Dispatch)
  "把 token 发给对应的专家，让它算"
  → 从 256 个 token 中挑出属于 Expert 0 的 ~64 个，集中送进 Expert 0 的 FFN
  → 对 Expert 1, 2, 3 重复

第三步: 加权合并 (Gather)
  "把每个专家算完的结果，乘以路由权重，放回原来的位置"
  → Expert 0 算出了 64 个结果，每个结果乘以该 token 选 Expert 0 时的路由权重
  → 加权后的结果放回输出 tensor 中对应的行
  → 对 Expert 1, 2, 3 重复
  → 如果 top-K > 1，同一个 token 会被多个专家处理，多份加权结果累加在一起
    （类似 Attention 里 softmax 权重 × V 的加权求和）
```

第一步（路由）和第二步（计算）都好理解。**难点在第三步——"加权后放回原来的位置"。**

这里有两个子问题：
- **加权**：专家输出要乘以路由权重（Router 给的概率值）。top-1 时权重归一化后恒为 1.0（等于不加权），top-2 时权重如 0.6 和 0.4，两个专家的输出按比例混合。
- **放回原位置**：结果不是按顺序排的，要散落地放回 y 的指定行。PyTorch 里做这件事的操作叫 `index_add_`。

### Step 3：怎么把专家的结果放回原位置？

Step 2 说了，MoE forward 的难点在第三步——"把专家算完的结果放回原来的位置"。

#### 问题是什么？

假设有 8 个 token，Expert 0 被分到了 token 1、2、5。Expert 0 算完后产出 3 个结果，**每个结果要乘以路由权重**，然后放回输出 tensor y 的第 1、2、5 行：

```
y 是一个 [8, 768] 的全零 tensor（等着被填充）

Expert 0 算完: raw_result = [3, 768]（3 个 token 的 FFN 输出）
  加权:        result = raw_result × weight   （乘以路由权重）
  result[0] 应该放到 y[1]
  result[1] 应该放到 y[2]
  result[2] 应该放到 y[5]
  y 的其他行先不动（等其他专家来填）
```

这不是按顺序填的——result 的第 0 份要放到 y 的第 1 行，第 1 份放第 2 行，第 2 份放第 5 行。**位置是散落的、不连续的。**

如果 top-K > 1（比如 top-2），token 1 可能同时被 Expert 0（权重 0.6）和 Expert 2（权重 0.4）处理。两个专家各自乘以权重后的结果需要**累加**到 y[1]，最终 y[1] = Expert 0 的加权输出 + Expert 2 的加权输出。这和 Attention 里 softmax 权重 × V 的加权求和是一个道理。

#### 方案 A：直接赋值 `y[positions] = result`

```python
y[[1, 2, 5]] = result    # 把 result 的 3 行分别赋值到 y 的第 1、2、5 行
```

top-1 时可以用。但 **top-2 时会出问题**：同一个 token 可能被两个专家处理，两个专家各输出一份结果。Expert 0 先把结果写到 y[1]，Expert 2 再把结果写到 y[1]——后者覆盖前者，Expert 0 的计算白费了。

#### 方案 B：用 for 循环逐个累加

```python
positions = [1, 2, 5]
for i in range(len(positions)):
    y[positions[i]] += result[i]    # 注意是 +=（累加），不是 =（覆盖）
```

累加就不怕被覆盖了——两个专家的结果叠加在一起。逻辑上完美，但 **Python for 循环太慢了**——每步都要在 Python 和 C++ 之间切换，256 个 token × 4 个专家 = 上千次循环。

#### 方案 C：`index_add_`——方案 B 的高速版

`index_add_` 做的事情和方案 B 的 for 循环**完全一样**，但 PyTorch 内部用 C++ 并行执行，快很多：

```python
y.index_add_(dim, index, source)
```

三个参数：
- **`dim`**：沿哪个维度操作。MoE 里每个 token 是一**行**，按行操作所以写 `0`。不能省略——PyTorch 要求显式指定。
- **`index`**：一维整数 tensor，告诉每一份结果应该放到 y 的哪一行
- **`source`**：要加上去的数据

> **名字拆解**：`index`（按索引）+ `add`（累加上去）+ `_`（in-place，直接修改 y 本身）

用最简单的一维例子理解：

```python
y = torch.zeros(5)             # 5 个格子，初始全是 0: [0, 0, 0, 0, 0]
source = torch.tensor([7, 9])  # 2 个结果: 7 和 9
index = torch.tensor([1, 3])   # 7 放第 1 格，9 放第 3 格

# 等价的 for 循环（方案 B）:
#   y[1] += 7   → y = [0, 7, 0, 0, 0]
#   y[3] += 9   → y = [0, 7, 0, 9, 0]

# 一行替代（方案 C）:
y.index_add_(0, index, source)
#            ↑   ↑      ↑
#          按行  放哪   放什么

# 结果: y = [0, 7, 0, 9, 0]
#              ↑     ↑
#            第1格  第3格
```

二维也一样，只是每个"格子"从一个数变成一行向量：

```python
y = torch.zeros(5, 3)                        # 5 行 3 列，全是 0
source = torch.tensor([[7, 8, 9],             # 第 0 份结果
                        [4, 5, 6]])           # 第 1 份结果
index = torch.tensor([1, 3])                  # 第 0 份放第 1 行，第 1 份放第 3 行

y.index_add_(0, index, source)

# 等价于:
#   y[1] += [7, 8, 9]
#   y[3] += [4, 5, 6]
#
# 结果:
#   y = [[0, 0, 0],   ← 第 0 行：没人放
#        [7, 8, 9],   ← 第 1 行：被 source[0] 加上了
#        [0, 0, 0],   ← 第 2 行：没人放
#        [4, 5, 6],   ← 第 3 行：被 source[1] 加上了
#        [0, 0, 0]]   ← 第 4 行：没人放
```

MoE 中：y 是 `[256, 768]` 的全零输出容器，source 是某个专家的 FFN 输出，index 是被分到这个专家的 token 编号。

#### `index_add_` 的限制 → 为什么要展平

`index_add_` 有一个限制：**index 必须是一维的——只能用一个数字定位一个位置。**

现在看 MoE 的输入。原始输入 x 是 `[B=2, S=128, 768]`（三维），一个 token 的位置需要**两个数字**：`(batch=1, pos=5)`。

```
x 是 [2, 128, 768]（三维）

  "token (batch=1, pos=5) 的结果要放回去"

  y.index_add_(0, ???, result)
  → index 只能放一维的 [1, 2, 5] 这样的东西
  → 没法表达 "batch=1 且 pos=5" 这种二维地址
  → 做不到！
```

**解法：展平。** 把 `[B, S, 768]` 变成 `[B×S, 768]`，每个 token 从二维地址 `(batch, pos)` 变成一维地址（一个数字）：

```
展平: x_flat = x.view(-1, 768)   →  [256, 768]（二维）

  (batch=0, pos=0)   → token 0
  (batch=0, pos=1)   → token 1
  ...
  (batch=0, pos=127) → token 127
  (batch=1, pos=0)   → token 128
  (batch=1, pos=5)   → token 133    ← 一个数字就能定位！

  y.index_add_(0, [37, 100, 133], result)   ← index 是一维的，搞定 ✓
```

**展平不是为了"打乱顺序"或"连续编号"，纯粹是因为 `index_add_` 的 index 只能是一维的，我们必须把 (batch, pos) 这个二维地址压成一个数字。** 最后 `y.view(B, S, D)` 恢复原始三维形状，顺序完全不变。

### Step 4：写代码——展平 + Router 打分 + Top-K + 权重归一化

理解了整体思路（Step 2）和 `index_add_` 为什么要展平（Step 3）之后，开始写 forward 的前半段。

> 写入 `MOEFeedForward.forward()`，替换 Step 1 的 `pass` 占位。这是 forward 的前半段，Step 5、6 继续往后接。

```python
    def forward(self, x):
        B, S, D = x.shape

        # ---- 展平：把二维地址(batch, pos)变成一维地址，为后面 index_add_ 服务 ----
        x_flat = x.view(-1, D)  # [B, S, 768] → [B×S, 768] = [256, 768]

        # ---- Router 打分 ----
        # gate 是 nn.Linear(768, 4)，把每个 token 映射到 4 个专家的偏好分数
        # softmax 归一化成概率（每行和为 1）
        scores = F.softmax(self.gate(x_flat), dim=-1)  # [256, 4]

        # ---- Top-K 选择（K=1：每个 token 只选 1 个专家）----
        topk_weight, topk_idx = torch.topk(
            scores, k=self.num_experts_per_tok, dim=-1, sorted=False
        )
        # topk_weight: [256, 1]  选中专家的概率值（如 0.42）
        # topk_idx:    [256, 1]  选中专家的编号（如 2）

        # ---- 权重归一化 ----
        if self.norm_topk_prob:
            # .sum(dim=-1, keepdim=True): 沿最后一维求和，保留该维度为 size=1
            #   keepdim=True:  [256, K] → [256, 1]  ← 能和 topk_weight [256, K] 正确广播做除法
            #   keepdim=False: [256, K] → [256]     ← 少一维，除法语义错误
            # + 1e-20: 防止除以 0（理论上 softmax 输出不为 0，但加了更安全）
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        # K=1 时: [0.42] / 0.42 = [1.0]（只有一个值，归一化后恒为 1）
        # K=2 时: [0.42, 0.31] → [0.575, 0.425]（确保权重和为 1）
```

K=1 时权重归一化实际无效（归一化后都是 1.0）。但代码要兼容 K>1 的情况。

**这里有一个容易忽略的后果**：K=1 + norm_topk_prob=True 时，topk_weight 恒为 1.0，专家输出不依赖 router 的概率值。而 topk_idx 是离散选择（"选第几个专家"），不可微。两条路都断了——**主任务 loss 几乎无法通过 MoE 输出给 router 提供梯度**。Router 主要靠后面讲的辅助损失（aux_loss）来训练，保持负载均衡。

这也意味着 `self.aux_loss` 必须在训练脚本里显式加到总 loss 才有效。MiniMind 的做法（`train_pretrain.py`）：

```python
loss = res.loss + res.aux_loss
```

如果漏了这一行，router 完全没有梯度，专家分配会退化成随机。

### Step 5：分发给专家计算

现在有了每个 token 的路由结果（topk_idx 告诉"去哪个专家"，topk_weight 告诉"权重多大"）。接下来做 Step 2 讲的第二步和第三步：遍历每个专家，找出属于它的 token，集中计算，用 `index_add_` 把结果放回原位置。

#### 关键操作解释

**初始化输出容器**：用 `zeros_like(x_flat)` 而不是 `torch.zeros(256, 768)`，好处是自动匹配 device 和 dtype，不用手动处理一致性。后面 `index_add_` 会往这个全零容器里逐步累加各专家的输出。

**构建 mask**：`(topk_idx == i)` 逐元素比较，得到同形状的布尔张量。比如 `topk_idx = [[2], [0], [0], [1]]`，`i=0` 时 `mask = [[F], [T], [T], [F]]`，形状 `[4, 1]`。

**跳过空专家**：`mask.any()`（不带参数）检查整个张量有没有 True——"有没有任何 token 选了这个专家？"没有就跳过，不白算。

**获取 token 一维索引**——`mask.any(dim=-1).nonzero().flatten()` 是一条链式操作，拆三步：

1. `mask.any(dim=-1)`：回答"这个 token 有没有选当前专家"。for 循环按专家遍历，每轮需要知道哪些 token 要送到当前专家。mask 形状是 `[T, K]`，每行有 K 个槽位，`any(dim=-1)` 把每行压成一个布尔值——"K 个槽位中有没有选中当前专家"。K=1 时等价于 squeeze（只有一个槽位，True 就是 True）；K>1 时才真正起到 any 的作用，例如 K=2: `[[F,T], [T,F], [F,F]]` → `[T, T, F]`，表示 token 0 和 token 1 要送去当前专家。多个专家的输出叠加不在这里发生，而是通过 for 循环的多轮 `index_add_` 累加实现
2. `.nonzero()`：返回所有 True 元素的坐标，`[F, T, T, F] → [[1], [2]]`，形状 `[num_true, 1]`
3. `.flatten()`：压成一维 `[1, 2]`，因为 `index_add_` 要求 index 必须是一维张量

**提取路由权重**：`topk_weight[mask]` 布尔索引取出 True 位置的值，返回一维张量 `[num_selected]`。`.view(-1, 1)` 重塑为列向量 `[num_selected, 1]`，这样和 FFN 输出 `[num_selected, 768]` 相乘时会广播到每一列。

**计算 + 放回**：`x_flat[token_idx]` 取出对应 token → `expert(...)` 过 SwiGLU FFN → `* weight` 乘路由权重（广播到 768 列）→ `.to(y.dtype)` 处理混合精度训练时的类型不一致。`index_add_(0, token_idx, source)` 沿第 0 维把 source 的第 j 行**累加**到 y 的第 `token_idx[j]` 行——累加而非覆盖，所以 top-K > 1 时同一 token 被多个专家处理，多份结果自然叠加。

#### 代码

> 接 Step 4，仍在 `forward()` 内。这是 for 循环的主体，Step 6 会补充 `elif` 分支和循环之后的 aux_loss 计算。

```python
        y = torch.zeros_like(x_flat)  # [256, 768]

        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx,
                             (expert(x_flat[token_idx]) * weight).to(y.dtype))
```

#### 完整的数据流（用实际规模）

```
256 个 token 的路由结果（假设大致均匀）：
  Expert 0 分到: token [1, 7, 12, ...]  ~64 个 token
  Expert 1 分到: token [3, 5, 18, ...]  ~65 个 token
  Expert 2 分到: token [0, 2, 9, ...]   ~70 个 token
  Expert 3 分到: token [4, 8, 11, ...]  ~57 个 token

Expert 2 处理它的 70 个 token:
  取出:       x_flat[[0, 2, 9, ...]] → [70, 768]
  gate_proj:  [70, 768] @ [768, 2432] = [70, 2432] → SiLU
  up_proj:    [70, 768] @ [768, 2432] = [70, 2432]
  相乘:       [70, 2432]
  down_proj:  [70, 2432] @ [2432, 768] = [70, 768]
  × weight:   [70, 768] × [70, 1] = [70, 768]
  index_add_: 把这 70 行结果放回 y 的第 0, 2, 9, ... 行

4 个专家各自处理完，y 里所有 256 行都填好了
```

### Step 6：训练时的两个特殊处理

#### (a) 没分到 token 的专家——保持梯度流

在 Step 5 的 for 循环里，给 `mask.any()` 加一个 `elif` 分支：

> 在 Step 5 的 `if mask.any():` 块后面追加 `elif`，仍在 for 循环内。

```python
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

如果某个专家在这个 batch 里一个 token 都没分到，它的参数不在计算图中。DDP（分布式训练）要求所有参数都参与梯度同步，否则报错。

`0 * sum(params)` 加了一个恒为 0 的值——不改变输出，但让参数出现在计算图里。

#### (b) 辅助损失（Auxiliary Loss）——防止专家坍缩

**问题：赢家通吃**

没有辅助损失时，路由器会正反馈失控：

```
初始: 4 个专家各分到 ~25% token
  ↓ Expert 2 碰巧稍好 → 更多 token 被路由到它 → 得到更多训练
  ↓ Expert 2 越来越好 → 路由器锁定 Expert 2
  ↓ 最终: Expert 2 分到 90% token，其他 3 个废掉
  ↓ 结果: 4 个专家 = 1 个专家，白白浪费 3× 参数
```

**解法思路：设计一个 loss，让"分配越不均匀，loss 越大"**

需要两个量来衡量"分配均不均匀"。这两个量都是**在当前 batch 的 forward 里实时算出来的**，不是预设的。

**辅助损失公式**

$$\mathcal{L}_{aux} = N \cdot \alpha \cdot \sum_{i=1}^{N} f_i \cdot p_i$$

| 符号 | 含义 | MiniMind 的值 |
|------|------|-------------|
| $N$ | 专家总数 | 4 |
| $\alpha$ | 辅助损失系数（`router_aux_loss_coef`），设得很小以不干扰主 loss | $5 \times 10^{-4}$ |
| $f_i$ | 第 $i$ 个专家在当前 batch 中实际被选中的频率。top-k 选完后，统计每个专家被选了多少次除以总 token 数。代码中用 `F.one_hot` 编码后取 `mean` 实现 | 均匀时 $f_i = 1/N = 0.25$ |
| $p_i$ | router 对第 $i$ 个专家的平均偏好。把当前 batch 所有 token 的 softmax 输出按列取平均得到（见下方解释） | 均匀时 $p_i = 1/N = 0.25$ |

**$f_i$ 和 $p_i$ 分别是怎么来的？**

用一个完整例子说清楚。假设句子"我 爱 写 代 码 ！"有 6 个 token，4 个专家。每个 token 过 router 后 softmax 输出一行概率，6 个 token 拼起来就是一个 $[6, 4]$ 的 softmax 矩阵——**每行是一个 token 对 4 个专家的偏好分布**（softmax 保证每行和为 1）。

$f_i$ 和 $p_i$ 都从这同一个矩阵算出来，但方式完全不同：

```
scores 矩阵 [6, 4]         Expert 0   Expert 1   Expert 2   Expert 3   │ top-1 选择
                           ┌──────────┬──────────┬──────────┬──────────┐│
 token 0 "我"               │  [0.45]  │   0.25   │   0.18   │   0.12   ││→ Expert 0 ✓
 token 1 "爱"               │   0.10   │  [0.55]  │   0.20   │   0.15   ││→ Expert 1 ✓
 token 2 "写"               │  [0.50]  │   0.15   │   0.22   │   0.13   ││→ Expert 0 ✓
 token 3 "代"               │   0.38   │   0.12   │  [0.40]  │   0.10   ││→ Expert 2 ✓
 token 4 "码"               │  [0.42]  │   0.18   │   0.30   │   0.10   ││→ Expert 0 ✓
 token 5 "！"               │  [0.35]  │   0.30   │   0.15   │   0.20   ││→ Expert 0 ✓
                           ├──────────┼──────────┼──────────┼──────────┤│
 ↓ 列平均 scores.mean(0)   │  0.367   │  0.258   │  0.242   │  0.133   ││
 = p_i                     └──────────┴──────────┴──────────┴──────────┘│
                                                                        │
 ↓ 统计 top-1 选中次数/总数   4/6       1/6        1/6       0/6        │
 = f_i                       0.667     0.167      0.167     0.000       │
```

方括号 `[0.45]` 表示该行最大值，即 top-1 选中的专家。top-1 选择就是 `torch.topk(scores, k=1)`——对每行取最大值的索引，是确定性操作（不涉及温度或采样，那些只在文本生成的最后一层用）。

- **$p_i$（列平均）**：把矩阵**按列取平均**（`scores.mean(0)`），得到 router 对每个专家的"平均偏好"。是 softmax 的连续输出，**可微**。即使 token 4 没选 Expert 2，它给 Expert 2 的概率 0.30 仍然被计入 $p_2$。
- **$f_i$（选中频率）**：统计每个专家被 top-1 选中了几次，除以总 token 数。是离散计数，**不可微**。只看"最终选了谁"，不关心具体概率值。

**$f$ 和 $p$ 的趋势相同但幅度不同**——$f$ 远比 $p$ 极端：

```
        Expert 0   Expert 1   Expert 2   Expert 3
p_i      0.367      0.258      0.242      0.133     ← 看着还算均匀（最大:最小 ≈ 3:1）
f_i      0.667      0.167      0.167      0.000     ← 实际极度不均（Expert 0 占 2/3，Expert 3 = 0）
```

Expert 0 的 softmax 均值只有 0.367，但因为它在 4 个 token 上都恰好是最大的，top-1 后 $f_0$ 被放大到 0.667。Expert 3 的 softmax 均值 0.133 不算零，但它在所有 token 上都不是最大的，top-1 后 $f_3 = 0$。**top-1 的离散化会把微小的概率优势放大成巨大的负载差距。**

**为什么公式用 $f_i \cdot p_i$，而不是 $p_i^2$ 或 $(p_i - 1/N)^2$？**

有三个看似更直觉的方案，逐个分析为什么不行：

**方案 A：$\sum f_i^2$（只用实际负载的平方和）**

数学上没问题——$\sum f_i^2$ 在 $\sum f_i = 1$ 约束下确实在均匀时取最小值。但 $f_i$ 来自 top-k 离散选择（选中 = 1，没选中 = 0，然后计数），**不可微**。对 $f_i$ 求导得 $2f_i$，但 $f_i$ 本身不可微，梯度断了，router 收不到任何信号。

**方案 B：$\sum p_i^2$（只用 softmax 均值的平方和）**

可微，但 **$p_i$ 看不到 top-1 离散化带来的放大效应**。用上面的例子算：

```
p = [0.367, 0.258, 0.242, 0.133]
sum(p_i²) = 0.135 + 0.067 + 0.059 + 0.018 = 0.278

均匀时:
sum(0.25²) × 4 = 0.250

差距: 0.278 vs 0.250 → loss 觉得"还行，不太需要调"
```

但实际负载 $f = [0.667, 0.167, 0.167, 0.000]$ 已经严重失衡了！$p$ 看起来最大:最小 ≈ 3:1，$f$ 实际上是 Expert 0 吃掉了 2/3 的 token、Expert 3 一个都没分到。**只看 $p$ 会严重低估不均匀程度。**

**方案 C：$\sum (p_i - 1/N)^2$（直接让 $p$ 逼近均匀分布 $1/N$）**

和方案 B 本质相同——展开 $(p_i - 1/N)^2 = p_i^2 - 2p_i/N + 1/N^2$，求和后 $= \sum p_i^2 - 1/N$（因为 $\sum p_i = 1$），和 $\sum p_i^2$ 只差一个常数，梯度完全一样。同样的问题：只看 $p$，看不到 top-1 放大效应。

**为什么 $f_i \cdot p_i$ 是对的**

$f_i \cdot p_i$ 结合了两者的优点：

| | $f_i$（实际负载） | $p_i$（softmax 均值） |
|---|---|---|
| 来源 | top-k 离散计数 | softmax 连续输出按列取平均 |
| 可微 | 不可微，反传时当常数 | 可微，梯度流回 router |
| 角色 | **探测器**：告诉 loss "谁真的被分多了" | **梯度载体**：让 loss 能调 router 权重 |

用上面例子的数字对比三种方案的信号强度：

```
方案 B  sum(p_i²)    = 0.278    均匀时 0.250    差距 0.028    ← 信号弱
方案 C  sum(p-1/N)²  = 0.028    均匀时 0.000    差距 0.028    ← 等价，信号一样弱
公式    sum(f_i·p_i) = 0.667×0.367 + 0.167×0.258 + 0.167×0.242 + 0×0.133
                     = 0.245 + 0.043 + 0.040 + 0 = 0.328
                                      均匀时 0.250    差距 0.078    ← 信号强 3 倍
```

$f_i$ 把 top-1 离散化后的真实不均匀注入到了 loss 里——Expert 0 的 $f_0 = 0.667$ 远大于 $p_0 = 0.367$，让乘积 $0.667 \times 0.367 = 0.245$ 远大于 $p_0^2 = 0.135$。

梯度角度：

$$\frac{\partial \mathcal{L}_{aux}}{\partial p_i} = N \cdot \alpha \cdot f_i$$

- $f_0 = 0.667$（严重过载）→ 给 $p_0$ 的梯度最大 → router 被强力推动降低对 Expert 0 的偏好
- $f_3 = 0.000$（完全空闲）→ 给 $p_3$ 的梯度为零 → $p_3$ 不被下压 → 相对地被"推高"

如果用 $p_i^2$，梯度是 $2p_i$——Expert 0 梯度 $2 \times 0.367 = 0.734$，Expert 3 梯度 $2 \times 0.133 = 0.266$，差距只有 2.8 倍。而用 $f_i$，Expert 0 梯度 0.667，Expert 3 梯度 0，差距是无穷大。**$f_i$ 作为梯度权重，比 $p_i$ 更能反映实际的不均匀程度。**

`aux_loss` 必须在训练脚本里显式加到总 loss 才有效：

```python
# train_pretrain.py
loss = res.loss + res.aux_loss   # 漏了 res.aux_loss → router 没有梯度 → 专家退化
```

#### 代码

> 接 for 循环之后，仍在 `forward()` 内。这段加上 `return` 完成整个 `forward()` 方法。

```python
        if self.training and self.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * self.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(B, S, D)
```

## 维度追踪

```
输入 x: [2, 128, 768]

展平:   x_flat = [256, 768]

Router:
  gate(x_flat): [256, 768] @ [768, 4] = [256, 4]
  softmax:      [256, 4]  概率分布
  top-1:        topk_weight [256, 1], topk_idx [256, 1]

Expert 分发（假设均匀，每个专家 ~64 个 token）:
  Expert_i 输入:  [~64, 768]
  gate_proj:      [~64, 768] @ [768, 2432] = [~64, 2432]  → SiLU
  up_proj:        [~64, 768] @ [768, 2432] = [~64, 2432]
  逐元素相乘:     [~64, 2432]
  down_proj:      [~64, 2432] @ [2432, 768] = [~64, 768]
  × weight + index_add_ 放回 y

输出 y: [256, 768] → reshape → [2, 128, 768]
```

## 参数量对比

```
Dense FFN（1 个 SwiGLU）:
  gate_proj: 768 × 2432 = 1,867,776
  up_proj:   768 × 2432 = 1,867,776
  down_proj: 2432 × 768 = 1,867,776
  合计: 5,603,328 ≈ 5.6M

MoE FFN（4 个 SwiGLU + Router）:
  4 × 5,603,328         = 22,413,312  （4 个专家）
  Router: 768 × 4       = 3,072       （路由器）
  合计: 22,416,384 ≈ 22.4M

参数量: ×4
每个 token 的计算量: 5.6M（只过 1 个专家，和 Dense 相同）
```

## 完整实现

> 追加到 `scratch/model/model_minimind.py`（FeedForward 已在上一章追加）

```python
class MOEFeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=2432,
                 num_experts=4, num_experts_per_tok=1,
                 norm_topk_prob=True, router_aux_loss_coef=5e-4):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(
            scores, k=self.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * self.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(B, S, D)
```

验证代码追加到 `if __name__ == '__main__':` 块中（参见 [01-Embedding](01-Embedding.md) 关于文件组织的说明）：

```python
    # ---- MOEFeedForward ----
    moe = MOEFeedForward()
    x = torch.randn(2, 128, 768)
    assert moe(x).shape == x.shape

    total = sum(p.numel() for p in moe.parameters())
    expert_params = sum(p.numel() for p in moe.experts.parameters())
    router_params = sum(p.numel() for p in moe.gate.parameters())
    print(f"MoE total params: {total:,}")          # 22,416,384
    print(f"  Expert params:  {expert_params:,}")   # 22,413,312 (4 × 5,603,328)
    print(f"  Router params:  {router_params:,}")   # 3,072
    print(f"  Dense FFN 单个: {expert_params // 4:,}")  # 5,603,328
    print(f"  参数是 Dense 的 {total / (expert_params // 4):.1f}×，计算量是 1×")

    with torch.no_grad():
        scores = F.softmax(moe.gate(x.view(-1, 768)), dim=-1)
        _, idx = scores.topk(1, dim=-1)
        for i in range(4):
            count = (idx == i).sum().item()
            print(f"Expert {i}: {count}/{idx.shape[0]} tokens ({count/idx.shape[0]*100:.1f}%)")
    print("MOEFeedForward verified!")
```

## 对照 MiniMind 源码

| 我们的实现 | MiniMind 源码（model_minimind.py:148-176） |
|-----------|------------------------------------------|
| `MOEFeedForward` 类 | 完全对应 |
| `self.gate = nn.Linear(hidden_size, num_experts)` | 一致 |
| `self.experts = nn.ModuleList([FeedForward(...)])` | 一致（传 `config.moe_intermediate_size`） |
| softmax → topk → norm_topk_prob | 一致 |
| for 循环 + mask + index_add_ | 一致 |
| `0 * sum(...)` 保持梯度流 | 一致 |
| aux_loss 计算 | 一致 |

## 业界对比：DeepSeek-V3 如何重新定义 MoE

DeepSeek-V3（2024.12）是 MoE 在大模型领域的标杆——671B 总参数，每个 token 只激活 37B（5.5%）。它在 MoE 的三个关键设计上都做了改进：

### 架构配置对比

| 特性 | MiniMind | DeepSeek-V3 | Mixtral 8×7B |
|------|---------|-------------|-------------|
| hidden_size | 768 | 7168 | 4096 |
| 层数 | 8 | 61 | 32 |
| 路由专家数 | 4 | 256 | 8 |
| 共享专家 | 无 | 1 个（所有 token 都过） | 无 |
| top-k | 1 | 8 | 2 |
| 单专家 intermediate | 2432 | 2048 | 14336 |
| 总参数 | Dense ~64M / MoE ~198M | 671B | 46.7B |
| 每 token 活跃参数 | ~64M（和 Dense 相同） | 37B | ~13B |
| 活跃参数占比 | ~32%（MoE 部分 1/4） | ~5.5%（37B/671B） | ~28%（2/8） |

### 改进 1：Sigmoid 路由替代 Softmax

MiniMind 和 Mixtral 用 softmax 路由——所有专家的分数竞争，和为 1：

```python
# MiniMind / Mixtral: softmax routing
scores = F.softmax(gate(x), dim=-1)  # [n_tokens, 4]，每行和为 1
```

DeepSeek-V3 用 sigmoid——每个专家独立打分，不互相竞争：

```python
# DeepSeek-V3: sigmoid routing
scores = F.sigmoid(gate(x))  # [n_tokens, 256]，每个值独立在 0~1 之间
```

区别：softmax 是零和博弈（一个专家分高了，其他必然降低），sigmoid 允许多个专家独立得高分。选中的专家 affinity 值再做归一化作为输出权重。

### 改进 2：动态 Bias 替代辅助损失做负载均衡

MiniMind 用经典辅助损失 $\mathcal{L}_{aux}$ 防止专家坍缩。DeepSeek-V3 认为辅助损失直接作用于主训练目标，会干扰 router 学习最优路由。他们用动态 bias 项替代了主要的 batch-level 辅助损失（仍保留了极小系数 0.0001 的 sequence-level 辅助损失防止单条序列内的极端不均衡）。

DeepSeek-V3 的替代方案：**给每个专家一个 bias 项 $b_i$，训练中动态调节**。

```
路由公式:
  原始分数:  s_i = sigmoid(x · e_i)           每个专家的亲和度
  路由决策:  选 top-K 的 (s_i + b_i)          bias 影响谁被选中
  输出权重:  用原始 s_i（不含 bias）做加权     bias 不影响输出值

bias 更新规则（每个训练 step 结束后）:
  如果专家 i 被选的次数 > 平均值 → b_i -= γ   减小 bias，下次少选它
  如果专家 i 被选的次数 < 平均值 → b_i += γ   增大 bias，下次多选它
  γ = bias_update_speed（超参数）
```

关键：**bias 只影响"谁被选中"，不影响"选中后的权重"**。这样减少了负载均衡目标对主训练目标的直接干扰。官方消融实验显示多数指标优于纯辅助损失方案。

### 改进 3：共享专家（Shared Expert）

DeepSeek-V3 有 1 个共享专家：所有 token 都过这个专家，不经过路由。

```
每个 token 的实际处理:
  1 个共享专家（必过） + 8 个路由专家（top-8 选出） = 9 个专家同时激活
```

共享专家的目标是捕获各类 token 的公共知识，减少路由专家之间的冗余。路由专家则负责专业化的特征提取。

## MoE 的代价

```
优势:
  ✓ 参数和计算解耦——知识容量大，推理不慢
  ✓ 天然适合并行——不同专家放不同 GPU

代价:
  ✗ 显存占用大——所有专家参数都要加载到显存，即使大部分不用
    （DeepSeek-V3 的 671B 参数全部要加载，即使每次只算 37B）
  ✗ 通信开销——分布式时 token 要发送到对应专家所在的 GPU
  ✗ 训练不稳定——路由器容易坍缩，需要辅助损失或 bias 调节
  ✗ batch 效率不均——某些专家分到多 token 某些少，GPU 利用率波动
```

## 面试话术

> "MoE 解决的是 Dense 模型'参数和计算绑定'的问题。FFN 占单层 76% 参数，每个 token 必须过所有参数。MoE 把 FFN 拆成多个独立专家，每个 token 只路由到 top-K 个，总参数增大但每 token 活跃参数不变。只对 FFN 做 MoE 是因为 FFN 对每个 token 独立计算、参数占比高，天然适合按 token 分发；Attention 涉及 token 间成对交互，不方便这样做。
>
> 我复现了 MiniMind 的 MoE——4 专家 top-1 + softmax 路由 + 经典辅助损失。复现中发现一个细节：top-1 且 norm_topk_prob=True 时，路由权重归一化后恒为 1，加上 topk_idx 是离散选择不可微，主 loss 几乎没法给 router 提供梯度，router 主要靠辅助损失训练，而且 aux_loss 必须在训练脚本里显式加进总 loss 才有效。
>
> 业界进化方向是 DeepSeek-V3：256 路由专家 + 1 共享专家，top-8，sigmoid 路由替代 softmax 避免零和博弈，用动态 bias 项替代主要的辅助损失做负载均衡——bias 只影响路由选择不影响输出权重，减少了对主训练目标的干扰。671B 总参数只激活 37B，活跃参数占比约 5.5%。"

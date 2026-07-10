# 06 - MOE FeedForward：参数和计算的解耦

## 先算账：Transformer 单层参数花在哪？

在讲 MoE 之前，先搞清楚一个基础问题——一层 Transformer 的参数到底有多少，花在哪里。

### Attention 参数量

线性层参数量 = 输入维度 × 输出维度（无 bias 时）。

MiniMind 的 Attention 有 4 个投影矩阵：

```
wq:  hidden_size × (n_heads × head_dim)    = 768 × (8 × 96)  = 768 × 768 = 589,824
wk:  hidden_size × (n_kv_heads × head_dim) = 768 × (4 × 96)  = 768 × 384 = 294,912
wv:  hidden_size × (n_kv_heads × head_dim) = 768 × (4 × 96)  = 768 × 384 = 294,912
wo:  (n_heads × head_dim) × hidden_size    = 768 × 768        =             589,824

Attention 合计: 589,824 + 294,912 + 294,912 + 589,824 = 1,769,472 ≈ 1.77M
```

为什么 wk/wv 比 wq 小？因为 GQA：8 个 Q head 共享 4 个 KV head，KV 的参数只需要一半。

> 还有 q_norm 和 k_norm 各一个 RMSNorm（96 个参数），合计 192，相比 1.77M 可以忽略。

### FFN（SwiGLU）参数量

上一章的 SwiGLU 有 3 个投影矩阵：

```
gate_proj:  hidden_size × intermediate_size = 768 × 2432 = 1,867,776
up_proj:    hidden_size × intermediate_size = 768 × 2432 = 1,867,776
down_proj:  intermediate_size × hidden_size = 2432 × 768 = 1,867,776

FFN 合计: 1,867,776 × 3 = 5,603,328 ≈ 5.60M
```

### 单层参数分布

```
Attention:  1.77M  →  24%
FFN:        5.60M  →  76%
──────────────────────────
单层合计:   7.37M  → 100%

8 层 × 7.37M = 58.96M
+ Embedding/LM Head: 768 × 6400 = 4.92M（共享权重只算一份）
+ 其他 Norm: 忽略
──────────────────────────
整个模型: ≈ 64M 参数（MiniMind-64M 名字的来源）
```

**结论：FFN 占了单层参数的 76%。Transformer 的参数主要花在 FFN 上。**

## FFN 的权重矩阵在"做什么"？

上一章讲了 SwiGLU 的公式，但没有解释 gate_proj 和 up_proj 的每一行**在语义上**代表什么。

### 矩阵乘法的本质：逐行点积

`gate_proj` 是一个 `[768, 2432]` 的矩阵。当输入 x（一个 768 维向量）乘以它时：

```
x @ gate_proj.T = [768] @ [768, 2432] → [2432]

输出的第 i 个值 = x 和 gate_proj 第 i 行的点积
                 = sum(x[j] * gate_proj[i][j] for j in 0..767)
```

点积衡量的是两个向量的**相似度**——x 和 gate_proj 第 i 行越"像"，输出值越大。

所以 gate_proj 的第 i 行 = 一个学到的**模式**。2432 行 = 2432 个不同的模式。输入 x 和每个模式做相似度计算，SiLU 激活后决定"哪些模式被触发"。

up_proj 同理：它的 2432 行是另一组模式，产出"候选特征"。gate 的输出和 up 的输出逐元素相乘，相当于 gate 决定"开关"，up 决定"内容"。

```
具体例子：假设训练后 gate_proj 的某一行学到了"代码关键字"的模式
  - 输入是 "def" token → 点积大 → SiLU 输出大 → 这个维度被激活
  - 输入是 "诉讼" token → 点积小 → SiLU 输出接近 0 → 这个维度被抑制

2432 行 = 2432 个这样的模式，覆盖各种语义特征
```

> 注意：这里说"代码关键字模式"只是帮助理解的比喻。实际训练出的模式不一定是人类能命名的语义概念，但数学结构确实是"逐行做模式匹配"。

### 想要更聪明 = 需要更多模式

intermediate_size 从 2432 增大到 10000？模式从 2432 个增加到 10000 个，模型能识别更细粒度的特征。

但代价是：

```
gate_proj: 768 × 10000 = 7,680,000（原来 1.87M → 现在 7.68M）
up_proj:   同上
down_proj: 10000 × 768 = 7,680,000
合计: 23.04M（原来 5.60M → 4 倍增长）
```

**而且每个 token 都要过全部 10000 个模式**——参数量和计算量同比例增长，完全绑定。

## Dense 模型的根本困境

```
想让模型更聪明
  → 需要更多模式（增大 intermediate_size）
  → 参数增多
  → 每个 token 过所有参数 → 计算量同比增长
  → 推理变慢、训练变贵

参数量 ∝ 计算量，完全绑定。
```

这是 Dense（稠密）模型的先天限制：**每个 token 不管是什么内容，都要过所有参数。**

但仔细想想——一个 token 的内容是"代码"，它真的需要过"法律术语""医学概念""古诗词格律"这些模式吗？大部分模式对它来说输出接近 0，算了也白算。

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

**Attention 做"路由"**。它决定 token 之间怎么互相看——谁和谁交互。这是结构性的计算，不能分科。翻译任务里，中文 token 必须 attend 到英文 token，不能说"中文用中文专家的 attention"。

还有一个实际原因：**FFN 占 76% 参数**，MoE 化 FFN 的杠杆远大于 Attention。

> DeepSeek-V2/V3 对 Attention 也做了稀疏化，但不是 MoE 路线——走的是 MLA（Multi-head Latent Attention），把 KV 维度做低秩压缩。目标是省 KV Cache 显存，思路和 MoE 完全不同。

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

## 逐步实现

### Step 1：Router 打分 + Top-K 选择

```python
# x: [B, S, 768] → 展平成 [B×S, 768]，每个 token 独立路由
x_flat = x.view(-1, hidden_dim)  # [256, 768]（B=2, S=128）

# Router 打分：每个 token 对 4 个专家的偏好
scores = F.softmax(self.gate(x_flat), dim=-1)  # [256, 4]，每行和为 1

# Top-K 选择（K=1：每个 token 只选 1 个专家）
topk_weight, topk_idx = torch.topk(scores, k=1, dim=-1, sorted=False)
# topk_weight: [256, 1]  选中专家的概率值（如 0.42）
# topk_idx:    [256, 1]  选中专家的编号（如 2）
```

### Step 2：权重归一化

```python
if norm_topk_prob:
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
# K=1 时: [0.42] / 0.42 = [1.0]（只有一个值，归一化后恒为 1）
# K=2 时: [0.42, 0.31] → [0.575, 0.425]（确保权重和为 1）
```

K=1 时这一步实际无效（归一化后都是 1.0）。但代码要兼容 K>1 的情况。

### Step 3：分发给专家计算

遍历每个专家，找出被路由到它的 token，集中计算：

```python
y = torch.zeros_like(x_flat)  # [256, 768]

for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)                                    # 哪些 token 选了专家 i
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()      # 这些 token 的索引
        weight = topk_weight[mask].view(-1, 1)                # 对应的路由权重
        y.index_add_(0, token_idx,                            # 放回原位置
                     (expert(x_flat[token_idx]) * weight).to(y.dtype))
```

具体数据流：

```
假设 256 个 token 的路由结果：
  Expert 0 分到: token [1, 7, 12, ...]  ~64 个 token
  Expert 1 分到: token [3, 5, 18, ...]  ~65 个 token
  Expert 2 分到: token [0, 2, 9, ...]   ~70 个 token
  Expert 3 分到: token [4, 8, 11, ...]  ~57 个 token

Expert 2 处理它的 70 个 token:
  输入:     [70, 768]
  gate_proj: [70, 768] @ [768, 2432] = [70, 2432] → SiLU
  up_proj:   [70, 768] @ [768, 2432] = [70, 2432]
  相乘:     [70, 2432]
  down_proj: [70, 2432] @ [2432, 768] = [70, 768]
  × weight:  [70, 768] × [70, 1] = [70, 768]
  放回 y 的对应行

4 个专家各自处理完，y 里所有 256 行都填好了
```

### Step 4：训练时的两个特殊处理

#### (a) 没分到 token 的专家——保持梯度流

```python
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

如果某个专家在这个 batch 里一个 token 都没分到，它的参数不在计算图中。DDP（分布式训练）要求所有参数都参与梯度同步，否则报错。

`0 * sum(params)` 加了一个恒为 0 的值——不改变输出，但让参数出现在计算图里。

#### (b) 辅助损失（Auxiliary Loss）——防止专家坍缩

没有辅助损失时，路由器会出现"赢家通吃"：

```
初始: 4 个专家各分到 ~25% token
  ↓ Expert 2 碰巧稍好 → 更多 token 被路由到它 → 得到更多训练
  ↓ Expert 2 越来越好 → 路由器锁定 Expert 2
  ↓ 最终: Expert 2 分到 90% token，其他 3 个废掉
  ↓ 结果: 4 个专家 = 1 个专家，白白浪费 3× 参数
```

MiniMind 用经典的 load-balance 辅助损失：

```python
if self.training and router_aux_loss_coef > 0:
    load = F.one_hot(topk_idx, num_experts).float().mean(0)   # [4] 每个专家被选中的频率
    self.aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
```

$$\mathcal{L}_{aux} = N \cdot \alpha \cdot \sum_{i=1}^{N} f_i \cdot p_i$$

- $f_i$ = 专家 i 被选中的频率
- $p_i$ = router 给专家 i 的平均概率
- $N$ = 专家数（MiniMind: 4），$\alpha$ = 5e-4

当所有专家均匀分配时 $\sum f_i p_i$ 最小（= $1/N$），任何不均匀都会增大这个值。梯度反传推动 router 分散路由。

## MiniMind MoE 配置

```python
num_experts = 4              # 4 个专家
num_experts_per_tok = 1      # 每个 token 选 1 个（top-1 路由）
moe_intermediate_size = 2432 # 每个专家和 Dense FFN 中间维度一样大
norm_topk_prob = True        # 归一化 top-k 权重
router_aux_loss_coef = 5e-4  # 辅助损失系数，很小，不干扰主 loss
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


# 验证
moe = MOEFeedForward()
x = torch.randn(2, 128, 768)
out = moe(x)
assert out.shape == x.shape  # [2, 128, 768]

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
| 总参数 | ~64M (MoE ~86M) | 671B | 46.7B |
| 每 token 活跃参数 | ~7.4M | 37B | ~13B |
| 活跃比 | 1/4 = 25% | (8+1)/257 ≈ 3.5% | 2/8 = 25% |

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

区别：softmax 是零和博弈（一个专家分高了，其他必然降低），sigmoid 允许多个专家同时得高分。256 个专家时 softmax 会把概率稀释得太薄，sigmoid 更稳定。

### 改进 2：无辅助损失的负载均衡

MiniMind 用经典辅助损失 $\mathcal{L}_{aux}$ 防止专家坍缩。但 DeepSeek-V3 发现**辅助损失会干扰主任务训练**——为了均衡负载，router 被迫偏离最优路由，牺牲模型质量。

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

关键：**bias 只影响"谁被选中"，不影响"选中后的权重"**。这样 router 可以自由学习最优路由，bias 只在选中边界做微调。比辅助损失优雅得多。

### 改进 3：共享专家（Shared Expert）

DeepSeek-V3 有 1 个共享专家：所有 token 都过这个专家，不经过路由。

```
每个 token 的实际处理:
  1 个共享专家（必过） + 8 个路由专家（top-8 选出） = 9 个专家同时激活
```

共享专家保证所有 token 都能获得基础的语言能力（语法、常识）。路由专家提供专业化的知识（代码、数学、法律等）。

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

> "MoE 解决的是 Dense 模型'参数和计算绑定'的问题。Transformer 里 FFN 占单层 76% 的参数——768×2432 的三个矩阵，5.6M/层。每个 token 要过所有参数，想加参数就得加计算。MoE 把 FFN 拆成多个独立专家，每个 token 只路由到 top-K 个。参数 ×4 但计算 ×1。
>
> 为什么只对 FFN 做 MoE？FFN 的每一行权重是一个学到的模式——输入和它做点积就是模式匹配。不同 token 需要不同的模式，天然适合分专家。Attention 是 token 间的交互，不能分。
>
> 我复现了 MiniMind 的 MoE——4 专家 top-1 + softmax 路由 + 经典辅助损失。业界进化方向是 DeepSeek-V3：256 专家 top-8，sigmoid 路由替代 softmax 避免零和竞争，用动态 bias 项替代辅助损失来做负载均衡——bias 只影响路由选择不影响输出权重，比辅助损失更干净。还引入了 1 个共享专家保证基础能力。671B 总参数只激活 37B，活跃比 3.5%。"

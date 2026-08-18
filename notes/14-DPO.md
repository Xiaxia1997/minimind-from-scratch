# 14 - DPO：不训 Reward Model 的偏好对齐

## 在整个项目中的位置

```
Chapter 13: RL 对齐基础（纯理论——Policy Gradient → PPO → RLHF）
                                                        ↓
Chapter 14（本章）: 从零写 DPO 训练
  → scratch/dataset/lm_dataset.py（追加 DPODataset）
  → scratch/trainer/train_dpo.py（DPO loss + 训练主循环）
```

> 前置：[13-RL对齐基础](13-RL对齐基础.md)（Policy Gradient、PPO、RLHF 全景）
> 上游参考（只读）：`trainer/train_dpo.py`、`dataset/lm_dataset.py`

## 一句话总结

DPO 证明了一个关键数学等价：**最优 reward 可以用 policy 和 reference model 的 log 概率比直接表达**，不需要显式训练 reward model。这把 RLHF 的三步（训 RM → PPO → KL 约束）简化成了一步——直接在偏好数据上优化 policy。

---

## Part 1: 动机——RLHF 哪里太复杂了

[上一章](13-RL对齐基础.md) Part 8 详细讲了 RLHF 的三步走和四大困难。简单回顾关键痛点：

```
RLHF 的工程代价:

1. 要训一个单独的 Reward Model
   - 额外的数据、算力、调参
   - RM 质量直接决定对齐效果——RM 不好，后面全白干

2. PPO 训练不稳定
   - 4-5 个模型同时运行
   - clip epsilon、KL beta、学习率互相耦合
   - reward hacking：policy 钻 RM 漏洞

3. 在线采样昂贵
   - 每步都要用 policy 生成回复（rollout）
   - 7B 模型生成 1024 token ≈ 数十秒

DPO 的思路: 能不能直接跳过 RM 和 PPO，在偏好数据上一步到位？
```

---

## Part 2: DPO 的数学推导

### 2.1 起点：RLHF 的优化目标

RLHF 的 PPO 阶段要解决的优化问题是：

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi_\theta(\cdot|x)} \big[r_\phi(x, y)\big] - \beta \, D_{\text{KL}}\big(\pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x)\big)$$

翻译：**最大化 reward model 的打分，同时别偏离 reference model 太远。**

- 第一项 $r_\phi(x, y)$：reward model 对回复的打分
- 第二项 $\beta \cdot D_{\text{KL}}$：KL 散度惩罚，$\beta$ 控制约束强度

### 2.2 这个优化问题有闭式解

Rafailov et al. (2023) 的关键发现：**上述受约束优化问题的最优策略有闭式解**。

对给定的 reward 函数 $r(x, y)$，最优策略为：

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \cdot \exp\!\left(\frac{r(x, y)}{\beta}\right)$$

其中 $Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \cdot \exp(r(x, y) / \beta)$ 是归一化常数。

**推导思路**：把 KL 展开，目标函数对 $\pi$ 逐点求导，令为 0，解出 $\pi^*$。

把 $D_{\text{KL}}$ 的定义 $\sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ 代入优化目标，整理成对每个 $y$ 独立的项：

$$J(\pi) = \sum_y \pi(y|x) \left[r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

对 $\pi(y|x)$ 求导令为 0（加上 $\sum_y \pi = 1$ 的拉格朗日乘子约束），得到：

$$r(x,y) - \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} - \beta - \lambda = 0 \quad \Rightarrow \quad \pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \cdot \exp\!\left(\frac{r(x,y)}{\beta}\right)$$

归一化后就是上面的闭式解，$Z(x)$ 吸收了 $\lambda$ 和 $\beta$ 常数项。

**直觉**：最优策略 = reference 策略 × reward 的指数缩放。reward 高的回复概率被放大，reward 低的被缩小，$\beta$ 控制放大/缩小的力度——$\beta$ 大则缩放温和（紧跟 reference），$\beta$ 小则缩放激进（大幅偏离 reference）。

### 2.3 反解 reward

把最优策略的闭式解对 $r(x, y)$ 反解：

两边取 log：

$$\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{r(x, y)}{\beta} - \log Z(x)$$

移项：

$$\boxed{r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)}$$

这就是 DPO 的核心等式——**最优 reward 可以用策略和 reference 的 log 概率比来表达**。$\beta \log Z(x)$ 是只依赖 prompt $x$ 的常数，对同一个 prompt 下的不同回复来说是相同的，后面会消掉。

### 2.4 代入 Bradley-Terry，消掉 RM

[上一章](13-RL对齐基础.md) Part 8.2 讲过 Reward Model 的训练用 Bradley-Terry 偏好模型：

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

现在把 2.3 的 reward 表达式代入：

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \cancel{\beta \log Z(x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \cancel{\beta \log Z(x)}$$

$Z(x)$ 消掉了（同一个 prompt 的常数项相减为 0）。剩下：

$$r(x, y_w) - r(x, y_l) = \beta \left[\log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]$$

代回 Bradley-Terry：

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \left[\log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)$$

**Reward model 完全消失了！** 偏好概率完全由策略 $\pi^*$ 和 reference $\pi_{\text{ref}}$ 的 log 概率比决定。

### 2.5 DPO 的 Loss

把上述概率的负对数似然作为 loss（最大化偏好概率 = 最小化负对数似然），并把 $\pi^*$ 换成我们正在训练的 $\pi_\theta$。**为什么能换？** $\pi^*$ 是满足闭式解的最优策略——我们训不出精确的 $\pi^*$，但最小化 DPO loss 就是在把 $\pi_\theta$ 往 $\pi^*$ 推。loss = 0 当且仅当 $\pi_\theta = \pi^*$。

$$\boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right]\right)\right]}$$

**逐个符号拆解**：

| 符号 | 全称 | 含义 |
|------|------|------|
| $y_w$ | winner / chosen | 人类标注员选出的**更好**回复（数据集中的 `chosen`） |
| $y_l$ | loser / rejected | 人类标注员选出的**更差**回复（数据集中的 `rejected`） |
| $\pi_\theta$ | policy | 正在训练的模型，参数为 $\theta$ |
| $\pi_{\text{ref}}$ | reference model | 冻结的 SFT 模型，训练全程不更新，作为 KL 约束的锚点 |
| $\beta$ | KL 惩罚系数 | 控制 policy 偏离 reference 的约束强度（MiniMind 默认 0.15） |
| $\sigma$ | sigmoid 函数 | $\sigma(z) = 1/(1+e^{-z})$，来自 Bradley-Terry 偏好模型，把 reward 差映射到 $[0,1]$ 概率 |

$y_w / y_l$ 的下标来自 Bradley-Terry 的 "winner vs loser" 记法。给定同一个 prompt $x$，标注员比较两条回复，选出 winner（$y_w$）和 loser（$y_l$）：

```
prompt x: "什么是机器学习？"

y_w (chosen):   "机器学习是 AI 的一个分支，通过从数据中学习模式来做预测..."
y_l (rejected): "机器学习就是让电脑自己学。"
```

$\beta$ 的直觉——它在 $\sigma(\beta \cdot \ldots)$ 中做缩放：

- $\beta$ **大**（如 0.5）：$\beta \times \text{logits}$ 被放大 → sigmoid 容易饱和 → 对 policy 偏离 reference 的惩罚更强 → 训练保守
- $\beta$ **小**（如 0.05）：$\beta \times \text{logits}$ 被缩小 → sigmoid 在线性区，梯度大 → 偏好学习更激进，但易过拟合

这个 $\beta$ 和 2.1 节 RLHF 目标函数里 KL 惩罚前面的 $\beta$ 是**同一个数学量**——DPO 推导的起点就是 $\max \mathbb{E}[r] - \beta \cdot D_{\text{KL}}$，闭式解反解后 $\beta$ 自然保留在 loss 里。

$\sigma$ 来自 2.4 节代入的 Bradley-Terry 模型：$P(y_w \succ y_l) = \sigma(r_w - r_l)$。DPO 用 $\beta \log(\pi_\theta / \pi_{\text{ref}})$ 替换了 $r$，所以 $\sigma$ 里面的参数变成了两个隐式 reward 的差值。外层取 $-\log$ 就是标准的负对数似然——最大化"正确排序 $y_w > y_l$"的概率。

定义 log-ratio：

$$\Delta(x, y) = \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

则 DPO Loss 简化为：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\big[\log \sigma\big(\beta \cdot (\Delta(x, y_w) - \Delta(x, y_l))\big)\big]$$

**直觉**：
- $\Delta(x, y_w) > \Delta(x, y_l)$：policy 相对于 reference 更偏好 chosen → sigmoid > 0.5 → loss 小 ✓
- $\Delta(x, y_w) < \Delta(x, y_l)$：policy 相对于 reference 更偏好 rejected → sigmoid < 0.5 → loss 大 ✗
- 梯度方向：增大 $\Delta(x, y_w)$、减小 $\Delta(x, y_l)$ → policy 越来越偏好 chosen

### 2.6 DPO 的梯度——隐式 reward 和隐式 KL

对 DPO loss 求梯度（推导从略），可以得到：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \mathbb{E} \left[\underbrace{\sigma\big(-\beta(\Delta_w - \Delta_l)\big)}_{\text{动态权重}} \left(\underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{提高 chosen 概率}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{降低 rejected 概率}}\right)\right]$$

$\sigma(-\beta(\Delta_w - \Delta_l))$ 是一个动态权重：

- 当 policy 已经正确排序（$\Delta_w > \Delta_l$）→ $\sigma$ 接近 0 → 梯度很小 → "已经学好了，不用再推"
- 当 policy 排序错误（$\Delta_w < \Delta_l$）→ $\sigma$ 接近 1 → 梯度大 → "排错了，大力纠正"

这个动态权重是**对排序错误程度的加权**——排错越狠，纠正梯度越大；已经排对就几乎不再推。

注意区分两件事：
- **动态权重 $\sigma(-\cdots)$**：控制梯度大小——"这个样本还需不需要学"，是训练效率机制
- **隐式 KL 约束**：来自 $\Delta$ 里的 $\pi_{\text{ref}}$ 项——$\Delta = \log \pi_\theta - \log \pi_{\text{ref}}$，reward 以 reference 为零点，policy 偏离 reference 越远，隐式 reward 的绝对值越大，sigmoid 越饱和，更新自然变慢

两者协同但不是同一个东西。面试时说"DPO 的 KL 约束是隐式的，体现在 loss 以 π_ref 为锚点"即可，不要说"动态权重就是 KL"。

---

## Part 3: 偏好数据格式

DPO 需要三元组：(prompt, chosen_response, rejected_response)。

```json
{
  "chosen": [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支，通过数据训练模型..."}
  ],
  "rejected": [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习就是让电脑自己学习。"}
  ]
}
```

`chosen` 和 `rejected` 共享相同的 system + user 部分，只有 assistant 回复不同。`chosen` 是标注员认为更好的回复。

注意：这和训 Reward Model 的数据格式**完全一样**（[上一章](13-RL对齐基础.md) Part 8.2）。区别在于用途：RM 数据训打分模型，DPO 数据直接训 policy。

### DPODataset 的处理

```python
# 写到: scratch/dataset/lm_dataset.py（追加）
def __getitem__(self, index):
    sample = self.samples[index]
    chosen = sample['chosen']        # 完整对话（含更好的回复）
    rejected = sample['rejected']    # 完整对话（含较差的回复）

    # ① 分别渲染成 ChatML 格式并分词
    chosen_prompt = self.tokenizer.apply_chat_template(chosen, ...)
    rejected_prompt = self.tokenizer.apply_chat_template(rejected, ...)
    chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=..., padding='max_length')
    rejected_encoding = self.tokenizer(rejected_prompt, ...)

    # ② 生成 loss mask（只标记 assistant 回复部分）
    chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
    rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

    # ③ 移位：x = input_ids[:-1], y = input_ids[1:]
    x_chosen = torch.tensor(chosen_input_ids[:-1])
    y_chosen = torch.tensor(chosen_input_ids[1:])
    mask_chosen = torch.tensor(chosen_loss_mask[1:])     # mask 也移位

    x_rejected = torch.tensor(rejected_input_ids[:-1])
    y_rejected = torch.tensor(rejected_input_ids[1:])
    mask_rejected = torch.tensor(rejected_loss_mask[1:])

    return {
        'x_chosen': x_chosen, 'y_chosen': y_chosen, 'mask_chosen': mask_chosen,
        'x_rejected': x_rejected, 'y_rejected': y_rejected, 'mask_rejected': mask_rejected
    }
```

DPODataset 在 Dataset 层面就做了移位——`x = input_ids[:-1]`，`y = input_ids[1:]`。这和预训练/SFT 不同（那些是在模型的 forward 里做移位）。原因是 DPO 不走 `model.forward(labels=labels)` 的标准 loss 计算，而是自己算 log probability。

```
DPODataset 返回的 6 个 tensor:

x_chosen:     [token_0, token_1, ..., token_{N-2}]       ← 模型输入
y_chosen:     [token_1, token_2, ..., token_{N-1}]       ← 对应的目标 token
mask_chosen:  [0, 0, ..., 0, 1, 1, ..., 1, 0, 0, ...]   ← 只有 assistant 回复=1

x_rejected:   同上，但 assistant 回复部分不同
y_rejected:   同上
mask_rejected: 同上
```

---

## Part 4: DPO Loss 的代码实现

### 4.1 从 logits 到 per-token log probability

```python
# 写到: scratch/trainer/train_dpo.py
def logits_to_log_probs(logits, labels):
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    log_probs = F.log_softmax(logits, dim=2)                           # [B, S, V]
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)                    # 取出每个位置真实 token 的 log prob
    ).squeeze(-1)                                                      # [B, S]
    return log_probs_per_token
```

对每个位置，取出模型给真实 token 的 log 概率：

```
举例:
  logits[b, t, :] = [1.2, -0.5, 3.1, ..., 0.8]     ← 6400 维 logits
  log_softmax:     [-2.1, -3.8, -0.2, ..., -2.5]    ← 6400 维 log 概率

  labels[b, t] = 42   (真实 token 的 id)
  log_probs_per_token[b, t] = -0.2                   ← token 42 的 log 概率
```

### 4.2 Policy 和 Reference Model 分别前向

```python
# 写到: scratch/trainer/train_dpo.py（训练循环内）
with autocast_ctx:
    # Reference model（冻结）前向
    with torch.no_grad():
        ref_outputs = ref_model(x)
        ref_logits = ref_outputs.logits
    ref_log_probs = logits_to_log_probs(ref_logits, y)    # [B, S]

    # Policy model（训练中）前向
    outputs = model(x)
    logits = outputs.logits
    policy_log_probs = logits_to_log_probs(logits, y)     # [B, S]
```

其中 `x = cat([x_chosen, x_rejected])`，`y = cat([y_chosen, y_rejected])`——chosen 和 rejected 拼在一起做一次前向，然后在 loss 函数里拆开。这是工程上的优化——两次前向合并成一次，减少 kernel launch 开销。

### 4.3 DPO Loss 计算

先理解公式和代码的对应关系：

$$\log \pi_\theta(y \mid x) = \sum_{t: \text{mask}_t = 1} \log \pi_\theta(y_t \mid y_{<t}, x)$$

序列级 log 概率 = 对 assistant 回复部分的 per-token log 概率求和。

```python
# 写到: scratch/trainer/train_dpo.py
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ① 序列级 log 概率（只对 mask=1 的 assistant 部分求和）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)         # [B]
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)   # [B]

    # ② 拆分 chosen 和 rejected（batch 前半是 chosen，后半是 rejected）
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # ③ 计算 log-ratio（对应公式里的 Δ(x, y)）
    #   Δ(x, y_w) = log π_θ(y_w|x) - log π_ref(y_w|x) → chosen 的
    #   但代码里先算的是 π 之间的差（不是 π/π_ref），效果一样：
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs

    # ④ DPO Loss
    #   logits = (Δ_w - Δ_l) 的等价变形
    #   = (log π_w - log π_l) - (log π_ref_w - log π_ref_l)
    #   = (log π_w - log π_ref_w) - (log π_l - log π_ref_l)
    #   = Δ(x, y_w) - Δ(x, y_l)
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

代码中的 `logits` 变量就是公式里的 $\Delta(x, y_w) - \Delta(x, y_l)$，只不过用了等价的代数变形——先在 policy 内做 chosen vs rejected 的差，再和 reference 的差相减。

用具体数字验证等价性：

```
假设:
  log π_θ(y_w|x) = -5.0    log π_ref(y_w|x) = -6.0    Δ_w = -5.0 - (-6.0) = 1.0
  log π_θ(y_l|x) = -4.0    log π_ref(y_l|x) = -4.0    Δ_l = -4.0 - (-4.0) = 0.0

公式路径: Δ_w - Δ_l = 1.0 - 0.0 = 1.0

代码路径:
  pi_logratios  = -5.0 - (-4.0) = -1.0
  ref_logratios = -6.0 - (-4.0) = -2.0
  logits = -1.0 - (-2.0) = 1.0  ✓ 一样

loss = -log sigmoid(0.15 × 1.0) = -log sigmoid(0.15) = -log(0.537) = 0.622
含义: policy 正确地偏好 chosen（logits > 0），loss 适中
```

---

## Part 5: Reference Model 的作用

### 5.1 为什么需要 Reference Model

```python
# 写到: scratch/trainer/train_dpo.py（主函数内）
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)    # 完全冻结，不参与训练
```

Reference model = SFT 后的模型（和 policy 初始权重相同，但训练过程中不更新）。它在 DPO loss 中的作用等价于 RLHF 中的 **KL 惩罚**。

```
没有 reference model（只看 policy 的 log prob）:
  loss = -log sigmoid(beta × (log π(y_w) - log π(y_l)))
  问题: policy 可能通过给 y_w 极高概率来最小化 loss
        → 生成高度重复、模板化的回复（mode collapse）
        → 丢失语言多样性

有 reference model:
  loss = -log sigmoid(beta × ([log π(y_w) - log π_ref(y_w)] - [log π(y_l) - log π_ref(y_l)]))
  效果: policy 需要在 "相对于 reference 更偏好 chosen" 的方向上优化
        不能简单地给 chosen 极高概率——因为 reference 也会给它高概率
        → 只学 "chosen 和 rejected 之间的差异"，不破坏语言能力
```

### 5.2 $\beta$ 参数

```python
# 写到: scratch/trainer/train_dpo.py（argparse 参数）
parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
```

$\beta$ 控制 reference 约束的强度——和 RLHF 中 KL 惩罚的 $\beta$ 是同一个东西：

| $\beta$ | 效果 |
|---------|------|
| 大（如 0.5） | 强约束，policy 不能偏离 reference 太远 → 保守，不易过拟合 |
| 小（如 0.05） | 弱约束，policy 可以大幅偏离 → 偏好学习效果强，但易过拟合 |
| MiniMind 默认 0.15 | 适中 |

### 5.3 显存代价

DPO 需要同时加载 policy 和 reference——相比纯 SFT 显存翻倍，但比 RLHF（5 个模型）少很多：

```
MiniMind (64M):
  policy:    128 MB (BF16)
  reference: 128 MB (BF16)
  总共: 256 MB + 优化器状态 + 激活值 ≈ 1 GB

7B 模型:
  policy:    14 GB（推理态权重）；训练态按 P×16 ≈ 112 GB（含梯度 + Adam）
  reference: 14 GB（只算推理态权重）
  总共: ≈ 126 GB → 单卡 A100 80GB 放不下，需要多卡 / ZeRO

对比 RLHF: 5 个模型 ≈ 266 GB（见 13 章）→ DPO 省了一半以上
```

---

## Part 6: 训练超参数

### DPO vs SFT 的超参数差异

| 超参数 | 全参 SFT | DPO | 为什么变了 |
|--------|---------|-----|-----------|
| `learning_rate` | 1e-5 | 4e-8 | 极低——DPO 只做微调整，太高会导致偏好过拟合 |
| `batch_size` | 16 | 4 | 每个样本含 chosen + rejected（等效 8 条） |
| `epochs` | 2 | 1 | DPO 数据量小，1 epoch 通常够了 |
| `max_seq_len` | 768 | 1024 | 两条回复需要更长空间 |
| `from_weight` | pretrain | full_sft | 必须从 SFT 权重开始 |

MiniMind 的 DPO 学习率是 **4e-8**——比 SFT 低了 250 倍。这是小模型 + 小偏好数据集下防遗忘的保守取值。

业界 7B+ 模型的 DPO 学习率通常在 $10^{-7} \sim 10^{-5}$ 量级（如 LLaMA 3 论文用 $10^{-5}$），因为大模型的参数空间更大，同样的学习率对权重的相对扰动更小。

---

## Part 7: DPO 的局限性

### 7.1 分布偏移（Distribution Shift）

DPO 用的是**离线**偏好数据——数据是在训练前收集好的，来自 SFT 模型（或人类标注员）的分布。但随着训练进行，policy 的分布会发生变化：

$$\text{训练开始: } \pi_\theta \approx \pi_{\text{ref}} \quad \xrightarrow{\text{训练}} \quad \pi_\theta \neq \pi_{\text{ref}}$$

问题：偏好数据中的 chosen 和 rejected 是从旧分布采样的，新策略可能根本不会生成这些回复。模型在"自己不会说的话"上学偏好，学到的信号可能不可靠。

### 7.2 Chosen/Rejected 质量差距

当 chosen 和 rejected 质量差距太大时（比如 chosen 是专家写的，rejected 是随机生成的），模型可能学到的是"避免 rejected 的表面特征"（比如避免短回复），而不是"学习 chosen 的内在质量"。

### 7.3 无法探索

DPO 是纯离线的——模型不会在训练过程中生成新的回复来试错。这意味着它无法发现训练数据覆盖不到的好策略。这是 DPO 和 GRPO 的根本区别。

### 7.4 业界的缓解方案

| 方案 | 做法 | 代表工作 |
|------|------|---------|
| Iterative DPO | 每轮用当前 policy 生成新的 rejected，迭代做 DPO | Qwen2.5 |
| Online DPO | 训练过程中实时生成回复 + 打分 | OAIF (Guo et al., 2024) |
| SimPO | 去掉 reference model，用长度归一化的 log prob 做隐式 reward | Meng et al., 2024 |
| DPO → GRPO | 直接换成在线 RL | DeepSeek-V3 |

---

## 完整实现

把本章的代码整合到 scratch/ 目录。

**文件 1：`scratch/dataset/lm_dataset.py`（追加 DPODataset）**

在 12 章已有的 `SFTDataset` 之后追加：

```python
# 写到: scratch/dataset/lm_dataset.py（追加）

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']
        rejected = sample['rejected']

        # ① 分别渲染成 ChatML 格式并分词
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # ② 移位：x = input_ids[:-1], y = input_ids[1:]
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen, 'y_chosen': y_chosen, 'mask_chosen': mask_chosen,
            'x_rejected': x_rejected, 'y_rejected': y_rejected, 'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """标记 assistant 回复部分为 1，其余为 0"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
```

**文件 2：`scratch/trainer/train_dpo.py`（完整训练脚本）**

包含 `logits_to_log_probs`、`dpo_loss`、`train_epoch` 和 `__main__` 四部分——本章 Part 4-6 的代码全部写到这个文件。完整训练循环参考上游 `trainer/train_dpo.py`，核心逻辑已在 Part 4 和 Part 5 逐步讲解。

```python
# 写到: scratch/trainer/train_dpo.py

import os, sys, time, argparse
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import DPODataset
from transformers import AutoTokenizer
from trainer_utils import get_lr, Logger


# ========== Part 4: 从 logits 到 per-token log prob ==========

def logits_to_log_probs(logits, labels):
    """从模型输出的 logits 中提取每个 token 的 log 概率。

    logits:  [batch, seq_len, vocab_size]  模型对每个位置的全词表打分
    labels:  [batch, seq_len]              目标 token id（用来"挑"对应的 log prob）
    返回:    [batch, seq_len]              每个位置目标 token 的 log 概率
    """
    # 沿词表维度做 log_softmax，得到归一化的 log 概率分布
    log_probs = F.log_softmax(logits, dim=2)               # [batch, seq_len, vocab_size]

    # gather: 用 labels 作为索引，从 vocab_size 维度中"挑出"对应 token 的 log prob
    # labels.unsqueeze(2) 把 [batch, seq_len] → [batch, seq_len, 1]，匹配 gather 的 index 要求
    # squeeze(-1) 把结果从 [batch, seq_len, 1] → [batch, seq_len]
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)                                           # [batch, seq_len]

    return log_probs_per_token


# ========== Part 5: DPO Loss ==========

def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """计算 DPO Loss。

    核心公式：L_DPO = -log σ(β · (log π(y_w|x)/π(y_l|x) - log π_ref(y_w|x)/π_ref(y_l|x)))

    输入的 ref_log_probs / policy_log_probs 是 per-token 的 log 概率，shape: [batch, seq_len]
    batch 的前半是 chosen，后半是 rejected（在 train_epoch 里用 torch.cat 拼起来的）
    mask: [batch, seq_len]，只对 assistant 回答部分计算 loss
    """
    # ① 对每个样本，把 token 级 log prob 按 mask 汇总成序列级 log prob
    #    mask=0 的位置（system/user prompt）不参与 loss 计算
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)       # [batch]
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) # [batch]

    # ② 分离 chosen 和 rejected（前半 chosen，后半 rejected）
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # ③ 计算 policy 和 reference 的 log ratio
    #    pi_logratios = log π(y_w|x) - log π(y_l|x)
    #    ref_logratios = log π_ref(y_w|x) - log π_ref(y_l|x)
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs

    # ④ DPO 的核心：隐式 reward 差 = policy ratio - reference ratio
    #    这就是 Part 3 推导的 r(x,y) = β·log(π/π_ref) 的差值形式
    logits = pi_logratios - ref_logratios

    # ⑤ Bradley-Terry: -log σ(β · Δr)
    loss = -F.logsigmoid(beta * logits)

    return loss.mean()


# ========== Part 6: 训练循环 ==========

def train_epoch(model, ref_model, loader, optimizer, scaler, autocast_ctx, epoch, args):
    """一个 epoch 的 DPO 训练。

    核心流程：
    1. 从 DPODataset 取 chosen/rejected 对
    2. 拼成一个 batch 一起过模型（省一次 forward）
    3. ref_model 冻结推理 → ref_log_probs
    4. policy model 正常推理 → policy_log_probs
    5. dpo_loss 计算 loss 并反向传播
    """
    model.train()
    iters = len(loader)
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        # ---- 数据准备：chosen 和 rejected 拼成一个 batch ----
        x_chosen = batch['x_chosen'].to(args.device)       # [B, seq_len]
        x_rejected = batch['x_rejected'].to(args.device)   # [B, seq_len]
        y_chosen = batch['y_chosen'].to(args.device)        # [B, seq_len]
        y_rejected = batch['y_rejected'].to(args.device)    # [B, seq_len]
        mask_chosen = batch['mask_chosen'].to(args.device)   # [B, seq_len]
        mask_rejected = batch['mask_rejected'].to(args.device)

        # 把 chosen 和 rejected 拼在一起，一次 forward 搞定
        # 拼完后 x 的前半是 chosen、后半是 rejected，dpo_loss 里按这个约定拆开
        x = torch.cat([x_chosen, x_rejected], dim=0)       # [2B, seq_len]
        y = torch.cat([y_chosen, y_rejected], dim=0)        # [2B, seq_len]
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # ---- LR 调度（cosine decay，和预训练一样） ----
        current_lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        with autocast_ctx:
            # ---- Reference model：冻结推理，提供 baseline ----
            # no_grad: ref_model 不参与梯度计算，只提供"SFT 模型会怎么回答"的参照
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits             # [2B, seq_len, vocab_size]
            ref_log_probs = logits_to_log_probs(ref_logits, y)  # [2B, seq_len]

            # ---- Policy model：正常 forward，梯度会回传 ----
            outputs = model(x)
            logits = outputs.logits                         # [2B, seq_len, vocab_size]
            policy_log_probs = logits_to_log_probs(logits, y)   # [2B, seq_len]

            # ---- 计算 DPO Loss ----
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=args.beta)
            # aux_loss: MOE 架构的负载均衡 loss（非 MOE 时为 0）
            loss = dpo_loss_val + outputs.aux_loss
            loss = loss / args.accumulation_steps

        # ---- 反向传播 + 梯度累积 ----
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ---- 日志 ----
        if step % args.log_interval == 0 or step == iters:
            real_loss = loss.item() * args.accumulation_steps
            eta_min = (time.time() - start_time) / step * (iters - step) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                   f'loss:{real_loss:.4f} dpo_loss:{dpo_loss_val.item():.4f} '
                   f'lr:{current_lr:.8f} eta:{eta_min:.1f}min')

        # ---- 保存 checkpoint ----
        if step % args.save_interval == 0 or step == iters:
            model.eval()
            moe_suffix = '_moe' if args.use_moe else ''
            ckp = f'{args.save_dir}/dpo_{args.hidden_size}{moe_suffix}.pth'
            torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, ckp)
            model.train()

        # 及时释放显存
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected
        del x, y, mask, ref_outputs, ref_logits, ref_log_probs
        del outputs, logits, policy_log_probs, loss

    # epoch 末尾：处理残余累积梯度（和预训练一样）
    if iters % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


# ========== __main__: 入口 ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO Training")
    parser.add_argument("--save_dir", default="../../out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=4e-8,
                        help="极低学习率，防止 DPO 训练遗忘 SFT 能力")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="训练的最大截断长度")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--data_path", default="../../dataset/dpo.jsonl")
    parser.add_argument("--from_weight", default="full_sft",
                        help="基于哪个权重训练（DPO 必须从 SFT 权重开始）")
    parser.add_argument("--beta", type=float, default=0.15,
                        help="DPO 的 β 参数，控制偏离 reference model 的惩罚力度")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- 1. 模型配置 ----
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    # ---- 2. 加载权重的辅助函数 ----
    def load_model(config, weight_name, device):
        """加载模型 + SFT 权重"""
        model = MiniMindForCausalLM(config)
        moe_suffix = '_moe' if config.use_moe else ''
        weight_path = f'{args.save_dir}/{weight_name}_{config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        return model.to(device)

    # ---- 3. 策略模型（要训练的）和参考模型（冻结的） ----
    model = load_model(lm_config, args.from_weight, args.device)
    Logger(f'Policy model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    # ref_model: 冻结的 SFT 模型，DPO 用它当 baseline
    # 为什么要冻结？DPO Loss 需要一个固定的 π_ref 来计算 KL 散度隐式约束
    # 如果 ref_model 也在变，loss 的含义就不对了
    ref_model = load_model(lm_config, args.from_weight, args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)     # 冻结所有参数，不参与梯度计算
    Logger(f'Reference model params: {sum(p.numel() for p in ref_model.parameters()) / 1e6:.2f}M')

    # ---- 4. 数据集和 DataLoader ----
    tokenizer = AutoTokenizer.from_pretrained('../../model')
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # ---- 5. 优化器和混合精度 ----
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (nullcontext() if "cpu" in args.device
                    else torch.amp.autocast('cuda', dtype=dtype))
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # ---- 6. 开始训练 ----
    for epoch in range(args.epochs):
        train_epoch(model, ref_model, loader, optimizer, scaler, autocast_ctx, epoch, args)
```

---

## 对照上游源码

| 概念 | 我们写到 scratch/ 的 | 上游源码位置 |
|------|---------------------|-------------|
| 偏好数据格式 | `scratch/dataset/lm_dataset.py` DPODataset | `dataset/lm_dataset.py` DPODataset |
| Loss mask | `generate_loss_mask` | `dataset/lm_dataset.py` generate_loss_mask |
| log prob 计算 | `logits_to_log_probs` | `trainer/train_dpo.py` logits_to_log_probs |
| DPO Loss | `dpo_loss` | `trainer/train_dpo.py` dpo_loss |
| Reference model | 冻结的 SFT 模型 | `trainer/train_dpo.py` init_model + freeze |
| beta | 0.15 | `trainer/train_dpo.py` argparse |
| 学习率 | 4e-8 | `trainer/train_dpo.py` argparse |
| Chosen/Rejected 拼接 | `cat([x_chosen, x_rejected])` | `trainer/train_dpo.py` train_epoch |

---

## 业界对比

| 特性 | MiniMind DPO | LLaMA 3 | DeepSeek-V3 | Qwen2.5 |
|------|-------------|---------|-------------|---------|
| 对齐方法 | DPO | Rejection Sampling + DPO | GRPO（不用 DPO） | Iterative DPO + GRPO |
| $\beta$ | 0.15 | 0.1（论文） | N/A | 未公开 |
| 学习率 | 4e-8 | 1e-5（论文） | N/A | 未公开 |
| Reference model | SFT 权重冻结 | SFT 权重冻结 | N/A | SFT 权重冻结 |
| 偏好数据量 | ~几千条 | ~数十万条 | ~数十万条 | ~数十万条 |
| 变体 | 标准 DPO | DPO + Rejection Sampling | GRPO | Iterative DPO |

几个值得注意的趋势：

1. **DPO 是 RLHF 的最简替代**——去掉了 RM 和 PPO，实现简单得多
2. **DeepSeek-V3 选择了 GRPO 而不是 DPO**——在线采样（on-policy）+ 组内比较更适合推理任务，能持续用当前策略的最新分布训练
3. **Iterative DPO 是 DPO 和在线 RL 的折中**——每轮用当前 policy 生成新数据，部分解决分布偏移

---

## 面试话术

> "DPO 是对传统 RLHF 的数学简化。核心洞察是：受 KL 约束的 reward 最大化问题有闭式解，最优 reward 可以写成 beta × log(pi/pi_ref) + 常数。把这个代入 Bradley-Terry 偏好模型后，reward model 完全消失——偏好概率只取决于 policy 和 reference 的 log 概率比。
>
> 所以 DPO loss = -log sigmoid(beta × (Delta_chosen - Delta_rejected))，其中 Delta = log(pi/pi_ref)。KL 约束是隐式的——体现在 Delta 以 pi_ref 为锚点，policy 偏离 reference 越远，隐式 reward 越饱和，更新自然变慢。梯度还有一个动态权重 sigma(-...)，已经排对的样本梯度自动减小，排错的大力纠正。
>
> 实现上，chosen 和 rejected 拼成一个 batch 做一次前向，拆开算 log-ratio。学习率极低（MiniMind 用 4e-8），防止破坏 SFT 学到的对话能力。
>
> 主要局限是分布偏移——DPO 是离线的，训练数据来自旧策略的分布。Iterative DPO 通过每轮重新生成数据来缓解，但根本解决需要在线 RL，这就是 GRPO 的路线。"

# 13 - DPO：不训 Reward Model 的偏好对齐

> 源码位置：`trainer/train_dpo.py`（训练主循环 + DPO loss 函数）、`dataset/lm_dataset.py:122-192`（DPODataset）
> 依赖：`model/model_minimind.py:245-253`（模型 forward）
> 前置：[12-SFT](12-SFT.md)（SFT 的 loss mask）、[10-训练阶段论](10-训练阶段论.md)（RLHF 阶段定位）

## 一句话总结

DPO（Direct Preference Optimization）把传统 RLHF 中"训 Reward Model → 用 RL 优化 Policy"两步压缩成一步：**直接用偏好数据（chosen/rejected 对）优化策略模型**，省掉了 reward model 的训练和推理开销。核心公式是对 chosen 和 rejected 的 log-ratio 做 sigmoid loss。

---

## Part 1: 从 RLHF 到 DPO——为什么要绕开 Reward Model

### 传统 RLHF 的流程

```
传统 RLHF（InstructGPT 风格）:

① 训 Reward Model:
   偏好数据 → 训练一个打分模型 → RM 可以给任意回复打分

② 用 PPO 优化 Policy:
   prompt → policy 模型生成回复 → RM 打分 → PPO 更新 policy
   同时加 KL 惩罚防止 policy 偏离 SFT 模型太远

问题:
  - 需要单独训练一个 RM（额外的模型、数据、算力）
  - PPO 训练不稳定（reward hacking、KL 散度调参）
  - 推理时 RM 也要常驻显存
```

### DPO 的简化

Rafailov et al. (2023) 发现：**最优的 reward function 可以用 policy 的 log 概率直接表达**，不需要显式训练一个 reward model。

```
DPO 核心洞察:

最优 reward:  r*(x, y) = beta * log(pi(y|x) / pi_ref(y|x)) + 常数

含义: 一个回复的 "好坏" 可以通过 policy 和 reference model 的概率比来衡量
     如果 policy 比 reference 更 "喜欢" 这个回复（概率更高），reward 就高

把这个代入 Bradley-Terry 偏好模型:
P(y_w > y_l | x) = sigmoid(r*(y_w) - r*(y_l))
                  = sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))

DPO Loss = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
```

---

## Part 2: DPO 的数据格式

### 偏好数据

DPO 需要三元组：(prompt, chosen_response, rejected_response)

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

`chosen` 和 `rejected` 共享相同的 system + user 部分，只有 assistant 回复不同。`chosen` 是人类标注员认为更好的回复，`rejected` 是相对较差的。

### DPODataset 的处理

```python
# dataset/lm_dataset.py:135-174
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

注意 DPODataset 在 Dataset 层面就做了移位——`x = input_ids[:-1]`，`y = input_ids[1:]`。这和预训练/SFT 不同（那些是在模型的 forward 里做移位）。原因是 DPO 不走 `model.forward(labels=labels)` 的标准 loss 计算，而是自己算 log probability。

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

## Part 3: DPO Loss 的代码实现

### 第一步：从 logits 到 per-token log probability

```python
# train_dpo.py:25-31
def logits_to_log_probs(logits, labels):
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]
    log_probs = F.log_softmax(logits, dim=2)                           # [B, S, V]
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)                    # 取出每个位置真实 token 的 log prob
    ).squeeze(-1)                                                      # [B, S]
    return log_probs_per_token
```

这个函数对每个位置，取出模型给真实 token 的 log 概率。

```
举例:
  logits[b, t, :] = [1.2, -0.5, 3.1, ..., 0.8]     ← 6400 维 logits
  log_softmax:     [-2.1, -3.8, -0.2, ..., -2.5]    ← 6400 维 log 概率

  labels[b, t] = 42   (真实 token 的 id)
  log_probs_per_token[b, t] = -0.2                   ← token 42 的 log 概率
```

### 第二步：policy 和 reference model 分别前向

```python
# train_dpo.py:73-81
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

其中 `x = cat([x_chosen, x_rejected])`，`y = cat([y_chosen, y_rejected])`——chosen 和 rejected 拼在一起做一次前向传播，然后在 loss 函数里拆开。

### 第三步：DPO Loss 计算

#### 先理解公式，再看代码

```
步骤 ①: 序列级 log 概率
  log P(y|x) = sum_t log P(y_t | y_{<t}, x)
  只对 mask=1 的位置求和（assistant 回复部分）

步骤 ②: 拆分 chosen / rejected
  batch 排布: [chosen_0, chosen_1, ..., rejected_0, rejected_1, ...]
  前 B/2 个是 chosen，后 B/2 个是 rejected

步骤 ③: log-ratio
  pi_logratios  = log pi(y_w|x) - log pi(y_l|x)          ← policy 对 chosen vs rejected 的偏好
  ref_logratios = log pi_ref(y_w|x) - log pi_ref(y_l|x)  ← reference model 的偏好

步骤 ④: DPO Loss
  logits = pi_logratios - ref_logratios
         = [log pi(y_w|x) - log pi_ref(y_w|x)] - [log pi(y_l|x) - log pi_ref(y_l|x)]
         = log(pi(y_w|x) / pi_ref(y_w|x)) - log(pi(y_l|x) / pi_ref(y_l|x))

  loss = -log sigmoid(beta * logits)

  直觉:
  当 policy 比 reference 更偏好 chosen（logits > 0）→ sigmoid > 0.5 → loss 小
  当 policy 比 reference 更偏好 rejected（logits < 0）→ sigmoid < 0.5 → loss 大
  梯度会推动 policy 更偏好 chosen，同时不偏离 reference 太远
```

#### 对照公式看代码

```python
# train_dpo.py:34-50
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ① 对 assistant 回复部分的 log prob 求和（序列级 log 概率）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)         # [B]
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)   # [B]

    # ② 拆分 chosen 和 rejected（batch 前半是 chosen，后半是 rejected）
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # ③ 计算 log-ratio
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs

    # ④ DPO Loss
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

---

## Part 4: Reference Model 的作用

### 为什么需要 reference model

```python
# train_dpo.py:186-188
ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
ref_model.eval()
ref_model.requires_grad_(False)    # 完全冻结，不参与训练
```

Reference model = SFT 后的模型（和 policy model 初始权重相同，但训练过程中不更新）。它的作用是**防止 policy 偏离太远**——这等价于传统 RLHF 中的 KL 惩罚项。

```
没有 reference model:
  loss = -log sigmoid(beta * (log pi(y_w) - log pi(y_l)))
  问题: policy 可能通过给 y_w 极高概率来最小化 loss
        → 生成高度重复、模板化的回复（reward hacking）
        → 丢失语言多样性

有 reference model:
  loss = -log sigmoid(beta * ([log pi(y_w) - log pi_ref(y_w)] - [log pi(y_l) - log pi_ref(y_l)]))
  效果: policy 需要在 "相对于 reference 更偏好 chosen" 的方向上优化
        不能简单地给 chosen 极高概率——因为 reference 也会给它高概率
        → 只学 "chosen 和 rejected 之间的差异"，不破坏语言能力
```

### beta 参数的含义

```python
# train_dpo.py:152
parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
```

beta 控制 reference model 约束的强度：

```
beta 大（如 0.5）: 强约束，policy 不能偏离 reference 太远
                   → 保守，不容易过拟合，但偏好学习效果弱
beta 小（如 0.05）: 弱约束，policy 可以大幅偏离 reference
                    → 激进，偏好学习效果强，但容易过拟合
MiniMind 默认 beta=0.15: 适中
```

### 显存代价

DPO 需要同时在显存中保留 policy model 和 reference model——显存占用翻倍：

```
MiniMind (64M):
  policy model:    128 MB (BF16)
  reference model: 128 MB (BF16)
  总共:            256 MB + 优化器状态 + 激活值 ≈ 1 GB

7B 模型:
  policy model:    14 GB
  reference model: 14 GB
  总共:            28 GB + 优化器状态 ≈ 60 GB → 需要 A100 80GB
```

---

## Part 5: 训练超参数

### DPO 的超参数特点

| 超参数 | 全参 SFT | DPO | 为什么变了 |
|--------|---------|-----|-----------|
| `learning_rate` | 1e-5 | 4e-8 | 极低——DPO 只做微调整，太高会导致偏好过拟合 |
| `batch_size` | 16 | 4 | 每个样本包含 chosen + rejected（等效 8 条） |
| `epochs` | 2 | 1 | DPO 数据量小，1 epoch 通常够了 |
| `max_seq_len` | 768 | 1024 | 两条回复需要更长空间 |
| `from_weight` | pretrain | full_sft | 必须从 SFT 权重开始 |

注意 MiniMind 的 DPO 学习率是 **4e-8**——比它的 SFT 低了 250 倍。（这是小模型 + 小偏好数据集下防遗忘的保守取值；业界 7B+ 模型的 DPO 学习率通常在 1e-7~1e-5 量级，如 LLaMA 3 论文用 1e-5。）这是因为 DPO 是在已经对齐的模型上做精细调整，太大的学习率会破坏 SFT 学到的对话能力。

---

## 对照 MiniMind 源码

| 概念 | 我们讲的 | 源码位置 |
|------|---------|---------|
| 偏好数据格式 | chosen / rejected | `lm_dataset.py:122-174` |
| Loss mask | `generate_loss_mask` | `lm_dataset.py:176-192` |
| log prob 计算 | `logits_to_log_probs` | `train_dpo.py:25-31` |
| DPO Loss | `dpo_loss` | `train_dpo.py:34-50` |
| Reference model | 冻结的 SFT 模型 | `train_dpo.py:186-188` |
| beta | 0.15 | `train_dpo.py:152` |
| 学习率 | 4e-8 | `train_dpo.py:137` |
| Chosen/Rejected 拼接 | `cat([x_chosen, x_rejected])` | `train_dpo.py:65-67` |

---

## 业界对比

| 特性 | MiniMind DPO | LLaMA 3 | DeepSeek-V3 | Qwen2.5 |
|------|-------------|---------|-------------|---------|
| 对齐方法 | DPO | Rejection Sampling + DPO | GRPO（不用 DPO） | DPO + GRPO |
| beta | 0.15 | 0.1（论文） | N/A | 未公开 |
| 学习率 | 4e-8 | 1e-5（论文） | N/A | 未公开 |
| Reference model | SFT 权重冻结 | SFT 权重冻结 | N/A | SFT 权重冻结 |
| 偏好数据量 | ~几千条 | ~数十万条 | ~数十万条 | ~数十万条 |
| 变体 | 标准 DPO | DPO + Rejection Sampling | GRPO | Iterative DPO |

几个值得注意的点：

1. **DPO 是 RLHF 的简化替代**——去掉了 reward model 和 PPO，实现简单得多
2. **DeepSeek-V3 选择了 GRPO 而不是 DPO**——官方未说明原因；一般认为在线采样（on-policy）+ 组内比较更适合推理类任务，能持续用当前策略的最新分布训练，而 DPO 是离线的、受限于静态偏好数据的分布
3. **Iterative DPO**：Qwen 等模型用多轮 DPO——每轮用当前 policy 重新生成 rejected，然后做一轮 DPO，迭代几轮。这解决了 DPO 的"分布偏移"问题
4. DPO 的一个已知缺陷：**chosen 和 rejected 的质量差距太大时**，模型可能学到"避免 rejected 的特征"而不是"学习 chosen 的特征"（overfitting to rejected）

---

## 面试话术

> "DPO 是对传统 RLHF 的简化——Rafailov 等人 2023 年证明，最优 reward 可以用 policy 和 reference model
> 的 log 概率比直接表达，不需要单独训练 reward model。
>
> 核心公式是 loss = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))，
> 其中 log_ratio = log pi(y|x) - log pi_ref(y|x)，衡量 policy 相对于 reference 的偏好强度。
> 梯度方向是让 policy 对 chosen 的概率提高、对 rejected 的概率降低，同时通过 reference model 约束
> policy 不偏离太远。
>
> 实现上，chosen 和 rejected 拼成一个 batch 做一次前向，然后拆开算 log-ratio。
> 学习率极低（4e-8），因为只做精细的偏好调整，不能破坏 SFT 学到的对话能力。
> 缺点是显存翻倍（需要同时加载 policy 和 reference），以及离线数据可能存在分布偏移。
> DeepSeek 选择了 GRPO 绕开了这两个问题。"

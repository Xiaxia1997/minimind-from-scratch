# 15 - GRPO：扔掉 Critic 的强化学习

> **Chapter 14 写了 DPO，本章：从零写 GRPO 训练循环**
>
> ```
> 本章产出文件:
> scratch/trainer/train_grpo.py       ← GRPO 训练主循环 + reward + loss
> scratch/trainer/rollout_engine.py   ← Rollout 推理引擎（生成回复 + 记录 logprob）
> scratch/dataset/lm_dataset.py      ← RLAIFDataset（追加到已有文件）
> ```
>
> 依赖：`scratch/model/model_minimind.py`（已写好的模型）、`trainer/trainer_utils.py`（LMForRewardModel，直接复用）
> 前置：[13-RL对齐基础](13-RL对齐基础.md)（PPO 的 ratio + clip + advantage）、[14-DPO](14-DPO.md)（DPO 的偏好对齐）

## 一句话总结

GRPO 保留了 PPO 的 ratio + clip + KL，但**扔掉了 Critic（Value Model）**，改用同一个 prompt 下 $G$ 个回复的组内 reward 归一化来算 advantage。省了一个模型的显存，但 advantage 从 per-token 退化成 per-response。

---

## Part 1: 动机——PPO 在 LLM 场景的问题

[13 章](13-RL对齐基础.md) Part 7 详细讲了 PPO 需要 4 个模型（Policy + Old Policy + Critic + Reference），加上 Reward Model 共 5 个。GRPO 的出发点就是**能不能省掉 Critic**。

```
PPO 的 Critic 问题:

1. 显存
   PPO 显存 = Actor + Critic + Ref + RM ≈ 4× 模型大小
   7B: 4 × 14GB = 56GB 还只是推理态权重；Actor/Critic 要训练，
       按 P×16 口径各 ~112GB，合计远超单卡 → 需要大量 A100 + ZeRO

2. Critic 难训准
   LLM 的状态空间（token 序列）维度巨大，V(s) 很难学
   Critic 估不准 → advantage 不准 → actor 更新方向错

3. Actor-Critic 互相干扰
   共享 backbone 时两个目标打架
   分开训则显存再翻倍
```

GRPO（Group Relative Policy Optimization，DeepSeekMath 2024）的解法：**不用 Critic 估 value baseline，用"同一道题多答几次"的组内统计量替代**。

---

## Part 2: GRPO 的核心思想——组内 Z-Score

### 2.1 从 PPO 的 Advantage 到 GRPO 的 Advantage

回顾 [13 章](13-RL对齐基础.md) 的演进链：PPO 的 advantage 是 per-token 的 $\hat{A}_t^{\text{GAE}} = \delta_t + \gamma\lambda \hat{A}_{t+1}$，需要 Critic $V(s_t)$。

GRPO 换了一个完全不同的思路：

1. 对每个 prompt $q$，用旧策略采样 $G$ 个回答 $o_1, o_2, \dots, o_G$
2. Reward model（或规则）给每个回答打分 $r_1, r_2, \dots, r_G$
3. Advantage 用**组内 z-score**：

$$\boxed{\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1, \dots, r_G)}{\operatorname{std}(r_1, \dots, r_G) + \epsilon}}$$

```
具体例子: G=4, prompt="2+3=?"

o_1 = "答案是5" → r_1 = 1.0
o_2 = "答案是4" → r_2 = 0.0
o_3 = "2+3=5"   → r_3 = 1.0
o_4 = "答案是6" → r_4 = 0.0

mean = 0.5, std = 0.5
A_1 = (1.0 - 0.5) / 0.5 = +1.0   ← 比组内平均好
A_2 = (0.0 - 0.5) / 0.5 = -1.0   ← 比组内平均差
A_3 = (1.0 - 0.5) / 0.5 = +1.0
A_4 = (0.0 - 0.5) / 0.5 = -1.0
```

**直觉**：组均值 $\mu$ 就是 value baseline 的替代品——"这道题平均能答多好"。Z-score 就是"这个回答比平均水平好（或差）多少个标准差"。

### 2.2 Advantage 粒度的根本差异

| | PPO | GRPO |
|---|-----|------|
| Advantage 来源 | Critic + GAE | 组内 z-score |
| 粒度 | **per-token**：每个 token 独立的 $\hat{A}_t$ | **per-response**：同一个回答的所有 token 共享 $\hat{A}_i$ |
| 需要 Critic | 需要 | 不需要 |
| 显存 | Actor + Critic + Ref | Actor + Ref |

**per-response advantage 的含义**：第 $i$ 个回答的所有 token，不管是第 1 个还是第 100 个，都用同一个 $\hat{A}_i$。这对 credit assignment 来说是一个退步——但对 math reasoning 来说通常没问题（Part 5 详细讨论）。

---

## Part 3: GRPO Loss 公式

### 3.1 完整公式

$$\boxed{\mathcal{L}^{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^{G}} \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\!\big(r_{i,t}(\theta) \cdot \hat{A}_i, \;\operatorname{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \cdot \hat{A}_i\big) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]}$$

逐项拆解：

| 符号 | 含义 |
|------|------|
| $\mathbb{E}_{q, \{o_i\}}$ | 对每个 prompt 和它的 $G$ 个回答求期望 |
| $\frac{1}{\|o_i\|} \sum_t$ | 对第 $i$ 个回答的每个 token 求平均（长度归一化） |
| $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\text{old}}(a_{i,t} \mid s_{i,t})}$ | 第 $i$ 个回答第 $t$ 个 token 的 importance sampling ratio |
| $\hat{A}_i$（注意下标是 $i$ 不是 $t$） | 同一个回答所有 token 共享同一个 advantage |
| $\beta \cdot D_{\text{KL}}$ | KL 散度惩罚，别偏离 reference 太远 |

### 3.2 和 PPO 的对比

| | PPO | GRPO |
|---|-----|------|
| Advantage 下标 | $\hat{A}_t$（per-token） | $\hat{A}_i$（per-response） |
| Advantage 计算 | GAE（需要 Critic） | 组内 z-score（不需要 Critic） |
| Clip 机制 | $\min(r_t A_t, \operatorname{clip}(r_t) A_t)$ | 完全一样 |
| KL 惩罚 | 加在 reward 上（通过 GAE 传播到每个 token） | 直接加在 loss 上 |
| Value loss | 有（训 Critic） | 无 |

---

## Part 4: 从零构建 GRPO 训练循环

### 4.0 数据集：RLAIFDataset

GRPO 训练的数据格式很简单——只需要 prompt，不需要标注好的回复（回复由 rollout 在线生成）。

```python
# 写到: scratch/dataset/lm_dataset.py（追加 RLAIFDataset）

class RLAIFDataset(Dataset):
    """GRPO 训练数据集：只提供 prompt，回复由 rollout 在线生成"""
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        # 按概率决定是否开启 thinking 模式（<think>...</think>）
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],          # 去掉最后一轮 assistant 回复
            tokenize=False,
            open_thinking=use_thinking,   # tokenizer 的 chat template 支持 thinking 开关
            add_generation_prompt=True    # 末尾加 "<|im_start|>assistant\n"
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        return {'prompt': prompt, 'answer': ""}
```

关键设计：`conversations[:-1]` 去掉最后一轮 assistant 回复——因为 GRPO 要自己生成回复（rollout），不用人标的。`thinking_ratio=0.9` 表示 90% 的时候在 prompt 末尾加 `<think>\n`，引导模型先思考再回答。

### 4.1 Rollout：用 Policy 生成回复

Rollout 是 GRPO 的核心——用当前 Policy 对每个 prompt 生成 $G$ 个回复，同时记录每个 token 的 log probability（后续算 ratio 要用）。

```python
# 写到: scratch/trainer/rollout_engine.py

def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    """计算序列最后 n_keep 个 token 的 log probability"""
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
    # 只保留最后 n_keep+1 个 token 的 logits（节省显存）
    logits = model(input_ids, attention_mask=attention_mask,
                   logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        # log_softmax → gather 取出实际 token 的 log prob
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)
```

Rollout 引擎的核心 `rollout` 方法：

```python
# 写到: scratch/trainer/rollout_engine.py

class TorchRolloutEngine(RolloutEngine):
    def rollout(self, prompt_ids, attention_mask, num_generations,
                max_new_tokens, temperature=0.8):
        model = self.policy_model
        with torch.no_grad(), self.autocast_ctx:
            output_ids = model.generate(
                # repeat_interleave: [B, P] → [B*G, P]，每个 prompt 复制 G 次
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,       # 0.8: 保证 G 个回复有差异
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()                          # [B*G, P+R]

            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]   # [B*G, R]
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()
            # 计算 completion 每个 token 的 log prob（= ratio 的分母 π_old）
            per_token_logps = compute_per_token_logps(
                self.policy_model, output_ids, completion_ids.size(1), full_mask
            )

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions,
                             prompt_ids.new_full((output_ids.size(0),), prompt_len),
                             attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))
```

`temperature=0.8` 是关键——如果温度太低，$G$ 个回复几乎一样，组内比较没有信息量。

### 4.2 Reward 计算：规则 + Reward Model

```python
# 写到: scratch/trainer/train_grpo.py

def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)
    with torch.no_grad():
        batch_size = len(prompts)
        reward_model_scores = []
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]

                # ===== 规则分（类似 DeepSeek-R1 的 accuracy + format reward）=====
                # 长度奖惩：太短或太长都扣分
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    # 思考内容长度奖惩
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    # 格式正确性：应该恰好一个 </think> 标签
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                else:
                    answer = response

                # 重复惩罚
                rewards[response_idx] -= rep_penalty(answer)

                # ===== RM 分（语义层面的质量评估）=====
                prompt = prompts[i]
                # 从 prompt 中解析 chat 格式的 messages
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores
    return rewards
```

类似 DeepSeek-R1 的思路——规则分（accuracy + format）+ RM 分的混合。规则分提供稳定的基础信号，RM 分提供语义层面的质量评估。

### 4.3 Advantage 归一化

```python
# 写到: scratch/trainer/train_grpo.py（grpo_train_epoch 函数内）

# rewards: [B*G] → 按组（G 个一组）归一化
grouped_rewards = rewards.view(-1, args.num_generations)          # [B, G]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*G]
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*G]
advantages = (rewards - mean_r) / (std_r + 1e-4)                 # [B*G]
```

这就是 Part 2 公式里的 $\hat{A}_i = (r_i - \mu) / (\sigma + \epsilon)$。

`repeat_interleave` 的作用：mean/std 是 per-prompt 的（`[B]`），需要复制 $G$ 次才能和 `[B*G]` 的 rewards 做逐元素运算。

`unbiased=False`：用 $N$ 而非 $N-1$ 做除数。$G=6$ 时差别不大，但 `False` 更稳定。

### 4.4 Policy Loss

MiniMind 默认用 CISPO 变体（Clipped Importance Sampling Policy Optimization）：

```python
# 写到: scratch/trainer/train_grpo.py（grpo_train_epoch 函数内）

# CISPO 变体
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()    # max=5.0, detach
per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
```

和标准 PPO-clip 的区别：CISPO 把 ratio clamp 后 `.detach()`，ratio 只作为重要性采样权重，**不参与梯度计算**。梯度只通过 `per_token_logps` 回传。

标准 GRPO loss 也实现了：

```python
# 写到: scratch/trainer/train_grpo.py（grpo_train_epoch 函数内，loss_type == "grpo" 分支）

clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
per_token_loss1 = ratio * advantages.unsqueeze(1)
per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```

这和 [13 章](13-RL对齐基础.md) Part 7.2 的 PPO-clip 公式完全对应：`min(ratio * A, clip(ratio) * A) - beta * KL`。

### 4.5 KL 散度惩罚

```python
# 写到: scratch/trainer/train_grpo.py（grpo_train_epoch 函数内）

kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1           # Schulman k3 estimator
```

这是 **KL 散度的 Schulman 近似**：

$$\hat{D}_{\text{KL}} = \exp\!\big(\log \pi_{\text{ref}} - \log \pi_\theta\big) - \big(\log \pi_{\text{ref}} - \log \pi_\theta\big) - 1 = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

性质：

- 当 $\pi_\theta = \pi_{\text{ref}}$ 时 $\hat{D}_{\text{KL}} = 0$
- 偏离时 $> 0$，始终非负
- 比简单的 log-ratio 更对称——不管 $\pi_\theta$ 比 $\pi_{\text{ref}}$ 大还是小，偏离都会被惩罚

为什么用近似？精确 KL $= \sum_a \pi_\theta(a) \log(\pi_\theta(a) / \pi_{\text{ref}}(a))$ 需要遍历整个词表（6400 个 token），太贵。Schulman 近似只需要 rollout token 的 logprob，$O(1)$ per token。

### 4.6 完整训练步骤

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO 一个训练 step                            │
│                                                                 │
│  ① DataLoader 产出 2 个 prompt                                  │
│     ↓                                                           │
│  ② Rollout: 每个 prompt 生成 6 个回复 = 12 个回复               │
│     同时记录 old_per_token_logps（= ratio 分母）                │
│     ↓                                                           │
│  ③ Reward: 规则分 + RM 打分 = 12 个 reward 值                  │
│     ↓                                                           │
│  ④ Advantage: 按组（6个一组）z-score 归一化                     │
│     A_i = (r_i - group_mean) / group_std                        │
│     ↓                                                           │
│  ⑤ Policy Forward: 12 条序列 → 当前 per_token_logps（= ratio 分子）│
│     ↓                                                           │
│  ⑥ Reference Forward: 12 条序列 → ref_per_token_logps（= KL）  │
│     ↓                                                           │
│  ⑦ Loss 计算:                                                   │
│     ratio = exp(new_logps - old_logps)                          │
│     kl = exp(ref_logps - new_logps) - (ref_logps - new_logps) -1│
│     loss = -(clip(ratio) × A_i × new_logps - beta × kl)        │
│     ↓                                                           │
│  ⑧ loss.backward() → clip_grad_norm → optimizer.step           │
│     ↓                                                           │
│  ⑨ 同步更新 rollout 引擎的 policy 权重                          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 完整实现

以上代码片段拼成的完整可运行文件：

| 文件 | 内容 |
|------|------|
| `scratch/trainer/rollout_engine.py` | `compute_per_token_logps` + `RolloutResult` + `RolloutEngine`（抽象基类）+ `TorchRolloutEngine` + 工厂函数 |
| `scratch/trainer/train_grpo.py` | `rep_penalty` + `calculate_rewards` + `grpo_train_epoch`（训练主循环）+ `__main__`（初始化环境 + 模型 + 训练） |
| `scratch/dataset/lm_dataset.py` | 追加 `RLAIFDataset`（只需 prompt，回复由 rollout 在线生成） |

#### 代码块 1：RLAIFDataset（追加到 scratch/dataset/lm_dataset.py）

和 SFTDataset 不同，RLAIF 不需要标注的 assistant 回复——回复由 rollout 在线生成。Dataset 只负责把原始 JSONL 渲染成 prompt 字符串。

```python
# 追加到: scratch/dataset/lm_dataset.py

import json
import random

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """概率性添加 system prompt，增加多样性"""
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
    ]
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.9):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 90% 概率在 prompt 末尾加 <think>\n
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],          # 去掉最后一轮 assistant 回复，让 rollout 自己生成
            tokenize=False,
            open_thinking=use_thinking,  # thinking 模式：末尾加 <think>\n
            add_generation_prompt=True   # 末尾加 "<|im_start|>assistant\n"
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        return {'prompt': prompt}
```

#### 代码块 2：rollout_engine.py（scratch/trainer/rollout_engine.py）

教学版只保留 `TorchRolloutEngine`（PyTorch 原生推理），去掉 SGLang HTTP 引擎。

```python
# 写到: scratch/trainer/rollout_engine.py

import torch
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional
from torch import Tensor


# ===== 计算每个 token 的 log probability =====
def compute_per_token_logps(model, input_ids: Tensor, n_keep: int,
                            attention_mask: Optional[Tensor] = None) -> Tensor:
    """
    对序列最后 n_keep 个 token 计算 log P(token | prefix)。
    n_keep 对应生成部分的长度 R。

    input_ids: [B, P+R]    完整序列（prompt + 生成）
    n_keep:    R            只算最后 R 个 token 的 logprob
    返回:      [B, R]       每个生成 token 的 log probability
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    # logits_to_keep: 只保留最后 n_keep+1 个位置的 logits（节省显存）
    # [:, :-1, :] 是因为 logits[t] 预测的是 token[t+1]
    logits = model(input_ids, attention_mask=attention_mask,
                   logits_to_keep=n_keep + 1).logits[:, :-1, :]   # [B, n_keep, vocab]

    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        # log_softmax 转概率分布 → gather 取出实际生成 token 的 log prob
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1,
                         ids_row.unsqueeze(1)).squeeze(1)          # [n_keep]
        )
    return torch.stack(per_token_logps)                            # [B, R]


# ===== Rollout 结果 =====
@dataclass
class RolloutResult:
    output_ids: Tensor        # [B*G, P+R] 完整序列（prompt + 生成）
    completion_ids: Tensor    # [B*G, R]   生成部分
    per_token_logps: Tensor   # [B*G, R]   每个生成 token 的 log prob（ratio 分母）
    completions: List[str]    # 解码后的文本，用于 reward 计算
    prompt_lens: Tensor       # [B*G]      每条的 prompt 长度
    completion_mask: Tensor   # [B*G, R]   生成部分的有效位 mask


# ===== Rollout 引擎抽象基类 =====
class RolloutEngine(ABC):
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor,
                num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


# ===== PyTorch 原生推理引擎 =====
class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model, tokenizer, device="cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx or nullcontext()

    def rollout(self, prompt_ids, attention_mask, num_generations,
                max_new_tokens, temperature=0.8):
        model = self.policy_model
        ctx = self.autocast_ctx

        with torch.no_grad(), ctx:
            # ① 每个 prompt 复制 G 次：[B, P] → [B*G, P]
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()                                               # [B*G, P+R]

            # ② 拆出生成部分
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]             # [B*G, R]

            # ③ 算 per-token logprob（作为后续 ratio 的分母 old_logps）
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()
            per_token_logps = compute_per_token_logps(
                self.policy_model, output_ids, completion_ids.size(1),
                attention_mask=full_mask
            )

        # ④ 解码文本（用于 reward 计算）
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=per_token_logps,
            completions=completions,
            prompt_lens=prompt_ids.new_full((output_ids.size(0),), prompt_len),
            completion_mask=attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)),
        )

    def update_policy(self, model):
        """训练步结束后，同步更新推理用的 policy 权重"""
        self.policy_model = model
```

#### 代码块 3：train_grpo.py（scratch/trainer/train_grpo.py）

教学版去掉了 DDP 分布式、wandb 日志、断点续训、SGLang 引擎、debug 打印，只保留 GRPO 算法核心。

```python
# 写到: scratch/trainer/train_grpo.py

import argparse
import math
import re
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.rollout_engine import TorchRolloutEngine, compute_per_token_logps


# ===== Reward Model 封装 =====
# 复用上游 trainer/trainer_utils.py 的 LMForRewardModel
# 这里简化写一个等价版本：用 HuggingFace 的 reward model 给 (messages, answer) 打分
class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = device

    def get_score(self, messages, answer):
        """给 (messages, answer) 打一个标量分"""
        chat = messages + [{"role": "assistant", "content": answer}]
        text = self.tokenizer.apply_chat_template(chat, tokenize=False)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=2048).to(self.device)
        with torch.no_grad():
            score = self.model(**inputs).logits[0].item()
        return score


# ===== 重复 n-gram 惩罚 =====
def rep_penalty(text, n=3, cap=0.5):
    """
    统计文本中重复 n-gram 的比例，返回惩罚值（0 ~ cap）。
    重复越多惩罚越大，鼓励模型生成多样化内容。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    # 重复比例 × 放大系数，但不超过 cap
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


# ===== Reward 计算 =====
def calculate_rewards(prompts, responses, reward_model, num_generations, device):
    """
    对 B*G 个回复计算 reward，返回 [B*G] 的 reward 向量。
    reward = 长度分 + 思考格式分 + RM 打分 - 重复惩罚
    """
    rewards = torch.zeros(len(responses), device=device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(num_generations):
                idx = i * num_generations + j
                response = responses[idx]

                # --- 规则分 ---
                # ① 长度奖惩
                rewards[idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                answer = response
                # ② 思考格式分（如果模型输出了 <think>...</think>）
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()

                # ③ 重复惩罚
                rewards[idx] -= rep_penalty(answer)

                # --- RM 分（语义质量）---
                prompt = prompts[i]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=device)
        rewards += reward_model_scores

    return rewards


# ===== GRPO 训练主循环 =====
def grpo_train_epoch(epoch, loader, iters, model, optimizer, scheduler,
                     rollout_engine, ref_model, reward_model, autocast_ctx, args):
    for step, batch in enumerate(loader, start=1):
        # ===== ① Tokenize prompts =====
        prompts = batch['prompt']  # list[str], 长度 B
        prompt_inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            return_token_type_ids=False, padding_side="left",
            add_special_tokens=False
        ).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ===== ② Rollout: 每个 prompt 生成 G 个回复 =====
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids                        # [B*G, P+R]
        completion_ids = rollout_result.completion_ids              # [B*G, R]
        completions = rollout_result.completions                   # list[str]
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        # logp_pos: 在 full 序列中，生成部分对应的位置索引
        logp_pos = (prompt_lens.unsqueeze(1) - 1
                    + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0))

        # ===== ③ Reward =====
        rewards = calculate_rewards(
            prompts, completions, reward_model, args.num_generations, args.device
        )

        # ===== ④ Advantage: 组内 z-score 归一化 =====
        grouped_rewards = rewards.view(-1, args.num_generations)   # [B, G]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)          # [B*G]

        # ===== ⑤ Policy Forward: 当前 per_token_logps（ratio 分子）=====
        with autocast_ctx:
            res = model(outputs, attention_mask=full_mask)
            per_token_logps = (F.log_softmax(res.logits[:, :-1, :], dim=-1)
                               .gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)
                               .gather(1, logp_pos))               # [B*G, R]

        # ===== ⑥ Reference Forward: ref_per_token_logps（KL 计算用）=====
        with torch.no_grad():
            ref_logits = ref_model(outputs, attention_mask=full_mask).logits
            ref_per_token_logps = (F.log_softmax(ref_logits[:, :-1, :], dim=-1)
                                   .gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)
                                   .gather(1, logp_pos))           # [B*G, R]

        # ===== ⑦ Completion Mask: 截断到第一个 EOS =====
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1,
                             dtype=torch.long, device=args.device)
        # 找到每行第一个 EOS 的位置
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 只保留 EOS 及之前的 token
        completion_mask = (
            (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1)
             <= eos_idx.unsqueeze(1)) & completion_pad_mask
        ).int()                                                     # [B*G, R]

        # ===== ⑧ KL 散度（Schulman k3 近似）=====
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1              # 非负，对称惩罚

        # ===== ⑨ Policy Loss =====
        ratio = torch.exp(per_token_logps - old_per_token_logps)    # [B*G, R]

        if args.loss_type == "cispo":
            # CISPO: clamp ratio 后 detach，梯度只通过 per_token_logps
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                               - args.beta * per_token_kl)
        else:
            # 标准 GRPO（PPO-clip）
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2)
                               - args.beta * per_token_kl)

        # 在 completion_mask 上求均值：每条序列除以有效 token 数，再 batch 均值
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1)
                       / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = policy_loss / args.accumulation_steps
        loss.backward()

        # ===== ⑩ 梯度累积 + 更新 =====
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ===== 日志 =====
        if step % args.log_interval == 0 or step == iters:
            avg_reward = rewards.mean().item()
            kl_ref = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() \
                     / max(completion_mask.sum().item(), 1)
            avg_len = completion_mask.sum(dim=1).float().mean().item()
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                  f'Reward: {avg_reward:.4f}, KL_ref: {kl_ref:.4f}, '
                  f'Loss: {policy_loss.item():.4f}, Avg Len: {avg_len:.1f}, LR: {lr:.2e}')

        # ===== 同步 rollout 引擎权重 =====
        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        # 释放显存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, advantages, completion_mask


# ===== 入口 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl")
    parser.add_argument("--num_generations", type=int, default=6)
    parser.add_argument("--beta", type=float, default=0.1, help="KL 惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="CISPO ratio 上界")
    parser.add_argument("--from_weight", type=str, default="full_sft")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    parser.add_argument("--thinking_ratio", type=float, default=0.9)
    args = parser.parse_args()

    # ========== 1. 模型配置 ==========
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
    )

    # ========== 2. 混合精度 ==========
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (nullcontext() if "cpu" in args.device
                    else torch.cuda.amp.autocast(dtype=dtype))

    # ========== 3. 初始化三个模型 ==========
    tokenizer = AutoTokenizer.from_pretrained('../model', trust_remote_code=True)

    # Policy 模型（要训练的）
    model = MiniMindForCausalLM(lm_config).to(args.device)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckpt = f'../out/{args.from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    model.load_state_dict(torch.load(ckpt, map_location=args.device), strict=False)
    model.train()

    # Reference 模型（冻结，用于 KL 约束）
    ref_model = MiniMindForCausalLM(lm_config).to(args.device)
    ref_model.load_state_dict(torch.load(ckpt, map_location=args.device), strict=False)
    ref_model.eval().requires_grad_(False)

    # Reward 模型（冻结，用于打分）
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device)

    # ========== 4. Rollout 引擎 ==========
    rollout_engine = TorchRolloutEngine(model, tokenizer, args.device, autocast_ctx)

    # ========== 5. 数据 + 优化器 + 调度器 ==========
    train_ds = RLAIFDataset(args.data_path, tokenizer,
                            max_length=lm_config.max_seq_len,
                            thinking_ratio=args.thinking_ratio)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = math.ceil(len(loader) / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps,
                                  eta_min=args.learning_rate / 10)

    # ========== 6. 训练 ==========
    for epoch in range(args.epochs):
        grpo_train_epoch(epoch, loader, len(loader), model, optimizer, scheduler,
                         rollout_engine, ref_model, reward_model, autocast_ctx, args)
```

运行命令：

```bash
cd scratch
python -m trainer.train_grpo \
    --batch_size 2 \
    --num_generations 6 \
    --learning_rate 3e-7 \
    --beta 0.1 \
    --loss_type cispo \
    --from_weight full_sft
```

---

## Part 5: GRPO 的局限性——per-response advantage 的代价

### 5.1 对 Math Reasoning 没问题

Math 场景的特征：最终答案对了 = 全对，答案错了 = 全错。per-response advantage 天然匹配这种 episode-level reward。整条推理链共享一个 advantage 没问题——反正我们关心的就是"整条链的结果"。

这也是为什么 DeepSeek-R1 用 GRPO 在 math/code reasoning 上效果极好。

### 5.2 对 Agent 场景有三个问题

**问题 1：per-response advantage → token 级信号错误**

```
Agent 轨迹（查天气任务）:

Turn 1: 用户: "帮我看下北京今天天气"
  → Think: "需要查天气API"                          ← 正确
  → Tool call: get_weather("北京")                   ← 正确
  → Observation: "22°C, 多云"

Turn 2: Think: "用户可能还想知道明天的"
  → Tool call: get_forecast("北京")                  ← 正确
  → Observation: "明天有雨, 18°C"

Turn 3: Think: "整理一下回复"
  → Final answer: "北京今天22°C多云，明天有雨18°C建议带伞"  ← 正确
  → 但输出格式少了 </tool_call> 闭合标签              ← 格式错误！

→ 整个 episode 失败, reward = 0
→ GRPO 给所有 token 同样的负 advantage
→ Turn 1~2 正确的 tool call 也被惩罚了！

PPO 的 per-token advantage 不会有这个问题——GAE 会把负 reward 按 γλ 衰减回传，
越早的正确步骤受影响越小。
```

**问题 2：组内归一化在所有回复都失败时崩溃**

```
G=4 个 trajectory 全部失败:  r = [0, 0, 0, 0]
  → std = 0，advantage = 0/0 → NaN 或归一化无意义

一个勉强成功 (r=0.3)，三个完全失败 (r=0):
  → 归一化后 0.3 吃正 advantage
  → 但 0.3 的成功率根本不够好
  → 学到的是"比烂"而不是"做好"
```

MiniMind 用 `std + 1e-4` 防止除零，但根本问题仍在。

**问题 3：轨迹长度方差大 → reward 方差大**

```
Math: 答案长度 100~500 token，reward 方差小
Agent: 轨迹 5~100 步，reward = 0 或 1（二值），方差极大
  → G=4 个轨迹的组内归一化非常不稳定
  → 需要增大 G 来降低方差，但 agent rollout 很贵（多轮工具调用）
```

### 5.3 MiniMind Agent RL 怎么缓解

MiniMind 的 `train_agent.py`（[16-AgentRL](16-AgentRL.md)）用了几个技巧部分缓解：

1. **response_mask**：工具返回的 token 设 mask=0 不参与 loss，只有模型生成的 token 受 advantage 影响
2. **多维 reward**：不只看最终结果，还奖励工具选择正确性、格式正确性，提供更密集的信号
3. **GT 弱监督**：用 ground truth 验证最终答案，给出连续的匹配分，不是纯 0/1

但根本性缺陷（per-response advantage）无法在 GRPO 框架内彻底解决。

---

## Part 6: DeepSeek-R1 的四阶段训练管线

GRPO 最成功的应用案例——理解它有助于理解 MiniMind 的设计决策。

```
Cold Start SFT → Reasoning RL (GRPO) → Rejection Sampling + SFT → Full RL (GRPO)
   Stage 1            Stage 2                 Stage 3                 Stage 4
```

| 阶段 | 做什么 | 为什么 |
|------|--------|--------|
| 1. Cold Start SFT | 几千条长 CoT 数据做 SFT | 给模型一个"推理的起点"——会写 `<think>...</think>` 格式 |
| 2. Reasoning RL | GRPO + **纯规则 reward**（accuracy + format） | 模型自己长出推理能力（"aha moment"），涌现反思、回溯、自我验证 |
| 3. Rejection Sampling + SFT | 用 Stage 2 模型生成 → 筛选 → 混合通用数据 → **从头 SFT** | Stage 2 模型推理强但通用能力退化，需要 SFT 拉回 |
| 4. Full RL | GRPO + accuracy + helpfulness + harmlessness reward | 最终对齐——推理 + 通用 + 安全 |

关键发现：Stage 2 用**纯规则 reward**（不需要 RM）就训出了强推理能力。MiniMind 的 GRPO reward 设计受此启发。

---

## 关键超参数

| 超参数 | MiniMind | DeepSeekMath / R1 | 含义 |
|--------|---------|-------------------|------|
| `learning_rate` | 3e-7 | 1e-6（DeepSeekMath 论文；R1 未公开） | RL 训练对学习率极度敏感 |
| `batch_size` | 2 | 大得多 | 每步 2 prompt × 6 回复 = 12 序列 |
| `num_generations` ($G$) | 6 | 64（DeepSeekMath 论文；R1 未公开） | 组越大 advantage 估计越稳 |
| `beta` | 0.1 | 0.04（DeepSeekMath 论文；R1 未公开） | KL 惩罚强度 |
| `epsilon` | 0.2 | 0.2 | PPO-clip 范围 |
| `max_gen_len` | 1024 | 更长 | 单个回复最大长度 |
| `from_weight` | full_sft | SFT | 必须从 SFT 模型开始 |
| `thinking_ratio` | 0.9 | 1.0 | 90% 概率开启思考模式 |

---

## 业界对比

| 特性 | MiniMind GRPO | DeepSeek-R1 | OpenAI PPO | LLaMA 3 |
|------|-------------|-------------|------------|---------|
| 方法 | GRPO + CISPO | GRPO + rule rewards | PPO + RM | DPO + Rejection Sampling |
| 需要 Reward Model | 是 | 推理阶段纯规则；对齐阶段用 RM | 是 | 否 |
| 需要 Value Model | 否 | 否 | 是 | 否 |
| 在线采样 | 是 | 是 | 是 | 否 |
| Advantage 粒度 | per-response | per-response | per-token | N/A |
| 组大小 $G$ | 6 | 64 | N/A | N/A |
| KL 估计 | Schulman 近似 | 类似 | 标准 KL / reward penalty | 隐式 |

### 最新的 RL 变体

| 方法 | 核心改动 | 代表工作 |
|------|---------|---------|
| **DAPO** | **去 KL + Clip-Higher + 动态采样 + token-level loss + 超长软惩罚（详见下方 Part 7）** | **ByteDance (2025)** |
| Dr. GRPO | 去掉长度归一化 $1/\|o_i\|$，避免奖励长回复的偏置 | Liu et al. (2025) |
| REINFORCE++ | GRPO 的极简变体——去掉 clip，只保留 z-score + KL | Hu (2025) |
| PRIME | Process Reward Model + 在线 RL，per-step advantage | Cui et al. (2025) |
| Leave-One-Out | baseline 用其他 $G-1$ 个样本的均值，不含自己 | Ahmadian et al. (2024) |
| **GSPO** | **序列级重要性比率 + 长度归一化，解决 MoE 训练不稳定（详见下方 Part 8）** | **Qwen / 阿里 (2025)** |

---

## Part 7: DAPO——GRPO 的进一步简化

> 论文：DAPO: An Open-Source LLM Reinforcement Learning System (ByteDance Seed, 2025)
> 用于 DeepSeek-R1 等推理模型的对齐训练

### 7.1 动机：PPO → GRPO → DAPO 的简化链

```
PPO 需要 4 个模型:  Actor + Old Policy + Critic + Reference (+ RM)
                    ↓ GRPO 去掉 Critic
GRPO 需要 3 个模型: Actor + Old Policy + Reference (+ RM)
                    ↓ DAPO 去掉 Reference + KL
DAPO 需要 2 个模型: Actor + Old Policy (+ RM/规则)
```

GRPO 去掉了 Critic，但仍然保留了 Reference Model 和 KL 惩罚来防止策略跑偏。DAPO 的核心问题是：**KL 惩罚本身就在限制模型探索——能不能用其他机制替代 KL，把 Reference Model 也省掉？**

### 7.2 去 KL：为什么敢不要 Reference Model

PPO/GRPO 都用 $\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 约束策略不要偏离 reference 太远。DAPO 直接把这个约束和 Reference Model 一起去掉，动机是：

**长 CoT 推理模型的分布本应大幅偏离初始模型。** R1 式的反思、回溯、自我纠错行为在 SFT 模型的分布里几乎不存在——把策略钉在 ref 附近，等于禁止这些行为涌现。KL 惩罚从"防跑偏的保险"变成了"限制探索的枷锁"。

```
GRPO:  loss = clip_loss - β × KL(π_θ || π_ref)     ← 需要 reference model 做前向传播
DAPO:  loss = clip_loss                              ← 不需要 reference model
```

去 KL 之后靠什么稳定训练？靠下面四件套里的 Clip-Higher（限制单步更新幅度）+ Dynamic Sampling（保证梯度信号非零）+ 规则 reward 本身无偏。工程上还省了一个模型的显存（7B 的 BF16 推理态权重约 14GB）和一次前向传播。

### 7.3 DAPO 的四个核心创新

#### 创新 1：Clip-Higher——不对称 clip 区间

PPO/GRPO 的 clip 是对称的：$\operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$，通常 $\varepsilon=0.2$。DAPO 用**不对称区间**：下界保留，**上界抬高**：

$$\operatorname{clip}(r_t, \; 1-\varepsilon_{\text{low}}, \; 1+\varepsilon_{\text{high}}) \quad \text{where} \quad \varepsilon_{\text{low}}=0.2, \; \varepsilon_{\text{high}}=0.28$$

注意：下界 $1-\varepsilon_{\text{low}}$ 没有去掉，是**上界从 1.2 抬到 1.28**——所以叫 Clip-**Higher**。

**动机：防熵塌缩（entropy collapse）。** DAPO 论文的观察：对称 clip 下，token 概率的绝对上升幅度被限制在 $\varepsilon \times \pi_{\text{old}}$——一个概率 0.01 的 token 最多涨到 0.012，而概率 0.9 的 token 能涨到接近 1。净效应是高概率 token 越来越 dominant、低概率 token 抬不起头，策略熵快速下降，模型过早收敛到少数模式、丧失探索。抬高上界 = 给低概率 token 更大的上升空间，把熵维持在健康水平。

```
PPO/GRPO 对称 clip (ε=0.2):
  好 token（A>0）: ratio 上限 1.2 → 低概率 token 的概率上升被压制 → 熵塌缩
  坏 token（A<0）: ratio 下限 0.8 → 下降通道始终在

DAPO Clip-Higher (ε_low=0.2, ε_high=0.28):
  好 token: ratio 上限 1.28 → 上升空间更大 → 熵维持
  坏 token: ratio 下限 0.8  → 与 PPO 相同
```

#### 创新 2：动态采样（Dynamic Sampling）

GRPO 对每个 prompt 采样固定 $G$ 个回复。问题：如果 $G$ 个回复**全对或全错**，组内 z-score 没有区分度（所有 advantage 全为 0，零梯度）。

DAPO 的解法：**oversampling + 过滤**——采样更多回复，过滤掉全对或全错的 prompt，只保留"有对有错"的 prompt 参与训练，直到凑满一个 batch。

```
GRPO:   对每个 prompt 采样 G=8 个回复，直接用
DAPO:   对每个 prompt 采样更多回复
         → 过滤: 全对或全错 → 扔掉这个 prompt（advantage 全 0，无梯度）
         → 保留: 至少有 1 对 1 错的 prompt → 参与训练

效果: 参与训练的每个 prompt 都有对比信号，
      组内 z-score 有意义，batch 的有效梯度密度更高
```

#### 创新 3：Token-Level Policy Gradient Loss

GRPO 的 loss 是 **sample 级聚合**——每条回复先在自己的 token 上平均，再在组内 $G$ 条回复间平均：

$$\mathcal{L}_{\text{GRPO}} = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\ell_{i,t}$$

问题：长回复的每个 token 对总 loss 的贡献是 $1/(G \cdot |o_i|)$——**回复越长，per-token 梯度被稀释得越厉害**。后果：长 CoT 中的学习/惩罚信号都被摊薄，而长推理链恰恰是 reasoning 训练的重点。

DAPO 改成 **token 级聚合**——整组所有 token 拉平，一起平均：

$$\mathcal{L}_{\text{DAPO}} = \frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\ell_{i,t}$$

```
例子: G=2, o_1 长 10 token, o_2 长 100 token

GRPO (sample 级):  o_1 的每个 token 权重 1/(2×10)  = 0.05
                   o_2 的每个 token 权重 1/(2×100) = 0.005  ← 稀释 10 倍
DAPO (token 级):   每个 token 权重 1/110 ≈ 0.0091          ← 一视同仁
```

注意这与"mask padding/EOS"不是一回事——padding mask 是两种方法都有的基本操作，DAPO 改的是**聚合粒度**。

#### 创新 4：Overlong Reward Shaping（超长软惩罚）

超长被截断的回复怎么处理，是个 reward 噪声源：直接给 0 分，会把"思路对但没写完"的回复判成全错；完全不管，又控制不了篇幅。

DAPO 的做法：**软惩罚区间**——设上限 $L_{\max}$ 和缓冲长度 $L_{\text{cache}}$，长度进入 $[L_{\max}-L_{\text{cache}},\, L_{\max}]$ 区间后，惩罚随长度线性增加；真超长的再截断（并可把被截断样本过滤出 loss）。

```
回复长度 L，上限 L_max，缓冲 L_cache:

L ≤ L_max − L_cache:          无惩罚
L_max − L_cache < L ≤ L_max:  线性软惩罚（越长扣得越多）
L > L_max:                    截断，按失败处理（可过滤出 loss）
```

效果：把"长度"从 0/1 硬开关变成连续信号，模型学到"控制篇幅"，而不是"要么随便写、要么突然被罚"。

### 7.4 PPO vs GRPO vs DAPO 对比

| 特性 | PPO | GRPO | DAPO |
|------|-----|------|------|
| 需要 Critic | 是 | 否 | 否 |
| 需要 Reference Model | 是（KL 锚点） | 是（KL 锚点） | **否** |
| KL 惩罚 | 有（加在 reward 上） | 有（加在 loss 上） | **无**（去 KL 释放探索） |
| Clip 方式 | 对称 $[1-\varepsilon, 1+\varepsilon]$ | 对称 | **不对称**（$\varepsilon_{\text{low}}=0.2$、$\varepsilon_{\text{high}}=0.28$，抬高上界防熵塌缩） |
| 采样策略 | 固定 | 固定 $G$ 个 | **动态 oversampling + 过滤全对/全错** |
| Loss 聚合粒度 | per-token（GAE） | sample 级 | **token 级**（防长回复梯度稀释） |
| 超长回复 | 截断 | 截断 | **软惩罚 + 过滤** |
| 同时需要的模型数 | 4 (+ RM) | 3 (+ RM) | **2 (+ RM/规则)** |
| 显存占用 | 最高 | 中等 | **最低** |
| 适用场景 | 通用 | Math / Code reasoning | 长 CoT reasoning |

### 7.5 面试话术（3 句话版本）

> "DAPO 是字节 Seed 团队在 GRPO 基础上为长 CoT reasoning 做的改进：去掉 KL，再加四个技术。去 KL 的动机是推理模型的分布本应大幅偏离初始模型——反思、回溯这些行为在 ref 分布里几乎不存在，KL 惩罚会压制这类探索。四个技术：第一，Clip-Higher，不对称 clip 区间，下界保留 ε_low=0.2、上界抬到 ε_high=0.28，动机是防熵塌缩——对称 clip 下低概率 token 的概率上升被压制，策略熵会快速下降。第二，Dynamic Sampling，过滤掉全对全错的 prompt，保证每个 batch 都有非零梯度。第三，Token-Level Policy Gradient Loss，把 loss 聚合粒度从 sample 级改成 token 级，解决长回复 per-token 梯度被稀释的问题。第四，Overlong Reward Shaping，对接近长度上限的回复给线性软惩罚，减少截断带来的 reward 噪声。本质上，DAPO 去掉了 GRPO 依赖 ref 的约束，用四个针对性机制分别解决熵塌缩、零梯度、梯度稀释、超长噪声这四个 reasoning 训练的具体失败模式。"

---

## Part 8: GSPO——解决 GRPO 在 MoE 上的训练崩溃

> 论文：Group Sequence Policy Optimization (Qwen Team, Alibaba, 2025)
> arXiv: 2507.18071 | 用于 Qwen3 系列模型训练

### 8.1 动机：GRPO 的 token-level 重要性采样在 MoE 模型上失效

GRPO 的重要性比率是 **token 级**的：

$$w_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}$$

每个 token 独立算新旧策略的概率比。这在 dense 模型上没问题，但 **MoE 模型有个致命问题**：

```
MoE 的 expert 路由问题:

  每次梯度更新后，同一个 rollout 样本：
  - 约 10% 的 expert 路由发生变化（Qwen3-30B-A3B-Base 实测）
  - 同一个 token，更新前走 expert 3，更新后走 expert 7
  - 不同 expert 产出完全不同 → 概率 π_θ(y_{i,t}|...) 剧烈波动
  
  token-level ratio 的后果：
  - 单个 token 的 ratio 可能从 0.5 跳到 5.0
  - 重要性采样的基本假设被破坏
  - 梯度噪声逐步累积 → 训练不稳定甚至崩溃
```

核心洞察：**token-level ratio 的波动会随回复长度累积**。回复越长，有越多 token 的路由发生变化，梯度噪声越大。

### 8.2 GSPO 的解法：序列级重要性比率

GSPO 把重要性比率从 token 级提升到 **序列级**，并做**长度归一化**：

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}\right)^{1/|y_i|}$$

展开成 token 级的对数形式：

$$s_i(\theta) = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}\right)$$

```
为什么序列级更稳定:

  token-level:  每个 token 的 ratio 独立波动
                → 个别 token 路由变化导致极端值
                → clip 只能截断，不能消除噪声

  sequence-level: 对所有 token 的 log-ratio 取平均
                  → 个别 token 的极端波动被平滑掉
                  → 整条回复的"整体偏离程度"是稳定的

  长度归一化 (1/|y_i|): 
    消除长回复 vs 短回复的 ratio 尺度差异
    100 token 和 500 token 的回复在同一尺度上比较
```

### 8.3 GSPO 的目标函数

$$J_{\text{GSPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\Big(s_i(\theta)\hat{A}_i,\;\text{clip}\big(s_i(\theta),\,1-\varepsilon,\,1+\varepsilon\big)\hat{A}_i\Big)\right]$$

advantage 的计算和 GRPO 一样是组内 z-score：

$$\hat{A}_i = \frac{r(x, y_i) - \text{mean}(\{r(x, y_j)\}_{j=1}^G)}{\text{std}(\{r(x, y_j)\}_{j=1}^G)}$$

形式上和 GRPO 的 PPO-clip 目标几乎一样，**唯一区别是 ratio 从 token 级 $w_{i,t}$ 换成了序列级 $s_i$**。但这一个改动解决了 MoE 训练不稳定的根本问题。

### 8.4 一个反直觉的发现：clip 更多，效果更好

GSPO 训练中被 clip 掉的 token 数量是 GRPO 的 **约 100 倍**——因为序列级 ratio 一旦触发 clip，整条回复的所有 token 都被 clip（而 GRPO 只 clip 少数极端 token）。

```
GRPO:  clip 1% 的 token  → 其余 99% 的 token 带着噪声梯度通过
GSPO:  clip 100% 的 token（按序列）→ 但每条通过的序列信号干净

直觉：宁可丢掉更多样本，也不让噪声梯度污染更新方向
```

这说明 GRPO 的低 clip 率不是优势，而是 **token-level ratio 天然分散导致难以触发 clip**——很多有问题的序列没被 clip 挡住。

### 8.5 GSPO 消除了 MoE 专属的工程 hack

GRPO 训练 MoE 模型时需要 **Routing Replay** 策略来稳定 expert 路由：每次梯度更新后重新计算 expert 分配。GSPO 不需要——序列级 ratio 天然对 expert 路由变化鲁棒。

```
GRPO + MoE:   需要 Routing Replay（每步额外的 forward pass）
GSPO + MoE:   不需要，直接训练就稳
```

### 8.6 PPO vs GRPO vs DAPO vs GSPO 对比

| 特性 | PPO | GRPO | DAPO | GSPO |
|------|-----|------|------|------|
| 重要性比率粒度 | token 级 | token 级 | token 级 | **序列级** |
| Critic | 要 | 不要 | 不要 | 不要 |
| Reference Model | 要 | 要 | 不要 | 要 |
| KL 惩罚 | 有 | 有 | 无 | 有 |
| Clip 方式 | 对称 | 对称 | 不对称 | 对称 |
| Loss 聚合 | per-token（GAE） | sample 级 | token 级 | **序列级** |
| MoE 训练稳定性 | 差 | 差（需 Routing Replay） | 未验证 | **好（无需额外策略）** |
| 解决的核心问题 | 通用 RL | 去 Critic 省显存 | 去 KL 释放探索 | **MoE 路由导致的梯度噪声** |
| 代表模型 | InstructGPT | DeepSeek-R1 | 字节开源推理模型 | **Qwen3** |

### 8.7 面试话术

> "GSPO 是 Qwen 团队提出的 GRPO 改进，核心改动是把重要性比率从 token 级换成序列级。动机来自 MoE 模型的训练不稳定：Qwen3-30B 训练时发现，每次梯度更新后约 10% 的 expert 路由会变，导致 token-level ratio 剧烈波动，噪声随回复长度累积，最终训练崩溃。GSPO 的做法很简单——对整条序列的所有 token log-ratio 取平均再做长度归一化，得到一个序列级的重要性比率。这样个别 token 因路由变化产生的极端值会被平滑掉。一个反直觉的发现是 GSPO 被 clip 的 token 数量是 GRPO 的 100 倍，效果反而更好——因为 GRPO 的 token-level ratio 太分散，很多有问题的序列没被 clip 挡住，噪声梯度直接污染了更新方向。本质上，GSPO 把优化单元从 token 对齐到 reward 的粒度——reward 是 per-sequence 的，ratio 也应该是 per-sequence 的。"

---

## 面试话术

> "GRPO 是 DeepSeek 提出的 RL 对齐方法，核心改动是扔掉 PPO 的 Critic。
>
> PPO 用 Critic 网络 V(s_t) 估 value baseline 来算 per-token advantage，但 Critic 显存大、难训准。GRPO 换了一个思路：对每个 prompt 采样 G 个回复，reward 在组内做 z-score 归一化得到 advantage——用组均值替代 value baseline，'这个回复比同组平均水平好多少'。
>
> 省了 value model，但 advantage 从 per-token 退化成 per-response——同一条回复的所有 token 共享同一个 advantage。对 math reasoning 没问题，最终答案对了就是全对。但对 agent 场景有根本缺陷——5 步正确 + 1 步格式错，per-response advantage 会惩罚所有 token。
>
> DeepSeek-R1 进一步证明，GRPO + 纯规则 reward 就能训出强推理能力——涌现了反思、回溯等行为，不是 SFT 教的而是 RL 自己'长出来的'。这是 LLM RL 最激动人心的发现。"

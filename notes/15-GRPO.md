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

**per-response advantage 的含义**：第 $i$ 个回答的所有 token，不管是第 1 个还是第 100 个，都用同一个 $\hat{A}_i$。这对 credit assignment 来说是一个退步——但对 math/code reasoning 来说通常没问题（Part 5 讨论适用场景）。

---

## Part 3: GRPO Loss 公式

### 3.1 完整公式

$$\boxed{\mathcal{L}^{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^{G}} \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\!\big(r_{i,t}(\theta) \cdot \hat{A}_i, \;\operatorname{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \cdot \hat{A}_i\big) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]}$$

逐项拆解（用 MiniMind 实际参数：B=2 个 prompt，G=6 个回复，回复长度 R=50 token）：

**① 期望 $\mathbb{E}_{q, \{o_i\}_{i=1}^{G}}$**

对每个 prompt $q$ 和它的 $G$ 个回复求期望。实践中用采样近似：一个 batch 2 个 prompt，每个 prompt rollout 6 个回复，共 12 条回复取平均。

**② 长度归一化 $\frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$**

对第 $i$ 个回复的每个 token 求平均。$|o_i|$ 是这条回复的有效 token 数（到 EOS 为止）。**为什么要除以长度？** 不除的话，长回复天然有更多 token 贡献 loss，梯度会偏向长回复——不管长回复质量好不好。除以长度后，一条 100 token 的好回复和一条 20 token 的好回复对 loss 的贡献相当。

把三种训练方式的 loss 公式放在一起对比，看 token 聚合的差异：

$$\text{SFT:} \quad \mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log \pi_\theta(y_t \mid y_{<t}, x)$$

$$\text{PPO:} \quad \mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \min\!\big(\rho_t \hat{A}_t,\; \operatorname{clip}(\rho_t) \hat{A}_t\big) - c_e H(\pi_\theta)$$

$$\text{GRPO:} \quad \mathcal{L} = -\frac{1}{G}\sum_{i=1}^{G}\left[\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\!\big(r_{i,t} \hat{A}_i,\; \operatorname{clip}(r_{i,t}) \hat{A}_i\big) - \beta \cdot D_{\text{KL}}\right]$$

关键差异一目了然：

| | 聚合方式 | 每个 token 的监督信号 | 长度偏差 |
|---|---------|---------------------|---------|
| **SFT** | $\frac{1}{T}\sum_t$：所有 token 打平取均值 | **独立**：每个 token 有标准答案 $y_t$ | 长回复 token 多但 loss 独立且有正有负，自然对冲 |
| **PPO** | $\frac{1}{T}\sum_t$：所有回复的 token 打平取均值 | **独立**：每个 token 有独立的 $\hat{A}_t$（来自 GAE） | 同 SFT，advantage 有正有负，自然对冲 |
| **GRPO** | $\frac{1}{G}\sum_i \frac{1}{\|o_i\|}\sum_t$：**两层均值**，先 per-response 再 batch | **共享**：同一条回复所有 token 共享 $\hat{A}_i$ | 必须除以长度——否则长的差回复因 token 多产生巨大负梯度 |

一句话：**GRPO 必须做长度归一化，是因为它的 advantage 是 per-response 的。** 当所有 token 乘同一个系数时，token 数量直接决定了这条回复对梯度的影响力，不归一化就不公平。SFT 和 PPO 的每个 token 有独立信号，长度偏差能自然对冲，所以打平取均值也能工作。

**③ Importance Sampling Ratio $r_{i,t}(\theta)$**

$$r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s_{i,<t})}{\pi_{\text{old}}(a_{i,t} \mid s_{i,<t})} = \exp\!\big(\log\pi_\theta(a_{i,t} \mid s_{i,<t}) - \log\pi_{\text{old}}(a_{i,t} \mid s_{i,<t})\big)$$

和 [13 章 Part 7.1](13-RL对齐基础.md) 的 $\rho_t$ 完全一样。这里涉及**两个不同时刻的同一个模型**：

| 符号 | 是谁 | 什么时候算 | 会变吗 |
|------|------|-----------|--------|
| $\pi_{\text{old}}$ | 做 rollout 时的 policy | ② Rollout 阶段，`torch.no_grad()` | 冻结，整个 step 不变 |
| $\pi_\theta$ | 正在被 `optimizer.step()` 更新的 policy | ⑤ Policy Forward，`with autocast` | 每次 backward 后变化 |

**直觉**：rollout 用"旧模型"采了一批数据，现在"新模型"的参数已经更新了一些，ratio 衡量的是"新模型觉得这个 token 好不好" vs "旧模型当时觉得这个 token 好不好"。ratio > 1 说明新模型更喜欢这个 token，ratio < 1 说明新模型不太认可。

**ratio 里的概率怎么来的？** 不是直接比 logits（softmax 前的原始分数），而是经过了完整的转换链：

```
模型前向传播
    ↓
logits = model(input_ids).logits        # LM head 输出，softmax 前，[B, seq, vocab]
    ↓
log_probs = logits.log_softmax(dim=-1)  # 对 vocab 维做 log_softmax → log 概率
    ↓
per_token_logps = gather(log_probs, token_ids)  # 取出实际 token 的 log prob，[B, seq]
    ↓
ratio = exp(log_prob_new - log_prob_old)         # log 空间减法 = 概率空间除法
```

为什么在 log 空间算？softmax 后的概率值可能极小（如 1e-8），直接做除法数值不稳定。log 空间里先减后 exp，等价于 $\pi_\text{new}/\pi_\text{old}$，但数值更安全。

**代码对应**（完整实现 4.7 的 ⑨ Policy Loss）：
```python
ratio = torch.exp(per_token_logps - old_per_token_logps)    # [B*G, R]
#                 ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
#                 ⑤ 新模型算的       ② rollout 时记录的（.detach()）
```

**④ Clip 机制 $\min\!\big(r \cdot \hat{A},\; \operatorname{clip}(r, 1-\varepsilon, 1+\varepsilon) \cdot \hat{A}\big)$**

和 PPO-clip 完全一样（[13 章 Part 7.2](13-RL对齐基础.md)）。Clip 限制 ratio 在 $[1-\varepsilon, 1+\varepsilon]$（MiniMind 默认 $\varepsilon=0.2$，即 $[0.8, 1.2]$），然后取 min：

- **advantage > 0**（好回复）：ratio 不能超过 1.2，防止"好回复"被过度强化
- **advantage < 0**（差回复）：ratio 不能低于 0.8，防止"差回复"被过度抑制

效果：每次更新的步幅不能太大，保证训练稳定。

**⑤ Per-response Advantage $\hat{A}_i$**

注意下标是 $i$（第几个回复）不是 $t$（第几个 token）。**同一个回复的所有 token，不管是第 1 个还是第 50 个，都用同一个 advantage 值**。这是 GRPO 和 PPO 最本质的区别——PPO 的 $\hat{A}_t$ 是 per-token 的（每个 token 有独立的 advantage），GRPO 的 $\hat{A}_i$ 是 per-response 的（Part 2.2 讨论了这个 credit assignment 的退步）。

$\hat{A}_i$ 怎么算：Part 2 的组内 z-score，$\hat{A}_i = (r_i - \mu_{\text{group}}) / \sigma_{\text{group}}$。

**⑥ KL 散度惩罚 $\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$**

注意这里出现了**第三个模型** $\pi_{\text{ref}}$——reference model，训练全程冻结不动，是 GRPO 训练开始前的 SFT 模型快照。

| 符号 | 是谁 | 作用 |
|------|------|------|
| $\pi_{\text{old}}$ | rollout 时的 policy（每个 step 更新一次） | ratio 的分母 |
| $\pi_\theta$ | 当前正在训练的 policy | ratio 的分子、KL 的一方 |
| $\pi_{\text{ref}}$ | 训练开始前的 SFT 模型快照（全程冻结） | KL 的另一方 |

$\pi_{\text{old}}$ 和 $\pi_{\text{ref}}$ 是两个不同的东西：$\pi_{\text{old}}$ 每做一次 rollout 就更新（跟着训练走），$\pi_{\text{ref}}$ 从头到尾不变。KL 惩罚的目的是防止 $\pi_\theta$ 偏离 $\pi_{\text{ref}}$ 太远（reward hacking），而 ratio 衡量的是 $\pi_\theta$ 相对 $\pi_{\text{old}}$ 的变化（importance sampling 修正）。

和 PPO 的区别：PPO 把 KL 加在 reward 上（$r_t = R - \beta \cdot \text{KL}_t$），通过 GAE 传播到每个 token 的 advantage 里（[13 章 6.3.3](13-RL对齐基础.md)）。GRPO 直接把 KL 加在 loss 上作为独立的惩罚项，更简单直接。

### 3.2 和 PPO 的对比

| | PPO | GRPO |
|---|-----|------|
| Advantage 下标 | $\hat{A}_t$（per-token） | $\hat{A}_i$（per-response） |
| Advantage 计算 | GAE（需要 Critic） | 组内 z-score（不需要 Critic） |
| Clip 机制 | $\min(r_t A_t, \operatorname{clip}(r_t) A_t)$ | 完全一样 |
| KL 惩罚 | 加在 reward 上（通过 GAE 传播到每个 token） | 直接加在 loss 上 |
| Value loss | 有（训 Critic） | 无 |
| Entropy bonus | 论文公式有（$c_e \cdot H(\pi_\theta)$），但 LLM 实践中几乎不用 | 无 |

**Entropy Bonus：从论文到 LLM 实践的真实地位**

回顾 [13 章 7.3](13-RL对齐基础.md)，PPO 论文的完整 loss 有三项：$\mathcal{L}^{\text{clip}} - c_v(V - G_t)^2 + c_e H(\pi_\theta)$。Entropy bonus $H(\pi_\theta)$ 的作用是鼓励策略分布保持分散，防止模型塌缩到只会说一种回答。

但**这个设计是 Atari 游戏时代的产物**。PPO 论文（2017）的动作空间只有几个方向键，策略极易坍缩成"永远按右"。到了 LLM 场景：

1. **LLM 的动作空间天然高熵**。词表规模 6400（MiniMind）到 10 万+（GPT-4），下一个 token 的概率分布天然分散，熵塌缩不像游戏那么容易发生。

2. **KL 惩罚已经在做类似的事**。LLM-RLHF 的 KL 惩罚 $\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 锚定在高熵的 SFT ref model 上，间接维持了策略多样性，和 entropy bonus 的作用重叠。

3. **LLM-PPO 实践中已经静默忽略了它**。InstructGPT（2022）的目标函数没有显式 entropy bonus，主流开源实现（TRL、DeepSpeed-Chat）默认 $c_e = 0$。

所以 **GRPO 不是"去掉了 PPO 的 entropy bonus"，而是在公式层面正式去掉了 LLM-PPO 实践中本来就不用的东西**——把实践惯例写进了定义。后面 Part 6 会看到，DAPO 连 KL 都去掉了，转而用不对称 clip 来防熵塌缩——说明对推理模型来说，真正需要的是更精细的机制，而不是一个笼统的 entropy bonus。

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

先明确三个模型的身份——GRPO 训练中同时存在三个"版本"的模型：

| 变量名 | 身份 | 何时更新 | 作用 |
|--------|------|---------|------|
| `rollout_engine.policy_model`（$\pi_{\text{old}}$） | 做 rollout 的旧 policy | 每 N 个 step 同步一次（step ⑩） | ratio 的分母、生成回复 |
| `model`（$\pi_\theta$） | 正在训练的当前 policy | 每次 `optimizer.step()`（step ⑨） | ratio 的分子、loss backward 更新的目标 |
| `ref_model`（$\pi_{\text{ref}}$） | SFT 模型快照 | **全程冻结不变** | KL 惩罚的锚点 |

训练刚开始时三者相同。随着训练进行，`model` 每个 step 都在变，`rollout_engine` 周期性同步，`ref_model` 永远不动。

```
┌────────────────────────────────────────────────────────────────────────┐
│                    GRPO 一个训练 step（B=2, G=6）                      │
│                                                                        │
│  ① DataLoader 产出 2 个 prompt                                        │
│     prompts: ["2+3=?", "中国的首都?"]                                  │
│                                                                        │
│  ② Rollout（用 π_old，torch.no_grad）                                 │
│     每个 prompt 生成 6 个回复 = 12 条序列                              │
│     同时记录每个生成 token 的 log prob → old_per_token_logps [12, R]   │
│     这些 log prob 是 ratio 的分母，整个 step 内冻结不变（.detach()）    │
│                                                                        │
│  ③ Reward（12 条回复 → 12 个标量分）                                   │
│     reward = 长度分 + 思考格式分 + RM 打分 - 重复惩罚                  │
│                                                                        │
│  ④ Advantage（组内 z-score）                                           │
│     每 6 个回复一组，A_i = (r_i - group_mean) / group_std              │
│     → advantages [12]，整个 step 内冻结不变                            │
│                                                                        │
│  ②③④ 的产出全部冻结——以下才是 loss 计算                               │
│  ──────────────────────────────────────────────────────────────────     │
│                                                                        │
│  ⑤ 当前 Policy Forward（用 π_θ，参与梯度计算）                        │
│     把 ② 生成的 12 条序列喂给当前 model（不是 rollout 的旧模型）       │
│     → per_token_logps [12, R]   这是 ratio 的分子                      │
│                                                                        │
│  ⑥ Reference Forward（用 π_ref，torch.no_grad）                       │
│     同样的 12 条序列喂给冻结的 ref_model                               │
│     → ref_per_token_logps [12, R]   用于计算 KL 惩罚                   │
│                                                                        │
│  ⑦ Completion Mask                                                     │
│     找到每条回复的第一个 EOS，截断后面的 padding                        │
│                                                                        │
│  ⑧ Loss 计算                                                           │
│     ratio = exp(per_token_logps - old_per_token_logps)                 │
│              ⑤ π_θ 的输出         ② π_old 的输出                       │
│                                                                        │
│     kl = exp(ref_logps - new_logps) - (ref_logps - new_logps) - 1     │
│             ⑥ π_ref 的输出  ⑤ π_θ 的输出                              │
│                                                                        │
│     标准 GRPO:                                                         │
│       loss = -[ min(ratio × A_i, clip(ratio) × A_i) - beta × kl ]    │
│     CISPO 变体:                                                        │
│       loss = -[ clamp(ratio).detach() × A_i × logps - beta × kl ]    │
│                                                                        │
│     在 completion_mask 上求均值 → 标量 loss                             │
│                                                                        │
│  ⑨ 更新 π_θ                                                           │
│     loss.backward() → clip_grad_norm → optimizer.step()               │
│     此时 model（π_θ）的参数已更新                                      │
│                                                                        │
│  ⑩ 周期性同步 rollout 引擎                                             │
│     每 save_interval 步：rollout_engine.update_policy(model)           │
│     下一次 rollout 用的 π_old 就是更新后的 π_θ                         │
│     （π_ref 永远不更新）                                                │
└────────────────────────────────────────────────────────────────────────┘
```

**关键：三次 forward pass 用的是三个不同的模型。** ② 用 $\pi_{\text{old}}$ 生成数据并记录 log prob（ratio 分母），⑤ 用当前 $\pi_\theta$ 重新算 log prob（ratio 分子），⑥ 用冻结的 $\pi_{\text{ref}}$ 算 log prob（KL 锚点）。梯度只从 ⑤ 回传——②⑥ 都在 `no_grad` 下运行。

> 以上 4.0–4.6 各步骤的完整代码，见本章末尾[完整代码](#完整代码)。接下来讨论 GRPO 的实践案例和后续演进（Part 5–7），完整可运行代码附在全章最后。

---

## Part 5: DeepSeek-R1——GRPO 最成功的实践

GRPO 的 per-response advantage（2.2 讨论过的 credit assignment 退步）决定了它**天然适配某些任务、不适配另一些**。适合的任务有两个共同特征：

1. **输出是原子的**——模型一口气生成整条回复，中间没有可以独立评价的步骤。Math reasoning 是一条推理链、code 是一段完整代码，对了就全对、错了就全错，per-response advantage 不会冤枉任何"正确的中间步"。
2. **Reward 可自动验证**——math 对答案、code 跑测试，不需要人工标注或 Reward Model。

```
适合 GRPO 的任务:
  Math:  prompt → [推理链]    → 对答案 → 对/错
  Code:  prompt → [一段代码]  → 跑测试 → 过/不过
                   ^^^^^^^^
                   原子输出，不可拆分

不适合 GRPO 的任务（见 16 章）:
  Agent: prompt → [思考→工具调用→观察→...→最终回答]  → 成功/失败
                   ^^^^   ^^^^^^^^
                   多步轨迹，每一步都是独立决策
```

DeepSeek-R1 正是把 GRPO 用在它最擅长的场景——math reasoning——训出了业界顶级的推理能力。理解 R1 的训练管线有助于理解 MiniMind 的设计决策。

```
DeepSeek-V3-Base
      ↓
Stage 1: Cold Start SFT（几千条长 CoT）
      ↓
Stage 2: Reasoning RL（GRPO + 纯规则 reward）
      ↓
Stage 3: Rejection Sampling + SFT（~80 万条混合数据，从 V3-Base 重新 SFT）
      ↓
Stage 4: Full RL（GRPO + accuracy + helpfulness + harmlessness）
      ↓
DeepSeek-R1
```

### Stage 1: Cold Start SFT

| 项目 | 细节 |
|------|------|
| 基座模型 | DeepSeek-V3-Base |
| 数据量 | 数千条（论文原文 "thousands"，未公开精确数字） |
| 数据来源 | ① few-shot prompting（给 V3 一个长 CoT 示例，让它仿写）；② 直接 prompting（让模型写详细的反思+验证过程）；③ 从 R1-Zero（纯 RL 训出的中间版本）的输出中筛选可读性好的；④ 人工后处理润色 |
| 数据内容 | **math + code 为主**的长推理链，格式为 `<think>逐步推理...</think>最终答案` |
| 目的 | 不是教会模型推理——几千条数据教不会。而是给模型一个"格式起点"：学会输出 `<think>...</think>` 结构。真正的推理能力由 Stage 2 的 RL 训出来 |
| 为什么不跳过 | DeepSeek 做过实验（R1-Zero）：直接从 base model 做 RL，会出现语言混杂（中英夹杂）、重复循环、格式崩溃。Cold Start SFT 解决这些表面问题，让 RL 能专注学推理 |

### Stage 2: Reasoning RL

| 项目 | 细节 |
|------|------|
| 起点 | Stage 1 的 SFT 模型 |
| 算法 | GRPO |
| 训练数据 | **math 和 code 的 prompt**（不需要标注答案，只需要题目）。Reward 由规则自动验证 |
| Reward 规则 | 只有两条：① **accuracy reward**——math 对答案（精确匹配）、code 跑测试用例（pass/fail）；② **format reward**——回复必须包含 `<think>...</think>` 结构 |
| 不需要 RM | 这是 R1 最重要的发现：纯规则 reward（不需要 Reward Model）就能训出强推理能力 |
| 采样参数 | 每个 prompt 采样 $G=16$ 个回复，max length 32768 token，每步 32 个 unique prompt（batch size = 32 × 16 = 512） |
| 关键现象 | 训练过程中模型自发涌现"aha moment"——学会反思（"wait, let me re-check"）、回溯（"I made an error, let me redo"）、自我验证。这些行为不在 SFT 数据里，是 RL 探索出来的 |

### Stage 3: Rejection Sampling + SFT

| 项目 | 细节 |
|------|------|
| 动机 | Stage 2 模型推理很强，但**通用能力严重退化**（写作、翻译、闲聊等变差）。需要把推理能力和通用能力合并 |
| 数据生成 | 用 Stage 2 模型对 math/code/science/逻辑推理等 prompt 生成多个候选回复，按 accuracy 筛选最优的（rejection sampling） |
| 混合数据 | 筛出的推理数据 + 通用 SFT 数据（写作、QA、翻译等），**总计约 80 万条**（论文原文 "roughly 800k training samples"，未公开两部分各自的精确比例） |
| 训练方式 | **从 DeepSeek-V3-Base 重新做 SFT**（不是接着 Stage 2 的 checkpoint 继续训）。这很关键——Stage 2 模型已经"偏"了，从 base 重训能拿到更干净的起点 |

### Stage 4: Full RL

| 项目 | 细节 |
|------|------|
| 起点 | Stage 3 的 SFT 模型 |
| 算法 | GRPO |
| Reward | 不再只有规则。加入 helpfulness reward（RM 打分）和 harmlessness reward（安全 RM），做最终对齐 |
| 目的 | 推理能力已经在 Stage 2-3 训好了，Stage 4 负责"打磨"——让模型在推理强的同时也有用、安全 |

**MiniMind 的设计受 Stage 2 启发**：用纯规则 reward（长度分 + 格式分 + RM 打分 + 重复惩罚），不需要人工标注偏好数据。但 MiniMind 比 R1 多用了一个 RM（语义质量打分），因为 MiniMind 的训练数据不像 R1 那样有大量可自动验证的 math/code 题。

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
| **DAPO** | **去 KL + Clip-Higher + 动态采样 + token-level loss + 超长软惩罚（详见下方 Part 6）** | **ByteDance (2025)** |
| Dr. GRPO | 去掉长度归一化 $1/\|o_i\|$，避免奖励长回复的偏置 | Liu et al. (2025) |
| REINFORCE++ | GRPO 的极简变体——去掉 clip，只保留 z-score + KL | Hu (2025) |
| PRIME | Process Reward Model + 在线 RL，per-step advantage | Cui et al. (2025) |
| Leave-One-Out | baseline 用其他 $G-1$ 个样本的均值，不含自己 | Ahmadian et al. (2024) |
| **GSPO** | **序列级重要性比率 + 长度归一化，解决 MoE 训练不稳定（详见下方 Part 7）** | **Qwen / 阿里 (2025)** |
| **R3** | **Rollout Routing Replay——推理时缓存 MoE 路由索引，训练时回放（详见下方 Part 7）** | **(2025)** |
| **GEPO** | **组级重要性权重——分母从单样本概率换成组期望概率（详见下方 Part 7）** | **鹏城 & 上海 AI Lab (ICLR 2026)** |

---

## Part 6: DAPO——GRPO 的进一步简化

> **DAPO = Decoupled Clip and Dynamic Sampling Policy Optimization**
> 论文：DAPO: An Open-Source LLM Reinforcement Learning System (ByteDance Seed, 2025)
> 用于 DeepSeek-R1 等推理模型的对齐训练

### 6.1 动机：PPO → GRPO → DAPO 的简化链

```
PPO 需要 5 个模型:  Actor + Old Policy + Critic + Reference + RM
                    ↓ GRPO 去掉 Critic
GRPO 需要 4 个模型: Actor + Old Policy + Reference + RM
                    ↓ DAPO 去掉 Reference + KL
DAPO 需要 3 个模型: Actor + Old Policy + RM/规则
```

GRPO 去掉了 Critic，但仍然保留了 Reference Model 和 KL 惩罚来防止策略跑偏。DAPO 的核心问题是：**KL 惩罚本身就在限制模型探索——能不能用其他机制替代 KL，把 Reference Model 也省掉？**

### 6.2 去 KL：为什么敢不要 Reference Model

PPO/GRPO 都用 $\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 约束策略不要偏离 reference 太远。DAPO 直接把这个约束和 Reference Model 一起去掉，动机是：

**长 CoT 推理模型的分布本应大幅偏离初始模型。** R1 式的反思、回溯、自我纠错行为在 SFT 模型的分布里几乎不存在——把策略钉在 ref 附近，等于禁止这些行为涌现。KL 惩罚从"防跑偏的保险"变成了"限制探索的枷锁"。

```
GRPO:  loss = clip_loss - β × KL(π_θ || π_ref)     ← 需要 reference model 做前向传播
DAPO:  loss = clip_loss                              ← 不需要 reference model
```

**去 KL 之后靠什么稳定训练？** 先想清楚 KL 到底在防什么——KL 的真正目的不是"保持和 reference 像"，而是**防止模型训崩**（熵塌缩、reward hacking、生成退化）。它通过"把策略钉在 reference 附近"来间接实现这个目标。DAPO 的思路是：**不再关心偏离 reference 多远，转而用更细粒度的机制直接防止训崩的各种症状**：

```
KL 惩罚（一刀切）: 把策略钉在 reference 附近 → 间接防训崩，但也限制探索

DAPO（精细替代）: 去掉 KL，用四个机制分别对症下药
  Clip-Higher       → 直接防熵塌缩（KL 最核心的作用）
  动态采样           → 防零梯度，保证每个 batch 都有有效学习信号
  Token-Level Loss  → 防长回复的梯度信号被稀释
  超长软惩罚         → 防截断带来的 reward 噪声
```

工程上还省了一个模型的显存（7B 的 BF16 推理态权重约 14GB）和一次前向传播。

### 6.3 DAPO 的四个核心创新

#### 创新 1：Clip-Higher——不对称 clip 区间

PPO/GRPO 的 clip 是对称的：$\operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$，通常 $\varepsilon=0.2$。DAPO 用**不对称区间**：下界保留，**上界抬高**：

$$\operatorname{clip}(r_t, \; 1-\varepsilon_{\text{low}}, \; 1+\varepsilon_{\text{high}}) \quad \text{where} \quad \varepsilon_{\text{low}}=0.2, \; \varepsilon_{\text{high}}=0.28$$

注意：下界 $1-\varepsilon_{\text{low}}$ 没有去掉，是**上界从 1.2 抬到 1.28**——所以叫 Clip-**Higher**。

**动机：防熵塌缩（entropy collapse）。**

回顾 ratio 的含义：$r_t = \pi_\theta(\text{token}) / \pi_{\text{old}}(\text{token})$。clip 限制的是这个**比值**，不是概率本身。ratio 上限 1.2 意味着 $\pi_\text{new} \leq 1.2 \times \pi_\text{old}$。

用具体数字看问题：

```
对称 clip (ε=0.2)，ratio 上限 1.2：

低概率 token:  pi_old = 0.01
  pi_new 最大 = 1.2 × 0.01 = 0.012，绝对涨幅 = 0.002

高概率 token:  pi_old = 0.5
  pi_new 最大 = 1.2 × 0.5 = 0.6，  绝对涨幅 = 0.1
```

**同样的 ratio 上限，高概率 token 每步绝对涨幅是低概率 token 的 50 倍。** 多轮更新后，高概率 token 越来越 dominant，低概率 token 始终抬不起来，策略熵快速下降，模型过早收敛到少数模式、丧失探索——这就是熵塌缩。

抬高上界到 1.28 怎么缓解：

```
Clip-Higher (ε_high=0.28)，ratio 上限 1.28：

低概率 token:  pi_old = 0.01
  pi_new 最大 = 1.28 × 0.01 = 0.0128，绝对涨幅 = 0.0028（比 0.002 多 40%）

高概率 token:  pi_old = 0.5
  pi_new 最大 = 1.28 × 0.5 = 0.64，  绝对涨幅 = 0.14（比 0.1 多 40%）
```

虽然两边都多了 40%，但**低概率 token 本来被卡得最死**（每步只涨 0.002），额外的空间对它们是"能不能活下来"的区别。多轮累积下来，低概率 token 有更大机会被探索到，策略不会过早锁死在少数 token 上。

```
PPO/GRPO 对称 clip (ε=0.2):
  好 token（A>0）: ratio 上限 1.2  → 低概率 token 每步涨幅极小 → 熵塌缩
  坏 token（A<0）: ratio 下限 0.8  → 下降通道始终在

DAPO Clip-Higher (ε_low=0.2, ε_high=0.28):
  好 token: ratio 上限 1.28 → 每步涨幅放宽，低概率 token 受益最大 → 熵维持
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

### 6.4 PPO vs GRPO vs DAPO 对比

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

### 6.5 面试话术（3 句话版本）

> "DAPO 是字节 Seed 团队在 GRPO 基础上为长 CoT reasoning 做的改进：去掉 KL，再加四个技术。去 KL 的动机是推理模型的分布本应大幅偏离初始模型——反思、回溯这些行为在 ref 分布里几乎不存在，KL 惩罚会压制这类探索。四个技术：第一，Clip-Higher，不对称 clip 区间，下界保留 ε_low=0.2、上界抬到 ε_high=0.28，动机是防熵塌缩——对称 clip 下低概率 token 的概率上升被压制，策略熵会快速下降。第二，Dynamic Sampling，过滤掉全对全错的 prompt，保证每个 batch 都有非零梯度。第三，Token-Level Policy Gradient Loss，把 loss 聚合粒度从 sample 级改成 token 级，解决长回复 per-token 梯度被稀释的问题。第四，Overlong Reward Shaping，对接近长度上限的回复给线性软惩罚，减少截断带来的 reward 噪声。本质上，DAPO 去掉了 GRPO 依赖 ref 的约束，用四个针对性机制分别解决熵塌缩、零梯度、梯度稀释、超长噪声这四个 reasoning 训练的具体失败模式。"

---

## Part 7: MoE 模型的 RL 训练——从路由噪声到三级修正

> 本节涵盖三个方案：
> - **R3**（Rollout Routing Replay, 2025）：工程层面消除路由不一致
> - **GSPO**（Group Sequence Policy Optimization, Qwen, 2025, arXiv: 2507.18071）：ratio 从 token 级 → 序列级
> - **GEPO**（Group Expected Policy Optimization, 鹏城实验室 & 上海 AI 实验室, ICLR 2026）：importance weight 分母从单样本 → 组期望

**与 Part 6 DAPO 的区别**：DAPO 和本节的方案解决的是**完全不同的问题**，思路甚至相反：

```
DAPO（ByteDance，面向推理模型）:
  问题: KL 约束太强，限制了模型探索（学不会反思、回溯等新行为）
  解法: 去掉 KL + Reference Model，放手让模型跑远
  保留: token-level ratio

R3 / GSPO / GEPO（面向 MoE 模型）:
  问题: token-level ratio 噪声太大，MoE 的 expert 路由变化导致训练崩溃
  解法: 消除路由不一致（R3）或改变 ratio 粒度（GSPO/GEPO）
  保留: KL + Reference Model（约束没问题，ratio 计算方式才是问题）
```

### 7.1 问题：为什么 GRPO 在 MoE 上崩溃

GRPO 的重要性比率是 **token 级**的：

$$w_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}$$

每个 token 独立算新旧策略的概率比。这在 dense 模型上没问题，但 **MoE 模型有两层不一致叠加**：

**第一层：参数更新导致的路由漂移。** 每次梯度更新后，router 参数变了，同一个 token 可能被分到不同的 expert。这不是概率分布的平滑变化，而是计算路径的**结构性切换**——token 走了完全不同的 expert，输出概率可能剧烈波动。Qwen3-30B-A3B-Base 训练实测，每次梯度更新后约 10% 的 expert 路由发生变化。

**第二层：路由器自身的随机性。** 即使输入完全相同、模型参数完全不变，router 的 top-k 选择在某些实现中也存在数值抖动（如 softmax 后的 tie-breaking），导致两次前向传播可能激活不同的 expert。

这双重不一致叠加，使得 MoE 模型的训推 KL 散度显著高于 dense 模型。有实验报告，在相同条件下 MoE 的平均训推 KL 约为 dense 的 2.4 倍（具体倍数因模型和实验设置而异）。

```
MoE 的 expert 路由问题:

  每次梯度更新后，同一个 rollout 样本：
  - 约 10% 的 expert 路由发生变化（Qwen3-30B-A3B-Base 实测）
  - 同一个 token，更新前走 expert 3，更新后走 expert 7
  - 不同 expert 产出完全不同 → 概率 π_θ(y_{i,t}|...) 剧烈波动
  
  token-level ratio 的后果：
  - 单个 token 的 ratio 可能从 0.5 跳到 5.0
  - clip 只能截断极端值，不能消除底层噪声
  - 大量 token 被 clip → 有效梯度信号趋近于零 → 训练打不动
```

核心洞察：**token-level ratio 的波动会随回复长度累积**。回复越长，有越多 token 的路由发生变化，梯度噪声越大。

**为什么不用 Monte Carlo 多次采样来降低方差？** 理论上可以对同一批输入做 $M$ 次前向传播，每次让 router 独立采样，对输出做平均。但在 GRPO 的场景中，每个 prompt 已经要生成 $G$ 个回复（$G=8 \sim 16$），如果每个回复内部再做 $M=5$ 次路由采样，计算量变成 $G \times M = 40 \sim 80$ 倍前向计算，工程上不可接受。工业界的主流思路是从**工程层面消除路由不一致**（7.2 R3）或从**算法层面改变 ratio 的计算粒度**（7.3 GSPO、7.6 GEPO），而不是用更多采样硬扛随机性。

### 7.2 工程解法：R3（Rollout Routing Replay）

> **R3 = Rollout Routing Replay**（2025）
> 核心操作只有一步：推理时的路由决策记录下来，训练时照着跑。

R3 的思路最直接——既然问题出在训推路由不一致，那就**缓存推理时的路由结果，训练时原样回放**：

```
② Rollout 阶段（用 π_old 生成回复，torch.no_grad）:
   正常前向传播，router 做 top-k 选择
   除了记录 per_token_logps，额外记录每层 router 的 top-k 索引
   → routing_mask: [B*G, num_layers, seq_len, k]   每层每 token 选了哪 k 个 expert

⑤ Policy Forward 阶段（用 π_θ，参与梯度计算）:
   前向传播时不重新执行 router 选择
   直接用 ② 缓存的 routing_mask 激活对应的 expert
   → 训练和推理走的 expert 完全一致
   → ratio 反映的是真实的策略变化，不掺杂路由噪声
```

**成本极低**：routing mask 就是整数索引（每层每 token 存 $k$ 个 int），存储开销几乎可忽略；训练 forward 时省去了 router 的 softmax + top-k 计算（虽然这部分本来也不大）。实验报告 R3 能把训推 KL 散度降低一个数量级以上，训练曲线从剧烈震荡变成稳定收敛。

**天然兼容现有缓存机制**：rollout 阶段本来就要缓存 KV Cache（4.6 的步骤 ②），routing mask 只是在 KV Cache 旁边多存一组整数序列。多轮对话和 preface caching（缓存 system prompt 的 KV）场景同理。

### 7.3 算法解法一：GSPO——序列级重要性比率

> **GSPO = Group Sequence Policy Optimization**
> 论文：Group Sequence Policy Optimization (Qwen Team, Alibaba, 2025)
> arXiv: 2507.18071 | 用于 Qwen3 系列模型训练

R3 从工程上消除了路由不一致，GSPO 从算法上改变了 ratio 的计算粒度，让它对路由噪声天然鲁棒。

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

### 7.4 GSPO vs GRPO 的目标函数对比

先回顾 GRPO 的目标函数，再看 GSPO 改了什么。

**GRPO 的目标函数**（Part 2 的公式）：

$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\Big(w_{i,t}(\theta)\hat{A}_i,\;\text{clip}\big(w_{i,t}(\theta),\,1-\varepsilon,\,1+\varepsilon\big)\hat{A}_i\Big)\right]$$

注意这个公式里有**两层求和**：外层遍历 $G$ 条回复，内层遍历每条回复的每个 token。每个 token 有自己的 ratio $w_{i,t}$，但共享同一个 advantage $\hat{A}_i$。


用具体例子：一条好回复（$\hat{A}_i > 0$）有 100 个 token，其中第 37 个 token 的 ratio 已经到了 1.5（新策略比旧策略更喜欢它），clip 到 1.2 后，这个 token 的梯度被截断——**防止单个 token 被过度强化**。其他 ratio 在正常范围的 token 正常更新。

**GSPO 的目标函数**：

$$J_{\text{GSPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\Big(s_i(\theta)\hat{A}_i,\;\text{clip}\big(s_i(\theta),\,1-\varepsilon,\,1+\varepsilon\big)\hat{A}_i\Big)\right]$$

advantage 的计算和 GRPO 一样是组内 z-score：

$$\hat{A}_i = \frac{r(x, y_i) - \text{mean}(\{r(x, y_j)\}_{j=1}^G)}{\text{std}(\{r(x, y_j)\}_{j=1}^G)}$$

对比两个公式，**唯一区别是 ratio 从 token 级 $w_{i,t}$ 换成了序列级 $s_i$**：

```
GRPO: 每个 token 各自算 ratio，各自 clip    → ratio 和 advantage 粒度不一致
GSPO: 整条回复共享一个 ratio，统一 clip      → ratio 和 advantage 粒度一致（都是 sequence-level）
```

GSPO 把 ratio 的粒度**对齐到了 reward 的粒度**——reward 是 per-sequence 的，advantage 是 per-sequence 的，ratio 也应该是 per-sequence 的。这个对齐不仅解决了 MoE 训练不稳定的问题，逻辑上也更自洽。

### 7.5 反直觉发现：clip 更多，效果更好

GSPO 训练中被 clip 掉的 token 数量是 GRPO 的 **约 100 倍**——因为序列级 ratio 一旦触发 clip，整条回复的所有 token 都被 clip（而 GRPO 只 clip 少数极端 token）。

```
GRPO:  clip 1% 的 token  → 其余 99% 的 token 带着噪声梯度通过
GSPO:  clip 100% 的 token（按序列）→ 但每条通过的序列信号干净

直觉：宁可丢掉更多样本，也不让噪声梯度污染更新方向
```

这说明 GRPO 的低 clip 率不是优势，而是 **token-level ratio 天然分散导致难以触发 clip**——很多有问题的序列没被 clip 挡住。

### 7.6 算法解法二：GEPO——组级重要性权重

> **GEPO = Group Expected Policy Optimization**
> 鹏城实验室 & 上海 AI 实验室，ICLR 2026 收录

如果说 GRPO 的 ratio 在 token 级，GSPO 推进到序列级，GEPO 则进一步推进到**组级**。

回顾 GRPO 的 importance weight（Part 3.1 的 ③）：

$$w_i = \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}$$

分母是"这条回复在旧策略下的概率"——**每条回复和过去的自己比**。

GEPO 的改动：分母换成**整组 $G$ 条回复在旧策略下的概率期望**：

$$w_i^{\text{GEPO}} = \frac{\pi_\theta(y_i \mid x)}{\frac{1}{G}\sum_{j=1}^{G}\pi_{\theta_{\text{old}}}(y_j \mid x)}$$

**每条回复和整组平均水平比**。

这一步为什么关键？用具体数字看 GSPO 没完全解决的问题：

```
G=4，4 条回复在 π_old 下的序列概率: [0.3, 0.2, 0.001, 0.25]
                                                   ↑ 路由扰动导致极低

第 3 条回复的 importance weight 分母:

GRPO:   分母 = π_old(y_3) = 0.001           → weight 可能炸到数百
GSPO:   分母 = π_old(y_3) 的序列级版本       → 仍以该样本自身概率为基准，极端时仍不稳
GEPO:   分母 = mean(0.3, 0.2, 0.001, 0.25) = 0.188  → 稳定，不受单样本异常影响
```

**关键洞察**：GSPO 的序列级平均改变了 ratio 的**计算粒度**（token → sequence），但分母仍然是该样本自身在 $\pi_{\text{old}}$ 下的概率。当整条回复因为大面积路由变化导致序列概率极低时，GSPO 的分母也会接近零。GEPO 把分母的**参考基准**从"自己"换成"组平均"——一条异常不影响整体。

GEPO 论文给出了理论证明：在 trainer（$\pi_\theta$）和 sampler（$\pi_{\theta_{\text{old}}}$）之间 KL 散度较高的情况下，GEPO 的 importance weight 方差**指数级低于** GRPO 和 GSPO（论文 Theorem，具体 bound 因假设条件而异）。实验报告在极端路由扰动场景下，GEPO 的性能下降不到 3%，而 GRPO 和 GSPO 直接崩溃（论文数据，具体数字因实验设置而异）。

### 7.7 三种方案的关系：消除噪声 vs 平滑噪声

R3、GSPO、GEPO 解决的是同一个问题（MoE + RL 不稳定）的**不同层面**，是**组合关系而非替代关系**：

| 方案 | 解决什么 | 怎么解决 |
|------|---------|---------|
| R3 | 路由不一致的**根源** | 缓存推理路由，训练回放——消除噪声源 |
| GSPO | token 级 ratio 的**噪声累积** | ratio 从 token 级聚合到序列级——平滑噪声 |
| GEPO | 单样本分母的**极端值风险** | 分母从单样本概率换成组期望概率——稳定参考基准 |

```
没有 R3，没有 GSPO/GEPO:
  ratio 噪声 = 路由噪声 + 策略变化噪声            → 训练崩溃

有 R3，没有 GSPO/GEPO:
  ratio 噪声 = 策略变化噪声（路由噪声被消除）      → 显著改善，KL 降一个数量级
  但 token 级 ratio 仍有累积噪声

没有 R3，有 GSPO:
  ratio 噪声 = 路由噪声 + 策略变化噪声，被序列级平均抹平
  → 单独就够用（Qwen3 全系列验证过）

有 R3 + 有 GSPO/GEPO:
  路由噪声被 R3 消除 + 剩余噪声被序列/组级方案平滑  → 最稳
```

工程选择：R3 成本极低（多存几个整数索引），GSPO/GEPO 是改算法。在实践中，**GSPO 单独已足够稳定**（Qwen3 全系列验证过，不需要 R3）。R3 的价值在于它是通用的——不管用哪种 loss 变体（GRPO/GSPO/GEPO），R3 都能叠加使用。

### 7.8 PPO vs GRPO vs DAPO vs GSPO vs GEPO 对比

| 特性 | PPO | GRPO | DAPO | GSPO | GEPO |
|------|-----|------|------|------|------|
| 重要性比率粒度 | token 级 | token 级 | token 级 | **序列级** | **组级** |
| Critic | 要 | 不要 | 不要 | 不要 | 不要 |
| Reference Model | 要 | 要 | 不要 | 要 | 要 |
| KL 惩罚 | 有 | 有 | 无 | 有 | 有 |
| Clip 方式 | 对称 | 对称 | 不对称 | 对称 | 对称 |
| Loss 聚合 | per-token（GAE） | sample 级 | token 级 | **序列级** | **组级** |
| MoE 训练稳定性 | 差 | 差 | 未验证 | **好** | **最好（理论方差指数级更低）** |
| 需要 R3 | — | 需要 | — | 不需要 | 不需要 |
| 解决的核心问题 | 通用 RL | 去 Critic 省显存 | 去 KL 释放探索 | MoE 路由梯度噪声 | MoE 极端路由扰动 |
| 代表模型 | InstructGPT | DeepSeek-R1 | 字节开源推理模型 | **Qwen3** | — |

### 7.9 面试话术

> "MoE 模型做 RL 训练的核心问题是训推不一致——路由器的离散 top-k 采样导致每次参数更新后，同一个 token 可能被分到不同 expert，token-level ratio 剧烈波动，量化数据显示 MoE 的训推 KL 约为 dense 的 2.4 倍。
>
> 解决方向有两条。工程上用 R3——rollout 时记录每层 router 的 top-k 索引，训练时直接回放，成本几乎为零，KL 散度降一个数量级。
>
> 算法上有两个层次。GSPO（Qwen3 团队）把 ratio 从 token 级聚合到序列级——对所有 token 的 log-ratio 取几何平均，个别 token 因路由变化产生的极端值被抹平。GEPO（鹏城实验室，ICLR 2026）更进一步，把 importance weight 的分母从单样本概率换成组内 G 个样本的概率期望——即使某条回复因路由扰动概率接近零，分母也不会跟着崩，理论上方差指数级降低。
>
> 这三个方案是组合关系：R3 消除路由噪声的根源，GSPO/GEPO 平滑剩余噪声。实践中 GSPO 单独已足够稳定，Qwen3 全系列验证过。"

---

## 面试话术

> "GRPO 是 DeepSeek 提出的 RL 对齐方法，核心改动是扔掉 PPO 的 Critic。
>
> PPO 用 Critic 网络 V(s_t) 估 value baseline 来算 per-token advantage，但 Critic 显存大、难训准。GRPO 换了一个思路：对每个 prompt 采样 G 个回复，reward 在组内做 z-score 归一化得到 advantage——用组均值替代 value baseline，'这个回复比同组平均水平好多少'。
>
> 省了 value model，但 advantage 从 per-token 退化成 per-response——同一条回复的所有 token 共享同一个 advantage。对 math reasoning 没问题，最终答案对了就是全对。但对 agent 场景有根本缺陷——5 步正确 + 1 步格式错，per-response advantage 会惩罚所有 token。
>
> DeepSeek-R1 进一步证明，GRPO + 纯规则 reward 就能训出强推理能力——涌现了反思、回溯等行为，不是 SFT 教的而是 RL 自己'长出来的'。这是 LLM RL 最激动人心的发现。"

---

## 完整代码

以上 Part 4 各步骤的代码片段拼成三个完整可运行文件：

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


### 本章实现覆盖范围

本章代码实现了 GRPO 训练的完整流程：

- **GRPO 训练主循环**：rollout → reward → advantage → policy loss → 梯度更新
- **Rollout 引擎**：用当前 Policy 在线生成回复 + 记录 per-token logprob
- **RLAIFDataset**：只提供 prompt，回复由 rollout 在线生成（与 SFT 数据集的关键区别）
- **两种 loss 变体**：标准 GRPO（PPO-clip）和 CISPO
- **KL 散度**：Schulman k3 近似（无需遍历词表）
- **混合 Reward**：规则分（长度 + 格式 + 重复惩罚）+ Reward Model 打分

未实现（仅讲解原理）：

- Part 6 DAPO 的四个改进（Clip-Higher、Dynamic Sampling、Token-Level Loss、Overlong Reward Shaping）
- Part 7 的 R3 路由回放、GSPO 序列级重要性比率、GEPO 组级重要性权重

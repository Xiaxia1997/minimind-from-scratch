# 15 - Agent RL：多轮工具调用的强化学习

> 源码位置：`trainer/train_agent.py`（训练主循环 + 工具定义 + 多轮 rollout + reward 计算）
> 依赖：`dataset/lm_dataset.py:226-252`（AgentRLDataset）、`trainer/rollout_engine.py`（推理引擎）
> 前置：[14-GRPO](14-GRPO.md)（GRPO 的 loss 和 advantage 计算）

## 一句话总结

Agent RL 在 GRPO 的基础上增加了**多轮工具调用**：模型生成回复 → 如果包含 `<tool_call>`，解析并执行工具 → 把结果拼回上下文 → 模型继续生成 → 直到没有工具调用或达到最大轮次。Reward 同时评估工具调用的正确性（选对工具、传对参数、拿到正确结果）和最终回答的质量。

---

## Part 1: 为什么 Agent 需要 RL

### SFT 教 Agent 的局限

SFT 可以教模型工具调用的格式——"当需要查天气时，生成 `<tool_call>{"name": "get_weather", ...}</tool_call>`"。但 SFT 有两个根本问题：

```
问题 1: 无法学习"何时该调工具"
  SFT 数据是固定的——每个问题对应固定的工具调用序列。
  模型学到的是"看到类似的问题就调用工具"，而不是"判断这个问题需不需要工具"。
  
问题 2: 无法学习"多轮规划"
  工具调用是多步过程：先调 A 拿到结果，根据结果决定是否调 B。
  SFT 只能模仿固定的调用序列，无法学习动态规划。
  
RL 的解决方案:
  让模型自己生成工具调用 → 执行工具拿到真实结果 → 根据最终答案质量打分
  → 梯度反向传播，模型学会"什么时候该调什么工具"
```

### Agent RL = GRPO + 多轮交互环境

```
GRPO (上一章):
  prompt → 生成 1 轮回复 → reward → 更新

Agent RL (本章):
  prompt → 生成回复 → 包含 tool_call？
           ├── 否 → 直接 reward → 更新
           └── 是 → 解析工具调用 → 执行工具 → 结果拼回上下文
                    → 生成下一轮回复 → 还有 tool_call？
                         ├── 否 → 最终 reward → 更新
                         └── 是 → 继续循环（最多 3 轮）
```

---

## Part 2: 工具定义与模拟执行

### 可用工具

```python
# train_agent.py:40-47
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", ...}},      # 数学计算
    {"type": "function", "function": {"name": "unit_converter", ...}},      # 单位换算
    {"type": "function", "function": {"name": "get_current_weather", ...}}, # 天气查询
    {"type": "function", "function": {"name": "get_current_time", ...}},    # 时间查询
    {"type": "function", "function": {"name": "get_exchange_rate", ...}},   # 汇率查询
    {"type": "function", "function": {"name": "translate_text", ...}},      # 文本翻译
]
```

6 个工具覆盖了常见的 agent 场景：计算、查询、转换。每个工具都有完整的 JSON Schema 定义（函数名、描述、参数类型、必填项），这些信息会通过 `apply_chat_template(tools=...)` 注入到 system prompt 中。

### 模拟执行：不连真实 API

```python
# train_agent.py:50-64
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), ...}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ...}

MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(args.get("expression")))},
    "get_current_weather": lambda args: {"city": args["location"], "temperature": ...},
    "get_exchange_rate": lambda args: {"rate": EXCHANGE_DATA.get(...)},
    ...
}
```

RL 训练不连真实 API，原因：
1. **速度**：每个 step 要执行 batch_size × num_generations × max_turns 次工具调用，真实 API 太慢
2. **成本**：训练几千步，调用量巨大
3. **确定性**：RL 需要环境确定（相同输入 → 相同输出），真实 API 有变化（天气会变）
4. **安全**：`calculate_math` 用 `eval()`——训练环境可控，但绝不能接真实用户输入

### 参数校验

```python
# train_agent.py:67-74
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    ...
}
```

校验函数检查模型是否传了正确的参数。如果模型调了 `calculate_math` 但没传 `expression` 参数，校验失败 → reward 扣分。

---

## Part 3: 多轮 Rollout——Agent 的核心

### 单样本的多轮交互

```python
# train_agent.py:98-157 (rollout_single)
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, ...):
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []

    for turn in range(max_turns):
        # ① 把当前对话拼成 prompt
        context = tokenizer.apply_chat_template(messages, tools=tools, ...)
        inputs = tokenizer(context, return_tensors="pt", ...)

        # ② policy 生成回复
        rollout_result = rollout_engine.rollout(inputs, num_generations=1, ...)
        new_text = rollout_result.completions[0]

        # ③ 解析工具调用
        calls = parse_tool_calls(new_text)
        if not calls:
            break         # 没有工具调用，结束

        # ④ 执行工具，结果拼回上下文
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            result = execute_tool(call["name"], call["arguments"])
            messages.append({"role": "tool", "content": json.dumps(result)})

        # ⑤ 工具结果的 token 不参与 loss（mask=0）
        observe_ids = tokenizer(observe_context, ...)
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))     # ← 关键：tool 结果不算 loss

    return final_output, prompt_ids, response_ids, response_mask, response_old_logps, ...
```

#### response_mask 的设计

```
一个多轮 Agent 交互的 token 序列:

[system prompt + tools + user question]  [assistant 第1轮回复 + tool_call]  [tool result]  [assistant 第2轮回复]
├──────── prompt_ids ────────────────┤  ├─── response_ids (mask=1) ───┤  ├── mask=0 ──┤  ├── mask=1 ────────┤

mask=1: 模型生成的 token → 参与 loss 计算（模型对自己生成的负责）
mask=0: 工具返回的 token → 不参与 loss（这不是模型生成的，不该算 loss）
```

这个设计非常精妙：
- **模型生成的 tool_call** → mask=1 → 梯度信号鼓励/抑制这次工具调用
- **工具返回的结果** → mask=0 → 这是环境的反馈，不是模型的行为
- **模型基于工具结果的最终回复** → mask=1 → 梯度信号评估模型如何利用工具结果

### 批量 Rollout

```python
# train_agent.py:159-180 (rollout_batch)
def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, ...):
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):                # 每个 prompt 生成 num_gen 个独立轨迹
            msgs_copy = [dict(m) for m in messages]     # 深拷贝，因为 rollout_single 会修改 messages
            completion, ..., = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, ...)
```

注意 `msgs_copy`——每次生成都从原始 messages 开始，而不是在上一次生成的基础上继续。因为我们需要 N 条独立的轨迹来做组内比较。

---

## Part 4: Agent Reward 设计

### 无工具调用的情况

如果模型直接回答（没有调用工具），和 GRPO 的 reward 计算一样：

```python
# train_agent.py:203-218
if not tool_calls:
    reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5     # 长度分
    if '</think>' in response:
        think, answer = response.split('</think>', 1)
        reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5   # 思考分
    score = reward_model.get_score(messages, answer)                  # RM 分
    reward -= rep_penalty(answer)                                     # 重复惩罚
```

### 有工具调用的情况

```python
# train_agent.py:220-238
else:
    gt = gt_batch[sample_idx]                    # ground truth 标注

    # ① 工具选择正确性
    valid_call_count = 0
    for tool_call in tool_calls:
        name = tool_call.get("name", "")
        raw = tool_call.get("arguments", {})
        check = CHECK_ARGS.get(name)
        valid_call_count += int(bool(name in valid_names and check and check(raw)))

    tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)
    reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap              # 工具对齐分

    # ② 最终答案正确性
    verified = validate_gt_in_text(final_text, gt) if gt else set()
    if gt: reward += 2.5 * len(verified) / len(gt)                    # GT 验证分

    # ③ 未完成惩罚
    if unfinished: reward -= 0.5                                       # 达到 max_turns 还没结束

    reward -= rep_penalty(final_text)
    rewards[idx] = max(min(reward, 3.0), -3.0)                        # Clip 到 [-3, 3]
```

#### Reward 设计的层次

```
Reward 信号（从强到弱）:

+2.5  最终答案包含正确结果（GT 验证）    ← 最重要的信号
+1.0  思考过程格式正确且长度合理
+0.5  调用了正确数量和类型的工具
+0.5  回复长度合理
+RM   Reward Model 打分（仅无工具调用时）

-0.5  工具数量/类型不匹配（每多/少一个扣 0.5）
-0.5  未完成（3 轮还没结束）
-重复  重复 n-gram 惩罚
-3~+3  总分 Clip
```

#### ground truth (gt) 的作用

```python
# 训练数据中的 gt 字段示例
{"gt": ["7.21"]}               # 1美元=7.21人民币
{"gt": ["28°C", "晴"]}          # 北京天气
{"gt": ["Hello World"]}        # 翻译结果
```

`validate_gt_in_text` 检查模型最终回答中是否包含正确答案。这是一个弱校验——不要求格式完全一致，只要答案出现在文本中就算对。

```python
# train_agent.py:183-186
def validate_gt_in_text(text, gt_list):
    # 字符串匹配或数值匹配（容忍浮点误差 1e-6）
    return {g for g in gt_list if s.lower() in text.lower()
            or any(abs(float(g) - n) < 1e-6 for n in nums)}
```

---

## Part 5: Loss 计算——和 GRPO 共享

Agent RL 的 loss 计算和 GRPO 完全一样——PPO-clip + KL 惩罚。唯一的区别在于序列结构：

```
GRPO:
  [prompt] [completion]
  mask:  [0...0] [1...1]

Agent RL:
  [prompt] [assistant 回复 1] [tool result] [assistant 回复 2]
  mask:  [0...0] [1.........1] [0.........0] [1.........1]
            ↑ 不算 loss         ↑ 不算 loss     ↑ 算 loss
```

```python
# train_agent.py:319-332 (和 train_grpo.py 几乎完全一样)
grouped_rewards = rewards.view(-1, args.num_generations)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)

kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1
ratio = torch.exp(per_token_logps - old_per_token_logps)

# CISPO loss（默认）
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
```

### 关键超参数对比

| 超参数 | GRPO | Agent RL | 为什么不同 |
|--------|------|----------|-----------|
| `num_generations` | 6 | 4 | 多轮交互计算量大，减少每个 prompt 的生成数 |
| `max_gen_len` | 1024 | 768 | 每轮生成限制，但可能有 3 轮 |
| `max_total_len` | 无限制 | 2500 | 多轮拼接后序列很长，需要截断 |
| `thinking_ratio` | 0.9 | 0.1 | Agent 场景思考链不太重要 |
| `batch_size` | 2 | 2 | 相同 |
| `learning_rate` | 3e-7 | 3e-7 | 相同 |

---

## Part 6: 训练数据

### AgentRLDataset

```python
# dataset/lm_dataset.py:226-252
class AgentRLDataset(Dataset):
    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"])
            messages.append(message)
        return messages[:-1], tools        # 去掉最后一个 message（通常是 assistant 回复）
                                           # 因为 RL 让模型自己生成

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}
```

注意 `messages[:-1]`——去掉了训练数据中的 assistant 回复。RL 训练不需要"标准答案"（由模型自己生成），只需要 prompt（包含 system + user）和 ground truth（用于 reward 计算）。

```
SFT 数据:   system + user + assistant(标准回复)       → 监督学习，模仿标准回复
Agent RL:   system + user + gt(正确结果)              → 强化学习，模型自己探索
                               ↑ 不给回复，只给正确答案用于打分
```

---

## Part 7: GRPO 在 Agent 场景的根本性缺陷

Agent RL 沿用了 GRPO 的 loss 框架，但 GRPO 的 per-response advantage 在 agent 场景有三个根本性问题。

### 问题 1：per-response advantage → token 级信号错误

```
Agent 轨迹: 5 步正确的 tool call + 最后一步格式错误
  → 整个 episode 失败，reward = 0
  → GRPO 给所有 token 同样的负 advantage
  → 前 5 步正确的 tool call 也被惩罚了！

如果是 PPO:
  → GAE 从最后一步往前递推，delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
  → 越早的正确步骤，负 advantage 被 gamma*lambda 衰减得越小
  → 前面正确的步骤受到的惩罚远小于最后错误的步骤
  → credit assignment 更精确
```

这是 per-response 和 per-token advantage 的根本差异。Math reasoning 不怕——答案对了就是全对。Agent 场景每一步都独立重要。

### 问题 2：组内归一化在全部失败时崩溃

```
G=4 个 trajectory 全部失败:  r = [0, 0, 0, 0]
  → std = 0，advantage = 0 / (0 + 1e-4) ≈ 0
  → 梯度几乎为零，模型什么都没学到

一个勉强成功 (r=0.3)，三个完全失败 (r=0):
  → 归一化后 0.3 吃正 advantage
  → 但 0.3 的成功率很低
  → 学到的是"比烂"而不是"做好"
```

Agent 任务比 math 难得多——全部失败的概率远高于 math（math 有一半概率蒙对，agent 需要多步全对）。

### 问题 3：轨迹长度方差大

```
Math: 答案长度 100~500 token，reward 方差小
Agent: 轨迹 5~100 步，reward = 0 或 1（几乎是二值），方差极大
  → G=4 个轨迹的组内归一化非常不稳定
  → 需要增大 G 来降低方差，但 agent rollout 很贵
    （每个轨迹要多轮 tool call，推理成本是 math 的 3~5 倍）
```

### MiniMind 的缓解措施

MiniMind 的 Agent RL 用了几个技巧部分缓解，但无法根治：

| 措施 | 缓解了什么 | 无法解决什么 |
|------|-----------|-------------|
| response_mask（tool result 不算 loss） | 防止工具结果的 token 被错误惩罚 | 模型自己生成的 tool_call token 仍共享 per-response advantage |
| 多维 reward（工具正确性 + 格式 + GT） | 提供更密集的 reward 信号 | 所有维度最终合成一个标量，仍是 per-response |
| GT 弱监督（连续匹配分） | 避免纯 0/1 reward | 组内归一化的统计性质问题仍在 |
| `max_total_len=2500` 截断 | 控制序列长度方差 | 截断可能丢掉关键信息 |

### 可能的改进方向

| 方法 | 思路 | 代价 |
|------|------|------|
| Step-wise reward | 每步 tool call 后给即时奖励 | 需要设计精细的 step reward function |
| PRM (Process Reward Model) | 训一个小模型判断中间步骤对不对 | 需要标注 process label |
| GAE + Group Norm 混合 | GAE 做 per-token，group norm 替代 critic baseline | 引入 value function（又回来了） |
| 增大 G | 更多轨迹 → 更稳定的组内统计 | agent rollout 太贵 |

---

## 对照 MiniMind 源码

| 概念 | 我们讲的 | 源码位置 |
|------|---------|---------|
| 工具定义 | 6 个工具的 JSON Schema | `train_agent.py:40-47` |
| 模拟执行 | MOCK_RESULTS + CHECK_ARGS | `train_agent.py:57-74` |
| 工具调用解析 | `parse_tool_calls` | `train_agent.py:77-81` |
| 单样本多轮 rollout | `rollout_single` | `train_agent.py:98-157` |
| 批量 rollout | `rollout_batch` | `train_agent.py:159-180` |
| GT 验证 | `validate_gt_in_text` | `train_agent.py:183-186` |
| Reward 计算 | `calculate_rewards` | `train_agent.py:188-239` |
| response_mask | tool result mask=0 | `train_agent.py:151` |
| 数据集 | `AgentRLDataset` | `lm_dataset.py:226-252` |
| Loss | 和 GRPO 完全一样 | `train_agent.py:322-332` |

---

## 业界对比

| 特性 | MiniMind Agent RL | OpenAI Function Calling | Claude Tool Use | Qwen-Agent |
|------|-----------------|----------------------|----------------|------------|
| 训练方式 | GRPO + 多轮 rollout | 未公开（推测 RL） | 未公开（推测 RLHF） | SFT + RL |
| 工具数量 | 6 个 | 数百个 | 数十个 | 可扩展 |
| 多轮交互 | 最多 3 轮 | 无限轮 | 无限轮 | 无限轮 |
| 工具执行 | 模拟 | 真实 API | 真实 API / 沙箱 | 真实 API |
| Reward 来源 | 规则 + RM + GT | 未公开 | 未公开 | 规则 + RM |
| response_mask | 工具结果不算 loss | 未公开 | 未公开 | 类似 |

几个值得注意的点：

1. **response_mask 的设计是 Agent RL 的关键**——工具返回的结果是环境反馈，不应该对模型的 loss 有直接贡献。模型只应该对"自己决定调用什么工具"和"如何利用工具结果"负责

2. **GT 验证是一种弱监督**——不需要人工标注完整的工具调用序列，只需要标注最终的正确答案。模型自己探索如何通过工具调用得到正确答案

3. **模拟执行 vs 真实执行**：训练用模拟，推理/部署用真实 API。这是一个 sim2real gap——训练时的模拟环境可能和真实 API 行为不完全一致

4. **安全问题**：`calculate_math` 的模拟执行用了 `eval()`，训练环境可控。但部署时必须用沙箱隔离——恶意用户可能通过 prompt injection 让模型生成危险的代码

---

## 面试话术

> "Agent RL 在 GRPO 基础上增加了多轮工具调用。每个 prompt 的 rollout 是一个最多 3 轮的循环：
> 模型生成回复，如果包含 `<tool_call>` 标签就解析执行工具，把结果拼回上下文让模型继续生成。
>
> 关键设计是 response_mask——模型生成的 token 设 mask=1 参与 loss，
> 工具返回的结果设 mask=0 不参与 loss。这确保梯度信号只影响模型的决策
> （选什么工具、传什么参数、如何利用结果），而不是评价工具本身。
>
> Reward 设计分两类：无工具调用时用长度+格式+RM 打分；
> 有工具调用时验证工具选择正确性（数量、类型、参数）和最终答案的 GT 匹配。
> GT 是弱监督——只标注正确结果，不标注调用序列，模型自己探索最优路径。
>
> 但 GRPO 框架在 agent 场景有根本性缺陷——per-response advantage 让同一轨迹的所有 token 共享
> 一个 advantage 分数。如果 5 步正确 + 1 步格式错导致失败，前 5 步也被惩罚了。
> PPO 的 per-token advantage（GAE 递推）能做更精确的 credit assignment，
> 但代价是需要 critic 模型和更多显存。这是 GRPO 在 math reasoning 和 agent 场景的核心 trade-off。"

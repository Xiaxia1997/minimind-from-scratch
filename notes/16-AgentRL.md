# 16 - Agent RL：多轮工具调用的强化学习

> **项目位置**
> ```
> scratch/
> ├── trainer/
> │   └── train_agent.py       ← 本章手写（工具定义 + 多轮 rollout + reward + 训练主循环）
> └── dataset/
>     └── lm_dataset.py        ← 追加 AgentRLDataset
>
> 原版参考（只读）: trainer/train_agent.py, trainer/rollout_engine.py
> ```
> 前置：[15-GRPO](15-GRPO.md)（GRPO 的 loss 和 advantage 计算）

---

## Part 1: 一个训练 Step 的全貌

**先建立全局画面，再拆解每个零件。** 这是理解 Agent RL 最重要的一节——后面所有 Part 都是往这个骨架里填肉。

### 1.1 Agent RL = GRPO + 多轮工具交互

[上一章](15-GRPO.md) 的 GRPO 是**单轮**的：prompt 进去，模型一口气生成完整回复，打分，更新。模型和外界没有交互。

Agent RL 把 rollout 从"单轮生成"变成"多轮交互循环"——模型生成回复，如果包含工具调用就执行工具，把结果拼回上下文让模型继续生成。但 **loss 计算和 GRPO 完全一样**，变的只是 rollout 和 reward。

### 1.2 一个 Training Step 完整流程

```
一个 training step 做了什么？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: 取数据
  从 DataLoader 拿一个 batch，比如 2 个 prompt：
  prompt_0 = "1美元等于多少人民币？"
  prompt_1 = "北京今天天气怎么样？"

Step 2: 多轮 Rollout（生成轨迹）
  每个 prompt 用当前 policy 独立生成 4 条轨迹（num_generations=4）
  每条轨迹是一个最多 3 轮的交互循环：

  prompt_0 的轨迹 A:
    轮1: 模型生成 → <tool_call>get_exchange_rate(USD,CNY)</tool_call>
         执行工具 → {"rate": 7.21}，拼回上下文
    轮2: 模型生成 → "1美元等于7.21元人民币。"（无 tool_call，结束）

  prompt_0 的轨迹 B:
    轮1: 模型生成 → <tool_call>calculate_math("1*7")</tool_call>（调错工具）
         执行工具 → {"result": "7"}，拼回上下文
    轮2: 模型生成 → "1美元大约等于7元。"（答案不精确）

  ...共生成 2×4=8 条轨迹

Step 3: 算 Reward（每条轨迹 1 个标量）
  对 8 条轨迹分别算 reward，得到 8 个数字：
  prompt_0: [3.0, 0.0, -0.5, 0.5]  ← 4 条轨迹的分数
  prompt_1: [2.5, 1.0, -1.0, 2.0]  ← 4 条轨迹的分数

  注意：不是每轮一个 reward，是整条轨迹结束后算 1 个总分。

Step 4: 组内 Z-Score → Advantage（GRPO 的核心）
  每个 prompt 内部做归一化（组内比较，不跨 prompt）：
  prompt_0: rewards = [3.0, 0.0, -0.5, 0.5]
            mean = 0.75, std = 1.35
            advantages = [+1.67, -0.56, -0.93, -0.19]

  prompt_1: rewards = [2.5, 1.0, -1.0, 2.0]
            mean = 1.125, std = 1.31
            advantages = [+1.05, -0.10, -1.62, +0.67]

Step 5: 算 Loss（和 GRPO 完全一样）
  对 8 条轨迹各自：
  - 前向推理，算每个 token 的 log_prob
  - 用 ref model 算 KL 惩罚
  - loss = -advantage × log_prob × mask + KL惩罚

  关键：同一条轨迹内所有 mask=1 的 token 共享同一个 advantage。
  轨迹 A 的 advantage = +1.67 → 轨迹 A 里每个模型生成的 token
  都乘以 +1.67 → 梯度鼓励这些 token 的概率变大。

Step 6: 反向传播，更新参数
  8 条轨迹的 loss 平均 → backward → optimizer.step()

Step 7: 同步 Rollout Engine
  把更新后的参数同步到 rollout engine
  → 下一个 step 用新模型生成轨迹
```

### 1.3 打分粒度 vs 梯度粒度（最容易混淆的地方）

Reward 的计算过程确实看了中间步骤——比如"调了几次工具"、"每次参数对不对"。但这些**中间信号最终合成了一个标量**，然后这个标量作为 advantage 被**均匀地涂到整条轨迹的所有 token 上**。

```
打分时：看了很多维度                    → 合成 1 个标量
         工具对齐 +0.5                    ↓
         GT 匹配 +2.5                     reward = 2.5
         未完成   0                        ↓
         重复惩罚 -0.1                    → z-score → advantage = 1.67

梯度时：1 个 advantage 涂到所有 token
         <tool_call>get_exchange_rate     × 1.67  (mask=1)
         {"rate": 7.21}                  × 0     (mask=0, 工具返回不算)
         1美元等于7.21元人民币            × 1.67  (mask=1)
```

这就是为什么说 GRPO 在 agent 场景有根本性缺陷——reward 的计算很精细，但**传递给梯度的信号很粗糙**。如果一条轨迹"前 2 轮做对了、第 3 轮格式崩了"导致总分 -0.5，那前 2 轮正确的 token 也一起被惩罚。Part 6 会详细讨论这个问题和业界的解法。

---

## Part 2: 轨迹长什么样——多轮 Rollout

### 2.1 一条轨迹的完整生命周期

```
用户问: "北京今天天气怎么样？"

┌─ 轮 1 ─────────────────────────────────────────────────┐
│ 模型生成:                                                │
│   <think>用户问天气，我需要调用天气工具</think>           │
│   <tool_call>{"name":"get_current_weather",              │
│    "arguments":{"location":"北京"}}</tool_call>           │
│                                                          │
│ 解析到 tool_call → 执行工具                              │
│ 工具返回: {"city":"北京","temperature":"28°C",            │
│            "humidity":"65%","condition":"晴"}             │
│ 把工具结果拼回对话上下文                                  │
└──────────────────────────────────────────────────────────┘
                          ↓ 上下文变长了，包含工具结果
┌─ 轮 2 ─────────────────────────────────────────────────┐
│ 模型生成:                                                │
│   北京今天天气晴朗，气温28°C，湿度65%。                    │
│                                                          │
│ 没有 <tool_call> → rollout 结束                          │
└──────────────────────────────────────────────────────────┘

这条轨迹经历了 2 轮，产出:
  - 轮1的模型输出（含 tool_call）
  - 工具返回结果
  - 轮2的模型输出（最终回答）
```

三种结束条件：
1. 模型输出不含 `<tool_call>` → 正常结束
2. 达到 `max_turns=3` → 强制结束，标记 `unfinished=True`
3. 模型第一轮就直接回答（没调工具）→ 1 轮结束

### 2.2 response_mask：谁的 token 算 loss

一条轨迹的 token 序列里混合了"模型生成的"和"工具返回的"，必须区分：

```
token 序列:
[system prompt + tools + user question]  [模型第1轮输出 + tool_call]  [tool result]  [模型第2轮输出]
├──────── prompt_ids ──────────────────┤ ├─── mask=1 ────────────────┤ ├── mask=0 ──┤ ├── mask=1 ───┤

mask=1: 模型生成的 token → 参与 loss（模型对自己的行为负责）
mask=0: 工具返回的 token → 不参与 loss（这是环境的反馈，不是模型的决策）
```

为什么工具返回不能算 loss？因为 loss 的目标是调整模型的行为概率——"让好的 token 序列更容易被生成"。工具返回的 `{"rate": 7.21}` 不是模型生成的，对它算 loss 毫无意义，还会引入噪声。

### 2.3 rollout_single：单样本多轮 Rollout

```python
# 写到: scratch/trainer/train_agent.py
def rollout_single(rollout_engine, tokenizer, messages, tools,
                   max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """单样本多轮 rollout：生成 → 解析工具调用 → 执行 → 拼回上下文 → 继续生成

    返回值：
        final_output:          最后一轮的模型输出文本
        final_context:         完整上下文（含所有轮次的 prompt + response + tool result）
        prompt_ids:            第一轮的 prompt token ids（不参与 loss）
        response_ids:          所有轮次的 response token ids（模型生成 + 工具返回拼接）
        response_mask:         response_ids 的 mask（模型生成=1, 工具返回=0）
        response_old_logps:    response_ids 的 log prob（模型生成=真实值, 工具返回=0.0）
        all_outputs:           每轮的模型输出文本列表
        unfinished:            是否因为达到 max_turns 而被截断
    """
    all_outputs = []
    prompt_ids = None
    response_ids = []       # 累积所有轮次的 token ids
    response_mask = []      # 累积所有轮次的 mask（1=模型生成, 0=工具返回）
    response_old_logps = [] # 累积所有轮次的 log prob
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio

    for turn in range(max_turns):
        # --- Step 1: 把当前 messages 编码成 prompt ---
        context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=tools, open_thinking=open_thinking,
        )
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()

        # 第一轮的 context 就是 prompt（后续轮的 context 包含了之前的 response）
        if prompt_ids is None:
            prompt_ids = context_ids

        # --- Step 2: 用 rollout engine 生成一段回复 ---
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )

        # 去掉 pad 和 eos，保留 (token_id, logprob) 配对
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps)
                 if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]

        all_outputs.append(new_text)
        # 模型生成的 token：mask=1，记录真实 logprob
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text

        # --- Step 3: 解析是否有工具调用 ---
        calls = parse_tool_calls(new_text)
        if not calls:
            break  # 没有工具调用，rollout 结束

        # 最后一轮还在调工具 → 标记为未完成
        unfinished = turn == max_turns - 1

        # --- Step 4: 执行工具，把结果拼回 messages ---
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            name = call.get("name", "")
            raw = call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            result = execute_tool(name, raw)
            result_str = (
                json.dumps(result, ensure_ascii=False) if result
                else '{"error": "tool not found"}'
            )[:2048]  # 截断防止工具返回值撑爆 tokenizer
            messages.append({"role": "tool", "content": result_str})

        # --- Step 5: 计算工具返回部分的 token ids ---
        # 重新编码包含工具返回的完整上下文
        observe_context = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=not unfinished,  # 未完成时不加 generation prompt
            tools=tools, open_thinking=open_thinking,
        )
        observe_ids = tokenizer(observe_context, return_tensors="pt",
                                add_special_tokens=False)["input_ids"][0].tolist()

        # 用差集得到工具返回部分的 token ids
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]  # 新增的部分 = 工具返回的 token

        # 工具返回的 token：mask=0（不参与 loss），logprob=0.0（不是模型生成的）
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return (final_output, final_context, prompt_ids, response_ids,
            response_mask, response_old_logps, list(all_outputs), unfinished)
```

### 2.4 rollout_batch：批量生成轨迹

```python
# 写到: scratch/trainer/train_agent.py
def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch,
                  num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """对 batch 中每个 (messages, tools) 重复 num_gen 次 rollout_single"""
    all_completions, all_contexts = [], []
    all_prompt_ids, all_response_ids = [], []
    all_response_masks, all_response_old_logps = [], []
    all_turn_outputs, all_unfinished = [], []

    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]  # 深拷贝，rollout_single 会修改 messages
            (completion, context, prompt_ids, response_ids,
             response_mask, response_old_logps, turn_outputs, unfinished
            ) = rollout_single(rollout_engine, tokenizer, msgs_copy, tools,
                               max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)

    return (all_completions, all_contexts, all_prompt_ids, all_response_ids,
            all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished)
```

注意 `msgs_copy`——每次生成都从原始 messages 深拷贝。因为 `rollout_single` 会往 messages 里 append（assistant 回复、tool 结果），如果不拷贝，第 2 条轨迹就会在第 1 条轨迹的基础上继续，而不是从头开始独立探索。

---

## Part 3: 工具定义与模拟执行

### 3.1 可用工具

```python
# 写到: scratch/trainer/train_agent.py
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math",
        "description": "计算数学表达式",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter",
        "description": "单位换算",
        "parameters": {"type": "object",
            "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
            "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather",
        "description": "获取天气",
        "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time",
        "description": "获取时间",
        "parameters": {"type": "object",
            "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}},
            "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate",
        "description": "查询汇率",
        "parameters": {"type": "object",
            "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
            "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text",
        "description": "翻译文本",
        "parameters": {"type": "object",
            "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}},
            "required": ["text", "target_language"]}}},
]
```

6 个工具覆盖了常见的 agent 场景：计算、查询、转换。每个工具都有完整的 JSON Schema 定义（函数名、描述、参数类型、必填项），这些信息会通过 `apply_chat_template(tools=...)` 注入到 system prompt 中。

### 3.2 模拟执行：不连真实 API

```python
# 写到: scratch/trainer/train_agent.py
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

代码对模拟执行做了三层防护：
- `eval()` 传入 `{"__builtins__": {}, "math": math}`——清空内置函数，只留 math 模块。这挡住了 `__import__('os')` 这类直接攻击，但空 `__builtins__` 并非完整沙箱，部署时仍需进程级隔离
- `execute_tool` 用 `signal.alarm(1)` 设 1 秒超时——防止模型生成 `9**9**9` 这类表达式卡死训练
- 工具结果截断到 2048 字符——防止天文数字的字符串撑爆 tokenizer

**模拟环境不是玩具妥协，而是工业界主流路线**。Kimi K2 技术报告（[arXiv 2507.20534](https://arxiv.org/abs/2507.20534)）摘要明确描述其后训练为 "a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments"——大规模合成环境 + 真实/合成环境联合 RL。MiniMind 的 MOCK_RESULTS 就是这条路线的最小实现。

### 3.3 参数校验

```python
# 写到: scratch/trainer/train_agent.py
CHECK_ARGS = {
    "calculate_math":      lambda a: bool(a.get("expression")),
    "unit_converter":      lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time":    lambda a: True,  # 无必填参数
    "get_exchange_rate":   lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text":      lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}
```

校验函数检查模型是否传了正确的参数。如果模型调了 `calculate_math` 但没传 `expression` 参数，校验失败 → reward 扣分。

### 3.4 工具调用解析与执行

```python
# 写到: scratch/trainer/train_agent.py

def parse_tool_calls(text):
    """从模型输出文本中提取 <tool_call>...</tool_call> 内的 JSON"""
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass  # JSON 解析失败就跳过这个调用
    return calls


def execute_tool(name, args):
    """查 MOCK_RESULTS 执行工具，带 1 秒超时防止 eval 卡死"""
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return None
    try:
        # 设置 1 秒超时（仅 Unix）
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except Exception:
        return None
    finally:
        try:
            signal.alarm(0)  # 取消定时器
        except Exception:
            pass
```

---

## Part 4: Reward 设计——一条轨迹一个分数

Part 1 已经说了：reward 是**整条轨迹结束后算 1 个标量**。现在来看这个标量怎么算。

### 4.1 总览：两条路径

模型拿到一个 prompt 后，有两种可能的行为：调工具、不调工具。Reward 计算走不同路径：

```
模型输出不含 <tool_call>                模型输出含 <tool_call>
（直接回答）                            （调用工具）
    │                                       │
    ▼                                       ▼
 长度分 ±0.5                            ① 工具对齐分 ±0.5×n
 思考格式分 +1.0/-0.5                   ② GT 匹配分 +2.5×(命中率)
 思考闭合分 ±0.25                       ③ 未完成扣分 -0.5
 RM 打分 (外部模型)                        重复惩罚
 重复惩罚                                  ↓
    ↓                                    不用 RM（有客观答案了）
 clip 到 [-3, +3]                       clip 到 [-3, +3]

两条路径的公共检查（先执行）：
  <tool_call> 标签不配对 → 每个 -0.5（惩罚格式崩坏）
```

### 4.2 公共检查：标签配对

```python
# 写到: scratch/trainer/train_agent.py —— 每个不配对的标签扣 0.5
reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)
```

惩罚的是**格式崩坏**——模型生成了 `<tool_call>` 却没闭合（或反之），解析器拿不到完整 JSON。这种输出连"错误的工具调用"都算不上。

### 4.3 路径一：没调工具

模型直接回答了问题，reward 由长度分 + 思考格式分 + RM 打分 + 重复惩罚组成。和 [上一章](15-GRPO.md) GRPO 的 reward 基本一样。

```python
# 写到: scratch/trainer/train_agent.py
if not tool_calls:
    reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5     # 长度分
    if '</think>' in response:
        think, answer = response.split('</think>', 1)
        reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5   # 思考长度分
        reward += 0.25 if response.count('</think>') == 1 else -0.25 # 思考闭合分（恰好 1 个 </think>）
    score = reward_model.get_score(messages, answer)                  # RM 分
    reward -= rep_penalty(answer)                                     # 重复惩罚
```

**RM（Reward Model）** 是 InternLM2-1.8B-Reward，上海 AI Lab 开源的 1.8B 打分模型。它接受 (prompt, answer) 对，输出一个标量分数表示回答质量。MiniMind 本身只有 64M 参数，让它给自己打分没意义，所以用一个能力更强的外部模型做 reward signal——这是 RLHF 的标准做法。

### 4.4 路径二：调了工具（Agent RL 的核心 reward）

这是本章的重点。reward 分三层，用一个完整数字例子走通。

**例子：** 用户问"1美元等于多少人民币？"，GT = `["7.21"]`

#### 第一层：工具对齐分——你选对工具了吗？

```python
# 写到: scratch/trainer/train_agent.py
valid_call_count = 0
for tool_call in tool_calls:
    name = tool_call.get("name", "")
    raw = tool_call.get("arguments", {})
    check = CHECK_ARGS.get(name)
    valid_call_count += int(bool(name in valid_names and check and check(raw)))

tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)
reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap              # 工具对齐分
```

`tool_gap` 衡量两个差距之和：
- `abs(valid_call_count - len(gt))`：有效调用数和期望数量的差距
- `max(0, len(tool_calls) - valid_call_count)`：无效调用数（调了不存在的工具、或参数不对）

```
例子 1: 模型调了 get_exchange_rate(USD, CNY) → valid=1, GT数=1
  tool_gap = |1-1| + max(0, 1-1) = 0  → +0.5

例子 2: 模型调了 get_exchange_rate 但漏了 from_currency → valid=0
  tool_gap = |0-1| + max(0, 1-0) = 2  → -0.5 × 2 = -1.0

例子 3: 模型调了 get_exchange_rate(对) + get_current_time(多余) → valid=2
  tool_gap = |2-1| + max(0, 2-2) = 1  → -0.5

例子 4: 模型调了 calculate_math（调错工具）→ valid=1(语法对但工具不对)
  tool_gap = |1-1| + max(0, 1-1) = 0  → +0.5
  注意：这里拿了对齐分，但下一层 GT 匹配会暴露错误——
  calculate_math 算不出汇率 7.21，GT 匹配会失败
```

#### 第二层：GT 匹配分——最终答案对不对？

```python
# 写到: scratch/trainer/train_agent.py
final_text = "" if unfinished else (
    answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer
)
verified = validate_gt_in_text(final_text, gt) if gt else set()
if gt: reward += 2.5 * len(verified) / len(gt)                    # GT 验证分
```

GT（ground truth）是训练数据里标注的正确答案。`validate_gt_in_text` 做两种匹配：
- **字符串匹配**：`"7.21"` 出现在回答文本中（忽略大小写）
- **数值匹配**：提取回答中的所有数字，和 GT 做浮点比较（容差 1e-6）

```python
# 写到: scratch/trainer/train_agent.py
def validate_gt_in_text(text, gt_list):
    """在文本中检查 ground truth 是否出现（支持字符串匹配和数值模糊匹配）"""
    text_str = str(text)
    # 去掉逗号后提取所有数字，用于数值比较
    text_num = text_str.replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    verified = set()
    for g in gt_list:
        s = str(g).strip()
        if not s:
            continue
        # 字符串匹配（忽略大小写）
        if s.lower() in text_str.lower():
            verified.add(g)
            continue
        # 数值匹配（容差 1e-6）
        if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', s.replace(',', '')):
            if any(abs(float(s.replace(',', '')) - n) < 1e-6 for n in nums):
                verified.add(g)
    return verified
```

```
例子: GT = ["28°C", "晴"]（北京天气有两个期望答案）

模型回答 "北京今天28°C，天气晴朗"
  → "28°C" 命中（字符串匹配），"晴" 命中（字符串匹配）
  → verified = 2/2 → reward += 2.5 × 1.0 = +2.5

模型回答 "北京今天28度"
  → "28°C" 未命中（"28度" ≠ "28°C"），"晴" 未命中
  → 数值匹配：28.0 和 28.0 差 < 1e-6 → "28°C" 这个 GT 不是纯数字，不走数值匹配
  → verified = 0/2 → reward += 0

模型回答 "汇率是7.21"（用了 get_exchange_rate）
  → "7.21" 命中 → verified = 1/1 → reward += 2.5
```

+2.5 是**所有 reward 信号中权重最大的**。设计意图：工具对齐只是手段，最终答案正确才是目的。

#### 第三层：惩罚项

```python
# 写到: scratch/trainer/train_agent.py
if unfinished: reward -= 0.5                                       # 达到 max_turns 还没结束
reward -= rep_penalty(final_text)
rewards[idx] = max(min(reward, 3.0), -3.0)                        # Clip 到 [-3, 3]
```

### 4.5 用完整数字例子走通：从 reward 到 advantage 到梯度

这个例子串联 Part 1 的全貌图和 Part 4 的 reward 计算，是理解 Agent RL 最关键的链条。

```
Prompt: "1美元等于多少人民币？"  GT = ["7.21"]
生成 4 条轨迹（num_generations=4）:

轨迹 A:
  轮1: <tool_call>get_exchange_rate(USD,CNY)</tool_call>
       → 工具返回 {"rate": 7.21}
  轮2: "1美元等于7.21元人民币。"
  reward 计算:
    标签配对 ok → 0
    工具对齐: valid=1, GT数=1, gap=0 → +0.5
    GT 匹配: "7.21" 命中 1/1 → +2.5
    未完成: 否 → 0
    重复惩罚: 0
    总分 = 3.0 → clip → 3.0 ✓

轨迹 B:
  轮1: <tool_call>get_exchange_rate(USD,CNY)</tool_call>
       → 工具返回 {"rate": 7.21}
  轮2: "大约7元左右吧。"
  reward 计算:
    工具对齐: gap=0 → +0.5
    GT 匹配: "7.21" 未出现在 "大约7元左右吧" → 0
    总分 = 0.5

轨迹 C:
  轮1: <tool_call>calculate_math("1*7")</tool_call>  ← 调错了工具
       → 工具返回 {"result": "7"}
  轮2: "1美元等于7元。"
  reward 计算:
    工具对齐: calculate_math 不在期望工具里？
      实际上 valid_names 是所有 6 个工具名，calculate_math 在列表中
      且 check("expression"="1*7") 通过 → valid=1, GT数=1, gap=0 → +0.5
      （对齐分不检查"用的工具对不对"，只检查"调用合法不合法"）
    GT 匹配: "7.21" 未出现在 "1美元等于7元" → 0
    总分 = 0.5

轨迹 D:
  轮1: "1美元大概等于7块钱吧"  ← 没调工具，直接回答
  走无工具路径:
    长度分: 11字符 ∈ [5, 800] → +0.5
    无 </think> 标签
    RM 打分: 假设 -1.0（回答不准确）
    重复惩罚: 0
    总分 = -0.5

4 条 reward: [3.0, 0.5, 0.5, -0.5]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

组内 z-score:
  mean = 0.875
  std = 1.31
  advantages:
    A: (3.0 - 0.875) / 1.31 = +1.62   ← 远好于组内平均
    B: (0.5 - 0.875) / 1.31 = -0.29   ← 略低于平均
    C: (0.5 - 0.875) / 1.31 = -0.29   ← 略低于平均
    D: (-0.5 - 0.875) / 1.31 = -1.05  ← 远差于组内平均

advantage 如何作用到 token:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

轨迹 A (advantage = +1.62):
  token: <tool_call>get_exchange_rate...  mask=1 → loss = -1.62 × log_prob
  token: {"rate": 7.21}                  mask=0 → 不参与 loss（工具返回）
  token: 1美元等于7.21元人民币            mask=1 → loss = -1.62 × log_prob
  → 梯度鼓励这些 token 的生成概率变大

轨迹 D (advantage = -1.05):
  token: 1美元大概等于7块钱吧             mask=1 → loss = -(-1.05) × log_prob
                                                      = +1.05 × log_prob
  → 梯度抑制这些 token 的生成概率（让模型下次别直接瞎答）
```

### 4.6 Reward 设计的哲学：弱监督

训练数据只标注了最终正确答案（GT），不标注"应该调哪个工具、传什么参数"。模型通过 RL 自己探索：哪条调用路径能让最终答案包含正确结果，哪条路径就拿高 reward。工具对齐分（+0.5）只是辅助信号，给模型一个"方向感"——"合法的工具调用大概率是好的"。

```
SFT 数据:   system + user + assistant(标准回复)  → 监督学习，模仿标准回复
Agent RL:   system + user + gt(正确结果)         → 强化学习，模型自己探索
                               ↑ 不给回复，只给正确答案用于打分
```

Reward 设计本身就是 tool-use RL 的核心研究变量——ToolRL（[arXiv 2504.13958](https://arxiv.org/abs/2504.13958)，题目就叫 "Reward is All Tool Learning Needs"）系统研究了工具调用场景下格式分与正确性分的配比、粒度和给分时机对训练的影响。MiniMind 这套"格式分 + 工具对齐分 + GT 分"的多维设计正是这类思路的极简版。

### 4.7 Reward Model（RM）补充说明

RM 只在"没调工具"路径使用。为什么？

有工具调用时，我们有**客观正确答案**（GT）可以验证——`"7.21"` 出现在回答里就是对的。RM 是个"主观评判"，在有客观标准的情况下引入它只会增加噪声。

没调工具时，模型是直接回答问题，没有客观答案可查，所以用 RM 来评估回答质量（是不是通顺、有没有回答到点上）。

```python
# 写到: scratch/trainer/train_agent.py（RM 使用方式）
# LMForRewardModel 封装了 InternLM2-1.8B-Reward
# 输入: 对话 messages + 模型回答
# 输出: 标量分数，clip 到 [-3, 3]
reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
score = reward_model.get_score(messages, answer)
```

---

## Part 5: Loss 计算——和 GRPO 共享

Loss 计算和 [上一章](15-GRPO.md) GRPO 完全一样——PPO-clip + KL 惩罚。唯一的区别在于序列结构：

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
# 写到: scratch/trainer/train_agent.py（和 GRPO 完全一样）
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

## Part 6: GRPO 在 Agent 场景的根本性缺陷

Part 1.3 已经预告了这个问题：reward 算得很精细，但传递给梯度的信号很粗糙。现在详细展开。

### 6.1 问题 1：per-response advantage → 中间步骤的 credit 错误

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

### 6.2 问题 2：组内归一化在全部失败时崩溃

```
G=4 个 trajectory 全部失败:  r = [0, 0, 0, 0]
  → std = 0，advantage = 0 / (0 + 1e-4) ≈ 0
  → 梯度几乎为零，模型什么都没学到

一个勉强成功 (r=0.3)，三个完全失败 (r=0):
  → 归一化后 0.3 吃正 advantage
  → 但 0.3 的成功率很低
  → 学到的是"比烂"而不是"做好"
```

Agent 任务的组内全败概率通常比 math 更高——math 一条链只要最终答案对就得分，agent 轨迹要**每一步都不崩**（选对工具、传对参数、格式合法、最后还要答对），错误率随步数累乘。步数越长，G 个轨迹全部失败、梯度归零的概率越大。

### 6.3 问题 3：轨迹更长、reward 更稀疏，组内统计更不稳定

```
MiniMind 场景（教学缩影）: 最多 3 轮，max_total_len=2500
业界 agent 场景: 轨迹可达 100+ 轮、100K~1M token
  （来源：credit assignment 综述 arXiv 2604.09459——
   reasoning RL 单链约 500~30K+ token；
   agentic RL 多轮交互 100+ turns，episode 级 credit 随 horizon 变长越来越无信息量）

轨迹越长 → outcome reward 越接近二值(成/败) → 组内 reward 方差越大
  → G=4 个轨迹的组内归一化非常不稳定
  → 想增大 G 来稳定统计，但 agent rollout 比单轮 rollout 贵得多：
    每轮生成完必须停下来执行工具、拼接上下文再继续，整条轨迹是串行的，
    没法像单轮任务一样一次生成到底
```

### 6.4 MiniMind 的缓解措施

MiniMind 的 Agent RL 用了几个技巧部分缓解，但无法根治：

| 措施 | 缓解了什么 | 无法解决什么 |
|------|-----------|-------------|
| response_mask（tool result 不算 loss） | 防止工具结果的 token 被错误惩罚 | 模型自己生成的 tool_call token 仍共享 per-response advantage |
| 多维 reward（工具正确性 + 格式 + GT） | 提供更密集的 reward 信号 | 所有维度最终合成一个标量，仍是 per-response |
| GT 弱监督（连续匹配分） | 避免纯 0/1 reward | 组内归一化的统计性质问题仍在 |
| `max_total_len=2500` 截断 | 控制序列长度方差 | 截断可能丢掉关键信息 |

### 6.5 业界的真实解法（2025-2026）

上面三个问题不是 MiniMind 独有的，而是整个 agent RL 方向的公认痛点。2025 年以来出现了一批针对性工作，每个都能精确对应到前面的某个问题：

**针对问题 1（credit 太粗）——不引入 critic 也能拿细粒度 advantage：**

- **GiGPO**（Group-in-Group Policy Optimization，NTU，[arXiv 2505.10978](https://arxiv.org/abs/2505.10978)，官方实现 [verl-agent](https://github.com/langfengQ/verl-agent)）：两层 advantage。episode 级沿用 GRPO 的组内比较算 macro advantage；step 级引入 **anchor state grouping**——在 G 条轨迹里回溯找出"重复出现的环境状态"，把从同一状态出发的不同动作归成一个 step 级小组，在小组内再做相对比较。这样同一状态下"调对工具"和"调错工具"的动作拿到不同的 advantage，而整个机制保持 critic-free、低显存
- **Turn-level reward**（Minnesota + Morgan Stanley，[arXiv 2505.11821](https://arxiv.org/abs/2505.11821)）：首个系统研究 multi-turn RL 的 turn 级 reward 设计，把 GRPO/PPO 扩展成 turn 级变体。设计了两类 turn-level reward：**verifiable**（规则可验证，如"这一轮的工具调用格式合法且参数正确"）和 **LLM-as-judge**（让另一个模型给每轮打分）。对 MiniMind 来说这是工程代价最小的升级路径——把现有的工具对齐分从"轨迹级汇总"改成"每轮即时判定"即可
- **TRACE**（[arXiv 2607.13988](https://arxiv.org/abs/2607.13988)，2026-07）：把 rollout 按 tool-call 边界切成状态转移序列，用一个 **frozen reference model 对 gold answer 的 log-prob** 作为每个状态的 value（log-ratio 形式），从 value 差分推出 turn 级 dense credit——不训 critic、不需人工标 step label。刚发布，代表最新方向

**针对问题 2/3（全败梯度消失、rollout 贵）——把采样预算花在刀刃上：**

- **ARPO**（Agentic Reinforced Policy Optimization，[arXiv 2507.19849](https://arxiv.org/abs/2507.19849)）：论文观察到 LLM 在**工具结果返回之后**生成 token 的熵显著升高（模型"拿到新信息后不确定下一步"）。ARPO 据此做 entropy-based adaptive rollout——在高熵的工具调用后节点做**分支采样**，其余位置沿用全局轨迹采样。等效于把有限的 rollout 预算集中投到决策分叉点，缓解"想增大 G 但太贵"的矛盾
- **异步 rollout 架构**（AReaL，蚂蚁技术研究院+清华，[GitHub](https://github.com/inclusionAI/AReaL)）：把 rollout 和训练完全解耦成异步流水线——训练不等最慢的轨迹。其上训出的 ASearcher 演示了超过 10 轮的 long-horizon 搜索 agent。这不改算法，纯靠系统架构把"agent rollout 串行且贵"的成本压下来

**传统路线仍然存在：**

| 方法 | 思路 | 代价 |
|------|------|------|
| PRM (Process Reward Model) | 训一个模型判断中间步骤对不对 | 需要标注 process label，agent 场景标注成本高 |
| 回到 PPO/GAE | per-token advantage 递推 | 需要 critic 模型和更多显存（[13 章](13-RL对齐基础.md)讨论过的 trade-off） |

### 6.6 方法演进脉络

```
GRPO (2024, DeepSeek) — critic-free 组内比较，在 math reasoning 成功
  ↓ 直接搬到 agent 场景（Search-R1 / ReTool / ToRL，2025.03-04，本章 MiniMind 所在位置）
  ↓ 暴露三个问题：credit 粗 / 全败梯度消失 / rollout 贵
GiGPO (2025.05)        — anchor state 分组 → critic-free 拿 step 级 advantage
Turn-level RL (2025.05) — turn 级 verifiable reward → GRPO/PPO 的 turn 级变体
ARPO (2025.07)         — 工具后熵升高 → 高熵节点分支采样，省 rollout 预算
AReaL (2025)           — 全异步 rollout → 系统层面解决"贵"
  ↓ 2026 上半年
TRACE / GEAR / StepPO 等 — turn/step 级 dense credit 成为爆发方向
```

---

## Part 7: 训练数据

### AgentRLDataset

```python
# 写到: scratch/dataset/lm_dataset.py
class AgentRLDataset(Dataset):
    """Agent RL 数据集：从 JSONL 加载 agent 训练数据

    JSONL 每行格式：
    {"conversations": [...], "gt": ["期望答案1", "期望答案2"]}

    conversations 中的 message 结构：
    - {"role": "system", "tools": "[...]"}  → 提取工具定义
    - {"role": "system", "content": "..."}  → 系统提示词
    - {"role": "user", "content": "..."}    → 用户问题
    - {"role": "assistant", ...}            → 训练时不需要（由 rollout 在线生成）
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample['conversations']
        if isinstance(conversations, str):
            conversations = json.loads(conversations)

        messages, tools, gt = [], None, []
        for msg in conversations:
            # 提取工具定义（system message 中带 tools 字段）
            if msg.get('role') == 'system' and msg.get('tools'):
                tools = json.loads(msg['tools']) if isinstance(msg['tools'], str) else msg['tools']
            # 保留 user 消息作为 prompt
            if msg.get('role') == 'user':
                messages.append({'role': 'user', 'content': msg['content']})
            # 保留不带 tools 的 system 消息（系统提示词）
            if msg.get('role') == 'system' and not msg.get('tools'):
                messages.append({'role': 'system', 'content': msg['content']})

        # gt = ground truth 列表，用于 reward 计算中的 GT 匹配
        gt = sample.get('gt', [])
        if isinstance(gt, str):
            gt = json.loads(gt)

        return {'messages': messages, 'tools': tools or [], 'gt': gt}
```

注意训练数据和 SFT 的关键区别：

```
SFT 数据:   system + user + assistant(标准回复)       → 监督学习，模仿标准回复
Agent RL:   system + user + gt(正确结果)              → 强化学习，模型自己探索
                               ↑ 不给回复，只给正确答案用于打分
```

---

## Part 8: 训练主循环

把 Part 1 的全貌图对应到代码。

```python
# 写到: scratch/trainer/train_agent.py
def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0):
    """Agent RL 训练主循环（loss 计算和 GRPO 完全一样，区别在 rollout 和 reward）"""
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']

        # === Part 1 的 Step 2: 多轮 rollout ===
        with torch.no_grad():
            (completions, contexts,
             prompt_ids_batch, response_ids_batch,
             response_masks_batch, response_old_logps_batch,
             turn_outputs_batch, unfinished_batch
            ) = rollout_batch(
                rollout_engine, tokenizer, messages_batch, tools_batch,
                args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len,
                thinking_ratio=args.thinking_ratio, device=args.device,
            )

        # === 打包成训练 tensor ===
        # 每个样本：[prompt_ids | response_ids]，用 response_mask 标记哪些算 loss
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t)
            for m, t in zip(messages_batch, tools_batch)
        ]
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch,
                                    response_masks_batch, response_old_logps_batch):
            ids = p + r                              # 拼接 prompt + response
            mask = [0] * len(p) + m                  # prompt 部分 mask=0
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp  # prompt 部分填 0
            # 截断超长序列
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            # 找到第一个 mask=1 的位置，即 prompt 长度
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))

        # pad 到 batch 内最长
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor(
            [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples],
            device=args.device,
        )
        full_response_masks = torch.tensor(
            [mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples],
            device=args.device, dtype=torch.float32,
        )
        old_per_token_logps = torch.tensor(
            [old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples],
            device=args.device, dtype=torch.float32,
        )
        full_mask = (input_ids != tokenizer.pad_token_id).long()

        # === Part 1 的 Step 3: 计算 reward ===
        rewards = calculate_rewards(
            prompts, completions, gt_batch, tools_batch, args.num_generations,
            reward_model, device=args.device,
            turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch,
        )

        # === Part 1 的 Step 5: 前向推理 + loss ===
        with autocast_ctx:
            res = model(input_ids, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]  # [B, L-1, V]
            per_token_logps = F.log_softmax(logits, dim=-1).gather(
                2, input_ids[:, 1:].unsqueeze(-1)  # 取每个位置实际 token 的 logprob
            ).squeeze(-1)  # [B, L-1]

        # ref model logprobs（用于 KL 惩罚）
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(
                ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask,
            )

        # 构造 completion_mask（基于 response_mask，而不是简单的 pad_mask）
        completion_mask = full_response_masks[:, 1:]  # 对齐到 logits 维度（shift by 1）
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1,
                             device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0

        # === Part 1 的 Step 4: 组内 z-score → advantage ===
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # KL 散度惩罚
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        # importance sampling ratio
        ratio = torch.exp(per_token_logps - old_per_token_logps)

        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                               - args.beta * per_token_kl)
        else:  # grpo
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2)
                               - args.beta * per_token_kl)

        policy_loss = (
            ((per_token_loss * completion_mask).sum(dim=1)[valid_rows]
             / token_counts[valid_rows].clamp(min=1)).mean()
            if valid_rows.any() else per_token_loss.sum() * 0.0
        )

        # === Part 1 的 Step 6: 反向传播 ===
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # === 日志 ===
        if step % args.log_interval == 0 or step == iters:
            pl = loss.item() * args.accumulation_steps
            ar = rewards.mean().item()
            al = token_counts.float().mean().item()
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                  f'Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, '
                  f'Loss:{pl:.4f}, AvgLen:{al:.2f}, LR:{lr:.8f}')

        # === Part 1 的 Step 7: 同步 rollout engine ===
        if step % args.save_interval == 0 or step == iters:
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            state_dict = model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            model.train()
            del state_dict
            rollout_engine.update_policy(model)

        del per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
```

---

## 完整文件

本章的手写代码分两个文件：

| 文件 | 内容 |
|------|------|
| `scratch/trainer/train_agent.py` | 工具定义 + 模拟执行 + 多轮 rollout + reward 计算 + 训练主循环 |
| `scratch/dataset/lm_dataset.py` | 追加 `AgentRLDataset`（从 JSONL 解析 messages + tools + gt） |

运行命令：

```bash
cd scratch
python -m trainer.train_agent \
    --batch_size 2 \
    --num_generations 4 \
    --learning_rate 3e-7 \
    --beta 0.1 \
    --loss_type cispo \
    --from_weight full_sft
```

### scratch/trainer/train_agent.py 完整代码

```python
# scratch/trainer/train_agent.py
# 教学版：保留核心算法，去掉 DDP/wandb/断点续训/SGLang/debug_mode
import os, sys, re, json, math, random, signal, argparse, warnings
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

# ================================ 工具与 Reward ================================

def rep_penalty(text, n=3, cap=0.5):
    """n-gram 重复惩罚：统计重复 n-gram 比例，越重复扣分越多，上限 cap"""
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


# ======== 工具定义（6 个工具的 JSON Schema） ========
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math",
        "description": "计算数学表达式",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter",
        "description": "单位换算",
        "parameters": {"type": "object",
            "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
            "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather",
        "description": "获取天气",
        "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time",
        "description": "获取时间",
        "parameters": {"type": "object",
            "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}},
            "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate",
        "description": "查询汇率",
        "parameters": {"type": "object",
            "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
            "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text",
        "description": "翻译文本",
        "parameters": {"type": "object",
            "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}},
            "required": ["text", "target_language"]}}},
]

# ======== 模拟数据（训练环境不联网，用字典模拟真实 API 返回） ========
WEATHER_DATA = {
    "北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"),
    "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"),
    "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"),
    "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"),
    "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗"),
}
TIME_DATA = {
    "Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00",
    "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00",
    "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00",
}
EXCHANGE_DATA = {
    ("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12,
    ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79,
    ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72,
}
TRANSLATE_DATA = {
    ("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好",
    ("今天天气真好", "english"): "The weather is nice today",
    ("I love programming", "chinese"): "我喜欢编程",
    ("机器学习很有趣", "english"): "Machine learning is interesting",
    ("Happy birthday", "chinese"): "生日快乐",
}
UNIT_DATA = {
    "km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462,
    "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048,
    "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556,
}

# ======== 模拟执行（每个工具一个 lambda，输入 args dict → 输出 result dict） ========
MOCK_RESULTS = {
    "calculate_math": lambda args: {
        "result": str(eval(
            str(args.get("expression", "0"))
            .replace("^", "**").replace("×", "*").replace("÷", "/")
            .replace("−", "-").replace("（", "(").replace("）", ")"),
            {"__builtins__": {}, "math": math}  # 沙盒：只暴露 math 模块
        ))
    },
    "unit_converter": lambda args: {
        "result": round(
            float(args.get("value", 0)) *
            UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1),
            4,
        )
    },
    "get_current_weather": lambda args: (
        lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]}
    )(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {
        "datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"),
        "timezone": args.get("timezone", "Asia/Shanghai"),
    },
    "get_exchange_rate": lambda args: {
        "from": args.get("from_currency"), "to": args.get("to_currency"),
        "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0),
    },
    "translate_text": lambda args: {
        "translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", "")),
    },
}

# ======== 参数校验（检查工具调用的必填参数是否齐全） ========
CHECK_ARGS = {
    "calculate_math":      lambda a: bool(a.get("expression")),
    "unit_converter":      lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time":    lambda a: True,  # 无必填参数
    "get_exchange_rate":   lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text":      lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}


# ======== 工具调用解析与执行 ========

def parse_tool_calls(text):
    """从模型输出文本中提取 <tool_call>...</tool_call> 内的 JSON"""
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass  # JSON 解析失败就跳过这个调用
    return calls


def execute_tool(name, args):
    """查 MOCK_RESULTS 执行工具，带 1 秒超时防止 eval 卡死"""
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return None
    try:
        # 设置 1 秒超时（仅 Unix）
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except Exception:
        return None
    finally:
        try:
            signal.alarm(0)  # 取消定时器
        except Exception:
            pass


# ======== 多轮 Rollout（Agent RL 的核心区别） ========

def rollout_single(rollout_engine, tokenizer, messages, tools,
                   max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio

    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=tools, open_thinking=open_thinking,
        )
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()

        if prompt_ids is None:
            prompt_ids = context_ids

        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )

        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps)
                 if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]

        all_outputs.append(new_text)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text

        calls = parse_tool_calls(new_text)
        if not calls:
            break

        unfinished = turn == max_turns - 1

        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            name = call.get("name", "")
            raw = call.get("arguments", {})
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            result = execute_tool(name, raw)
            result_str = (
                json.dumps(result, ensure_ascii=False) if result
                else '{"error": "tool not found"}'
            )[:2048]
            messages.append({"role": "tool", "content": result_str})

        observe_context = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=not unfinished,
            tools=tools, open_thinking=open_thinking,
        )
        observe_ids = tokenizer(observe_context, return_tensors="pt",
                                add_special_tokens=False)["input_ids"][0].tolist()

        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]

        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return (final_output, final_context, prompt_ids, response_ids,
            response_mask, response_old_logps, list(all_outputs), unfinished)


def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch,
                  num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    all_completions, all_contexts = [], []
    all_prompt_ids, all_response_ids = [], []
    all_response_masks, all_response_old_logps = [], []
    all_turn_outputs, all_unfinished = [], []

    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            (completion, context, prompt_ids, response_ids,
             response_mask, response_old_logps, turn_outputs, unfinished
            ) = rollout_single(rollout_engine, tokenizer, msgs_copy, tools,
                               max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)

    return (all_completions, all_contexts, all_prompt_ids, all_response_ids,
            all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished)


# ======== Reward 计算 ========

def validate_gt_in_text(text, gt_list):
    text_str = str(text)
    text_num = text_str.replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    verified = set()
    for g in gt_list:
        s = str(g).strip()
        if not s:
            continue
        if s.lower() in text_str.lower():
            verified.add(g)
            continue
        if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', s.replace(',', '')):
            if any(abs(float(s.replace(',', '')) - n) < 1e-6 for n in nums):
                verified.add(g)
    return verified


def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen,
                      reward_model=None, device="cuda",
                      turn_outputs_batch=None, unfinished_batch=None):
    rewards = torch.zeros(len(completions), device=device)

    for idx, response in enumerate(completions):
        reward = 0.0
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch else False

        turn_answers = [
            turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip()
            for turn in turn_outputs
        ]
        answer = turn_answers[-1] if turn_answers else response.strip()

        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers:
            tool_calls.extend(parse_tool_calls(turn_answer))

        reward -= 0.5 * sum(
            abs(turn.count('<tool_call>') - turn.count('</tool_call>'))
            for turn in turn_answers
        )

        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
                reward += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer.strip()
            if reward_model is not None:
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward += score
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)

        else:
            gt = gt_batch[sample_idx]

            valid_call_count = 0
            for tool_call in tool_calls:
                name = tool_call.get("name", "")
                raw = tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except Exception:
                        raw = {}
                check = CHECK_ARGS.get(name)
                valid_call_count += int(bool(name in valid_names and check and check(raw)))

            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap

            final_text = "" if unfinished else (
                answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer
            )
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt:
                reward += 2.5 * len(verified) / len(gt)

            if unfinished:
                reward -= 0.5

            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)

    return rewards


# ================================ 训练主循环 ================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent RL (教学版)")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='agent', type=str)
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
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument("--max_gen_len", type=int, default=768)
    parser.add_argument("--max_total_len", type=int, default=2500)
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument('--from_weight', default='full_sft', type=str)
    parser.add_argument("--thinking_ratio", type=float, default=0.1)
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
        use_moe=bool(args.use_moe),
    )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # --- 初始化三个模型 ---
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)  # ref model 冻结

    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

    # --- Rollout 引擎 ---
    rollout_engine = create_rollout_engine(
        engine_type="torch",
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
    )

    # --- 数据集 + DataLoader ---
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    def collate_fn(batch):
        return {
            'messages': [b['messages'] for b in batch],
            'tools': [b['tools'] for b in batch],
            'gt': [b['gt'] for b in batch],
        }
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # --- 优化器 + 调度器 ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    iters = len(loader)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps,
                                  eta_min=args.learning_rate / 10)

    rollout_engine.update_policy(model)

    # --- 训练 ---
    for epoch in range(args.epochs):
        rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model)
```

---

## 对照表

| 概念 | 笔记讲解 | scratch 手写 | 原版参考 |
|------|---------|-------------|---------|
| 训练全貌 | Part 1（Step 1-7 + 数字例子） | `scratch/trainer/train_agent.py` `rl_train_epoch` | `trainer/train_agent.py:242-340` |
| 工具定义 | Part 3.1 | `scratch/trainer/train_agent.py` TOOLS | `trainer/train_agent.py:40-47` |
| 模拟执行 | Part 3.2-3.4 | `scratch/trainer/train_agent.py` MOCK_RESULTS | `trainer/train_agent.py:57-74` |
| 多轮 rollout | Part 2.3 | `scratch/trainer/train_agent.py` `rollout_single` | `trainer/train_agent.py:98-157` |
| response_mask | Part 2.2 | `scratch/trainer/train_agent.py` | `trainer/train_agent.py:151` |
| 标签配对扣分 | Part 4.2 | `scratch/trainer/train_agent.py` | `trainer/train_agent.py:201` |
| GT 验证 | Part 4.4 | `scratch/trainer/train_agent.py` `validate_gt_in_text` | `trainer/train_agent.py:183-186` |
| Reward 计算 | Part 4（完整数字例子） | `scratch/trainer/train_agent.py` `calculate_rewards` | `trainer/train_agent.py:188-239` |
| Loss | Part 5 | `scratch/trainer/train_agent.py` | `trainer/train_agent.py:322-332` |
| GRPO 缺陷 | Part 6（三个问题 + 业界解法） | — | — |
| 数据集 | Part 7 | `scratch/dataset/lm_dataset.py` `AgentRLDataset` | `dataset/lm_dataset.py:226-252` |

---

## 业界对比

对比对象选**开源 agent RL 训练框架**——它们和 `train_agent.py` 解决同一个问题（多轮 rollout + loss mask + reward），且代码可查证。闭源产品（OpenAI/Claude 的工具调用 API）的训练方式未公开，不列入对比。

### 开源训练框架（star 数为 2026-07 实测）

| 框架 | star | 出处 | 与 MiniMind 的关系 |
|------|------|------|------------------|
| [verl](https://github.com/volcengine/verl) | 22.5k | 字节跳动 | 事实标准。官方支持 multi-turn tool calling（`examples/tutorial/agent_loop_get_started/`）+ SGLang multi-turn rollout，ReTool recipe 已内置。它的 multi-turn loss mask 和 MiniMind 的 response_mask 是同一问题的工业化实现 |
| [agent-lightning](https://github.com/microsoft/agent-lightning) | 17.4k | 微软 | 训练框架与 agent 代码解耦——任意 agent 零改动接入 RL 训练 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 9.8k | 社区 | 已转型为 Ray-based agentic RL 框架（PPO/DAPO/REINFORCE++/异步 RL） |
| [AReaL](https://github.com/inclusionAI/AReaL) | 5.6k | 蚂蚁+清华 | 全异步 rollout 架构，专攻"agent rollout 贵"的系统解法 |
| [RAGEN](https://github.com/RAGEN-AI/RAGEN) | 2.8k | Northwestern 等 | 研究多轮 RL 训练稳定性的学术框架，论文分析了 agent 自我演化中的训练崩塌 |
| [Agent-R1](https://github.com/0russwest0/Agent-R1) | 1.5k | 中科大 | 端到端 agent RL，代码规模适中，适合 MiniMind 之后的进阶阅读 |

### MiniMind vs 工业框架：同一问题的两种规模

| 维度 | MiniMind `train_agent.py` | 工业框架（以 verl 为代表） |
|------|--------------------------|--------------------------|
| 多轮 rollout | Python for 循环串行生成，max_turns=3 | 异步 rollout / agent loop，轮数由任务定 |
| loss mask | `response_mask`，tool result 置 0 | 同思路；multi-turn tokenization + masking 是社区专题文档级的工程问题 |
| 工具执行 | 进程内 MOCK_RESULTS 模拟 | 沙箱服务 / 真实 API / 合成环境 |
| advantage | GRPO per-response 组内 z-score | GRPO 及其变体（GiGPO/turn-level 等 recipe） |
| 推理引擎 | 自带 rollout_engine（torch/SGLang 可选） | vLLM / SGLang 服务化 |

### 大模型技术报告怎么做 agentic RL

- **Kimi K2**（Moonshot，[arXiv 2507.20534](https://arxiv.org/abs/2507.20534)）：摘要描述后训练为"大规模 agentic 数据合成 pipeline + 与真实和合成环境交互的联合 RL"；agentic 成绩 Tau2-Bench 66.1、SWE-Bench Verified 65.8（摘要报告值）
- **GLM-4.5**（智谱，[arXiv 2508.06471](https://arxiv.org/abs/2508.06471)）：355B/32B MoE，"expert model iteration + RL"的多阶段后训练；摘要报告 TAU-Bench 70.1、SWE-bench Verified 64.2。智谱同时开源了 multi-turn multi-task 训练框架 [AgentRL](https://github.com/THUDM/AgentRL)
- 两份报告都只到"做法"层面，具体 reward 设计和 mask 细节未公开——想看完整实现仍要读开源框架

### 几个值得注意的点

1. **response_mask 的设计是 Agent RL 的关键**——工具返回的结果是环境反馈，不应该对模型的 loss 有直接贡献。模型只应该对"自己决定调用什么工具"和"如何利用工具结果"负责。这个设计是主流开源 multi-turn RL 框架的通行做法，verl 社区甚至有专门文档讨论 multi-turn tokenization 和 masking 的对齐陷阱

2. **GT 验证是一种弱监督**——不需要人工标注完整的工具调用序列，只需要标注最终的正确答案。模型自己探索如何通过工具调用得到正确答案

3. **模拟执行 vs 真实执行**：训练用模拟，推理/部署用真实 API。这是一个 sim2real gap——训练时的模拟环境可能和真实 API 行为不完全一致。工业界的缓解方式是把合成环境做大做逼真（Kimi K2 的 agentic data synthesis），而不是放弃合成环境

4. **安全问题**：`calculate_math` 的模拟执行用了受限 `eval()`（空 `__builtins__` + 1 秒超时），训练环境可控。但部署时必须用进程级沙箱隔离——恶意用户可能通过 prompt injection 让模型生成危险的代码

---

## 面试话术

> "Agent RL 在 GRPO 基础上增加了多轮工具调用。一个 training step 是这样的：
> 每个 prompt 用当前 policy 生成 4 条独立轨迹，每条轨迹最多经历 3 轮"生成 → 执行工具 → 拼回上下文"的循环。
> 轨迹结束后，对每条轨迹算一个 reward 标量——有工具调用时看工具选择正确性和最终答案的 GT 匹配，没有工具调用时用 RM 打分。
> 然后 4 条轨迹做组内 z-score 得到 advantage，同一条轨迹内所有模型生成的 token 共享这个 advantage。
>
> 关键设计是 response_mask——模型生成的 token 设 mask=1 参与 loss，
> 工具返回的结果设 mask=0 不参与 loss。这确保梯度信号只影响模型的决策
> （选什么工具、传什么参数、如何利用结果），而不是评价工具本身。
>
> 但这套做法有根本性缺陷：reward 是整条轨迹一个分数，
> 5 步正确 + 1 步格式错导致失败时，前 5 步也被惩罚了——这是 per-response advantage 的 credit assignment 问题。
> PPO 的 per-token advantage（GAE 递推）能做更精确的归因，但需要额外的 Critic 模型。"

**追问"业界怎么解决 credit assignment"时的进阶版：**

> "这正是 2025 年以来 agent RL 最热的方向。我了解的代表性工作有三类：
> 一是 GiGPO，在保持 critic-free 的前提下做两层 advantage——episode 级沿用组内比较，
> step 级用 anchor state grouping，把不同轨迹里从同一环境状态出发的动作归成小组再比较，
> 这样'同一状态下调对工具和调错工具'就能拿到不同的信号；
> 二是 turn-level reward 的系统研究，把 GRPO/PPO 扩展成 turn 级变体，
> 每轮用 verifiable 规则或 LLM-as-judge 即时打分；
> 三是 ARPO，它观察到模型在工具结果返回后生成熵会显著升高，
> 就在这些高熵节点做分支采样，把 rollout 预算花在决策分叉点上。
>
> 如果让我改进 MiniMind 的实现，我会先上 turn-level 的 verifiable reward——
> 它不需要额外模型，把现有的工具对齐分从轨迹级改成每轮即时判定就行，工程代价最小。
> 工业界大规模的做法可以参考 Kimi K2 报告：大规模合成环境做 agentic 数据合成，
> 加上真实与合成环境的联合 RL。"

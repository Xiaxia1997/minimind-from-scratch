# 14 - GRPO：从 Policy Gradient 到组内比较的强化学习

> 源码位置：`trainer/train_grpo.py`（训练主循环 + rollout + reward + loss）、`trainer/rollout_engine.py`（推理引擎）
> 依赖：`dataset/lm_dataset.py:195-224`（RLAIFDataset）、`trainer/trainer_utils.py`（LMForRewardModel）
> 前置：[13-DPO](13-DPO.md)（DPO 的偏好对齐）、[10-训练阶段论](10-训练阶段论.md)（RLHF 阶段定位）

## 一句话总结

GRPO 保留了 PPO 的 ratio + clip + KL，但**扔掉了 critic/value model**，改用同一个 prompt 下 G 个回复的组内 reward 归一化来算 advantage。省了一个模型的显存，advantage 从 per-token 变成 per-response。

---

## Part 0: 从 SFT 到 RL——为什么学完 SFT 还要学 RL

### 0.1 SFT 和 RL 的根本区别

前几章我们做了 Pretrain → Full SFT → LoRA → DPO，全都是**有标准答案**的训练方式。现在到 GRPO，进入了真正的强化学习。先搞清楚这两者的根本区别：

```
SFT（有监督微调）:
  输入: prompt + 标准答案（人写的 gold response）
  训练: 对每个 token 算 cross-entropy，让模型输出尽量接近标准答案
  本质: 模仿学习——"老师写了标准答案，你照着学"

RL（强化学习）:
  输入: 只有 prompt，没有标准答案
  训练: 模型自己生成回复 → 打分 → 根据分数调整生成策略
  本质: 试错学习——"没有标准答案，你自己试，我只告诉你好不好"
```

类比：

| | SFT | RL |
|---|-----|-----|
| 教学方式 | 老师给你范文，你照着写 | 老师只打分不给范文，你自己摸索 |
| 数据来源 | 人工标注的 (prompt, response) 对 | 模型自己生成的 response |
| Loss 信号 | 每个 token 都有监督信号 | 只有整条回复的打分（稀疏信号） |
| 上限 | 被标注数据质量限制——学生最多和老师一样好 | 可以超越标注数据——自己探索出更优解 |

### 0.2 为什么 SFT 不够

SFT 有一个根本性天花板：**模型最多学到和标注数据一样好**。

```
SFT 的问题:

1. 天花板问题
   标注员写的答案质量有限 → 模型也就这个水平
   想要更好？需要更好的标注 → 但优质标注极贵

2. 只会模仿，不会探索
   SFT 只见过标准答案，从没尝试过其他回答方式
   → 不知道什么是"比标准答案更好"
   → 不知道什么是"错误但看起来合理"

3. 推理能力难以通过模仿获得
   DeepSeek-R1 的关键发现:
   SFT 可以教会模型"写出 <think>...</think> 格式"
   但真正的反思、回溯、自我验证 → 是 RL 自己"长出来的"
   这些行为在 SFT 数据里根本没有模板可模仿
```

### 0.3 RL 怎么做到"超越标注数据"

核心机制是 **explore + exploit**（探索 + 利用）：

```
第 1 轮: 模型对 prompt "2+3=?" 生成 6 个回答:
  "5"       → reward = 1.0  ✓
  "8"       → reward = 0.0  ✗
  "23"      → reward = 0.0  ✗
  "2+3=5"   → reward = 1.0  ✓
  "六"      → reward = 0.0  ✗
  "答案是5" → reward = 1.0  ✓

→ 更新策略: 增大 "5"、"2+3=5"、"答案是5" 的生成概率
            减小 "8"、"23"、"六" 的生成概率

第 N 轮: 模型学会了新的推理方式，SFT 数据里从来没见过
  "<think>让我算一下，2加3等于5</think>答案是5"
  → 这种带推理过程的回答也拿到了高分
  → 模型自己发现了"先想再答"的策略
```

**SFT 是从人类标注里学（bounded by human）；RL 是从自己的尝试里学（bounded by reward signal）。**

这就是为什么 DeepSeek-R1 能通过 RL 涌现出 SFT 数据里根本没有的推理行为。

### 0.4 DPO vs GRPO：都是对齐，有什么不同

上一章的 DPO 也是"偏好对齐"，和 GRPO 有什么关系？

```
DPO（上一章）:
  数据: 人标注的 (chosen, rejected) 偏好对
  训练: 离线，直接在偏好对上优化，不需要模型生成
  本质: 把 RL 的 reward 信号藏进了 loss 函数里（implicit reward）
  局限: 仍然依赖离线标注数据，模型不会自己探索

GRPO（本章）:
  数据: 模型自己生成 + reward model 打分
  训练: 在线，模型先生成 → 打分 → 用分数更新策略
  本质: 显式的 RL 循环（explore → score → update）
  优势: 模型可以探索标注数据覆盖不到的空间
```

| | DPO | GRPO |
|---|-----|------|
| 需要模型生成 | 不需要（离线数据） | 需要（在线 rollout） |
| 偏好信号来源 | 人工标注 | reward model + 规则 |
| 能否超越标注数据 | 很难 | 可以 |
| 训练成本 | 低（和 SFT 差不多） | 高（要在线推理 + 多个模型） |
| 训练稳定性 | 高 | 低（需要 clip、KL 等技巧） |

理解了 SFT → DPO → GRPO 的递进关系后，接下来的问题是：**RL 具体怎么根据打分来更新模型参数？** 这就要从 Policy Gradient 说起。

---

## Part 1: RL 怎么更新模型——从 Policy Gradient 到 GRPO

### 1.1 RL 核心概念：状态、动作、策略、轨迹

RL 的世界观很简单：

```
看到状态 s → 选一个动作 a → 环境给奖励 r → 进入新状态 s' → 循环
```

在 LLM 场景下，这些概念对应什么？

| RL 概念 | 符号 | LLM 里是什么 | 举例 |
|--------|------|------------|------|
| 状态 State | $s_t$ | prompt + 已生成的前 $t$ 个 token | `"2+3=? 答案是"` |
| 动作 Action | $a_t$ | 下一个要生成的 token | `"5"` |
| 策略 Policy | $\pi_\theta(a \mid s)$ | 模型的 softmax 输出概率 | $P(\text{"5"} \mid \text{"2+3=? 答案是"}) = 0.7$ |
| 奖励 Reward | $r$ | reward model / 规则给的分数 | 答对了 $+1.0$ |
| 轨迹 Trajectory | $\tau$ | 一条完整的生成序列 | `"2+3=? 答案是5"` |

**策略 $\pi_\theta(a \mid s)$** 就是模型本身——给定当前状态（已有的 token 序列），输出词表上每个 token 的概率分布。比如当前序列是 `"2+3=? 答案是"`，策略输出 $\pi(\text{"5"} \mid s) = 0.7, \;\pi(\text{"4"} \mid s) = 0.2, \;\pi(\text{"6"} \mid s) = 0.1$。训练目标就是调整参数 $\theta$，让策略更倾向于选好的 token。

**轨迹 $\tau$** 是一条完整的「状态-动作-奖励」链：

$$\tau = (s_0, a_0, r_0, \; s_1, a_1, r_1, \; \dots, \; s_T)$$

**奖励 $r$** 在 LLM 场景有一个关键特征：**只有生成完整条回复后才有**。中间每一步的 $r_t = 0$，只有最后一步 $r_T = R$（reward model 的打分）。这和游戏 RL（每步都有即时奖励）有本质区别，后面很多设计选择都源于此。

### 1.2 Policy Gradient：怎么根据分数算梯度

#### 目标函数

SFT 的 loss 是 cross-entropy——有标准答案，直接算偏差。RL 没有标准答案，目标换成：**最大化策略的期望回报**。

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big]$$

读作："按当前策略 $\pi_\theta$ 采样出的所有可能轨迹，对它们的总回报 $R(\tau)$ 求期望"。

$R(\tau)$ 是整条轨迹的总 reward——不管回复有多少个 token，最终只有一个标量分数。

训练目标：找到 $\theta$ 使得 $J(\theta)$ 最大。和 SFT 最大化正确 token 的 log 概率异曲同工，只是 SFT 的"正确"由标准答案定义，RL 的"好"由 reward 定义。

#### 推导：怎么对 $J(\theta)$ 求梯度

$J(\theta)$ 展开成对所有可能轨迹的加权求和：

$$J(\theta) = \sum_{\tau} P(\tau \mid \theta) \cdot R(\tau)$$

$P(\tau \mid \theta)$ 是"在策略 $\pi_\theta$ 下，轨迹 $\tau$ 被采样出来的概率"。

对 $\theta$ 求梯度。$R(\tau)$ 是环境给的分数，不依赖 $\theta$：

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau \mid \theta) \cdot R(\tau)$$

问题：$\nabla_\theta P(\tau \mid \theta)$ 不好直接算。

**关键技巧：log-derivative trick**

数学恒等式：对任意 $f(x) > 0$，有 $\nabla \log f = \frac{\nabla f}{f}$，两边乘 $f$ 得：

$$\nabla f = f \cdot \nabla \log f$$

应用到 $P(\tau \mid \theta)$：

$$\nabla_\theta P(\tau \mid \theta) = P(\tau \mid \theta) \cdot \nabla_\theta \log P(\tau \mid \theta)$$

代回梯度公式：

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau \mid \theta) \cdot \nabla_\theta \log P(\tau \mid \theta) \cdot R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\big[\nabla_\theta \log P(\tau \mid \theta) \cdot R(\tau)\big]$$

**又变回了期望！** 这意味着可以通过采样来估计梯度——不需要遍历所有可能的轨迹，只要采样几条算平均就行。

**最后一步：把轨迹概率拆成每一步**

一条轨迹的概率 = 每一步策略选择的概率 × 环境转移概率：

$$P(\tau \mid \theta) = P(s_0) \cdot \prod_{t=0}^{T-1} \underbrace{\pi_\theta(a_t \mid s_t)}_{\text{策略选择（模型决定的）}} \cdot \underbrace{P(s_{t+1} \mid s_t, a_t)}_{\text{环境转移（模型控制不了的）}}$$

取 log 后乘积变求和：

$$\log P(\tau \mid \theta) = \log P(s_0) + \sum_{t=0}^{T-1}\big[\log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t)\big]$$

对 $\theta$ 求导时，$P(s_0)$ 和环境转移概率 $P(s_{t+1} \mid s_t, a_t)$ 和 $\theta$ 无关，导数为 0：

$$\nabla_\theta \log P(\tau \mid \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

**只剩策略部分！** 环境怎么转移、初始状态怎么分布——都不影响梯度。这就是 Policy Gradient 的核心优势：不需要知道环境的动力学模型。

#### 最终公式

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau)\right]}$$

逐块翻译：

| 部分 | 含义 |
|------|------|
| $\mathbb{E}_{\tau \sim \pi_\theta}[\cdot]$ | 用当前策略采样 $N$ 条轨迹，对括号里的东西求平均 |
| $\sum_t$ | 轨迹里每一步都贡献一份梯度 |
| $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | **方向向量**——沿它更新参数，可增大"在 $s_t$ 时选 $a_t$"的概率 |
| $R(\tau)$ | 整条轨迹的总分做权重：分高 → 大步走；分低 → 反方向走 |

直觉版：

> 采样一批轨迹 → 每条轨迹的每一步 → 把"增大这个 token 概率"的方向向量 × 这条轨迹的总分 → 全部加起来求平均 → 这就是梯度，照着走一步。

#### 和 SFT 的对比

```
SFT:  对每个 token: grad(log P(标准答案token | 前缀)) × 1
                                                       ↑ 权重固定为 1
      含义: 标准答案里每个 token 都是"对的"，一视同仁地提高概率

RL:   对每个 token: grad(log P(生成token | 前缀)) × R(τ)
                                                     ↑ 权重是 reward 分数
      含义: reward 高的回复里的 token 大力提高概率
           reward 低/负的回复里的 token 降低概率
```

唯一区别就是权重：SFT 权重固定 1，RL 权重是 reward。

#### 问题：所有 token 用同一个分数

$R(\tau)$ 是整条回复的总分。但梯度公式里，**每个 token 都乘这同一个 $R(\tau)$**：

```
模型生成: "我觉得答案是8不对应该是5"
reward = 1.0（因为最后说了5，reward model 给了正分）

每个 token 乘的系数:
  "我"     × 1.0   ← 无所谓的 token，也被强化了
  "觉"     × 1.0
  "得"     × 1.0
  "答案是" × 1.0
  "8"      × 1.0   ← 错误的 token！也被强化了！
  "不对"   × 1.0
  "应该是" × 1.0
  "5"      × 1.0   ← 真正对的 token
```

"8"明明是错的，但因为整条回复最终答对了，它也跟着被强化。这就是 **credit assignment 问题**——所有 token 共享同一个分数，分不清谁好谁差。

### 1.3 引入 Return $G_t$——两个改进，但在 LLM 里都失效

1.2 的问题是所有 token 乘同一个 $R(\tau)$。$G_t$ 试图改进这一点，做了**两件事**：

$$G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$$

**改进 1：求和从 $t$ 开始，不从 0 开始**

$R(\tau)$ 把整条轨迹的 reward 全加起来，$G_t$ 只加从第 $t$ 步到最后一步的 reward。逻辑很直觉：第 5 步生成的 token 不应该为第 1~4 步产生的 reward 负责。

**改进 2：$\gamma$（gamma）是衰减系数，让远处的 reward 权重更低**

$\gamma$ 通常取 0.99 之类的值。离当前 token 越远的 reward，被乘以越多次 $\gamma$，权重越低。直觉：第 5 步生成的 token 对紧接着第 6 步的 reward 影响大，对第 100 步的 reward 影响小。

梯度公式相应变成：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]$$

对比 1.2，唯一的变化就是每个 token 乘的系数从 $R(\tau)$ 变成了 $G_t$。

用具体数字看这两个改进的效果：

```
游戏场景（每步都有即时 reward，gamma=0.9）:
  step 0: r_0=+1    step 1: r_1=+5    step 2: r_2=-10

  1.2 每个 token 的系数:
    step 0 × R(τ) = × (1+5-10) = × (-4)     ← 全部一样
    step 1 × R(τ) = × (-4)
    step 2 × R(τ) = × (-4)

  1.3 每个 token 的系数:
    step 0 × G_0 = × (1 + 0.9×5 + 0.81×(-10)) = × (-2.6)
    step 1 × G_1 = × (5 + 0.9×(-10))           = × (-4.0)
    step 2 × G_2 = × (-10)                      = × (-10.0)
    ↑ 三个改进同时生效:
      (a) step 0 不包含它之前的 reward（这里 step 0 就是第一步，没区别）
      (b) gamma 让远处的 -10 对 step 0 的影响被衰减（0.81×(-10) 而不是 1×(-10)）
      (c) step 2 自己犯的错 -10 全额承担
```

**但在 LLM 场景，这两个改进都失效了：**

```
LLM 场景的特殊性:
  1. reward 只有最后一步才有（reward model 看完整条回复才打分）
     token 0: r=0   token 1: r=0   ...   最后一个 token: r=1.0
     → 改进 1 失效: 求和从 t 开始还是从 0 开始，反正前面全是 0
       G_0 = 0 + 0 + ... + 1.0 = 1.0
       G_1 = 0 + ... + 1.0     = 1.0
       G_T = 1.0
       → 所有 token 的 G_t 都一样，和 R(τ) 没区别

  2. gamma 通常设为 1（不打折）
     → 改进 2 失效: gamma^(k-t) = 1^(k-t) = 1，没有衰减
```

结论：$G_t$ 在游戏 RL（每步都有即时 reward）里确实有用，但在 LLM RL 里**退化成了 1.2**——所有 token 还是乘同一个数。需要别的办法。

### 1.4 Advantage：让不同 token 真正有不同的权重

1.3 在 LLM 场景失效，因为 $G_t$ 对所有 token 都是同一个值。怎么办？**减掉一个因 token 位置而异的 baseline**。

#### Critic（Value Model）是什么

Critic 是一个**额外的神经网络** $V_\psi(s_t)$，和生成模型（Actor/Policy）分开。它的输入是当前的 token 序列（prompt + 已生成部分），输出是一个标量——**预测从这个状态继续生成下去，平均能拿多少 reward**。

```
Critic 的工作方式:

  输入: "2+3=? 答案是"      → V = 0.5    "生成到这里，大约 50% 概率最终答对"
  输入: "2+3=? 答案是8"     → V = 0.2    "写了 8，形势不好，大约 20% 概率答对"
  输入: "2+3=? 答案是8不对"  → V = 0.7    "自我纠正了，70% 概率答对"
  输入: "2+3=? ...应该是5"  → V = 0.9    "基本对了，90% 概率答对"
```

Critic 怎么知道这些数字？它也是训练出来的——用实际的 $G_t$（回复的真实 reward）作为监督信号，训练 critic 去预测。Critic 有自己的 loss：$\mathcal{L}_{\text{value}} = (V_\psi(s_t) - G_t)^2$。

**这也是 PPO 显存大的原因——除了 Actor（生成模型），还要同时训一个 Critic（打分模型）。**

#### Advantage = 实际结果 - 预期水平

$$A_t = G_t - V(s_t)$$

$G_t$ 是实际发生的 reward，$V(s_t)$ 是 critic 预测的"这个位置的平均预期"。

梯度公式相应变成：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t\right]$$

每个 token 乘的系数从 $R(\tau)$（1.2）/ $G_t$（1.3）变成了 $A_t$（1.4）：

```
1.2: 每个 token 的梯度 × R(τ)              ← 所有 token 同一个数
1.3: 每个 token 的梯度 × G_t               ← LLM 场景下还是同一个数
1.4: 每个 token 的梯度 × (G_t - V(s_t))    ← 终于不同了！因为 V(s_t) 不同
```

用具体例子看：

```
prompt: "2+3=?"
模型生成: "答案是8不对应该是5"
最终 reward R = 1.0（因为最后答了5）

G_t = 1.0（对所有 token 都一样——LLM 只有最后给分）

V(s_t) 在每个位置不同（critic 的预测）:

位置    当前前缀                    V(s_t)   A_t = 1.0 - V     含义
─────────────────────────────────────────────────────────────────────
t=0    "2+3=? 答"                  0.6      +0.4         轻微正向——实际比预期好一点
t=1    "2+3=? 答案是"              0.5      +0.5         类似
t=2    "2+3=? 答案是8"             0.2      +0.8         V 很低（写了 8 形势差），
                                                          但最终翻盘了 → advantage 大
t=3    "2+3=? 答案是8不对"          0.7      +0.3         自我纠正拉回了预期
t=4    "2+3=? ...应该是5"          0.9      +0.1         已经快对了，超出预期不多
```

**含义**：

- $A_t > 0$：实际结果比 critic 预测的好 → 增大这个 token 的概率
- $A_t < 0$：实际结果比 critic 预测的差 → 减小这个 token 的概率
- $A_t \approx 0$：和预期差不多 → 不怎么更新

注意 $A_t$ 衡量的是"**惊喜程度**"，不是"绝对好坏"。写了"8"之后 V 只有 0.2（critic 觉得完了），但最终居然拿到了 1.0 → advantage = 0.8 很高。这不是因为"8"本身好，而是因为后面的自我纠正超出了预期。

### 1.5 GAE：Advantage 的两种极端算法及折中

1.4 说了 $A_t = G_t - V(s_t)$，但实际中这个公式有两种极端的计算方式，各有问题。GAE 就是在这两个极端之间做折中。

#### 两个极端

**极端 A：Monte Carlo Advantage**——用完整的真实 reward 来算

$$A_t^{\text{MC}} = G_t - V(s_t) = (\text{实际拿到的总 reward}) - (\text{critic 预测})$$

好处：$G_t$ 是真实发生的，**没有偏差**。
坏处：同一个 prompt 生成 10 次，每次 $G_t$ 可能差很多（有时答对有时答错），**方差大**，训练不稳定。

**极端 B：TD error**——只看一步真实 reward，后面全靠 critic 估

$$A_t^{\text{TD}} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

含义："实际走一步拿到的 reward $r_t$，加上下一个状态 critic 的预测 $V(s_{t+1})$"，和"当前状态 critic 的预测 $V(s_t)$"的差。

```
具体例子:
  当前状态 s_t = "2+3=? 答案是"，critic 预测 V(s_t) = 0.5
  模型生成了 token "8"，即时 reward r_t = 0（中间步没有 reward）
  下一状态 s_{t+1} = "2+3=? 答案是8"，critic 预测 V(s_{t+1}) = 0.2

  delta_t = 0 + 0.2 - 0.5 = -0.3
  含义: 生成 "8" 之后，critic 觉得形势从 0.5 跌到了 0.2 → 这一步是负向的
```

好处：只依赖一步的随机性，**方差小**。
坏处：$V(s_{t+1})$ 是 critic 预测的，critic 不准就全错了，**偏差大**。

```
两个极端的 tradeoff:

                方差（不稳定程度）    偏差（系统性偏移）
Monte Carlo     高                   低（用的是真实 reward）
TD error        低                   高（依赖 critic 估值）
```

#### GAE：在两个极端之间滑动

GAE 把不同步长的 TD error 加起来，用 $\lambda$ 控制看多远：

$$\hat{A}_t^{\text{GAE}} = \delta_t + \gamma\lambda \cdot \delta_{t+1} + (\gamma\lambda)^2 \cdot \delta_{t+2} + \cdots$$

递推形式：$\hat{A}_t^{\text{GAE}} = \delta_t + \gamma\lambda \cdot \hat{A}_{t+1}^{\text{GAE}}$

$\lambda$ 控制滑动位置：

| $\lambda$ | 效果 | 等价于什么 |
|-----------|------|-----------|
| $\lambda = 0$ | 只用 $\delta_t$ | 极端 B（TD error）——方差低、偏差高 |
| $\lambda = 1$ | 加到无穷远 | 接近极端 A（Monte Carlo）——偏差低、方差高 |
| $\lambda = 0.95$（常用） | 看 ~20 步的 TD error | 折中——方差和偏差都适中 |

$\gamma$（另一个参数）控制的是不同的事——未来 reward 打多少折：$\gamma = 0.99$ 意思是 100 步后的 reward 只算 $0.99^{100} \approx 37\%$。

**关键：和 1.4 一样，GAE 给每个 token 一个独立的 advantage $\hat{A}_t$。** $T=100$ 的序列有 100 个不同的值。这就是 PPO 的 per-token advantage。

**代价：GAE 依赖 critic $V(s_t)$。** critic 要跟 actor 一起训练，显存翻倍。这就是后面 GRPO 要解决的问题。

### 1.6 演进链路小结

回顾一下 1.2~1.5 走过的路——每一步解决了什么问题、引入了什么新问题：

| 阶段 | 每个 token 乘的系数 | 解决了什么 | 新问题 |
|------|-------------------|-----------|--------|
| 1.2 原始 PG | $R(\tau)$ | 能根据 reward 更新策略 | 所有 token 同一个分数，好坏不分 |
| 1.3 Return $G_t$ | $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ | 只对之后的 reward 负责 + gamma 衰减 | LLM 场景退化成 $R(\tau)$ |
| 1.4 Advantage | $A_t = G_t - V(s_t)$ | 减掉 baseline，不同 token 终于有不同系数 | 需要训 critic $V(s_t)$ |
| 1.5 GAE | $\hat{A}_t^{\text{GAE}}$ | MC 和 TD 的折中，降低方差 | 仍然需要 critic |

一句话版本：

> **系数从 $R(\tau)$ → $G_t$ → $A_t$ → $\hat{A}_t^{\text{GAE}}$，越来越精细，但代价是需要 critic。**
> PPO 用 ratio + clip 稳定更新，GRPO 用组内 z-score 干掉 critic。接下来分别讲。

---

## Part 2: PPO——怎么安全地更新策略

### 2.1 Importance Sampling Ratio

Policy Gradient 要求数据来自当前策略 $\pi_\theta$，但每次更新后采样太贵。PPO 用旧策略 $\pi_{\text{old}}$ 的数据多轮更新，通过 importance sampling ratio 修正：

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

工程里用 log 形式：$r_t(\theta) = \exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{old}}(a_t \mid s_t))$

| $r_t$ | 含义 |
|--------|------|
| $= 1.0$ | 新旧策略对这个 token 的概率一样 |
| $= 1.5$ | 新策略更想生成这个 token（概率涨了 50%） |
| $= 0.7$ | 新策略更不想生成（概率降了 30%） |

ratio 不是 reward ratio，它是**概率比**——衡量新策略相对于旧策略对同一个 token 的偏好变化。

### 2.2 PPO Clip

$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t\left[\min\!\big(r_t(\theta) \hat{A}_t, \;\operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\big)\right]$$

公式看着复杂，用具体数字拆开就清楚了。设 $\varepsilon = 0.2$，则 $\operatorname{clip}(r_t, 0.8, 1.2)$。

**情况 A：$\hat{A}_t = +1.0$（好 token，想增大概率）**

| $r_t$ | $\operatorname{clip}(r_t)$ | $r_t \cdot \hat{A}_t$ | $\operatorname{clip}(r_t) \cdot \hat{A}_t$ | $\min$ 取值 | 效果 |
|------|---------------------------|----------------------|-------------------------------------------|------------|------|
| 1.3 | 1.2 | 1.3 | 1.2 | **1.2** | 被 clip 拦住——概率已经涨够了，不让再涨 |
| 1.1 | 1.1 | 1.1 | 1.1 | **1.1** | 没超界，正常通过 |
| 0.9 | 0.9 | 0.9 | 0.9 | **0.9** | 没触下限，正常通过 |

**情况 B：$\hat{A}_t = -1.0$（坏 token，想减小概率）**

| $r_t$ | $\operatorname{clip}(r_t)$ | $r_t \cdot \hat{A}_t$ | $\operatorname{clip}(r_t) \cdot \hat{A}_t$ | $\min$ 取值 | 效果 |
|------|---------------------------|----------------------|-------------------------------------------|------------|------|
| 1.3 | 1.2 | -1.3 | -1.2 | **-1.3** | 没被拦（$-1.3 < -1.2$）——坏 token 概率还在涨，梯度继续纠正 |
| 0.7 | 0.8 | -0.7 | -0.8 | **-0.8** | 被 clip 拦住——概率已经降够了，别过度惩罚 |
| 0.3 | 0.8 | -0.3 | -0.8 | **-0.8** | 同上，防止把坏 token 概率踩到地板 |

**一句话总结**：

- 好 token（$\hat{A} > 0$）：$\min$ 用上限 $1+\varepsilon$ 防止概率无限提高——**别一步登天**
- 坏 token（$\hat{A} < 0$）：$\min$ 用下限 $1-\varepsilon$ 防止概率过度减小——**别一步踩死**
- 但如果模型把好 token 的概率**错误地降低了**（$r_t < 1$），clip 不拦，梯度继续纠正回来

### 2.3 PPO 的完整 loss

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{actor}}}_{\text{ratio+clip}} + c_v \underbrace{\mathcal{L}_{\text{value}}}_{\text{训 critic}} - c_e \underbrace{\mathcal{L}_{\text{entropy}}}_{\text{防塌缩}} + \beta \underbrace{\mathcal{L}_{\text{KL}}}_{\text{约束偏离}}$$

| loss 项 | 作用 | 更新谁 |
|---------|------|--------|
| $\mathcal{L}_{\text{actor}}$ | PPO-clip，让好 token 概率升、坏 token 概率降 | actor/policy |
| $\mathcal{L}_{\text{value}}$ | 训练 critic 预测 return，才能算 GAE | critic |
| $\mathcal{L}_{\text{entropy}}$ | 防止 policy 过早塌缩到少数 token | actor/policy |
| $\mathcal{L}_{\text{KL}}$ | 防止 policy 偏离 reference model 太远（reward hacking） | actor/policy |

### 2.4 PPO 需要的 4 个模型

| 模型 | 对应公式项 | 是否训练 |
|------|-----------|---------|
| Policy $\pi_\theta$ | $\pi_\theta(a_t \mid s_t)$ ratio 分子 | 训练 |
| Old Policy $\pi_{\text{old}}$ | $\pi_{\text{old}}(a_t \mid s_t)$ ratio 分母 | 每轮冻结 |
| Critic $V_\psi$ | $V_\psi(s_t)$ TD error / GAE | 训练 |
| Reference $\pi_{\text{ref}}$ | $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 长期锚点 | 长期冻结 |
| Reward Model | $r_t^{\text{total}}$ 打分 | 通常冻结 |

---

## Part 3: GRPO——扔掉 Critic

### 3.1 动机：PPO 在 LLM 场景的问题

```
PPO 显存 = Actor + Critic + Ref + Reward Model ≈ 4× 模型大小
7B 模型: 4 × 14GB = 56GB → 需要大量 A100

Critic 的额外问题:
  1. LLM 的状态（token 序列）维度巨大，V(s) 很难训准
  2. Critic 和 Actor 共享 backbone 时容易互相干扰
  3. 训练不稳定——critic 估不准 → advantage 不准 → actor 更新方向错
```

### 3.2 GRPO 的核心想法

**同一道题多答几次，好答案和坏答案之间的差距自然就出来了。不需要 critic。**

对每个 prompt $q$，用旧策略采样 $G$ 个回答 $o_1, o_2, \dots, o_G$，然后 reward model 打分 $r_1, r_2, \dots, r_G$。

Advantage 用组内 z-score：

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

### 3.3 Advantage 粒度的根本差异

| | PPO | GRPO |
|---|-----|------|
| Advantage 来源 | critic + GAE | 组内 z-score |
| 粒度 | **per-token**：每个 token 独立的 $\hat{A}_t$ | **per-response**：同一个回答的所有 token 共享 $\hat{A}_i$ |
| 需要 critic | 需要 | 不需要 |
| 显存 | Actor + Critic + Ref | Actor + Ref |

**per-response advantage 的含义**：第 $i$ 个回答的所有 token，不管是第 1 个还是第 100 个，都用同一个 $\hat{A}_i$。

对 math reasoning 来说这通常没问题——最终答案对了就是全对。
对 agent 场景来说这是一个根本性缺陷——第五部分会详细讨论。

### 3.4 GRPO Loss 完整公式

$$\boxed{\mathcal{L}^{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^{G}} \left[ \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\!\big(r_{i,t}(\theta) \cdot \hat{A}_i, \;\operatorname{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \cdot \hat{A}_i\big) - \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]}$$

逐项拆解：

| 符号 | 含义 |
|------|------|
| $\mathbb{E}_{q, \{o_i\}}$ | 对每个 prompt 和它的 G 个回答求期望 |
| $\frac{1}{\|o_i\|} \sum_t$ | 对第 $i$ 个回答的每个 token 求平均 |
| $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\text{old}}(a_{i,t} \mid s_{i,t})}$ | 第 $i$ 个回答第 $t$ 个 token 的 ratio |
| $\hat{A}_i$（不是 $\hat{A}_{i,t}$） | 下标是 $i$ 不是 $t$——同一个回答所有 token 共享 |
| $\beta \cdot \mathrm{KL}$ | KL 散度惩罚，别偏离 reference 太远 |

---

## Part 4: MiniMind 的 GRPO 实现

### 4.1 Rollout：用 policy 生成回复

```python
# train_grpo.py:80-91
rollout_result = rollout_engine.rollout(
    prompt_ids=prompt_inputs["input_ids"],
    attention_mask=prompt_inputs["attention_mask"],
    num_generations=args.num_generations,       # G=6
    max_new_tokens=args.max_gen_len,            # 最多 1024 token
    temperature=0.8,                            # 温度采样，保证 6 个回复有差异
)
old_per_token_logps = rollout_result.per_token_logps   # [B*6, R] rollout 时记录的 log prob
```

`temperature=0.8` 是关键——如果温度太低，6 个回复几乎一样，组内比较没有信息量。

### 4.2 Reward 计算：规则 + Reward Model

```python
# train_grpo.py:37-68
def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)
    for i in range(batch_size):
        for j in range(args.num_generations):
            response = responses[response_idx]

            # 规则分
            rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5     # 长度
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5  # 思考
                rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25           # 格式
            rewards[response_idx] -= rep_penalty(answer)              # 重复惩罚

            # RM 分
            score = reward_model.get_score(messages, answer)
            rewards[response_idx] += score
    return rewards
```

类似 DeepSeek-R1 的思路——规则分（accuracy + format）+ RM 分的混合。

### 4.3 Advantage 归一化

```python
# train_grpo.py:121-124
grouped_rewards = rewards.view(-1, args.num_generations)          # [B, 6]
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*6]
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*6]
advantages = (rewards - mean_r) / (std_r + 1e-4)                 # [B*6]
```

这就是上面公式里的 $\hat{A}_i = (r_i - \mu) / (\sigma + \epsilon)$。`repeat_interleave` 让 mean/std 广播到组内每个回复。

### 4.4 Policy Loss

MiniMind 默认用 CISPO 变体（clipped importance sampling）：

```python
# train_grpo.py:135-137 (loss_type == "cispo")
clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()    # max=5.0, detach 不参与梯度
per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
```

和标准 PPO-clip 的区别：CISPO 把 ratio clamp 后 `.detach()`，ratio 只作为重要性采样权重，不参与梯度计算。

标准 GRPO loss 也实现了：

```python
# train_grpo.py:138-142 (loss_type == "grpo")
clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
per_token_loss1 = ratio * advantages.unsqueeze(1)
per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```

这和 PPO clip 公式完全对应：`min(ratio * A, clip(ratio) * A) - beta * KL`。

### 4.5 KL 散度惩罚

```python
# train_grpo.py:132-133
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1           # Schulman k3 estimator
```

这是 **KL 散度的 Schulman 近似**：

$$\hat{\mathrm{KL}} = \exp\!\big(\log \pi_{\text{ref}} - \log \pi_\theta\big) - \big(\log \pi_{\text{ref}} - \log \pi_\theta\big) - 1 = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

性质：当 $\pi_\theta = \pi_{\text{ref}}$ 时 $\hat{\mathrm{KL}} = 0$，偏离时 $> 0$，始终非负，比 log-ratio 更对称。

为什么用近似而不是精确 KL？精确 KL $= \sum_a \pi_\theta(a) \log(\pi_\theta(a) / \pi_{\text{ref}}(a))$ 需要遍历整个词表（6400 个 token），太贵。Schulman 近似只需要 rollout token 的 logprob，O(1) per token。

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
│     loss = -(clip(ratio) * A_i * new_logps - beta * kl)        │
│     ↓                                                           │
│  ⑧ loss.backward() → clip_grad_norm → optimizer.step           │
│     ↓                                                           │
│  ⑨ 同步更新 rollout 引擎的 policy 权重                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 5: GRPO 的局限性——per-response advantage 的代价

### 5.1 对 Math Reasoning 没问题

Math 场景的特征：最终答案对了 = 全对，答案错了 = 全错。

per-response advantage 天然匹配这种 episode-level reward。整条推理链共享一个 advantage 没问题——反正我们关心的就是"整条链的结果"。

这也是为什么 DeepSeek-R1 用 GRPO 在 math/code reasoning 上效果极好。

### 5.2 对 Agent 场景有三个问题

**问题 1：per-response advantage → token 级信号错误**

用一个具体的 agent 轨迹来看这个问题：

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

  Turn 1 "get_weather" 的梯度 × A_i = × (-1.0)   ← 这步明明做对了
  Turn 2 "get_forecast" 的梯度 × A_i = × (-1.0)   ← 这步也做对了
  Turn 3 格式错误 token 的梯度  × A_i = × (-1.0)   ← 真正该惩罚的
  → 三步拿到完全相同的信号，模型学到的是噪音
```

PPO 的 per-token advantage 不会有这个问题——GAE 会把最终的负 reward 按 $\gamma\lambda$ 衰减回传，越早的正确步骤受影响越小。

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

MiniMind 的 `train_agent.py`（[15-AgentRL](15-AgentRL.md)）用了几个技巧部分缓解：

1. **response_mask**：工具返回的 token 设 mask=0 不参与 loss，只有模型生成的 token 受 advantage 影响
2. **多维 reward**：不只看最终结果，还奖励工具选择正确性、格式正确性，提供更密集的信号
3. **GT 弱监督**：用 ground truth 验证最终答案，给出连续的匹配分（`2.5 * len(verified) / len(gt)`），不是纯 0/1

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
| 1. Cold Start SFT | 几千条长 CoT 数据做 SFT | 给模型一个"推理的起点"——格式统一，会写 `<think>...</think>` |
| 2. Reasoning RL | GRPO + 纯规则 reward（accuracy + format） | 模型自己长出推理能力（"aha moment"），涌现反思、回溯、自我验证 |
| 3. Rejection Sampling + SFT | 用 Stage 2 模型生成 → 筛选 → 混合通用数据 → **从头 SFT** | Stage 2 模型推理强但通用能力退化，需要 SFT 拉回 |
| 4. Full RL | GRPO + accuracy + helpfulness + harmlessness reward | 最终对齐——推理 + 通用 + 安全 |

关键发现：Stage 2 用**纯规则 reward**（不需要 RM）就训出了强推理能力。MiniMind 的 GRPO reward 设计受此启发。

---

## 关键超参数

| 超参数 | MiniMind | DeepSeekMath / R1 | 含义 |
|--------|---------|-------------------|------|
| `learning_rate` | 3e-7 | 1e-6（DeepSeekMath 论文；R1 未公开） | RL 训练对学习率极度敏感 |
| `batch_size` | 2 | 大得多 | 每步 2 prompt × 6 回复 = 12 序列 |
| `num_generations` (G) | 6 | 64（DeepSeekMath 论文；R1 未公开） | 组越大 advantage 估计越稳 |
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
| 需要 Reward Model | 是 | 推理 RL 阶段纯规则；最终对齐阶段用 RM | 是 | 否 |
| 需要 Value Model | 否 | 否 | 是 | 否 |
| 在线采样 | 是 | 是 | 是 | 否 |
| Advantage 粒度 | per-response | per-response | per-token | N/A |
| 组大小 G | 6 | 64 | N/A | N/A |
| KL 估计 | Schulman 近似 | 类似 | 标准 KL / reward penalty | 隐式 |

---

## 面试话术

> "GRPO 是 DeepSeek 提出的 RL 对齐方法。要理解它，先看演进链：
> Policy Gradient 用整条轨迹 reward 更新所有 step，太粗；
> 引入 advantage = G_t - V(s_t) 后可以区分好坏 token，但需要 critic；
> PPO 加 ratio + clip 实现数据复用和稳定更新，但 critic 显存翻倍。
>
> GRPO 的核心改动：扔掉 critic，用组内 z-score 替代。
> 对每个 prompt 采样 G 个回复，reward 在组内归一化得到 advantage。
> 这等价于用组均值替代 value baseline——'这个回复比同组平均水平好多少'。
> 好处是省掉 value model，坏处是 advantage 从 per-token 变成 per-response，
> credit assignment 变粗了。
>
> 对 math reasoning 没问题——最终答案对了就是全对。
> 对 agent 有根本性缺陷——5 步正确 + 1 步格式错，per-response advantage 会惩罚全部 token。
> DeepSeek-R1 进一步证明，纯规则 reward 的 GRPO 就能训出强推理能力——
> 涌现了反思、回溯等行为，不是 SFT 教的而是 RL 自己长出来的。"

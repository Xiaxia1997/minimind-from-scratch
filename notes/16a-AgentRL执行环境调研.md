# 16a - Agent RL 执行环境调研：业界怎么做工具执行

> 调研日期：2026-07-21
> 动机：MiniMind 第 16 章用 `MOCK_RESULTS` 字典模拟工具执行，业界真实做法是什么？

---

## 一句话结论

**MiniMind 的 mock 字典不是玩具简化，而是"合成环境"这条主流路线的最小实现。** 业界实际做法是一个**光谱**——从纯 mock 字典到 Docker 沙箱到 Kubernetes 万级并发实例，选择取决于任务类型和规模，但核心原则一样：训练时需要**确定性、速度、安全**，所以几乎都不直连真实 API。

---

## Part 1：业界工具执行环境的三条路线

### 路线 1：合成/模拟环境（MiniMind 所在位置）

**原理**：用代码模拟工具的输入输出，不连真实 API。

```
MiniMind:    MOCK_RESULTS 字典 → lambda 函数返回固定结果
Kimi K2:     "tool simulator (functionally equivalent to a world model)" → 维护状态的模拟器
Agent World Model: 1000 个 code-driven + database-backed 合成环境
EnvFactory:  85 个从真实资源验证的可执行环境
```

**为什么这是主流**：

| 优势 | 说明 |
|------|------|
| 确定性 | 相同输入 → 相同输出，RL 需要稳定的环境信号 |
| 速度 | 进程内执行，无网络延迟，单次 < 1ms |
| 成本 | 零 API 费用，batch × num_gen × max_turns 次调用全免费 |
| 安全 | 模型生成的代码不接触真实系统 |
| 可扩展 | 可以程序化生成任意多环境（K2 生成了 20,000+ 合成工具） |

**Kimi K2 的合成环境细节**（来源：[arXiv 2507.20534](https://arxiv.org/abs/2507.20534) HTML 全文）：
> - 三阶段 pipeline：工具规范生成（3000+ 真实 MCP 工具 + LLM 合成工具）→ Agent 和任务生成 → 轨迹生成
> - "domain evolution"：从关键类别（金融交易、软件应用、机器人控制）出发，演化出具体域
> - 生成 20,000+ 合成工具
> - LLM 生成的用户 persona（不同沟通风格和偏好）进行多轮对话
> - **tool simulator** 维护状态：每次执行后更新环境状态（不像 MiniMind 的无状态 lambda）
> - LLM-as-judge 评估每条轨迹，只保留满足成功标准的

**Agent World Model**（来源：[arXiv 2602.10090](https://arxiv.org/abs/2602.10090)）：
> - 1000 个覆盖日常场景的合成环境
> - **code-driven + database-backed**：比 LLM 模拟的环境有更可靠一致的状态转移
> - 完全可执行的环境 + 可访问的数据库状态 → 可以设计可靠的 reward 函数
> - 仅在合成环境上训练，就能获得强 OOD 泛化

### 路线 2：真实沙箱执行（代码类任务的标配）

**原理**：在隔离容器/VM 中执行真实代码，用单元测试等客观指标做 reward。

**这条路线主要用于 SWE/Coding Agent 训练**——因为代码有天然的 verifiable reward（测试通过/失败）。

#### 隔离技术对比

| 技术 | 启动时间 | 隔离级别 | 代表项目 |
|------|---------|---------|---------|
| **Docker 容器** | 27-90ms | 共享宿主内核 | SWE-bench 标准 harness、DeepSWE（R2E-Gym） |
| **Firecracker microVM** | ~150ms | 独立 Linux 内核 | E2B（专为 AI agent 设计） |
| **gVisor 用户态内核** | 亚秒 | 拦截系统调用 | Modal |
| **mount namespace + chroot** | 极低 | 内核级文件隔离 | SWE-MiniSandbox（存储降到 Docker 的 5%，时间降到 25%） |
| **LLM 模拟执行** | 0 | 无真实执行 | SWE-World（完全 Docker-free） |

#### Kimi K2 的真实沙箱（来源：K2 技术报告）：
> - "built on a robust sandbox infrastructure, powered by **Kubernetes** for scalability and security"
> - 支持 **10,000+ 并发沙箱实例**
> - 软件工程任务：从 GitHub 拉 PR/issue，构建开发环境，用可执行单元测试做 reward
> - 并发 rollout 分摊延迟；partial rollout 允许暂停/恢复长尾任务

#### GLM-5 的真实沙箱（来源：[arXiv 2602.15763](https://arxiv.org/abs/2602.15763)）：
> - 10,000+ 真实软件工程场景（从 GitHub issues 构建）
> - 异步 RL 基础设施：解耦生成与训练，最大化 GPU 利用率
> - 终端环境、搜索任务等 verifiable 场景

#### SWE-MiniSandbox（来源：[arXiv 2602.11210](https://arxiv.org/abs/2602.11210)）：
> - 用 Linux 内核级机制（mount namespaces + chroot）替代 Docker
> - 存储降到容器方案的 ~5%，setup 时间降到 ~25%
> - 性能与 Docker 方案可比

#### "Infrastructure Tax" 论文（来源：[arXiv 2607.01415](https://arxiv.org/abs/2607.01415)）：
> - 对比了 4 种执行基底：单 Docker 容器、托管沙箱、K8s 编排容器、云 VM
> - 冷启动延迟差异达 **110×**
> - 执行 100 万条 150 步轨迹的工时差异达 **1.8×**
> - 结论：沙箱选型应作为训练 pipeline 的战略优化，而非事后运维

### 路线 3：混合路线（合成 + 真实沙箱联合）

**这是 2025-2026 大厂的实际选择**——不是二选一，而是两条路线混合使用。

**Kimi K2 的混合策略**（来源：K2 技术报告）：
> "we complement our simulated environments with real execution sandboxes for scenarios where authenticity is crucial"

```
训练任务类型              执行环境
──────────────────────────────────────
工具调用（天气/汇率/翻译）  → 合成环境（tool simulator 维护状态）
代码执行（编程题/算法）     → 真实沙箱（K8s + 单元测试）
软件工程（SWE-bench）      → 真实沙箱（GitHub 环境 + 测试套件）
数学/逻辑/安全             → 规则验证（Verifiable Rewards Gym）
```

**GLM-5 的混合策略**：
> - 异步 RL 基础设施解耦生成与训练
> - 真实 GitHub issue 环境 + terminal 环境
> - 不同任务类型用不同执行后端

---

## Part 2：开源框架怎么做

### verl（字节，⭐ 22.5k）— 事实标准

来源：[verl 文档](https://verl.readthedocs.io/en/latest/sglang_multiturn/sandbox_fusion.html)、[GitHub](https://github.com/volcengine/verl)

**架构设计**：verl 把工具执行抽象成可插拔的 `AgentLoop`：

```
verl 的工具执行架构:

Actor Rollout (SGLang multi-turn)
  ↓ 模型生成 tool_call
AgentLoop (可插拔)
  ├── ToolAgentLoop（内置 ReAct 实现）
  │    ├── @function_tool    → 无状态函数（类似 MiniMind 的 lambda）
  │    └── BaseTool 子类    → 有状态工具（如代码执行）
  └── 自定义 AgentLoop（继承 AgentLoopBase）
       ↓ 执行工具
  ├── sandbox-fusion（veFaas 代码执行服务）
  │    ├── TokenBucketWorker（Ray 全局 Actor，限流）
  │    ├── ExecutionWorker（异步执行池）
  │    └── 默认超时 30 秒
  ├── 搜索工具（外部检索服务）
  └── 任意自定义工具
       ↓ 结果拼回上下文
继续生成下一轮
```

**sandbox-fusion 细节**：
- 是 verl 的代码执行系统，基于 **veFaas 服务**（字节内部 FaaS）
- Ray 全局 Actor 做分布式限流（默认 10 并发），防 429 错误
- 异步执行 + 线程池，与 SGLang 的 multi-turn 协议集成
- 配置项：`num_workers`（每 DP rank 的工作线程数）、`default_timeout`（30s）、`sandbox_fusion_url`

**verl-agent（GiGPO 官方实现）**：[GitHub](https://github.com/langfengQ/verl-agent)，扩展了 verl 的 agent 训练能力

### agent-lightning（微软，⭐ 17.4k）

来源：[GitHub](https://github.com/microsoft/agent-lightning)

**核心设计**：训练框架与 agent 代码完全解耦

```
agent-lightning 架构:

任意 Agent（零改动）
  ↓ 执行任务
Tracer（捕获 prompt、tool calls、reward 信号）
  ↓ 结构化执行轨迹
LightningStore（中央存储：任务、资源、轨迹）
  ↓
Algorithm Engine（处理轨迹，学习策略）
  ↓
Trainer（编排训练流程，更新推理系统）
```

- 把 agent 执行视为 **MDP**：每步有 state → action → reward
- **不关心你的工具怎么执行**——Docker、API、mock 都行，它只看轨迹
- 微软单独有 **Agent Governance Toolkit**（[GitHub](https://github.com/microsoft/agent-governance-toolkit)）负责执行安全：
  - Microsoft Execution Containers (MXC)：OS 原生进程隔离
  - 覆盖 OWASP Agentic Top 10

### OpenRLHF（社区，⭐ 9.8k）

来源：[GitHub](https://github.com/OpenRLHF/OpenRLHF)

- Ray-based 异步 RL 框架
- 支持 PPO/DAPO/REINFORCE++ 等算法
- 已转型支持 agentic RL（异步 rollout）
- 工具执行依赖外部环境（不内置沙箱）

### AReaL（蚂蚁+清华，⭐ 5.6k）

来源：[GitHub](https://github.com/inclusionAI/AReaL)

- **全异步 rollout 架构**：训练不等最慢轨迹
- 解决的核心问题："agent rollout 串行且贵"
- 上面训出的 ASearcher 支持超过 10 轮的 long-horizon 搜索 agent
- 不改算法，纯靠系统架构压成本

### RAGEN（Northwestern 等，⭐ 2.8k）

来源：[GitHub](https://github.com/RAGEN-AI/RAGEN)

- 研究多轮 RL 训练稳定性
- 分析了 agent 自我演化中的训练崩塌
- 学术框架，适合理解问题

### Agent-R1（中科大，⭐ 1.6k）

来源：[GitHub](https://github.com/0russwest0/Agent-R1)

- 端到端 agent RL
- 代码规模适中，适合进阶阅读

---

## Part 3：代码执行沙箱技术选型（Agent 推理/部署阶段）

训练阶段的沙箱选型看的是吞吐和成本；推理/部署阶段更关注安全和延迟。

### 主流沙箱方案对比

| 方案 | 隔离技术 | 冷启动 | 适用场景 | 特点 |
|------|---------|--------|---------|------|
| **E2B** | Firecracker microVM | ~150ms | AI agent 代码执行 | AI-first SDK，session 限 24h |
| **Modal** | gVisor 用户态内核 | 亚秒 | 推理+训练+batch | GPU 友好，海量自动扩缩 |
| **Docker** | OCI 容器 | 27-90ms | 通用 | 成熟稳定，内核攻击面大 |
| **Daytona** | Docker 默认 | 中等 | 开发环境 | BYOC 支持 |
| **SWE-ReX** | 多后端 | 依赖后端 | SWE-agent | 本地/Docker/AWS/Modal 统一接口 |

来源：[E2B vs Modal 对比](https://northflank.com/blog/e2b-vs-modal)、[AI Agent 沙箱对比 2026](https://amux.io/guides/ai-agent-sandboxing/)

### SWE-ReX（⭐ 555）

来源：[GitHub](https://github.com/SWE-agent/SWE-ReX)

- SWE-agent 团队出品的沙箱执行框架
- 同一份 agent 代码，执行后端可以是本地/Docker/AWS/Modal
- 支持大规模并行，易于扩展
- 最近更新：2026-07-20

---

## Part 4：技术报告摘要（按时间线排列）

### DeepSeek R1 → V3.1 → V4（DeepSeek）

**R1**（2025.01）：
- 用 GRPO 做 math reasoning RL，rule-based reward（答案对/错）
- 不需要工具执行环境——纯推理链，没有工具调用

**V3.1**（2025）：
- 加强了工具使用：1,800+ 环境，85,000 复杂 prompt
- 合成数据驱动 RL

**V4**（[arXiv 2606.19348](https://arxiv.org/abs/2606.19348)，2026.04）：
- 1.6T 总参数，49B 激活
- 后训练采用 **N 个域专家独立 RL** → **On-Policy Distillation (OPD) 合并**
  - 域专家包括：math、competitive coding、agent use、instruction following 等
  - 每个专家有独立 RL loop + 高质量域内数据
  - OPD 合并：全词表 KL 散度加权求和，student 在自身 on-policy rollout 上训练
- **Agent 训练用真实工具环境做 RL**
- 来源：[HuggingFace DeepSeek-V4 博客](https://huggingface.co/blog/deepseekv4)

**DeepSeek Elastic Compute (DSec)**——V4 的执行基础设施：
> - Rust 写的沙箱平台
> - 支持 4 种执行基底：**函数调用、Docker 容器、Firecracker microVM、QEMU 完整 VM**
> - 分层 3FS 存储做快速镜像加载，避免 RL rollout 延迟
> - Preemption-safe trajectory replay：可恢复的训练
> - 统一 API 跨所有基底

**工具调用的 3 个后训练设计决策**：
1. Interleaved thinking：跨工具调用轮次和用户消息边界保持推理链
2. `|DSML|` 特殊 token + XML 格式的 tool-call schema（减少转义失败）
3. 字符串 vs JSON 参数分离（消除解析错误）

### Kimi K1.5 → K2 → K2.5 → K3（Moonshot）

**K2**（[arXiv 2507.20534](https://arxiv.org/abs/2507.20534)，2025.07）：
- 1T 参数 MoE，32B 激活
- 后训练：大规模 agentic 数据合成 + 真实/合成环境联合 RL
- **合成环境**：
  - tool simulator（functionally equivalent to a world model）维护跨轮状态
  - 20,000+ 合成工具 + 3,000+ 真实 MCP 工具
  - LLM 用户 persona（不同沟通风格和偏好）
  - LLM-as-judge 评估轨迹，只保留满足成功标准的
- **真实沙箱**：K8s 编排，10,000+ 并发实例，SWE 任务用单元测试做 reward
- **Verifiable Rewards Gym**：5 个域（数学/指令遵循/忠实性/代码/安全）
- **rollout 优化**：大量并发 rollout 分摊延迟；partial rollout 暂停/恢复长尾任务；重型环境部署为独立可扩展服务

**K2.5**（[arXiv 2602.02276](https://arxiv.org/abs/2602.02276)，2026.02）：
- 原生多模态视觉 + **Agent Swarm Mode**
  - 自导向并行 agent 编排框架
  - 动态分解复杂任务为异构子问题并发执行
  - 比单 agent 延迟降低 4.5×
- Joint text-vision RL
- 训练在 **Cursor 的真实 harness 内**执行——"shadow deployment of the Cursor backend during training so that tool behaviors (like semantic search) work identically to production"
- 统一沙箱架构：消除容器间开销同时保持隔离
- 零停机调度 + K8s 混合云架构

**Kimi-Researcher**（2026）：
- 来源：[moonshotai.github.io/Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/)
- 纯端到端 agentic RL 训练（REINFORCE 算法）
- 3 个工具：内部搜索引擎、文本浏览器、代码执行工具
- HLE 得分从 8.6% → 26.9%（纯 RL，无 SFT）
- Reward：格式分（惩罚无效工具调用）+ 正确性分（与 GT 比对）
- gamma-decay 鼓励更短更高效的探索
- 全自动 pipeline 生成和验证 QA 对，几乎无人工干预

**K3**（2026.07.17 发布，报告预计 07.27 随权重公开）：
- 2.8T 参数，100 万 token 上下文
- Kimi Delta Attention + Stable LatentMoE
- 技术报告尚未公开，agentic RL 细节待确认

### GLM-4.5 → GLM-5 → 5.1 → 5.2（智谱/Z.ai）

**GLM-5**（[arXiv 2602.15763](https://arxiv.org/abs/2602.15763)，2026.02）：
- 异步 RL 基础设施（基于 SLIME 框架）：解耦生成与训练，最大化 GPU 利用率
- 10,000+ 真实 GitHub issue 场景
- 异步 agent RL 算法，支持 long-horizon 交互
- 终端环境 + 搜索任务

**GLM-5.1**（2026 上半年）：
- agentic engineering 旗舰
- SWE-Bench Pro SOTA
- NL2Repo（仓库生成）、Terminal-Bench 2.0（真实终端任务）大幅领先 GLM-5

**GLM-5.2**（2026.06.13）：
- coding-first，100 万 token 上下文
- 两种思考模式（High / Max）
- "最强开源 coding 模型"（2026 年中）
- 在 real bug-fixing 和 long-horizon benchmark 上击败 GPT-5.5，工具使用匹配 Claude Opus 4.8
- 技术报告未单独公开 5.1/5.2 的 agentic RL 细节

### MiniMax M1 → M2.1（MiniMax）

**M1**（[arXiv 2506.13585](https://arxiv.org/abs/2506.13585)，2025.06）：
- 456B 总参数，45.9B 激活，混合注意力 + Lightning Attention
- 提出 **CISPO** 算法（clip importance sampling weights，而非 token updates）
- 512 H800 GPU，3 周完成全部 RL 训练，$534,700

**M2.1 后训练 Agent 模型的经验**（来源：[MiniMax 官方博客](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)）：
- 230B 总参数（MoE），~10B 激活
- **3 种数据合成路径**：
  1. **Real-Data-Driven（SWE Scaling）**：GitHub PR/commit 为原料 → **每个 PR 构建可运行 Docker 环境** → bug 注入/commit 合并/测试编写等增广
  2. **Expert-Driven（AppDev）**：前后端/iOS/Android 领域专家参与 → **Agent-as-a-Verifier**：agent 在沙箱中部署应用，用 Playwright 等工具交互验证 → rubric-based reward
  3. **Synthetic Long-Horizon（WebExplorer）**：agent 自由浏览网页构建信息丰富的种子问题，迭代演化复杂度

**Forge 框架**（来源：[MiniMax 官方博客](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm)）：
- MiniMax 内部 agent RL 训练框架
- 处理过 **10 万+ 种不同的真实 agent scaffold 和环境**
- 每天处理数百万样本，上下文窗口达 200K token
- 三模块架构：Agent Side（抽象白盒/黑盒 agent）→ Middleware（网关 + 分布式数据池，异步收集轨迹）→ Training/Inference Side
- 支持"数百种 scaffold 和数千种不同的工具调用格式"
- 兼容 Sandbox + MCP 环境
- **3 个关键工程方案**：
  1. **Windowed FIFO Scheduling**：滑动窗口（如 batch 的 30%）防止 head-of-line blocking
  2. **Prefix Tree Merging**：利用多轮对话的共享前缀，**40× 训练加速**
  3. **Multi-Token Prediction 投机解码** + 异构 prefill/decode 分离 + 全局 L3 KV cache pool
- **复合 Reward**：过程奖励 + 任务完成时间奖励（含工具执行延迟和并行性激励）+ Reward-to-go（长 horizon 方差缩减）

### Cursor Composer 2（Anysphere）

来源：[philschmid.de](https://www.philschmid.de/kimi-composer-context)

- **Firecracker VM 执行环境**（自建平台 Anyrun）
  - 完整开发环境（含浏览器和 GUI）
  - 500+ pods/秒的调度吞吐
  - 文件系统级 snapshotting：rollout 中途可 checkpoint
- 解耦四服务架构：异步训练（Ray + PyTorch）→ Firecracker VM 环境 → Fireworks AI 推理 → 固定生产后端副本评估
- 每个训练步通过 delta-compressed uploads 到 S3 做权重同步

---

## Part 5：方法演进时间线

```
2024:
  GRPO (DeepSeek-Math) — critic-free 组内比较，math reasoning 成功
  MiniMind MOCK_RESULTS — 无状态 lambda 字典，6 个工具
    = 合成环境路线的最小可运行实现

2025 上半年:
  DeepSeek R1 (2025.01) — GRPO + rule-based reward，纯推理链无工具
  MiniMax M1 (2025.06)  — CISPO 算法，混合注意力，512 H800 三周训完
  Kimi K2 (2025.07)     — 工业级：20K+ 合成工具 + K8s 万级沙箱 + Verifiable Rewards Gym

2026 Q1:
  Kimi K2.5 (2026.02)      — Agent Swarm Mode + 在 Cursor 真实 harness 内训练
  GLM-5 (2026.02)          — 异步 RL 基础设施 + 10K 真实 GitHub issue 场景
  SWE-MiniSandbox (2026.02) — mount namespace + chroot，存储降到 Docker 5%
  SWE-World (2026.02)       — 完全 Docker-free，LLM 模拟执行反馈
  Agent World Model (2026.02) — 1000 个 code-driven + DB-backed 合成环境
  Kimi-Researcher (2026)    — 纯 RL 训出 agentic 能力（HLE 8.6% → 26.9%）

2026 Q2:
  DeepSeek V4 (2026.04)     — DSec 沙箱平台（Rust，4 种执行基底）+ N 域专家独立 RL → OPD 合并
  EnvFactory (2026.05)      — 85 个从真实资源验证的可执行环境
  MiniMax M2.1 (2026)       — Forge 框架（10 万+ scaffold）+ 3 种数据合成路径 + 40× prefix tree 加速
  GLM-5.1 → 5.2 (2026.06)  — SWE-Bench Pro SOTA，100 万 token 上下文

2026 Q3:
  Infrastructure Tax (2026.07) — 4 种执行基底系统对比，冷启动差 110×
  Kimi K3 (2026.07.17)       — 2.8T 参数，技术报告待公开

框架路线:
  verl (字节, 22.5K⭐)     → AgentLoop + sandbox-fusion（veFaas）
  agent-lightning (微软)    → 训练与 agent 完全解耦
  Forge (MiniMax, 内部)    → 10 万+ scaffold，prefix tree 40× 加速
  AReaL (蚂蚁+清华)       → 全异步 rollout
  DSec (DeepSeek, 内部)    → Rust 沙箱，4 种隔离基底统一 API

趋势:
  2025: 合成环境为主，代码任务用 Docker
  2026: 混合路线成主流（合成 + 真实沙箱按任务类型分配）
        Docker 替代方案涌现（microVM / namespace / LLM 模拟）
        异步 rollout 成标配（解耦生成与训练）
        N 域专家独立 RL → 合并 成为新范式（DeepSeek V4）
        训练在生产 harness 内执行（Kimi K2.5 在 Cursor 内训练）
```

---

## Part 6：对 MiniMind 的启示

### MiniMind 的 MOCK_RESULTS 做对了什么

1. **确定性**：固定字典 → 相同输入相同输出，RL 训练稳定
2. **速度**：进程内执行，零网络开销
3. **安全**：`eval()` 加了 `__builtins__: {}` + `signal.alarm(1)` 超时
4. **response_mask 设计**：工具结果 mask=0 不算 loss，这是业界通行做法

### MiniMind 和工业实现的差距

| 维度 | MiniMind | 工业级（以 Kimi K2 / verl 为代表） |
|------|---------|----------------------------------|
| 环境数量 | 6 个工具 | 20,000+ 合成工具 |
| 状态维护 | 无状态 lambda | tool simulator 维护跨轮状态 |
| 环境生成 | 手写字典 | 程序化生成 + LLM 合成 |
| 执行隔离 | 受限 eval | Kubernetes / Docker / microVM / gVisor |
| 质量控制 | 无 | LLM-as-judge 筛选轨迹 |
| 并发规模 | 串行 | 10,000+ 并发实例 |
| 真实环境 | 无 | 合成 + 真实沙箱混合 |

### 如果要升级 MiniMind 的执行环境

```
Level 1（最小改动）: 给 MOCK_RESULTS 加状态
  → 每个工具维护一个 state dict，跨轮更新
  → 例：translate_text 记住上次翻译的内容，get_weather 可以"上次查的城市"

Level 2（接 Docker 沙箱）: calculate_math 改用真实执行
  → 启动一个 Docker 容器，把表达式发进去 eval
  → 用 subprocess + timeout 替代 signal.alarm

Level 3（接 verl 框架）: 换用 verl 的 AgentLoop
  → 复用 sandbox-fusion 或自定义工具
  → 获得异步 rollout + 分布式训练
```

---

## 面试话术

> "MiniMind 用 mock 字典模拟工具执行，这其实是业界合成环境路线的最小实现。
> 大厂的真实做法是**混合路线**——按任务类型选不同执行环境：
>
> **合成环境处理通用工具调用**：Kimi K2 用 tool simulator 维护跨轮状态，
> 生成了 20,000+ 合成工具 + 3,000+ 真实 MCP 工具，
> LLM 用户 persona 做多轮对话，LLM-as-judge 筛选轨迹。
> MiniMax 的 Forge 框架更激进，处理过 10 万+ 种不同的真实 agent scaffold。
>
> **真实沙箱处理代码执行**：Kimi K2 用 Kubernetes 编排 10,000+ 并发沙箱实例，
> 用单元测试做 verifiable reward。DeepSeek V4 更系统化——
> 自建了 DSec 沙箱平台（Rust 写的），统一支持函数调用、Docker 容器、
> Firecracker microVM、QEMU VM 四种执行基底。
> Cursor 用 Firecracker VM 跑完整开发环境，500+ pods/秒的调度吞吐。
>
> **Verifiable reward 处理数学/逻辑**：Kimi 的 Verifiable Rewards Gym 覆盖数学、
> 指令遵循、忠实性、代码、安全五个域，binary reward（对/错）。
>
> 训练时不连真实 API 的根本原因：确定性、速度、安全。
> 一个有意思的趋势是 Kimi K2.5 在 Cursor 的**真实 harness 内**训练——
> shadow deployment 生产后端，让训练时的工具行为和生产完全一致，
> 直接消除 sim-to-real gap。
>
> DeepSeek V4 的后训练还有一个新范式：N 个域专家（math、coding、agent 等）
> 独立做 RL，最后用 On-Policy Distillation 合并成一个统一模型。
> 这比一个模型在所有域上做联合 RL 更稳定——每个专家的 reward 信号不会互相干扰。"

---

## 参考来源

### 技术报告（按时间排序）
- 📄 [DeepSeek R1](https://arxiv.org/abs/2501.12948) — GRPO + rule-based reward，纯推理链
- 📄 [MiniMax M1](https://arxiv.org/abs/2506.13585) — CISPO 算法，混合注意力推理模型
- 📄 [Kimi K2 技术报告](https://arxiv.org/abs/2507.20534) — 合成环境 + 真实沙箱混合 RL，最详细的 agentic RL 工程披露
- 📄 [Kimi K2.5 技术报告](https://arxiv.org/abs/2602.02276) — Agent Swarm Mode + 视觉 agentic
- 📄 [GLM-5 技术报告](https://arxiv.org/abs/2602.15763) — 异步 RL 基础设施 + 10K 真实场景
- 📄 [DeepSeek-V4 技术报告](https://arxiv.org/abs/2606.19348) — 架构创新（CSA/HCA/mHC），agentic 细节见 [HF 博客](https://huggingface.co/blog/deepseekv4)
- 📄 Kimi K3 — 2026.07.17 发布，技术报告预计 07.27 随权重公开

### 博客/官方披露
- 📄 [MiniMax M2.1 Agent 后训练经验](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models) — 3 种数据合成路径 + Docker 环境 + Agent-as-a-Verifier
- 📄 [MiniMax Forge 框架](https://www.minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm) — 10 万+ scaffold，Prefix Tree 40× 加速
- 📄 [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/) — 纯端到端 agentic RL（HLE 8.6%→26.9%）
- 📄 [Kimi/Cursor/Chroma RL 训练对比](https://www.philschmid.de/kimi-composer-context) — Cursor Firecracker VM 细节 + Kimi 在 Cursor harness 内训练
- 📄 [HuggingFace DeepSeek-V4 博客](https://huggingface.co/blog/deepseekv4) — DSec 沙箱平台（Rust，4 种执行基底）

### 论文
- 📄 [Infrastructure Tax in Coding-Agent RL](https://arxiv.org/abs/2607.01415) — 4 种沙箱基底对比，冷启动差 110×
- 📄 [Agent World Model](https://arxiv.org/abs/2602.10090) — 1000 个 code-driven + DB-backed 合成环境
- 📄 [EnvFactory](https://arxiv.org/abs/2605.18703) — 从真实资源合成可执行环境
- 📄 [SWE-MiniSandbox](https://arxiv.org/abs/2602.11210) — container-free RL（存储降到 Docker 5%）
- 📄 [SWE-World](https://arxiv.org/abs/2602.03419) — 完全 Docker-free 的 SWE agent 训练
- 📄 [SENTINEL](https://arxiv.org/abs/2606.12908) — failure-driven RL for tool agents
- 📄 [Synthesize and Reward](https://arxiv.org/abs/2606.03892) — 合成 + live 环境联合 RL
- 📄 [SEED](https://arxiv.org/abs/2607.14777) — self-evolving on-policy distillation for agentic RL

### 开源框架（⭐ 数为 2026-07-21 ✅ gh repo view 实测）
- ✅ [verl](https://github.com/volcengine/verl) — ⭐ 22,583 — 字节，RL 后训练标准框架
- ✅ [agent-lightning](https://github.com/microsoft/agent-lightning) — ⭐ 17,406 — 微软，训练与 agent 解耦
- ✅ [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — ⭐ 9,830 — 社区，Ray-based agentic RL
- ✅ [AReaL](https://github.com/inclusionAI/AReaL) — ⭐ 5,579 — 蚂蚁+清华，全异步 rollout
- ✅ [RAGEN](https://github.com/RAGEN-AI/RAGEN) — ⭐ 2,753 — Northwestern，多轮 RL 稳定性
- ✅ [Agent-R1](https://github.com/0russwest0/Agent-R1) — ⭐ 1,557 — 中科大，端到端 agent RL
- ✅ [SWE-ReX](https://github.com/SWE-agent/SWE-ReX) — ⭐ 555 — SWE-agent 团队，多后端沙箱执行
- ✅ [Awesome Agent RL](https://github.com/0russwest0/Awesome-Agent-RL) — ⭐ 511 — 论文集合
- ✅ [SWE-MiniSandbox](https://github.com/lblankl/SWE-MiniSandbox) — ⭐ 72 — container-free RL

### 沙箱方案对比
- 📄 [E2B vs Modal 对比](https://northflank.com/blog/e2b-vs-modal)
- 📄 [AI Agent 沙箱技术对比 2026](https://amux.io/guides/ai-agent-sandboxing/)
- 📄 [Best Code Execution Sandboxes 2026](https://modal.com/resources/best-code-execution-sandboxes-ai-agents)

### 未覆盖 + 原因
- Kimi K3 技术报告：尚未发布（预计 2026.07.27）
- GLM-5.1/5.2 独立技术报告：未发布，增量更新无独立论文
- Anthropic/OpenAI 的 agent 训练细节：闭源未公开
- DeepSeek V4 完整 PDF 中的 agentic RL 章节：HF 博客已提取核心信息，PDF 全文未逐节读取

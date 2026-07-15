<div align="center">

<img src="images/logo2.png" alt="MiniMind From Scratch" width="200">

# MiniMind From Scratch

### 11,000 行图文笔记 · 逐行手写 · 论文溯源 · 业界对比 · 面试话术

**别再"跑通就行"了。这次，搞懂每一行。**

基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（52.9k+ Star）的 64M 参数大模型，拆成 20+ 章从零手写复现。

[![GitHub stars](https://img.shields.io/github/stars/Xiaxia1997/minimind-from-scratch?style=social)](https://github.com/Xiaxia1997/minimind-from-scratch/stargazers)
[![License](https://img.shields.io/github/license/Xiaxia1997/minimind-from-scratch)](LICENSE)

[**快速开始**](#5-分钟尝鲜) · [**笔记目录**](#笔记目录) · [**和原版的区别**](#这个仓库和原版-minimind-有什么不同)

</div>

---

## 这个仓库和原版 MiniMind 有什么不同？

原版 MiniMind 是一个优秀的可运行项目。本仓库不是替代它，而是在它的基础上做了一件事：**把"能跑"变成"能讲"**。

| | 原版 MiniMind | 本仓库 |
|---|---|---|
| **目标** | 2 小时跑通训练流程 | 逐行理解每一行代码 |
| **代码** | 完整可运行的模型 | 每个组件从零手写，逐步拼装 |
| **讲解深度** | README 概览 | **20+ 篇深度笔记，11,000+ 行图文** |
| **论文溯源** | 未标注来源 | 每个设计追溯到原始论文，区分"实验发现"和"理论推导" |
| **业界对比** | 单一模型 | 每个组件对比 Qwen、DeepSeek、GLM 的实际参数 |
| **面试价值** | 无 | 每章末尾 30 秒面试话术，体现"我动手做过" |

## 谁适合读？

**准备面试的算法工程师** — 需要讲清楚 "Attention 为什么要多头"、"GQA 怎么来的"、"DPO 的 loss 怎么推的"

**想真正理解 Transformer 的人** — 不满足于调库，想知道每一行代码在做什么

**想自己改模型的研究者** — 理解了原理才能改得动

## 5 分钟尝鲜

随便点开一章感受一下笔记风格：

> **[04-Attention](notes/04-Attention.md)**（953 行，本仓库最长的一章）
>
> 从最简单的 QKV 点积写起，经过 9 个 Step 逐步演化到完整的 GQA + KV Cache + Flash Attention 分支，每一步都有可运行的代码和维度标注。

或者看看模型的数据流全景（来自 [00-学习路线](notes/00-学习路线.md)）：

```
token_ids  [batch, seq_len]
    ↓
① Embedding                         → [batch, seq_len, hidden_size]
    ↓
 ┌───────────────── × N 层 ─────────────────┐
 │  x ─────────────────────┐ (残差连接)      │
 │  ↓                      │                 │
 │  RMSNorm → Attention    │                 │
 │    ├─ RoPE (位置编码)    │                 │
 │    ├─ GQA (分组查询)     │                 │
 │    └─ KV Cache (推理)    │                 │
 │  ↓                      │                 │
 │  + ←────────────────────┘                 │
 │  ↓                                        │
 │  x'─────────────────────┐ (残差连接)      │
 │  ↓                      │                 │
 │  RMSNorm → FFN/MoE      │                 │
 │  ↓                      │                 │
 │  + ←────────────────────┘                 │
 └────────────────────────────────────────────┘
    ↓
RMSNorm → LM Head          → [batch, seq_len, vocab_size] logits
```

## 笔记目录

### 第一部分：模型架构（从零拼装一个完整 LLM）

从 Embedding 到完整模型，每一章写一个组件，最终拼成可运行的 `scratch/model/model_minimind.py`。

| 章节 | 组件 | 核心内容 | 行数 |
|------|------|---------|------|
| [01](notes/01-Embedding.md) | **Embedding** | token → 向量，LM Head 权重共享的数学原理 | 234 |
| [02](notes/02-RMSNorm.md) | **RMSNorm** | 为什么不用 LayerNorm，Pre-Norm vs Post-Norm | 228 |
| [03](notes/03-RoPE.md) | **RoPE** | 从二维旋转推导到高维，YaRN 外推扩展 | 720 |
| [03b](notes/03b-线性代数补充.md) | 数学补充 | 旋转矩阵与 Givens 分解（RoPE 前置知识） | 335 |
| [04](notes/04-Attention.md) | **Attention** | 9 个 Step 从 QKV 点积演化到 GQA + KV Cache | 953 |
| [04b](notes/04b-Attention变体演进.md) | Attention 变体 | MHA → MQA → GQA → MLA 演化链 | 458 |
| [05](notes/05-FeedForward.md) | **FeedForward** | sigmoid → ReLU → SiLU → SwiGLU 演化（配图） | 224 |
| [06](notes/06-MOEFeedForward.md) | **MoE** | TopK 专家路由 + 辅助 loss + 负载均衡 | 785 |
| [07](notes/07-MiniMindBlock.md) | **MiniMindBlock** | 组装 Attention + FFN + 残差 + Pre-Norm | 192 |
| [08](notes/08-FullModel.md) | **完整模型** | Config + 所有组件组装，从 token 到 logits | 632 |

### 第二部分：训练全流程（从预训练到 RLHF）

| 章节 | 主题 | 核心内容 | 行数 |
|------|------|---------|------|
| [09](notes/09-训练基础.md) | **训练基础** | 交叉熵 loss、SGD → Adam → AdamW、训练循环 | 775 |
| [09b](notes/09b-ML数学基础.md) | 数学补充 | 激活函数、损失函数的数学推导 | 806 |
| [10](notes/10-训练阶段论.md) | **训练阶段** | 预训练 → CPT → SFT → RLHF 全景 | 310 |
| [11](notes/11-预训练.md) | **预训练** | tokenizer、BPE、数据加载、超参数 | 335 |
| [12](notes/12-SFT.md) | **SFT** | assistant-only loss mask、LoRA 低秩分解 | 721 |
| [13](notes/13-DPO.md) | **DPO** | reference model + chosen/rejected log-ratio | 357 |
| [14](notes/14-GRPO.md) | **GRPO** | Policy Gradient → GRPO、组内 advantage、KL 惩罚 | 878 |
| [15](notes/15-AgentRL.md) | **Agent RL** | 多轮工具调用的 reward 设计 | 463 |

### 第三部分：训练工程（面试高频考点）

| 章节 | 主题 | 核心内容 | 行数 |
|------|------|---------|------|
| [16](notes/16-分布式训练全景.md) | **分布式训练** | DDP + DeepSpeed + FSDP | 188 |
| [17](notes/17-FlashAttention.md) | **Flash Attention** | 分块计算，O(S) 内存 | 138 |
| [18](notes/18-GradientCheckpointing.md) | **梯度检查点** | 时间换空间 | 134 |
| [19](notes/19-训练工程知识图谱.md) | 知识图谱 | 训练工程全景串联索引 | 57 |
| [20](notes/20-长上下文技术.md) | **长上下文** | RoPE 外推 + YaRN + 动态 NTK | 608 |

## 笔记有什么不一样？

### 追溯论文原文，区分"实验发现"和"理论推导"

很多教程把后人的解读当成设计动机来讲。本笔记严格标注每个设计选择的来源。

例如 MHA 的动机：
- **后人解读**："把 768 维拆成 8x96 是为了子空间分离"
- **论文原话**："we found it beneficial to linearly project the queries, keys and values h times"（实验发现，不是理论推导）

### 业界模型对比，具体到参数级别

每个组件都对比 MiniMind、Qwen2.5-7B、DeepSeek-V2、GLM-4 的实际配置：

| 特性 | MiniMind (64M) | Qwen2.5-7B | DeepSeek-V2 (236B) |
|------|---------------|------------|-------------------|
| Q 头数 | 8 | 28 | 128 |
| KV 头数 | 4 (GQA) | 4 (GQA) | 1 (MQA/MLA) |
| QK Norm | 有 | 有 | 有 |

### 每一层都标注维度

不说"经过线性变换"，而是说 `[batch=2, seq=128, dim=768]` 经过 `W_q [768, 768]` 变成 `[2, 128, 8, 96]` 再 transpose 成 `[2, 8, 128, 96]`。

### 配图讲解

激活函数演化、SwiGLU 门控机制等都有专门绘制的图解：

<div align="center">
<img src="notes/step3_swiglu.png" alt="SwiGLU 门控机制" width="500">
</div>

### 面试话术

每章末尾提供 30 秒面试话术，体现"我动手做过"：

> "Attention 我是从最简的 QKV 点积开始写的——先理解 MHA，再改成 GQA 省 KV Cache。GQA 就是让 W_k/W_v 的投影矩阵只有 W_q 的一半大小，8 个 Q 头共享 4 个 KV 头..."

## 仓库结构

```
minimind-from-scratch/
├── notes/                  # 20+ 篇深度笔记（本仓库核心）
│   ├── 00-学习路线.md       #   数据流全景 + 学习顺序
│   ├── 01~08               #   模型架构篇（Embedding → 完整模型）
│   ├── 09~15               #   训练全流程（预训练 → Agent RL）
│   └── 16~20               #   训练工程篇（分布式/FlashAttn/长上下文）
├── scratch/                # 手写复现代码（逐组件拼装）
│   └── model/
│       └── model_minimind.py  # 276 行，从零手写的完整模型
├── model/                  # 原版 MiniMind 模型代码（只读参考）
├── trainer/                # 原版训练脚本（只读参考）
└── images/                 # 图片资源
```

## 更新计划

- [x] 模型架构篇 01-08（已完成）
- [x] 训练全流程笔记 09-15（已完成）
- [x] 训练工程篇 16-20（已完成）
- [ ] 手写复现完整模型组装（scratch/model）
- [ ] 在 RTX 4060 上实际训练 + 训练日志
- [ ] 手写训练脚本（scratch/trainer）

## Star History

如果这个仓库对你有帮助，请点一个 Star，这是我持续更新的最大动力。

[![Star History Chart](https://api.star-history.com/svg?repos=Xiaxia1997/minimind-from-scratch&type=Date)](https://star-history.com/#Xiaxia1997/minimind-from-scratch&Date)

## 致谢

- [jingyaogong/minimind](https://github.com/jingyaogong/minimind) — 本项目基于此仓库，原版代码保留在 `model/`、`trainer/` 目录下供参考
- 原版采用 Apache 2.0 协议开源

## License

本项目继承原版 [Apache 2.0](LICENSE) 协议。笔记和手写复现代码为原创内容。

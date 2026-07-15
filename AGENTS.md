# MiniMind 逐步复现教程

## 项目定位

基于 [MiniMind](https://github.com/jingyaogong/minimind)（52.9k star），从零逐步复现 64M 参数大模型的完整代码。
目标：每个组件手写一遍，理解到能在面试中讲清楚"我是怎么实现的"。

## 工作流

1. **教程笔记** → `notes/` 目录，按组件编号（01-Embedding、02-RMSNorm、03-RoPE...）
2. **手写复现** → `scratch/` 目录，逐步往文件里追加代码，最终拼成完整可运行的模型
3. **原版参考** → `model/model_minimind.py` 和 `trainer/` 是上游原版代码，只读参考不改

## 复现进度与文件映射

### scratch/model/model_minimind.py（逐步追加）

| 步骤 | 组件 | 对应笔记 | 状态 |
|------|------|---------|------|
| 1 | Embedding + LM Head（权重共享） | notes/01-Embedding.md | done |
| 2 | RMSNorm | notes/02-RMSNorm.md | done |
| 3 | RoPE（precompute_freqs_cis + apply_rotary_pos_emb） | notes/03-RoPE.md | done |
| 4 | Attention（GQA + KV Cache + Flash Attention） | notes/04-Attention.md | 进行中 |
| 5 | FeedForward（SwiGLU） | notes/05-FeedForward.md | done |
| 6 | MOEFeedForward | notes/06-MOEFeedForward.md | done |
| 7 | MiniMindBlock（Attention + FFN + 残差 + Pre-Norm） | notes/07-MiniMindBlock.md | |
| 8 | 完整模型组装 | notes/08-FullModel.md | |

### scratch/trainer/（后续）

| 步骤 | 脚本 | 核心学习点 |
|------|------|-----------|
| 9 | train_pretrain.py | 全文 loss、数据分块 |
| 10 | train_full_sft.py | 只对 assistant 回复算 loss |
| 11 | train_lora.py | 低秩矩阵分解 |
| 12 | train_dpo.py | reference model + chosen/rejected log-ratio |
| 13 | train_grpo.py | group sampling、组内 advantage、KL 惩罚 |
| 14 | train_agent.py | 多轮工具调用的 reward 设计 |

## 学习者背景

- AI 算法工程师，求职冲刺期（DDL 2026-08-30）
- 目标岗位：DeepSeek Agent / 阿里千问 / 腾讯混元
- 理论有基础，缺训练实操，学习要高效、面试导向

## 环境

- MacBook Air M5 16GB — 读代码、写笔记、小实验（MPS）
- Windows 台式机 RTX 4060 8GB — 实际训练

## 讲解规则

### 先讲动机再讲方案
- 从"要解决什么问题"出发，先明确目标，再推导方案
- 设计选择必须有依据，不确定的不编

### 代码必须标明写到哪个文件
- 每段代码必须注明目标文件路径（如 `scratch/model/model_minimind.py`）
- 说清楚是"追加到文件末尾"还是"替换某段代码"

### 细致入微
- 每一层的输入输出维度用具体形状标注：`[batch, seq_len, 768]`
- 用 MiniMind 实际参数（768, 8 heads, 96 head_dim）举例

### 业界对比
- 每个组件对比 Qwen、DeepSeek、GLM 怎么做，具体到参数级别

### 面试话术
- 每个章节末尾写 30 秒面试话术，体现"我动手做过"

### 格式
- 写md文档数学公式用 LaTeX（`$...$`、`$$...$$`，对答不要写latex避免渲染问题
- 代码用代码块，标注语言
- 架构图/数据流图用普通代码块（等宽对齐）

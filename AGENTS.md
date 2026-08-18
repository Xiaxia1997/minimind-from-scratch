# AGENTS.md — MiniMind From Scratch

> 面向 AI 编码助手（Claude Code / Kimi Code 等）的项目说明。读完本文件即可理解仓库结构、工作方式与约定。

## 项目概述

本仓库是 **MiniMind 的逐行手写复现教程**（面试导向的学习项目，非生产代码）：

- 基于上游 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（52.9k+ Star）的 64M 参数 decoder-only 大模型
- 把模型拆成 20+ 章中文深度笔记（`notes/`，约 14,600 行图文），并逐组件手写复现到 `scratch/`
- 目标：每个组件手写一遍，理解到能在面试中讲清楚"我是怎么实现的"
- 模型配置：hidden_size=768、8 层、8 Q 头 / 4 KV 头（GQA）、head_dim=96、vocab=6400、SwiGLU（intermediate_size=2432）、RoPE（theta=1e6）、RMSNorm、权重共享（tie_word_embeddings），可选 MoE（4 专家 top-1）
- License：Apache 2.0（继承上游，笔记与手写代码为原创内容）

**Git 远端**：`origin` = 上游 jingyaogong/minimind；`myscratch` = 作者 fork（Xiaxia1997/minimind-from-scratch）。

## 仓库结构与模块划分

```
├── notes/                  # 核心产出：20+ 篇中文笔记，按组件编号
│   ├── 00-学习路线.md       #   数据流全景 + 章节索引 + 面试策略（先读这个）
│   ├── 01~08               #   模型架构篇（Embedding → RMSNorm → RoPE → Attention → FFN → MoE → Block → 完整模型）
│   ├── 09~16               #   训练篇（训练基础 → 预训练 → SFT → RL对齐/DPO/GRPO/AgentRL）
│   ├── 17~18               #   训练工程篇（分布式、长上下文）
│   ├── 19-前沿模型技术全景.md #   前沿技术全景（1900 行）：DeepSeek V4/GLM-5/Kimi/Qwen3/MiniMax/MiMo 完整对比
│   └── *.png, plot_activations.py  # 笔记配图及绘图脚本
├── scratch/                # 手写复现代码（本仓库主线工作区）
│   └── model/model_minimind.py  # 454 行，按章节注释分块逐步追加，__main__ 块内含自验证
├── model/                  # 上游原版模型代码（model_minimind.py / model_lora.py / tokenizer）——只读参考，不要改
├── trainer/                # 上游原版训练脚本（train_pretrain/full_sft/lora/dpo/grpo/ppo/agent 等）——只读参考，不要改
├── docs-local/             # 技术报告研究笔记（*.md）+ 15 篇原始论文 PDF（papers/*.pdf）
├── dataset/                # lm_dataset.py（Dataset 定义，只读参考）+ 数据集文件存放处（数据本身不入库）
├── scripts/                # 上游推理/部署脚本（web_demo、serve_openai_api、chat_api、convert_model、eval_toolcall）
├── eval_llm.py             # 上游交互式对话评估脚本（从仓库根目录运行）
├── requirements.txt        # 依赖（精确版本）；torch/torchvision/peft/matplotlib 被注释，需按平台单独装
└── images/                 # README 图片资源
```

**关键边界**：`model/`、`trainer/`、`dataset/lm_dataset.py`、`scripts/`、`eval_llm.py` 是上游原版代码，**只读参考、不要修改**。所有手写工作发生在 `notes/`（笔记）和 `scratch/`（复现代码）。

## 技术栈与环境

- 纯 Python 项目，**无 pyproject.toml / setup.py / package.json / 构建系统 / CI**
- 依赖：PyTorch + transformers==4.57.6 + trl==0.13.0 + datasets，训练日志用 swanlab/wandb；完整列表见 `requirements.txt`（torch 等被注释，需按平台单独安装）
- 双机环境：
  - **MacBook Air M5 16GB**（本机）— 读代码、写笔记、小实验（MPS）。本机已建 conda 环境 `minimind`（python 3.13 + torch 2.13.0 + numpy，MPS 可用；在 `~/miniconda3/envs/`，不进仓库）；运行 python 用 `~/miniconda3/envs/minimind/bin/python` 或先 `conda activate minimind`
  - **Windows 台式机 RTX 4060 8GB** — 实际训练（CUDA + bf16/fp16 AMP + DDP）
- 数据集不入库：下载后放 `dataset/` 目录（见 `dataset/dataset.md`），默认路径如 `dataset/pretrain_t2t_mini.jsonl`
- 训练产物（`out/`、`checkpoints/`）被 .gitignore 排除

## 运行命令

```bash
# 1. 手写复现模型的自验证（本机用 conda 环境 minimind，torch 已装）
~/miniconda3/envs/minimind/bin/python scratch/model/model_minimind.py

# 2. 上游训练脚本（只读参考，实际训练在 Windows/4060 上进行）
#    注意：脚本内相对路径为 ../dataset、../out、../checkpoints，须从 trainer/ 目录运行
cd trainer && python train_pretrain.py --data_path ../dataset/pretrain_t2t_mini.jsonl
cd trainer && torchrun --nproc_per_node=N train_pretrain.py   # 多卡 DDP

# 3. 推理对话（上游脚本，从仓库根目录运行，权重在 ./out/）
python eval_llm.py --weight full_sft
```

## 测试与验证策略

- **没有 pytest 等测试框架，也没有 CI**。验证方式是 `scratch/model/model_minimind.py` 末尾 `if __name__ == '__main__':` 块内的 **assert 自验证**：逐组件检查输出形状、KV Cache 行为、参数量，全部打印 "verified!"
- 修改 `scratch/` 代码后，应在有 torch 的环境运行该脚本确认全部 assert 通过
- 笔记中的代码片段应与 `scratch/` 或上游源码保持一致；引用上游行号以 `model/model_minimind.py`、`trainer/*.py` 实际内容为准

## 复现进度（当前状态）

### 架构篇：`scratch/model/model_minimind.py`（逐步追加，可直接运行自验证）

| 步骤 | 组件 | 对应笔记 | scratch 手写 |
|------|------|---------|-------------|
| 1 | Embedding + LM Head（权重共享） | notes/01-Embedding.md | done（在 `__main__` 验证块） |
| 2 | RMSNorm | notes/02-RMSNorm.md | done |
| 3 | RoPE（precompute_freqs_cis + apply_rotary_pos_emb） | notes/03-RoPE.md | done |
| 4 | Attention（GQA + QK Norm + KV Cache + SDPA 分支） | notes/04-Attention.md | done |
| 5 | FeedForward（SwiGLU） | notes/05-FeedForward&激活函数.md | done |
| 6 | MOEFeedForward | notes/06-MOEFeedForward.md | done |
| 7 | MiniMindBlock（Attention + FFN + 残差 + Pre-Norm） | notes/07-MiniMindBlock.md | done |
| 8 | MiniMindConfig + 完整模型组装 | notes/08-FullModel.md | done（含 KV Cache + Generate 自验证） |

### 训练篇：笔记 09–16 全部 done，`scratch/trainer/` 未开始（下一阶段：在 4060 上实跑）

notes/09-训练基础、09a-ML数学基础、10-训练阶段论、11-预训练、12-SFT、13-RL对齐基础、14-DPO、15-GRPO、16-AgentRL。

### 训练工程篇（纯知识补充，无对应 scratch 代码）

notes/17-分布式训练全景、18-GradientCheckpointing、19-长上下文技术、20-DeepSeek-V4架构创新，全部 done。Flash Attention 已合入 04a，训练工程知识图谱已合入 00-学习路线。

> 注意：笔记文件名可能带历史后缀（如 `04a-Attention变体演进-todo.md` 实际已 done），以本表和 `notes/00-学习路线.md` 为准。训练工程篇的业界数字部分来自模型内生知识，笔记中已标注"论文/未公开/推测"，引用前需核对原始出处。

## 代码与文档约定（写笔记 / 写 scratch 代码时必须遵守）

1. **先讲动机再讲方案**：从"要解决什么问题"出发推导；设计选择必须有依据，不确定的不编
2. **代码必须标明目标文件**：每段代码注明写到哪个文件（如 `scratch/model/model_minimind.py`），说清是"追加到文件末尾"还是"替换某段代码"
3. **维度标注细致入微**：每层输入输出用具体形状标注（`[batch, seq_len, 768]`），用 MiniMind 实际参数（768、8 heads、96 head_dim）举例
4. **不常见操作必须加注释**：`.nonzero()`、`index_add_`、`zeros_like`、布尔索引、`.view(-1,1)` 广播等，注释要说清做了什么、形状变化、为什么这么做；链式操作拆开逐步注释
5. **业界对比**：每个组件对比 Qwen、DeepSeek、GLM 的实际参数配置
6. **面试话术**：每章笔记末尾写 30 秒面试话术，体现"我动手做过"
7. **格式**：写 md 文档时数学公式用 LaTeX（`$...$` / `$$...$$`）；直接对话回复时禁止 LaTeX（终端不渲染），用代码块或纯文本；代码块标注语言；架构图用等宽对齐的普通代码块
8. **语言**：笔记、注释、文档一律用中文（与现有内容保持一致）
9. **提纲先行，一个概念一个家**：写长笔记（多 Part）前先输出完整目录 + 每节一句话职责，冻结提纲再逐节写；每个概念只在一个小节完整展开，其他地方只允许一句话引用（"见 X.Y"），禁止在两处重复讲解；承上启下的预告同一内容只写一次
10. **数字必须可溯源**：笔记中的具体数字要么来自可验算的例子（给出算式），要么来自论文/官方资料（标注出处）；解释性的比例、占比、贡献度一律用定性说法（"贡献极小"），禁止编造精确百分比
11. **结构变更后必须全文校验**：移动/拆分/合并小节后，通读全章修正所有交叉引用编号（"见 X.Y"）、检查概念是否重复展开；长笔记交付前用 teach skill 的 Review 模式过一遍（建议在新会话中做，避免写作上下文偏向）

## Git 与协作注意

- 不要主动执行 `git commit` / `push` / `reset` 等变更操作，除非用户明确要求
- `.gitignore` 已排除：`out/`、`docs-local/`（私有文档）、`.agents/`、`.vscode/`、`skills-lock.json`、`CLAUDE.md`、notes 临时图片 —— 这些不要提交
- `docs-local/` 是本地私有材料（B站视频 PPT、推广计划），不要读取传播
- 上游参考代码（`model/`、`trainer/`、`scripts/`、`dataset/lm_dataset.py`、`eval_llm.py`）只读不改；如发现上游 bug，在笔记中说明而不是改源码
- 仓库无密钥类敏感文件；tokenizer 文件（`model/tokenizer*.json`）是正常模型资产

## 学习者背景（帮助理解项目取向）

- 用户是 AI 算法工程师，求职冲刺期（DDL 2026-08-30），目标岗位：DeepSeek Agent / 阿里千问 / 腾讯混元
- 理论有基础、缺训练实操，学习和讲解要高效、面试导向，不要给"跑通就行"式的浅层解释

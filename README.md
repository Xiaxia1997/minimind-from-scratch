<div align="center">

# 🔬 MiniMind From Scratch

**逐组件手写复现 + 深度图解笔记 + 论文原文追溯 + 业界模型对比**

基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（52.9k ⭐）

[![GitHub stars](https://img.shields.io/github/stars/Xiaxia1997/minimind-from-scratch?style=social)](https://github.com/Xiaxia1997/minimind-from-scratch)
[![License](https://img.shields.io/github/license/Xiaxia1997/minimind-from-scratch)](LICENSE)

</div>

---

## 这个仓库和原版 MiniMind 有什么不同？

| | 原版 MiniMind | 本仓库 |
|---|---|---|
| **定位** | 跑通训练流程 | 逐行理解每一行代码 |
| **代码** | 完整可运行的模型 | 每个组件从零手写，逐步拼装 |
| **讲解** | README 概览 | 18 篇深度笔记，4000+ 行图文 |
| **论文** | 未标注来源 | 每个设计追溯到原始论文，标注 "we found it beneficial" 还是理论推导 |
| **对比** | 单一模型 | 每个组件对比 Qwen、DeepSeek、GLM 的实际参数 |
| **面试** | 无 | 每章末尾 30 秒面试话术 |

## 谁适合读？

- 🎯 **准备面试的算法工程师**：需要讲清楚 "Attention 为什么要多头"、"GQA 怎么来的"、"DPO 的 loss 怎么推的"
- 📖 **想真正理解 Transformer 的人**：不满足于调库，想知道每一行代码在做什么
- 🔧 **想自己改模型的研究者**：理解了原理才能改得动

## 笔记目录

### 模型架构篇（从零拼装一个完整模型）

| 编号 | 组件 | 核心内容 | 亮点 |
|------|------|---------|------|
| [01](notes/01-Embedding.md) | Embedding | token → 向量，权重共享 | LM Head 权重共享的数学原理 |
| [02](notes/02-RMSNorm.md) | RMSNorm | 均方根归一化 | 为什么不用 LayerNorm，Pre-Norm vs Post-Norm |
| [03](notes/03-RoPE.md) | RoPE | 旋转位置编码 | 从二维旋转推导到高维，YaRN 扩展 |
| [03b](notes/03b-线性代数-旋转矩阵与Givens分解.md) | 数学基础 | 旋转矩阵与 Givens 分解 | RoPE 的线性代数前置知识 |
| [04](notes/04-Attention.md) | Attention | GQA + KV Cache + Flash Attention | MHA→MQA→GQA 演化链（附论文原文），QK Norm |
| [05](notes/05-FeedForward.md) | FeedForward | SwiGLU 门控机制 | sigmoid→ReLU→SiLU→SwiGLU 演化，配图解 |
| [06](notes/06-MOEFeedForward.md) | MoE | TopK 专家路由 + 负载均衡 | 辅助 loss 的作用 |
| [07](notes/07-MiniMindBlock.md) | MiniMindBlock | 组装 Attention + FFN + 残差 | Pre-Norm 的数据流 |
| [08](notes/08-FullModel.md) | 完整模型 | 组装所有组件 | 从 token 到 logits 的完整前向传播 |

### 训练工程篇

| 编号 | 主题 | 核心内容 |
|------|------|---------|
| [10](notes/10-混合精度训练.md) | 混合精度训练 | FP16/BF16 + loss scaling |
| [11](notes/11-多轮对话Loss计算.md) | 多轮对话 Loss | 只对 assistant 回复算 loss |
| [12](notes/12-梯度累积.md) | 梯度累积 | 小显存模拟大 batch |
| [13](notes/13-分布式训练全景.md) | 分布式训练 | DDP + DeepSpeed |
| [14](notes/14-FlashAttention.md) | Flash Attention | 分块计算，O(S) 内存 |
| [15](notes/15-GradientCheckpointing.md) | 梯度检查点 | 时间换空间 |
| [16](notes/16-学习率调度.md) | 学习率调度 | warmup + cosine decay |
| [17](notes/17-训练工程知识图谱.md) | 知识图谱 | 训练工程全景串联 |
| [18](notes/18-长上下文技术.md) | 长上下文 | RoPE 外推 + YaRN |

## 仓库结构

```
minimind-from-scratch/
├── notes/                  # 📖 18 篇深度笔记（本仓库核心产出）
│   ├── 00-学习路线.md
│   ├── 01-Embedding.md
│   ├── ...
│   └── 18-长上下文技术.md
├── scratch/                # ✍️ 手写复现代码（逐组件从零拼装）
│   └── model/
│       └── model_minimind.py
├── model/                  # 📦 原版 MiniMind 模型代码（只读参考）
├── trainer/                # 📦 原版训练脚本（只读参考）
├── scripts/                # 📦 原版工具脚本
├── dataset/                # 📦 数据集说明
└── images/                 # 📦 原版图片资源
```

## 笔记特色

### 📐 追溯论文原文，区分"实验发现"和"理论推导"

> 很多教程把后人的解读当成设计动机来讲。本笔记严格标注每个设计选择的来源——论文原话是 "we found it beneficial"（实验发现）还是数学推导。

例如 MHA 的动机：
- ❌ "把 768 维拆成 8×96 是为了子空间分离"（后人解读）
- ✅ 论文原话："we found it beneficial to linearly project the queries, keys and values h times"（实验发现）

### 📊 业界模型对比，具体到参数级别

每个组件都对比 MiniMind、Qwen2.5-7B、DeepSeek-V2、GLM-4 的实际配置：

| 特性 | MiniMind (64M) | Qwen2.5-7B | DeepSeek-V2 (236B) |
|------|---------------|------------|-------------------|
| Q 头数 | 8 | 28 | 128 |
| KV 头数 | 4 (GQA) | 4 (GQA) | 1 (MQA/MLA) |
| QK Norm | 有 | 有 | 有 |

### 🎨 配图讲解，不画不写

激活函数演化、SwiGLU 门控机制等都有专门绘制的图解：

![SwiGLU 门控机制](notes/step3_swiglu.png)

### 🎤 面试话术

每章末尾提供 30 秒面试话术，体现"我动手做过"的信号：

> "Attention 我是从最简的 QKV 点积开始写的——先理解 MHA，再改成 GQA 省 KV Cache。GQA 就是让 W_k/W_v 的投影矩阵只有 W_q 的一半大小，8 个 Q 头共享 4 个 KV 头..."

## 致谢

- [jingyaogong/minimind](https://github.com/jingyaogong/minimind) — 本项目基于此仓库，原版代码保留在 `model/`、`trainer/` 目录下供参考
- 原版采用 Apache 2.0 协议开源

## License

本项目继承原版 [Apache 2.0](LICENSE) 协议。笔记和手写复现代码为原创内容。

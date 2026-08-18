# 21 - RAG 检索与 Reranker：从 TF-IDF 到 BGE

> 前置：[04-Attention](04-Attention.md)（Self-Attention 机制）、[08-FullModel](08-FullModel.md)（Decoder-only 全模型）
> 本章不涉及 MiniMind 代码——纯知识篇，讲 Encoder-only 架构和检索系统

## 在整个项目中的位置

```
01-08 架构篇: Decoder-only（生成）
  Embedding → RMSNorm → RoPE → Attention → FFN → Block → FullModel
  ↑ 只用了 Transformer 的右半边（Decoder）

09-18 训练篇: 预训练 → SFT → DPO → GRPO → 分布式训练

【21 本章: Encoder-only（理解/检索）+ 检索应用】
  ↑ 只用了 Transformer 的左半边（Encoder）
```

## 一句话总结

01-08 章学的是 Decoder-only（自回归生成）。本章学另一半——Encoder-only（双向理解），以及它在检索场景的两种用法：Bi-Encoder 把句子压成向量做召回，Cross-Encoder 把 query+doc 拼在一起做精排。

## 演进全景

```
                    原始 Transformer（翻译）
                     ↙              ↘
              Encoder-only         Decoder-only
              (本章)                (01-08 已学)
                 ↓
               BERT（MLM + NSP 预训练）           TF-IDF（词频统计）
                 ↓                                    ↓
            RoBERTa（训练改良）                     BM25（非线性改良）
                 ↓                                    ↓
          ┌──────┴──────┐                     学习的词权重模型
          ↓             ↓                    （BERT + CRF + 回归头）
     Bi-Encoder    Cross-Encoder                  ↓
     (Embedding)   (Reranker)               稀疏检索召回
          ↓             ↓
     BGE Embedding  BGE Reranker
     训练流程        训练流程
```

左边是**稠密检索**路线（语义匹配），右边是**稀疏检索**路线（关键词匹配）。生产系统两条路都用。

---

## Part 1: 为什么需要检索

### 1.1 LLM 的两个硬伤

大语言模型的知识来自预训练数据：
1. **知识截止**：训练数据有截止日期，之后发生的事不知道
2. **幻觉**：对不确定的事实也会自信地编答案

RAG（Retrieval-Augmented Generation）的思路：先从外部知识库里**检索**出相关文档片段，再塞进 prompt 让 LLM 基于真实材料回答。

### 1.2 检索的两条路线

```
路线 1: 稀疏检索（关键词匹配）
  "RTX 4060 显存" → 在倒排索引里精确找包含这些关键词的文档
  特点: 速度极快，精确匹配能力强
  问题: "猫" 匹配不到 "喵星人"（没有语义理解）

路线 2: 稠密检索（语义匹配）
  "RTX 4060 显存" → 编码成向量 → 在向量空间里找最近邻
  特点: 能做语义匹配（"猫" ≈ "喵星人"）
  问题: 压缩成向量后关键词精确匹配能力反而变弱
```

两条路互补，生产系统通常两条都用。

### 1.3 三阶段检索链路

```
用户 Query
    ↓
━━ 第 1 阶段: 召回（稀疏 + 稠密，并行）━━━━━━━━━━━━━━━━━━━━
  稀疏: 词权重 → 倒排索引检索        ← Part 2 讲
  稠密: Bi-Encoder → 向量检索        ← Part 5-6 讲
  两路取并集 → Top-K 候选（K ≈ 100-1000）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
━━ 第 2 阶段: 精排（Rerank）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Cross-Encoder → query+doc 拼接送入同一个模型 → 精细打分
  特点: 精度最高，但每对都要在线算    ← Part 7 讲
  输出: Top-N 最终结果（N ≈ 5-10）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
塞进 LLM prompt → 生成回答
```

| 阶段 | 候选数 | 速度 | 精度 | 能预计算？ |
|------|--------|------|------|----------|
| 稀疏召回 | 全库（百万+） | ~1ms | 关键词精确 | 是（倒排索引） |
| 稠密召回 | 全库（百万+） | ~5ms | 语义匹配 | 是（向量索引） |
| Reranker 精排 | 50-100 | ~50ms | 最高 | 否（每对在线算） |

Cross-Encoder 精度最高，但每对 (query, doc) 都要过一遍前向传播。100 万文档 × 10ms = 10000 秒，不可能在线用。所以先用第 1 阶段快速缩小范围，再用精排。

---

## Part 2: 稀疏检索演进——从统计到学习

稀疏检索的核心问题：**给 query/doc 中的每个词打一个重要性权重**。"RTX 4060 显存不够用怎么办"里，"RTX 4060""显存"是关键信息，"不够用""怎么办"是修饰。

### 2.1 倒排索引：稀疏检索的数据结构基础

在讲 TF-IDF 和 BM25 之前，先理解它们的载体——**倒排索引（Inverted Index）**。

正常存储是"文档 → 包含哪些词"（正排），倒排索引反过来：**"词 → 出现在哪些文档里"**。

```
假设有 3 篇文档（已分词）：
  doc1: "猫 喜欢 吃 鱼"
  doc2: "狗 喜欢 吃 骨头"
  doc3: "猫 喜欢 晒太阳"

正排索引（存储原文）:
  doc1 → [猫, 喜欢, 吃, 鱼]
  doc2 → [狗, 喜欢, 吃, 骨头]
  doc3 → [猫, 喜欢, 晒太阳]

倒排索引（检索用）:
  猫     → [(doc1, TF=1), (doc3, TF=1)]
  喜欢   → [(doc1, TF=1), (doc2, TF=1), (doc3, TF=1)]
  吃     → [(doc1, TF=1), (doc2, TF=1)]
  鱼     → [(doc1, TF=1)]
  狗     → [(doc2, TF=1)]
  骨头   → [(doc2, TF=1)]
  晒太阳 → [(doc3, TF=1)]
```

每个词条（term）指向一个**倒排列表（posting list）**，记录该词出现在哪些文档中以及词频。查询时不需要扫描所有文档，直接按词查表——这就是为什么 BM25 能在百万文档上做到毫秒级响应。

中文需要先分词（jieba 等工具），英文按空格和标点分割即可。

### 2.2 TF-IDF：最朴素的词权重

TF-IDF 给每个词的权重 = **词频 × 逆文档频率**。

```
TF（Term Frequency）: 这个词在当前文档中出现了几次
  "感冒" 在文档 A 中出现 5 次 → TF = 5

IDF（Inverse Document Frequency）: 这个词在整个语料库中有多罕见
  语料库 10 万篇文档，"感冒" 出现在 1000 篇 → IDF = log(100000/1000) = 2.0
  "的" 出现在 99000 篇 → IDF = log(100000/99000) = 0.004

TF-IDF = TF × IDF
  "感冒": 5 × 2.0 = 10.0   ← 高权重（在本文档频繁出现 + 全局罕见）
  "的":   8 × 0.004 = 0.032 ← 低权重（虽然出现多次，但全局太普遍）
```

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{\text{DF}(t)}$$

其中 $t$ 是词、$d$ 是文档、$N$ 是语料库文档总数、$\text{DF}(t)$ 是包含词 $t$ 的文档数。

**TF-IDF 的问题**：词频是线性的——"感冒"出现 100 次的文档不比出现 10 次的文档相关 10 倍。实际上出现 5 次以上就饱和了，再多不增加多少信息。

### 2.3 BM25：TF-IDF 的非线性改良

BM25 对 TF-IDF 做了两个改进：① 词频饱和（非线性）② 文档长度归一化。

$$\text{BM25}(t, d) = \text{IDF}(t) \times \frac{\text{TF}(t, d) \times (k_1 + 1)}{\text{TF}(t, d) + k_1 \times \left(1 - b + b \times \frac{|d|}{\text{avgdl}}\right)}$$

公式看着复杂，拆开看只有两个改动：

**改动 1：词频饱和**

分子是 $\text{TF} \times (k_1 + 1)$，分母是 $\text{TF} + k_1 \times (\ldots)$。当 TF 很大时，整个分数趋近于 $k_1 + 1$——不管出现多少次，权重最多就是 $k_1 + 1$。

```
k1 = 1.2（默认值）时:
  TF = 1:   1 × 2.2 / (1 + 1.2) = 1.0
  TF = 5:   5 × 2.2 / (5 + 1.2) = 1.77
  TF = 10:  10 × 2.2 / (10 + 1.2) = 1.96
  TF = 100: 100 × 2.2 / (100 + 1.2) = 2.17  ← 接近上限 2.2

对比 TF-IDF: TF = 100 时权重是 TF = 1 的 100 倍
BM25:        TF = 100 时权重只是 TF = 1 的 2.17 倍
```

**改动 2：文档长度归一化**

分母里的 $1 - b + b \times \frac{|d|}{\text{avgdl}}$：$|d|$ 是当前文档长度，$\text{avgdl}$ 是平均文档长度，$b$ 控制归一化强度（默认 0.75）。

```
长文档（|d| = 2 × avgdl）: 分母变大 → 权重被压低
  "感冒" 在 2000 字的长文档中出现 10 次不稀奇
短文档（|d| = 0.5 × avgdl）: 分母变小 → 权重被抬高
  "感冒" 在 200 字的短文档中出现 10 次说明这篇很相关
```

**BM25 仍然是 2025 年稀疏检索的默认选择**——Elasticsearch、Lucene、Meilisearch 默认都用它。简单、无需训练、效果好。

### 2.4 完整检索流程：倒排索引 + BM25 如何协作

倒排索引负责**快速找到候选**（从百万文档缩小到几百篇），BM25 负责**给候选打分排序**。走一遍完整流程：

```
═══ 离线建索引阶段 ═══

所有文档 → 分词（中文用 jieba，英文按空格）→ 建倒排索引
  每个词条记录: [doc_id, 词频 TF, 文档长度 |d|]
  同时预算好全局统计量: 每个词的 IDF、所有文档的平均长度 avgdl

  IDF 和 avgdl 只依赖语料库整体，文档入库后就能算好
  TF 是每个 (词, 文档) 对的局部统计，也在建索引时记录

═══ 在线查询阶段 ═══

query "猫 吃 什么"
    ↓ 分词
  ["猫", "吃", "什么"]
    ↓ 查倒排索引（毫秒级，只查 3 个词条的 posting list）
  "猫" → [doc1, doc3]
  "吃" → [doc1, doc2]
  "什么" → [doc5, doc8, ...]
    ↓ 合并候选集（取并集）
  {doc1, doc2, doc3, doc5, doc8, ...}
    ↓ 对每个候选文档算 BM25 总分（求和所有 query 词的贡献）
  score(doc1) = BM25(猫, doc1) + BM25(吃, doc1) + BM25(什么, doc1)
  score(doc2) = BM25(猫, doc2) + BM25(吃, doc2) + BM25(什么, doc2)
  ...
    ↓ 按得分排序，返回 Top-K
```

**为什么这么快？** BM25 公式里的所有参数（TF、IDF、$|d|$、avgdl、$k_1$、$b$）在建索引时就已经确定了。在线查询只需要：① 查倒排列表找到候选文档 ② 代入公式算加法——没有任何矩阵运算或模型推理。百万文档上通常 1-5ms 就能返回结果。

### 2.5 BM25 的局限：没有语义，不会学

BM25 的权重完全由统计公式决定，有两个无法解决的问题：

```
问题 1: 无法区分"同一个词在不同语境下的重要性"
  query = "学科技的人"
  文档 A: "学科技改变命运"       ← "学" = 学习（动词），相关
  文档 B: "文学科的录取分数线"    ← "学" = 学科（名词），不相关
  BM25 给两个"学"打一样的分——它不理解上下文

问题 2: 功能词权重过高
  query = "RTX 4060 显存不够用怎么办"
  BM25: "不够用""怎么办" 的 IDF 可能不低，白白匹配到无关文档
  理想: 模型应该学会"不够用""怎么办"在检索中不重要
```

### 2.6 学习的词权重模型：BERT + CRF + 回归头

解决办法：**用 BERT 理解上下文，然后为每个词学一个重要性权重**。

```
输入: [CLS] R T X 4 0 6 0 显 存 不 够 用 ...
                ↓
       ┌──────────────────────┐
       │  BERT (bert-base)     │
       │  12 层 Encoder         │
       │  hidden_size = 768    │
       └──────────────────────┘
                ↓
        sequence_output          ← [batch, seq_len, 768]
           ↙          ↘
  ┌──────────────┐   ┌──────────────────┐
  │ CRF 分类头     │   │  权重回归头         │
  │ Linear(768→4) │   │  Linear(768→1)    │
  │ → CRF 解码     │   │  → Sigmoid        │
  │ → BMES 标注    │   │  → 权重 ∈ [0, 1]   │
  └──────────────┘   └──────────────────┘
       ↓                    ↓
  分词结果              每个词的重要性权重
  "RTX" "4060" "显存"    0.95  0.92  0.88
  "不够用" "怎么办"       0.15  0.05
```

两个头各做一件事：

**CRF 分类头**做中文分词标注（BMES）——B=词首、M=词中、E=词尾、S=单字词。

Linear(768→4) 输出每个位置的 4 个 logits（发射分数），如果直接用 Softmax + argmax，每个位置独立取最大值，可能产生非法序列（如 B 后面直接跟 B）。CRF 在发射分数之外多了一个**可学习的 4×4 转移矩阵**，编码标签间的合法转移：

```
Softmax（逐 token 独立决策）:
  位置 0: argmax → B
  位置 1: argmax → B   ← B 后面跟 B，非法！
  位置 2: argmax → E

CRF（全局最优解码）:
  转移矩阵 T（学出来的）:
         → B    → M    → E    → S
  B →  [ -∞    0.9    0.8   -∞   ]  ← B 后面只能跟 M 或 E
  M →  [ -∞    0.7    0.9   -∞   ]  ← M 后面只能跟 M 或 E
  E →  [ 0.8   -∞     -∞    0.9  ]  ← E 后面只能跟 B 或 S
  S →  [ 0.8   -∞     -∞    0.9  ]  ← S 后面只能跟 B 或 S

  解码时用维特比算法找全局最优路径:
  路径总分 = Σ(发射分数) + Σ(转移分数)
  位置 0 选 B → 位置 1 选 B 的转移分数 = -∞ → 这条路被堵死
  → 最终选 B M E（合法，"三个字组成一个词"）
```

CRF 的 "Conditional Random Field"（条件随机场）：**"条件"**= 给定输入序列（BERT 输出）作为条件；**"随机场"**= 对整条标签序列建模联合概率，而不是对每个标签独立建模。

转移矩阵里的分数不是写死的，是**训练出来的**——训练数据里从来没有 B→B 的标注，梯度会把 B→B 的分数压到很低。

维特比解码的完整例子：

```
输入 "自然语言" 四个字，BERT 给每个位置的发射分数:

         B     M     E     S
位置0(自): 3.0   0.1   0.2   1.5    ← BERT 觉得"自"很可能是 B
位置1(然): 0.3   1.0   2.8   0.5    ← BERT 觉得"然"很可能是 E
位置2(语): 2.5   0.2   0.3   1.0
位置3(言): 0.1   0.8   3.0   0.2

候选路径（总分 = 发射分之和 + 转移分之和）:
  B-E-B-E: 发射 = 3.0+2.8+2.5+3.0=11.3  转移 = T[B→E]+T[E→B]+T[B→E] 全合法
           → 总分高 → "自然" "语言" 两个词 ✓

  B-B-E-S: 发射 = 3.0+0.3+0.3+0.2       转移 = T[B→B]=-100 → 直接淘汰 ✗

维特比算法高效地找到最优路径，不需要穷举全部 4⁴=256 种组合
```

**一句话：Softmax 是逐 token 独立分类，CRF 是对整个序列做全局最优解码。**

**权重回归头**给每个 token 输出一个 $[0, 1]$ 的重要性分数：

$$w_i = \sigma(W \cdot h_i + b)$$

$h_i$ 是第 $i$ 个 token 的 BERT 输出（768 维），$W$ 是 $1 \times 768$ 的矩阵，$\sigma$ 是 sigmoid。sigmoid 把输出映射到 $[0, 1]$，但这**不是概率分布**——所有 token 的权重不需要加起来等于 1。每个 token 独立打分：0.95 = 很重要，0.05 = 不重要。

代码结构（对应 NAIETermweight 项目）：

```python
class BertSeg(BertPreTrainedModel):
    def __init__(self, config):
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # BMES 分类
        self.crf = CRF(config.num_labels, batch_first=True)                # 序列标注
        self.weight_regressor = nn.Linear(config.hidden_size, 1)           # 词权重回归
```

### 2.7 词权重模型的数据构建

训练数据的权重标签不是人工标注的，而是根据 query 在正/负文档中的匹配情况**自动生成**：

```python
for word in query_words:
    synonyms = get_synonyms(word)  # 同义词扩展
    if any(syn in positive_doc for syn in synonyms):
        # 词（或同义词）出现在正文档中 → 基础权重 1.0
        # 每在一个负文档中也出现，权重减 0.08（区分度降低）
        weight = 1.0 - 0.08 * neg_count
    else:
        weight = 0.0  # 不在正文档中 → 不重要
```

**直觉**：如果一个词在正文档有、负文档没有 → 区分正负的关键信号 → 权重高。如果正负文档都有 → 区分度低 → 权重打折。

**难负例挖掘**：用不同匹配阈值 `[0.5, 0.7]`、`[0.2, 0.5]`、`[0, 0.2]` 查找关键词部分重叠但整体不相关的文档。

### 2.8 词权重模型的 Loss

总 Loss 由两个任务组成：

$$L_{total} = \frac{1}{2} L_{CRF} + L_{MSE}$$

#### 2.6.1 $L_{MSE}$：权重回归任务（简单的那个）

每个 token 的预测权重 $w_i^{pred}$（sigmoid 输出）和目标权重 $w_i^{true}$（2.7 节自动生成的）算均方误差：

$$L_{MSE} = \frac{1}{N} \sum_{i=1}^{N}(w_i^{pred} - w_i^{true})^2$$

这个好理解，不展开。

#### 2.6.2 $L_{CRF}$：分词任务——从交叉熵推广到序列

**先回顾交叉熵**：Part 3.6 的 MLM 在每个 mask 位置做分类——从词表的 30522 个词里挑出正确的那个：

$$L_{CE} = -\log \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}$$

$z_j$ 是第 $j$ 个类的 logit（模型输出的原始分数），$k$ 是正确类的下标，$C$ 是类别数。本质：让正确类的分数占所有类分数的比例最大。

**CRF 做的事一模一样，只是把"类"换成了"序列"**：不再是从 $C$ 个类里挑一个，而是从所有可能的标签序列里挑正确的那条。

$$L_{CRF} = -\log \frac{e^{S(\mathbf{y}^*)}}{\sum_{\tilde{\mathbf{y}}} e^{S(\tilde{\mathbf{y}})}}$$

| 符号 | 含义 |
|------|------|
| $\mathbf{y}^*$ | 正确的标签序列，如 $(B, E, B, E)$ |
| $\tilde{\mathbf{y}}$ | 任意一条可能的标签序列 |
| $S(\mathbf{y})$ | 一条标签序列的**总分** |

结构和交叉熵完全一样：分子是正确答案的分数，分母是所有候选的分数之和。区别在于 $S(\mathbf{y})$ 怎么算——交叉熵里 $z_k$ 就是一个 logit，CRF 里 $S(\mathbf{y})$ 由两部分组成。

#### 2.6.3 序列总分 $S(\mathbf{y})$：发射分 + 转移分

$$S(\mathbf{y}) = \underbrace{\sum_{t=1}^{T} E(t, y_t)}_{\text{发射分之和}} + \underbrace{\sum_{t=1}^{T-1} T(y_t, y_{t+1})}_{\text{转移分之和}}$$

| 符号 | 含义 | 来源 | 维度 |
|------|------|------|------|
| $E(t, y_t)$ | 位置 $t$ 取标签 $y_t$ 的**发射分** | BERT 输出经 Linear(768→4) | 每个位置 4 个值 |
| $T(y_t, y_{t+1})$ | 从标签 $y_t$ 转移到 $y_{t+1}$ 的**转移分** | CRF 层的可学习参数 | 4×4 矩阵 |

**发射分**：BERT 在每个位置输出 768 维向量，经过 `Linear(768→4)` 得到 4 个分数——分别对应 B、M、E、S 四个标签。发射分衡量的是"根据上下文，这个位置**本身**最像哪个标签"。

**转移分**：一个 $4 \times 4$ 的可学习参数矩阵，$T(i, j)$ 表示"从标签 $i$ 跳到标签 $j$ 要加多少分"。合法转移（B→M、B→E 等）的分数会被训高，非法转移（B→B、M→B 等）的分数会被训到很低。

用 "自然语言" 四个字的具体例子：

```
BERT 输出经 Linear(768→4)，得到每个位置的发射分:

         B     M     E     S
位置1(自): 3.0   0.1   0.2   1.5
位置2(然): 0.3   1.0   2.8   0.5
位置3(语): 2.5   0.2   0.3   1.0
位置4(言): 0.1   0.8   3.0   0.2

正确序列 y* = (B, E, B, E)，表示 "自然" "语言" 两个词

S(B,E,B,E) = 发射分之和 + 转移分之和
           = [E(1,B) + E(2,E) + E(3,B) + E(4,E)] + [T(B,E) + T(E,B) + T(B,E)]
           = [3.0 + 2.8 + 2.5 + 3.0]             + [1.8 + 2.5 + 1.8]
           = 11.3                                  + 6.1
           = 17.4

错误序列 (B, B, E, S):
S(B,B,E,S) = [3.0 + 0.3 + 0.3 + 0.2]             + [T(B,B) + T(B,E) + T(E,S)]
           = 3.8                                    + [-100 + 1.8 + 2.2]
           = 3.8 + (-96)
           = -92.2   ← T(B,B) = -100 把这条路径的分数压到极低
```

#### 2.6.4 分母怎么算：前向算法

$L_{CRF}$ 的分母 $\sum_{\tilde{\mathbf{y}}} e^{S(\tilde{\mathbf{y}})}$ 要对所有可能的标签序列求和。$K$ 个标签、$T$ 个位置，共 $K^T$ 条序列——4 个标签 × 500 个位置 = $4^{500}$ 种，不可能穷举。

**前向算法**用动态规划在 $O(T \times K^2)$ 时间内算出这个和。核心思想：

定义 $\alpha_t(j)$ = 位置 1 到位置 $t$、且位置 $t$ 的标签为 $j$ 的所有部分路径的分数之和：

$$\alpha_t(j) = \sum_{\text{所有到达}(t, j)\text{的路径}} e^{S(\text{该路径})}$$

递推关系（从位置 $t-1$ 到位置 $t$）：

$$\alpha_t(j) = \sum_{i=1}^{K} \alpha_{t-1}(i) \cdot e^{T(i,j)} \cdot e^{E(t,j)}$$

读作：位置 $t$ 取标签 $j$ 的所有路径分数和 = 对位置 $t-1$ 的每个可能标签 $i$，把"到 $i$ 的路径和" × "从 $i$ 转移到 $j$ 的分数" × "位置 $t$ 发射 $j$ 的分数"，全部加起来。

```
用 "自然语言" 的例子（K=4, T=4）:

初始化（位置 1）:
  α₁(B) = exp(E(1,B)) = exp(3.0) = 20.1
  α₁(M) = exp(E(1,M)) = exp(0.1) = 1.1
  α₁(E) = exp(E(1,E)) = exp(0.2) = 1.2
  α₁(S) = exp(E(1,S)) = exp(1.5) = 4.5

递推到位置 2:
  α₂(E) = [α₁(B)·exp(T(B,E)) + α₁(M)·exp(T(M,E)) + ...] · exp(E(2,E))
         = 对 4 个来源标签求和 × 发射分
  ...依此类推，每步 4×4=16 次乘加

最终:
  分母 = α₄(B) + α₄(M) + α₄(E) + α₄(S)    ← 所有到达位置 4 的路径分数之和

总计算量: T × K × K = 4 × 4 × 4 = 64 次乘加（而非 4⁴ = 256 次穷举）
对于 T=500: 500 × 16 = 8000 次（而非 4⁵⁰⁰ 次）
```

#### 2.6.5 CRF Loss 完整流程总结

$$L_{CRF} = -\log \frac{e^{S(\mathbf{y}^*)}}{\sum_{\tilde{\mathbf{y}}} e^{S(\tilde{\mathbf{y}})}} = -S(\mathbf{y}^*) + \log \sum_{\tilde{\mathbf{y}}} e^{S(\tilde{\mathbf{y}})}$$

展开后变成两项：
- $-S(\mathbf{y}^*)$：正确序列的总分取负（分子，直接算，$O(T)$）
- $\log \sum e^{S(\tilde{\mathbf{y}})}$：所有序列的 log-sum-exp（分母，前向算法，$O(TK^2)$）

和交叉熵的对比：

| | 交叉熵 | CRF Loss |
|---|---|---|
| 正确答案 | 一个标签（如"苹果"） | 一条标签序列（如 B-E-B-E） |
| 候选数 | $C$ 个类（词表大小） | $K^T$ 条序列（指数级） |
| 分数 | 每个类一个 logit $z_j$ | 每条序列一个总分 $S(\mathbf{y}) = \Sigma E + \Sigma T$ |
| 分母计算 | 直接求和（$C$ 次加法） | 前向算法（$O(TK^2)$ 动态规划） |
| 本质 | 让正确类的概率最大 | 让正确序列的概率最大 |

#### 2.6.6 总 Loss

$$L_{total} = \frac{1}{2} L_{CRF} + L_{MSE}$$

$L_{CRF}$ 除以 2 是为了平衡两个任务的梯度量级——CRF loss 通常比 MSE 大。

### 2.9 稀疏检索演进总结

| | TF-IDF | BM25 | 学习的词权重 |
|---|---|---|---|
| **权重来源** | 统计公式 | 统计公式（非线性） | BERT 学习 |
| **理解上下文？** | 否 | 否 | 是 |
| **需要训练？** | 否 | 否（只需设 k1, b） | 是（需要标注数据） |
| **能区分同形异义？** | 否 | 否 | 是（"学科技" vs "学科"） |
| **功能词处理** | IDF 自动降权，但不够精确 | 同左 | 模型学会功能词权重 ≈ 0 |
| **部署** | 倒排索引 | 倒排索引 | 倒排索引（权重来自模型推理） |

---

## Part 3: BERT——Encoder-only 的诞生

稀疏检索（Part 2）解决了关键词匹配的问题。但"猫"匹配不到"喵星人"——需要语义理解。稠密检索的底座就是 BERT 及其后续模型。

### 3.1 从 Transformer 拆零件

01-08 章我们写的 MiniMind 是 Decoder-only——只保留了原始 Transformer 的右半边。BERT 是 Encoder-only——只保留了左半边。
![alt text](image-5.png)

| 模型 | 保留了什么 | 扔掉了什么 | 特点 |
|------|-----------|-----------|------|
| **BERT** | 只用 Encoder | 扔掉 Decoder | 双向注意力（没有 Mask） |
| **GPT / LLaMA / MiniMind** | 只用 Decoder 的子层 1 和 3 | 扔掉 Encoder 和 Cross-Attention | 单向（有 Mask），自回归生成 |

**"双向"的含义**：注意力矩阵没有遮挡，每个 token 能看到序列里所有其他 token（前面的和后面的）。MiniMind 的 causal mask 让注意力矩阵变成下三角，每个 token 只能看前面的。

### 3.2 BERT 的架构

**BERT 的创新不在架构——它和 Transformer Encoder 完全一样——而在训练方式。**

```
BERT-base 一层:
┌─────────────────────┐
│     Add & Norm      │
│         ↑           │
│    Feed Forward     │
│         ↑           │
│     Add & Norm      │
│         ↑           │
│  Multi-Head Attention│
│  (无 Mask，双向)      │
└─────────────────────┘
        × 12 层
```

| | BERT-base | BERT-large |
|---|---|---|
| Encoder 层数 | 12 | 24 |
| 隐藏维度 | 768 | 1024 |
| 注意力头数 | 12 | 16 |
| 参数量 | ~110M | ~340M |

### 3.3 特殊 token

| 作用 | 原始 Transformer | BERT | GPT-2 | LLaMA |
|------|-----------------|------|-------|-------|
| 序列开始 | `<BOS>` | `[CLS]` | 无 | `<s>` |
| 序列结束 | `<EOS>` | `[SEP]` | `<\|endoftext\|>` | `</s>` |
| 句子分隔 | 无 | `[SEP]` | 无 | 无 |
| 填充 | `<PAD>` | `[PAD]` | `<\|endoftext\|>` | `<pad>` |
| 遮蔽 | 无 | `[MASK]` | 无 | 无 |

这些 token 不是手写的，是 **tokenizer 自动加的**：

```python
tokenizer("我喜欢吃苹果")
# input_ids: [101, 2769, 1599, 3614, 1391, 5765, 3362, 102]
#             ↑                                          ↑
#           [CLS]=101                                [SEP]=102

tokenizer("猫吃什么", "猫喜欢吃鱼")  # 双句输入
# [101,  猫, 吃, 什么, 102,  猫, 喜欢, 吃, 鱼, 102]
#  [CLS]              [SEP]                    [SEP]
```

### 3.4 特殊 token 的 embedding 本质

[CLS]、[SEP] 都有可学习的 embedding 向量，跟普通词一模一样——模型内部不"知道"谁是特殊 token。但这些 embedding 不需要编码复杂语义。要区分两个阶段：

```
阶段 1: 输入 embedding（查表得到，进入模型前）
  [CLS] 的输入 embedding = 永远同一个 768 维向量
  它学到的只是一个"身份标记"：告诉模型"我是序列开头"

  经过 12 层 Self-Attention（每一层都 attend 到所有 token）

阶段 2: 输出向量（12 层变换后）
  [CLS] 的输出向量 = 每个句子都不同，因为它 attend 了不同的内容
  这才是承载语义信息的向量
```

**输入 embedding 是门票（告诉模型我是谁），输出向量是会议纪要（融合了所有人说的话）。**

```python
cls_embedding = model.embeddings.word_embeddings.weight[101]  # 永远同一个向量

out1 = model(**tokenizer("我喜欢猫", return_tensors="pt"))
out2 = model(**tokenizer("今天天气真好", return_tensors="pt"))
cls_out_1 = out1.last_hidden_state[0, 0, :]  # [768] 句子 1 的 [CLS] 输出
cls_out_2 = out2.last_hidden_state[0, 0, :]  # [768] 句子 2 的 [CLS] 输出
# cls_out_1 ≠ cls_out_2（内容不同，语义不同）
```

所有 token 都经过 attention，但不是所有输出都被使用：

```
输入:  [CLS]  我  喜欢  猫  [SEP]
        ↓     ↓    ↓    ↓    ↓      ← 全部进入 12 层 attention
     ┌───────────────────────────┐
     │  12 层 Transformer Encoder  │
     └───────────────────────────┘
        ↓     ↓    ↓    ↓    ↓      ← 全部有输出向量
       h_cls  h_我 h_喜欢 h_猫 h_sep

       用 ✅   ×    ×    ×   ×  ← Reranker 只取 [CLS]，其他扔掉
```

### 3.5 BERT 的输入 Embedding

BERT 的输入 embedding 由**三部分相加**：

```
Token Embedding:     [CLS]  我  喜欢  [SEP]  苹果  很甜  [SEP]
                       +     +    +     +     +     +     +
Segment Embedding:     A     A    A     A     B     B     B    ← 区分句子 A/B
                       +     +    +     +     +     +     +
Position Embedding:    0     1    2     3     4     5     6    ← 位置信息
                       =     =    =     =     =    =     =
最终输入:            送入 Transformer Encoder
```

Segment Embedding 是 BERT 特有的，告诉模型哪些 token 属于第一个句子、哪些属于第二个。对比 MiniMind：我们在 03 章用 RoPE（旋转位置编码），BERT 用可学习的绝对位置编码。

### 3.6 预训练任务 1：MLM（Masked Language Model）

随机选 15% 的 token 做处理：

```
原句: 我 喜欢 吃 苹果 很甜

选中"苹果"后:
  80% 概率 → 替换成 [MASK]:   我 喜欢 吃 [MASK] 很甜
  10% 概率 → 替换成随机词:     我 喜欢 吃  西瓜  很甜
  10% 概率 → 保持不变:         我 喜欢 吃  苹果  很甜
```

**为什么不全用 [MASK]？** fine-tune 时输入里没有 [MASK]。如果预训练全用 [MASK]，模型只学会"看到 [MASK] 就预测"，而不是真正理解上下文。

MLM Head 结构——**只在被 mask 的位置上**接预测头：

```
h₃ (768 维) → Linear(768→768) + GELU + LayerNorm → Linear(768→30522) → Softmax
                                                     ↑ 投射到词表大小
→ {苹果: 0.82, 香蕉: 0.05, ...} → CrossEntropyLoss
```

### 3.7 预训练任务 2：NSP（Next Sentence Prediction）

50% 正例 50% 负例：

```
正例: A="我今天去了超市"  B="买了很多水果"       → IsNext
负例: A="我今天去了超市"  B="法国的首都是巴黎"    → NotNext
```

NSP Head：只看 [CLS] 输出，接二分类头：

```
h₀ (768 维) → Linear(768→2) → Softmax → {IsNext: 0.92, NotNext: 0.08}
```

### 3.8 两个任务同时训练

```
输入: [CLS] 我 今天 去了 [MASK] [SEP] 买了 很多 水果 [SEP]

                ┌────────────────────────────┐
                │   12 层 Transformer Encoder  │
                └────────────────────────────┘
                  ↓                    ↓
                 h₀                   h₃
                  │                    │
                  ↓                    ↓
              NSP Head             MLM Head
           (IsNext/NotNext)      (预测"超市")

        总 Loss = Loss_mlm + Loss_nsp
```

**预训练完成后，MLM Head 和 NSP Head 都扔掉**，只保留 12 层 Encoder 的参数。这些参数已经学会了语义表示能力，可以接各种新的 Head 做下游任务——接分类头变 Reranker，接 pooling 变 Embedding 模型。

### 3.9 [CLS] 能聚合语义的原因

[CLS] 能代表整句语义，不是因为它的 embedding 特殊，而是因为 **NSP 任务训练时专门取它的输出做分类** → 梯度反传迫使 12 层 attention 把有用信息汇聚到这个位置。

但后来 SBERT 论文实验发现：[CLS] 直接当句子向量效果其实不太好，不如对所有位置做 mean pooling（平均池化）。原因是 MLM 的梯度分散在所有 token 上，[CLS] 只被 NSP 弱弱地训过——没有被强迫成为高质量的句子表示。

**这个问题在 Part 6 的 RetroMAE 中被解决。**

---

## Part 4: RoBERTa——训练改良版

**架构和 BERT 完全一样。所有区别都在训练策略上。**

| 差异点 | BERT | RoBERTa |
|--------|------|---------|
| 训练数据 | 16GB | **160GB**（10 倍） |
| Mask 策略 | 静态（预处理时定好，每个 epoch 不变） | **动态**（每个 epoch 重新随机 mask） |
| NSP 任务 | 有 | **去掉了** |
| Batch Size | 256 | **8192**（32 倍） |
| 分词器 | WordPiece | **Byte-level BPE** |

**动态 Mask**：同一条样本在不同 epoch 被 mask 的词不同，相当于数据增强。

```
Epoch 1: 我 喜欢 吃 [MASK] 很甜     ← mask 了"苹果"
Epoch 2: 我 [MASK] 吃 苹果 很甜     ← mask 了"喜欢"
Epoch 3: 我 喜欢 [MASK] 苹果 很甜   ← mask 了"吃"
```

**为什么去掉 NSP？** RoBERTa 实验观察到：去掉 NSP 后几乎所有下游任务效果持平或更好。NSP 太简单——随机抽的负例和正例差异太大，模型靠主题词就能判断。不如把算力全给 MLM。

### 4.2 XLM-RoBERTa：多语言版 RoBERTa

BGE 系列模型用的底座不是原始 BERT 或 RoBERTa，而是 XLM-RoBERTa（X = Cross-Lingual，跨语言）。架构和 RoBERTa **完全一样**，区别在训练数据和词表：

| | BERT | RoBERTa | XLM-RoBERTa |
|---|---|---|---|
| 训练数据 | 英文 16GB | 英文 160GB | **CommonCrawl 100+ 语言 2.5TB** |
| 分词器 | WordPiece | Byte-level BPE | **SentencePiece BPE**（语言无关，见 11b） |
| 词表大小 | ~30K | ~50K | **~250K**（覆盖 100+ 语言） |
| 参数量 | base 110M | 同 BERT | base 270M / large 550M |
| 语言 | 英文为主 | 英文 | **100+ 语言** |

**为什么 BGE 选它？** BGE-M3 要做多语言检索（M3 的第一个 M = Multi-Linguality）。XLM-RoBERTa 在 100+ 语言上预训练过，天然支持中英文混合场景。词表 250K 能覆盖各种语言的 token。

从这里开始，BERT/RoBERTa/XLM-RoBERTa 底座已经准备好了。下一步是接不同的"头"变成不同的下游模型。

---

## Part 5: 两种用法分叉——Cross-Encoder vs Bi-Encoder

同一个 BERT/RoBERTa 底座，接不同的"头"就变成不同的模型。**这是理解整个检索系统的分叉点。**

### 5.1 架构对比

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cross-Encoder（Reranker）              Bi-Encoder（Embedding 模型）

      相关性分数: 0.87                    余弦相似度(u, v) = 0.87
          ↑                                    ↑
     Linear 线性层                        cosine_sim(u, v)
     (768 → 1)                            ↑           ↑
          ↑                           Pooling       Pooling
    [CLS] 的输出向量                      ↑               ↑
          ↑                        ┌────────────┐  ┌────────────┐
┌─────────────────────┐            │ BERT/RoBERTa│  │ BERT/RoBERTa│
│  一个 BERT/RoBERTa   │            │  (共享权重)   │  │  (共享权重)   │
│  query 和 doc 的每个  │            └────────────┘  └────────────┘
│  token 互相 attend   │                  ↑               ↑
└─────────────────────┘            ┌────────────┐  ┌────────────┐
          ↑                        │   query     │  │  document   │
┌─────────────────────┐            └────────────┘  └────────────┘
│[CLS] query [SEP] doc│
└─────────────────────┘              各自独立编码       无交互！
   拼接成一个序列输入

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

| | Cross-Encoder (Reranker) | Bi-Encoder (Embedding) |
|---|---|---|
| **BERT 几个？** | 1 个 | 2 个（共享权重的孪生网络） |
| **输入方式** | query+doc 拼接一起送入 | query 和 doc 分别独立送入 |
| **token 交互** | 深度交互（每个 token 互相 attend） | 无交互 |
| **输出** | [CLS] → Linear(768→1) → 分数 | pooling → 向量 |
| **精度** | 更高 | 较低 |
| **速度** | 慢（每对都要前向传播） | 快（向量预计算） |
| **能离线索引？** | 不能 | 能（向量存数据库） |

### 5.2 "Cross" ≠ Cross-Attention

**Cross-Encoder 内部没有 Cross-Attention。** 它用的就是普通 BERT、普通的 Self-Attention。

"Cross" 的意思是 **"两段文本交叉地互相看到了"**——靠**拼接**实现：把 query 和 doc 拼成一个序列 `[CLS] query [SEP] doc [SEP]` 送进普通 BERT。因为 Self-Attention 没有 Mask，query 的每个 token 自然能看到 doc 的每个 token。

```
Cross-Encoder 的注意力矩阵（全是 1，就是普通 Self-Attention）:
             [CLS] 猫  吃  什么 [SEP] 猫  喜欢 吃  鱼 [SEP]
             ─── query 部分 ───  ────── doc 部分 ──────
   猫(q)  [   1    1   1   1    1    1    1   1   1   1  ]
   什么(q)[   1    1   1   1    1    1    1   1   1   1  ]  ← query 看到 doc
   猫(d)  [   1    1   1   1    1    1    1   1   1   1  ]  ← doc 看到 query
   鱼(d)  [   1    1   1   1    1    1    1   1   1   1  ]

Bi-Encoder: 两个独立的注意力矩阵，互不相关
BERT-1（query）:           BERT-2（doc）:
     猫  吃  什么               猫  喜欢  吃  鱼
猫 [ 1   1    1 ]          猫 [ 1    1    1   1 ]
吃 [ 1   1    1 ]         喜欢[ 1    1    1   1 ]
什么[ 1   1    1 ]          吃 [ 1    1    1   1 ]
                            鱼 [ 1    1    1   1 ]
query 看不到 doc ❌        doc 看不到 query ❌
```

### 5.3 为什么 Cross-Encoder 精度更高

**两种模型都是"输入一个 (query, doc) pair，输出一个分数"**，区别只在内部怎么算。

```
query = "companies that did NOT go bankrupt"

  对 doc1 = "XX 公司宣布破产":
    Bi-Encoder:    cosine(embed(query), embed(doc1))           → 0.57
    Cross-Encoder: "[CLS] query [SEP] doc1 [SEP]" 整体送入 BERT → 0.30

  对 doc2 = "XX 公司持续盈利":
    Bi-Encoder:    cosine(embed(query), embed(doc2))           → 0.57
    Cross-Encoder: "[CLS] query [SEP] doc2 [SEP]" 整体送入 BERT → 0.54
```

| | doc1（"宣布破产"）| doc2（"持续盈利"）| 能区分吗？ |
|---|---|---|---|
| Bi-Encoder | 0.57 | 0.57 | 不能 |
| Cross-Encoder | 0.30 | 0.54 | 能 |

**Bi-Encoder 为什么分不出来？** query 和 doc 各自独立压成一个 768 维向量。query 里的 "not" 和 doc 里的 "破产" 在编码过程中**从未见过面**。"not" 的否定信号被 "破产、公司" 这些强词的语义淹没——768 维要压缩整句话，否定词在向量空间里权重太小。两个 doc 都包含 "破产" 相关语义，向量就很像。

**Cross-Encoder 为什么能区分？** query 和 doc 拼成一条序列进同一个 BERT。self-attention 让 "not" 的 Q 向量直接 attend 到 "破产" 的 K 向量，模型学会了 "not + 破产 = 不相关" 这种组合判断。

代价：Cross-Encoder 不能预计算 doc 向量，每个候选都要在线跑一遍前向传播 → 只用于 top-k rerank。

### 5.4 预训练 → Fine-tune 链路

```
预训练阶段（MLM + NSP）→ 扔掉 MLM Head 和 NSP Head → 保留 Encoder 参数

                    ↙                     ↘
Fine-tune 为 Reranker（Cross-Encoder）     Fine-tune 为 Embedding（Bi-Encoder）
  新接: Linear(768→1) 分类头                新接: 无额外参数（直接用 pooling）
  输入: [CLS] query [SEP] doc [SEP]         输入: 单个句子
  输出: h₀ → Linear → 相关性分数             输出: pooling(h₀~hₙ) → 768 维向量
  损失: BCEWithLogitsLoss                   损失: InfoNCE（对比学习）
```

---

## Part 6: BGE Embedding 的训练——Bi-Encoder 怎么训

> **本节讲的是 Embedding 模型（Bi-Encoder）的训练**——Part 5 讲的两种用法中的左边那条路。Reranker（Cross-Encoder）的训练在 Part 7。

BGE 是目前最主流的 embedding 体系。以下基于 C-PACK 论文（arXiv:2309.07597）和 BGE-M3 论文（arXiv:2402.03216）。

### 6.1 核心问题

Part 3.9 讲过：BERT 的 MLM 预训练没有强迫 [CLS] 成为好的句子向量。直接拿 BERT 的 [CLS] 做句子相似度，效果甚至不如 GloVe 词向量取平均。

BGE 的三阶段训练就是为了解决这个问题——把 BERT/RoBERTa 的 [CLS] 从一个"没什么用的位置标记"变成一个"高质量的句子向量"。

### 6.2 三阶段 Pipeline

```
起点: BERT/RoBERTa MLM 预训练权重（已学会 token 级语义）
  ↓ 用这套权重初始化 encoder
Stage 1: RetroMAE 后预训练（无标注数据）
  悟道/Pile 等无标注文本 → 非对称 masked autoencoder → 逼 [CLS] 压缩全句语义
  解决: MLM 训出的 [CLS] 不是好的句子向量
  ↓
Stage 2: 无监督对比学习（1 亿弱监督文本对）
  标题-正文等自动构造的文本对 → InfoNCE + in-batch negatives（batch=19,200）
  解决: 让模型学会区分相关 vs 不相关
  ↓
Stage 3: 有监督微调（84 万标注 query-doc 对）
  人工标注数据 → InfoNCE + hard negatives + 指令前缀
  解决: 在高质量数据上提精度
```

**三段各解决一个问题，缺一不可**：RetroMAE 把 CLS 逼成合格的句子向量（打底）→ 1 亿弱监督对 + 19,200 大 batch InfoNCE（练区分度）→ 84 万标注对 + hard negative + 指令微调（提精度）。

**自始至终都是同一个 encoder-only 模型**，三个 Stage 是逐步改进它的句子表示质量。

### 6.3 Stage 1：RetroMAE——让 [CLS] 成为合格的句子向量

RetroMAE 本质上是一个 **post-pretraining**（后预训练）：encoder 从 BERT 预训练权重初始化（`--model_name_or_path bert-base-uncased`），decoder 随机初始化（1 层小 Transformer，训完扔掉）。它不是替代 MLM，而是**站在 MLM 的肩膀上**，换一个更适合检索的任务继续训练——逼 [CLS] 成为合格的句子向量。

具体做法是把标准 MLM 改成一个"编码-解码"任务，三个关键设计：**非对称结构**（encoder 全尺寸，decoder 只有 1 层）、**非对称 mask 率**（encoder 轻度 15-30%，decoder 重度 50-70%）、**信息瓶颈**（decoder 必须从 [CLS] 向量里拿信息才能重建）。

用一个具体例子走完整流程：

```
原文: "自然语言处理是人工智能的核心方向"

═══ 编码器（完整的 12 层 Encoder，训完保留）═══

输入（轻度 mask，只遮了 2 个词）:
  "自然 [MASK] 处理 是 人工智能 的 [MASK] 方向"
  大部分词还在 → 编码器能理解整句意思

输出: 每个 token 都有 768 维向量，但只取 [CLS] 位置的那一个
      → [CLS] 向量 = 768 维的"整句压缩摘要"

═══ 解码器（1-2 层小 Transformer，训完扔掉）═══

输入有两部分:
  ① 编码器传来的 [CLS] 向量（768 维，就一个向量）
  ② 同一句话的重度 mask 版本（遮了 70%）:
     "[MASK] [MASK] [MASK] 是 [MASK] [MASK] 的 [MASK] [MASK]"

[CLS] 向量被当作解码器的一个额外输入 token——
解码器做 self-attention 时，每个位置都能 attend 到它

解码器的任务: 在每个 [MASK] 位置预测原来的词

  位置 0 的 [MASK] → Linear(768→vocab) → softmax
    → {自然: 0.85, 语言: 0.03, ...}  正确答案"自然" → CrossEntropy
  位置 1 的 [MASK] → Linear(768→vocab) → softmax
    → {语言: 0.78, 自然: 0.05, ...}  正确答案"语言" → CrossEntropy
  ...

Loss = 所有被 mask 位置的 CrossEntropy 的平均
```

**为什么这能让 [CLS] 变好？** 解码器的输入被 mask 得很狠（70%），光靠残余 token 根本猜不出原文。它**不得不依赖编码器传来的 [CLS] 向量**。反向传播时：解码器预测错了 → 梯度传回 [CLS] → 传回编码器所有 12 层 → 逼编码器把更多信息塞进 [CLS]。

**一句话总结：MLM 训的是"每个 token 的表示"，RetroMAE 训的是"那个唯一的句子向量"。**

**语料来源**：BGE 中文版用悟道（Wudao），英文版用 Pile。C-Pack 论文没有公布这一步的具体数据量和训练步数——这是 recipe 里唯一不够透明的部分。

### 6.4 Stage 2：InfoNCE + 大 Batch——练区分度

RetroMAE 让 [CLS] 成为合格的句子向量。Stage 2 的目标是让模型学会"这两个向量该近还是该远"。

**数据**：从 Wudao、知乎、百科、新闻等语料中自动构造约 **1 亿对**文本对（中文），比如同一篇文章的标题和正文就是一对正样本。英文数据量约为中文的 2 倍。使用 text2vec-chinese 对候选对打分，0.43 阈值过滤噪声对。

#### InfoNCE Loss = 多分类交叉熵

你已经很熟悉分类任务的交叉熵——Part 3.6 的 MLM 就是在词表上做分类：

```
MLM（从词表里挑出被 mask 的词）:
  logits = [2.1, 0.5, 3.8, 0.1, ...]   ← 词表大小个数
  正确答案 = 第 k 个类（"苹果"）
  Loss = -log(softmax(logits)[k])
```

InfoNCE **完全一样**，只不过把"从词表里挑词"换成了"从一堆文档里挑出正确的那个"：

```
一个 batch 有 4 个 query-doc 对:
  (q1, d1+)  (q2, d2+)  (q3, d3+)  (q4, d4+)

对 q1，算它和所有 doc 的相似度，除以温度 τ:
  logits = [sim(q1,d1+), sim(q1,d2+), sim(q1,d3+), sim(q1,d4+)] / τ
         = [0.9,         0.3,         0.1,         0.2        ] / 0.05
         = [18,          6,           2,           4          ]

正确答案 = 第 0 个（d1+ 是 q1 的正样本）

Loss = -log(softmax(logits)[0])
     = CrossEntropyLoss(logits, label=0)   ← 完全等价！
```

公式：

$$L_{\text{InfoNCE}} = -\log \frac{e^{\text{sim}(q, d^+) / \tau}}{e^{\text{sim}(q, d^+) / \tau} + \sum_{i=1}^{N} e^{\text{sim}(q, d_i^-) / \tau}}$$

> **记忆 tips**：InfoNCE = `CrossEntropyLoss(相似度 / τ, label=正样本位置)`。
> 和 MLM 完全同构：MLM 是"从 3 万个词里挑对的"，InfoNCE 是"从 B 个 doc 里挑对的"。
> 公式里的 exp、求和、取 log 不用单独记——就是 softmax + cross-entropy 展开后的样子。

其中 $\text{sim}(q, d) = \cos(q, d)$ 是余弦相似度，$\tau$ 是温度参数（BGE 用 0.02）。

用 PyTorch 写出来就三行：

```python
similarity_matrix = queries @ docs.T       # [B, B] 相似度矩阵
labels = torch.arange(B)                   # [0, 1, ..., B-1] 对角线是正样本
loss = F.cross_entropy(similarity_matrix / temperature, labels)
```

#### 温度 τ 的作用

$\tau$ 就是 softmax 之前对 logits 做的缩放。$\tau$ 越小 → logits 被放大 → softmax 越尖锐 → 模型被迫把正样本的分数拉得远远高于负样本 → 梯度集中在最难区分的负例上。

```
假设: sim(q, d+) = 0.8,  sim(q, d-) = 0.5

τ = 1.0:   softmax([0.8, 0.5]) → 正样本概率 = 0.575   ← 差距不大
τ = 0.1:   softmax([8.0, 5.0]) → 正样本概率 = 0.953   ← 差距放大
τ = 0.05:  softmax([16, 10])   → 正样本概率 = 0.9975  ← 几乎确定
```

#### In-batch Negatives：batch 里的其他样本就是负例

不需要为每个 query 显式准备负样本。一个 batch 里有 $B$ 个 (query, pos_doc) 对，每个 query 的负样本就是 batch 内其他 query 的 pos_doc：

```
一个 batch 有 4 个 query-doc 对:
  (q₁, d₁⁺)  (q₂, d₂⁺)  (q₃, d₃⁺)  (q₄, d₄⁺)

对 q₁: 正样本 = d₁⁺，负样本 = d₂⁺, d₃⁺, d₄⁺

相似度矩阵（B × B）:
          d₁⁺    d₂⁺    d₃⁺    d₄⁺
  q₁   [ 0.92   0.31   0.15   0.28 ]  ← 对角线是正样本
  q₂   [ 0.22   0.88   0.19   0.33 ]
  q₃   [ 0.11   0.25   0.91   0.17 ]
  q₄   [ 0.35   0.29   0.14   0.85 ]

InfoNCE 就是让对角线的值尽量大，其余尽量小
→ 等价于 B 类分类问题，对角线位置是正确类别
```

#### 为什么 batch 要撑到 19,200

In-batch negatives 下，负例数量 = batch size。batch 越大，负例越多，里面撞上"语义很像但不是答案"的难负例概率越高，模型区分度越高。

C-Pack 消融实验（论文 Table 4）：

| batch size | Retrieval 分数 | Average 分数 |
|-----------|---------------|-------------|
| 256 | 57.25 | 56.43 |
| 2,048 | 60.96 | 57.92 |
| **19,200** | **63.90** | **59.00** |

batch 从 256 到 19,200，检索分数涨了 **6.65 个点**——整个 recipe 里单点收益最大的 trick。

**19,200 怎么塞进显存？GradCache。** 把 19,200 拆成显存吃得下的小 chunk，每个 chunk 只前向、缓存 embedding，不算梯度；全部 embedding 拿到后算一次 InfoNCE 损失，再逐 chunk 重放前向并反传。用重算换显存，等效于单卡跑超大 batch。

### 6.5 Stage 3：有监督微调——高质量数据 + Hard Negatives

Stage 2 用 1 亿弱监督对练出了初步的区分度。Stage 3 的 loss 函数还是 InfoNCE，**变的是数据和负例**。

**数据**：C-MTP (labeled)，共 **838,465 对**，来源是 T2-Ranking、DuReader、mMARCO（中文版）、CMedQA-v2 等人工标注集。对比 Stage 2 的 1 亿弱监督对，数据量小两个数量级但质量高得多。

#### Hard Negative Mining

Stage 2 的 in-batch negatives 是随机的，大多数和 query 差异很大——比如 query 是"感冒了吃什么药"，随机负例可能是"Python 3.12 新特性"，模型已经能轻松区分，梯度几乎为零。

更有价值的是**难负例**——当前模型觉得和 query 很像、但实际不是答案的文档。

**前提：Stage 2 训完后已经有一个初步能用的 embedding 模型**，用它来找"它自己分不清的样本"：

```
query = "感冒了吃什么药"
正样本 = "感冒常用药包括布洛芬、对乙酰氨基酚..."

第 1 步: 用 Stage 2 训完的 embedding 模型，把语料库所有文档编码成向量

第 2 步: 把 query 也编码成向量，用 ANN 检索 top-200 最相似的文档:
         top-1:  "感冒常用药包括布洛芬..."     ← 正样本本身，跳过
         top-3:  "流感和普通感冒的区别"         ← 语义很近，但不是答案！
         top-8:  "小儿感冒的家庭护理"           ← 问的不是药！
         top-15: "咳嗽吃什么药比较好"           ← 相关但不是感冒！
         ...

第 3 步: 从 top-2 ~ top-200 里采样 15 个作为难负例

为什么叫"难"？
  当前模型觉得它们和 query 很像（不然不会排进 top-200）
  但标注告诉我们它们不是正确答案
  → 模型当前分不清的样本 → 用它们训练收益最大
```

#### 假负例问题：万一采到了真正的正样本？

top-200 里确实可能混进未标注的正确答案——比如 top-12 恰好是一篇"感冒药有哪些？布洛芬、对乙酰氨基酚..."，内容完全正确但没被标注。它会被当成负样本，InfoNCE 会试图压低它的相似度。

实际操作中的处理：

```
1. top-1 直接跳过（几乎肯定是正样本或近义改写）
2. 和标注数据中已有的 pos 列表做完全匹配去重
3. 剩下的"漏网之鱼"——不处理

为什么不处理也行？
  15 个难负例里混进 1-2 个假负例 → 噪声比例 ~10%
  训练数据有 84 万对，整体噪声被平均掉了
  少量假负例的噪声 远小于 难负例带来的训练收益
  C-Pack 论文没有做去噪，效果依然 SOTA
```

后续工作有针对性的改进——如 Conan-Embedding（2024）提出用 Cross-Encoder 对候选负例二次过滤，把真正的正样本挑出来剔除。但 BGE 的 recipe 里没做这一步。

Stage 3 的 loss 在 in-batch negatives 之外显式加入难负例：

$$L = -\log \frac{e^{\text{sim}(q, d^+) / \tau}}{e^{\text{sim}(q, d^+) / \tau} + \sum_{j \in \text{batch}} e^{\text{sim}(q, d_j) / \tau} + \sum_{k \in \text{hard}} e^{\text{sim}(q, d_k^-) / \tau}}$$

#### 指令前缀

query 侧拼接任务指令（如"为这个句子生成表示以用于检索相关文章："），doc 侧不加。C-Pack 消融显示去掉指令会掉点——指令帮助模型区分不同任务需要什么样的表示。

**效果**：BGE-i（Stage 2 checkpoint）平均 59.00 → BGE-f（Stage 3 最终）63.96，涨了约 5 个点。

#### Stage 3 超参数

来自官方 FlagEmbedding 示例脚本，论文未完整公布：

| 参数 | encoder-only（bge-v1.5/M3 系） | decoder-only（gemma2/ICL 系） |
|------|-------------------------------|------------------------------|
| lr | 1e-5 | 1e-4（LoRA, rank 32 / alpha 64） |
| temperature τ | 0.02 | 0.02 |
| epochs | 2 | 1 |
| warmup ratio | 0.1 | 0.1 |
| pooling | cls + normalize | last_token + normalize |
| 精度 | fp16 + gradient checkpointing | fp16 + deepspeed |

**工程细节：`same_dataset_within_batch`**。M3 和 gemma2 微调时强制同 batch 样本来自同一数据集。因为 in-batch negatives 里如果混着别的任务的样本（比如把 STS 的正例对当检索负例），会产生假负例污染梯度。

### 6.6 BGE-M3：同一个模型，三种读法

BGE-M3 的 "M3" 是 Multi-Linguality、Multi-Granularity、Multi-Functionality。其中 Multi-Functionality 的意思是：**同一个 XLM-RoBERTa，同一次前向传播，三种不同的方式读输出**。

```
输入: "感冒了吃什么药"
         ↓
┌─────────────────────────────┐
│  XLM-RoBERTa（只跑一次前向）   │
│  输出: 每个 token 一个向量      │
│  [CLS]  h₁  h₂  h₃  h₄  h₅  │
└─────────────────────────────┘
         ↓ 同一份输出，三种读法
   ┌─────┼──────────┐
   ↓     ↓          ↓

读法 1: Dense（和 BGE v1.5 一样）
  取 [CLS] 的输出 → 一个 768 维向量
  检索时: cosine(query_vec, doc_vec)
  用途: 向量检索

读法 2: Sparse（类似 Part 2 的词权重模型，但训练方式完全不同）
  取每个 token 的输出 → Linear(768→1) → ReLU → 每个 token 一个非负权重
  检索时: 稀疏向量内积（倒排索引）
  用途: 关键词精确匹配

读法 3: ColBERT（多向量）
  取所有 token 的输出 → 每个 token 保留为一个独立向量
  检索时: query 的每个 token 和 doc 的每个 token 算相似度
          每个 query token 取最大匹配 → 求和（MaxSim）
  用途: 比 Dense 精确（保留了 token 级信息），比 Cross-Encoder 快（doc 向量可预计算）
```

**没有额外的"头"**——Dense 直接取 [CLS]，Sparse 只加了一个 `Linear(768→1) + ReLU`，ColBERT 直接用所有 token 的输出。三种读法**共享同一个 backbone 的所有参数**。

#### M3 Sparse vs Part 2 词权重模型

M3 Sparse 和 Part 2 的词权重模型都输出"每个 token 一个权重"，但工作方式完全不同：

| | Part 2 词权重模型 | BGE-M3 Sparse |
|---|---|---|
| **需要显式分词？** | 是（CRF 做 BMES 标注） | 否（直接用 tokenizer 的子词结果） |
| **分词粒度** | 词级（"科技" 整体一个权重） | subword token 级（"科""技" 各一个权重） |
| **需要权重标签？** | 是（人工规则：正文档有→1.0，负文档也有→减 0.08） | 否 |
| **Loss** | CRF_loss + MSE_loss | InfoNCE（和 Dense 模式用同一个 loss） |
| **和 Dense 共享参数？** | 不共享（独立模型） | 共享（同一个 backbone） |

**M3 Sparse 怎么不需要权重标签也能学？** 训练信号完全来自 InfoNCE 的对比：

```
给定 (query, pos_doc, neg_doc):
  query 过 BERT → 每个 subword token 过 Linear+ReLU → 稀疏权重向量
  pos_doc 和 neg_doc 同理

  稀疏相似度 = 共同 token 的权重乘积之和

  喂进 InfoNCE: 让 sim(query, pos) > sim(query, neg)

反向传播时:
  "科" 的权重太低 → pos_doc 匹配分不够高 → 梯度增大 "科" 的权重
  "的" 的权重太高 → neg_doc 匹配分也很高 → 梯度减小 "的" 的权重
  → 不需要人告诉模型哪个词重要，从对比信号里自己学会了
```

#### 自蒸馏：三种读法互相当老师

三种读法各有擅长——Dense 擅长语义匹配，Sparse 擅长精确匹配，ColBERT 擅长细粒度交互。把三种分数融合起来比任何单路都更准。

```
对同一个 (query, doc) 对，三种读法各给一个分数:
  s_dense   = cosine(CLS_q, CLS_d)
  s_sparse  = 稀疏向量内积
  s_colbert = MaxSim

融合成 teacher score:
  s_teacher = w₁ · s_dense + w₂ · s_sparse + w₃ · s_colbert

然后用 s_teacher 作为每种读法的训练目标:
  Dense 的 InfoNCE: 用 s_teacher 排序正负样本（而不是 one-hot）
  Sparse 的 InfoNCE: 同上
  ColBERT 的 InfoNCE: 同上
```

**为什么叫"自蒸馏"？** 没有外部 teacher 模型——三种读法自己给自己当老师。Dense 从 Sparse 那里学到精确匹配信号，Sparse 从 Dense 那里学到语义泛化能力，各自的短板被其他读法补上。

### 6.7 BGE Embedding 训练总结

| 阶段 | 数据量 | 数据来源 | 方法 | 关键 trick |
|------|--------|---------|------|-----------|
| Stage 1: RetroMAE | 未公布 | 悟道 / Pile | 非对称 masked autoencoder | encoder 从 BERT 权重初始化，重度 mask 解码器逼 [CLS] 压缩全句 |
| Stage 2: 无监督对比 | ~1 亿对 | 标题-正文、知乎、百科 | InfoNCE + in-batch negatives | **batch=19,200**（GradCache）|
| Stage 3: 有监督微调 | 84 万对 | T2-Ranking、DuReader 等 | InfoNCE + hard negatives | 指令前缀 + same_dataset_within_batch |

> 数据来源：C-Pack 论文（arXiv:2309.07597）Table 4 消融实验、BGE-M3 论文（arXiv:2402.03216）、FlagEmbedding 官方示例脚本。

---

## Part 7: Reranker 的训练——Cross-Encoder 怎么训

> Part 5 讲的两种用法中的右边那条路。和 Part 6 的 BGE Embedding 训练相比，Reranker 的训练简单得多——不需要 RetroMAE、不需要对比学习、不需要大 batch。拿一个预训练好的 BERT/RoBERTa，接个 Linear 头，用 BCE loss 微调就完了。

### 7.1 模型结构

```python
class CrossEncoder(nn.Module):
    def __init__(self, hf_model, tokenizer, loss_type='classfication'):
        super().__init__()
        self.model = hf_model  # AutoModelForSequenceClassification（已含分类头）
        self.tokenizer = tokenizer
        self.loss_type = loss_type

    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels=1):
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(hf_model, tokenizer)
```

`AutoModelForSequenceClassification` 内部 = BERT + `Linear(768→1)`。输入 `[CLS] query [SEP] document [SEP]`，由 tokenizer 自动拼接。

### 7.2 数据构建

训练数据格式：

```json
{"query": "RTX 4060 显存多大", "pos": ["配备 8GB GDDR6 显存"], "neg": ["RTX 4090 配备 24GB", ...]}
```

每个 query 的正负样本展开成 (query, doc, label) 三元组：

```python
for item in data:
    for doc in item['pos']:
        train_data.append([item['query'], doc, 1])        # 正样本 label=1
    for doc in item['neg'][:15]:                          # 最多取 15 个负样本
        train_data.append([item['query'], doc, 0])        # 负样本 label=0
```

**正负比例 1:15**——负样本越多，模型越能学会区分"看起来相关但实际不相关"的文档。

也支持多级标签的回归模式：

| 标签值 | 数值 | 含义 |
|--------|------|------|
| 3 | 1.0 | 强正例 |
| 2 | 0.75 | 中等相关 |
| 1 | 0.5 | 弱相关 |
| 0 | 0.25 | 弱正例 |

### 7.3 Loss 设计

Reranker 的任务：给 (query, doc) 对打分，判断相关（1）还是不相关（0）——二分类问题。

#### 为什么用 Sigmoid 不用 Softmax？

Sigmoid 和 Softmax 是两个完全不同的函数：

```
Sigmoid: 把一个标量映射到 (0, 1)，每个值独立映射
  sigmoid(x) = 1 / (1 + exp(-x))
  x = -10 → 0.00005 ≈ 0
  x = 0   → 0.5
  x = 10  → 0.99995 ≈ 1

Softmax: 把一组数映射成概率分布（加起来 = 1），值之间有竞争关系
  输入 [2.0, 1.0, 0.5] → [0.59, 0.24, 0.17]  和 = 1.0
```

Reranker 每次只输出一个标量分数（`num_labels=1`），不是在多个类别之间选择，所以用 sigmoid 把这一个分数映射成"相关的概率"。Softmax 需要至少两个值才有意义。

#### 分类模式（主流）：BCEWithLogitsLoss

BCE = Binary Cross-Entropy（二元交叉熵）。分两步理解：

**第 1 步：Sigmoid 归一化**——模型输出的原始分数 $z$（logit，任意实数）经过 sigmoid 变成概率 $p$：

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

```
z = -3   → p = 0.05  （模型觉得"不相关"）
z = 0    → p = 0.5   （不确定）
z = 2.5  → p = 0.92  （模型觉得"很相关"）
```

**第 2 步：用 $p$ 算交叉熵**——和标签 $y$ 比较：

$$L = -\left[y \cdot \log(p) + (1-y) \cdot \log(1-p)\right]$$

合在一起写（把 $p = \sigma(z)$ 代入）：

$$L = -\left[y \cdot \log\sigma(z) + (1-y) \cdot \log(1-\sigma(z))\right]$$

| 符号 | 含义 | 取值范围 |
|------|------|---------|
| $z$ | 模型输出的原始分数（logit） | 任意实数 |
| $\sigma(z)$ | sigmoid，把 $z$ 归一化成概率 $p$ | $(0, 1)$ |
| $y$ | 标签：正样本=1，负样本=0 | $\{0, 1\}$ |

**$y$ 是正负样本的开关**——决定走公式的哪一项：

```
当 y=1（正样本，query 和 doc 相关）:
  (1-y) = 0，第二项消失
  L = -log(σ(z))
  → 想让 loss 小 → 要 σ(z) 大 → 要 z 大 → 模型要给正样本打高分 ✓

当 y=0（负样本，query 和 doc 不相关）:
  y = 0，第一项消失
  L = -log(1 - σ(z))
  → 想让 loss 小 → 要 σ(z) 小 → 要 z 小 → 模型要给负样本打低分 ✓
```

**PyTorch 用 `BCEWithLogitsLoss` 而不是分开做 sigmoid + BCELoss**——因为 $\log(\sigma(z))$ 在 $z$ 很大或很小时会数值溢出。`BCEWithLogitsLoss` 内部把 sigmoid 和 log 合并成一个数值稳定的表达式，避免中间步骤溢出。

**推理时不做 sigmoid**——排序只看分数高低，sigmoid 是单调函数（$z$ 大则 $\sigma(z)$ 大），加不加 sigmoid 排序结果一样。

> **记忆方法：BCE = sigmoid 归一化 + 交叉熵。正样本要高分，负样本要低分，$y$ 是开关。**

用一个完整的训练 batch 走一遍：

```
batch 有 4 个样本（1 正 3 负），每个都是 [CLS] query [SEP] doc [SEP] 送进 BERT:

样本 1: query="感冒吃什么药"  doc="感冒常用药有布洛芬"  y=1（正样本）
  → z₁ = 2.5 → loss = -log(σ(2.5)) = -log(0.92) = 0.08   ← 打分正确

样本 2: query="感冒吃什么药"  doc="流感和感冒的区别"    y=0（负样本）
  → z₂ = 0.8 → loss = -log(1-σ(0.8)) = -log(0.31) = 1.17  ← 大！给负样本打了偏高分

样本 3: query="感冒吃什么药"  doc="Python 3.12 新特性"  y=0（负样本）
  → z₃ = -3  → loss = -log(1-σ(-3)) = -log(0.95) = 0.05   ← 轻松区分

样本 4: query="感冒吃什么药"  doc="咳嗽吃什么药"        y=0（负样本）
  → z₄ = 1.5 → loss = -log(1-σ(1.5)) = -log(0.18) = 1.71  ← 很大！难负例没区分开

batch loss = (0.08 + 1.17 + 0.05 + 1.71) / 4 = 0.75
```

**BCE vs InfoNCE 的区别**：

| | BCE（Reranker 用） | InfoNCE（Embedding 用） |
|---|---|---|
| 输入 | 一个 (query, doc) 对 | 一个 query 对比 batch 里所有 doc |
| 决策 | "这一对相关还是不相关" | "从一堆里挑出对的那个" |
| 正负样本 | 每个样本独立算 loss | 正负样本在同一个 softmax 里竞争 |
| 等价于 | 二分类 | 多分类交叉熵 |

```python
if self.loss_type == 'classfication':
    loss = BCEWithLogitsLoss()(logits.squeeze(), labels.squeeze())
```

#### 回归模式：先 sigmoid 再 MSE

适用于多级标签（0.25/0.5/0.75/1.0）：

```python
if self.loss_type == 'regression':
    logits = torch.sigmoid(logits)  # 映射到 [0,1]
    loss = MSELoss()(logits.squeeze(), labels.squeeze())
```

### 7.4 长文档切片

BERT 的 max_length 通常 512 token。doc 超长时用**滑动窗口 + overlap + 取最大分数**：

```
Query: "RTX 4060 显存"（10 token）
Doc: 2000 token 的长文档
可用于 doc 的长度 = 512 - 10 - 3([CLS]+两个[SEP]) = 499 token

切片（overlap = 80 token）:
  片段 1: doc[0:499]     → 送入 BERT → 0.32
  片段 2: doc[419:918]   → 送入 BERT → 0.78  ← 最高
  片段 3: doc[838:1337]  → 送入 BERT → 0.45
  片段 4: doc[1257:1756] → 送入 BERT → 0.21
  片段 5: doc[1676:2000] → 送入 BERT → 0.15

该 doc 的最终分数 = max(所有片段) = 0.78
```

**为什么要 overlap？** 避免关键信息被切断。如果答案恰好跨越边界，没有 overlap 两边都拿不到完整信息。

### 7.5 蒸馏

大模型 Reranker（如 560M 的 large 版本）精度高但推理慢。蒸馏的目标：训一个小模型（如 66M），精度接近但推理快 5-8 倍。

#### 为什么学 teacher 的输出比学标签更好？

```
标签只有 0 和 1（硬标签）:
  (query, doc_A) → 1    ← 只知道"相关"
  (query, doc_B) → 0    ← 只知道"不相关"

Teacher 的输出（软标签）:
  (query, doc_A) → 0.95  ← "非常相关"
  (query, doc_B) → 0.42  ← "有点相关但不太对"——标签说 0，但 teacher 看出了部分相关性
  (query, doc_C) → 0.03  ← "完全不相关"

软标签比硬标签多了细粒度信息:
  doc_B 标签是 0，但 teacher 给了 0.42
  → 这种"暗知识"帮 student 学到更精细的判断
```

#### 三种蒸馏方法：区别在"学 teacher 的哪一层"

以 TinyBERT 为例具体展开。假设 Teacher 是 12 层 BERT-large（hidden_size=1024），Student 是 4 层小 BERT（hidden_size=512）。

**第一个问题：Teacher 12 层，Student 4 层，怎么对齐？**

手动指定层映射：Student 第 1 层模仿 Teacher 第 3 层，第 2 层模仿第 6 层，以此类推——均匀采样。

```
Teacher:  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]  [10] [11] [12]
                     ↑              ↑              ↑               ↑
Student:            [1]            [2]            [3]             [4]
```

**隐藏层对齐**（通用层蒸馏的核心）：

```
同一条输入 "感冒了吃什么药" 分别送进 Teacher 和 Student:

Teacher 第 3 层输出: [batch, seq_len, 1024]   ← 1024 维
Student 第 1 层输出: [batch, seq_len, 512]    ← 512 维，维度不一样！

Student 侧加一个可学习的投影矩阵 W_proj: [512, 1024]
  student_projected = student_output @ W_proj   → [batch, seq_len, 1024]

Loss_hidden = MSE(student_projected, teacher_output)
            = 每个位置、每个维度的差的平方的平均
```

**Attention 矩阵对齐**：

```
Teacher 第 3 层的 attention: [batch, 16 heads, seq_len, seq_len]
Student 第 1 层的 attention: [batch, 8 heads, seq_len, seq_len]

头数不一致 → 跳过 attention 对齐（设 alpha_attn = 0）
头数一致 → Loss_attn = MSE(student_attention, teacher_attention)
```

三种方法的区别：

| 方法 | 学 Teacher 的什么 | Loss |
|------|------------------|------|
| DistilBERT | 只学最终 logits | $\text{MSE}(z_{student}, z_{teacher})$ |
| TinyBERT | 中间层隐藏状态 + attention 矩阵 + 最终 logits | $\alpha L_{hidden} + \beta L_{attn} + \gamma L_{logits}$ |
| MiniLMv2 | 中间层的 Q/K/V 矩阵（比 attention 矩阵更精细） | $\alpha L_{qkv}$ |

#### 蒸馏三阶段

```
Stage 1: 通用层蒸馏（大规模数据 1000 万+）
  同一条文本分别过 Teacher 和 Student
  对齐 4 组映射层的隐藏状态和 attention
  不需要标签——只需要 Teacher 和 Student 看同一条文本
  目的: 让 Student 学到通用的语义表示能力

Stage 2: 预测层蒸馏（中规模数据 100 万）
  不再对齐中间层，只对齐最终输出:
  Loss = MSE(z_student, z_teacher)
  例: Teacher 给 2.5 分，Student 给 1.0 分 → loss = (1.0-2.5)² = 2.25
  目的: 让 Student 的打分和 Teacher 对齐

Stage 3: 领域微调（小规模数据 <10 万）
  不再用 Teacher，直接用标签:
  Loss = BCEWithLogitsLoss(z_student, y)
  目的: 在目标领域的标注数据上微调
```

蒸馏基础原理（temperature、soft label 等）见 12b-知识蒸馏.md。

### 7.6 训练配置

| 参数 | 值 |
|------|-----|
| 底座模型 | bge-reranker-base (XLM-RoBERTa) |
| optimizer | AdamW（bias/LayerNorm 不衰减） |
| lr | 5e-5 |
| scheduler | cosine with warmup |
| batch_size | 4（单卡）× 32（梯度累积）= 等效 128 |
| max_len | 512 |
| epochs | 1-2 |
| mixed_precision | fp16 |
| 分布式 | Accelerate |

---

## Part 8: 业界最新演进——Pooling 变迁与 Decoder 化趋势

### 8.1 Pooling 策略的三个时代

Bi-Encoder 需要把序列输出聚合成一个句子向量。三种做法：

```python
# CLS Pooling: 取 [CLS] 位置的输出
sentence_vec = last_hidden_state[:, 0, :]  # [batch, 768]

# Mean Pooling: 所有非 PAD token 取平均
mask = attention_mask.unsqueeze(-1)
sentence_vec = (last_hidden_state * mask).sum(1) / mask.sum(1)

# Last Token Pooling: 取最后一个 token（decoder-only 用）
last_idx = attention_mask.sum(1) - 1
sentence_vec = last_hidden_state[torch.arange(B), last_idx, :]
```

**真正的规律不是"encoder 用 CLS、decoder 用 last token"，而是：池化方式跟着训练方式走，推理时必须和训练时一致。**

| 时代 | 代表模型 | 默认 Pooling | 原因 |
|------|----------|-------------|------|
| SBERT/E5 时代 | all-MiniLM-L6-v2、E5 v1 | **Mean** | SBERT 论文发现 CLS 直接用效果差 |
| BGE/GTE 时代 | BGE v1.5、BGE-M3、GTE | **CLS** | 大规模对比学习把 CLS 训成聚合点。**BGE 必须用 CLS，用 mean 效果会掉** |
| LLM 底座时代 | bge-gemma2、e5-mistral、Qwen3-Embedding | **Last Token** | Decoder-only 没有 CLS，最后一个 token 天然"看过"前面所有 token |

### 8.2 Embedding 侧最新架构

| 模型 | 底座 | Pooling | 参数量 | 特点 |
|------|------|---------|--------|------|
| BGE v1.5 | XLM-RoBERTa | CLS | 278M/560M | 部署主力 |
| BGE-M3 | XLM-RoBERTa | CLS + sparse + ColBERT | 568M | 三路混合检索 |
| bge-multilingual-gemma2 | Gemma2 | Last Token | 9B | LLM 底座，刷榜 |
| e5-mistral | Mistral-7B | Last Token | 7B | 微软出品 |
| Qwen3-Embedding | Qwen3 | Last Token | 0.6B/8B | 2025 年，多语言 |

### 8.3 Reranker 侧最新架构

| 模型 | 底座 | 打分方式 | 参数量 |
|------|------|---------|--------|
| bge-reranker-base/large | XLM-RoBERTa | CLS → Linear(768→1) | 278M/560M |
| bge-reranker-v2-m3 | XLM-RoBERTa | CLS → Linear | 568M |
| bge-reranker-v2-gemma | Gemma2-9B | Last Token → 分类头 | 9B |
| Qwen3-Reranker | Qwen3 | 生成式：取 "yes"/"no" 的 logit 算概率 | 0.6B/8B |

**Qwen3-Reranker 的生成式打分**——不再接分类头，而是让 decoder-only 模型"说" yes/no：

```
输入 prompt: "Given a query and a document, judge if the document is relevant.
              Query: {query}  Document: {doc}  Relevant:"

打分: P(relevant) = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))
```

### 8.4 趋势总结

底座都在从 BERT 系迁到 LLM 系，池化从 CLS 变成 last token。区别在节奏：
- **Embedding 侧**：decoder 化已比较主流（刷榜基本都是 LLM-based）
- **Reranker 侧**：encoder-only + CLS 在落地场景仍然活得很好（bge-reranker-v2-m3 只有 568M，打 9B 的 gemma 系差距不大）

---

## 面试话术

> **RAG 检索链路**
>
> 检索有两条路线：稀疏和稠密。稀疏路线从 TF-IDF 到 BM25 到学习的词权重模型，核心演进是把统计权重换成 BERT 学出来的上下文感知权重。稠密路线从 BERT/RoBERTa 出发，分叉成两种用法——Bi-Encoder 把句子各自编码成向量做召回，Cross-Encoder 把 query+doc 拼在一起做精排。"Cross"不是 Cross-Attention，是靠拼接让 Self-Attention 自然交叉。Cross-Encoder 精度高是因为 token 级深度交互能捕捉否定等细粒度信号，但不能预计算，所以只用于 top-k rerank。
>
> BGE Embedding 的训练分三阶段：RetroMAE 逼 [CLS] 压缩全句语义（MLM 训 token 表示，RetroMAE 训句子向量）→ 1 亿弱监督对 + batch=19,200 的 InfoNCE（InfoNCE 本质是多分类交叉熵）→ 84 万标注对 + hard negative mining（用当前模型检索 top-200 中采样难负例）。batch 从 256 到 19,200 检索分涨了 6.65 点，是 recipe 里收益最大的 trick，用 GradCache 塞进显存。
>
> Reranker 训练简单得多：BERT + Linear 头 + BCEWithLogitsLoss，长文档用滑动窗口切片 + overlap + max score。业界趋势是底座从 encoder-only 迁到 decoder-only，Qwen3-Reranker 已经用生成式打分（取 yes/no 的 logit 算概率）。

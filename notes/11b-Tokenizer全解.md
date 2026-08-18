# 11b - Tokenizer 全解：词表怎么从零构建

> 前置：[11-预训练](11-预训练.md)（预训练流程中的 tokenizer 训练部分）
> 后续：[21-RAG检索与Reranker](21-RAG检索与Reranker.md)（Part 4 RoBERTa 提到 WordPiece vs Byte-level BPE 的差异）
> 本文聚焦：词表怎么从零构建、四种主流分词算法的原理和差异、不同模型家族的具体做法、中文场景的特殊挑战。

## 在整个模型中的位置

```
原始文本 "我喜欢吃苹果"
    ↓
 ┌──────────────────────────────────────────────────────┐
 │  本文的范围                                            │
 │                                                        │
 │  ① Tokenizer 分词: "我喜欢吃苹果" → ["我","喜欢","吃","苹果"]  │
 │  ② 查词表得到 ID:  ["我","喜欢","吃","苹果"] → [2769, 1599, 1391, 5765] │
 │                                                        │
 └──────────────────────────────────────────────────────┘
    ↓
 Embedding 查表 (01章) → 向量 → Transformer → 任务头
```

11 章我们在预训练流程中用过 tokenizer，但没有详细讲词表是怎么造出来的。本文补全这个环节：词表构建是预训练之前的**独立步骤**，只跟语料有关，跟模型架构无关。

**核心问题**：词表是从哪来的？怎么决定"我喜欢吃苹果"该切成哪几个 token？不同模型为什么切法不同？

---

## 1. 词表在模型生命周期中的位置

```
造词表（本文内容）
  大量文本语料 → 分词算法（BPE / WordPiece / Unigram） → 词表文件
  ↓
预训练（11章）
  词表 → 初始化 embedding 矩阵 → 训练（MLM/自回归） → 学好的模型
  ↓
Fine-tune / 推理
  复用造好的词表 + 预训练的模型 → 接任务头（Reranker、Embedding 模型等）
  词表在 Fine-tune 时不变，原封不动继承
```

**关键认知**：词表一旦造好就固定了。后续预训练、Fine-tune 都复用同一份词表。

---

## 2. 造词表：四种子词算法

### 分词的三个层级（为什么需要子词？）

在讲具体算法之前，先理解分词的三个层级——它们是所有分词算法的背景：

```
层级 1: Word-level（词级分词）
  英文按空格切分: "I love cats" → ["I", "love", "cats"]
  中文需要分词器: "我喜欢猫" → ["我", "喜欢", "猫"]（jieba 等）
  
  问题: 词表极大（英文 50万+），且无法处理未登录词（OOV）
        训练时没见过 "tokenizer" → 直接变成 [UNK]

层级 2: Character-level（字符级分词）
  英文按字母: "cats" → ["c", "a", "t", "s"]
  中文按字:  "猫咪" → ["猫", "咪"]
  
  问题: 序列太长，单个字符缺乏语义（"c" 是什么意思？）

层级 3: Sub-word-level（子词级分词）← 当前主流
  "tokenizer" → ["token", "izer"]    常见词保持完整，罕见词拆成有意义的子词
  "unhappiness" → ["un", "happi", "ness"]
  
  核心原则: 高频词不拆，低频词分解为有意义的子词片段
```

下面介绍的 BPE、WordPiece、Unigram 都是子词级分词算法，只是**构建词表的策略不同**。

### 2.1 BPE 训练：从字符开始，自底向上合并

BPE = Byte Pair Encoding（字节对编码）。核心操作：找到语料中最高频的一**对**（Pair）相邻单元，合并成一个新 token，重复直到词表达到目标大小。

#### 训练过程（完整示例）

```
训练语料（统计词频后）:
  l o w       出现 5 次
  l o w e r   出现 2 次
  n e w e s t 出现 6 次
  w i d e r   出现 3 次

初始词表 = 所有出现过的字符: {d, e, i, l, n, o, r, s, t, w}  共 10 个

第 1 轮: 统计所有相邻字符对的频率
  (e, s): 6次  (s, t): 6次  (l, o): 7次  (o, w): 7次  (e, r): 5次 ...
  最高频: (l, o) = 7次 → 合并成 "lo"
  记录合并规则 #1: l + o → lo
  词表变成: {d, e, i, l, n, o, r, s, t, w, lo}  共 11 个

第 2 轮: 用 "lo" 重新表示语料，重新统计
  lo w       出现 5 次
  lo w e r   出现 2 次
  n e w e s t 出现 6 次
  w i d e r   出现 3 次
  最高频: (e, s) = 6次 → 合并成 "es"
  记录合并规则 #2: e + s → es
  词表: 12 个

第 3 轮: (es, t) = 6次 → "est"
  规则 #3: es + t → est
  词表: 13 个

第 4 轮: (lo, w) = 7次 → "low"
  规则 #4: lo + w → low
  词表: 14 个

...不断重复，直到词表达到目标大小（如 30000 / 50000 / 150000）
```

**严格每轮只合并一个是教科书的简化描述**，真实实现会做优化：

```
1. 批量合并不冲突的 pair
   如果 (l,o) 和 (e,s) 在语料中没有位置重叠
   → 可以同时合并，不影响结果

2. 增量更新统计，不全量重算
   合并 (l,o) → lo 后，只更新 lo 左右邻居的 pair 频率
   时间复杂度从 O(语料长度) 降到 O(受影响的位置数)

3. 优先队列（堆）维护频率排序
   用 max-heap，O(log n) 取最高频 pair
```

#### 训练产出的文件

```
vocab.json（或 tokenizer.json）:
  {"lo": 10, "es": 11, "est": 12, "low": 13, ...}   ← 每个 token 对应一个 ID

merges.txt:
  l o          ← 合并规则 #1
  e s          ← 合并规则 #2
  es t         ← 合并规则 #3
  lo w         ← 合并规则 #4
  ...          ← 按学到的顺序排列
```

**vocab.json 就是词表，merges.txt 就是合并规则。** 推理时 tokenizer 读取这两个文件来分词。

#### 代码：训练一个 BPE tokenizer

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())

trainer = BpeTrainer(
    vocab_size=50000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
)

tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("my_tokenizer.json")
```

### 2.2 WordPiece 训练：思路类似，选择标准不同

WordPiece 也是从字符开始合并，但**选择合并哪一对的标准不同**：

```
BPE:       选频率最高的 pair
WordPiece: 选合并后能最大化语言模型似然的 pair
           即：合并 (a, b) → ab 后，P(ab) / (P(a) × P(b)) 最大的那对
```

#### 用数字理解这个公式

```
假设某一轮，语料中有这些候选 pair:
  (t, h): 相邻出现 100 次    其中 "t" 单独出现 500 次, "h" 单独出现 300 次
  (x, z): 相邻出现  20 次    其中 "x" 单独出现  25 次, "z" 单独出现  22 次

BPE（按频率）:
  (t, h) = 100 次  ← 频率最高，先合并它

WordPiece（按互信息）:
  score(x,z) = 20 / (25×22)   = 0.036   ← 最高！先合并它
  score(t,h) = 100 / (500×300) = 0.00067 ← 排后面
```

**分母 P(x)×P(z) 在问：如果 x 和 z 只是各自出现、碰巧挨在一起，我们预期看到多少次？**
- t 和 h 都是超高频字母，随机排列也经常挨着 → 100 次并不意外
- x 和 z 都很罕见，但 80%（20/25）的 x 旁边都有 z → 它们高度绑定

**直觉：BPE 问"谁出现得最多"，WordPiece 问"谁最不可能是巧合"。** 实际效果差异不大，因为高频 pair 和高互信息 pair 往往高度重叠。

#### 推理时的差异：合并规则 vs 贪心匹配

```
BPE 推理（回放合并规则）:
  输入 "lower" → 先拆成字符 [l, o, w, e, r]
  按 merges.txt 的顺序逐条回放:
    规则 #1: l+o → lo  → [lo, w, e, r]
    规则 #4: lo+w → low → [low, e, r]
    规则 #7: e+r → er  → [low, er]
  结果: ["low", "er"]
  特点: 必须保存 merges.txt，且顺序不能乱

WordPiece 推理（贪心最长匹配）:
  输入 "lower" → 从左往右，每次取词表中能匹配的最长 token:
    "lower" 在 vocab.txt 里？ 不在
    "lowe" 在 vocab.txt 里？  不在
    "low" 在 vocab.txt 里？   在！→ 取出 "low"
    剩余 "er" → 加 ## 前缀 → "##er" 在 vocab.txt 里？ 在！→ 取出 "##er"
  结果: ["low", "##er"]
  特点: 不需要合并规则文件，只需要 vocab.txt
       ## 前缀标记"这不是词首"
```

### 2.3 产出文件对比

```
WordPiece（BERT）:
  vocab.txt              ← 每行一个 token，行号就是 ID
  第 0 行: [PAD]
  第 101 行: [CLS]
  第 2769 行: 我
  ...共 30522 行

BPE（RoBERTa / GPT）:
  vocab.json             ← token → ID 的映射
  merges.txt             ← 合并规则（按学到的顺序排列）
  推理时两个文件配合使用
```

### 2.4 Byte-level BPE：为什么能保证不 [UNK]

普通 BPE 从**字符**开始合并——如果某个字符在训练语料中没出现过，它就不在初始词表里，会变成 [UNK]。

Byte-level BPE 从**字节**（0-255）开始合并——任何文本最终都是字节序列，而字节只有 256 种可能，初始词表就能覆盖所有情况：

```
普通 BPE 初始词表:  {a, b, c, ..., 我, 你, 他, ...}  ← 取决于训练语料，可能漏掉罕见字符
Byte-level 初始词表: {0x00, 0x01, ..., 0xFF}          ← 固定 256 个，覆盖一切

"🤖" 的 UTF-8 字节: [0xF0, 0x9F, 0xA4, 0x96]
即使没见过这个 emoji，也能拆成 4 个字节 token → 不会 [UNK]
```

### 2.5 Unigram：从大词表开始，自顶向下裁剪

BPE 和 WordPiece 都是**自底向上**——从最小单元开始合并。Unigram 是**反过来**的：从一个很大的初始词表开始，逐步删掉不重要的 token。

```
BPE / WordPiece:  小词表 → 不断合并 → 大词表（自底向上）
Unigram:          大词表 → 不断裁剪 → 小词表（自顶向下）
```

#### 训练过程

```
第 1 步: 构建一个非常大的初始词表
  包含所有单个字符 + 语料中出现的常见子串，可能有几十万个 token

第 2 步: 用 Unigram 语言模型给每个 token 打分
  模型假设: 每个 token 独立出现，P(句子) = P(x₁) × P(x₂) × ... × P(xₙ)
  其中 P(xᵢ) = count(xᵢ) / total_count
  整体 loss = -Σ log(P(xᵢ))

第 3 步: 对每个 token 计算"如果移除它，整体 loss 会增加多少？"
  增加少 → 不重要，可以删
  增加多 → 有用，保留

第 4 步: 移除 loss 增加最少的那 10-20% token
  基础字符永远不删（保证任何输入都能被编码）

第 5 步: 重复第 2-4 步，直到词表缩减到目标大小
```

#### 推理时的关键差异：概率最优路径

BPE 推理是确定性的（按 merges.txt 逐步回放），Unigram 推理要找**概率最大的分词方式**——同一个词可能有多种拆法，用维特比算法找最优：

```
"tokenizer" 的可能分词方式:
  方式 A: ["token", "izer"]      → P = P("token") × P("izer") = 0.003      ← 选这个
  方式 B: ["to", "ken", "izer"]  → P = P("to") × P("ken") × P("izer") = 0.001
  方式 C: ["t","o","k","e","n","i","z","e","r"] → 概率极小

维特比算法在 O(n²) 时间内找到最优分词，无需穷举
```

Unigram 还有独特能力：**带随机性的分词采样**。训练时可以对同一句话按概率生成不同的分词结果，相当于数据增强。

使用 Unigram 的代表模型：T5、ALBERT、XLNet、mBART——通常通过 SentencePiece 框架。

### 2.6 SentencePiece：不是算法，是统一框架

**SentencePiece 是 Google 开发的分词框架**，内部支持 BPE 和 Unigram 两种算法。它解决的核心问题是**语言无关的统一处理**。

#### 它解决什么问题？

```
传统 BPE/WordPiece 的隐含假设:
  文本已经被正确地按空格预分割了
  "I love cats" → 先按空格切成 ["I", "love", "cats"] → 再对每个词做子词分词
  
  问题 1: 中文、日文没有空格，怎么"先按空格切"？
  问题 2: 空格本身也是信息（代码里的缩进），丢掉就不可逆了
  问题 3: 需要针对不同语言写不同的预处理规则
```

SentencePiece 的做法：**把空格也当成普通字符，用特殊符号 ▁ 表示**。

```
SentencePiece 处理流程:
  原文: "I love cats"
  → 替换空格: "▁I▁love▁cats"（▁ = U+2581）
  → 直接在这个字符串上跑 BPE 或 Unigram，不需要预分词
  
  分词结果: ["▁I", "▁love", "▁cats"]
  反向拼接: 把 ▁ 换回空格 → "I love cats" ← 无损还原

对比传统方式:
  传统: "I love cats" → 按空格切 → 子词分词 → 空格信息丢失
  SP:  "I love cats" → "▁I▁love▁cats" → 子词分词 → 空格信息保留在 ▁ 里
```

#### 核心特性

| 特性 | 说明 |
|------|------|
| 语言无关 | 输入是 raw Unicode 字节流，不依赖任何语言的空格/标点规则 |
| 可逆分词 | 分词结果可以无损还原成原文 |
| 直接训练 | 输入原始文本文件，不需要先做分词/清洗等预处理 |
| NFKC 规范化 | 自动将 Unicode 变体统一（全角→半角、①→1） |
| 双算法支持 | 内部可选 `model_type='bpe'` 或 `model_type='unigram'` |

#### 产出文件对比

```
SentencePiece:       .model（二进制，一个文件包含词表+规则） + .vocab（可读）
HuggingFace BPE:     vocab.json + merges.txt（两个文件配合）
WordPiece (BERT):    vocab.txt（每行一个 token，行号=ID）
tiktoken (OpenAI):   .tiktoken 或 tokenizer.json（基于正则预分词 + BPE）
```

### 2.7 预分词（Pre-tokenization）：正则切分

在子词分词之前，很多模型会先用**正则表达式**把文本粗切一刀。目的是限制子词合并的范围——不让 BPE 跨越词边界合并出奇怪的 token。

```
GPT-2 / RoBERTa 的预分词正则:
  r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+"

  "I don't like 100cats!" →
  ["I", " don", "'t", " like", " 100", "cats", "!"]
  然后 BPE 只在每个片段内部做合并

LLaMA (SentencePiece):
  不需要预分词正则——SentencePiece 直接处理原始文本
```

**预分词正则是一个经常被忽略但影响很大的设计决策。** 如果微调一个模型但换了 tokenizer，预分词正则的差异就是 bug 来源之一。

### 2.8 四种子词算法总结对比

| | BPE | WordPiece | Unigram | SentencePiece |
|---|---|---|---|---|
| **构建方向** | 自底向上（合并） | 自底向上（合并） | 自顶向下（裁剪） | 框架（支持 BPE/Unigram） |
| **合并/裁剪标准** | 频率最高的 pair | 互信息最大的 pair | 移除后 loss 增加最小 | 取决于内部算法 |
| **训练起点** | 所有字符（或字节） | 所有字符 | 大初始词表 | 原始文本 |
| **推理方式** | 按 merges.txt 回放 | 贪心最长匹配 | 维特比找最优路径 | 取决于内部算法 |
| **是否确定性** | 是 | 是 | 可确定/可采样 | 取决于内部算法 |
| **需要预分词？** | 需要（正则切分） | 需要 | 不需要 | 不需要 |
| **产出文件** | vocab.json + merges.txt | vocab.txt | .model | .model + .vocab |

---

## 3. 中文 vs 英文：为什么分词差异这么大？

### 3.1 中英文分词的本质差异

```
英文: 天然有空格做词边界
  "I love natural language processing"
  → BPE 只需要处理罕见词的拆分

中文: 没有空格，没有天然的词边界
  "我喜欢自然语言处理"
  → 必须依赖算法决定怎么拆
  → 而且中文一个字的 UTF-8 编码 = 3 个字节（英文字母 = 1 个字节）
```

**核心问题在 UTF-8 编码长度不对等**：

```
英文 "cat":  UTF-8 = 3 个字节 [0x63, 0x61, 0x74]     → 3 字节表达 3 个字母
中文 "猫":   UTF-8 = 3 个字节 [0xE7, 0x8C, 0xAB]     → 3 字节只表达 1 个字

对 Byte-level BPE 来说:
  如果训练语料英文占 90%，"cat" 这 3 个字节会被高频合并成一个 token
  而 "猫" 的 3 个字节在英文语料里几乎不出现，频率太低无法合并
  → "猫" 被拆成 3 个字节 token → 一个汉字要花 3 个 token
```

### 3.2 中文 Token 税：具体有多严重？

```
输入: "人工智能正在重塑全球的信息基础设施"（16 个汉字）

LLaMA 2（词表 32K，中文语料极少）:
  每个汉字拆成 3 个字节 token → 可能 40+ 个 token

BERT 中文（WordPiece 逐字切分）:
  ["人", "工", "智", "能", ...] → 16+ 个 token

GPT-4:
  基本一字一 token → 19 个 token

Qwen 2.5（词表 152K，中文优化）:
  ["人工智能", "正在", "重塑", "全球", "的", "信息基础设施"] → 6 个 token
```

**这不只是效率问题，直接影响成本和能力**：

| 影响 | 说明 |
|------|------|
| API 费用 | 同样的中文对话，LLaMA 2 的 token 数可能是 Qwen 的 6 倍 |
| 上下文窗口 | 128K 窗口：Qwen 能装约 20 万字中文，LLaMA 2 只能装约 4 万字 |
| 推理速度 | token 越多 → 自回归步数越多 → 响应越慢 |

### 3.3 各模型中文编码效率排名

| 模型 | 中文压缩率（字符/token） | 说明 |
|------|------------------------|------|
| **Qwen 2.5/3** | ~1.5-1.8 | 中文最优之一，词表 152K |
| **DeepSeek V3** | ~1.7 | 官方数据：1 个汉字约 0.6 个 token |
| **ChatGLM 3** | ~1.3-1.5 | 中英平衡训练，词表 65K |
| **LLaMA 3** | ~1.0-1.3 | 比 LLaMA 2 大幅改善，词表扩到 128K |
| **GPT-4o** | ~1.0-1.2 | 比 GPT-4 改善 |
| **BERT 中文** | ~1.0 | 逐字切分，一字一 token |
| **LLaMA 2** | ~0.3-0.5 | 最差，每个汉字要 2-3 个 token |

> 数据来源：36氪《AI 大模型的「中文税」》、Qwen/DeepSeek 技术报告、社区实测。不同文本类型压缩率有差异，表中为中文通用文本的典型值。

### 3.4 Qwen 的词表怎么做的？

Qwen 的 tokenizer 是目前中文编码效率最好的之一。

#### 技术选型：tiktoken + Byte-level BPE

```
Qwen 不用 SentencePiece，而是选了 OpenAI 的 tiktoken 库:
  - 基于 cl100k_base 词表起步（GPT-4 的词表基础）
  - 大规模扩充中文和多语言 token
  - 最终词表: 151,643 + 208 特殊 token = 151,851

为什么选 tiktoken？
  - Rust 实现，编解码速度更快
  - 正则预分词的灵活性更高
```

#### 词表构建策略：大量注入中文 token

```
普通英文优先词表（如 LLaMA 2，32K）:
  中文相关 token 极少
  → "自然语言" 被拆成 "自""然""语""言" 四个单字

Qwen 的做法（152K）:
  专门添加大量中文词级 token:
  → "自然语言" 是一个 token
  → "人工智能" 是一个 token
  → "信息基础设施" 是一个 token

关键: 不只是加汉字，更重要的是加中文「词」
  BERT 中文词表有 7322 个 CJK 字符，但都是单字
  Qwen 的词表不仅有单字，还有大量多字词汇
  → 这才是压缩率高的真正原因
```

#### Qwen 的预分词正则

```python
PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# 关键: 中文字符属于 \p{L}（Unicode Letter），不会被逐字强制切开
# BPE 可以自由合并相邻汉字 → 能学到"人工智能"这种词级 token
```

### 3.5 过度合并 vs 不合并：中文分词的两个陷阱

#### 陷阱 1：不合并

```
LLaMA 2:
  "机器学习" → [0xE6, 0x9C, 0xBA, ...] → 12 个字节 token
  一个 4 字词花 12 个 token
```

#### 陷阱 2：错误合并（跨词边界）

Digital Orientalist 的研究发现，一些模型的 BPE 合并出了不该合并的 token：

```
句子: "他是学科技的，他不相信神秘和超自然的事物"

LLaMA:    错误地把 "的事" 合并成一个 token（而不是 "事物"）
DeepSeek: 错误地把 "他不" 和 "的物" 合并
ChatLM-mini: 错误地把 "学科" 合并（在 "学科技" 中应该是 "学+科技"）

根本原因: BPE 只看统计频率，不懂语义
```

#### 陷阱 3：词表污染

```
GPT-4o 词表中最长的中文合并 token 很多来自赌博和色情领域
→ 训练语料质量问题——这些领域文本量大、用词重复，BPE 统计频率高
→ 宝贵的词表空间被低质量 token 占据
```

#### 核心权衡

```
不合并:  安全但低效，中文要花 2-3 倍 token
过度合并: 高效但可能跨词错切，且浪费词表空间
平衡点:  Qwen 的做法——高质量中文语料训练 BPE + 足够大的词表（152K）
```

### 3.6 词表大小的权衡

词表越大，编码效率越高，但 embedding 矩阵也越大：

```
LLaMA 2: embedding = (32000, 4096)  ≈ 0.5 GB   中文极差
LLaMA 3: embedding = (128256, 4096) ≈ 2.0 GB   中文大幅改善
Qwen:    embedding = (151936, 3584) ≈ 2.1 GB   中文最优之一
DeepSeek: embedding = (102400, 5120) ≈ 2.0 GB  中文优秀
```

对于 7B+ 的大模型，embedding 占总参数比例很小（2GB / 14GB ≈ 14%），词表大的收益远大于成本。但对于 1B 以下的小模型，embedding 可能占总参数的 30%+，需要更谨慎地控制词表大小。

---

## 4. 汇总：各模型分词方案一览

| 模型 | 底层算法 | 框架/实现 | Byte-level | 词表大小 | 产出文件 | 中文压缩率 |
|------|---------|----------|-----------|---------|---------|-----------|
| BERT 中文 | WordPiece | Google 原生 | 否 | ~21K | vocab.txt | ~1.0 |
| GPT-2 / RoBERTa | BPE | HuggingFace | 是 | ~50K | vocab.json + merges.txt | — |
| T5 / ALBERT | Unigram | SentencePiece | 否 | ~32K | tokenizer.model | — |
| LLaMA 2 | BPE | SentencePiece | 是 | ~32K | tokenizer.model | ~0.3-0.5 |
| LLaMA 3 | BPE | tiktoken | 是 | ~128K | tokenizer.json | ~1.0-1.3 |
| GPT-4 / GPT-4o | BPE | tiktoken | 是 | ~100K | — | ~1.0-1.2 |
| ChatGLM 2/3 | BPE | SentencePiece | 是 | ~65K | tokenizer.model | ~1.3-1.5 |
| **Qwen 2.5/3** | **BPE** | **tiktoken** | **是** | **~152K** | **tokenizer.json** | **~1.5-1.8** |
| **DeepSeek V2/V3** | **BPE** | **tiktoken** | **是** | **~100K** | **tokenizer.json** | **~1.7** |

**规律**：2024 年之后的主流大模型基本都是 **Byte-level BPE + tiktoken**。中文效率差异取决于：① 词表大小（100K+ 才有空间给中文词）② 训练语料中的中文占比。

> 数据来源：Qwen 技术报告（arXiv:2309.16609、arXiv:2407.10671）、LLaMA/GLM/DeepSeek 技术报告、36氪《AI 大模型的「中文税」》、社区实测。

---

## 面试话术

> **大模型分词器（Tokenizer）**
>
> 主流有三种子词分词算法：BPE 自底向上，每轮合并频率最高的相邻 pair；WordPiece 也是自底向上但按互信息选 pair（不是看谁最多，而是看谁最不可能是巧合）；Unigram 反过来，自顶向下从大词表裁剪 loss 贡献最小的 token。SentencePiece 是 Google 的统一框架，支持 BPE 和 Unigram，核心贡献是语言无关（把空格当字符用 ▁ 保留，不需要预分词）。
>
> 2024 年后的主流大模型基本都是 Byte-level BPE + tiktoken（从 256 个字节开始合并，保证不 [UNK]），差异在词表大小和中文覆盖：BERT 用 WordPiece 逐字切分词表约 21K；LLaMA 2 用 SentencePiece BPE 词表只有 32K，中文每个字要拆成 3 个字节 token；Qwen 用 tiktoken BPE 词表 152K，大量注入中文词级 token（"人工智能""自然语言"整体编码），同样 16 字的句子 GPT-4 要 19 token，Qwen 只要 6 个。
>
> **追问预备**：
>
> *"BPE 和 Unigram 的核心区别？"*——方向相反：BPE 自底向上合并，Unigram 自顶向下裁剪；BPE 是确定性分词，Unigram 可以概率采样做数据增强。
>
> *"SentencePiece 和 BPE 什么关系？"*——SentencePiece 是框架不是算法，内部可选 BPE 或 Unigram，核心贡献是语言无关的统一处理（空格用 ▁ 保留，分词可逆）。
>
> *"中文场景哪家 tokenizer 做得好？为什么？"*——Qwen 和 DeepSeek 目前中文编码效率最高。关键是在大量高质量中文语料上训练 BPE + 足够大的词表（152K），让常见中文词整体合并成单个 token。LLaMA 2 最差因为词表只有 32K 且中文语料极少。核心权衡是词表大→编码效率高但 embedding 矩阵也大，对 7B+ 模型影响可忽略，对 1B 以下需要注意。
>
> *"BPE 合并中文会有什么问题？"*——BPE 只看统计频率不懂语义，可能跨词边界合并（"学科技"错切成"学科+技"）。另外训练语料质量也重要，GPT-4o 的词表里最长中文 token 很多来自赌博色情领域——语料污染。

---

## 参考资源

- [全网最全的大模型分词器（Tokenizer）总结 — LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)
- [HuggingFace Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [SentencePiece GitHub](https://github.com/google/sentencepiece)
- [Qwen Technical Report (arXiv:2309.16609)](https://arxiv.org/pdf/2309.16609)
- [Qwen2 Technical Report (arXiv:2407.10671)](https://arxiv.org/pdf/2407.10671)
- [DeepSeek-V2 Technical Report (arXiv:2405.04434)](https://arxiv.org/pdf/2405.04434)
- [AI 大模型的「中文税」— 36氪](https://www.36kr.com/p/3793050208984071)
- [The Pitfalls of Chinese Tokenization in LLMs — Digital Orientalist](https://digitalorientalist.com/2025/02/04/to-merge-or-not-to-merge-the-pitfalls-of-chinese-tokenization-in-general-purpose-llms/)

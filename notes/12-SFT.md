# 12 - SFT：从零开始写指令微调

## 在整个项目中的位置

```
Chapter 11: 预训练数据管道 + 训练脚本
  → scratch/dataset/lm_dataset.py（PretrainDataset）
  → scratch/trainer/train_pretrain.py

Chapter 12（本章）: SFT 数据管道 + LoRA 模块 + 训练脚本
  → scratch/dataset/lm_dataset.py（追加 SFTDataset）
  → scratch/trainer/train_full_sft.py（全参 SFT 训练循环）
  → scratch/model/model_lora.py（LoRA 结构 + 注入 + 保存/加载/合并）
  → scratch/trainer/train_lora.py（LoRA 训练循环）

Chapter 13: RL 对齐基础
```

SFT（Supervised Fine-Tuning）和预训练的区别只有一个：**Loss 只对 assistant 回复部分计算**。全参 SFT 更新所有参数，LoRA 只更新低秩增量——两者共享相同的数据格式和 loss mask。

---

## 从零开始写

> 以下 Step 1-8 逐步构建 SFT 训练的完整链路，每一步在真实文件上迭代修改：
>
> | Step | 操作 | 文件 |
> |------|------|------|
> | Step 1-2 | 理解 SFT 与 ChatML 格式 | 不涉及文件操作 |
> | Step 3 | 追加 `SFTDataset` 类 | `scratch/dataset/lm_dataset.py` |
> | Step 4 | **新建**，写全参 SFT 训练脚本 | `scratch/trainer/train_full_sft.py` |
> | Step 5 | 理解 LoRA 数学原理 | 不涉及文件操作 |
> | Step 6 | **新建**，写 LoRA 模块 | `scratch/model/model_lora.py` |
> | Step 7 | **新建**，写 LoRA 训练脚本 | `scratch/trainer/train_lora.py` |
> | Step 8 | 追加保存/加载/合并方法 | `scratch/model/model_lora.py` |
>
> 运行前提：Chapter 11 的 `PretrainDataset` 和 `train_pretrain.py` 已可用。

### Step 1: SFT 和预训练的区别——为什么需要 loss mask

预训练时，模型学习预测**所有** token——每个位置的 label 都是下一个 token（padding 除外）：

```
预训练:
input_ids: [BOS, 中, 国, 首, 都, 是, 北, 京, EOS, PAD]
labels:    [BOS, 中, 国, 首, 都, 是, 北, 京, EOS, -100]
           ↑    所有非 padding 位置都算 loss
```

SFT 时，数据是对话格式（system + user + assistant）。模型应该学"怎么回答"，不应该学"怎么提问"：

```
SFT:
input_ids: [<|im_start|>, system, ..., <|im_end|>, <|im_start|>, user, 中, 国, ..., <|im_end|>, <|im_start|>, assistant, 北, 京, <|im_end|>]
labels:    [   -100,       -100,  ...,   -100,        -100,      -100, -100,-100,...,  -100,        -100,       -100,    北, 京, <|im_end|>]
                                                                                                                          ↑
                                                                                                               只有 assistant 回复部分算 loss
```

**为什么只对 assistant 回复算 loss？** 三个原因：

1. **信息不对等**：用户指令是"输入"，模型不需要学习"怎么提问"。如果对用户指令也算 loss，模型会把一部分学习能力浪费在拟合"用户怎么说话"上。

2. **避免模式混淆**：模型学到的生成模式应该是"给定 context，生成 assistant 风格的回答"。如果也学了用户的说话模式，推理时可能混入用户的表达方式。

3. **梯度效率**：SFT 数据中用户指令和 system prompt 占了很大比例的 token，如果都算 loss，有效梯度信号被稀释。只对回复部分算 loss，每个 token 的梯度信号都是有用的。

#### loss mask 怎么生效——cross_entropy 的 ignore_index

label = -100 的位置在 loss 计算中被自动跳过：

```python
# Chapter 08 的 MiniMindForCausalLM.forward 中
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

PyTorch 的 cross_entropy 内置了 `ignore_index` 参数：
- $\text{label} = -100$ 的位置 → loss 贡献为 0，梯度也为 0
- $\text{label} \neq -100$ 的位置 → 正常计算 cross_entropy
- 最终 $\text{loss} = \dfrac{\sum_{\text{valid}} \text{CE}_i}{\, N_{\text{valid}} \,}$（mean reduction）

#### 和 DPO 的 loss mask 对比

后面 Chapter 14 会讲 DPO。DPO 也需要 loss mask，但用的是 0/1 的 `loss_mask` 而不是 -100——因为 DPO 的 loss 不用 cross_entropy 的 ignore_index，而是手动把 log_probs 乘以 mask 来实现选择性计算。

---

### Step 2: ChatML 格式——对话数据长什么样

#### 原始数据是 JSON 对话

预训练数据是纯文本：

```json
{"text": "中国的首都是北京。"}
```

SFT 数据是多轮对话：

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "中国的首都是哪里？"},
    {"role": "assistant", "content": "中国的首都是北京。"}
  ]
}
```

#### ChatML：把对话变成 token 序列

模型看到的不是 JSON，而是一串用特殊 token 分隔的文本。MiniMind 使用 ChatML 格式：

```
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
中国的首都是哪里？<|im_end|>
<|im_start|>assistant
中国的首都是北京。<|im_end|>
```

这个转换由 tokenizer 的 `apply_chat_template` 完成——它读取 tokenizer 配置里的 Jinja2 模板（`tokenizer_config.json` 中的 `chat_template` 字段），自动把 `[{role, content}]` 渲染成带特殊标记的纯文本。不用它的话就得手动拼字符串，换一个 tokenizer（比如 LLaMA 格式和 Qwen 的不一样）就得重写。

#### `add_generation_prompt` 参数

MiniMind 的模板末尾有这样一段逻辑：当 `add_generation_prompt=True` 时，在整个字符串末尾额外追加 `<|im_start|>assistant\n<think>...`，提示模型"该你说了，开始生成"。

- **推理时用 `True`**：只有用户的问题，没有 assistant 回复，需要在末尾加上提示让模型开始续写
- **SFT 训练时用 `False`**：assistant 回复已经在数据里了，模板渲染时自然会包含 `<|im_start|>assistant\n...内容...<|im_end|>`。如果还设 `True`，末尾会凭空多出一个 `<|im_start|>assistant\n`，把模型教歪

同一文件里的 `RLAIFDataset` 用的就是 `add_generation_prompt=True`，因为 RLHF/GRPO 需要模型自己生成回复。

---

### Step 3: 写 SFTDataset——构造 input_ids + loss mask

> 追加到: `scratch/dataset/lm_dataset.py`（在 PretrainDataset 之后）

SFTDataset 的核心任务：把多轮对话 JSON → ChatML 文本 → token 序列 + loss mask。以下 3.1-3.4 逐步构建这个类，每一步直接给出要写的代码。

#### 3.1 写 `__init__` + `__len__`——加载数据与边界标记

先写类骨架。和 PretrainDataset 相比，SFTDataset 的 `__init__` 多了两样东西：

1. **Features schema**：SFT 的 JSONL 每条有 `conversations` 字段（对话列表），每个对话可能有 `role`、`content`、`reasoning_content`、`tools`、`tool_calls` 等字段。用 `Features` 显式声明类型，防止 `load_dataset` 自动推断出错（比如 `None` 值字段被推断为 int）

2. **边界标记** `bos_id` 和 `eos_id`：用来在 token 序列中找到 assistant 回复的起止位置，后面 `generate_labels` 会用到

先在文件顶部补 import（Chapter 11 没用到 `Features`、`Value`、`json`、`random`）：

```python
# 补到 scratch/dataset/lm_dataset.py 顶部 import 区域
import json
import random
from datasets import load_dataset, Features, Value   # 替换原来的 from datasets import load_dataset
```

然后在 PretrainDataset 之后追加两个辅助函数（SFTDataset 的 `__getitem__` 会用到）：

```python
# 追加到: scratch/dataset/lm_dataset.py（在 PretrainDataset 类之后）

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """概率性添加 system prompt，让模型学会有/无 system 都能回答"""
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations):
        return conversations
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
    ]
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """以 80% 概率移除空思考标签——避免模型总是输出空 <think></think>"""
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content
```

接着写 SFTDataset 的 `__init__` 和 `__len__`：

```python
# 追加到: scratch/dataset/lm_dataset.py（在辅助函数之后）

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 显式定义 schema，防止 load_dataset 自动推断类型出错
        features = Features({
            'conversations': [{
                'role': Value('string'),
                'content': Value('string'),
                'reasoning_content': Value('string'),
                'tools': Value('string'),
                'tool_calls': Value('string')
            }]
        })
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 边界标记：用来在 token 序列中定位 assistant 回复的起止位置
        # bos_id: "<|im_start|>assistant\n" 的 token 序列——回复的起始标记
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        # eos_id: "<|im_end|>\n" 的 token 序列——回复的结束标记
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
```

#### 3.2 create_chat_prompt：对话 → ChatML 字符串

```python
# 写到: scratch/dataset/lm_dataset.py（SFTDataset 的方法）
def create_chat_prompt(self, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)           # 拷贝，避免修改原数据
        # 如果 system 消息里带了 tools 字段（Agent 场景），提取出来
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        # tool_calls 字段可能是 JSON 字符串，需要解析成 dict
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        messages.append(message)
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,              # 只渲染文本，不分词（分词由 __getitem__ 统一做）
        add_generation_prompt=False,  # SFT 训练不加生成提示
        tools=tools                   # 传给模板渲染 tool 描述
    )
```

#### 3.3 generate_labels：找出 assistant 回复的边界

逻辑：扫描 token 序列，每遇到一段 `bos_id...eos_id`，就把中间（含 eos_id）的 label 从 -100 改为真实 token id，其他位置保持 -100。

**用具体例子走一遍**：

假设多轮对话有两个 assistant 回复：

```
token 序列（简化表示，数字为位置编号）:

pos:  0     1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19
tok:  sys   你   是   助   手   eos  usr  首   都   eos  ast  北   京   eos  usr  天   气   eos  ast  晴

pos:  20
tok:  eos

其中:
  sys = <|im_start|>system\n   (对应 system 起始)
  usr = <|im_start|>user\n
  ast = <|im_start|>assistant\n  (= bos_id)
  eos = <|im_end|>\n            (= eos_id)

labels 构造:
pos:  0     1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20
lab: -100  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 [北] [京] [eos]-100 -100 -100 -100 -100 [晴] [eos]
                                                                    ↑                                   ↑
                                                               第一段 assistant 回复               第二段 assistant 回复
```

**理解了流程再看代码**：

```python
# 写到: scratch/dataset/lm_dataset.py（SFTDataset 的方法）
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)     # 默认所有位置不算 loss
    i = 0
    while i < len(input_ids):
        # 找到 assistant 回复的起始标记
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)     # 跳过 "<|im_start|>assistant\n"
            end = start
            # 找到回复的结束标记
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # 只把 assistant 回复部分的 label 设为真实 token id（含 eos_id）
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            # 跳到 eos_id 之后继续扫描下一段 assistant 回复
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

#### 3.4 \_\_getitem\_\_：完整的数据处理管道

```python
# 写到: scratch/dataset/lm_dataset.py（SFTDataset 的方法）
def __getitem__(self, index):
    sample = self.samples[index]
    conversations = pre_processing_chat(sample['conversations'])

    # 1. 把对话渲染成 ChatML 格式字符串
    prompt = self.create_chat_prompt(conversations)
    prompt = post_processing_chat(prompt)

    # 2. 分词 + 截断到 max_length
    input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

    # 3. Padding: 不够长的补 pad_token_id
    input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

    # 4. 生成 loss mask——只有 assistant 回复部分的 label 不是 -100
    labels = self.generate_labels(input_ids)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

完整流程图：

```
原始数据:  {"conversations": [{role: system, ...}, {role: user, ...}, {role: assistant, ...}]}
    ↓ pre_processing_chat（概率性添加 system prompt）
    ↓ create_chat_prompt（apply_chat_template 渲染）
ChatML:    "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n"
    ↓ post_processing_chat（概率性移除空思考标签）
    ↓ tokenizer（分词）
分词:      [token_id_0, token_id_1, ..., token_id_N]
    ↓ 截断 + Padding
截断+Pad:  [token_id_0, ..., token_id_N, PAD, PAD, ...]     长度 = max_length
    ↓ generate_labels（扫描 bos_id/eos_id 边界）
Labels:    [-100, ..., -100, tok_a1, tok_a2, ..., <|im_end|>, -100, ..., -100]
                             ↑ assistant 回复的内容 ↑
```

#### 3.5 多轮对话为什么一条序列就够——Teacher Forcing

上面的例子里，两轮 assistant 回复放在同一条序列里训练。你可能会想：模型预测 assistant1 的结果和语料不一样，那预测 assistant2 时前文不就错了吗？

答案是：**训练时模型的输入不是自己的预测，而是语料里的真实 token**。

**推理时（自回归）**：模型用自己的输出当下一步输入

```
step 1: 输入 [user1]           → 模型生成 "你"
step 2: 输入 [user1, 你]       → 模型生成 "好"
step 3: 输入 [user1, 你, 好]   → 模型生成 "啊"    ← 如果这里生成错了
step 4: 输入 [user1, 你, 好, 啊] → 后面全歪      ← 误差累积
```

**训练时（Teacher Forcing）**：输入永远是语料真值

```
一次前向传播，input_ids 就是完整的 ground truth:

位置:    0      1     2     3      4     5     6     7     8
input:  user1  你    好    user2  北    京    是    首    都
               ↑ 语料里 assistant1 的真实回复，不是模型生成的

模型在每个位置独立输出预测:
位置 0 → 预测位置 1（目标: "你"）
位置 1 → 预测位置 2（目标: "好"）
位置 2 → 预测位置 3（目标: user2）
位置 3 → 预测位置 4（目标: "北"）  ← 看到的是真实的 assistant1，不是模型自己编的
...
```

causal mask 让每个位置只能看前面、看不到后面，模拟了"一步一步生成"的效果，但所有位置**并行计算**，输入全是真值。

**和"拆成 n 条"有什么区别？** 直觉上 3 轮对话似乎应该拆成 3 条训练样本：

```
拆分式:
样本1: [user1]                                → 预测 [assistant1]
样本2: [user1, assistant1, user2]             → 预测 [assistant2]
样本3: [user1, assistant1, user2, assistant2, user3] → 预测 [assistant3]
```

这样做**数学上等价**（causal mask 保证了可见范围一样），但不这样做是因为：
- 前缀在每条里重复计算（user1 算了 3 遍）
- 数据膨胀 n 倍
- 一条序列 + label mask 更高效，效果完全一样

**一句话记住：训练不是让模型"推理"，而是给它真实答案问"你猜到了吗"。** 猜错了算 loss、更新参数，但下一个位置看到的输入仍然是语料里的正确答案——这就是 Teacher Forcing。

---

### Step 4: 写全参 SFT 训练脚本

> 写到: `scratch/trainer/train_full_sft.py`

全参 SFT 的训练循环和预训练**完全相同**——同样的 forward、同样的 loss.backward()、同样的 AdamW + 余弦退火 + 梯度裁剪。差异全在**数据层面**（SFTDataset 的 label mask）和**超参数**。

#### 和预训练的 diff

```
train_pretrain.py  vs  train_full_sft.py

数据集:     PretrainDataset         →  SFTDataset
数据路径:   pretrain_data.jsonl      →  sft_t2t_mini.jsonl
学习率:     5e-4                     →  1e-5（低 50 倍）
序列长度:   340                      →  768
batch_size: 32                       →  16
from_weight: none                    →  pretrain（加载预训练权重）
save_weight: pretrain                →  full_sft
```

#### 怎么写

因为训练循环和 Chapter 11 的 `train_pretrain.py` 完全一样，只需要修改三处：

```python
# 写到: scratch/trainer/train_full_sft.py

# 1. 导入 SFTDataset（替换 PretrainDataset）
from dataset.lm_dataset import SFTDataset

# 2. 修改默认超参数
parser.add_argument("--save_weight", default='full_sft', type=str)
parser.add_argument("--learning_rate", default=1e-5, type=float)      # 低 50 倍
parser.add_argument("--batch_size", default=16, type=int)              # 更小
parser.add_argument("--max_seq_len", default=768, type=int)            # 更长
parser.add_argument("--from_weight", default='pretrain', type=str)     # 加载预训练权重
parser.add_argument("--data_path", default='../dataset/sft_t2t_mini.jsonl', type=str)

# 3. 用 SFTDataset 创建数据集
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

其余所有代码（train_epoch 函数、混合精度、分布式、检查点保存等）与 `train_pretrain.py` 完全一致，直接复用。

#### 关键超参数

| 超参数 | 预训练 | 全参 SFT | 为什么变了 |
|--------|--------|---------|-----------|
| `learning_rate` | 5e-4 | 1e-5 | SFT 是微调，学习率太高会破坏预训练学到的知识（灾难性遗忘） |
| `max_seq_len` | 340 | 768 | 对话数据通常比纯文本长（包含 system + user + assistant） |
| `batch_size` | 32 | 16 | 序列更长，显存占用更大，需要减小 batch |
| `accumulation_steps` | 8 | 1 | 等效 batch = 16 x 1 = 16，SFT 数据量小不需要太大 batch |
| `from_weight` | none | pretrain | 必须从预训练权重开始，否则模型没有基础知识 |

#### 数据量级对比

```
预训练数据:    GB~TB 级，数百万~数十亿条
SFT 数据:      MB 级，1 万~100 万条
               → SFT 只需要 1~3 个 epoch
               → 训练时间从天/周缩短到小时
```

#### 全参 SFT 的显存分析

全参 SFT 意味着所有 64M 参数都参与训练。显存主要占用：

```
模型参数:         64M x 4 bytes (FP32) = 256 MB
                  或 64M x 2 bytes (BF16) = 128 MB

梯度:             和参数一样大 = 128~256 MB

AdamW 优化器状态: 每个参数需要存一阶动量 + 二阶动量
                  = 64M x 4 bytes x 2 = 512 MB（始终 FP32）

激活值:           取决于 batch_size x seq_len
                  MiniMind: 16 x 768 x 768 x 8层 ≈ 几百 MB

总计:             MiniMind 约 1~2 GB（64M 参数很小）
                  7B 模型: 约 112 GB 固定开销（P x 16，不含激活值；单卡 A100 80GB 放不下，需 ZeRO/多卡）
```

这就是为什么大模型通常用 LoRA 而不是全参 SFT——7B 全参 SFT 固定开销就要 112 GB，一张 A100 80GB 装不下。LoRA 在一张 24GB RTX 4090 上就能跑。

#### 完整的 train_full_sft.py

以下是基于 Chapter 11 的 `train_pretrain.py` 改写的完整文件，可直接运行：

```python
# 新建: scratch/trainer/train_full_sft.py

import os, sys, time, argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset                   # ← 替换 PretrainDataset
from transformers import AutoTokenizer
from trainer_utils import get_lr, Logger


def train_epoch(model, loader, optimizer, scaler, autocast_ctx, epoch, args):
    model.train()
    iters = len(loader)
    start_time = time.time()

    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        current_lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            real_loss = loss.item() * args.accumulation_steps
            eta_min = (time.time() - start_time) / step * (iters - step) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                   f'loss:{real_loss:.4f} lr:{current_lr:.8f} eta:{eta_min:.1f}min')

        if step % args.save_interval == 0 or step == iters:
            model.eval()
            ckp = f'{args.save_dir}/full_sft_{args.hidden_size}.pth'
            torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, ckp)
            model.train()

        del input_ids, labels, res, loss

    if iters % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="../../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)             # ← 比预训练小
    parser.add_argument("--learning_rate", type=float, default=1e-5)      # ← 比预训练低 50 倍
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=1)      # ← SFT 数据量小，不需要大 batch
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)           # ← 比预训练长（对话更长）
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--data_path", default="../../dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--from_weight", default="pretrain")              # ← 加载预训练权重
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(lm_config).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained('../../model')

    # ← 新增：加载预训练权重（预训练脚本没有这一步）
    if args.from_weight != 'none':
        ckp = f'{args.save_dir}/{args.from_weight}_{args.hidden_size}.pth'
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
        Logger(f'Loaded weights from {ckp}')

    Logger(f'Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)  # ← SFTDataset
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (nullcontext() if "cpu" in args.device
                    else torch.amp.autocast('cuda', dtype=dtype))
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))

    for epoch in range(args.epochs):
        train_epoch(model, loader, optimizer, scaler, autocast_ctx, epoch, args)
```

和 `train_pretrain.py` 相比，只改了 4 处（用 `←` 标注）：导入 `SFTDataset`、调低超参数、加载预训练权重、用 `SFTDataset` 创建数据集。训练循环本身一行没改。

---

### Step 5: LoRA 的数学原理——低秩分解

#### 为什么需要 LoRA

全参 SFT 更新所有参数，对于 MiniMind 这种 64M 小模型还可以接受，但放到大模型场景：

```
模型规模       全参 SFT 显存（固定开销）   LoRA SFT 显存       需要的硬件
7B            ~112 GB                 ~16 GB              2xA100 80GB → RTX 4090 24GB
13B           ~208 GB                 ~24 GB              3xA100 → 1xA100
70B           ~1120 GB                ~80 GB              14xA100 → 1xA100
```

（全参列 = P x 16 固定开销口径，不含激活值——激活值随 batch x seq 另算；硬件按 A100 80GB 估算，全参 SFT 实际还要配 ZeRO 切分。LoRA 列 ≈ BF16 权重 + LoRA 参数/梯度/优化器。）

除了显存，全参 SFT 还有一个实际问题：**每个下游任务保存一份完整模型权重**。7B 模型 = 14GB 权重文件，10 个任务 = 140GB。LoRA 只保存增量权重，7B 模型的 LoRA 权重通常只有 10~50MB。

#### LoRA 的直觉

Aghajanyan et al. (2020) 发现：预训练模型虽然参数量巨大，但微调时的权重变化 $\Delta W$ 是**低秩**的——也就是说，$\Delta W$ 的有效信息可以用一个远小于原始维度的矩阵来表示。

LoRA (Hu et al., 2021) 把这个观察变成了方法：

```
原始:  y = Wx                    W: [d_out, d_in]

LoRA:  y = Wx + BAx              W: [d_out, d_in]  ← 冻结，不训练
                                 A: [rank, d_in]    ← 训练
                                 B: [d_out, rank]   ← 训练

rank << d_in, d_out
```

用 MiniMind 的实际参数算一下：

```
W: [768, 768]   = 589,824 个参数
A: [16, 768]    = 12,288 个参数
B: [768, 16]    = 12,288 个参数
LoRA 参数:       24,576 (占 W 的 4.2%)
```

#### 为什么低秩就够了

一个 $[768, 768]$ 的权重矩阵可以表示 768 维空间到 768 维空间的任意线性变换。但微调时，权重的变化 $\Delta W$ 通常只需要调整几个"方向"——比如让模型更关注医疗术语、调整回答风格。

```
ΔW 的秩 = ΔW 中独立变化方向的数量

rank=16 意味着 ΔW 最多有 16 个独立方向
对于从"通用回答"到"医疗领域回答"的适配，16 个方向通常足够

类比: 768 维空间中，调整"说话风格"可能只需要改几个主成分方向
      就像 PCA 降维——大部分变化被前几个主成分捕获
```

#### $\alpha/r$ 缩放（MiniMind 没实现）

标准 LoRA（Hu et al., 2021）的完整形式是：

$$\Delta W = \frac{\alpha}{r} \cdot BA$$

$\alpha$ 是缩放系数（常见默认 $\alpha=16$ 或 $\alpha=2r$），作用是**让 LoRA 的有效更新幅度不随 rank 变化**——调 rank 时不用重新调学习率。MiniMind 直接输出 $BA$，等效于固定 $\alpha = r$（缩放恒为 1），rank 改大时更新幅度会同比变大，学习率要自己留意。**面试常问"$\alpha$ 的作用"**：归一化更新幅度，把 rank 和学习率调参解耦。

#### 初始化策略："A 高斯，B 全零"

训练开始时，$B \times A$ 的输出 = 0（因为 B 全零），所以 LoRA 模块不会改变原始模型的输出。这是一个关键设计——**确保微调从预训练模型的原始状态开始**，而不是被随机初始化扰乱。

```
训练开始:  y = Wx + BAx = Wx + 0 = Wx      ← 和原始模型完全一样
训练过程:  B 和 A 逐渐被更新              ← 逐渐学习任务特定的调整
训练结束:  y = Wx + BAx                    ← BA 学到了"如何从原始输出调整到期望输出"
```

如果 A 和 B 都随机初始化，训练开始时 $BAx$ 就不是 0，模型一开始就偏离了预训练状态，可能导致训练不稳定。

#### rank 的选择

```
rank    参数量(每层)    效果          适用场景
4       6,144         较差          极度资源受限
8       12,288        可用          简单任务适配
16      24,576        标准选择      大部分场景
32      49,152        更好          复杂任务
64      98,304        接近全参      需要大幅调整的场景

MiniMind 默认 rank=16
业界常见: rank=8~64，LLaMA 系列通常用 rank=16 或 32
```

rank 太小 → 表达能力不够，欠拟合
rank 太大 → 参数量增加，LoRA 的优势减弱，接近全参

---

### Step 6: 从零写 LoRA 模块

> 写到: `scratch/model/model_lora.py`（新建文件）

#### 6.1 LoRA 类

```python
# 写到: scratch/model/model_lora.py
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        # A: 降维投影 [d_in] → [rank]，高斯初始化提供梯度方向
        self.A = nn.Linear(in_features, rank, bias=False)
        # B: 升维投影 [rank] → [d_out]，全零初始化确保训练开始时 BA=0
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        # x: [B, S, d_in] → A(x): [B, S, rank] → B(A(x)): [B, S, d_out]
        return self.B(self.A(x))
```

#### 6.2 哪些层需要加 LoRA

不是所有 Linear 层都需要 LoRA。MiniMind 用一个简单的筛选条件：`in_features == out_features`（方阵 Linear 层）。在 MiniMind 中：

```
Attention:
  wq:  [768, 768]  ← 加 LoRA (方阵)
  wk:  [768, 384]  ← 不加（GQA，4 KV 头 x 96 head_dim = 384）
  wv:  [768, 384]  ← 不加（GQA，V 头数少）
  wo:  [768, 768]  ← 加 LoRA (方阵)

FeedForward:
  w1:  [768, 2432] ← 不加（非方阵）
  w2:  [2432, 768] ← 不加（非方阵）
  w3:  [768, 2432] ← 不加（非方阵）

每层 LoRA 参数 = 2 x (768 x 16 + 16 x 768) = 2 x 24,576 = 49,152
8 层总计 = 8 x 49,152 = 393,216 ≈ 0.39M
```

#### 6.3 apply_lora：用 monkey-patch 注入

`apply_lora` 使用 monkey-patch（运行时替换方法）来修改 Linear 层的 forward：

```
修改前:  module.forward(x) → x @ W.T + bias
修改后:  module.forward(x) → x @ W.T + bias + B(A(x))
```

```python
# 写到: scratch/model/model_lora.py
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        # 只对方阵 Linear 加 LoRA（in == out，即 Attention 的 Q 和 O 投影）
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            # setattr 把 lora 挂到 module 上，后续可以通过 module.lora 访问
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 关键: 用默认参数绑定闭包
            # 如果不用 layer1=original_forward, layer2=lora，
            # 而是在函数体内直接引用外层变量 original_forward 和 lora，
            # 所有层的 forward 会指向最后一次循环的值（Python 闭包陷阱）
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)        # y = Wx + BAx

            module.forward = forward_with_lora
```

**注意**：monkey-patch 和 `torch.compile` 不兼容——`torch.compile` 假设 forward 方法不会被运行时替换。所以 LoRA 训练中 `use_compile` 要关闭。

---

### Step 7: 写 LoRA 训练脚本

> 写到: `scratch/trainer/train_lora.py`

LoRA 训练脚本基于全参 SFT 脚本，但有三个关键区别：

#### 7.1 参数冻结与优化器

```python
# 写到: scratch/trainer/train_lora.py（模型初始化部分）

# 先加载模型并注入 LoRA
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
apply_lora(model)

# 冻结非 LoRA 参数，收集 LoRA 参数
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True      # LoRA 参数：可训练
        lora_params.append(param)
    else:
        param.requires_grad = False     # 原始参数：冻结

# 优化器只接收 LoRA 参数——不冻结的话 AdamW 也会为冻结参数分配动量/方差，浪费显存
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

冻结非 LoRA 参数有两个作用：
1. **不计算梯度** → 节省反向传播的计算量和显存
2. **不更新权重** → 预训练知识完全保留，天然抗灾难性遗忘

#### 7.2 梯度裁剪只作用于 LoRA 参数

```python
# 全参 SFT:
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

# LoRA SFT:
torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
#                               ↑ 不是 model.parameters()
```

#### 7.3 保存时只存 LoRA 权重

```python
# 全参 SFT:
torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

# LoRA SFT（调用 Step 8 的 save_lora）:
save_lora(model, lora_save_path)
```

#### 7.4 自动关闭 torch.compile

```python
# 写到: scratch/trainer/train_lora.py（编译和分布式包装部分）
if args.use_compile == 1:
    args.use_compile = 0
    Logger('[LoRA] monkey-patch forward 与 torch.compile 不兼容，use_compile 已自动关闭')
```

#### 7.5 超参数对比

| 超参数 | 全参 SFT | LoRA SFT |
|--------|---------|----------|
| `learning_rate` | 1e-5 | 1e-4（高 10 倍） |
| `batch_size` | 16 | 32 |
| `epochs` | 2 | 10 |
| `max_seq_len` | 768 | 340 |
| `from_weight` | pretrain | full_sft |
| `save_weight` | full_sft | lora_medical |

LoRA 学习率比全参 SFT 高 10 倍——这是因为 LoRA 参数量少，梯度信号需要更大的学习率才能在有限参数上产生足够的调整。epoch 数也更多（10 vs 2），因为参数少、数据小，需要更多遍训练。

注意 MiniMind 的 LoRA 是从 `full_sft` 权重开始的（不是从 pretrain），因为 LoRA 的典型应用场景是在一个已经 SFT 过的通用模型上做领域适配（如医疗 LoRA、法律 LoRA）。

#### 全参 vs LoRA：怎么选

| 维度 | 全参 SFT | LoRA |
|------|---------|------|
| 效果 | 更好（所有参数自由调整） | 性价比高（接近全参 90%+ 效果） |
| 显存 | 高（参数 + 梯度 + 优化器状态全量） | 低（只存 LoRA 参数的梯度和优化器状态） |
| 权重存储 | 每个任务一份完整权重 | 每个任务只存增量（~1MB vs ~128MB） |
| 灾难性遗忘 | 需要低学习率控制 | 天然抗遗忘（大部分参数冻结） |
| 适用场景 | 数据量大 + 资源充足 + 追求最优效果 | 数据量小 + 资源有限 + 多任务适配 |

#### 完整的 train_lora.py

以下是基于 `train_full_sft.py` 改写的完整文件，可直接运行：

```python
# 新建: scratch/trainer/train_lora.py

import os, sys, time, argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, save_lora                       # ← 新增
from dataset.lm_dataset import SFTDataset
from transformers import AutoTokenizer
from trainer_utils import get_lr, Logger


def train_epoch(model, loader, optimizer, scaler, autocast_ctx, epoch, lora_params, args):
    """和全参 SFT 的唯一区别：梯度裁剪作用于 lora_params 而不是 model.parameters()"""
    model.train()
    iters = len(loader)
    start_time = time.time()

    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        current_lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)  # ← lora_params
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            real_loss = loss.item() * args.accumulation_steps
            eta_min = (time.time() - start_time) / step * (iters - step) // 60
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                   f'loss:{real_loss:.4f} lr:{current_lr:.8f} eta:{eta_min:.1f}min')

        if step % args.save_interval == 0 or step == iters:
            model.eval()
            ckp = f'{args.save_dir}/{args.lora_name}_{args.hidden_size}.pth'
            save_lora(model, ckp)                                        # ← 只存 LoRA 权重
            model.train()

        del input_ids, labels, res, loss

    if iters % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="../../out")
    parser.add_argument("--lora_name", default="lora_medical")            # ← LoRA 权重名称
    parser.add_argument("--epochs", type=int, default=10)                 # ← 更多轮
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)      # ← 比全参 SFT 高 10 倍
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)           # ← 更频繁打日志
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--data_path", default="../../dataset/lora_medical.jsonl")
    parser.add_argument("--from_weight", default="full_sft")              # ← 从 SFT 权重开始
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    model = MiniMindForCausalLM(lm_config).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained('../../model')

    # 加载 SFT 权重
    ckp = f'{args.save_dir}/{args.from_weight}_{args.hidden_size}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    Logger(f'Loaded weights from {ckp}')

    # ← 新增：注入 LoRA
    apply_lora(model)

    # ← 新增：冻结非 LoRA 参数，收集 LoRA 参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    lora_count = sum(p.numel() for p in lora_params)
    Logger(f'Total: {total_params/1e6:.2f}M, LoRA: {lora_count/1e6:.2f}M ({lora_count/total_params*100:.1f}%)')

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    # ← 优化器只接收 LoRA 参数
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (nullcontext() if "cpu" in args.device
                    else torch.amp.autocast('cuda', dtype=dtype))
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))

    for epoch in range(args.epochs):
        train_epoch(model, loader, optimizer, scaler, autocast_ctx, epoch, lora_params, args)
```

和 `train_full_sft.py` 相比的区别（用 `←` 标注）：导入 `apply_lora`/`save_lora`、注入 LoRA 并冻结非 LoRA 参数、优化器只接收 `lora_params`、梯度裁剪作用于 `lora_params`、保存时只存 LoRA 权重。

---

### Step 8: LoRA 权重的保存/加载/合并

> 写到: `scratch/model/model_lora.py`（在 LoRA 类和 apply_lora 之后追加）

#### 8.1 保存：只存 LoRA 权重

```python
# 写到: scratch/model/model_lora.py
def save_lora(model, path):
    # getattr 兼容 torch.compile 包装：compile 后模型被包了一层 _orig_mod
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 清理 DDP 前缀 "module."
            clean_name = name[7:] if name.startswith("module.") else name
            # .cpu().half() 节省存储空间
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half()
                         for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

只保存带 `lora` 属性的模块的参数。64M 模型的全参权重 ~128MB，LoRA 权重 ~1MB。

#### 8.2 加载：把 LoRA 权重恢复到 monkey-patch 后的模型

```python
# 写到: scratch/model/model_lora.py
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    # 兼容 DDP 保存的权重：去掉 "module." 前缀
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 从全局 state_dict 中筛选出属于当前 module 的 LoRA 参数
            # 并把 key 从 "layers.0.attention.wq.lora.A.weight" 简化为 "A.weight"
            lora_state = {k.replace(f'{name}.lora.', ''): v
                         for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
```

#### 8.3 合并：把 LoRA 权重融入原始权重

训练完成后，可以把 LoRA 的增量权重永久融入原始模型，推理时不再需要 LoRA 模块：

```python
# 写到: scratch/model/model_lora.py
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    # 先提取所有非 LoRA 参数
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # W_merged = W + B @ A
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
```

合并的数学：

$$W_{\text{new}} = W + BA$$

$$y = W_{\text{new}} x = (W + BA)x = Wx + BAx$$

合并后的模型和原始模型结构完全一样（没有 LoRA 模块），推理时没有额外开销。

---

## SFT 训练监控（面试高频）

> 网易云一面原题："SFT loss 一般没有太大意义，因为它一定会降。那有没有别的指标可以帮助我们侧面关注一下训练的状态？"
> 以及："每个 epoch 结束的时候 loss 都会有一个大的下降，怎么理解这个大的下降？"

### 为什么 SFT loss "没有意义"

预训练的 loss 是有意义的——数据量大（几 T tokens）、训练时间长，loss 曲线能反映模型是否在持续学到新知识。但 SFT 的场景完全不同：

1. **数据量极小**（几千~几万条），模型几乎一定能"背下来" → loss 一定会降
2. **Loss 降不代表模型变好**：loss 从 1.5 降到 0.3，可能只是模型记住了训练样本的回答方式，泛化能力反而变差
3. **过拟合的 loss 也在降**：模型开始复读训练数据、丧失多样性，但 train loss 看起来一切正常

所以光看 SFT train loss 曲线是"假信号"——它只能告诉你训练在跑、没有 NaN，不能告诉你模型在变好。

### Epoch 边界 Loss 突降的原因

SFT 训 2~3 个 epoch 时，每个 epoch 结束、下一个 epoch 开始的瞬间，loss 会有一个明显的下跌。原因：

```
Epoch 1: 模型第一次见到所有样本 → 逐渐学会
          epoch 结束时，loss 已经在低位

Epoch 2 开始: DataLoader reshuffle 后，模型再次见到同样的数据
              但这些数据它已经在 epoch 1 学过了
              → 第一个 batch 的 loss 就比 epoch 1 末尾更低
              → 看起来像"突然下降"

本质: 模型在第二遍看到已学过的数据，记忆效应 → loss 立刻比上个 epoch 末还低
```

这个现象本身是正常的，但如果每个 epoch 的 loss 下降幅度越来越大、最终接近 0，那就是严重过拟合。

### 该看什么指标

| 指标 | 含义 | 怎么看 | 工具 |
|------|------|--------|------|
| **Eval Loss** | 留出 10~20% 数据做验证集的 loss | 和 train loss 对比：eval loss 开始上升 = 过拟合 | TensorBoard / WandB |
| **Gradient Norm** | 每步所有参数梯度的 L2 范数 | 突然飙升 = 训练不稳定，可能需要降 lr 或 clip grad；趋近 0 = 学不动了 | `torch.nn.utils.clip_grad_norm_` 返回值 |
| **Learning Rate** | 当前步的实际学习率 | 确认 scheduler 在正常工作，warmup → cosine decay 曲线符合预期 | `scheduler.get_last_lr()` |
| **Token-Level Accuracy** | 预测正确的 token 比例（只算 assistant 部分） | 比 loss 更直觉——90% 以上开始要注意过拟合 | 自己写：`(logits.argmax(-1) == labels).float().mean()` |
| **生成质量采样** | 定期拿固定 prompt 生成回复，人工看 | 最直接的信号：回复是否多样、流畅、没有复读 | 每隔 N 步跑 `model.generate()` |
| **输出熵** | 模型在词表上的概率分布有多分散（Shannon entropy，不是交叉熵） | 熵持续下降 = 模型越来越"自信" = 输出多样性在丧失。如果后续要接 RL（GRPO），这是最关键的预警——详见下面"SFT 何时该停 → 切 RL" | 每隔 N 步对固定 prompt 集做一次前向传播，算平均 token 熵 |

**最关键的实操建议**：

1. **一定要有 eval set**。SFT 数据少，哪怕只留 200 条做 eval 也比没有强。eval loss 上升是最可靠的过拟合信号
2. **Gradient norm 是训练稳定性的晴雨表**。如果用了 gradient clipping（MiniMind 用的 `grad_clip=1.0`），实际 gradient norm > clip 值的频率也值得监控——太频繁说明 lr 太大或数据有异常样本
3. **生成采样是唯一能看到"模型到底学成什么样"的指标**。loss/accuracy 都是代理指标，只有看实际生成的文本才知道模型是不是在往对的方向走

### SFT 何时该停 → 切 RL

这也是面试高频问题。2025-2026 年的最新研究颠覆了传统做法——**SFT 表现最好的 checkpoint 往往是 RL 最差的起点**。

#### 传统做法（2024 及以前）

判断 SFT 到瓶颈的经典信号：

1. **Eval loss 平了或开始涨**：eval loss 涨 = 模型在背训练数据而不是学泛化能力。train loss 继续降只是记忆效应，模型对没见过的问题反而答得更差了——典型过拟合信号

2. **Pass@k 指标**：pass@k 来自 Codex 论文（Chen et al., 2021），定义：对同一道题采样 k 次，至少 1 次答对的概率。pass@1 衡量**一致性**（每次都能做对吗），pass@10 衡量**能力上限**（有没有能力做对）。

   用 pass@10 判断是否该切 RL 的逻辑链：
   - 如果你的模型 pass@10 高但 pass@1 低 → 模型知道正确答案（10 次里能答对），但不总能选对 → **差距在一致性，不在知识**
   - RL（如 GRPO）的核心机制就是"强化好的、惩罚差的"：对同一题采样一组 rollout，答对的得正 advantage，答错的得负 advantage → **把已有的能力推到稳定输出**
   - 反过来，如果 pass@10 也很低 → 模型根本不知道怎么做 → GRPO 采样全是错的，无正 advantage → RL 学不到东西 → 该继续 SFT 加数据
   - 一句话：**RL 不教新知识，它只提升已有知识的命中率**

   和强模型对标（如 pass@10 ≈ DeepSeek-R1 的 pass@1）是判断"能力上限是否够用"的参考线——如果你 10 次的上限都不如人家 1 次的常态，说明知识还不够，先补 SFT

3. **生成多样性下降**：模型的回复开始高度相似、套路化 → SFT 已经过头了。判断方法：
   - **人眼看**：固定 prompt 集生成，看回复是否套路化、结构趋同
   - **输出熵**：Shannon entropy 持续下降 = 概率分布越来越集中
   - **Self-BLEU**：对同一 prompt 生成多条回复，两两算 BLEU → 值越高说明回复越相似
   - **Response 长度方差**：方差塌缩 = 模型在用固定模板回答

大多数团队的做法是：SFT 训 1-3 epochs → 选 eval loss 最低或 pass@1 最高的 checkpoint → 切 RL。

#### 新发现：Rank Inversion（排名反转）

[SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR](https://arxiv.org/pdf/2606.18487)（2026）发现了一个反直觉的现象：

**选 pass@1 最高的 SFT checkpoint 做 GRPO 初始化，在每个种子、每个深度上都选到了最差的 GRPO 起点**——差距高达 0.325 pass@10。

具体来说：实验中 SFT 训练过程中保存了多个 checkpoint，按 pass@1 从高到低排序是 A > B > C > D > E。传统做法选 A（SFT 表现最好的）。但从每个 checkpoint 出发跑 GRPO 后，**排名完全反过来了**：E 的 GRPO 后 pass@10 最高，A 最低，差距达 0.325。这就是"rank inversion"——SFT 的排名和 GRPO 后的排名是反的。"在每个种子、每个深度上"意味着换随机种子、换模型大小，结论都一样：这不是偶然，而是结构性规律。

原因是 **entropy collapse（熵坍缩）**。SFT 训得越久，模型输出分布越集中（熵越低），采样多样性越低。当 GRPO 做组内采样时，所有 rollout 都差不多，advantage 信号消失，RL 学不动了。

```
SFT 训得少:  模型输出分布分散，熵高
             → GRPO rollout 多样: [好回答, 差回答, 中回答, 好回答]
             → 组内有明确的好坏对比 → advantage 信号清晰 → RL 学得动

SFT 训得多:  模型输出分布集中，熵低
             → GRPO rollout 同质: [标准回答, 标准回答, 标准回答, 标准回答]
             → 组内无法区分好坏 → advantage ≈ 0 → RL 学不动
```

#### 输出熵（Shannon Entropy）——不是交叉熵

这里说的"熵"是模型输出分布自身的 **Shannon 熵**，和训练用的交叉熵 loss 是完全不同的东西：

```
交叉熵 (cross-entropy):  H(p, q) = -Σ p(x) log q(x)
  → 需要真实标签 p（one-hot），衡量"模型预测 vs 正确答案"的差距
  → 用途：训练 loss

输出熵 (Shannon entropy): H(q) = -Σ q(x) log q(x)
  → 不需要标签，只看模型自己在词表上的概率分布 q
  → 用途：衡量模型有多"犹豫" / 回复有多多样
```

用代码说清楚：

```python
import torch
import torch.nn.functional as F

# 假设模型在某个位置的 logits: [vocab_size]
logits = model_output.logits[:, -1, :]           # 取最后一个位置的 logits

# 交叉熵（需要标签）
label = torch.tensor([target_token_id])
cross_entropy = F.cross_entropy(logits, label)    # 标量，衡量"预测 vs 真值"

# 输出熵（不需要标签）
probs = F.softmax(logits, dim=-1)                 # softmax → 概率分布
output_entropy = -(probs * probs.log()).sum(dim=-1)  # Shannon 熵，标量
# 或等价地: output_entropy = torch.distributions.Categorical(probs=probs).entropy()
```

| | 交叉熵 | 输出熵 |
|---|--------|--------|
| 公式 | $H(p, q) = -\sum p_i \log q_i$ | $H(q) = -\sum q_i \log q_i$ |
| 需要标签？ | 需要 | 不需要 |
| 衡量什么 | 预测和真值的差距 | 输出分布的分散程度 |
| 值域 | $[0, +\infty)$ | $[0, \log V]$（$V$ = 词表大小） |
| 低值含义 | 预测越准 | 模型越确定（分布越集中） |
| 高值含义 | 预测越差 | 模型越犹豫（分布越分散） |

**实操：怎么算一个 checkpoint 的"输出熵"**

```python
def compute_checkpoint_entropy(model, eval_prompts, tokenizer, num_samples=8):
    """对固定 prompt 集生成多次，计算平均 token 级输出熵"""
    entropies = []
    model.eval()
    with torch.no_grad():
        for prompt in eval_prompts:
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
            # 前向传播拿到所有位置的 logits
            logits = model(input_ids).logits          # [1, seq_len, vocab_size]
            probs = F.softmax(logits, dim=-1)
            # 每个位置的 Shannon 熵
            token_entropy = -(probs * probs.log()).sum(dim=-1)   # [1, seq_len]
            entropies.append(token_entropy.mean().item())
    return sum(entropies) / len(entropies)
```

代价很低——只需一次前向传播，不需要生成（没有自回归解码的开销）。

#### 为什么 Over-SFT 会让 RL 失败：Plasticity Loss

[When RL Fails after SFT](https://arxiv.org/pdf/2606.09932)（2026）进一步揭示了机制：过度 SFT 不仅是"过拟合"那么简单，它让模型**丧失了被 RL 继续改进的能力**（plasticity loss）：

| 问题 | 机制 | 后果 |
|------|------|------|
| **Entropy collapse** | 输出分布被压缩，多样性消失 | GRPO 的组内 advantage 信号消失 |
| **Plasticity loss** | 模型参数被锁死在 SFT 的模式里，对 RL 梯度不敏感 | RL 更新无法有效转化为性能提升 |
| **Distributional forgetting** | 模型偏离 base model 的分布太远 | 即使没有传统意义上的"过拟合"（eval loss 还在降），探索能力也已丧失 |

关键洞察：**SFT 需要的训练量（学会指令跟随格式）和 RL 需要的最佳起点（保留探索能力）往往不一致。**

#### 新的 Checkpoint 选择标准

传统标准（eval loss / pass@1）测量的是"模仿质量"，不是"被 RL 改进的潜力"。

| 选择标准 | 测量什么 | 选 RL 起点？ |
|---------|---------|------------|
| **Eval loss 最低** | 模仿 SFT 数据的质量 | ❌ 可能选到过度拟合的 checkpoint |
| **Pass@1 最高** | 单次采样正确率 | ❌ 会选到熵最低的 checkpoint（rank inversion） |
| **Pass@10 最高** | 采样 10 次至少对一次的概率 | ✅ 直接衡量 GRPO 能否采到正样本，但计算成本高（10 次采样 + 评测） |
| **输出熵最高** | 模型回复的多样性 | ✅ 与 pass@10 高度相关，且只需一次前向传播——实操中常作为 pass@10 的廉价代理 |
| **Self-BLEU 最低** | 不同回复之间的重复度 | ✅ 低 self-BLEU = 高多样性 |

为什么 pass@10 和输出熵都有效？GRPO 需要组内 rollout 既有对的（正 advantage）又有错的（负 advantage）。pass@10 直接衡量"能不能采到对的"，输出熵衡量"分布够不够宽"——两者在实际 checkpoint 之间高度相关（分布宽 → 更容易碰到正确路径）。论文用熵做指标，主要因为计算成本低得多（一次前向 vs 10 次采样 + 评测），而不是因为熵严格优于 pass@10。

实验证据：用 entropy triage 替代 pass@1 选 checkpoint，GRPO 后 pass@10 提升了 +0.090（0.643 vs 0.553），代价只是多做一次前向传播。

#### 业界最新方案

| 方案 | 思路 | 来源 |
|------|------|------|
| **Entropy triage** | 选熵最高（而非 pass@1 最高）的 SFT checkpoint | [Entropy Collapse 论文](https://arxiv.org/pdf/2606.18487) |
| **Lightweight SFT** | 刻意少训 SFT（1 epoch 甚至更少），保留探索能力 | [OpenReview 2025](https://openreview.net/forum?id=yezWGJmODg) |
| **Model rejuvenation** | SFT 训完后用 model fusion + neuron reset 恢复可塑性 | [When RL Fails after SFT](https://arxiv.org/pdf/2606.09932) |
| **Adaptive interleaving** | 不硬切，而是根据 rollout 表现动态切换 SFT/RL | BRIDGE / SRFT / SASR |
| **Entropy flow optimization** | 在 GRPO 中加熵正则化，防止 RL 阶段的熵坍缩 | [ACL Findings 2026](https://aclanthology.org/2026.findings-acl.879.pdf) |

#### 实操建议（综合传统经验和新发现）

```
1. SFT 训 1-2 epochs（不超过 3），每隔 N 步保存 checkpoint
2. 对每个 checkpoint 算两个指标：
   - pass@1（或 eval loss）：确认模型已学会基本格式（IFEval > 某阈值）
   - pass@10 或输出熵：衡量 GRPO 的改进潜力
     - pass@10 更直接（能不能采到正样本），但需要 10 次采样 + 评测
     - 输出熵是 pass@10 的廉价代理，一次前向传播即可，实操中优先用熵
3. 选 checkpoint 的原则：
   - 先过格式门槛（eval loss 够低、IFEval 达标）
   - 在通过门槛的 checkpoint 中，选 pass@10 最高或熵最高的那个
4. 切 RL
```

一句话总结：**SFT 该停的时机不是"模型学到最好"的时候，而是"刚好学会格式但还保有探索能力"的时候。**

> **核心理念：评测先行。** 没有评测体系就不知道模型好没好、该不该切阶段。评测 > 数据 > 模型——先搭评测，再做数据，最后才是训练模型。

---

## 完整实现

> 以下 4 个文件的完整代码，可直接写到 `scratch/` 对应路径。

### 文件 1: SFTDataset

> 追加到: `scratch/dataset/lm_dataset.py`（在 PretrainDataset 之后）

```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({
            'conversations': [{
                'role': Value('string'),
                'content': Value('string'),
                'reasoning_content': Value('string'),
                'tools': Value('string'),
                'tool_calls': Value('string')
            }]
        })
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

### 文件 2: model_lora.py

> 新建: `scratch/model/model_lora.py`

```python
import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v
                         for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half()
                         for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
```

验证：

```python
# 验证 LoRA 模块
import torch
from torch import nn

lora = LoRA(768, 768, rank=16)
x = torch.randn(2, 10, 768)
out = lora(x)
assert out.shape == (2, 10, 768)

# 验证初始化: B 全零 → 输出全零
assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)
print("LoRA module verified!")

# 验证 apply_lora
model = nn.Sequential(nn.Linear(768, 768), nn.Linear(768, 768))
model.device = torch.device('cpu')
x_before = model(x).clone()
apply_lora(model)
x_after = model(x)
# apply_lora 后因为 B=0，输出应该不变
assert torch.allclose(x_before, x_after, atol=1e-6)
print("apply_lora verified!")
```

### 文件 3: train_full_sft.py

> 新建: `scratch/trainer/train_full_sft.py`

完整代码见 Step 4 中"完整的 train_full_sft.py"部分，可直接复制。

### 文件 4: train_lora.py

> 新建: `scratch/trainer/train_lora.py`

完整代码见 Step 7 中"完整的 train_lora.py"部分，可直接复制。

---

## 对照上游源码

| 我们写的 | 上游文件 |
|---------|---------|
| SFTDataset（SFT 数据格式、ChatML、loss mask） | `dataset/lm_dataset.py` 的 SFTDataset 类 |
| 全参 SFT 训练脚本 | `trainer/train_full_sft.py` |
| LoRA 模块（LoRA 类、apply_lora、save/load/merge） | `model/model_lora.py` |
| LoRA 训练脚本 | `trainer/train_lora.py` |

我们的实现与上游完全对齐。训练循环中用到的工具函数（`get_lr`、`Logger`、`init_model`、`SkipBatchSampler` 等）来自 `trainer/trainer_utils.py`，在 Chapter 11 已实现。

---

## 业界对比

### 全参 SFT

| 特性 | MiniMind (64M) | Qwen2.5 | LLaMA 3 | DeepSeek-V3 |
|------|---------------|---------|---------|-------------|
| SFT Loss | 只算 assistant | 只算 assistant | 只算 assistant | 只算 assistant |
| Chat Template | ChatML | ChatML | LLaMA 格式 | ChatML 变体 |
| SFT 学习率 | 1e-5 | 7e-6~2e-5 | 1e-5~2e-5 | ~1e-5 |
| SFT 数据量 | ~几万条 | ~100 万条 | ~1000 万条 | ~150 万条 |
| SFT epoch | 2 | 2~3 | 2 | 2 |
| 序列长度 | 768 | 32768 | 8192 | 8192 |
| 特殊处理 | 无 | 长短样本混合打包 | Packing + document_attention_mask | Packing；多轮 tool_call 也参与 loss |

几个值得注意的点：

1. **Loss mask 是全行业标准做法**——所有主流 LLM 的 SFT 都只对 assistant 回复算 loss
2. **Packing（样本打包）**：大模型会把多个短对话打包到一个序列里，避免大量 padding 浪费算力。MiniMind 没做这个优化
3. **SFT 数据质量远比数量重要**——Alpaca 论文用 52K 条高质量数据就训出了不错的效果
4. **SFT epoch 通常很少**（1~3 个），多了容易过拟合，表现为模型过度模仿训练数据的说话风格

### LoRA

| 特性 | MiniMind LoRA | QLoRA | LLaMA-Adapter | LoRA+ |
|------|-------------|-------|--------------|-------|
| LoRA 应用层 | Q, O (方阵) | Q, K, V, O | 只加 Adapter 层 | Q, K, V, O, FFN |
| rank | 16 | 64 | N/A | 16~64 |
| 量化 | 无 | 4-bit NF4 | 无 | 无 |
| A 初始化 | 高斯 | 高斯 | 高斯 | 高斯 |
| B 初始化 | 全零 | 全零 | 全零 | 全零 |
| A/B 学习率 | 相同 | 相同 | 相同 | B 的 lr ≈ 16 倍 A（固定比率） |

几个值得注意的点：

1. **MiniMind 只对方阵加 LoRA**，业界更常见的做法是对 Q/K/V/O 全部加 LoRA（效果更好）
2. **QLoRA**（Dettmers et al., 2023）把基础模型量化到 4-bit，只用 LoRA 做微调，70B 模型可以在单张 48GB GPU 上训练
3. **LoRA+**（Hayou et al., 2024）发现 A 和 B 用不同学习率效果更好——B 的学习率应该约为 A 的 16 倍（固定比率 $\lambda = 2^4$，来自大宽度极限下的特征学习分析）
4. **DoRA**（Liu et al., 2024）把权重分解为幅度和方向两个分量，分别用 LoRA 适配，效果更接近全参

---

## SFT 监控清单

> 故障分类与定位方法论见 [10b-训练监控](10b-训练监控.md)（四种死法 + 事故案例库）。

**核心转变**：teacher forcing 下，loss 衡量的是"对标准答案的拟合程度"，和生成质量弱相关。SFT 的监控重心从数值转向**行为**——而行为只有让模型实际生成才能看到。

### 指标清单

| 指标 | 为什么留下 | 怎么看 |
|------|----------|-------|
| **分任务 eval loss** | 只算 assistant 部分（先验证 prompt masking 没写错——mask 错了 loss 会异常低或高）；按任务类型分开看 | 总 loss 降而某类升 = 常态，需定位 |
| **Gradient Norm** | 全局 L2 范数 | 飙升→不稳定；持续很小→梯度消失 |
| **Token Accuracy** | 逐 token 预测准确率（只算 assistant） | 和 loss 不一致时以 accuracy 为准 |
| **EOS 结束率**（定期生成） | SFT 第一高发事故：**模型学不会停**。teacher forcing 训练时看不到——只有让模型自己生成才暴露 | 正常应接近 100%；异常下降先查 truncation 率和 chat 模板拼接 |
| **truncation 率**（数据体检） | 训练样本因超 max_length 被截断的比例。被截断的样本学不到 EOS——这是 EOS 学不会的**头号根因** | **训练前查一次**：>1% 就要处理（扩 max_length 或滤掉） |
| **定期生成评审** | 固定 50~100 条 held-out prompt，每个 checkpoint 生成，RM 打分 + **人眼抽查至少 20 条** | 复读、格式崩、语气突变只有人眼看得到 |
| **response 长度分布**（生成层面） | SFT 是 teacher forcing，训练时的"长度"是数据属性不是模型行为；只有**生成时**的长度漂移才是过拟合/模板化的信号 | 均值偏移 or 方差塌缩 = 模型在套路化 |
| **输出熵** | 词表概率分布的 Shannon entropy | 持续下降→多样性丧失→后续 RL 会失效（rank inversion） |
| **通用能力回归** | SFT 前后各跑一次 MMLU/GSM8K 级 benchmark | 掉 2+ 个点 = epoch 太多或 lr 太高 |

**已删除**：~~per-turn / per-length 分桶 loss~~（量级太小）；~~distinct-n 等文本统计~~（人工抽查覆盖）；~~与 base 模型的 KL~~（SFT 阶段信息量低，留到 RL 再盯）。

### SFT 多 epoch 的 loss 曲线（面试高频）

面试题："SFT 训了三个 epoch，每个 epoch 结束 loss 大降一次，怎么理解？"

**正解是记忆效应**：epoch 边界没有特殊计算，但第二遍起模型已经见过这些数据，同一批样本的 loss 系统性更低。逐 epoch 下台阶 = 模型在背训练集，小数据集尤其明显——这是**过拟合信号**。看 eval loss：train 降 eval 平就 early stop。

### Checkpoint 策略

```
1. 定时保存: 每 500-2000 步
2. 最优保存: eval loss 最低的 top-3
3. 如果后接 RL: 在通过格式门槛的 checkpoint 中选 pass@10 最高或输出熵最高的（熵是 pass@10 的廉价代理）
4. 如果不接 RL: 用 downstream metric（不是 loss）选最终模型
```

---

## 面试话术

### 全参 SFT

> "全参 SFT 我是从预训练脚本改过来的——训练循环完全一样，唯一的区别在数据层面。
> 我写了 SFTDataset，核心是 `generate_labels` 方法：扫描 token 序列找到
> `<|im_start|>assistant\n` 和 `<|im_end|>\n` 之间的区间作为有效 label，其余全部设为 -100。
> PyTorch 的 cross_entropy 用 `ignore_index=-100` 自动跳过这些位置。
>
> 超参数方面，学习率从预训练的 5e-4 降到 1e-5 防灾难性遗忘，序列长度从 340 拉到 768 适应对话格式。
>
> 为什么不对 user 指令算 loss？一是模型不需要学怎么提问，二是避免推理时混入用户表达方式，
> 三是指令占了很多 token 但贡献无用梯度。这是业界标准做法。
>
> 多轮对话不拆成 n 条样本——整个对话拼成一条序列，靠 causal mask 保证只看到前文，
> 靠 label mask 保证只对 assistant 算 loss，一次前向传播同时训练所有轮次。
> 这行得通是因为 Teacher Forcing——训练时每个位置的输入是语料真值而非模型自己的预测。"

### LoRA

> "LoRA 我也是自己写的，不依赖 HuggingFace PEFT。核心就一个 LoRA 类——A 矩阵高斯初始化、
> B 矩阵全零初始化，确保训练开始时 BA=0 不影响原始输出。然后用 monkey-patch 把
> 原始 Linear 的 forward 替换成 original(x) + lora(x)。
>
> 在 MiniMind 中只对 Attention 的 Q 和 O 投影加 LoRA（方阵判断 in==out），
> rank=16，可训练参数约 0.39M，占总参数 0.6%。
>
> 训练时冻结所有非 LoRA 参数，优化器只接收 LoRA 参数——天然抗灾难性遗忘。
> 学习率比全参高一个量级（1e-4 vs 1e-5），因为参数少需要更大步长。
>
> 训练完可以 merge——把 BA 加回原始权重 W_new = W + BA，推理时没有任何额外开销，
> 模型结构和普通模型一模一样。一个 7B 模型的 LoRA 权重只有 10~50MB，全参要 14GB。"

### SFT 训练监控 & 何时切 RL

> "SFT 的 train loss 不太能说明问题——数据量小、模型容量大，loss 一定会降，过拟合了也会降。
> 我更关注几个信号：第一是 eval loss，留一部分数据做验证，eval loss 开始上升就是过拟合信号。
> 第二是 gradient norm，突然飙升说明训练不稳定、需要检查数据或调 lr。
> 第三是定期做生成采样，拿固定的 prompt 看模型实际输出质量——loss 和 accuracy 都是代理指标，
> 生成的文本才是真实反馈。至于 epoch 边界 loss 突降，本质是模型第二遍看到已学过的数据，
> 记忆效应导致的——这也是为什么 SFT 一般只训 1~3 个 epoch。
>
> 如果后续要接 RL，还有一个 2026 年的新发现很关键：SFT 表现最好的 checkpoint 往往是
> RL 最差的起点——这叫 rank inversion。原因是 SFT 训得越久，模型输出熵越低、回复多样性
> 越差，GRPO 的组内 rollout 趋于同质化，advantage 信号消失。所以选 RL 起点不应该选
> eval loss 最低或 pass@1 最高的 checkpoint，而是在通过格式门槛后选 pass@10 最高的
> 那个——pass@10 直接衡量 GRPO 能不能采到正样本。实操中输出熵是 pass@10 的廉价代理：
> 两者高度相关，但熵只需一次前向传播，不用做 10 次采样和评测。"

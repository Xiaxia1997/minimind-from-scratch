# 12 - SFT：全参微调与 LoRA

> 源码位置：`trainer/train_full_sft.py`（全参 SFT 主循环）、`trainer/train_lora.py`（LoRA 训练主循环）、`model/model_lora.py`（LoRA 结构 + 注入 + 保存/加载/合并）、`dataset/lm_dataset.py:58-119`（SFTDataset）
> 依赖：`model/model_minimind.py:245-253`（MiniMindForCausalLM.forward 中的 loss 计算）
> 前置：[11-预训练](11-预训练.md)

## 一句话总结

SFT 和预训练的区别只有一个：**Loss 只对 assistant 回复部分计算**。全参 SFT 更新所有参数，LoRA 只更新低秩增量——两者共享相同的数据格式和 loss mask。

---

## Part 1: SFT 数据格式——从纯文本到指令-回复对

### 数据从"一段话"变成"对话"

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

### Chat Template：把对话变成 token 序列

模型看到的不是 JSON，而是一串用特殊 token 分隔的文本。MiniMind 使用 ChatML 格式：

```
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
中国的首都是哪里？<|im_end|>
<|im_start|>assistant
中国的首都是北京。<|im_end|>
```

```python
# dataset/lm_dataset.py:71-86
def create_chat_prompt(self, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"])
        messages.append(message)
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )
```

`apply_chat_template` 是 HuggingFace tokenizer 提供的方法，它根据 tokenizer 配置的 Jinja2 模板把 `[{role, content}]` 列表渲染成带特殊 token 的字符串。`add_generation_prompt=False` 表示不在末尾追加 `<|im_start|>assistant\n`——因为训练时 assistant 回复已经在数据里了。

---

## Part 2: Loss Mask——SFT 的核心区别

### 预训练 vs SFT 的 label 对比

```
预训练:
input_ids: [BOS, 中, 国, 首, 都, 是, 北, 京, EOS, PAD]
labels:    [BOS, 中, 国, 首, 都, 是, 北, 京, EOS, -100]
           ↑    所有非 padding 位置都算 loss

SFT:
input_ids: [<|im_start|>, system, ..., <|im_end|>, <|im_start|>, user, 中, 国, ..., <|im_end|>, <|im_start|>, assistant, 北, 京, <|im_end|>]
labels:    [   -100,       -100,  ...,   -100,        -100,      -100, -100,-100,...,  -100,        -100,       -100,    北, 京, <|im_end|>]
                                                                                                                          ↑
                                                                                                               只有 assistant 回复部分算 loss
```

### generate_labels：找出 assistant 回复的边界

#### 边界标记是什么

要区分"哪些 token 是 assistant 说的"，需要两个标记：

```python
# dataset/lm_dataset.py:65-66
self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

- `bos_id`：`<|im_start|>assistant\n` 的 token 序列——ChatML 格式中 assistant 回复的起始标记
- `eos_id`：`<|im_end|>\n` 的 token 序列——回复的结束标记

`generate_labels` 的逻辑：扫描 token 序列，每遇到一段 `bos_id...eos_id`，就把中间（含 eos_id）的 label 从 -100 改为真实 token id，其他位置保持 -100。

#### 用具体例子走一遍

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

#### 理解了流程再看代码

```python
# dataset/lm_dataset.py:88-104
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
            # 只把 assistant 回复部分的 label 设为真实 token id
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id)
        else:
            i += 1
    return labels
```

### 为什么只对 assistant 回复算 loss

三个原因：

1. **信息不对等**：用户指令是"输入"，模型不需要学习"怎么提问"。如果对用户指令也算 loss，模型会把一部分学习能力浪费在拟合"用户怎么说话"上。

2. **避免模式混淆**：模型学到的生成模式应该是"给定 context，生成 assistant 风格的回答"。如果也学了用户的说话模式，推理时可能混入用户的表达方式。

3. **梯度效率**：SFT 数据中用户指令和 system prompt 占了很大比例的 token，如果都算 loss，有效梯度信号被稀释。只对回复部分算 loss，每个 token 的梯度信号都是有用的。

### SFTDataset 完整流程

```python
# dataset/lm_dataset.py:106-119
def __getitem__(self, index):
    sample = self.samples[index]
    conversations = pre_processing_chat(sample['conversations'])

    # ① 把对话渲染成 ChatML 格式字符串
    prompt = self.create_chat_prompt(conversations)
    prompt = post_processing_chat(prompt)

    # ② 分词 + 截断到 max_length
    input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

    # ③ Padding
    input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

    # ④ 生成 loss mask——只有 assistant 回复部分的 label 不是 -100
    labels = self.generate_labels(input_ids)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

```
完整流程:

原始数据:  {"conversations": [{role: system, ...}, {role: user, ...}, {role: assistant, ...}]}
    ↓
ChatML:    "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n"
    ↓
分词:      [token_id_0, token_id_1, ..., token_id_N]
    ↓
截断+Pad:  [token_id_0, ..., token_id_N, PAD, PAD, ...]     长度 = max_length
    ↓
Labels:    [-100, ..., -100, tok_a1, tok_a2, ..., <|im_end|>, -100, ..., -100]
                             ↑ assistant 回复的内容 ↑
```

### 多轮对话为什么一条序列就够——Teacher Forcing

Part 2 的例子里，两轮 assistant 回复放在同一条序列里训练。你可能会想：模型预测 assistant1 的结果和语料不一样，那预测 assistant2 时前文不就错了吗？

答案是：**训练时模型的输入不是自己的预测，而是语料里的真实 token**。

#### 推理时（自回归）：模型用自己的输出当下一步输入

```
step 1: 输入 [user1]           → 模型生成 "你"
step 2: 输入 [user1, 你]       → 模型生成 "好"
step 3: 输入 [user1, 你, 好]   → 模型生成 "啊"    ← 如果这里生成错了
step 4: 输入 [user1, 你, 好, 啊] → 后面全歪      ← 误差累积
```

#### 训练时（Teacher Forcing）：输入永远是语料真值

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

#### 这和"拆成 n 条"有什么区别

直觉上 3 轮对话似乎应该拆成 3 条训练样本：

```
拆分式:
样本1: [user1]                                → 预测 [assistant1]
样本2: [user1, assistant1, user2]             → 预测 [assistant2]
样本3: [user1, assistant1, user2, assistant2, user3] → 预测 [assistant3]
```

这样做**数学上等价**（causal mask 保证了可见范围一样），但 MiniMind 不这样做，因为：
- 前缀在每条里重复计算（user1 算了 3 遍）
- 数据膨胀 n 倍
- 一条序列 + label mask 更高效，效果完全一样

**一句话记住：训练不是让模型"推理"，而是给它真实答案问"你猜到了吗"。** 猜错了算 loss、更新参数，但下一个位置看到的输入仍然是语料里的正确答案——这就是 Teacher Forcing。

#### cross_entropy 的 ignore_index=-100

label = -100 的位置是怎么被"跳过"的：

```python
# model/model_minimind.py:252
loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
```

PyTorch 的 cross_entropy 内置了 `ignore_index` 参数：
- $\text{label} = -100$ 的位置 → loss 贡献为 0，梯度也为 0
- $\text{label} \neq -100$ 的位置 → 正常计算 cross_entropy
- 最终 $\text{loss} = \dfrac{\sum_{\text{valid}} \text{CE}_i}{\, N_{\text{valid}} \,}$（mean reduction）

#### 和 DPO 的 loss mask 对比

```python
# dataset/lm_dataset.py:176-192 (DPODataset)
def generate_loss_mask(self, input_ids):
    loss_mask = [0] * len(input_ids)    # 用 0/1 mask 而不是 -100
    # 同样只对 assistant 回复段标 1
    ...
    return loss_mask
```

DPO 用 0/1 的 loss_mask 而不是 -100，因为 DPO 的 loss 不用 cross_entropy 的 ignore_index，而是手动把 log_probs 乘以 mask 来实现选择性计算（详见 [13-DPO](13-DPO.md)）。

---

## Part 3: 全参 SFT 训练

### 和预训练的 diff

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

训练循环代码**完全相同**——同样的 forward、同样的 loss.backward()、同样的 AdamW + 余弦退火 + 梯度裁剪。差异全在数据层面（SFTDataset 的 label mask）。

### 关键超参数

| 超参数 | 预训练 | 全参 SFT | 为什么变了 |
|--------|--------|---------|-----------|
| `learning_rate` | 5e-4 | 1e-5 | SFT 是微调，学习率太高会破坏预训练学到的知识（灾难性遗忘） |
| `max_seq_len` | 340 | 768 | 对话数据通常比纯文本长（包含 system + user + assistant） |
| `batch_size` | 32 | 16 | 序列更长，显存占用更大，需要减小 batch |
| `accumulation_steps` | 8 | 1 | 等效 batch = 16 × 1 = 16，SFT 数据量小不需要太大 batch |
| `from_weight` | none | pretrain | 必须从预训练权重开始，否则模型没有基础知识 |

### 数据量级对比

```
预训练数据:    GB~TB 级，数百万~数十亿条
SFT 数据:      MB 级，1 万~100 万条
               → SFT 只需要 1~3 个 epoch
               → 训练时间从天/周缩短到小时
```

### 全参 SFT 的显存分析

全参 SFT 意味着所有 64M 参数都参与训练。显存主要占用：

```
模型参数:         64M × 4 bytes (FP32) = 256 MB
                  或 64M × 2 bytes (BF16) = 128 MB

梯度:             和参数一样大 = 128~256 MB

AdamW 优化器状态: 每个参数需要存一阶动量 + 二阶动量
                  = 64M × 4 bytes × 2 = 512 MB（始终 FP32）

激活值:           取决于 batch_size × seq_len
                  MiniMind: 16 × 768 × 768 × 8层 ≈ 几百 MB

总计:             MiniMind 约 1~2 GB（64M 参数很小）
                  7B 模型: 约 60~80 GB（需要多张 A100）
```

这就是为什么大模型通常用 LoRA 而不是全参 SFT——7B 全参 SFT 需要至少一张 80GB A100，而 LoRA 在一张 24GB RTX 4090 上就能跑。

### 对照 MiniMind 源码

| 概念 | 我们讲的 | 源码位置 |
|------|---------|---------|
| SFT 数据格式 | conversations 多轮对话 | `lm_dataset.py:58-64` |
| Chat Template | `apply_chat_template` | `lm_dataset.py:81-86` |
| Loss Mask | `generate_labels` | `lm_dataset.py:88-104` |
| 边界标记 | `bos_id` / `eos_id` | `lm_dataset.py:65-66` |
| 训练循环 | 和预训练完全一样 | `train_full_sft.py:24-73` |
| 学习率 | 1e-5 | `train_full_sft.py:90` |
| from_weight | pretrain | `train_full_sft.py:103` |

---

## Part 4: LoRA——低秩微调

### 为什么需要 LoRA

#### 全参 SFT 的问题

全参 SFT 更新所有 64M 参数，对于 MiniMind 这种小模型还可以接受，但放到大模型场景：

```
模型规模       全参 SFT 显存         LoRA SFT 显存       需要的硬件
7B            ~60 GB               ~16 GB              A100 80GB → RTX 4090 24GB
13B           ~120 GB              ~24 GB              2×A100 → 1×A100
70B           ~560 GB              ~80 GB              8×A100 → 1×A100
```

除了显存，全参 SFT 还有一个实际问题：**每个下游任务保存一份完整模型权重**。7B 模型 = 14GB 权重文件，10 个任务 = 140GB。LoRA 只保存增量权重，7B 模型的 LoRA 权重通常只有 10~50MB。

#### LoRA 的直觉

Aghajanyan et al. (2020) 发现：预训练模型虽然参数量巨大，但微调时的权重变化 ΔW 是**低秩**的——也就是说，ΔW 的有效信息可以用一个远小于原始维度的矩阵来表示。

LoRA (Hu et al., 2021) 把这个观察变成了方法：

```
原始:  y = Wx                    W: [d_out, d_in]

LoRA:  y = Wx + BAx              W: [d_out, d_in]  ← 冻结，不训练
                                 A: [rank, d_in]    ← 训练
                                 B: [d_out, rank]   ← 训练

rank << d_in, d_out

MiniMind 的例子:
  W: [768, 768]   = 589,824 个参数
  A: [16, 768]    = 12,288 个参数
  B: [768, 16]    = 12,288 个参数
  LoRA 参数:       24,576 (占 W 的 4.2%)
```

### MiniMind 的 LoRA 实现

#### LoRA 模块

```python
# model/model_lora.py:6-18
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)     # 降维: [d_in] → [rank]
        self.B = nn.Linear(rank, out_features, bias=False)     # 升维: [rank] → [d_out]
        self.A.weight.data.normal_(mean=0.0, std=0.02)         # A 高斯初始化
        self.B.weight.data.zero_()                             # B 全零初始化

    def forward(self, x):
        return self.B(self.A(x))                               # x → A(x) → B(A(x))
```

#### 初始化策略为什么是"A 高斯，B 全零"

训练开始时，`B @ A` 的输出 = 0（因为 B 全零），所以 LoRA 模块不会改变原始模型的输出。这是一个关键设计——**确保微调从预训练模型的原始状态开始**，而不是被随机初始化扰乱。

```
训练开始:  y = Wx + BAx = Wx + 0 = Wx      ← 和原始模型完全一样
训练过程:  B 和 A 逐渐被更新              ← 逐渐学习任务特定的调整
训练结束:  y = Wx + BAx                    ← BA 学到了"如何从原始输出调整到期望输出"
```

如果 A 和 B 都随机初始化，训练开始时 `BAx` 就不是 0，模型一开始就偏离了预训练状态，可能导致训练不稳定。

#### LoRA 注入：apply_lora

**哪些层需要加 LoRA**

不是所有 Linear 层都需要 LoRA。MiniMind 用一个简单的筛选条件：`in_features == out_features`（方阵 Linear 层）。在 MiniMind 中：

```
Attention:
  wq:  [768, 768]  ← 加 LoRA (方阵)
  wk:  [768, 256]  ← 不加（GQA，K 头数少）
  wv:  [768, 256]  ← 不加（GQA，V 头数少）
  wo:  [768, 768]  ← 加 LoRA (方阵)

FeedForward:
  w1:  [768, 2688] ← 不加（非方阵）
  w2:  [2688, 768] ← 不加（非方阵）
  w3:  [768, 2688] ← 不加（非方阵）

每层 LoRA 参数 = 2 × (768 × 16 + 16 × 768) = 2 × 24,576 = 49,152
8 层总计 = 8 × 49,152 = 393,216 ≈ 0.39M
```

**注入方式：monkey-patch**

`apply_lora` 使用 monkey-patch（运行时替换方法）来修改 Linear 层的 forward：

```
修改前:  module.forward(x) → x @ W.T + bias
修改后:  module.forward(x) → x @ W.T + bias + B(A(x))

关键: 闭包绑定
def forward_with_lora(x, layer1=original_forward, layer2=lora):
    return layer1(x) + layer2(x)

layer1 和 layer2 用默认参数绑定，确保每个 module 的 lora 不会指错
如果用 module.lora 而不是显式绑定，所有层的 forward 会指向最后一个 lora
```

这种实现简单但有副作用——`torch.compile` 不兼容 monkey-patch，所以 LoRA 训练中 `use_compile` 被自动关闭（`train_lora.py:165-166`）。

**理解了机制再看代码**

```python
# model/model_lora.py:21-32
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        # 只对方阵 Linear 加 LoRA（in == out，即 Attention 的 Q/K/V/O 投影）
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)        # y = Wx + BAx

            module.forward = forward_with_lora
```

#### 参数冻结与优化器

```python
# train_lora.py:140-152
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True      # LoRA 参数：可训练
        lora_params.append(param)
    else:
        param.requires_grad = False     # 原始参数：冻结

# 优化器只接收 LoRA 参数
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

冻结非 LoRA 参数有两个作用：
1. **不计算梯度** → 节省反向传播的计算量和显存
2. **不更新权重** → 预训练知识完全保留，天然抗灾难性遗忘

注意梯度裁剪也只作用于 LoRA 参数：

```python
# train_lora.py:45
torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
#                               ↑ 不是 model.parameters()
```

### LoRA 权重的保存、加载与合并

#### 保存：只存 LoRA 权重

```python
# model/model_lora.py:45-53
def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half()
                         for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

只保存带 `lora` 属性的模块的参数。64M 模型的全参权重 ~128MB，LoRA 权重 ~1MB。

#### 加载：把 LoRA 权重恢复到 monkey-patch 后的模型

```python
# model/model_lora.py:35-42
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v
                         for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
```

#### 合并：把 LoRA 权重融入原始权重

```python
# model/model_lora.py:56-65
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'lora'):
            # W_merged = W + B @ A
            state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
```

合并后的模型和原始模型结构完全一样（没有 LoRA 模块），推理时没有额外开销。

```
合并的数学:
  训练时:   y = Wx + BAx = (W + BA)x
  合并后:   W_new = W + BA
            y = W_new × x                ← 和普通模型一样，无额外计算
```

### LoRA vs 全参 SFT 的训练差异

#### 超参数对比

| 超参数 | 全参 SFT | LoRA SFT |
|--------|---------|----------|
| `learning_rate` | 1e-5 | 1e-4（高 10 倍） |
| `batch_size` | 16 | 32 |
| `epochs` | 2 | 10 |
| `max_seq_len` | 768 | 340 |
| `from_weight` | pretrain | full_sft |
| `accumulation_steps` | 1 | 1 |
| `save_weight` | full_sft | lora_medical |

LoRA 学习率比全参 SFT 高 10 倍——这是因为 LoRA 参数量少，梯度信号需要更大的学习率才能在有限参数上产生足够的调整。epoch 数也更多（10 vs 2），因为参数少、数据小，需要更多遍训练。

#### 训练流程对比

```
全参 SFT:
  加载预训练权重 → 所有参数可训练 → AdamW(model.parameters()) → 保存全部权重

LoRA SFT:
  加载 SFT 权重 → apply_lora() 注入 LoRA → 冻结非 LoRA 参数
  → AdamW(lora_params) → 只保存 LoRA 权重 → 可选: merge_lora 融合
```

注意 MiniMind 的 LoRA 是从 `full_sft` 权重开始的（不是从 pretrain），因为 LoRA 的典型应用场景是在一个已经 SFT 过的通用模型上做领域适配（如医疗 LoRA、法律 LoRA）。

### LoRA 的数学直觉

#### 为什么低秩就够了

一个 [768, 768] 的权重矩阵可以表示 768 维空间到 768 维空间的任意线性变换。但微调时，权重的变化 ΔW 通常只需要调整几个"方向"——比如让模型更关注医疗术语、调整回答风格。

```
ΔW 的秩 = ΔW 中独立变化方向的数量

rank=16 意味着 ΔW 最多有 16 个独立方向
对于从"通用回答"到"医疗领域回答"的适配，16 个方向通常足够

类比: 768 维空间中，调整"说话风格"可能只需要改几个主成分方向
      就像 PCA 降维——大部分变化被前几个主成分捕获
```

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

## 全参 vs LoRA：怎么选

| 维度 | 全参 SFT | LoRA |
|------|---------|------|
| 效果 | 更好（所有参数自由调整） | 性价比高（接近全参 90%+ 效果） |
| 显存 | 高（参数 + 梯度 + 优化器状态全量） | 低（只存 LoRA 参数的梯度和优化器状态） |
| 权重存储 | 每个任务一份完整权重 | 每个任务只存增量（~1MB vs ~128MB） |
| 灾难性遗忘 | 需要低学习率控制 | 天然抗遗忘（大部分参数冻结） |
| 适用场景 | 数据量大 + 资源充足 + 追求最优效果 | 数据量小 + 资源有限 + 多任务适配 |

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
| A/B 学习率 | 相同 | 相同 | 相同 | B 的 lr = 2~8 倍 A |

几个值得注意的点：

1. **MiniMind 只对方阵加 LoRA**，业界更常见的做法是对 Q/K/V/O 全部加 LoRA（效果更好）
2. **QLoRA**（Dettmers et al., 2023）把基础模型量化到 4-bit，只用 LoRA 做微调，70B 模型可以在单张 48GB GPU 上训练
3. **LoRA+**（Hayou et al., 2024）发现 A 和 B 用不同学习率效果更好——B 的学习率应该是 A 的 2~8 倍
4. **DoRA**（Liu et al., 2024）把权重分解为幅度和方向两个分量，分别用 LoRA 适配，效果更接近全参

---

## 面试话术

### 全参 SFT

> "全参 SFT 和预训练的唯一区别在 Loss mask——SFT 只对 assistant 回复部分算 loss，
> system 和 user 的 label 全部设为 -100。SFTDataset 用 `generate_labels` 方法扫描 token 序列，
> 找到 `<|im_start|>assistant\n` 和 `<|im_end|>\n` 之间的区间作为有效 label，其余全部 mask 掉。
>
> 训练循环和预训练完全一样，差异在超参数：学习率从 5e-4 降到 1e-5（防止灾难性遗忘），
> 从预训练权重 `from_weight=pretrain` 初始化，序列长度从 340 拉到 768 适应对话格式。
>
> 这里有个关键设计选择——为什么不对 user 指令算 loss？
> 一是信息不对等，模型不需要学怎么提问；
> 二是避免模式混淆，推理时混入用户的表达方式；
> 三是梯度效率，指令占了很多 token 但贡献无用梯度。这是业界标准做法。
>
> 多轮对话不需要拆成 n 条样本——整个对话拼成一条序列，靠 causal mask 保证每个位置只看到前文，
> 靠 label mask（-100）保证只对 assistant 回复算 loss，一次前向传播同时训练所有轮次。
> 这之所以可行是因为 Teacher Forcing——训练时每个位置的输入是语料真值而非模型自己的预测，
> 所以不存在'assistant1 预测错了导致 assistant2 也错'的问题。"

### LoRA

> "LoRA 的核心思想是低秩分解——预训练模型微调时的权重变化 ΔW 是低秩的，
> 可以分解为两个小矩阵 B 和 A 的乘积，rank 远小于原始维度。
> 在 MiniMind 中，rank=16，只对 Attention 的 Q 和 O 投影加 LoRA，
> 可训练参数量约 0.39M，占总参数的 0.6%。
>
> 实现上，B 全零初始化确保训练开始时不改变原始输出，A 高斯初始化提供梯度方向。
> 原始权重完全冻结，只有 LoRA 参数参与梯度更新和优化器状态存储，
> 显存占用大幅降低。训练完成后可以把 BA 合并到原始权重 W_new = W + BA，
> 推理时没有任何额外开销。
>
> 相比全参 SFT，LoRA 学习率通常高一个量级（1e-4 vs 1e-5），因为参数少需要更大步长。
> 同时天然抗灾难性遗忘——大部分参数冻结，预训练知识完全保留。"

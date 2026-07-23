# 09b - PyTorch 与 HuggingFace 基础

## 这章在干什么

前面 8 章我们从零搭建了 MiniMind 的每一个组件（Embedding、RMSNorm、RoPE、Attention、FFN、MOE、Block、完整模型）。接下来要进入训练篇和完整模型组装，代码里会大量出现 PyTorch 和 HuggingFace transformers 的基础设施类。

本章的目标：**把这些"脚手架"搞清楚**，这样后面写训练代码和读 `MiniMindForCausalLM` 的时候不会被框架 API 挡住。

覆盖三件事：
1. PyTorch 核心类（`nn.Module`、`nn.Parameter`、常用层）
2. HuggingFace transformers 核心类（`PretrainedConfig`、`PreTrainedModel`、`GenerationMixin`）
3. MiniMind 代码中每个类用了哪些特性（一张对照表）

---

## Part 1：PyTorch 核心类

### 1.1 nn.Module —— 所有模型的基类

PyTorch 里每一个"模型"或"层"都继承自 `nn.Module`。我们前面写的 `RMSNorm`、`Attention`、`FeedForward`、`MiniMindBlock` 全都是 `nn.Module` 的子类。

它提供了五个关键能力：

#### (1) `__init__` 定义结构，`forward` 定义计算

```python
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()                          # 必须调用 super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # 在 __init__ 里定义可学习参数

    def forward(self, x):                           # 在 forward 里定义前向计算
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

规则很简单：**`__init__` 里声明"有什么"，`forward` 里声明"怎么算"**。调用时直接 `norm(x)` 而不是 `norm.forward(x)`——原因见下面的 `__call__` 机制。

#### 为什么写 `norm(x)` 而不是 `norm.forward(x)`

```python
norm = RMSNorm(768)   # __init__ 执行完，norm 是一个 RMSNorm 对象

# 这两种写法不等价：
norm(x)            # ✅ 正确：走 __call__，完整执行 hook + forward
norm.forward(x)    # ❌ 错误：绕过了 __call__，hook 不会执行
```

Python 的 `对象(参数)` 语法会调用 `对象.__call__(参数)`。`nn.Module` 重写了 `__call__`，在里面做了三件事：

```python
# nn.Module.__call__ 的简化版逻辑（PyTorch 源码 torch/nn/modules/module.py）
def __call__(self, *input, **kwargs):
    # 1. 执行所有 forward pre-hook（前置钩子）
    for hook in self._forward_pre_hooks.values():
        hook(self, input)

    # 2. 调用你写的 forward
    result = self.forward(*input, **kwargs)

    # 3. 执行所有 forward hook（后置钩子）
    for hook in self._forward_hooks.values():
        hook(self, input, result)

    return result
```

所以 `norm(x)` = "执行 pre-hook → 执行 `forward(x)` → 执行 post-hook"。如果你直接调 `norm.forward(x)`，就绕过了整个 hook 机制。

#### Hook 是什么？

Hook = **空插槽**。`nn.Module` 在 `__call__` 里预留了几个位置，你可以选择往里面塞一个自定义函数，也可以什么都不塞。

**关键认知：新建的模型默认没有任何 hook。** 上面 `__call__` 里的那两个 for 循环遍历的是空列表，直接跳过。只有你**主动调 `register_xxx_hook()`** 往上面挂一个函数，它才会在每次 forward 时被调用。

```python
norm = RMSNorm(768)
norm._forward_hooks        # OrderedDict()，空的——没有任何 hook
norm._forward_pre_hooks    # OrderedDict()，空的

# 此时 norm(x) 的效果就是纯粹的 norm.forward(x)，没有区别
# 因为 hook 列表为空，for 循环什么也不做
```

#### 什么时候会用到 hook？

只有两种场景你会接触到 hook：

**场景 1：你自己注册（调试/提取中间结果）**

```python
# 例：想看第 3 层 Attention 输出了什么，但不想改模型代码
activations = {}

def save_output(module, input, output):
    activations['layer3_attn'] = output

# 往第 3 层的 self_attn 上挂一个 forward hook
model.layers[3].self_attn.register_forward_hook(save_output)

# 之后每次调 model(input_ids)，经过这一层时 save_output 就会被自动调用
model(input_ids)
print(activations['layer3_attn'].shape)  # [B, S, 768]
```

注册之后，**每次** forward 经过这个模块都会触发。想取消就调 `hook_handle.remove()`。

**场景 2：PyTorch/HuggingFace 内部注册（你看不到但它在用）**

`torch.compile`、分布式训练（DDP）等高级功能会在你调 `model = DDP(model)` 时，自动往每个模块上挂 hook 来同步梯度。你不需要知道细节，只需要知道"这就是为什么要用 `model(x)` 而不是 `model.forward(x)`"——绕过 `__call__` 会导致这些内部 hook 失效。

#### Hook 类型速查

| Hook 类型 | 注册方法 | 触发时机 |
|-----------|---------|---------|
| Forward Pre-Hook | `register_forward_pre_hook(fn)` | `forward()` 之前 |
| Forward Hook | `register_forward_hook(fn)` | `forward()` 之后 |
| Backward Hook | `register_full_backward_hook(fn)` | `backward()` 经过该模块时 |

日常写模型代码**几乎不需要手动注册 hook**。

---

#### 保存激活值 vs Gradient Checkpoint

先澄清一个常见混淆：**保存激活值不是 hook，是 autograd 的内置行为。**

当一个 tensor 的 `requires_grad=True` 时（`nn.Parameter` 默认就是 True，你从来不用手动设），前向计算时 PyTorch 自动保存中间结果——这是写死在引擎里的，不需要 hook，也关不掉。

```python
# nn.Parameter 的构造函数：
class Parameter(Tensor):
    def __new__(cls, data, requires_grad=True):  # ← 默认 True
        ...

# 所以你写 self.weight = nn.Parameter(torch.ones(768)) 时
# 这个 tensor 自动 requires_grad=True
# 前向计算涉及它的所有中间结果都会被 autograd 保存
```

这意味着：只要你在训练（有 backward），autograd 就会保存所有中间激活值。这是默认行为，不可配置。

**Gradient Checkpoint 是主动覆盖这个默认行为**——告诉 PyTorch"这一层的中间激活不要保存，反向时我宁愿重新算一遍"。用时间换空间：

```
默认（autograd 内置）：
  前向：Layer1 → 保存激活 → Layer2 → 保存激活 → ... → Layer8 → 保存激活
  反向：直接读取保存的激活来算梯度
  显存：8 层的激活值全部常驻

用 gradient checkpoint 包住（你主动选择）：
  前向：Layer1 → 不保存 → Layer2 → 不保存 → ... → Layer8
  反向：需要某层的激活？→ 从该层输入开始重新前向算一遍
  显存：只存每层的输入，中间激活用完即扔
```

**它不是 hook**，而是 autograd 的**自定义 Function** 机制——PyTorch 允许你定义一个操作，自己控制"前向时保存什么、反向时怎么算"。`checkpoint()` 就是这样一个自定义操作。

在 MiniMind 中的使用：

```python
# model/model_minimind.py 中的 MiniMindModel.forward
from torch.utils.checkpoint import checkpoint

for layer in self.layers:
    if self.gradient_checkpointing and self.training:
        # 用 checkpoint 包住：前向时不保存中间激活，反向时重算
        hidden_states = checkpoint(layer, hidden_states, ...)
    else:
        # 默认：前向时 autograd 自动保存所有中间激活
        hidden_states = layer(hidden_states, ...)
```

你只需要知道：

- **效果**：显存从 $O(L)$ 降到 $O(\sqrt{L})$（$L$ = 层数），代价是约 33% 额外计算（每层前向算两遍）
- **使用条件**：只在训练时有意义（推理时没有 backward，根本不需要保存激活）
- **在哪开启**：`model.gradient_checkpointing_enable()`（HuggingFace 方法）

#### 三者关系总结

| | 保存激活值 | Hook | Gradient Checkpoint |
|---|---|---|---|
| **是什么** | autograd 内置行为 | `__call__` 里的空插槽 | autograd 自定义 Function |
| **默认状态** | 训练时自动发生 | 空的，什么都不做 | 关闭（不用） |
| **需要你配置吗** | 不需要，也关不掉 | 需要主动 register | 需要主动开启 |
| **目的** | 给 backward 提供中间结果 | 让你无侵入地插入自定义逻辑 | 用重算替代保存，省显存 |

> Gradient Checkpoint 的完整原理、显存计算和业界实践见 [09-训练基础.md](09-训练基础.md) Step 2「Gradient Checkpointing」小节。

#### (2) `.parameters()` 返回所有可学习参数

```python
norm = RMSNorm(768)

# .parameters() 递归返回所有 nn.Parameter
for name, param in norm.named_parameters():
    print(name, param.shape)
# 输出: weight torch.Size([768])
```

这是 optimizer 的入口。训练时我们写 `optimizer = AdamW(model.parameters(), lr=1e-4)`，optimizer 就知道要更新哪些张量。

`nn.Module` 会**递归收集**所有子模块的参数。对于完整的 MiniMind 模型，一个 `.parameters()` 就能拿到所有层、所有注意力头、所有 FFN 的全部权重。

#### (3) `.to(device)` 递归搬移

```python
model = MiniMindModel(config)

# 把模型所有参数和 buffer 搬到 GPU
model = model.to("cuda")

# .to() 是递归的：
# model.embed_tokens 的权重 → cuda
# model.layers[0].self_attn.q_proj 的权重 → cuda
# model.freqs_cos (buffer) → cuda
# ... 所有子模块都会被搬过去
```

关键词是"递归"——不管模型嵌套多深，`.to()` 会遍历所有子模块，把每个 `nn.Parameter` 和 `register_buffer` 注册的张量都搬到目标设备。

#### (4) `.train()` / `.eval()` 切换模式

```python
model.train()  # 训练模式：Dropout 生效，BatchNorm 用当前 batch 统计量
model.eval()   # 推理模式：Dropout 关闭，BatchNorm 用全局统计量
```

MiniMind 里主要影响的是 `nn.Dropout`。在 `Attention` 类中：

```python
# model/model_minimind.py 中的 Attention
self.attn_dropout = nn.Dropout(config.dropout)
self.resid_dropout = nn.Dropout(config.dropout)
```

`.train()` 时 Dropout 会随机丢弃 neuron，`.eval()` 时自动关闭。这也是递归的——对顶层模型调一次 `.eval()`，所有子模块都切换。

#### (5) `.state_dict()` / `load_state_dict()` 序列化

```python
# 保存：把所有参数导出为有序字典 {参数名: 张量}
state = model.state_dict()
torch.save(state, "model.pt")

# 加载：把保存的参数灌回模型
state = torch.load("model.pt")
model.load_state_dict(state)
```

`state_dict()` 返回的 key 是参数的完整路径名，比如：

```
model.embed_tokens.weight          → [6400, 768]
model.layers.0.self_attn.q_proj.weight → [768, 768]
model.layers.0.self_attn.k_proj.weight → [384, 768]
model.norm.weight                  → [768]
lm_head.weight                    → [6400, 768]
```

这个命名结构直接对应 `__init__` 里的属性名。`self.layers = nn.ModuleList(...)` 使得子模块按索引编号（`layers.0`、`layers.1`...）。

---

### 1.2 nn.Parameter vs register_buffer —— 哪些张量跟着模型走

模型里有两类张量需要"跟着模型走"（保存、加载、搬设备）：

| | `nn.Parameter` | `register_buffer` |
|---|---|---|
| **有梯度？** | 有，optimizer 会更新 | 无，不参与梯度计算 |
| **`.to()` 搬移？** | 搬 | 搬 |
| **`.state_dict()` 包含？** | 包含 | 取决于 `persistent` 参数 |
| **`.parameters()` 返回？** | 返回 | 不返回 |

#### nn.Parameter：可学习的权重

```python
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # nn.Parameter 告诉 PyTorch："这是需要训练的参数"
        self.weight = nn.Parameter(torch.ones(dim))  # [768]
```

`nn.Parameter` 本质是一个 `requires_grad=True` 的 `torch.Tensor`，但它额外被 `nn.Module` 的参数注册机制追踪。所有 `nn.Linear`、`nn.Embedding` 内部的 `.weight` 和 `.bias` 都是 `nn.Parameter`。

#### register_buffer：不学习但要跟着走的张量

```python
class MiniMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # RoPE 的 cos/sin 是预计算的常量，不需要梯度
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,          # 96
            end=config.max_position_embeddings,  # 32768
            rope_base=config.rope_theta   # 1e6
        )
        # persistent=False：不保存到 state_dict（因为可以重算）
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

为什么用 `register_buffer` 而不是普通属性 `self.freqs_cos = freqs_cos`？

```python
# 如果用普通属性：
self.freqs_cos = freqs_cos  # 一个普通 tensor

model.to("cuda")
# model.freqs_cos 还在 CPU！
# 后面 forward 里 model.freqs_cos 和 GPU 上的 hidden_states 做运算就会报错：
# RuntimeError: Expected all tensors to be on the same device

# 如果用 register_buffer：
self.register_buffer("freqs_cos", freqs_cos)

model.to("cuda")
# model.freqs_cos 也在 GPU 了，.to() 会递归搬移所有 buffer
```

**MiniMind 中的对照**：
- **nn.Parameter**：所有 `nn.Linear` 的权重（q_proj、k_proj、v_proj、o_proj、gate_proj、down_proj、up_proj）、`RMSNorm.weight`、`nn.Embedding.weight`
- **register_buffer**：RoPE 的 `freqs_cos` 和 `freqs_sin`

---

### 1.3 常用层

#### nn.Linear —— 矩阵乘法

```python
linear = nn.Linear(in_features=768, out_features=384, bias=False)
# 内部参数：linear.weight 的形状是 [384, 768]（注意转置）

x = torch.randn(2, 128, 768)   # [batch, seq_len, hidden_size]
y = linear(x)                   # [2, 128, 384]

# 等价于：y = x @ linear.weight.T
# 如果 bias=True，则 y = x @ linear.weight.T + linear.bias
```

MiniMind 中所有线性层都用 `bias=False`（LLaMA 系列的惯例，减少参数量）：

```python
# Attention 中的投影
self.q_proj = nn.Linear(768, 768, bias=False)   # Q: hidden → n_heads * head_dim = 8 * 96 = 768
self.k_proj = nn.Linear(768, 384, bias=False)   # K: hidden → n_kv_heads * head_dim = 4 * 96 = 384
self.v_proj = nn.Linear(768, 384, bias=False)   # V: 同 K
self.o_proj = nn.Linear(768, 768, bias=False)   # O: 拼接后投影回 hidden_size
```

#### nn.Embedding —— 查表

```python
embed = nn.Embedding(num_embeddings=6400, embedding_dim=768)
# 内部参数：embed.weight 的形状是 [6400, 768]
# 6400 = vocab_size，768 = hidden_size

input_ids = torch.tensor([[1, 42, 100]])  # [1, 3]，三个 token id
vectors = embed(input_ids)                 # [1, 3, 768]

# 等价于：vectors = embed.weight[input_ids]
# 就是用 token id 作为索引去查表
```

`nn.Embedding` 不做任何矩阵乘法，就是一个 `[vocab_size, hidden_size]` 的查找表。输入整数索引，输出对应行的向量。

#### nn.ModuleList —— 为什么不能用普通 list

```python
# 错误写法：普通 list
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(768, 768) for _ in range(8)]

model = BadModel()
list(model.parameters())  # 空的！nn.Module 找不到 self.layers 里的参数
model.to("cuda")          # layers 里的 Linear 还在 CPU
```

`nn.Module` 只会追踪直接赋值为属性的 `nn.Module` 子类。普通 Python list 不被追踪。

```python
# 正确写法：nn.ModuleList
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(768, 768) for _ in range(8)])

model = GoodModel()
len(list(model.parameters()))  # 8 个权重矩阵都被追踪
model.to("cuda")               # 所有 layer 都搬到 GPU
```

MiniMind 里 Transformer 的 8 层就是用 `nn.ModuleList` 装的：

```python
# MiniMindModel.__init__
self.layers = nn.ModuleList([
    MiniMindBlock(l, config) for l in range(config.num_hidden_layers)  # 8 层
])
```

类似的还有 `nn.ModuleDict`，用于需要按名字索引子模块的场景。

#### nn.Dropout —— 训练时随机丢弃

```python
dropout = nn.Dropout(p=0.1)

x = torch.ones(2, 4)
dropout.train()
print(dropout(x))
# 输出类似: tensor([[1.1111, 0.0000, 1.1111, 1.1111],
#                   [1.1111, 1.1111, 0.0000, 1.1111]])
# 约 10% 的元素被置零，其余乘以 1/(1-p) 补偿

dropout.eval()
print(dropout(x))
# 输出: tensor([[1., 1., 1., 1.],
#               [1., 1., 1., 1.]])
# eval 模式下 Dropout 完全透传，不改变输入
```

MiniMind 默认 `dropout=0.0`（不丢弃），但保留了 Dropout 层以便实验时开启。

---

### 1.4 torch.inference_mode() vs torch.no_grad()

训练时 PyTorch 会为每个操作记录计算图（autograd graph），以便反向传播算梯度。推理时不需要梯度，应该关掉来省内存、加速。

```python
# 方式 1：torch.no_grad()
with torch.no_grad():
    output = model(input_ids)
# 不记录梯度，但仍然创建可以 requires_grad 的中间张量

# 方式 2：torch.inference_mode()（推荐）
with torch.inference_mode():
    output = model(input_ids)
# 更彻底：完全禁用 autograd 机制，不创建 version counter
# 比 no_grad() 更快，内存更省
```

**区别**：`inference_mode` 创建的张量是 `InferenceTensor`，不能用于任何需要梯度的后续操作。`no_grad` 创建的张量还是普通张量，只是 `requires_grad=False`。

一句话：**训练时偶尔关梯度用 `no_grad()`（比如计算 validation loss），纯推理用 `inference_mode()`**。

MiniMind 的 `generate` 方法就用了 `@torch.inference_mode()` 装饰器：

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    @torch.inference_mode()
    def generate(self, inputs=None, max_new_tokens=8192, temperature=0.85, ...):
        # 整个生成过程不需要梯度，用 inference_mode 最合适
        ...
```

---

### 1.5 Dataset 与 DataLoader —— 数据怎么喂给模型

#### Dataset：定义"一条数据长什么样"

PyTorch 的 `Dataset` 是一个抽象类，你需要继承它并实现两个方法：

```python
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=340):
        # __init__：加载原始数据（文件/数据库/API...）
        with open(data_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # 数据集有多少条（DataLoader 用它计算一个 epoch 的步数）
        return len(self.samples)

    def __getitem__(self, index):
        # 给一个整数索引，返回一条处理好的数据
        # 返回的是 tensor，直接能被模型吃
        text = self.samples[index]['text']
        tokens = self.tokenizer.encode(text)[:self.max_length]
        input_ids = torch.tensor(tokens, dtype=torch.long)  # [max_length]
        return input_ids, input_ids.clone()  # (input_ids, labels)
```

`Dataset` 只管"一条数据怎么取"，不管 batching、shuffle、并行加载——这些全交给 `DataLoader`。

#### DataLoader：把多条数据组装成 batch

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,              # 你写的 Dataset 对象
    batch_size=32,        # 每次取 32 条
    shuffle=True,         # 每个 epoch 打乱顺序（训练必须，验证不需要）
    num_workers=4,        # 4 个子进程并行预加载下一批数据
    pin_memory=True,      # 锁页内存，加速 CPU→GPU 传输
    drop_last=True,       # 最后不满 32 条的尾巴丢掉（保持 batch 形状一致）
)

# 使用：直接 for 循环
for input_ids, labels in loader:
    # input_ids: [32, 340]  ← DataLoader 自动把 32 条拼成一个 batch tensor
    # labels:    [32, 340]
    input_ids = input_ids.to('cuda')
    ...
```

DataLoader 内部做了：
1. 按 `shuffle` 决定取数据的顺序
2. 调 `dataset[i]` 拿 `batch_size` 条数据
3. 用 `collate_fn` 把它们拼成一个 batch（默认是 `torch.stack`）
4. 在后台用 `num_workers` 个进程预加载下一批

#### collate_fn：自定义拼 batch 的逻辑

默认情况下 DataLoader 直接 `torch.stack` 把所有样本拼在一起——前提是所有样本形状相同。如果样本长度不等（padding），你需要自定义 `collate_fn`：

```python
def collate_fn(batch):
    # batch 是一个 list，每个元素是 dataset[i] 的返回值
    input_ids_list, labels_list = zip(*batch)
    # 对齐长度：短的补 pad_token_id，labels 里短的补 -100
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    return input_ids, labels

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

MiniMind 预训练时所有文本已经截断/padding 到 `max_length=340`，形状统一，所以不需要自定义 collate_fn。SFT 阶段文本长度不等时会用到。

---

### 1.6 torch.nn.functional —— 函数式 API

PyTorch 有两种风格做同一件事：**模块式**（`nn.Module` 子类）和**函数式**（`torch.nn.functional`）。

```python
import torch.nn.functional as F

# 模块式：先创建对象，再调用
dropout = nn.Dropout(0.1)
y = dropout(x)

# 函数式：直接调函数，把参数传进去
y = F.dropout(x, p=0.1, training=self.training)
```

**什么时候用哪个？**

| 用模块式 | 用函数式 |
|---------|---------|
| 有可学习参数（`nn.Linear`、`nn.Embedding`） | 无参数的纯计算（`F.softmax`、`F.silu`、`F.cross_entropy`） |
| 有内部状态（`nn.Dropout` 需要知道 train/eval） | 不需要 train/eval 切换的操作 |

MiniMind 代码中常见的函数式调用：

```python
F.cross_entropy(logits, labels, ignore_index=-100)   # loss 计算
F.softmax(scores, dim=-1)                            # attention 权重
F.silu(gate)                                         # SwiGLU 激活
F.scaled_dot_product_attention(q, k, v, ...)         # SDPA（PyTorch 2.0+）
F.dropout(x, p=self.dropout, training=self.training) # dropout
```

---

### 1.7 @dataclass —— 轻量数据容器

Python 标准库的 `dataclasses.dataclass` 用来快速定义"只存数据"的类，省去手写 `__init__`：

```python
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    aux_loss: Optional[torch.Tensor] = None
```

等价于手写：

```python
class ModelOutput:
    def __init__(self, loss=None, logits=None, aux_loss=None):
        self.loss = loss
        self.logits = logits
        self.aux_loss = aux_loss
```

MiniMind 的 `MiniMindForCausalLM.forward` 返回 HuggingFace 的 `MoeCausalLMOutputWithPast`，它就是一个 dataclass：

```python
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# forward 的返回值
return MoeCausalLMOutputWithPast(
    loss=loss,                      # 交叉熵 loss
    logits=logits,                  # [B, S, 6400]
    aux_loss=aux_loss,              # MOE 辅助 loss
    past_key_values=past_key_values # KV Cache
)
```

调用方通过属性名访问：`res.loss`、`res.logits`——比返回一个 tuple 清晰很多。

---

## Part 2：HuggingFace transformers 核心类

MiniMind 的完整模型（`MiniMindForCausalLM`）不只是 `nn.Module`，它还继承了 HuggingFace transformers 的两个类。为什么？因为 HuggingFace 生态提供了一整套"模型发布 → 加载 → 推理"的基础设施，继承这些类就能接入这个生态。

### 2.1 PretrainedConfig —— 超参数管理

#### 先澄清命名："Pretrained" 不是指"预训练阶段"

HuggingFace 里的 `PretrainedConfig`、`PreTrainedModel` 中的 "Pretrained" 意思是**"支持从已训练好的 checkpoint 加载"**——别人训练好了发布到 Hub 上，你能 `from_pretrained()` 一行加载。

它**不是说**"这个 Config 只用于预训练阶段"。Config 存的是模型架构参数（几层、多宽、几个头），和训练阶段无关。同一个 Config，预训练用它、SFT 用它、DPO 用它、推理还是用它——因为模型架构没变，变的只是权重的值。

#### 问题：超参散落各处

如果不用 Config 类，超参数可能散落在代码的各个角落：

```python
# 不好的方式：超参硬编码在各处
model = SomeModel(hidden_size=768, num_layers=8, ...)
trainer = Trainer(lr=1e-4, ...)
# 保存模型时还得单独保存这些参数
# 别人加载你的模型时要猜参数
```

#### 解决：PretrainedConfig

```python
from transformers import PretrainedConfig

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"  # 模型类型标识符，后面解释用途

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        # ... 更多超参
```

继承 `PretrainedConfig` 之后，`MiniMindConfig` 自动获得：

```python
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)

# 能力 1：序列化为 JSON 文件
config.save_pretrained("./my_model/")
# 生成 ./my_model/config.json：
# {
#   "model_type": "minimind",
#   "hidden_size": 768,
#   "num_hidden_layers": 8,
#   "vocab_size": 6400,
#   ...
# }

# 能力 2：从 JSON 反序列化
loaded_config = MiniMindConfig.from_pretrained("./my_model/")
print(loaded_config.hidden_size)  # 768

# 能力 3：从 HuggingFace Hub 加载
# config = MiniMindConfig.from_pretrained("jingyaogong/MiniMind2-Small")
```

#### model_type 的作用

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
```

`model_type` 是 HuggingFace 的模型注册机制的 key。当你把模型上传到 HuggingFace Hub，别人用 `AutoModel.from_pretrained("your-model")` 加载时，transformers 会读 `config.json` 里的 `model_type`，找到对应的模型类来实例化。没有这个字段，`AutoModel` 不知道该用哪个类。

---

### 2.2 PreTrainedModel —— 给模型加上 save/load 能力

`PreTrainedModel` 继承自 `nn.Module`，在它的基础上加了模型序列化、权重初始化等能力。

```
nn.Module                    ← PyTorch 基类
  └── PreTrainedModel        ← HuggingFace 加的一层
        └── MiniMindForCausalLM  ← 我们的模型
```

#### config_class：绑定 Config

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig  # 告诉 HF：我的配置类是 MiniMindConfig
```

有了这个绑定，`from_pretrained` 加载模型时会自动用 `MiniMindConfig` 来解析 `config.json`。

#### _tied_weights_keys：声明权重共享

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

这告诉 HuggingFace：`lm_head.weight` 和 `model.embed_tokens.weight` 是**同一个张量**（Weight Tying，Chapter 01 讲过）。

为什么要显式声明？两个原因：
1. **保存时不重复存**：`state_dict` 里不需要存两份相同的权重
2. **加载时自动绑定**：`from_pretrained` 加载权重后会把这两个 key 指向同一个张量

对应的实现在 `__init__` 里：

```python
def __init__(self, config):
    ...
    self.model = MiniMindModel(self.config)
    self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
    # 让 embedding 和 lm_head 共享同一个权重张量
    if self.config.tie_word_embeddings:
        self.model.embed_tokens.weight = self.lm_head.weight
```

#### post_init()：初始化后的后处理

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()  # ← 在最后调用
```

`post_init()` 是 `PreTrainedModel` 提供的方法，做两件事：
1. **权重初始化**：对所有 `nn.Linear`、`nn.Embedding` 等层应用初始化策略（如正态分布）
2. **梯度 checkpoint 设置**：如果配置了 gradient checkpointing 就启用

必须在 `__init__` 的最后调用，因为此时所有子模块都已经创建完毕。

#### save/load 全流程

```python
model = MiniMindForCausalLM(config)

# 保存：config.json + model.safetensors
model.save_pretrained("./my_model/")
# ./my_model/
# ├── config.json          ← 由 config_class 生成
# └── model.safetensors    ← 所有参数的权重

# 加载：自动读 config.json 创建模型，再灌入权重
model = MiniMindForCausalLM.from_pretrained("./my_model/")
```

这套机制让模型发布变得标准化：任何人只要有模型目录（或 Hub 地址），一行代码就能加载。

---

### 2.3 GenerationMixin —— 通用 generate 方法

#### 先搞清楚：什么是 Mixin 类

普通继承的思路是"子类 **是** 父类的一种"——`Dog` 继承 `Animal`，因为狗是动物。

Mixin 不是这个关系。Mixin 是**"给你附加一组方法，但自己不是一个完整的东西"**。它：

- **没有 `__init__`**（不能单独实例化，单独创建没意义）
- **只提供一组相关方法**（像一个"方法包"）
- **设计上就是被混入到别的类里用的**

```python
# Mixin 的经典例子：
class JsonSerializableMixin:
    """混入这个类的对象自动获得 .to_json() 方法"""
    # 注意：没有 __init__
    def to_json(self):
        return json.dumps(self.__dict__)

class User(DatabaseModel, JsonSerializableMixin):
    # User 继承了两个类：
    # - DatabaseModel：主类，提供 save/load/查数据库
    # - JsonSerializableMixin：附加能力，提供 .to_json()
    def __init__(self, name):
        super().__init__()
        self.name = name

user = User("Alice")
user.save()       # 来自 DatabaseModel
user.to_json()    # 来自 JsonSerializableMixin（附加的）
```

**GenerationMixin 就是这个模式**：它不是一个完整的模型，它只提供 `.generate()` 方法。混入之后，你的模型类就自动有了文本生成能力。

---

`GenerationMixin` 提供了通用的 `.generate()` 方法，支持 greedy search、beam search、sampling（top-k、top-p）等解码策略。

```python
from transformers import GenerationMixin

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    ...
```

继承后，模型就有了 `.generate()` 方法，可以做自回归文本生成。

**但 MiniMind 自己写了一个 `generate` 覆盖了它**：

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    @torch.inference_mode()
    def generate(self, inputs=None, max_new_tokens=8192, temperature=0.85,
                 top_p=0.85, top_k=50, eos_token_id=2, streamer=None,
                 use_cache=True, repetition_penalty=1.0, **kwargs):
        # 自定义的生成逻辑
        ...
```

为什么不用 HuggingFace 自带的？几个原因：
1. **轻量**：HF 的 `generate` 功能非常多（beam search、constrained generation、diverse beam...），但 MiniMind 只需要最基本的 top-k + top-p sampling
2. **可控**：自己写的代码好理解、好调试
3. **教学目的**：手写 generate 能理解自回归生成的每一步细节

实际大规模生产中，大多数团队也会自己写生成逻辑（或用 vLLM 等推理框架），HF 的 `generate` 主要用于快速原型。

---

### 2.4 Python 多继承 —— MRO

```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    ...
```

Python 支持多继承，一个类可以同时继承多个父类。这里 `MiniMindForCausalLM` 同时继承了 `PreTrainedModel`（模型管理能力）和 `GenerationMixin`（生成能力）。

当两个父类有同名方法时，Python 用 **MRO（Method Resolution Order，方法解析顺序）** 决定调哪个。规则是 **C3 线性化**，实际操作中可以用 `.__mro__` 查看：

```python
print(MiniMindForCausalLM.__mro__)
# (MiniMindForCausalLM, PreTrainedModel, ..., GenerationMixin, ..., nn.Module, object)
```

查找顺序：先自己 → 先左后右按继承列表 → 最后到 `object`。如果 `MiniMindForCausalLM` 自己定义了 `generate`，就用自己的（覆盖了 `GenerationMixin` 的默认实现）。如果没定义，就沿 MRO 往上找。

**Mixin 模式的核心思想**：`GenerationMixin` 不是一个完整的类，它只提供 `generate` 相关的方法，不定义 `__init__`。它设计上就是被"混入"到其他类里使用的——给主类"附加"一组方法，而不是作为独立的基类。

---

### 2.5 AutoTokenizer —— 文本和 token id 之间的转换器

模型只能处理数字（token id），不能直接处理文字。Tokenizer 负责**文本 → token id**（编码）和**token id → 文本**（解码）的双向转换。

```python
from transformers import AutoTokenizer

# 加载 MiniMind 的 tokenizer（从本地目录或 Hub）
tokenizer = AutoTokenizer.from_pretrained('./model')

# 编码：文本 → token id 列表
ids = tokenizer.encode("中国首都是北京")
# [1, 3456, 2890, 1234, 5678]
# 其中 1 是 BOS（Begin of Sequence）

# 解码：token id → 文本
text = tokenizer.decode([3456, 2890, 1234, 5678])
# "中国首都是北京"
```

#### 关键属性和方法

```python
tokenizer.vocab_size           # 6400（词表大小）
tokenizer.bos_token_id         # 1（序列开始标记）
tokenizer.eos_token_id         # 2（序列结束标记）
tokenizer.pad_token_id         # 0（填充标记）

# encode：文本 → id 列表
ids = tokenizer.encode("hello")  # [1, 4523, 2891]

# decode：id 列表 → 文本
text = tokenizer.decode(ids)     # "hello"

# __call__：encode 的增强版，返回 dict 包含 attention_mask 等
result = tokenizer("hello", max_length=32, padding='max_length', truncation=True)
# result['input_ids']:      [1, 4523, 2891, 0, 0, ..., 0]  长度 32
# result['attention_mask']:  [1, 1, 1, 0, 0, ..., 0]        有效位置为 1
```

#### BPE（Byte Pair Encoding）：tokenizer 怎么切词

MiniMind 用 BPE 算法把文本切成 token。BPE 不是按字切、也不是按词切，而是**按统计频率学出来的子词单元**切：

```
"训练基础" → ["训练", "基", "础"]     ← 常见词"训练"作为整体
"Transformer" → ["Trans", "former"]  ← 英文也按子词切
```

BPE 的训练过程（Chapter 11 详细讲）：从单字符开始，不断合并出现频率最高的字符对，直到词表大小达到上限（6400）。

#### Auto 前缀的含义

`AutoTokenizer` 里的 "Auto" 和 `AutoModel` 一样——**自动根据 config 里的信息选择正确的 tokenizer 类**。你不需要知道底层是 BPE、SentencePiece 还是 WordPiece，`Auto` 帮你分发。

---

### 2.6 ACT2FN —— 激活函数注册表

`transformers.activations.ACT2FN` 是一个字符串→函数的字典，让你用字符串名字选激活函数：

```python
from transformers.activations import ACT2FN

# ACT2FN 内部大概长这样：
# {
#     "silu": F.silu,
#     "gelu": F.gelu,
#     "relu": F.relu,
#     "tanh": torch.tanh,
#     ...
# }

# 在 FeedForward 中通过 config 选激活函数：
self.act_fn = ACT2FN[config.hidden_act]  # config.hidden_act = "silu"
# 等价于 self.act_fn = F.silu
```

好处是激活函数可以通过 config 切换，不用改代码。MiniMind 用 `"silu"`。

---

### 2.7 torch.distributed（预告，Chapter 17 详讲）

多 GPU 训练需要的分布式通信库。这里只列出你会看到的关键 API，详细原理留到 Chapter 17：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# 初始化进程组（每个 GPU 是一个"进程"）
dist.init_process_group(backend="nccl")

# DDP 包装模型：自动在 backward 时同步各 GPU 的梯度
model = DDP(model, device_ids=[local_rank])

# DistributedSampler：让每个 GPU 只看数据集的一个子集，互不重叠
sampler = DistributedSampler(dataset)
loader = DataLoader(dataset, sampler=sampler)
```

你在 MiniMind 的 `train_pretrain.py` 上游版本里会看到这些代码。单卡训练时（我们的 scratch 版本）不需要它们。

---

## Part 3：MiniMind 代码对照表

回头看 MiniMind 的代码，每个类用了上面哪些概念：

| MiniMind 类 | 继承自 | 用了什么 PyTorch/HF 特性 |
|---|---|---|
| `RMSNorm` | `nn.Module` | `nn.Parameter`（缩放权重 `self.weight`） |
| `Attention` | `nn.Module` | `nn.Linear`（Q/K/V/O 投影）、`nn.Dropout`（attn + resid） |
| `FeedForward` | `nn.Module` | `nn.Linear`（gate/up/down 三个投影） |
| `MOEFeedForward` | `nn.Module` | `nn.Linear`（路由门控）、`nn.ModuleList`（专家列表） |
| `MiniMindBlock` | `nn.Module` | 组合 Attention + FFN + RMSNorm |
| `MiniMindConfig` | `PretrainedConfig` | `model_type`、`save_pretrained` / `from_pretrained` |
| `MiniMindModel` | `nn.Module` | `nn.Embedding`、`nn.ModuleList`（Transformer 层）、`nn.Dropout`、`register_buffer`（RoPE cos/sin） |
| `MiniMindForCausalLM` | `PreTrainedModel` + `GenerationMixin` | `config_class`、`_tied_weights_keys`、`post_init()`、自定义 `generate`、`@torch.inference_mode()` |

注意继承链的层次：

```
nn.Module
├── RMSNorm
├── Attention
├── FeedForward
├── MOEFeedForward
├── MiniMindBlock
├── MiniMindModel          （包含 nn.Embedding + nn.ModuleList + register_buffer）
│
└── PreTrainedModel        （继承自 nn.Module，加了 save/load 能力）
      └── MiniMindForCausalLM  （+ GenerationMixin，最顶层的"可用"模型）
```

底层组件（`RMSNorm`、`Attention`、`FeedForward`）直接继承 `nn.Module`，因为它们只需要最基本的"定义结构 + 前向计算"能力。只有最外层的 `MiniMindForCausalLM` 才需要 HuggingFace 的 save/load/generate 能力，所以只有它继承 `PreTrainedModel`。

`MiniMindModel`（中间层）也只是 `nn.Module`，不需要 HuggingFace 的能力——它被 `MiniMindForCausalLM` 包含，后者负责对外接口。这是一个常见的分层模式：**内部组件用纯 PyTorch，最外层接入生态框架**。

---

## 小结

| 概念 | 一句话 |
|------|--------|
| `nn.Module` | 所有层和模型的基类，提供参数追踪、设备搬移、模式切换、序列化 |
| `nn.Parameter` | 可学习的张量（默认 `requires_grad=True`），optimizer 会更新 |
| `register_buffer` | 不可学习但要跟着 `.to()` 搬移的张量（如 RoPE 的 cos/sin） |
| `nn.ModuleList` | 让 Python list 里的子模块被 `nn.Module` 追踪 |
| `__call__` vs `forward` | `model(x)` 走 `__call__`（触发 hook），永远不要直接调 `forward` |
| `Dataset` / `DataLoader` | Dataset 定义一条数据怎么取，DataLoader 负责 batching/shuffle/并行加载 |
| `F.xxx` | 函数式 API，用于无参数的纯计算（cross_entropy、silu、softmax） |
| `@dataclass` | 轻量数据容器，MiniMind 的 forward 返回值就是 dataclass |
| `inference_mode` | 推理时关掉 autograd，比 `no_grad` 更彻底 |
| `PretrainedConfig` | 超参数打包成对象，支持 JSON 序列化和 Hub 加载（名字里的 "Pretrained" 不是指预训练阶段） |
| `PreTrainedModel` | 给模型加上 `save_pretrained` / `from_pretrained` 能力 |
| `GenerationMixin` | Mixin 类，提供通用 `.generate()` 方法，可覆盖自定义 |
| `AutoTokenizer` | 文本↔token id 的转换器，`Auto` 前缀自动选择正确的 tokenizer 类 |
| `ACT2FN` | 字符串→激活函数的注册表，让激活函数可通过 config 切换 |
| `_tied_weights_keys` | 声明权重共享关系，避免重复保存和加载 |
| `model_type` | 模型注册标识符，`AutoModel` 靠它找到正确的类 |

这些概念在后续的训练篇（Chapter 09-16）和 Chapter 08 的完整模型组装中会反复出现。理解了这些"脚手架"，后面读代码就不会被框架 API 卡住。

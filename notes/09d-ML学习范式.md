# 09d - 机器学习全景：5 个维度理解 ML

> 本章用 5 个正交维度系统梳理机器学习的分类体系：
> 学习范式、问题类型、数据模态、模型架构、数据条件。
> 任何一个 ML 任务都可以用这 5 个维度来定位。

---

## ML 的 5 个分类维度

| 维度 | 回答的问题 | 例子 |
|------|-----------|------|
| **学习范式** | 监督信号从哪来？ | 有监督 / 无监督 / 自监督 / 强化学习 |
| **问题类型** | 做什么任务？ | 分类 / 回归 / 生成 / 检测 / 检索 |
| **数据模态** | 对什么数据做？ | 文本 / 图像 / 语音 / 时序 / 表格 |
| **模型架构** | 用什么结构？ | Transformer / CNN / RNN / GNN / 树模型 |
| **数据条件** | 有多少标注可用？ | full-shot / few-shot / zero-shot / transfer |

5 个维度是**正交的**，可以自由组合。任何 ML 任务都可以用一个 5 元组来描述：

| 任务 | 范式 | 问题类型 | 模态 | 架构 | 数据条件 |
|------|------|---------|------|------|---------|
| GPT 预训练 | 自监督 | 生成 | 文本 | Transformer | full-shot（海量无标注文本） |
| ImageNet 分类 | 有监督 | 分类 | 图像 | CNN/ViT | full-shot（128 万标注图片） |
| ChatGPT 回答问题 | 自监督+SFT+RL | 生成 | 文本 | Transformer | pretrain→SFT→RLHF |
| Stable Diffusion 文生图 | 无监督 | 生成 | 图像+文本 | U-Net/DiT | 弱监督（图文配对） |
| CLIP zero-shot 分类 | 自监督（对比） | 分类 | 图像+文本 | Transformer | zero-shot |
| XGBoost 信用评分 | 有监督 | 分类 | 表格 | 树模型 | full-shot |
| AlphaGo 下围棋 | 强化学习 | 决策 | 棋盘状态 | CNN+MCTS | 自博弈 |
| 推荐系统 | 有监督/RL | 检索排序 | 用户行为 | MLP/Transformer | full-shot |

下面逐个维度展开。

---

## 维度一：学习范式（监督信号从哪来）

### 范式全景图

```
                        Machine Learning
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
   ┌──────┴──────┐    ┌──────┴──────┐    ┌───────┴───────┐
   │  有监督学习   │    │  无监督学习   │    │   强化学习     │
   │ Supervised  │    │ Unsupervised │    │ Reinforcement │
   │             │    │              │    │               │
   │ 有 label    │    │ 无 label     │    │ 有 reward     │
   │ 学 P(y|x)  │    │ 学 P(x) 或   │    │ 最大化累积奖励 │
   │             │    │   P(z|x)    │    │               │
   └──────┬──────┘    └──────┬───────┘    └───────────────┘
          │                  │
          │       ┌──────────┼──────────┐
          │       │          │          │
          │  传统无监督  ┌───┴───┐  生成模型
          │  聚类/降维   │ 自监督  │  VAE/GAN/
          │  密度估计   │  SSL  │  Diffusion
          │            └───┬───┘
          │                │
          │       ┌────────┴────────┐
          │       │                 │
          │   预测类 SSL         重建类 SSL
          │   MLM/CLM            MAE/BEiT
          │
  ┄┄┄┄┄┄┄┼┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄  对比学习横切多种范式  ┄┄┄┄┄┄┄┄┄┄
          │
  ┌───────┴────────────────────────────────────────────────┐
  │  对比学习（Contrastive Learning）= 方法论，不是范式       │
  │                                                         │
  │  ┌──────────────┬──────────────┬──────────────────┐    │
  │  │ + 自监督数据   │ + 有监督数据   │ + 弱监督数据      │    │
  │  │ SimCLR, MoCo │ SupCon, BGE  │ CLIP (图片+alt)  │    │
  │  │ BYOL, DINO   │ 人标 query-doc│ 天然共现配对      │    │
  │  └──────────────┴──────────────┴──────────────────┘    │
  └────────────────────────────────────────────────────────┘
```

**三大顶层范式**（有监督、无监督、强化学习）彼此独立，分类依据是**监督信号从哪来**。自监督学习是无监督的子集。**对比学习不是范式，而是方法论**——它定义的是"怎么学"（拉近正样本、推远负样本），至于正负样本从哪来（自动构造 / 人工标注 / 天然共现），决定了它属于哪种范式。

---

### 1. 有监督学习（Supervised Learning）

#### 1.1 核心思想

训练数据里每个输入 $x$ 都有一个人工标注的正确答案 $y$，模型学习从 $x$ 预测 $y$。

$$\min_\theta \sum_{i} \mathcal{L}(f_\theta(x_i),\ y_i)$$

#### 1.2 两大任务类型

| 任务 | 输出 | Loss | 热门例子 |
|------|------|------|---------|
| 分类 | 离散类别 | Cross-Entropy：$L = -\sum_c y_c \log \hat{y}_c$ | 垃圾邮件检测、人脸识别、情感分析 |
| 回归 | 连续数值 | MSE：$L = \frac{1}{n}\sum(y - \hat{y})^2$ | 房价预测、股价预测 |

还有变体：序列标注（NER）、排序（搜索排序）、结构化预测（机器翻译）。

#### 1.3 有监督的局限

有监督学习效果好，但根本瓶颈是**需要人工标注**：

- 标注成本高：ImageNet-1k 标注了约 128 万张图片（完整的 ImageNet-21k 约 1400 万张）
- 标注质量不一致：不同标注员对同一样本判断可能不同
- 规模受限：互联网上有海量文本，但不可能给每句话标注"正确的下一个词"

这个局限催生了无监督学习和自监督学习：**能不能不要标签，直接从数据本身学？**

---

### 2. 无监督学习（Unsupervised Learning）

#### 2.1 核心思想

训练数据**没有人工标注**，模型从数据本身的分布和结构中学习。两大方向：

- **表示方向**：把高维数据压缩到低维空间（降维、聚类、Autoencoder）
- **生成方向**：学习 $P(x)$，从分布中采样生成新数据（VAE、GAN、Diffusion）

#### 2.2 聚类与降维（传统无监督）

**聚类**：把数据分成若干组，组内相似、组间不同。

| 方法 | 核心思路 | 适用场景 |
|------|---------|---------|
| K-Means | 最小化每个点到簇中心的距离 | 球状簇、已知 K |
| DBSCAN | 基于密度连通性，不需要指定 K | 任意形状簇 |
| GMM | 假设数据由多个高斯分布混合生成 | 软聚类（概率分配）|

**降维**：把高维数据映射到低维空间，便于可视化或去噪。

| 方法 | 核心思路 | 线性/非线性 |
|------|---------|-----------|
| PCA | 找方差最大的方向做投影 | 线性 |
| t-SNE | 保持高维空间中的邻近关系 | 非线性，适合可视化 |
| UMAP | 类似 t-SNE 但更快、保持全局结构 | 非线性 |

#### 2.3 Autoencoder：学压缩表示

**要解决什么问题**：让神经网络自动学到数据的压缩表示——压缩到低维向量（编码），再从低维向量还原原始输入（解码）。

```
输入 x ──→ Encoder ──→ 瓶颈 z ──→ Decoder ──→ 重建 x̂
[784]       (压缩)     [32]       (还原)      [784]
                    (bottleneck)
```

以 MNIST（28x28=784 像素的手写数字）为例：Encoder 把 784 维压到 32 维，Decoder 再从 32 维恢复成 784 维。

**Loss**：重建误差。

$$L_{AE} = \frac{1}{n}\sum_{i=1}^{n} \lVert x_i - \hat{x}_i \rVert^2$$

**为什么属于无监督？** "标签"就是输入本身——不需要人工标注。

**变体**：
- **Denoising AE (DAE)**：给输入加噪声，学去噪——表示更鲁棒
- **Sparse AE**：加正则让瓶颈层激活稀疏——每个维度更有意义

**局限**：学的是确定性映射（$x \to z \to \hat{x}$），瓶颈层向量分布没有约束——**不能随机采样生成新数据**。这催生了 VAE。

#### 2.4 VAE：从压缩到生成

**要解决什么问题**：Autoencoder 能压缩，但不能生成。想在低维空间中**随机采样**生成新的合理数据。

**核心思路**：Encoder 输出不是确定向量 $z$，而是**概率分布的参数**（均值 $\mu$ 和方差 $\sigma^2$），再从分布中采样。

```
输入 x ──→ Encoder ──→ μ, σ² ──→ 采样 z ──→ Decoder ──→ 重建 x̂
                        │
                    z = μ + σ · ε    (ε ~ N(0, I))
                    (reparameterization trick)
```

**Reparameterization Trick**：采样不可求导。解决方法：$z = \mu + \sigma \cdot \epsilon$（$\epsilon \sim \mathcal{N}(0, I)$），把随机性分离出来，梯度可以通过 $\mu$ 和 $\sigma$ 反向传播。

**VAE 的 Loss（ELBO）**：

$$L_{VAE} = \underbrace{-\mathbb{E}_{q(z \mid x)}[\log p(x \mid z)]}_{\text{重建项}} + \underbrace{D_{KL}(q(z \mid x) \| p(z))}_{\text{KL 正则项}}$$

| 项 | 含义 | 作用 |
|---|------|------|
| 重建项 | 从 $z$ 能多好地还原 $x$ | 保证信息不丢失 |
| KL 项 | $q(z \mid x)$ 和标准正态 $p(z) = \mathcal{N}(0, I)$ 的差异 | 约束隐空间形状——让不同输入的 $z$ 都接近标准正态 |

**为什么要 KL 项？** 没有 KL 约束，Encoder 会把不同输入映射到隐空间中毫不相关的区域，随机采样落在"空白区域"，Decoder 不知道怎么解码。KL 项把所有 $z$ 推向标准正态，让隐空间**连续且平滑**——相邻 $z$ 对应相似输出，随机采样也能生成合理数据。

**ELBO 是什么？** Evidence Lower BOund（证据下界）。真正想最大化的是 $\log p(x)$，但算不出来（需要对所有 $z$ 积分）。ELBO 是它的下界，最大化 ELBO 相当于间接最大化 $\log p(x)$。

**$\beta$-VAE**：KL 权重放大为 $\beta > 1$，迫使每个维度独立编码不同语义因素（disentangled representation）。代价是重建质量下降。

#### 2.5 VQ-VAE：用离散编码替代连续分布

**要解决什么问题**：VAE 隐空间是连续高斯分布，但很多数据结构是离散的。能不能让隐空间也**离散**？

**核心思路**：维护一个 **codebook**（码本）——一组固定数量的向量（如 512 个）。Encoder 输出先找 codebook 中最近的向量替代。

```
输入 x ──→ Encoder ──→ z_e ──→ 找最近码本向量 ──→ z_q ──→ Decoder ──→ x̂
                                    │
                              codebook: {e_1, e_2, ..., e_K}
                              z_q = e_k, where k = argmin ‖z_e - e_k‖
```

**Loss** 三部分：

$$L_{VQ} = \underbrace{\lVert x - \hat{x} \rVert^2}_{\text{重建}} + \underbrace{\lVert \text{sg}(z_e) - z_q \rVert^2}_{\text{codebook loss}} + \underbrace{\beta \lVert z_e - \text{sg}(z_q) \rVert^2}_{\text{commitment loss}}$$

- $\text{sg}$ 是 stop-gradient——求梯度时当常数
- argmin 不可导 → 用 **straight-through estimator**：前向用 $z_q$，反向把梯度直接传给 $z_e$

VQ-VAE 是 DALL-E 和视觉 tokenizer 的基础——先把图像编码成离散 token 序列，再用 Transformer 生成。

#### 2.6 GAN：用对抗学生成

**要解决什么问题**：VAE 生成的图像偏模糊（MSE loss 对所有可能输出取平均 → 模糊）。能不能生成更**清晰、逼真**的图像？

**核心思路**：两个网络互相对抗。

```
随机噪声 z ──→ Generator G ──→ 生成图像 G(z) ─┐
                                                ├──→ Discriminator D ──→ 真/假
真实图像 x ──────────────────────────────────────┘
```

- **Generator**：从噪声生成假图像，目标是骗过 D
- **Discriminator**：区分真假图像，目标是不被骗

**原始 GAN Loss（Minimax）**：

$$\min_G \max_D\ \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**实际训练的改进**：

| 问题 | 解决方案 | 代表 |
|------|---------|------|
| G 早期梯度消失 | G 改用 $-\log D(G(z))$ | 原始 GAN 技巧 |
| JS 散度在分布不重叠时梯度为 0 | Wasserstein 距离 | WGAN |
| D 的 Lipschitz 约束 | gradient penalty | WGAN-GP |
| 大规模高分辨率 | spectral normalization + hinge loss | BigGAN |
| 高质量人脸 | progressive growing + style mapping | StyleGAN |

**GAN vs VAE**：
- GAN 生成更清晰（对抗学分布），但训练不稳定（mode collapse：G 只生成几种样本、训练震荡）
- VAE 隐空间规整（KL 约束），训练稳定，但生成偏模糊

#### 2.7 Diffusion Model：加噪声再去噪

**要解决什么问题**：GAN 的训练稳定性问题（见 2.6）始终难以根治。能不能**训练稳定**又**高质量**？

**核心思路**：

```
前向（固定，不需要学）：
x_0 (清晰) → x_1 (微噪) → ... → x_T (纯噪声 ≈ N(0,I))

反向（需要学）：
x_T (纯噪声) → x_{T-1} → ... → x_0 (清晰)
                  ↑ 神经网络预测每一步加的噪声 ε
```

**训练 Loss**：

$$L_{diffusion} = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2\right]$$

本质就是 MSE：让预测噪声和实际噪声一致。训练非常简单，不需要维护 GAN 的对抗平衡。

**Classifier-Free Guidance (CFG)**：条件生成时，同时训练有条件和无条件去噪。推理时：$\epsilon_{guided} = \epsilon_{uncond} + w \cdot (\epsilon_{cond} - \epsilon_{uncond})$，$w > 1$ 放大条件引导。

**热门应用**：Stable Diffusion（文生图）、Midjourney（文生图）、Sora（文生视频）。

**变体**：

| 变体 | 改进 | 代表 |
|------|------|------|
| LDM | 在 VAE 隐空间做扩散，大幅降低计算量 | Stable Diffusion |
| DiT | Transformer 替代 U-Net 做去噪骨架 | Sora 的基础 |
| Flow Matching | 直接回归 ODE 速度场 | SD 3, Flux |

#### 2.8 无监督学习结构选型

| 结构 | 核心能力 | 代表 | 关键特点 |
|------|---------|------|---------|
| Autoencoder | 压缩、去噪 | AE, DAE | 确定性映射，不能生成 |
| VAE | 可控生成、隐空间插值 | VAE, $\beta$-VAE, VQ-VAE | KL 约束，训练稳定 |
| GAN | 高清图像生成 | StyleGAN3 | 对抗训练，清晰但不稳定 |
| Diffusion | SOTA 生成 | LDM, DiT | 训练稳定+质量高，推理慢 |

---

### 3. 自监督学习（Self-Supervised Learning）

#### 3.1 核心思想

SSL 是无监督的子集，核心做法：**从数据本身构造标签（pseudo label）**，把无监督问题转成有监督形式。

GPT 预训练就是典型 SSL——用 next token 作为标签来训练。不需要人工标注（无监督），但显式构造了 pseudo label（自监督）。

**为什么 SSL 成为大模型的基础范式？** 解决了有监督的规模瓶颈：海量无标注数据可以直接用。

#### 3.2 预测类 SSL

> CLM 在 MiniMind 中的完整实现见 Ch11。

**MLM（Masked Language Model）——BERT 的核心**

随机 mask 约 15% token，模型根据上下文预测被 mask 的 token。

$$L_{MLM} = -\sum_{i \in \text{masked}} \log p(x_i \mid x_{\backslash i})$$

双向——能看前后文，适合理解任务。

**CLM（Causal Language Model）——GPT 的核心**

给定前面 token，预测下一个 token。

$$L_{CLM} = -\sum_{t=1}^{T} \log p(x_t \mid x_{<t})$$

单向（causal mask），适合生成。所有 token 都参与 loss，数据利用率比 MLM 高。当前大模型几乎全用 CLM。

**ELECTRA**：用小模型替换部分 token，discriminator 判断每个位置是否被替换（binary CE）。所有 token 位置参与训练，比 MLM 高效。

#### 3.3 重建类 SSL

**MAE（Masked Autoencoder）**

> 注意：MAE 在统计领域是 Mean Absolute Error（平均绝对误差），这里指 Masked Autoencoder。

由何恺明（Kaiming He）等人于 2021 年在 FAIR 提出（论文："Masked Autoencoders Are Scalable Vision Learners"，CVPR 2022）。

随机 mask 75% 图像 patch，重建被 mask 的像素。mask 比例远高于文本（15%），因为图像信息冗余度高——相邻 patch 高度相关，低 mask 比例模型靠插值就能蒙对。

**BEiT**：先用 VQ-VAE 把 patch 离散化成 visual token（离散编码，ID 来自 codebook），再做 MLM 预测 visual token。

**data2vec / I-JEPA**：不重建像素，而是重建 **teacher network 的特征**（smooth L1 loss）。I-JEPA（Yann LeCun 提出）只在 latent space 做预测，不做像素级重建。

#### 3.4 结构选型

| 结构 | SSL 方法 | 代表 | 适用场景 |
|------|---------|------|---------|
| Encoder-only Transformer | MLM / RTD | BERT, ELECTRA | 理解（NLU）|
| Decoder-only Transformer | CLM | GPT, LLaMA, DeepSeek | 生成 / 大模型基座 |
| Encoder-Decoder | Span Corruption | T5, BART | seq2seq |
| ViT + 重建 | MAE / BEiT | MAE, EVA | 视觉 representation |
| ViT + 自蒸馏 | 对比类 | DINO, DINOv2 | 通用视觉特征 |
| 双塔 | 对比类 | CLIP, SigLIP | 多模态对齐 |

---

### 4. 对比学习（Contrastive Learning）

#### 4.1 核心思想

对比学习是**方法论**，不是范式——在 embedding 空间中**拉近正样本对，推远负样本对**。

| 正负样本来源 | 属于什么范式 | 例子 |
|------------|-----------|------|
| 数据增强自动构造 | 自监督 | SimCLR：同一张图做两次裁剪 → 正样本对 |
| 人工标注的配对 | 有监督 | BGE 增训：人标的 query-doc 对 |
| 互联网天然共现 | 弱监督 / 自监督 | CLIP：(图片, alt-text) 对 |

学术界默认指自监督场景（SimCLR、MoCo），但工程中经常搭配有监督数据。

#### 4.2 需要负样本的 Loss

**InfoNCE / NT-Xent**（最核心）：

$$L = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

- $\tau$：温度，典型值 0.05-0.1。太大 → 正负难区分；太小 → 只关注最 hard 负样本
- 本质是 $(N-1)$-way softmax 分类——从负样本中识别出正样本

**Triplet Loss**：$L = \max(0,\ d(a, p) - d(a, n) + \text{margin})$。FaceNet 用这个做人脸验证。

#### 4.3 不需要负样本的 Loss

负样本需要大 batch（SimCLR 用 4096）或 memory bank（MoCo 用 queue）。以下方法绕过这个约束：

| 方法 | 核心机制 | 防 collapse 靠什么 |
|------|---------|------------------|
| **BYOL** | Online network 预测 Target network（EMA 更新）的输出，$L = \lVert q - \text{sg}(z') \rVert^2$ | EMA + stop-gradient 打破对称性 |
| **Barlow Twins** | 让正样本对的 cross-correlation matrix 逼近单位矩阵 | 对角线→1 保信息，非对角线→0 去冗余 |
| **VICReg** | 三项显式正则：Variance（保方差）+ Invariance（正样本近）+ Covariance（去相关） | 每一项都显式防 collapse |
| **SimSiam** | 和 BYOL 类似但没有 EMA，纯靠 stop-gradient | stop-gradient 是唯一关键 |

**Representation Collapse**（表示坍缩）是对比学习的核心技术挑战：所有输入 embedding 坍缩到同一常数向量。解法总结：负样本（推力）、stop-gradient（打破对称）、特征正则化（保方差去相关）、聚类（SwAV 的 Sinkhorn 均匀分配）、centering（DINO）。

#### 4.4 方法选型

| 方法 | 负样本 | 大 batch | 关键机制 | 热门应用 |
|------|:---:|:---:|---------|---------|
| SimCLR | 需要 | 需要 (4096+) | augmentation + projection head | CV 表示学习 |
| MoCo v1/v2/v3 | 需要 | 不需要 | momentum encoder + queue | CV 表示学习 |
| BYOL | 不需要 | 不需要 | EMA + stop-gradient | CV 表示学习 |
| CLIP | 需要 | 需要 (32768) | 双塔 + 双向 InfoNCE | zero-shot 分类、文生图的文本编码器 |
| SigLIP | 隐式 | 不需要 | sigmoid CE 替代 softmax | 多模态理解 |
| DINO | 不需要 | 中等 | self-distillation + centering | 通用视觉特征 |

**Projection Head**（SimCLR 关键发现）：在 backbone 后接 MLP projection head，对比在 projection head 输出空间做。训练完后**去掉 projection head 用 backbone 输出**做下游任务——projection head 吸收了 augmentation 相关信息，backbone 保留更通用语义。

#### 4.5 对比学习在 LLM 中的应用

- **SimCSE**：句子 embedding 的对比学习（Ch21 详细展开）
- **DPO**：chosen vs rejected 本质是正负样本对的对比（Ch14 详细展开）
- **CLIP / SigLIP**：image-text pair + 双向 InfoNCE

---

### 5. 强化学习（Reinforcement Learning）

> LLM 对齐中的 RL（RLHF、DPO、GRPO）在 Ch13-15 已详细推导。

#### 5.1 核心思想

Agent 在环境中通过 trial-and-error 最大化累积奖励。

$$\max_\pi \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

**RL 和有监督的根本区别**：label 告诉你"正确答案是什么"，reward 只告诉你"这次做得好不好"——你得了 +3 分，但可能存在 +10 的动作你没试过。加上数据非 i.i.d.（行为改变环境）、奖励可延迟（credit assignment），这些使 RL 成为独立范式。

#### 5.2 三大算法家族

| 家族 | 核心思路 | 代表 | 适用场景 |
|------|---------|------|---------|
| **Value-Based** | 学 $Q(s,a)$，选 Q 值最大的动作 | DQN, Rainbow | 离散动作（Atari） |
| **Policy-Based** | 直接学策略 $\pi(a \mid s)$ | REINFORCE, PPO | 连续/离散都行 |
| **Actor-Critic** | Actor 选动作，Critic 评价 | A2C, SAC, TD3 | 连续动作（机器人） |

**PPO-Clip**（最常用的 on-policy 算法）：

$$L = -\mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\epsilon,\ 1+\epsilon)\hat{A}_t\right)\right]$$

clip 限制每步更新幅度，防止策略跳太远。是 TRPO 的一阶近似简化版。

**热门应用**：AlphaGo/AlphaZero（围棋）、OpenAI Five（Dota2）、ChatGPT 的 RLHF 对齐、DeepSeek-R1 的 GRPO。

---

### 范式对比表

| 维度 | 有监督 | 无监督（含自监督） | 强化学习 |
|------|-------|-----------------|---------|
| **监督信号** | 人工标注 label | 无 / 数据构造 pseudo label | 环境 reward |
| **数据假设** | i.i.d. | i.i.d. | 序贯、非 i.i.d. |
| **优化目标** | $P(y \mid x)$ | $P(x)$ / pretext task | 累积 reward |
| **核心挑战** | 标注成本 | 评估困难 / pretext 设计 | exploration + credit assignment |
| **数据效率** | 高 | 中-高 | 低 |

对比学习不在此表中——它是方法论（定义"怎么学"），不是范式（定义"监督信号从哪来"）。

### 交叉融合

| 方向 | 做法 | 代表 |
|------|------|------|
| SSL + RL | 序贯决策建模成 sequence prediction | Decision Transformer |
| 自博弈 | RL + SSL：自己和自己对弈生成训练数据 | AlphaZero |
| RLHF 中的对比 | preference pair = 正负样本对 | DPO (Ch14) |
| World Model + SSL | SSL 学环境模型 → planning 替代 model-free RL | I-JEPA, Genie |

---

## 维度二：问题类型（做什么任务）

| 问题类型 | 输出是什么 | 典型 Loss | 热门例子 |
|---------|----------|----------|---------|
| **分类** | 离散类别 | Cross-Entropy | 垃圾邮件检测、ImageNet、情感分析 |
| **回归** | 连续数值 | MSE / MAE (Mean Absolute Error) | 房价预测、年龄估计 |
| **生成** | 新的数据样本 | CE（文本）/ MSE（图像）/ 对抗 | ChatGPT、Stable Diffusion、Sora |
| **检测** | 位置（bbox）+ 类别 | 分类 + 回归 | YOLO（自动驾驶目标检测）、人脸检测 |
| **分割** | 像素级标签 | pixel-wise CE | 医学影像分割、自动驾驶语义分割 |
| **检索/排序** | 排序后的候选列表 | 对比 / pairwise ranking | Google 搜索、推荐系统、RAG 检索 |
| **序列标注** | 每个 token 一个标签 | token-level CE | NER（命名实体识别）、POS tagging |
| **聚类** | 分组 | 组内距离最小化 | 用户分群、文档聚类 |
| **翻译/转换** | 另一种形式的数据 | seq2seq CE | Google Translate、语音转文字 |
| **强化决策** | 动作序列 | policy gradient / TD | AlphaGo、机器人控制 |

**分类 vs 检测 vs 分割的区别**（CV 中容易混淆）：

```
分类：这张图里有猫吗？ → "有" / "没有"（整张图一个标签）
检测：猫在哪？ → 画一个矩形框 [x, y, w, h] + "猫"
分割：哪些像素是猫？ → 每个像素标注 "猫" / "背景"
```

**生成类任务的多样性**：

| 生成什么 | 输入 | 代表 |
|---------|------|------|
| 文本 | 文本 prompt | ChatGPT, Claude, DeepSeek |
| 图像 | 文本描述 | Stable Diffusion, Midjourney, DALL-E 3 |
| 视频 | 文本描述 | Sora, Runway |
| 语音 | 文本 | ChatTTS, Bark |
| 代码 | 自然语言指令 | Cursor, GitHub Copilot |
| 音乐 | 文本描述 | Suno, Udio |

---

## 维度三：数据模态（对什么数据）

| 模态 | 数据长什么样 | 代表任务 | 主流模型 |
|------|-----------|---------|---------|
| **文本 (NLP)** | token 序列 | 生成、翻译、摘要、问答、分类 | GPT, BERT, T5, LLaMA |
| **图像 (CV)** | 像素矩阵 [H, W, C] | 分类、检测、分割、生成 | ResNet, ViT, YOLO, Diffusion |
| **语音/音频** | 波形 / 频谱图 | ASR（语音→文字）、TTS（文字→语音）、说话人识别 | Whisper, ChatTTS |
| **视频** | 图像序列 [T, H, W, C] | 动作识别、视频生成、视频理解 | Sora, VideoMAE |
| **时序数据** | 数值序列 [T, features] | 预测（股价、天气）、异常检测（设备故障） | LSTM, Transformer, Prophet |
| **表格/结构化** | 行×列 | 分类、回归 | XGBoost, LightGBM（树模型至今称霸） |
| **图 (Graph)** | 节点 + 边 | 节点分类、链接预测、社区发现 | GCN, GAT, GraphSAGE |
| **3D/点云** | (x, y, z) 坐标集合 | 3D 检测、三维重建 | PointNet, NeRF |
| **多模态** | 混合以上多种 | 图文理解、文生图、视频问答 | GPT-4o, CLIP, LLaVA |

**模态之间的趋势**：Transformer 正在统一各种模态——文本（GPT）、图像（ViT）、语音（Whisper）、视频（Sora）都在用 Transformer。多模态大模型（GPT-4o, Gemini）进一步把多种模态统一到一个模型里。

**表格数据的特殊性**：在表格/结构化数据上，树模型（XGBoost、LightGBM）至今仍然是首选，Transformer 和深度学习在这个模态上并没有明显优势。原因：表格数据的特征之间没有空间/时序上的局部关系（不像图像有相邻像素、文本有前后 token），树模型的 split-based 决策天然适合处理异构特征和非线性边界。

---

## 维度四：模型架构（用什么结构）

### 架构选型总表

| 架构 | 核心机制 | 擅长什么 | 代表模型 | 现状 |
|------|---------|---------|---------|------|
| **Transformer** | Self-Attention（全局依赖建模） | 几乎所有模态和任务 | GPT, BERT, ViT, Whisper | 当前统治地位 |
| **CNN** | 卷积核（局部特征提取） | 图像、短距离模式 | ResNet, YOLO, U-Net | CV 中仍有用，被 ViT 部分替代 |
| **RNN / LSTM** | 循环连接（序列处理） | 短序列 | LSTM, GRU | 大量被 Transformer 替代 |
| **GNN** | 消息传递（图结构） | 图数据 | GCN, GAT, GraphSAGE | 图数据的首选 |
| **树模型** | 特征分裂决策 | 表格数据 | XGBoost, LightGBM | 表格数据至今称霸 |
| **SSM / Mamba** | 状态空间模型（线性复杂度） | 长序列 | Mamba, RWKV | Transformer 的潜在替代者 |

### Transformer 为什么统治

Transformer（Vaswani et al., 2017, "Attention Is All You Need"）能统治大部分模态的原因：

1. **全局依赖**：Self-Attention 让每个 token 都能直接关注序列中的任意其他 token，不像 RNN 必须逐步传递信息（长距离依赖衰减）
2. **并行训练**：RNN 必须按时间步顺序计算，Transformer 的所有 token 可以并行计算，GPU 利用率高
3. **缩放性**：模型越大、数据越多，效果持续提升（scaling law），这对 CNN/RNN 不明显

**代价**：Self-Attention 的计算复杂度是 $O(n^2)$（$n$ 是序列长度），长序列下计算量和内存都很大。这催生了 SSM/Mamba 等线性复杂度替代方案。

### 各架构的典型用法

| 场景 | 推荐架构 | 理由 |
|------|---------|------|
| LLM / 大模型 | Transformer (Decoder-only) | scaling law + 生成能力 |
| 图像分类 | ViT 或 CNN (ResNet) | ViT 在大数据下更强，CNN 在小数据下更稳 |
| 目标检测 | CNN backbone + Transformer head | YOLO 系列、DETR |
| 表格数据 | XGBoost / LightGBM | 树模型在表格上仍然优于深度学习 |
| 图数据 | GNN | 社交网络、分子结构、推荐系统的图结构 |
| 超长序列 | Mamba / RWKV | 线性复杂度，替代 Transformer 的 $O(n^2)$ |
| 语音 | Transformer (Whisper) 或 CNN+RNN | Whisper 几乎统一了 ASR |

---

## 维度五：数据条件（有多少标注）

| 数据条件 | 含义 | 典型做法 | 热门例子 |
|---------|------|---------|---------|
| **Full-shot** | 有大量标注数据 | 标准有监督训练 | ImageNet 分类（128 万标注图片） |
| **Few-shot** | 只有几个标注样本 | in-context learning / meta-learning | GPT-4 给几个例子就能做新任务 |
| **Zero-shot** | 没见过这个类别/任务 | 靠语言描述或指令泛化 | CLIP（用文本描述分类，不需要训练）、ChatGPT 做没见过的任务 |
| **Semi-supervised** | 少量标注 + 大量无标注 | pseudo labeling / consistency regularization | 医学影像（标注昂贵，大量未标注 X 光片） |
| **Transfer learning** | 在任务 A 上学，迁移到任务 B | pretrain + fine-tune | BERT 在通用文本上预训练 → 微调到情感分类 |
| **Meta-learning** | 学会"学习"本身 | 学习快速适应新任务的能力 | MAML（几步梯度就能适配新任务） |
| **Continual learning** | 持续学新任务，不忘旧任务 | 防灾难性遗忘 | 聊天机器人持续学习新知识 |
| **Active learning** | 模型主动选哪些样本值得标注 | 不确定性采样 | 标注预算有限时，优先标注模型最不确定的样本 |

### 当前主流：Pretrain + Fine-tune

当前 ML 的主流范式是**两阶段**：

```
阶段 1：大规模预训练（SSL，海量无标注数据）
    → 学到通用表示

阶段 2：下游微调（有监督，少量标注数据）
    → 适配到具体任务
```

这本质是 **Transfer Learning + Self-Supervised Learning** 的组合。GPT/BERT/ViT/CLIP 都是这个范式。

**为什么这个范式有效？** 预训练阶段从海量数据中学到了通用的语言/视觉理解能力（语法、语义、视觉特征），微调阶段只需要少量标注就能把这些能力迁移到具体任务。

### Few-shot vs Zero-shot

| 条件 | 给模型什么 | 例子 |
|------|----------|------|
| Few-shot | 几个带答案的例子 | "判断情感：'好吃' → 正面，'难吃' → 负面。请判断：'一般般' → ?" |
| Zero-shot | 只给任务描述，不给例子 | "请判断以下评论的情感是正面还是负面：'一般般'" |

GPT-4 等大模型的 few-shot / zero-shot 能力是 **in-context learning**——不更新模型参数，只通过 prompt 里的例子或指令让模型理解任务。这和传统 few-shot learning（meta-learning，需要梯度更新）在机制上完全不同。

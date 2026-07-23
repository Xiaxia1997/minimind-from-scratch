# ML/DL 面试数学基础 -- todo（整体待系统学习）

> 面试中可能考到的数学知识全览。每个知识点给出核心公式 + 直觉解释 + 面试考法。
> 重点标注 ⭐ = 高频考点，🔥 = Transformer/LLM 相关必考。
>
> **学习进度标注**：✅ = 已学过 / 已理解，❌ = 待学习

---

## 一、线性代数

### 1.1 向量与矩阵基础 ✅

**向量内积（点积）**：

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta
$$

- 衡量两个向量的"方向一致性"
- $\cos \theta = 1$ 完全同向，$0$ 正交，$-1$ 反向
- 🔥 Attention 的 $Q \cdot K$ 就是内积，衡量 query 和 key 的相似度

**矩阵乘法** ⭐：

$$
C = A \cdot B \implies C_{ij} = \sum_{k} A_{ik} B_{kj}
$$

维度规则：$A$ 是 $m \times n$，$B$ 是 $n \times p$ → $C$ 是 $m \times p$（中间维度必须相等，然后消掉）。

**转置性质**：

$$
(AB)^T = B^T A^T
$$

$$
(A^T)^T = A
$$

$$
(A + B)^T = A^T + B^T
$$

**逐元素乘法（Hadamard 积）**：

$$
(A \odot B)_{ij} = A_{ij} \cdot B_{ij}
$$

- 两个矩阵必须同形
- 🔥 Mask 机制就是 Hadamard 积：$y = S \odot M$

### 1.2 特殊矩阵 ✅（单位矩阵与残差连接、对角矩阵与 BatchNorm 已理解）

| 类型 | 性质 | ML 中出现在 |
|------|------|------------|
| 单位矩阵 $I$ | $AI = IA = A$ | 残差连接 $y = x + F(x)$ |
| 对角矩阵 | 只有对角线非零 | BatchNorm 的 scale |
| 对称矩阵 | $A = A^T$ | 协方差矩阵、核矩阵 |
| 正交矩阵 | $A^T A = I$ | 权重初始化 |
| 正定矩阵 | 所有特征值 $> 0$（特征值见 1.3） | $A^T A$ 半正定、二阶优化 |

### 1.3 矩阵乘向量与线性变换 ✅

在 1.1 节我们学了矩阵乘矩阵的维度规则：$A$ 是 $m \times n$，$B$ 是 $n \times p$，结果是 $m \times p$。

**向量就是"只有一列的矩阵"**。一个 $n$ 维列向量 $\mathbf{x}$ 可以看作 $n \times 1$ 的矩阵。所以矩阵乘向量就是矩阵乘法的特例：

$$A: m \times n, \quad \mathbf{x}: n \times 1 \quad \Rightarrow \quad A\mathbf{x}: m \times 1$$

**数字例子**：

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

$$A\mathbf{x} = \begin{bmatrix} 2 \times 1 + 1 \times 0 \\ 1 \times 1 + 3 \times 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

输入是 $[1, 0]$（指向 x 轴正方向），输出变成了 $[2, 1]$（既不在 x 轴上了，长度也变了）。这就是矩阵对向量的作用：**一般情况下，既改变方向，又改变长度**。

再试一个向量：

$$\mathbf{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad A\mathbf{x} = \begin{bmatrix} 2+1 \\ 1+3 \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$$

输入是 $[1, 1]$（45° 方向），输出是 $[3, 4]$——方向变了（不再是 45°），长度也变了。

**小结**：矩阵 $A$ 乘以向量 $\mathbf{x}$，就是对 $\mathbf{x}$ 做一次"线性变换"——拉伸、压缩、旋转、翻转的组合。对于大多数向量，方向和长度都会改变。

### 1.4 行列式与矩阵可逆性 ⭐ ❌

#### 行列式（determinant）

行列式是一个**把方阵映射成一个标量**的运算，记作 $\det(A)$ 或 $|A|$。

**2×2 矩阵的行列式**（直接背公式）：

$$\det \begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**数字例子**：

$$\det \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} = 4 \times 3 - 1 \times 2 = 10$$

$$\det \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix} = 2 \times 2 - 4 \times 1 = 0$$

#### 行列式的几何含义

$|\det(A)|$ = 矩阵 $A$ 对应的线性变换**对面积的缩放倍数**。

- $\det(A) = 10$：面积变成原来的 10 倍
- $\det(A) = 0$：面积被压成零（整个平面被压成一条线甚至一个点）
- $\det(A) < 0$：面积缩放的同时，方向翻转（镜像）

**直觉**：把一个单位正方形丢进矩阵 $A$，看它变成什么形状。行列式就是变换后面积和原面积的比值。

#### 3×3 矩阵的行列式

沿第一行展开（cofactor expansion）：

$$\det \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = a(ei - fh) - b(di - fg) + c(dh - eg)$$

规则：第一行的每个元素，乘以它对应的"余子式"（划掉所在行和列剩下的 2×2 行列式），符号交替 $+, -, +$。

更大的矩阵同理递归展开，但实际中几乎不手算，都用算法。

#### 逆矩阵

如果存在矩阵 $A^{-1}$ 使得：

$$A A^{-1} = A^{-1} A = I \quad \text{（单位矩阵）}$$

就说 $A$ 是**可逆的**（invertible），$A^{-1}$ 叫 $A$ 的逆矩阵。

直觉：$A$ 做了一个变换，$A^{-1}$ 做的是**完全反向的变换**，两个连着做等于什么都没做（$I$）。

#### 可逆 ↔ 行列式不为零

**核心定理**：

$$A \text{ 可逆} \iff \det(A) \neq 0$$

$$A \text{ 不可逆（奇异）} \iff \det(A) = 0$$

**为什么？** 回忆行列式的几何含义：$\det(A) = 0$ 意味着面积被压成零——信息丢失了。一条线不可能恢复成一个平面，所以没有逆变换。

**数字例子**：

$$A = \begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix}, \quad \det(A) = 0$$

$A$ 的第二列 $= 2 \times$ 第一列，两列线性相关。$A$ 把所有输入压到同一条线上：

$$A \begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}2\\1\end{bmatrix}, \quad A \begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}4\\2\end{bmatrix} = 2 \begin{bmatrix}2\\1\end{bmatrix}$$

不管输入什么，输出都在 $[2, 1]$ 方向上——整个 2D 平面被压成 1D 的线。不同的输入映射到同一个输出，信息丢失，不可能反推回去。

#### "不可逆"的另一个含义：$A\mathbf{x} = \mathbf{0}$ 有非零解

如果 $A$ 可逆，$A\mathbf{x} = \mathbf{0}$ 两边乘 $A^{-1}$ 得 $\mathbf{x} = A^{-1}\mathbf{0} = \mathbf{0}$，只有零解。

如果 $A$ 不可逆（$\det(A) = 0$），就存在非零的 $\mathbf{x}$ 使得 $A\mathbf{x} = \mathbf{0}$——某些非零向量被 $A$ "压死"了。

这个性质在 1.6 求特征值时要用到。

### 1.5 线性无关与秩（rank） ⭐ ❌

#### 线性无关——"不能被其他向量凑出来"

一组向量**线性无关**，意味着其中任何一个都不能写成其他向量的线性组合。

**例 1**：$[1, 0]$ 和 $[0, 1]$——方向完全不同，谁也凑不出谁。✅ 线性无关。

**例 2**：$[1, 2]$ 和 $[2, 4]$——第二个 $= 2 \times$ 第一个，方向相同。❌ 线性相关。

**例 3**：$[1, 0]$、$[0, 1]$、$[1, 1]$——第三个 = 第一个 + 第二个，可以被凑出来。❌ 线性相关。

#### 秩——"有多少个独立方向"

**秩 = 矩阵的列向量中，线性无关的最多有几个。**

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 6 \\ 5 & 10 \end{bmatrix}$$

第二列 $= 2 \times$ 第一列，两列线性相关，独立方向只有 1 个 → $\text{rank}(A) = 1$。

$$B = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

两列方向不同（$[1,0,1]$ 和 $[0,1,1]$ 谁也凑不出谁） → $\text{rank}(B) = 2$。

#### 满秩

$m \times n$ 矩阵的秩最大是 $\min(m, n)$，达到上限就叫**满秩**。

对于 $n \times n$ 方阵，四个说法等价：

$$\text{满秩}（\text{rank} = n）\iff n \text{ 列线性无关} \iff \text{可逆} \iff \det \neq 0$$

回忆 1.4：不可逆 = 信息丢失 = 有方向被压死 = 列之间有"冗余"（线性相关） = 不满秩。

#### 为什么 LLM 里关心秩

神经网络的权重矩阵（比如 $W_Q$：$768 \times 768$）理论上是满秩的（秩 = 768），但研究发现微调时的**权重变化量** $\Delta W$ 通常集中在一个低维子空间——秩远小于 768。这是 LoRA 的出发点（1.9 详讲）。

### 1.6 特征值与特征分解 ⭐ ❌

#### 动机：找"只缩放不旋转"的特殊方向

1.3 节看到矩阵乘向量一般会同时改变方向和长度。但有没有一些**特殊的向量**，被 $A$ 乘了之后**方向不变，只是被拉长或缩短**？

如果能找到这些特殊方向，就可以把矩阵的行为拆解成"每个方向上独立缩放"——从而真正理解一个矩阵到底在"做什么"。

#### 定义

如果一个非零向量 $\mathbf{v}$ 满足：

$$A \mathbf{v} = \lambda \mathbf{v}$$

就叫 $\mathbf{v}$ 是 $A$ 的**特征向量**（eigenvector），$\lambda$ 是对应的**特征值**（eigenvalue）。

等式的含义：$A$ 乘以 $\mathbf{v}$，结果等于 $\mathbf{v}$ 自己乘以一个标量 $\lambda$——方向不变（或反向），只是长度被缩放了 $\lambda$ 倍。

| $\lambda$ 的值 | 几何含义 | 例子 |
|---------|---------|------|
| $\lambda > 1$ | 沿该方向拉伸 | $\lambda = 5$：拉长 5 倍 |
| $0 < \lambda < 1$ | 沿该方向压缩 | $\lambda = 0.5$：缩短一半 |
| $\lambda = 1$ | 沿该方向不变 | 单位矩阵的所有特征值都是 1 |
| $\lambda = 0$ | 该方向被压扁到零（信息丢失） | 奇异矩阵 |
| $\lambda < 0$ | 方向反转再缩放 | $\lambda = -2$：翻转并拉长 2 倍 |

#### 先用直觉"猜"，再验证

用 1.3 的矩阵：

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$$

**猜 $\mathbf{v} = [1, 1]^T$**：

$$A\begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix}3\\4\end{bmatrix} \neq \lambda \begin{bmatrix}1\\1\end{bmatrix}$$

$[3, 4]$ 不等于 $[1, 1]$ 的倍数——不是特征向量。显然不能靠猜，需要一个系统的方法来求。

#### 推导特征方程

从定义出发：

$$A\mathbf{v} = \lambda\mathbf{v}$$

右边的 $\lambda\mathbf{v}$ 可以写成 $\lambda I \mathbf{v}$（$I$ 是单位矩阵），这样两边形式统一：

$$A\mathbf{v} - \lambda I \mathbf{v} = \mathbf{0} \implies (A - \lambda I)\mathbf{v} = \mathbf{0}$$

现在问题变成：矩阵 $(A - \lambda I)$ 乘以向量 $\mathbf{v}$ 等于零向量。我们要找**非零**的 $\mathbf{v}$（零向量不算特征向量）。

根据 1.4 的结论：$A\mathbf{x} = \mathbf{0}$ 有非零解 $\iff$ $A$ 不可逆 $\iff$ $\det(A) = 0$。

所以条件是：

$$\det(A - \lambda I) = 0 \quad \text{（特征方程）}$$

解这个关于 $\lambda$ 的方程，得到特征值；把特征值代回 $(A - \lambda I)\mathbf{v} = \mathbf{0}$，解出特征向量。

#### 数字例子：完整手算

$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

**第一步：构造 $A - \lambda I$**

$$A - \lambda I = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix}$$

**第二步：列特征方程 $\det(A - \lambda I) = 0$**

用 2×2 行列式公式 $ad - bc$（1.4 节）：

$$\det \begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix} = (4-\lambda)(3-\lambda) - 1 \times 2$$

展开 $(4-\lambda)(3-\lambda)$：

$$(4-\lambda)(3-\lambda) = 12 - 4\lambda - 3\lambda + \lambda^2 = \lambda^2 - 7\lambda + 12$$

所以：

$$\lambda^2 - 7\lambda + 12 - 2 = \lambda^2 - 7\lambda + 10 = 0$$

**第三步：解方程**

$$(\lambda - 5)(\lambda - 2) = 0 \implies \lambda_1 = 5, \quad \lambda_2 = 2$$

**第四步：回代求特征向量**

**$\lambda_1 = 5$**：解 $(A - 5I)\mathbf{v} = \mathbf{0}$

$$A - 5I = \begin{bmatrix} 4-5 & 1 \\ 2 & 3-5 \end{bmatrix} = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}$$

写成方程组：

$$-v_1 + v_2 = 0 \implies v_1 = v_2$$

$$2v_1 - 2v_2 = 0 \implies v_1 = v_2 \quad \text{（和第一行一样，不提供新信息）}$$

取 $v_1 = v_2 = 1$，得 $\mathbf{v}_1 = \begin{bmatrix}1\\1\end{bmatrix}$

（特征向量的长度不唯一，$[2, 2]^T$、$[100, 100]^T$ 也是，只要方向对就行。）

**$\lambda_2 = 2$**：解 $(A - 2I)\mathbf{v} = \mathbf{0}$

$$A - 2I = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}$$

$$2v_1 + v_2 = 0 \implies v_2 = -2v_1$$

取 $v_1 = 1$，得 $\mathbf{v}_2 = \begin{bmatrix}1\\-2\end{bmatrix}$

**第五步：验证**

$$A\mathbf{v}_1 = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix} = 5 \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \lambda_1 \mathbf{v}_1 \quad \checkmark$$

$$A\mathbf{v}_2 = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 4-2 \\ 2-6 \end{bmatrix} = \begin{bmatrix} 2 \\ -4 \end{bmatrix} = 2 \begin{bmatrix} 1 \\ -2 \end{bmatrix} = \lambda_2 \mathbf{v}_2 \quad \checkmark$$

**几何含义**：矩阵 $A$ 在 $[1, 1]$ 方向上拉伸 5 倍，在 $[1, -2]$ 方向上拉伸 2 倍。知道了这两个"主轴"和对应的缩放系数，就完全理解了 $A$ 的几何行为。

#### 特征分解（对角化）

知道了特征值和特征向量之后，能用它们把矩阵 $A$ "拆开"。

如果一个 $n \times n$ 方阵有 $n$ 个线性无关（1.5 节）的特征向量，可以分解为：

$$A = P \Lambda P^{-1}$$

- $P$：列是特征向量 $[\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n]$
- $\Lambda$：对角矩阵，对角线是对应的特征值 $\text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$

**为什么成立**：把 $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$ 写成矩阵形式：

$$A [\mathbf{v}_1, \mathbf{v}_2] = [\lambda_1 \mathbf{v}_1, \lambda_2 \mathbf{v}_2]$$

左边就是 $AP$。右边可以拆成 $P \Lambda$：

$$[\lambda_1 \mathbf{v}_1, \lambda_2 \mathbf{v}_2] = [\mathbf{v}_1, \mathbf{v}_2] \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix} = P \Lambda$$

所以 $AP = P\Lambda$，两边右乘 $P^{-1}$：$A = P\Lambda P^{-1}$。

前提：$P$ 必须可逆，即 $P$ 满秩（1.5 节），即 $n$ 个特征向量线性无关。

**数字例子**：

$$P = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}, \quad \Lambda = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}$$

$$A = P \Lambda P^{-1} = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix} \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}^{-1}$$

**几何含义**：$A = P \Lambda P^{-1}$ 把"矩阵对向量的变换"拆成三步：

1. $P^{-1}$：换到特征向量坐标系（把向量用特征向量作为基来表示）
2. $\Lambda$：在每个特征方向上独立缩放（乘以对应的特征值）
3. $P$：换回原坐标系

**直觉**：在普通坐标系下，$A$ 的行为很复杂（既旋转又拉伸）。但换到特征向量坐标系下，$A$ 的行为变成了对角矩阵——每个轴独立缩放，简单到不能再简单。

#### 特征分解的限制

特征分解 $A = P\Lambda P^{-1}$ 有两个限制：

**限制 1：只能用于方阵**。特征值定义为 $A\mathbf{v} = \lambda\mathbf{v}$，要求 $A\mathbf{v}$ 和 $\mathbf{v}$ 维度相同，所以 $A$ 必须是 $n \times n$ 方阵。非方阵（比如 $3 \times 2$）没有特征值的概念。

**限制 2：$P$ 必须满秩（$n$ 个特征向量线性无关）**。即使是方阵，如果找不到 $n$ 个线性无关的特征向量，$P$ 不满秩、不可逆，分解做不了。这种情况叫"不可对角化"。

**不可对角化的例子**：

$$A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$$

特征方程：$\det \begin{bmatrix} -\lambda & 1 \\ 0 & -\lambda \end{bmatrix} = \lambda^2 = 0$，只有 $\lambda = 0$（重根）。

代回求特征向量：$\begin{bmatrix}0 & 1 \\ 0 & 0\end{bmatrix}\mathbf{v} = \mathbf{0}$，得 $v_2 = 0$，只有 $\mathbf{v} = [1, 0]^T$ 这一个方向。

$P$ 需要 2 列才能是 $2 \times 2$ 方阵，但只有 1 个特征向量 → $P$ 凑不成方阵 → 无法对角化。

**总结**：

| 情况 | 能否特征分解 | 例子 |
|------|-------------|------|
| $n \times n$ 方阵，$P$ 满秩（$n$ 个独立特征向量） | ✅ 可以 | 上面的 $A$（大多数方阵都行） |
| 实对称矩阵（$A = A^T$） | ✅ 一定可以，且特征值全是实数，特征向量正交 | 协方差矩阵 $X^T X$ |
| 非方阵（$m \times n$，$m \neq n$） | ❌ 特征值只对方阵定义 | $3 \times 2$ 矩阵 |
| 方阵但 $P$ 不满秩（特征向量不够 $n$ 个） | ❌ 不能分解成 $P\Lambda P^{-1}$ | 上面 $\begin{bmatrix}0&1\\0&0\end{bmatrix}$ 的例子 |
| 实矩阵但特征值是复数 | 有复数特征值，实数域无法对角化 | 旋转矩阵 $\begin{bmatrix}0&-1\\1&0\end{bmatrix}$（$\lambda = \pm i$） |

**结论**：特征分解很有用，但受限于方阵且 $P$ 满秩。1.7 节的 SVD 会突破这两个限制。

#### 对称矩阵的特殊性

实对称矩阵（$A = A^T$）有两个极好的性质：

1. **特征值一定是实数**（不会出现复数）
2. **特征向量一定互相正交**（互相垂直，内积为 0）

这意味着对称矩阵**一定能做特征分解**，而且分解出来的 $P$ 是正交矩阵（$P^T P = I$），此时特征分解可以写成更漂亮的形式：

$$A = Q \Lambda Q^T \qquad \text{（$Q$ 正交，$Q^{-1} = Q^T$）}$$

**为什么关心对称矩阵？** 因为 ML 中的协方差矩阵 $X^T X$、核矩阵（kernel matrix）都是对称矩阵。对称矩阵的这些好性质是 1.7 节 SVD 和 1.8 节 PCA 的数学基础。

#### 秩和特征值的关系

回忆 1.5 节讲的秩。特征值 $\lambda = 0$ 意味着该方向被"压死"了。一个方阵有多少个非零特征值，就有多少个"存活"的方向——这正好就是它的秩。

#### 面试考法

- "特征值和特征向量的几何含义？" → 特征向量是矩阵"只缩放不旋转"的方向，特征值是对应的缩放系数。特征分解把矩阵拆成"换坐标系→各轴独立缩放→换回来"。
- "为什么 PCA 找最大特征值的特征向量？" → 协方差矩阵的特征值 = 对应方向上的方差。最大特征值 = 数据变化最大的方向 = 信息最多的方向。
- "对称矩阵的特征值一定是实数，为什么？" → 因为 $A = A^T$ 保证了特征方程的判别式非负（2×2 情况可手算验证；一般情况用 $\lambda = \mathbf{v}^T A \mathbf{v} / \mathbf{v}^T \mathbf{v}$ 证明结果是实数）。
- "什么情况下矩阵不能做特征分解？" → 两种：非方阵（特征值只对方阵定义）、方阵但特征向量不够 $n$ 个（$P$ 不可逆）。

### 1.7 SVD（奇异值分解） ⭐ ❌

#### 动机：突破特征分解的两个限制

1.6 节我们知道了特征分解 $A = P\Lambda P^{-1}$ 有两个限制：只能用于方阵，且方阵还必须可对角化。

但实际中大量矩阵是非方阵的。比如一张 $1000 \times 2000$ 的图片、一个 $n \times d$ 的数据矩阵（$n$ 个样本、$d$ 个特征）。我们需要一种**对任意形状、任意矩阵都适用**的分解方法——这就是 SVD。

#### 数学基础：为什么 $A^T A$ 是关键

在讲 SVD 公式之前，先理解一个关键事实：**$A^T A$ 永远是实对称方阵，永远能做特征分解**。

**为什么 $A^T A$ 一定是方阵**：$A$ 是 $m \times n$，$A^T$ 是 $n \times m$，所以 $A^T A$ 是 $n \times n$——方阵。

**为什么 $A^T A$ 一定对称**：

$$(A^T A)^T = A^T (A^T)^T = A^T A$$

转置等于自己，所以是对称矩阵。根据 1.6 节对称矩阵的性质，**一定能做特征分解**，特征值一定是实数，特征向量一定正交。

**为什么 $A^T A$ 的特征值一定非负**：

设 $\mathbf{v}$ 是 $A^T A$ 的特征向量，$\lambda$ 是对应的特征值：$A^T A \mathbf{v} = \lambda \mathbf{v}$

两边左乘 $\mathbf{v}^T$：

$$\mathbf{v}^T A^T A \mathbf{v} = \lambda \mathbf{v}^T \mathbf{v}$$

左边 $= (A\mathbf{v})^T (A\mathbf{v}) = \|A\mathbf{v}\|^2 \geq 0$，右边 $\mathbf{v}^T \mathbf{v} = \|\mathbf{v}\|^2 > 0$（$\mathbf{v} \neq 0$），所以 $\lambda \geq 0$。

**小结**：不管 $A$ 是什么形状、什么内容，$A^T A$ 都是对称、半正定的方阵，一定能做特征分解，特征值全是非负实数。

#### 从 $A^T A$ 的特征分解，正向推导出 SVD

下面不做任何假设，从"$A^T A$ 能特征分解"出发，一步一步**构造**出 $A = U\Sigma V^T$。

##### 第一步：对 $A^T A$ 做特征分解，得到 $V$ 和 $\Sigma$

$A^T A$ 是实对称方阵（上面已证），所以一定能做特征分解（1.6 节）：

$$A^T A = V \Lambda V^T$$

其中 $V$ 是正交矩阵（列是特征向量 $\mathbf{v}_1, \ldots, \mathbf{v}_n$），$\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$，所有 $\lambda_i \geq 0$（上面已证）。

定义**奇异值** $\sigma_i = \sqrt{\lambda_i}$，排成对角矩阵 $\Sigma$（$m \times n$，对角线是 $\sigma_1, \sigma_2, \ldots$）。

到这里，**$V$ 和 $\Sigma$ 都有了**。

##### 第二步：用 $V$ 和 $\Sigma$ 构造 $U$

对每个 $\sigma_i > 0$，定义：

$$\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$$

现在需要验证这些 $\mathbf{u}_i$ 是正交归一的（否则 $U$ 不是正交矩阵，构造就失败了）。

算 $\mathbf{u}_i$ 和 $\mathbf{u}_j$ 的内积：

$$\mathbf{u}_i^T \mathbf{u}_j = \frac{1}{\sigma_i \sigma_j} (A\mathbf{v}_i)^T (A\mathbf{v}_j) = \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T A^T A \mathbf{v}_j$$

因为 $\mathbf{v}_j$ 是 $A^T A$ 的特征向量，所以 $A^T A \mathbf{v}_j = \lambda_j \mathbf{v}_j = \sigma_j^2 \mathbf{v}_j$：

$$= \frac{1}{\sigma_i \sigma_j} \mathbf{v}_i^T (\sigma_j^2 \mathbf{v}_j) = \frac{\sigma_j}{\sigma_i} \mathbf{v}_i^T \mathbf{v}_j$$

$V$ 是正交矩阵，所以 $\mathbf{v}_i^T \mathbf{v}_j = \delta_{ij}$（$i = j$ 时为 1，否则为 0）：

$$\mathbf{u}_i^T \mathbf{u}_j = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

**$\mathbf{u}_i$ 确实正交归一**，把它们排成矩阵就是正交矩阵 $U$。✓

##### 第三步：验证 $A = U\Sigma V^T$

从 $\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$ 出发，两边乘以 $\sigma_i$：

$$\sigma_i \mathbf{u}_i = A \mathbf{v}_i$$

这对所有 $i$（$\sigma_i > 0$）成立。写成矩阵形式：

$$A V = U \Sigma$$

两边右乘 $V^T$（$V$ 正交，所以 $V V^T = I$）：

$$A = U \Sigma V^T \quad \checkmark$$

##### 小结

整个推导没有假设 SVD 存在，而是从 $A^T A$ 的特征分解出发，**构造性地造出了** $U$、$\Sigma$、$V$，并验证了它们满足 $A = U\Sigma V^T$。

关键在于 $A^T A$ 是实对称方阵——这保证了第一步的特征分解一定成功（特征值实数且非负、特征向量正交且够 $n$ 个），从而整条构造链都能走通。这就是 SVD 能对**任意矩阵**使用的根本原因。

#### SVD 公式

$$A = U \Sigma V^T$$

- $A$：$m \times n$ 矩阵（要分解的对象）
- $U$：$m \times m$ 正交矩阵（$U^T U = I$，列向量叫"左奇异向量"）
- $\Sigma$：$m \times n$ 对角矩阵（对角线上是**奇异值** $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$，从大到小排列）
- $V$：$n \times n$ 正交矩阵（$V^T V = I$，列向量叫"右奇异向量"）

#### 和特征分解的对比

| | 特征分解 $A = P\Lambda P^{-1}$ | SVD $A = U\Sigma V^T$ |
|--|--|--|
| 适用范围 | 方阵 + 可对角化 | **任意矩阵** |
| 分解成几块 | 3 块（$P$、$\Lambda$、$P^{-1}$） | 3 块（$U$、$\Sigma$、$V^T$） |
| 中间矩阵 | 特征值对角阵 | 奇异值对角阵 |
| 左右矩阵 | 同一个 $P$（及其逆） | **两个不同的**正交矩阵 $U, V$ |

特征分解用同一组基做"换坐标系→缩放→换回来"，而 SVD 用两组不同的基——这就是它能处理非方阵的原因（输入空间和输出空间维度不同，自然需要两组基）。

#### 几何直觉

任何线性变换（矩阵乘法）都可以拆成三步——旋转 → 拉伸/压缩 → 再旋转：

$$\underbrace{A}_{\text{原始变换}} = \underbrace{U}_{\text{旋转 2}} \underbrace{\Sigma}_{\text{各轴拉伸}} \underbrace{V^T}_{\text{旋转 1}}$$

#### 怎么算——三步法

**第一步**：对 $A^T A$ 做特征分解（1.6 节） → 得到特征值 $\lambda_i$ 和特征向量 $\mathbf{v}_i$
- 特征向量排成矩阵就是 $V$
- 奇异值 $\sigma_i = \sqrt{\lambda_i}$，排成对角矩阵就是 $\Sigma$

**第二步**：用公式 $\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$ 算出 $U$ 的每一列

**第三步**：验证 $A = U\Sigma V^T$

#### 数字例子：$2 \times 2$ 矩阵手算

$$A = \begin{bmatrix} 4 & 0 \\ 3 & 5 \end{bmatrix}$$

这个矩阵不对称（$A \neq A^T$），不能直接用特征分解拆。但 SVD 总是可以做的。

**第一步：算 $A^T A$ 并做特征分解**

$$A^T A = \begin{bmatrix} 4 & 3 \\ 0 & 5 \end{bmatrix} \begin{bmatrix} 4 & 0 \\ 3 & 5 \end{bmatrix} = \begin{bmatrix} 25 & 15 \\ 15 & 25 \end{bmatrix}$$

特征方程 $\det(A^T A - \lambda I) = 0$：

$$(25 - \lambda)^2 - 225 = 0 \implies \lambda^2 - 50\lambda + 400 = 0$$

$$\lambda_1 = 40, \quad \lambda_2 = 10$$

奇异值 = 特征值的正平方根：$\sigma_1 = \sqrt{40} \approx 6.32$，$\sigma_2 = \sqrt{10} \approx 3.16$

对应的特征向量（归一化后）就是 $V$ 的列：

$$\lambda_1 = 40: \quad \mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \qquad \lambda_2 = 10: \quad \mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$

$$V = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}, \qquad \Sigma = \begin{bmatrix} \sqrt{40} & 0 \\ 0 & \sqrt{10} \end{bmatrix}$$

**第二步：算 $U$**

$$\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$$

$$\mathbf{u}_1 = \frac{1}{\sqrt{40}} \begin{bmatrix} 4 & 0 \\ 3 & 5 \end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{80}} \begin{bmatrix} 4 \\ 8 \end{bmatrix} = \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

$$\mathbf{u}_2 = \frac{1}{\sqrt{10}} \begin{bmatrix} 4 & 0 \\ 3 & 5 \end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{20}} \begin{bmatrix} -4 \\ 2 \end{bmatrix} = \frac{1}{\sqrt{5}} \begin{bmatrix} -2 \\ 1 \end{bmatrix}$$

**第三步：验证**

$$U \Sigma V^T = \frac{1}{\sqrt{5}} \begin{bmatrix} 1 & -2 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} \sqrt{40} & 0 \\ 0 & \sqrt{10} \end{bmatrix} \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 4 & 0 \\ 3 & 5 \end{bmatrix} \checkmark$$

**奇异值的含义**：$\sigma_1 = 6.32$ 远大于 $\sigma_2 = 3.16$——说明 $A$ 在 $\mathbf{v}_1$ 方向上的"拉伸"是 $\mathbf{v}_2$ 方向的 2 倍。$\sigma_1$ 方向承载了更多"信息"。

#### 秩与奇异值的关系

**关键事实：矩阵的秩（1.5 节）= 非零奇异值的个数。**

上面的 $A$ 有 2 个非零奇异值（$\sigma_1 = 6.32, \sigma_2 = 3.16$），所以 $\text{rank}(A) = 2$（满秩）。

再看 1.5 节的例子：

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 6 \\ 5 & 10 \end{bmatrix}$$

做 SVD 会得到 $\sigma_1 = \sqrt{135} \approx 11.6$，$\sigma_2 = 0$。只有 1 个非零奇异值 → $\text{rank}(A) = 1$，和我们在 1.5 节通过观察"两列共线"得出的结论一致。

**为什么这很重要？** 因为很多真实世界的矩阵，虽然技术上满秩（所有奇异值 > 0），但大部分奇异值非常小、接近于零。这意味着**信息实际上集中在少数几个方向上**——矩阵"本质上"是低秩的，那些小奇异值只是噪声。

#### 面试考法

- "SVD 和特征分解什么关系？" → 特征分解只能用于方阵，SVD 对任意矩阵适用。SVD 通过对 $A^T A$（对称矩阵，保证能特征分解）找到奇异值和右奇异向量。
- "SVD 的几何含义？" → 任何线性变换都可以拆成"旋转→缩放→旋转"三步。

### 1.8 低秩近似、PCA 与低秩分解 ⭐ ❌

#### 低秩近似——"砍掉不重要的方向"

##### 动机

1.7 节我们知道了奇异值从大到小反映每个方向的重要性。那能不能只保留大的、扔掉小的，得到一个更简单（秩更低）的矩阵来"近似"原矩阵？

##### 做法

在 SVD 的结果 $A = U \Sigma V^T$ 中，只保留前 $r$ 个最大的奇异值，把其余的直接置零。

##### 数字例子

用 1.7 节的矩阵：$A = \begin{bmatrix} 4 & 0 \\ 3 & 5 \end{bmatrix}$，$\sigma_1 = \sqrt{40}, \sigma_2 = \sqrt{10}$

取 $r = 1$（只保留最大的奇异值），把 $\sigma_2$ 扔掉：

$$A_1 = \sigma_1 \cdot \mathbf{u}_1 \cdot \mathbf{v}_1^T = \sqrt{40} \cdot \frac{1}{\sqrt{5}} \begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \end{bmatrix}$$

$$= \frac{\sqrt{40}}{\sqrt{10}} \begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 1 & 1 \end{bmatrix} = 2 \begin{bmatrix} 1 & 1 \\ 2 & 2 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 4 & 4 \end{bmatrix}$$

对比一下：

$$A = \begin{bmatrix} 4 & 0 \\ 3 & 5 \end{bmatrix}, \qquad A_1 = \begin{bmatrix} 2 & 2 \\ 4 & 4 \end{bmatrix}$$

$A_1$ 的秩 = 1（第二行 = 2 × 第一行），用 1 个方向就"近似"了原来的 2 个方向。当然这个例子里 $\sigma_2$ 并不算特别小（$\sigma_1 / \sigma_2 = 2$），所以近似误差不小。在实际应用里，比如一张 1000×1000 的图片，前 50 个奇异值占了 99% 的信息量，$r = 50$ 时近似几乎完美。

##### 近似误差的保证（Eckart-Young 定理）

$$\|A - A_r\|_F^2 = \sum_{i=r+1}^{\min(m,n)} \sigma_i^2$$

即误差 = 被扔掉的那些奇异值的平方和。在上面的例子中：

$$\|A - A_1\|_F^2 = \sigma_2^2 = 10$$

这里的 $\|\cdot\|_F$ 是 Frobenius 范数（矩阵所有元素的平方和再开根号，详见 1.10 节）。

**Eckart-Young 定理保证**：$A_r$ 是所有秩 $\leq r$ 的矩阵中，和 $A$ 差距最小的那个。不存在比它更好的 rank-$r$ 近似。

##### 存储量对比

以 $1000 \times 1000$ 矩阵、保留前 $r = 50$ 个奇异值为例：

| 存储方式 | 需要存的数 | 数量 |
|---------|-----------|------|
| 原始矩阵 $A$ | 所有 $1000 \times 1000$ 个元素 | $1{,}000{,}000$ |
| 低秩近似 $U_r, \Sigma_r, V_r^T$ | $U_r$：$1000 \times 50$ + $\Sigma_r$：$50$ + $V_r^T$：$50 \times 1000$ | $100{,}050$ |

压缩了约 10 倍。图像压缩、推荐系统（用户-物品评分矩阵）都用这个原理。

##### 一句话定义

低秩近似：对一个**已有的矩阵**做 SVD，扔掉小奇异值，得到一个秩更低、更简单的矩阵来近似原矩阵。是一种**后处理**操作。

---

#### PCA（主成分分析）——低秩近似在数据降维上的应用

##### 动机

高维数据中很多维度是冗余的（高度相关）。比如身高和臂展几乎线性相关，知道了身高就大致知道了臂展，两个特征本质上只有一个方向的信息。

PCA 的目标：找到数据变化最大的几个方向，把数据投影上去，用更少的维度保留最多的信息。

##### 步骤

1. 数据 $X$（$n$ 个样本 × $d$ 维特征），先中心化（每个特征减去均值）
2. 计算协方差矩阵 $C = \frac{1}{n-1} X^T X$（$d \times d$ 对称矩阵）
3. 对 $C$ 做特征分解 $C = V \Lambda V^T$（1.6 节对称矩阵性质保证一定能做），特征向量按特征值从大到小排列
4. 取前 $k$ 个特征向量作为投影方向，得到降维后的数据 $Z = X V_k$（$n \times k$）

##### PCA 和 SVD 的关系——一步推导

对中心化后的 $X$ 做 SVD：$X = U \Sigma V^T$，代入协方差矩阵：

$$C = \frac{1}{n-1} X^T X = \frac{1}{n-1} V \Sigma^T \underbrace{U^T U}_{= I} \Sigma V^T = V \cdot \frac{\Sigma^2}{n-1} \cdot V^T$$

对比特征分解 $C = V \Lambda V^T$，结论：

- **SVD 的右奇异向量 $V$ = PCA 的主成分方向**
- **奇异值的平方 $\sigma_i^2 / (n-1)$ = 对应方向上的方差（特征值）**

所以 **PCA 本质上就是对数据矩阵做 SVD，然后取前 $k$ 个右奇异向量作为投影方向**。

##### 数字例子

100 个二维数据点散布在 $y = 2x$ 附近（有少量噪声）。PCA 会找到：
- 第一主成分：沿 $y = 2x$ 方向（方差最大，$\sigma_1^2$ 很大）
- 第二主成分：垂直于 $y = 2x$ 方向（只有噪声，$\sigma_2^2$ 很小）

降到 1 维：只保留第一主成分，100 个 2D 点变成 100 个 1D 数。丢失的只是噪声方向的信息。

---

#### 低秩分解——LoRA / MLA 的思想（和低秩近似完全不同）

> 这里是关键区分点。很多人把低秩近似和低秩分解混为一谈，但它们的出发点完全不同。

##### 低秩近似 vs 低秩分解

```
低秩近似（上面讲的，SVD 截断）：
  你已经有一个满秩矩阵 A
  → 对 A 做 SVD
  → 扔掉小奇异值
  → 得到一个近似 A 的低秩矩阵 A_r
  关键词：已有矩阵、后处理、数学最优

低秩分解（LoRA / MLA）：
  你从来没有那个满秩矩阵
  → 直接设计网络结构：大矩阵 = 瘦矩阵 × 矮矩阵
  → 让梯度下降去训练这两个小矩阵
  → 结果自动就是低秩的（秩 ≤ r）
  关键词：结构预设、端到端训练、无需 SVD
```

##### LoRA 的具体做法

微调大模型时，原始权重 $W$（$d \times d$，比如 $768 \times 768$）冻结不动，只训练一个低秩增量：

$$\Delta W = B \cdot A$$

- $B$：$d \times r$ 矩阵（$768 \times 8$）—— 从 $r$ 维升回 $d$ 维
- $A$：$r \times d$ 矩阵（$8 \times 768$）—— 从 $d$ 维降到 $r$ 维
- $r$ 是你**人为选定**的瓶颈宽度（LoRA 典型取 $r = 8$）

$B \cdot A$ 的结果是一个 $768 \times 768$ 的矩阵，但它的秩**最多只有 $r = 8$**。为什么？因为 $B$ 只有 8 列，$A$ 只有 8 行，乘出来的结果中任何一列都是 $B$ 那 8 列的线性组合——独立方向最多 8 个（回忆 1.5 节对秩的定义）。

##### 信息流示意

```
输入 x [768维]
    ↓
    A：768维 → 8维（挤过瓶颈，被迫只保留 8 个方向的信息）
    ↓
    B：8维 → 768维（从 8 个方向恢复回高维）
    ↓
输出 ΔWx [768维]

最终：h = Wx + BAx
         ↑原始    ↑低秩增量
```

##### 参数量对比

| | 参数数量 | 秩 |
|------|---------|-----|
| 全参数微调 $\Delta W$ | $768^2 = 589{,}824$ | 最高 768 |
| LoRA $\Delta W = BA$ | $768 \times 8 + 8 \times 768 = 12{,}288$ | 最高 8 |
| 压缩比 | **48 倍** | |

LoRA 的论文（Hu et al., 2021）发现 $r = 8$ 就够了——微调时权重变化确实集中在少数几个方向上。

##### MLA 的做法（同一思想，用在 KV Cache 压缩）

详见 04a §1.2。原理一样：不缓存完整的 K/V（高维），而是设计一个瓶颈，只缓存低维的潜在向量 $c_{kv}$，需要时再投影回来。

$$X \xrightarrow{W_{\text{down}} (d_{\text{model}} \times d_c)} c_{kv} \xrightarrow{W_{\text{up}} (d_c \times d_{\text{model}})} K, V$$

DeepSeek-V2 中 $d_{\text{model}}$ 对应完整 KV 维度约 $32768$，$d_c = 512$，压缩比约 57 倍。

---

#### 全景对比

| | SVD | 低秩近似（SVD 截断） | PCA | 低秩分解（LoRA/MLA） |
|------|-----|-------------------|-----|-------------------|
| **是什么** | 矩阵分解工具 | 用 SVD 做矩阵压缩 | 用 SVD 做数据降维 | 网络结构设计 |
| **输入** | 一个矩阵 $A$ | 一个矩阵 $A$ | 一批数据 $X$ | 无（直接定义结构） |
| **需要原始矩阵吗** | 是 | 是 | 是（需要数据） | **不需要** |
| **怎么得到低秩** | 分解出来的 | 截断小奇异值 | 丢弃小方差方向 | **结构上保证**（$B \cdot A$ 秩 $\leq r$） |
| **何时发生** | 后处理 | 后处理 | 后处理 | 训练时 |
| **有正交约束吗** | 有（$U, V$ 正交） | 有 | 有 | **没有** |
| **最优性保证** | 精确分解 | Eckart-Young 最优 | 方差最大方向 | **无**（靠训练逼近） |
| **典型用途** | 矩阵分析的基础工具 | 图像压缩、推荐系统 | 数据可视化、去噪 | LoRA 微调、MLA 推理 |

**一句话区分低秩近似 vs 低秩分解**：

- **低秩近似**："我有一个大矩阵，怎么用一个小矩阵来近似它？" → SVD 截断，数学最优
- **低秩分解**："我一开始就不要大矩阵，直接用两个小矩阵的乘积，让训练来决定内容" → LoRA/MLA

#### 面试话术（30s）

> SVD 把任意矩阵分解成"旋转-缩放-旋转"三步，奇异值从大到小反映每个方向的重要性。截断小奇异值就是低秩近似——Eckart-Young 定理保证这是数学上最优的 rank-r 近似。PCA 本质就是对数据矩阵做 SVD，取方差最大的几个方向来降维。LoRA 和 MLA 用的是低秩分解——完全不同的思路，不是对已有矩阵做 SVD，而是直接在网络结构里设计一个降维-升维的瓶颈，两个瘦矩阵相乘自动保证低秩，通过端到端训练学出压缩表示。核心区别：低秩近似是"先有大矩阵再压缩"，低秩分解是"从一开始就不造大矩阵"。

### 1.9 正交矩阵 ❌

**定义**：$A^T A = I$，即 $A^{-1} = A^T$。列向量互相正交且长度为 1。

**性质**：正交变换不改变向量长度（保范），只做旋转/反射。

**ML 中出现在**：
- SVD 的 $U, V$ 是正交矩阵（1.7 节）
- 对称矩阵特征分解的 $Q$（1.6 节）
- 权重初始化（正交初始化防止梯度消失/爆炸）

### 1.10 范数 ⭐ ❌

**向量范数**：

$$
\|x\|_1 = \sum_i |x_i| \quad \text{（L1，稀疏性）}
$$

$$
\|x\|_2 = \sqrt{\sum_i x_i^2} \quad \text{（L2，欧氏距离）}
$$

$$
\|x\|_\infty = \max_i |x_i| \quad \text{（最大值范数）}
$$

**矩阵范数**：

$$
\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2} \quad \text{（Frobenius 范数，类似 L2）}
$$

**面试考法**：
- L1 正则化为什么产生稀疏？→ L1 在坐标轴上有"角"，优化容易落在轴上（某些维度变 0）
- L2 正则化为什么是权重衰减？→ 梯度更新时等价于每步把权重乘以 $(1 - \lambda)$
- L1 vs L2 正则化的几何直觉（正方形 vs 圆形约束区域与等高线的交点）

### 1.11 矩阵求逆与伪逆 ❌

$$
A^{-1}A = AA^{-1} = I
$$

- 不是所有矩阵都可逆（行列式 $= 0$ 的不行）
- 非方阵用 Moore-Penrose 伪逆：$A^+ = (A^T A)^{-1} A^T$
- 面试考法：最小二乘法的解析解 $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$ 怎么来的

---

## 二、微积分

### 2.1 导数基本公式 ⭐ ✅

| 函数 | 导数 | ML 中的场景 |
|------|------|------------|
| $x^n$ | $nx^{n-1}$ | 多项式特征 |
| $e^x$ | $e^x$ | Softmax、指数族分布 |
| $\ln(x)$ | $1/x$ | 交叉熵损失 |
| $\frac{1}{1+e^{-x}}$ (sigmoid) | $\sigma(x)(1-\sigma(x))$ | 二分类、门控 |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | RNN 激活 |
| $\max(0, x)$ (ReLU) | $1 \text{ if } x>0,\ 0 \text{ if } x<0$ | 最常用激活函数 |

### 2.2 链式法则 ⭐⭐⭐ ✅（标量版 + 多变量版均已理解）

**标量版**：

$$
\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
$$

**多变量版（反向传播的核心）**：

如果 $L$ 依赖 $y_1, y_2, \ldots, y_n$，每个 $y_i$ 依赖 $x$：

$$
\frac{\partial L}{\partial x} = \sum_{i=1}^{n} \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x}
$$

**直觉**：从 Loss 出发，沿着计算图往回走，每经过一个节点就乘上这个节点的局部梯度。

### 2.3 常见激活函数及其导数 🔥 ✅（Sigmoid 梯度消失 + ReLU + SiLU 对比已理解，含可视化）

![Sigmoid vs ReLU vs SiLU 函数及梯度对比](09a_activations.png)

**Sigmoid**：

$$
\sigma(x) = \frac{1}{1 + e^{-x}} \qquad \sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

- 输出范围 $(0, 1)$
- 问题：两端梯度接近 0（梯度消失）

**Tanh**：

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \qquad \tanh'(x) = 1 - \tanh^2(x)
$$

- 输出范围 $(-1, 1)$，零中心
- 同样有梯度消失问题

**ReLU 家族**：

$$
\text{ReLU}(x) = \max(0, x)
$$

$$
\text{LeakyReLU}(x) = \max(\alpha x, x) \quad (\alpha \approx 0.01)
$$

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

$$
\text{SiLU/Swish}(x) = x \cdot \sigma(x)
$$

- 🔥 现代 LLM（LLaMA、Qwen）统一用 SiLU

**面试考法**：
- 为什么 ReLU 比 Sigmoid 好？→ 正区间梯度恒为 1，缓解梯度消失
- ReLU 的死神经元问题是什么？→ 一旦输入为负，梯度永远为 0，再也学不动
- GELU vs SiLU 的区别？→ GELU 用正态分布 CDF，SiLU 用 sigmoid，效果接近

### 2.4 Softmax 及其导数 🔥 ❌

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Jacobian 矩阵（$i$ 对 $j$ 求导）：

$$
\frac{\partial \text{softmax}_i}{\partial z_j} = \text{softmax}_i (\delta_{ij} - \text{softmax}_j)
$$

其中 $\delta_{ij}$ 是 Kronecker delta（$i = j$ 时为 1，否则为 0）。

**面试考法**：
- Softmax 为什么要减去 $\max(z)$？→ 数值稳定性，防止 $e^z$ 溢出，不影响结果
- Softmax 的温度参数 $T$ 是什么？→ $\text{softmax}(z/T)$，$T$ 大 → 均匀，$T$ 小 → 尖锐，$T \to 0$ → argmax

  > **数字例子**：设 $z = [2, 5, 1]$，最大值是 5
  >
  > | $T$ | $z / T$ | softmax 输出 | 效果 |
  > |-----|---------|-------------|------|
  > | $1.0$ | $[2, 5, 1]$ | $[0.04, 0.93, 0.02]$ | 偏向最大值 |
  > | $0.1$ | $[20, 50, 10]$ | $[\approx 0, \approx 1, \approx 0]$ | 几乎全给最大值 |
  > | $0.01$ | $[200, 500, 100]$ | $[\approx 0, \approx 1, \approx 0]$ | 等价 argmax |
  > | $10$ | $[0.2, 0.5, 0.1]$ | $[0.30, 0.40, 0.27]$ | 接近均匀分布 |
  > | $\to \infty$ | $[0, 0, 0]$ | $[0.33, 0.33, 0.33]$ | 完全均匀 |
  >
  > 原理：$T$ 越小，$z_i / T$ 把 logits 差距放大得越厉害（原来差 3，除以 0.01 后差 300），$e^{500}$ 远大于 $e^{200}$，概率全部集中到最大值 → 等价 argmax。注意 $T \to 0$ 是极限行为，不是真的除以 0。

- 🔥 Attention 里为什么除以 $\sqrt{d}$？→ 内积随 $d$ 增大而增大，使 softmax 进入饱和区，梯度消失

#### 补充：为什么 Softmax 需要 Jacobian 矩阵？

**Jacobian 的本质**：向量对向量求导，结果是一个矩阵（详见 3.3 节）。

- 标量对标量求导 → 一个数
- 标量对向量求导 → 一个向量（梯度）
- 向量对向量求导 → 一个矩阵（Jacobian）

**为什么 ReLU 不需要显式 Jacobian**：ReLU 是逐元素操作，$y_i$ 只跟 $z_i$ 有关，Jacobian 是对角矩阵，记个向量就够了。

**为什么矩阵乘法不需要显式 Jacobian**：$S = Q \cdot K^T$ 的反传恰好能用另一个矩阵乘法表达（$\partial L/\partial Q = \partial L/\partial S \cdot K$），不用写出 Jacobian。

**为什么 Softmax 需要**：每个输出依赖所有输入（分母是全部的和），Jacobian 是满矩阵，且无法化简为更简单的运算。

但实际训练中 Softmax + CrossEntropy 组合后 Jacobian 被约掉，梯度简化为 $\hat{y} - y$（见 2.5 节）。

### 2.5 损失函数的导数 ⭐ ❌

**MSE 损失**：

$$
L = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2 \qquad \frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

**交叉熵损失（二分类）**：

$$
L = -[y \ln \hat{y} + (1-y) \ln(1-\hat{y})]
$$

$$
\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}
$$

**交叉熵 + Softmax（多分类，组合求导）** 🔥：

$$
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
$$

- 这个简洁的结果是 softmax 导数和 log 导数恰好约掉的结果
- 面试常考：推导为什么 softmax + 交叉熵的梯度这么简单

---

## 三、矩阵微积分（反向传播核心）🔥

### 3.1 标量对向量求导 ✅

如果 $L$ 是标量，$\mathbf{x}$ 是 $n$ 维向量，则：

$$
\frac{\partial L}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial L}{\partial x_1} \\ \vdots \\ \frac{\partial L}{\partial x_n} \end{bmatrix}
$$

梯度和原变量同形。

### 3.2 标量对矩阵求导 ✅

如果 $L$ 是标量，$X$ 是 $m \times n$ 矩阵，则：

$$
\left(\frac{\partial L}{\partial X}\right)_{ij} = \frac{\partial L}{\partial X_{ij}}
$$

梯度矩阵和原矩阵同形。这是维度验证法的基础。

### 3.3 向量对向量求导：Jacobian 矩阵 🔥 ❌

**导数形状的完整图谱**：

| 输入 | 输出 | 导数形状 | 名称 |
|------|------|---------|------|
| 标量 | 标量 | 标量（1 个数） | 导数 |
| $n$ 维向量 | 标量 | $n$ 维向量 | 梯度 |
| 标量 | $m$ 维向量 | $m$ 维向量 | — |
| $n$ 维向量 | $m$ 维向量 | $m \times n$ 矩阵 | **Jacobian** |

**为什么是矩阵？**

设 $\mathbf{y} = f(\mathbf{x})$，其中 $\mathbf{x} \in \mathbb{R}^n$，$\mathbf{y} \in \mathbb{R}^m$。每个输出 $y_i$ 对每个输入 $x_j$ 都有一个偏导数 $\partial y_i / \partial x_j$，一共 $m \times n$ 个数，自然排成一个矩阵：

$$
J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}
$$

第 $i$ 行 = $y_i$ 对所有输入的敏感度；第 $j$ 列 = 所有输出对 $x_j$ 的敏感度。

**即使 $m = n$（同维度），Jacobian 仍然是矩阵**——维度相同不代表 $y_i$ 只依赖 $x_i$。

#### 对角 Jacobian vs 满 Jacobian

**逐元素操作（ReLU）→ 对角矩阵**：

$y_i = \text{ReLU}(x_i)$，每个 $y_i$ 只依赖对应的 $x_i$，交叉偏导全为 0：

$$
J_{\text{ReLU}} = \begin{bmatrix}
\text{ReLU}'(x_1) & 0 & 0 \\
0 & \text{ReLU}'(x_2) & 0 \\
0 & 0 & \text{ReLU}'(x_3)
\end{bmatrix}
$$

实践中只需存对角线上的向量，不需要构造完整矩阵。

**耦合操作（Softmax）→ 满矩阵**：

$y_i = e^{x_i} / \sum_j e^{x_j}$，每个 $y_i$ 的分母包含所有 $x_j$，所以 $\partial y_i / \partial x_j \neq 0$ 对所有 $i, j$：

$$
J_{\text{softmax}} = \text{diag}(\mathbf{y}) - \mathbf{y}\mathbf{y}^T
$$

即 $J_{ij} = y_i(\delta_{ij} - y_j)$，是一个 $n \times n$ 的满矩阵。

#### 链式法则的 Jacobian 形式（向量版）

标量链式法则 $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$ 推广到向量：

$$
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = J_2 \cdot J_1
$$

链式法则变成 **Jacobian 矩阵的连乘**。

反向传播中，Loss $L$ 是标量，所以最终要的是 $\partial L / \partial \mathbf{x}$（行向量 $1 \times n$）：

$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot J
$$

即上游梯度（行向量）左乘 Jacobian = 当前层梯度。这就是 **VJP（Vector-Jacobian Product）**，PyTorch autograd 的核心操作。

#### 为什么实践中很少显式构造 Jacobian？

| 场景 | 原因 |
|------|------|
| 逐元素操作（ReLU、SiLU） | Jacobian 是对角矩阵，VJP 退化为逐元素乘 |
| 矩阵乘法 $S = AB$ | VJP 可用另一个矩阵乘法表达，不需要写出 Jacobian |
| Softmax | Jacobian 是满矩阵且无法简化，但 Softmax + CrossEntropy 组合后梯度约分为 $\hat{y} - y$ |

只有当 Jacobian 无法化简且没有组合约分时，才需要显式处理——这在 Transformer 中很少见。

**面试话术**：
> "向量对向量求导得到 Jacobian 矩阵，维度是输出维×输入维。反向传播本质上是上游梯度和 Jacobian 的向量-矩阵乘（VJP）。但实际大多数层的 Jacobian 有特殊结构——逐元素操作是对角的，矩阵乘法可以用另一个矩阵乘法代替——所以 PyTorch 从不显式构造 Jacobian，而是直接实现每个 op 的 VJP。"

### 3.4 矩阵乘法的反传 ⭐⭐⭐ ✅

**核心公式**：若 $S = A \cdot B$，已知 $\partial L / \partial S$，则：

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial S} \cdot B^T
$$

$$
\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial S}
$$

**推导**（从元素级出发）：

$S_{ij} = \sum_k A_{ik} B_{kj}$，所以 $\partial S_{ij} / \partial A_{ab} = \delta_{ia} \cdot B_{bj}$

$$
\frac{\partial L}{\partial A_{ab}} = \sum_{i,j} \frac{\partial L}{\partial S_{ij}} \cdot \frac{\partial S_{ij}}{\partial A_{ab}} = \sum_j \frac{\partial L}{\partial S_{aj}} \cdot B_{bj} = \sum_j \left(\frac{\partial L}{\partial S}\right)_{aj} (B^T)_{jb}
$$

这就是矩阵乘法 $(\partial L / \partial S) \cdot B^T$ 的第 $(a,b)$ 个元素。

**记忆法**：
1. 梯度必须和原变量同形 → 用维度凑
2. 🔥 $\partial L / \partial(\text{左矩阵}) = \partial L / \partial S \cdot (\text{右矩阵})^T$
3. 🔥 $\partial L / \partial(\text{右矩阵}) = (\text{左矩阵})^T \cdot \partial L / \partial S$

### 3.5 逐元素运算的反传 ✅

**逐元素乘（Hadamard）**：$y = A \odot B$

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial y} \odot B
$$

$$
\frac{\partial L}{\partial B} = \frac{\partial L}{\partial y} \odot A
$$

**逐元素函数**：$y = f(x)$（如 ReLU、sigmoid 逐元素应用）

$$
\frac{\partial L}{\partial x_{ij}} = \frac{\partial L}{\partial y_{ij}} \cdot f'(x_{ij})
$$

### 3.5.1 实战推导：Masked Attention Score 的反传 ✅

**题目**：已知前向计算 $y = (Q \cdot K^T) \odot M$，其中 $Q, K \in \mathbb{R}^{L \times d}$，$M \in \mathbb{R}^{L \times L}$（mask，常量），$y \in \mathbb{R}^{L \times L}$。给定 $\partial L / \partial y$，求 $\partial L / \partial Q$ 和 $\partial L / \partial K$。无 softmax，无 scale 系数。

这是 Attention 反传的简化版，组合了 3.4（矩阵乘法）和 3.5（Hadamard 积）两个基础公式。

#### 拆计算图

前向分两步：

$$
\text{Step 1: } S = Q \cdot K^T \qquad \text{（矩阵乘法，3.4 节）}
$$

$$
\text{Step 2: } y = S \odot M \qquad \text{（逐元素乘，3.5 节）}
$$

反向沿计算图从 $y$ 倒着走回 $Q$ 和 $K$。

#### 反传 Step 1：过 Hadamard（3.5 节）

前向 $y = S \odot M$，由 3.5 节公式：

$$
\frac{\partial L}{\partial S} = \frac{\partial L}{\partial y} \odot M
$$

$M$ 是常量 mask，不需要对它求梯度。直觉：mask 为 0 的位置，前向时 $y = 0$，反向时梯度也被"遮掉"——梯度流不过 $M = 0$ 的位置。

#### 反传 Step 2：过矩阵乘法（3.4 节）

前向 $S = Q \cdot K^T$，左矩阵是 $Q$，右矩阵是 $K^T$。记 $G = \partial L / \partial y$，上一步得到 $\partial L / \partial S = G \odot M$。

**求 $\partial L / \partial Q$**（左矩阵的梯度），套 3.4 记忆法 $\partial L / \partial(\text{左}) = \partial L / \partial S \cdot (\text{右})^T$：

$$
\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial S} \cdot (K^T)^T = (G \odot M) \cdot K
$$

**求 $\partial L / \partial K$**（注意右矩阵是 $K^T$，不是 $K$），先套 3.4 记忆法 $\partial L / \partial(\text{右}) = (\text{左})^T \cdot \partial L / \partial S$：

$$
\frac{\partial L}{\partial K^T} = Q^T \cdot \frac{\partial L}{\partial S} = Q^T \cdot (G \odot M)
$$

但我们要的是 $\partial L / \partial K$，不是 $\partial L / \partial K^T$。关键性质：**如果 $\partial L / \partial (K^T) = G'$，则 $\partial L / \partial K = G'^T$**。原因：$K^T$ 的第 $(i,j)$ 元素就是 $K$ 的第 $(j,i)$ 元素，对 $K^T_{ij}$ 求的梯度 = 对 $K_{ji}$ 求的梯度，转置一下就行。

$$
\frac{\partial L}{\partial K} = \left(Q^T \cdot (G \odot M)\right)^T = (G \odot M)^T \cdot Q
$$

#### 最终结果

$$
\boxed{\frac{\partial L}{\partial Q} = (G \odot M) \cdot K}
$$

$$
\boxed{\frac{\partial L}{\partial K} = (G \odot M)^T \cdot Q}
$$

其中 $G = \partial L / \partial y$。

#### 维度验证

| 表达式 | 形状 | 目标形状 | 匹配 |
|--------|------|---------|------|
| $G \odot M$ | $[L, L] \odot [L, L] = [L, L]$ | — | ✅ |
| $(G \odot M) \cdot K$ | $[L, L] \cdot [L, d] = [L, d]$ | 和 $Q$ 同形 $[L, d]$ | ✅ |
| $(G \odot M)^T \cdot Q$ | $[L, L] \cdot [L, d] = [L, d]$ | 和 $K$ 同形 $[L, d]$ | ✅ |

#### 面试话术

> "Masked Attention Score 的反传就是矩阵乘法反传和 Hadamard 反传的组合。把 $y = (QK^T) \odot M$ 拆成两步：先过 Hadamard，梯度乘以 mask；再过矩阵乘法，套'左矩阵梯度 = 上游梯度乘右矩阵转置'的公式。唯一的坑是右矩阵是 $K^T$ 不是 $K$，需要对结果再转置一次。加上 softmax 和 scale 就是完整 Attention 的反传。"

### 3.6 常见层的反传总结 🔥 ❌

| 前向 | $\partial L / \partial(\text{输入})$ | 备注 |
|------|------------|------|
| $y = Wx$ | $\partial L/\partial W = (\partial L/\partial y) \cdot x^T$，$\partial L/\partial x = W^T \cdot (\partial L/\partial y)$ | 全连接层 |
| $y = x + b$ | $\partial L/\partial x = \partial L/\partial y$，$\partial L/\partial b = \sum \partial L/\partial y$ | 偏置 |
| $y = A \odot B$ | $\partial L/\partial A = \partial L/\partial y \odot B$ | Mask、门控 |
| $y = \text{softmax}(x)$ | Jacobian 矩阵（见 2.4） | Attention |
| $y = \text{LayerNorm}(x)$ | 较复杂（见 3.7） | 归一化 |

### 3.7 LayerNorm / RMSNorm 的反传 🔥 ❌

**LayerNorm 前向**：

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

**RMSNorm 前向**（现代 LLM 用这个）：

$$
\hat{x}_i = \frac{x_i}{\text{RMS}(x)} \qquad \text{RMS}(x) = \sqrt{\frac{1}{n}\sum_i x_i^2 + \epsilon}
$$

- 比 LayerNorm 少了 mean centering，计算更快
- 反传时分母含输入的函数 → 非线性来源之一

**面试考法**：
- 为什么 LLM 用 RMSNorm 不用 LayerNorm？→ 去掉 mean shift 不影响效果但减少计算
- Pre-Norm vs Post-Norm 的区别？→ Pre-Norm 训练更稳定，是现代 LLM 标配

---

## 四、概率论与统计 ⭐ ❌

### 4.1 基础概率 ❌

**条件概率**：

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

**贝叶斯公式** ⭐：

$$
P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}
$$

- 后验 = 似然 × 先验 / 证据
- 面试考法：MLE vs MAP 的区别 → MLE 只最大化似然，MAP 加了先验（相当于正则化）

**全概率公式**：

$$
P(A) = \sum_i P(A|B_i) P(B_i)
$$

### 4.2 常见分布 ❌

| 分布 | 概率密度/质量函数 | ML 场景 |
|------|------------------|---------|
| 伯努利 $\text{Bernoulli}(p)$ | $P(x=1) = p$ | 二分类 |
| 二项 $\text{Binomial}(n,p)$ | $\binom{n}{k} p^k (1-p)^{n-k}$ | 多次伯努利试验 |
| 高斯 $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ | 到处都是 |
| 多项 Multinomial | softmax 输出 | 多分类 |
| 均匀 $\text{Uniform}(a,b)$ | $\frac{1}{b-a}$ | 随机初始化 |

### 4.3 期望与方差 ⭐ ✅

$$
E[X] = \sum_x x \cdot P(x) \quad \text{或} \quad \int x \cdot f(x) dx
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2
$$

**重要性质**：

$$
E[aX + b] = aE[X] + b
$$

$$
\text{Var}(aX + b) = a^2 \text{Var}(X)
$$

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)
$$

**面试考法**：
- 🔥 Attention 为什么除以 $\sqrt{d}$？→ $\mathbf{q} \cdot \mathbf{k}$ 的方差随 $d$ 线性增长，除以 $\sqrt{d}$ 使方差归一
- 推导：如果 $q_i, k_i$ 独立同分布，均值 0 方差 1，则 $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = \text{Var}\!\left(\sum_i q_i k_i\right) = d$

### 4.4 最大似然估计 (MLE) ⭐ ❌

$$
\hat{\theta}_{MLE} = \arg\max_\theta \prod_i P(x_i | \theta) = \arg\max_\theta \sum_i \ln P(x_i | \theta)
$$

**面试考法**：
- 高斯分布的 MLE → $\mu$ 的 MLE 是样本均值，$\sigma^2$ 的 MLE 是样本方差
- 交叉熵损失就是负对数似然
- 为什么取 log？→ 乘法变加法，数值更稳定，不改变 argmax

---

## 五、信息论 ⭐

### 5.1 自信息（Self-Information）：听到一件事后，你的认知变了多少

#### 先澄清"信息量"这个词

日常说的"信息量"容易和"知识"混淆。Shannon 定义的信息量**不是在问"这件事是不是知识"，而是在问"告诉你这件事后，你有多惊讶"**——更准确地说，是你的**不确定性减少了多少**。

所以 Shannon 叫它 **self-information**（自信息），而不是 knowledge。它是**相对于接收者已有的认知**来定义的，不是绝对的。

**核心规律：事件发生前你越觉得"意料之中"（概率高），事件发生后你获得的新信息就越少。**

#### 从 LLM 的角度理解：自信息 = loss

这个概念直接对应 LLM 训练中的 cross-entropy loss（见 09 Part 1）：

$$I(\text{token}) = -\log P(\text{token}) = \text{该 token 的 loss}$$

| 场景 | 模型预测概率 | loss = $-\log P$ | 直觉 |
|------|------------|------------------|------|
| "太阳从\_\_边升起"，下一个 token 是"东" | $P \approx 0.99$ | $\approx 0.01$ | 模型早就"知道"了，不惊讶 |
| "今天的新闻是\_\_"，下一个 token 是"陨石" | $P \approx 0.0001$ | $\approx 9.2$ | 模型没料到，被"惊"到了 |

**训练 LLM 的过程 = 不断降低模型对训练数据的"平均惊讶程度"，直到模型的认知（权重）和训练数据的分布尽可能一致。**

#### 公式

Shannon 定义事件 $x$ 的自信息：

$$I(x) = -\log P(x)$$

**$-\log$ 有哪些性质恰好适合描述自信息？** 三条：

1. **确定事件 → 自信息为 0**：$P = 1$ 时 $I = -\log 1 = 0$ ✅
2. **概率越小 → 自信息越大**：$P$ 减小 → $-\log P$ 增大 ✅
3. **可加性**：独立事件同时发生的概率是乘法（$P(x,y) = P(x) \cdot P(y)$），而 $\log$ 天然把乘法变加法

第 3 点是 $\log$ 最核心的性质。两个独立事件 $x, y$：

$$I(x, y) = -\log P(x, y) = -\log [P(x) \cdot P(y)] = -\log P(x) - \log P(y) = I(x) + I(y)$$

满足前两条的函数很多（比如 $1/P - 1$、$1 - P$ 等），但同时满足可加性的只有对数函数族。至于底数（$\log_2$、$\ln$、$\log_{10}$）可以随便选，它们之间只差一个常数系数，不影响数学性质，只改变度量单位。

反例：$f(P) = 1/P$ 满足前两条，但不满足可加性：$\frac{1}{P(x) \cdot P(y)} \neq \frac{1}{P(x)} + \frac{1}{P(y)}$。

用 $\log_2$ 时单位是 **bit**（二进制问题的最小单位）：

| 事件 | 概率 $P$ | 自信息 $I = -\log_2 P$ | 直觉 |
|------|---------|----------------------|------|
| 公平硬币正面 | $0.5$ | $1$ bit | 一个 yes/no 问题就能确定 |
| 骰子掷出 3 | $1/6$ | $\approx 2.58$ bit | 需要约 2.6 个 yes/no |
| 抽到红心 A | $1/52$ | $\approx 5.70$ bit | 需要约 5.7 个 yes/no |
| 必然事件 | $1$ | $0$ bit | 没有新信息 |

> ML 中一般用 $\ln$（自然对数），单位叫 **nat**。只差一个常数系数：$\log_2 x = \ln x / \ln 2$。

### 5.2 熵（Entropy）：平均下来，每次抽样你会有多惊讶

#### 从自信息到熵：为什么需要"平均"

5.1 的自信息 $I(x) = -\log P(x)$ 衡量的是**单个事件**的惊讶程度。但一个随机变量有很多种可能结果，每次抽样可能抽到不同的事件，每个事件的惊讶程度不一样。

**熵要回答的问题是：平均下来，每次抽样你会有多惊讶？**

"平均"怎么算？就是期望——每个事件的惊讶程度（自信息）乘以它出现的概率，再求和：

$$H(X) = E[I(X)] = \sum_i P(x_i) \cdot [-\log P(x_i)] = -\sum_i P(x_i) \log P(x_i)$$

#### 用骰子算一遍

**公平骰子**（每面 $1/6$）：

| 事件 | 概率 $P$ | 自信息 $-\log_2 P$ | $P \times$ 自信息 |
|------|---------|-------------------|------------------|
| 掷出 1 | $1/6$ | $2.58$ bit | $0.43$ |
| 掷出 2 | $1/6$ | $2.58$ bit | $0.43$ |
| 掷出 3 | $1/6$ | $2.58$ bit | $0.43$ |
| 掷出 4 | $1/6$ | $2.58$ bit | $0.43$ |
| 掷出 5 | $1/6$ | $2.58$ bit | $0.43$ |
| 掷出 6 | $1/6$ | $2.58$ bit | $0.43$ |
| | | **熵 $H$** | **$2.58$ bit** |

公平骰子每面概率一样，所以每次都同样惊讶，熵 = 单个事件的自信息。

**作弊骰子**（1 出现概率 90%，其余各 2%）：

| 事件 | 概率 $P$ | 自信息 $-\log_2 P$ | $P \times$ 自信息 |
|------|---------|-------------------|------------------|
| 掷出 1 | $0.90$ | $0.15$ bit | $0.14$ |
| 掷出 2 | $0.02$ | $5.64$ bit | $0.11$ |
| 掷出 3 | $0.02$ | $5.64$ bit | $0.11$ |
| 掷出 4 | $0.02$ | $5.64$ bit | $0.11$ |
| 掷出 5 | $0.02$ | $5.64$ bit | $0.11$ |
| 掷出 6 | $0.02$ | $5.64$ bit | $0.11$ |
| | | **熵 $H$** | **$0.69$ bit** |

大部分时候掷出 1（不惊讶），偶尔掷出别的（很惊讶但很少发生），加权平均后整体惊讶程度很低。

**结论：熵低 = 这个随机变量很"可预测"，熵高 = 很"不确定"。**

| 分布 | 熵 | 直觉 |
|------|-----|------|
| 公平骰子（均匀分布） | $2.58$ bit（最大） | 完全猜不到，每次都惊讶 |
| 作弊骰子（$0.9/0.02/.../0.02$） | $0.69$ bit | 猜 1 就对了，偶尔惊讶 |
| 确定骰子（只会出 1） | $0$ bit | 永远不惊讶 |

#### 性质

- $H(X) \geq 0$：惊讶程度不可能为负
- **均匀分布时 $H$ 最大**：$n$ 个等概率事件的熵 = $\log_2 n$
- $H = 0$ 当且仅当某事件概率 $= 1$：完全确定 → 零惊讶

### 5.3 交叉熵（Cross-Entropy）🔥

#### 熵 vs 交叉熵：一句话区分

两个公式只差一个地方——$\log$ 里面是谁：

$$\text{熵：} \quad H(P) = \sum_i P(x_i) \cdot [-\log \underbrace{P(x_i)}_{\text{真实概率}}] \qquad \leftarrow P \text{ 乘 } {-\log P}\text{，同一个分布}$$

$$\text{交叉熵：} \quad H(P, Q) = \sum_i P(x_i) \cdot [-\log \underbrace{Q(x_i)}_{\text{模型预测}}] \qquad \leftarrow P \text{ 乘 } {-\log Q}\text{，两个不同分布}$$

两个公式的权重都是 $P$（真实分布），因为事件实际上按 $P$ 发生。区别在于你用谁的概率来衡量惊讶程度：

- **熵**：用**真实概率** $P$ 来衡量惊讶。你完全了解这个世界，惊讶程度是最低的。这是理论下界。
- **交叉熵**：用**模型预测** $Q$ 来衡量惊讶。模型对世界的理解不完美，惊讶程度一定 $\geq$ 熵。

#### 接回 LLM 训练

在 LLM 训练里：$P$ = 训练数据中 token 的真实频率，$Q$ = 模型预测的概率。

| 阶段 | 模型状态 | 遇到"的"（真实频率 10%） | 遇到"陨石"（真实频率 0.01%） | 平均 loss（交叉熵） |
|------|---------|------------------------|--------------------------|-------------------|
| 刚初始化 | $Q$ = 均匀分布 $1/6400$ | $-\log(1/6400) = 8.8$（对最常见词也很惊讶） | $-\log(1/6400) = 8.8$ | 8.8（很高） |
| 训练 100 步 | 学会了常见词 | $-\log(0.08) = 2.5$（不太惊讶了） | $-\log(0.001) = 6.9$（还是过度惊讶） | 8.6（降了一点） |
| 训练收敛 | $Q \approx P$ | $-\log(0.10) = 2.3$ | $-\log(0.0001) = 9.2$ | $\approx$ 熵（逼近下界） |

**训练过程 = 交叉熵从一个很高的值，逐步降到接近熵的下界。**

熵在实际训练中不用算（$P$ 固定，它就是个常数），但它告诉你 loss 不可能降到 0——因为语言本身有不确定性，同一个上下文后面可以接很多种合理的 token。

#### 关键公式

**$Q$ 越接近 $P$，交叉熵越小，下界就是熵 $H(P)$。**

**分类任务中**：$P$ 是 one-hot，代入后只剩正确类别那一项 → $H(P, Q) = -\log Q_{\text{正确类别}}$ = 负对数似然（NLL）。

### 5.4 Perplexity（困惑度）：模型平均在几个选项里猜 🔥 ✅

#### 动机：为什么不直接看 loss？

交叉熵 loss（5.3 节）是训练时最直接的指标，但它的单位是 bit（或 nat），不够直觉。"loss = 1.5"是好是坏？很难一眼判断。

**Perplexity 把 loss 换算成一个更直觉的数字——"等价于在几个等概率选项中随机猜"。**

#### 公式

设模型在数据上的**平均交叉熵**（即平均 loss per token）为 $L$，则：

$$\text{PPL} = 2^{L} \quad \text{（用 } \log_2 \text{ 算 loss 时）}$$

$$\text{PPL} = e^{L} \quad \text{（用 } \ln \text{ 算 loss 时）}$$

> 注意：这里的 $L$ 是**交叉熵** $H(P, Q)$（和模型有关），不是**熵** $H(P)$（和模型无关）。熵是真实分布的固有性质，算不到也不用算；交叉熵就是训练时的 loss，每个 batch 都在算。

本质就是对 loss 做**指数还原**——交叉熵用 $\log$ 把概率压缩成 bit 数，perplexity 用指数把它还原回"等效选项数"。

#### loss 到底在衡量什么

**loss 不是说"编码一个 token 本身需要多少 bit"**（那是固定的 $\log_2(6400) \approx 12.6$ bit），而是说**"给定了上下文之后，模型还需要多少 bit 的额外信息才能确定这个 token"**——即模型看完前文后，剩余的不确定性。

| 场景 | 剩余不确定性（loss） | PPL = $2^L$ | 含义 |
|------|---------------------|-------------|------|
| 没有模型，纯瞎猜 | $\log_2(6400) = 12.6$ bit | 6400 | 全词表瞎猜 |
| 差的模型 | 5 bit | 32 | 还在 32 个选项里猜 |
| 好的模型 | 1.5 bit | 2.83 | 差不多只剩 3 个选项 |
| 完美模型 | 0 bit | 1 | 直接知道答案，不用猜 |

具体例子：模型看了前文"中国的首都是"，给出 $P(\text{北}) = 0.85$。此时 loss $= -\log_2(0.85) = 0.23$ bit——模型已经帮你排除了绝大部分可能性，剩余不确定性很小。PPL $= 2^{0.23} = 1.17$，几乎不用猜。

**模型越好 → 前文已经提供了越多信息 → 剩余不确定性越小 → loss 越低 → PPL 越接近 1。**

#### 用具体数字讲清楚

**例 1：模型平均 loss = 1.5 bit**

$$\text{PPL} = 2^{1.5} \approx 2.83$$

直觉："模型在预测每个 token 时，平均相当于在 2.83 个等概率选项里猜。"——选项很少，模型已经学得不错了。

**例 2：完美模型**

每个 token 都预测对，$P = 1$，loss $= -\log_2(1) = 0$：

$$\text{PPL} = 2^{0} = 1$$

"只有 1 个选项，不用猜"——完美预测。

**例 3：随机猜（词表大小 6400）**

均匀分布，每个 token 概率 $= 1/6400$，loss $= \log_2(6400) \approx 12.6$：

$$\text{PPL} = 2^{12.6} = 6400$$

"在整个词表里瞎猜"——最差情况（均匀分布的 PPL = 词表大小）。

#### 直觉对照表

| 平均 loss（bit） | PPL | 直觉含义 |
|------------------|-----|---------|
| 0 | 1 | 完美预测，不用猜 |
| 1 | 2 | 相当于猜硬币正反 |
| 1.5 | 2.83 | 不到 3 个选项 |
| 3 | 8 | 相当于猜 8 选 1 |
| 5 | 32 | 32 个等概率选项 |
| 12.6 | 6400 | 在整个词表里瞎猜（MiniMind 词表大小） |

**数字越小模型越好。PPL = 1 是理论下界，PPL = 词表大小是理论上界。**

#### 和 LLM 实践的联系

- **PPL 是评估 LLM 最常用的指标之一**：论文中报告模型性能时，通常会给出在 validation set 上的 perplexity
- **不同模型直接可比**：只要评估数据集和 tokenizer 一致，PPL 越低 = 模型越好
- **04a-Attention变体演进-todo.md 中用 PPL 对比模型性能**：比较不同 Attention 变体（MHA、MQA、GQA）时，PPL 是核心评估指标

> 注意：PPL 只衡量模型对数据分布的拟合程度（"猜得准不准"），不衡量生成质量（"说得好不好"）。一个 PPL 很低的模型可能背下了训练数据（过拟合），生成时并不一定好用。所以实际评估 LLM 还需要结合 BLEU、ROUGE、人类评分等指标。

#### 面试话术

> "Perplexity 本质就是对 cross-entropy loss 做指数还原，把 bit 数换算成'等价于在几个等概率选项里猜'。PPL = 1 是完美预测，PPL = 词表大小是纯随机猜。评估 LLM 时，PPL 是最基础的指标——直接反映模型对下一个 token 的预测能力。但 PPL 只衡量拟合程度，不衡量生成质量，所以还需要其他指标配合。"

### 5.5 KL 散度 🔥

#### 定义

KL 散度衡量两个概率分布 $P$、$Q$ 之间的差距：

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

公式的结构：**左边的分布（$P$）既是求和的权重，也是 log 里的分子**。这个"谁在左边"很重要，后面会讲。

两个分布越像，KL 越小；完全相同时 KL = 0。

注意：$P$ 和 $Q$ 只是两个分布的代号，**不一定是"真实分布"和"模型分布"**——在不同场景下谁是 $P$ 谁是 $Q$ 含义不同（下面会逐个展开）。

#### KL 散度恒 >= 0（Gibbs 不等式）

$\log \frac{P(x)}{Q(x)}$ 有些项正有些项负，加起来会不会变成负数？**不会。**

直觉：KL 散度 = "多浪费的编码 bit 数"——用错误编码传信息不可能比正确编码更省 bit，所以浪费量不可能是负数。

证明（用 $\ln t \leq t - 1$ 不等式，对所有 $t > 0$ 成立，等号当且仅当 $t = 1$）：

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = -\sum_x P(x) \log \frac{Q(x)}{P(x)}$$

对 $\log$ 应用 $\ln t \leq t - 1$（令 $t = \frac{Q(x)}{P(x)}$）：

$$\log \frac{Q(x)}{P(x)} \leq \frac{Q(x)}{P(x)} - 1$$

两边乘以 $P(x) \geq 0$ 再求和：

$$\sum_x P(x) \log \frac{Q(x)}{P(x)} \leq \sum_x P(x) \left(\frac{Q(x)}{P(x)} - 1\right) = \sum_x Q(x) - \sum_x P(x) = 1 - 1 = 0$$

所以：

$$D_{KL}(P \| Q) = -\sum_x P(x) \log \frac{Q(x)}{P(x)} \geq 0$$

等号成立当且仅当 $P = Q$。

**KL(P||Q) >= 0，KL(Q||P) 也 >= 0——两者都是非负的，不是互为负数。** 

#### 和交叉熵的关系

把 KL 的公式拆开：

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \underbrace{-\sum_x P(x) \log Q(x)}_{H(P, Q) \text{（交叉熵）}} - \underbrace{\left(-\sum_x P(x) \log P(x)\right)}_{H(P) \text{（P 的熵）}}$$

$$\boxed{D_{KL}(P \| Q) = H(P, Q) - H(P)}$$

等价地：

$$\boxed{H(P, Q) = H(P) + D_{KL}(P \| Q)}$$

交叉熵 = 熵 + KL 散度。

**训练 loss 的场景**：$P$ = one-hot label，$Q$ = 模型输出。one-hot 的熵 $H(P) = 0$（5.2 节讲过），所以：

$$H(P, Q) = 0 + D_{KL}(P \| Q) = D_{KL}(P \| Q)$$

**最小化交叉熵 = 最小化 KL 散度**（在训练 loss 的场景下）。

但在 GRPO 里，$P$ 和 $Q$ 都是模型的 softmax 输出，都不是 one-hot，$H(P) \neq 0$——这时交叉熵和 KL 散度**不相等**。

#### 不对称性：KL(P||Q) ≠ KL(Q||P)

交换 $P$ 和 $Q$ 的位置，结果**不同**，而且**不是互为负数**。写出两个公式对比：

$$D_{KL}(P \| Q) = \sum_x \boldsymbol{P(x)} \log \frac{P(x)}{Q(x)} \quad \geq 0$$

$$D_{KL}(Q \| P) = \sum_x \boldsymbol{Q(x)} \log \frac{Q(x)}{P(x)} \quad \geq 0$$

两者都 >= 0。差别在哪？

- **$\log$ 里面**：分子分母确实翻转了
- **但求和的权重也变了**：一个用 $P(x)$ 加权，一个用 $Q(x)$ 加权

**加权分布不同，就是根本性的不同——不是一个正一个负，是两个不同的非负数。** 因此数学上叫"散度"而不叫"距离"（距离要求对称性）。

用数字感受一下：

```
假设只有两个 token: x₁, x₂
P = [0.9, 0.1]
Q = [0.5, 0.5]

KL(P||Q) = 0.9 × ln(0.9/0.5) + 0.1 × ln(0.1/0.5)
         = 0.9 × 0.588 + 0.1 × (-1.609)
         = 0.529 - 0.161
         = 0.368

KL(Q||P) = 0.5 × ln(0.5/0.9) + 0.5 × ln(0.5/0.1)
         = 0.5 × (-0.588) + 0.5 × 1.609
         = -0.294 + 0.805
         = 0.511

两个都是正数，但值不同：0.368 ≠ 0.511，也不是互为负数。
```

#### 在 LLM 中的具体使用

**① 训练 loss（交叉熵）**——固定 label，优化 model

$$\text{loss} = H(P_{\text{label}}, Q_{\text{model}}) = D_{KL}(P_{\text{label}} \| Q_{\text{model}}) + \underbrace{H(P_{\text{label}})}_{=0 \text{（one-hot）}}$$

label 固定在左边，model 在右边被优化 → mode-covering。模型必须给正确答案足够的概率。

**② 知识蒸馏**——固定 teacher，优化 student

$$D_{KL}(P_{\text{teacher}} \| Q_{\text{student}})$$

teacher 固定在左边，student 在右边被优化 → mode-covering。教师认为"北京"和"上海"都可能，学生也必须给两者概率——不能只学一个。

**③ GRPO/RLHF KL 惩罚**——固定 π_ref，优化 π_θ

$$\text{总 loss} = -\text{reward} + \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

注意：**两边都是模型的 softmax 输出，都不是 one-hot，不存在"真实分布"**。

$\pi_{\text{ref}}$ 固定在右边，$\pi_\theta$ 在左边被优化 → mode-seeking。这意味着：

- $\pi_\theta$ **不能凭空造出 $\pi_{\text{ref}}$ 不认可的行为**——如果 $\pi_{\text{ref}}$ 给某个 token 概率极低，$\pi_\theta$ 也不敢给它高概率（否则 KL 爆炸）
- $\pi_\theta$ **可以放弃 $\pi_{\text{ref}}$ 的部分模式**——RL 的目的就是让策略朝高 reward 方向偏移，不需要复刻参考模型的全部行为
- **本质是在 $\pi_{\text{ref}}$ 的概率空间内做"择优放大"**：GRPO 采样 G 条回答，把高 reward 的概率往上提、低 reward 的往下压，但不能强化出 $\pi_{\text{ref}}$ 完全不认识的新行为

用数字感受这个约束：

```
假设 π_ref 在某个位置:  "北京" 0.30,  "上海" 0.25,  "火星" 0.001

π_θ 把 "北京" 提到 0.50（已有模式内微调比例）:
  KL 贡献 = 0.50 × ln(0.50/0.30) = 0.26        ← 小惩罚，可以接受

π_θ 把 "火星" 提到 0.30（强行拉起极低概率的 token）:
  KL 贡献 = 0.30 × ln(0.30/0.001) = 1.71       ← 巨大惩罚，被拉回来
```

这是**软约束**，不是硬禁止——$\beta$ 控制约束的松紧：

- $\beta$ 大 → KL 惩罚重 → $\pi_\theta$ 被拽得紧，几乎不偏离 $\pi_{\text{ref}}$
- $\beta$ 小 → KL 惩罚轻 → $\pi_\theta$ 有更大自由度

如果没有 KL 惩罚（$\beta = 0$），策略可能找到 reward model 的漏洞——生成不通顺但高分的文本（reward hacking）。KL 惩罚拉住它：你可以变好，但不能变得面目全非。

**为什么是 $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ 而不是 $D_{KL}(\pi_{\text{ref}} \| \pi_\theta)$？**

两种方向都能约束 $\pi_\theta$ 不要偏离 $\pi_{\text{ref}}$，但**同样的偏移量，惩罚力度完全不同**。

差别的根源在于**权重是谁的**。KL 公式里左边的分布是求和的权重——当 $\pi_\theta$ 在左边时，权重跟着 $\pi_\theta$ 一起变；当 $\pi_{\text{ref}}$ 在左边时，权重固定不动。

用具体数字对比。假设 $\pi_{\text{ref}} = [0.50, 0.50]$，$\pi_\theta$ 把北京提到 0.90：

$$\pi_\theta = [0.90, 0.10] \quad \text{（偏移相同，KL 方向不同）}$$

GRPO 的 loss 有两项：$-\text{reward}$（越大越好）和 $\beta \cdot D_{KL}$（越小越好）。策略在追 reward 的同时，KL 惩罚在拉它。**两种 KL 方向的核心差异在于：追 reward 的时候，KL 惩罚会不会同步升高。**

**当前用法 $D_{KL}(\pi_\theta \| \pi_{\text{ref}})$——权重是 $\pi_\theta$（跟着变）**：

```
"上海" 的贡献:  权重 = π_θ("上海") = 0.10（已经被压下去了）
               0.10 × ln(0.10/0.50) = -0.16

总 KL = 0.90 × ln(0.90/0.50) + 0.10 × ln(0.10/0.50) = 0.53 - 0.16 = 0.37
```

$\pi_\theta$ 为了追 reward 把"上海"从 0.50 压到 0.10（reward 那边涨了），同时**这个 token 的 KL 权重也从 0.50 降到了 0.10**——KL 惩罚自动减轻了。reward 升了，KL 惩罚没怎么涨，**跷跷板两边可以同时往好的方向走**。策略有空间学。

**反过来 $D_{KL}(\pi_{\text{ref}} \| \pi_\theta)$——权重是 $\pi_{\text{ref}}$（固定不动）**：

```
"上海" 的贡献:  权重 = π_ref("上海") = 0.50（纹丝不动）
               0.50 × ln(0.50/0.10) = 0.80

总 KL = 0.50 × ln(0.50/0.90) + 0.50 × ln(0.50/0.10) = -0.29 + 0.80 = 0.51
```

$\pi_\theta$ 同样为了追 reward 把"上海"压到 0.10（reward 那边涨了），但**KL 权重还是 $\pi_{\text{ref}}$("上海") = 0.50，纹丝不动**。0.50 × 1.61 = 0.80，KL 惩罚巨大。reward 升一寸，KL 惩罚就升一寸——**跷跷板：一头上去另一头必定上来**，两边互相抵消，策略推不动。

偏移越大，跷跷板效应越明显：

```
π_θ = [0.99, 0.01]（极端偏移）

KL(π_θ  || π_ref) = 0.64    ← 权重跟着变，惩罚增长慢，留出学习空间
KL(π_ref || π_θ)  = 1.62    ← 权重固定，惩罚增长 2.5 倍，策略被锁死
```

**总结**：$D_{KL}(\pi_\theta \| \pi_{\text{ref}})$ 的权重随 $\pi_\theta$ 偏移**自动衰减**——追 reward 和控 KL 不矛盾，策略可以在 $\pi_{\text{ref}}$ 的概率空间内择优偏移。$D_{KL}(\pi_{\text{ref}} \| \pi_\theta)$ 的权重固定——追 reward 必然涨 KL，跷跷板互相抵消，reward 信号推不动策略。

### 5.6 等价关系总结

| 视角 | 目标 | 等价关系 |
|------|------|---------|
| 信息论 | 最小化交叉熵 $H(p,q)$ | 直接定义 |
| 分布距离 | 最小化 KL 散度 $D_{KL}(p \| q)$ | $= H(p,q) - H(p)$，$H(p)$ 是常数 |
| 统计学 | 最大似然估计（MLE） | $\max \sum \log P(x_i) = \min \sum -\log P(x_i)$ |
| 损失函数 | 最小化负对数似然（NLL） | $= -\log P(\text{正确 token})$ = 交叉熵 |

### 5.7 KL 散度在 ML 中的核心应用

| 应用 | 公式中的 KL | KL 方向 | 作用 |
|------|------------|---------|------|
| 分类损失 | $H(p,q) = H(p) + D_{KL}(p \Vert q)$，one-hot 时 $H(p)=0$ | 前向 | 最小化交叉熵 = 最小化 KL |
| 知识蒸馏 | $D_{KL}(p_{\text{teacher}} \Vert p_{\text{student}})$ | 前向 | 学生覆盖教师的所有预测模式 |
| VAE 正则化 | $D_{KL}(q_\phi(z \mid x) \Vert p(z))$ | 反向 | 拉近编码器后验与先验 $\mathcal{N}(0,I)$ |
| DPO/RLHF | $\beta \cdot D_{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ | 反向 | 策略不能在参考模型不认可的地方乱放概率 |

**面试话术**：

> "信息论的核心用编码论来理解：熵是最优编码的平均 bit 数，交叉熵是用错误编码的 bit 数，KL 散度是多浪费的 bit 数——所以 KL 恒非负。分类任务用交叉熵做损失，one-hot 代入后就是 NLL，本质等价于 MLE 和最小化 KL 散度。
> KL 散度的不对称性在于求和的权重不同——固定谁、优化谁决定了行为差异：固定左边优化右边是 mode-covering（被优化的分布必须覆盖固定分布的所有模式），固定右边优化左边是 mode-seeking（被优化的分布只能在固定分布的概率空间内活动）。
> GRPO 用 KL(π_θ || π_ref) 而不是反过来，是因为 π_θ 在左边时权重跟着 π_θ 一起变——策略压掉某个 token 追 reward 时，KL 权重也自动降低，reward 和 KL 不形成跷跷板，策略有学习空间。反过来的话权重是 π_ref（固定），reward 升一寸 KL 就涨一寸，策略被锁死。"

---

## 六、优化理论 ⭐ ❌

优化器的演化是一条清晰的问题驱动链，每一步都是"上一个方法有什么问题 → 新方法怎么解决"：

```
GD（全量梯度下降）
 └─ 问题：每步要过全量数据，太慢
     └─ SGD（随机采样加速）
         └─ 问题：梯度噪声大，方向震荡
             └─ SGD + Momentum（累积历史梯度平滑噪声）
                 └─ 问题：所有参数共享同一个 lr
                     └─ Adam（自适应学习率）
                         └─ 问题：weight decay 被自适应缩放扭曲
                             └─ AdamW（解耦 weight decay）  ← MiniMind / LLaMA / DeepSeek 都用这个
```

训练循环的完整数据流（前向保存激活值、backward 写入 `.grad`、optimizer 读 `.grad` 更新参数和 m/v 状态、zero_grad 清空梯度）见 **[09-训练基础.md](09-训练基础.md)** Part 2。下面各节只讲循环中的每个环节。


### 6.2 GD → SGD：怎么算梯度 ✅

> 可视化交互：[梯度下降可视化](https://claude.ai/code/artifact/1bd44654-03d9-4682-905b-2250842752c1)

#### Gradient Descent（GD，梯度下降）

训练神经网络就是找一组参数 $\theta$ 使得 loss 函数 $L(\theta)$ 尽可能小。GD 的做法：每一步用**全部** $N$ 条训练数据算出精确梯度，然后往梯度反方向走：

$$
g_t = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta L(x_i, \theta_t)
$$

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

**为什么是减法？** 梯度 $g_t$ 指向 loss **上升最快**的方向。我们要让 loss 下降，所以往**反方向**走——减法。所有优化器的参数更新都是减法。

**GD 的问题：太慢。** 每一步要遍历全部训练数据。MiniMind 预训练数据约 500 万条，过一遍全量数据才走一步，不可接受。

#### SGD：用随机采样加速

"Stochastic"= 随机。原始定义是每步只抽 **1 条**样本算梯度，太快但噪声极大。实际中用的是折中方案——**Mini-batch SGD**，每步随机抽一个 batch（如 32 条）：

$$
g_t = \frac{1}{B} \sum_{i \in \text{batch}} \nabla_\theta L(x_i, \theta_t)
$$

| 方法 | 每步数据量 | 梯度质量 | 实际使用 |
|------|-----------|---------|---------|
| GD | 全部 $N$ 条 | 精确 | 几乎不用（太慢） |
| SGD（原始） | 1 条 | 噪声极大 | 几乎不用 |
| Mini-batch SGD | 一个 batch（32/64） | 噪声可控 | **标准做法** |

**术语注意**：实际中说"SGD"**几乎都是指 Mini-batch SGD**。MiniMind 预训练的 batch_size = 32。本笔记后续的"SGD"都指 Mini-batch SGD。

**SGD 的问题：梯度方向震荡。** Mini-batch 的梯度是随机近似，有噪声。在 loss surface 的"峡谷"地形中（一个方向陡、另一个方向缓），梯度在陡的方向上来回震荡，在缓的方向上走得很慢。

#### 学习率的数字例子

以 MiniMind 预训练为例，lr = 5e-4：

- 某参数当前值 $\theta = 0.5$，梯度 $g = 0.1$
- 更新量 $= 5 \times 10^{-4} \times 0.1 = 5 \times 10^{-5}$，参数变为 $0.49995$——微调，合理

**lr 太大**（比如 lr = 0.5）：
- 更新量 $= 0.5 \times 0.1 = 0.05$，参数变为 $0.45$——单步跳了 10%，可能跳过最优点
- 极端情况：loss 不降反升，震荡（oscillation）甚至发散到 NaN

**lr 太小**（比如 lr = 1e-8）：
- 更新量 $= 10^{-8} \times 0.1 = 10^{-9}$——参数几乎不动，浪费算力

> **面试话术**："梯度下降的核心是：梯度指向 loss 上升方向，所以往反方向走就能让 loss 下降。GD 用全量数据算梯度太慢，实际训练用 Mini-batch SGD，用一个 batch 的梯度近似全局梯度，在计算效率和梯度精度之间取平衡。"

### 6.3 SGD + Momentum：平滑梯度方向 ❌

#### 动机：SGD 的梯度方向有噪声，来回震荡

SGD 每步的梯度只看当前这一个 batch，方向不稳定。能不能把过去几步的梯度也考虑进来，平滑掉噪声？

#### Momentum 的公式

引入"速度"变量 $u_t$，累积历史梯度方向：

$$
u_t = \beta \cdot u_{t-1} + g_t
$$

$$
\theta_t = \theta_{t-1} - \eta \cdot u_t
$$

和 SGD 的 $\theta_t = \theta_{t-1} - \eta \cdot g_t$ 唯一的区别：把"只看这一步的梯度 $g_t$"换成了"累积的历史梯度 $u_t$"。

- $g_t = \nabla_\theta L(\theta_{t-1})$：当前梯度
- $\beta$：动量系数，通常取 0.9
- $u_0 = 0$：第一步和没有动量一样；从第二步开始才有累积效果

**公式来源**：这个公式本质上是**指数移动平均（EMA）**——统计学中平滑噪声信号的标准工具。"动量"只是一个物理比喻（历史梯度的累积让更新有了惯性），公式本身不是从物理推导出来的。

注意：Momentum 的系数没有归一化（$\beta + 1 \neq 1$），$u_t$ 会越滚越大，是一个**累积器**，靠膨胀来加速。Adam 的一阶矩用 $\beta_1 + (1-\beta_1) = 1$ 归一化，是**平均器**不是累积器，区别见 6.4。

#### 数字例子：动量的加速效果

假设连续 5 步的梯度方向相同，都是 $g = 0.1$，$\beta = 0.9$：

| 步数 $t$ | $u_t = 0.9 \cdot u_{t-1} + 0.1$ | 相比无动量的加速倍数 |
|-----------|----------------------------------|---------------------|
| 1 | $0.9 \times 0 + 0.1 = 0.1$ | 1.0x |
| 2 | $0.9 \times 0.1 + 0.1 = 0.19$ | 1.9x |
| 3 | $0.9 \times 0.19 + 0.1 = 0.271$ | 2.7x |
| 4 | $0.9 \times 0.271 + 0.1 = 0.344$ | 3.4x |
| 5 | $0.9 \times 0.344 + 0.1 = 0.410$ | 4.1x |

理论极限：当梯度恒定时，$u_t$ 收敛到 $g / (1 - \beta) = 0.1 / 0.1 = 1.0$，即 10x 加速。

反过来，如果梯度交替为 $+0.1$ 和 $-0.1$（纯震荡），$u_t$ 会在 $\pm 0.01$ 附近——震荡被抑制了 10 倍。

#### Momentum 的问题：所有参数共享同一个学习率

以 MiniMind 为例：
- **Embedding 层**（`tok_embeddings`，$6400 \times 768$）：大量行梯度为 0——**稀疏参数**
- **Attention 的 $W_Q$**（$768 \times 768$）：每步都有梯度——**密集参数**

用同一个 lr：稀疏参数更新太慢，密集参数更新太频繁。需要给每个参数独立的有效学习率。

> **面试话术**："Momentum 通过累积历史梯度解决了 SGD 的方向震荡——连续同向时加速、来回震荡时抑制。但它没解决自适应学习率的问题，这就是 Adam 要做的事。"

### 6.4 Adam：自适应学习率 ⭐ ✅

#### 出发点：参数更新的通用方程

截至 6.3，我们的参数更新公式是：

$$\theta_t = \theta_{t-1} - \eta \cdot g_t$$

这个公式里有**两个可以优化的东西**：

| 要优化的 | 当前问题 | 目标 |
|---------|---------|------|
| **$g_t$（梯度方向）** | 单步梯度噪声大，方向震荡 | 用更稳定的信号替代原始梯度 |
| **$\eta$（学习率）** | 所有参数共享同一个 $\eta$，但不同参数的 loss 曲面陡峭程度不同 | 给每个参数独立的有效学习率 |

6.3 的 Momentum 解决了第一个（用历史梯度的累积 $u_t$ 替代 $g_t$），但第二个没碰。Adam 同时解决两个：**Step 1 优化 $g_t$（用一阶矩 $m_t$ 替代），Step 2 优化 $\eta$（用二阶矩 $v_t$ 实现自适应缩放）**。

最终 Adam 的更新公式会变成：

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

其中 $\hat{m}_t$ 替代了 $g_t$（更稳定的方向），$\frac{\eta}{\sqrt{\hat{v}_t}}$ 替代了 $\eta$（每个参数独立的有效学习率）。下面逐步推导。

#### Step 1：优化 $g_t$ → 用一阶矩 $m_t$ 替代（稳定梯度方向）

Adam 用 EMA 把多步梯度平均起来，消除噪声：

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

- $\beta_1 = 0.9$：历史占 90%，当前梯度占 10%
- 系数 $\beta_1 + (1-\beta_1) = 1$，所以 $m_t$ 永远在 $g$ 的量级附近，不会膨胀——**平均器**

**和 Momentum 的区别**：Momentum 的 $u_t = \beta u_{t-1} + g_t$，系数不归一（$0.9 + 1 \neq 1$），是**累积器**，靠膨胀加速。Adam 的一阶矩系数归一，是**平均器**，靠平滑稳定。

一阶矩让更新方向更稳定，但做不到让不同参数有不同的步长。这就是二阶矩要解决的。

#### Step 2：优化 $\eta$ → 用二阶矩 $v_t$ 实现自适应学习率（Adam 核心创新）

Step 1 已经把 $g_t$ 换成了更稳定的 $m_t$。现在解决第二个问题：给每个参数加一个独立的缩放系数 $c_i$：

$$
\theta_i = \theta_i - \frac{\eta}{c_i} \cdot m_{t,i}
$$

$c_i$ 应该反映"这个参数的梯度通常有多大"。梯度有正有负，直接求平均会抵消，所以用梯度**平方**的 EMA：

```
连续 5 步梯度 g:   +3,  -2,  +4,  -3,  +2
g 的平均:          (+3-2+4-3+2)/5 = 0.8  ← 正负抵消，看不出幅度
g² 的平均:         (9+4+16+9+4)/5 = 8.4  ← 全是正数，反映真实幅度
√(g²的平均):       √8.4 ≈ 2.9            ← 还原到和 g 同量级
```

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

$$
c_i = \sqrt{v_t} \qquad \Rightarrow \qquad \text{有效学习率} = \frac{\eta}{\sqrt{v_t}}
$$

- $g_t^2$ 是逐元素平方——**每个参数分别算自己的 $v_t$**，所以每个参数有独立的 $c_i$
- $\beta_2 = 0.999$，比 $\beta_1 = 0.9$ 大，因为梯度幅度比方向变化慢，需要更长的记忆窗口

**为什么梯度大的参数需要更小的学习率？** 梯度大意味着 loss 曲面在这个参数方向上很陡。陡坡上步子太大，参数会直接跨过最优点，来回震荡（oscillation）；严重时 loss 不降反升，发散（divergence）到 NaN。所以梯度持续大的参数，应该自动缩小有效学习率，在陡峭的方向上"踩刹车"。反过来，梯度持续小的参数（如稀疏的 Embedding），loss 曲面平缓，可以放心走大步，自动"踩油门"。$\sqrt{v_t}$ 正好衡量了梯度的历史幅度，除以它就实现了这个自适应缩放。

| 参数 | 梯度特性 | $v_t$ | $\sqrt{v_t}$（= $c_i$） | 有效学习率 $\eta / \sqrt{v_t}$ | 效果 |
|------|---------|-------|------------------------|-------------------------------|------|
| Embedding | 稀疏，大多数 $g=0$ | 小 | 小 | 大 | 偶尔有梯度时走大步 |
| $W_Q$ | 密集，每步都有 $g$ | 大 | 大 | 小 | 每步更新但步子小 |

#### Step 3：偏差修正——前几步的 $m_t$ 和 $v_t$ 严重偏低

**问题出在哪？** EMA 公式 $m_t = 0.9 \cdot m_{t-1} + 0.1 \cdot g_t$，$m_0 = 0$。假设每步梯度都是 $g = 1.0$，第一步：

$$
m_1 = 0.9 \times \underbrace{0}_{m_0} + 0.1 \times 1.0 = 0.1
$$

$m_0 = 0$ 不是真实梯度，是随便初始化的，所以 $m_1 = 0.1$ 只有真实值 $1.0$ 的十分之一。

**修正方法**：除以 $(1 - \beta^t)$：

| 步数 $t$ | $m_t$ | $1 - 0.9^t$ | $\hat{m}_t = m_t / (1 - 0.9^t)$ |
|-----------|-------|-------------|--------------------------------|
| 1 | $0.1$ | $0.1$ | $0.1 / 0.1 = 1.0$ ✅ |
| 2 | $0.19$ | $0.19$ | $0.19 / 0.19 = 1.0$ ✅ |
| 100 | $\approx 1.0$ | $\approx 1.0$ | $\approx 1.0$ ✅ |

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

随着 $t$ 增大，$\beta^t \to 0$，修正因子自动消失——前几百步才需要修正，之后 EMA 自己就稳定了。

#### Step 4：组装——Adam 的完整更新公式

把 Step 1-3 的三个组件组装起来，对照开头的通用方程 $\theta_t = \theta_{t-1} - \eta \cdot g_t$：

$$\theta_t = \theta_{t-1} - \underbrace{\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{替代 } \eta \text{（自适应学习率）}} \cdot \underbrace{\hat{m}_t}_{\text{替代 } g_t \text{（平滑梯度）}}$$

- **$\hat{m}_t$**：替代了原始梯度 $g_t$，方向更稳定（Step 1）
- **$\eta / (\sqrt{\hat{v}_t} + \epsilon)$**：替代了全局 $\eta$，每个参数独立缩放（Step 2）
- **$\hat{}$（偏差修正）**：消除 EMA 从零初始化导致的低估（Step 3）
- **$\epsilon = 10^{-8}$**：防止除以零

注意：这个公式里**没有 weight decay**。Adam 原论文用的是纯 Adam，weight decay 是后来的事（见 6.5 节）。

#### 显存开销：MiniMind 的具体数字

Adam 对**每个参数**额外维护 $m_t$ 和 $v_t$ 两个状态张量：

| 存储项 | 数量 | FP32 显存 |
|--------|------|----------|
| 模型参数 $\theta$ | 64M | 256 MB |
| 一阶矩 $m$ | 64M | 256 MB |
| 二阶矩 $v$ | 64M | 256 MB |
| **合计** | **192M 个浮点数** | **768 MB** |

优化器状态占 2/3。7B 模型的 FP32 固定开销 = 参数 28 GB + 梯度 28 GB + 优化器状态 56 GB = **112 GB**（不含激活值），这就是为什么大模型训练需要多卡并行（ZeRO 把优化器状态、梯度甚至参数切分到多卡）。

> **面试话术**："Adam 追踪两个统计量。一阶矩是梯度方向的 EMA，消除噪声、稳定方向。二阶矩是梯度平方的 EMA，衡量每个参数的梯度幅度——除以它的平方根实现自适应学习率：梯度大的参数步子小（防震荡），梯度小的参数步子大（加速收敛）。偏差修正解决 EMA 从零初始化的低估问题。"

### 6.5 正则化、Weight Decay 与 AdamW ✅

#### 动机：正则化从哪来、为什么要加

**正则化不是 Adam 发明的，也不是 Adam 才开始加的。** 它从线性回归时代（1970s）就存在了，比神经网络火起来还早。

**最初的问题**：统计学中，当特征数多于样本数时，线性回归的解不唯一，且对数据噪声极度敏感（病态问题）。1970 年 Hoerl & Kennard 提出 Ridge Regression：在 loss 上加一个参数平方和的惩罚项，约束参数大小，让解变得稳定——这就是 L2 正则化。

后来人们发现，这个惩罚项在神经网络训练中也有类似效果：**加了 weight decay 的模型在验证集上表现更好**。这个经验结论在 CV（AlexNet 2012, weight_decay=0.0005）和 NLP 中被反复验证。

**为什么 weight decay 改善泛化？诚实的回答是：理论上没有简单因果，但有多种部分解释：**

| 解释视角 | 核心论点 | 严格程度 |
|---------|---------|---------|
| 平滑性 | 小权重让函数对输入扰动不敏感，更不容易拟合噪声 | 直觉，非严格证明 |
| Bayesian | L2 正则化等价于对参数加高斯先验 $\mathcal{N}(0, 1/\lambda)$，偏好接近 0 的参数 | 数学严格 |
| Occam's razor | 参数小 → 函数简单 → 更可能泛化 | 哲学层面 |
| 训练稳定性 | 防止个别参数长到极端值导致数值问题 | 实践层面 |

**对于 LLM**：预训练数据极多（几 TB），通常跑不完一个 epoch，传统意义的"背答案"式过拟合不是主要问题。weight decay 在 LLM 中的作用更偏向**训练稳定性**和**隐式控制参数分布**。但所有主流 LLM（LLaMA、DeepSeek、Qwen）都用 weight decay，经验上确实有效。

#### 历史演进

```
1970  Ridge Regression（Hoerl & Kennard）
      → 线性回归的 loss 上加 L2 惩罚，解决病态问题 → L2 正则化
1988  Weight Decay（Hanson & Pratt）
      → 神经网络训练中直接每步缩小权重
      → 发现在 SGD 里 weight decay 和 L2 正则化数学等价
      → 此后整个学界把两个词混用
2012  AlexNet 用 SGD + weight_decay = 0.0005，标配
2014  Adam（Kingma & Ba）
      → 优化器本身不含正则化，但使用时大家沿用老习惯加 L2 正则化
      → PyTorch 的 Adam(weight_decay=0.01) 就是把 lambda*theta 加到梯度里
      → 没人意识到 Adam 的自适应缩放会破坏 L2 和 weight decay 的等价性
2019  AdamW（Loshchilov & Hutter, "Decoupled Weight Decay Regularization"）
      → 发现 Adam + L2 ≠ Adam + weight decay
      → 实验证明解耦后泛化更好 → 提出 AdamW
```

下面按演进顺序讲每一步。

#### Step 1：L2 正则化（1970，从 loss 入手）

在 loss 里加一个惩罚项——参数越大，loss 越大，优化器在降低交叉熵的同时也要压低参数大小：

$$
L_{\text{total}} = \underbrace{L_{\text{original}}}_{\text{交叉熵 loss}} + \underbrace{\frac{\lambda}{2} \|\boldsymbol{\theta}\|_2^2}_{\text{L2 惩罚}}
$$

其中 $\|\boldsymbol{\theta}\|_2^2 = \sum_i \theta_i^2$ 是所有参数的**平方和**（L2 范数的平方，不开根号）。

**L1 vs L2 对比**：

| | L1 正则化（Lasso） | L2 正则化（Ridge） |
|---|---|---|
| 惩罚项 | $\lambda \sum_i \|\theta_i\|$（绝对值之和） | $\frac{\lambda}{2} \sum_i \theta_i^2$（平方和） |
| 效果 | 把不重要的参数直接压到 0（稀疏） | 把所有参数均匀缩小（不会精确到 0） |
| 用途 | 特征选择 | 防过拟合（LLM 都用这个） |

**$\frac{\lambda}{2}$ 里的常数从哪来？**

- $\lambda$ 是人工设的超参数，控制正则化强度。MiniMind 用 $\lambda = 0.01$
- $\frac{1}{2}$ 纯粹是为了求导方便：$\frac{d}{d\theta_i}\left[\frac{\lambda}{2}\theta_i^2\right] = \lambda \theta_i$，系数 $2$ 和 $\frac{1}{2}$ 恰好约掉，梯度干净

**惩罚项的梯度推导**：

$$
\frac{\partial}{\partial \theta_i} \left[ \frac{\lambda}{2} \sum_j \theta_j^2 \right] = \frac{\lambda}{2} \cdot \frac{\partial}{\partial \theta_i} \left[ \theta_1^2 + \theta_2^2 + \cdots + \theta_i^2 + \cdots \right] = \frac{\lambda}{2} \cdot 2\theta_i = \lambda \theta_i
$$

对 $\theta_i$ 求偏导时，其他 $\theta_j^2$（$j \neq i$）都是常数，导数为 0，求和自然消失。所以总梯度是：

$$
\nabla_\theta L_{\text{total}} = \nabla_\theta L_{\text{original}} + \lambda \boldsymbol{\theta}
$$

**效果：每个参数的梯度里多了 $\lambda \theta_i$，方向是"把参数往 0 拉"**——参数越大，拉力越大。

#### Step 2：Weight Decay（1988，直接缩小参数）

换一种更直接的思路：不改 loss，不加惩罚项，直接在每步更新参数时把参数**等比例缩小**一点。

先看**不加 weight decay** 时 SGD 的公式：

$$
\theta_{t} = \theta_{t-1} - \eta \cdot g_t
$$

**加了 weight decay** 后，每步分两个操作顺序执行：

$$
\underbrace{\theta_{t-1}' = (1 - \lambda) \cdot \theta_{t-1}}_{\text{操作 1：先把参数缩小}} \qquad \underbrace{\theta_{t} = \theta_{t-1}' - \eta \cdot g_t}_{\text{操作 2：再用梯度更新}}
$$

合并成一个公式：

$$
\theta_{t} = (1 - \lambda) \cdot \theta_{t-1} - \eta \cdot g_t
$$

- $\lambda$ 是 weight decay 系数（超参数，人工设定），MiniMind 用 $\lambda = 0.01$
- $(1 - \lambda)$ 是参数更新时乘的**衰减因子**，不是 weight decay 本身；每步先把参数缩小 $\lambda$（如 $\lambda=0.01$ 时缩小 1%），然后再做梯度更新
- 缩小的是**上一步的参数 $\theta_{t-1}$**（操作 1），不是更新后的结果（操作 2 还没做）

> **符号注意**：$(1-\lambda)$ 只是衰减因子，$\lambda$ 才是 weight decay 系数。PyTorch `AdamW` 的真实实现中，衰减项还会乘上学习率，即每步实际衰减 $\eta\lambda$，对应衰减因子 $(1-\eta\lambda)$。下文写成 $(1-\lambda)$ 是为了突出 weight decay 与 Adam 梯度更新“解耦”的思想。

直觉：如果某个参数真的对预测有用，梯度更新的推力 > 衰减的拉力，参数会保持在合理值。如果某个参数没用，每步衰减 1%，几百步后就接近 0——自动淘汰。

#### Step 3：在 SGD 里，L2 正则化和 Weight Decay 数学等价

先看两种方案各自的公式，然后证明它们结果一样。

**SGD + L2 正则化**：在 loss 里加了 $\frac{\lambda}{2}\sum\theta_i^2$，所以梯度变成 $g_t + \lambda\theta_{t-1}$，代入 SGD 更新公式：

$$
\theta_{t} = \theta_{t-1} - \eta \cdot \underbrace{(g_t + \lambda\theta_{t-1})}_{\text{梯度 + L2 惩罚项的梯度}}
$$

展开：

$$
\theta_{t} = \theta_{t-1} - \eta g_t - \eta\lambda\theta_{t-1} = (1 - \eta\lambda) \cdot \theta_{t-1} - \eta \cdot g_t
$$

**SGD + weight decay**（Step 2 的公式）：

$$
\theta_{t} = (1 - \lambda') \cdot \theta_{t-1} - \eta \cdot g_t
$$

**两者等价的条件**：令 $\lambda' = \eta\lambda$，两个公式完全一样。也就是说，只要 weight decay 系数等于 $\eta \times$ L2 系数，两种方案在 SGD 里是同一件事的两种写法。

因为数学等价，从 1988 到 2014 这 26 年间，整个学界把"weight decay"和"L2 正则化"当同义词混用。**但这个等价只在 SGD 里成立——到了 Adam 就不成立了。**

#### Step 4：到 Adam 里，等价性被破坏

先回顾 6.4 节 Adam **不加正则化**时的更新公式：

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

其中 $\hat{m}_t$ 和 $\hat{v}_t$ 都是从原始梯度 $g_t$ 算出来的。

现在要加正则化。大家沿用 SGD 时代的老思路——用 L2 正则化，把 $\lambda\theta$ 加到梯度里。**但 Adam 和 SGD 不同：Adam 会用 $\sqrt{v_t}$ 对梯度做自适应缩放。**

**Adam + L2 正则化（PyTorch `Adam(weight_decay=...)` 的做法）**：

$$
g_t' = g_t + \lambda\theta_{t-1}
$$

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \cdot g_t', \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) \cdot {g_t'}^2
$$

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

对比不加正则化的版本，唯一的区别在第一行：$g_t$ 变成了 $g_t' = g_t + \lambda\theta_{t-1}$。问题就出在这里：$\lambda\theta$ 被混进了 $g_t'$，然后进入 $m_t$ 和 $v_t$ 的 EMA，最终被 $\sqrt{\hat{v}_t}$ 除。**正则化项被 Adam 的自适应缩放扭曲了**：

| 参数 | 梯度特性 | $v_t$ | 正则化项 $\lambda\theta$ 的实际效果 |
|------|---------|-------|----------------------------------|
| Embedding（稀疏） | $g \approx 0$ 居多 | 小 | 被小 $\sqrt{v_t}$ 除 → **放大**，衰减过度 |
| $W_Q$（密集） | $g$ 每步都有 | 大 | 被大 $\sqrt{v_t}$ 除 → **缩小**，几乎不衰减 |

这不是前几步的偏差问题——不同参数的 $v_t$ 贯穿整个训练过程始终不同，正则化强度的扭曲是结构性的。本意是所有参数均匀衰减，结果稀疏参数被过度惩罚，密集参数几乎没有正则化。

**为什么 Adam 当时不直接用 weight decay？** 因为从 1988 到 2014 这 26 年间，所有人都以为 weight decay 和 L2 正则化是同一件事（在 SGD 里确实如此）。没人意识到换了优化器后等价性会被破坏。

#### Step 5：AdamW——解耦 weight decay（2019）

Loshchilov & Hutter 的论文"Decoupled Weight Decay Regularization"（ICLR 2019）的核心思路：

```
观察：大家用 Adam 时，都通过 PyTorch 的 weight_decay 参数加正则化
     → PyTorch 的实现是把 lambda*theta 加到梯度里（= L2 正则化）
     → 但在 Adam 里，梯度会被 sqrt(v_t) 自适应缩放
     → 所以正则化效果被不同参数的 v_t 扭曲了（Step 4 的问题）

实验：把 weight decay 从梯度管道中拿出来，直接作用在参数上
     → 泛化效果明显提升（test error 相对改善 ~15%）

结论：Adam + L2 正则化 ≠ Adam + weight decay
     → 应该用解耦的 weight decay → AdamW
```

**AdamW 的公式**——对比 Step 4，改动只有一处：$g_t$ 里不再混入 $\lambda\theta$，weight decay 独立作用在参数上：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) \cdot g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) \cdot g_t^2
$$

$$
\theta_t = \underbrace{(1 - \lambda) \cdot \theta_{t-1}}_{\text{weight decay：独立于 Adam}} - \underbrace{\eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{Adam 梯度更新}}
$$

weight decay 的衰减因子 $(1-\lambda)$ 和 Adam 的 $\sqrt{v_t}$ 完全独立，互不干扰。每个参数都受到相同的 weight decay 项作用，不受自己梯度大小的影响。

> **再次强调**：这里的 $\lambda$ 是 weight decay 系数；$(1-\lambda)$ 只是把它作用到参数上时乘的衰减因子，不要把二者混为一谈。在 PyTorch `AdamW` 的真实实现中，每步对参数的衰减量是 $\eta \times \text{weight\_decay}$，所以衰减因子为 $(1-\eta\lambda)$。

```python
# PyTorch 中的区别：
optimizer = torch.optim.Adam(params, lr=5e-4, weight_decay=0.01)   # ← L2 正则化（有扭曲）
optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)  # ← 解耦 weight decay（正确）
```

MiniMind、LLaMA、DeepSeek 全部用 AdamW。

#### 业界超参数对比

| 模型 | 优化器 | $\beta_1$ | $\beta_2$ | weight_decay | 备注 |
|------|--------|-----------|-----------|-------------|------|
| MiniMind | AdamW | 0.9 | 0.999 | 0.01 | PyTorch 默认值 |
| LLaMA-3 | AdamW | 0.9 | 0.95 | 0.1 | $\beta_2$ 降低以更快适应梯度幅度变化 |
| DeepSeek-V3 | AdamW | 0.9 | 0.95 | 0.1 | 同 LLaMA 配置 |

为什么大模型把 $\beta_2$ 从 0.999 降到 0.95？$\beta_2 = 0.999$ 的有效记忆窗口约 1000 步，$0.95$ 约 20 步。LLM 训练数据分布在不同阶段变化大（预训练→退火），更短的记忆窗口能更快适应新的梯度幅度。

> **面试话术**："正则化通过约束参数大小来改善泛化。L2 正则化在 loss 上加参数平方和惩罚项，weight decay 直接每步缩小参数。在 SGD 里两者数学等价（令 weight decay 系数 = lr × L2 系数），但 Adam 的自适应缩放会扭曲 L2 正则化的效果——不同参数的 $v_t$ 不同，导致正则化强度在参数间不均匀。AdamW 把 weight decay 从梯度管道中解耦出来，直接作用在参数上，所有参数均匀衰减，不受 $v_t$ 影响。所有主流 LLM 都用 AdamW。"

### 6.6 PyTorch 优化器总结 ✅

回到 6.1 的训练循环，`optimizer.step()` 这一步内部发生什么，取决于优化器类型：

```python
# vanilla SGD
optimizer = torch.optim.SGD(params, lr=5e-4)
# step() 内部：theta -= lr * g

# SGD + Momentum
optimizer = torch.optim.SGD(params, lr=5e-4, momentum=0.9)
# step() 内部：u = 0.9*u + g; theta -= lr * u

# AdamW（MiniMind 用这个）
optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.01)
# step() 内部：
#   m = 0.9*m + 0.1*g           （一阶矩）
#   v = 0.999*v + 0.001*g²      （二阶矩）
#   m_hat = m / (1-0.9^t)       （偏差修正）
#   v_hat = v / (1-0.999^t)     （偏差修正）
#   theta = (1-lr*0.01)*theta - lr * m_hat / sqrt(v_hat + 1e-8)
```

训练循环的代码**完全一样**——都是 `loss.backward()` + `optimizer.step()`。区别只在 `step()` 内部。

### 6.7 梯度裁剪：训练的安全阀 ❌

#### 它解决什么问题

训练过程中，偶尔某个 batch 的数据会导致梯度突然变得很大（gradient spike）。如果不加限制，一次大梯度更新就能把参数推到离谱的值，loss 飙升甚至变成 NaN。

梯度裁剪不是每步都生效——正常训练时梯度范数一般在阈值以内，裁剪不触发。只有梯度异常大时才会启动，相当于一个**安全阀**。

#### 在训练循环中的位置

```python
loss.backward()                                        # Step 2：算出梯度
clip_grad_norm_(model.parameters(), max_norm=1.0)      # Step 3：裁剪 ← 就在这里
optimizer.step()                                       # Step 4：用（裁剪后的）梯度更新参数
```

**在 backward 之后、step 之前**。裁剪的是反向传播算出来的原始梯度。

#### 公式

把所有参数的梯度拼成一个大向量 $g$，计算其 L2 范数 $\|g\|$，设阈值 $\tau$：

$$
g \leftarrow \begin{cases}
g & \text{if } \|g\| \leq \tau \\[4pt]
g \cdot \dfrac{\tau}{\|g\|} & \text{if } \|g\| > \tau
\end{cases}
$$

方向完全不变，只是长度被截断到 $\tau$。

#### 数字例子

模型有 3 个参数，梯度 $g = [3.0, 4.0, 0.0]$，阈值 $\tau = 1.0$：

1. 全局 L2 范数：$\|g\| = \sqrt{3^2 + 4^2 + 0^2} = 5.0$
2. $5.0 > 1.0$，触发裁剪
3. 缩放因子：$\tau / \|g\| = 1.0 / 5.0 = 0.2$
4. 裁剪后：$g' = [0.6, 0.8, 0.0]$，$\|g'\| = 1.0$，方向不变

#### 梯度裁剪 vs Weight Decay：完全不同的东西

| | 梯度裁剪 | Weight Decay |
|---|---------|--------------|
| 作用对象 | 梯度（backward 的输出） | 参数本身 |
| 目的 | 防止偶尔的梯度爆炸 | 防止参数长得太大（正则化） |
| 在训练循环中的位置 | backward 之后、step 之前 | optimizer.step() 内部 |
| 触发频率 | 偶尔（只有梯度真的大时） | 每步都做 |

> **面试话术**："梯度裁剪是训练的安全阀——放在 backward 之后、optimizer.step 之前，当梯度范数超过阈值时等比缩小（保持方向），防止偶尔的 gradient spike 导致训练崩溃。几乎所有 LLM 训练都用 max_norm=1.0。它和 weight decay 解决的是完全不同的问题。"

### 6.8 学习率调度 🔥 ❌

#### 动机：Adam 的 $v_t$ 解决不了的问题

6.4 的 Adam 用 $\sqrt{v_t}$ 给每个参数**不同的**有效学习率——这解决的是**空间问题**（参数 A 和参数 B 应该用不同步长）。但还有一个**时间问题**没解决：**训练的不同阶段应该用不同大小的步长。**

$v_t$ 不会随训练进行而整体变小——它跟踪的是梯度幅度，如果后期梯度幅度和初期差不多，$v_t$ 也差不多，所有参数的有效 lr 就不会自动缩小。结果：训练后期参数已经接近最优，但步长还是那么大，参数在最优点附近来回震荡，loss 卡在 plateau 上不再下降。

所以需要一个**随时间调整全局 $\eta$ 的机制**——学习率调度。

#### 历史

学习率调度比 Adam 更早，SGD 时代就在用了：

```
2015  ResNet（He et al.）：SGD + step decay（每 30 epoch lr 除以 10）
2017  SGDR（Loshchilov & Hutter，后来提 AdamW 的同一批人）：
      提出 cosine annealing，用余弦曲线代替阶梯式衰减
2017  Transformer 原论文（Vaswani et al.）：warmup + 反比例衰减
2020+ LLM 时代：warmup + cosine decay 成为标配
```

固定学习率有两个矛盾：
- **训练初期**需要大 lr 快速收敛，但参数还是随机初始化的，大 lr 可能导致 loss 爆炸
- **训练后期**参数已经接近最优，大 lr 会在最优点附近震荡，需要小 lr 做精调

学习率调度就是让 lr 在训练过程中动态变化，解决这个矛盾。

#### Warmup：为什么初期要用小学习率

训练开始时，所有参数都是随机初始化的，此时的梯度可能非常大且方向不稳定（因为随机参数产生的 loss 很大）。如果直接用目标 lr（如 5e-4），大梯度 $\times$ 大 lr = 巨大的参数更新，可能导致：

1. loss 直接 NaN（数值溢出）
2. 参数跳到一个更差的位置，之后难以恢复

Warmup 的做法：前 $T_{\text{warmup}}$ 步，lr 从接近 0 线性增长到目标值：

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}, \quad t < T_{\text{warmup}}
$$

Adam 的二阶矩 $v_t$ 也需要几百步才能稳定估计梯度幅度。Warmup 期间用小 lr，给 $v_t$ 足够时间积累统计信息，等 $v_t$ 稳定了再上大 lr，训练就稳定了。

#### Cosine Decay：为什么用余弦而不是线性衰减

Warmup 结束后，lr 按余弦函数从 $\eta_{\max}$ 衰减到 $\eta_{\min}$：

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \cdot \pi\right)\right), \quad t \geq T_{\text{warmup}}
$$

余弦曲线的特点：

```
lr
│
η_max ────╮
│          ╲
│           ╲          ← 前期：衰减慢，保持较大 lr 快速学习
│            ╲
│             ╲
│              ╲       ← 中期：衰减最快
│               ╲
│                ╲
│                 ╲    ← 后期：衰减慢，小 lr 精调
η_min ─────────────╰──
│
└─────────────────────→ step
    warmup  cosine decay
```

和线性衰减相比，cosine 在后期衰减更慢，给模型更多时间在小 lr 下精调——实验表明这对最终性能有帮助。

#### $\eta_{\max}$ 和 $\eta_{\min}$ 从哪来

- $\eta_{\max}$：就是传给 optimizer 的 `lr`（如 `AdamW(lr=5e-4)` 里的 `5e-4`）
- $\eta_{\min}$：MiniMind 硬编码在 `get_lr()` 函数里，设为 $0.1 \times \eta_{\max}$（即 `5e-5`）

#### 谁来执行 cosine decay？

没有单独的模块。就是训练循环里**你自己写的几行代码**——每步算一个新 lr，手动塞回 optimizer：

```python
def get_lr(current_step, total_steps, lr):
    #                                 ^^^ 这就是 eta_max，传给 optimizer 的那个 lr
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
    #           ^^^
    #           0.1 × lr = eta_min，硬编码在这里

# 训练循环的每一步结尾
new_lr = get_lr(step, total_steps, 5e-4)
for param_group in optimizer.param_groups:
    param_group['lr'] = new_lr    # 直接覆盖 optimizer 里的 lr
```

optimizer 自己不知道 cosine decay 的存在，它只知道"这步的 lr 是多少就用多少"。也可以用 PyTorch 内置的 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)` 来自动调度，但 MiniMind 选择了手动方式。

> 具体实现代码见 [09-训练基础.md](09-训练基础.md) Step 9 和 `scratch/trainer/trainer_utils.py`。

#### Cosine Decay vs Adam 的 $\sqrt{v_t}$ vs Weight Decay：三者的区别

回到 AdamW 的更新公式，把三个机制标出来：

$$
\theta_t = \underbrace{(1-\lambda)}_{\text{weight decay}} \cdot \theta_{t-1} - \underbrace{\eta_t}_{\text{cosine decay 控制}} \cdot \frac{\hat{m}_t}{\underbrace{\sqrt{\hat{v}_t} + \epsilon}_{\text{Adam 自适应}}}
$$

| 机制 | 作用在什么上 | 干什么 | 随时间变化？ |
|------|------------|--------|------------|
| Adam 的 $\sqrt{v_t}$ | 每个参数的步长（分母） | 梯度大的参数步子小，小的步子大 | 不随时间衰减，根据梯度幅度动态调整 |
| Weight decay $(1-\lambda)$ | 参数本身 | 每步缩小参数（正则化） | 系数固定，不变 |
| Cosine decay | 全局学习率 $\eta$ | 训练后期缩小所有参数的步长 | 从 $\eta_{\max}$ 降到 $\eta_{\min}$ |

**为什么 Adam 的自适应不够，还需要 cosine decay？** Adam 的 $\sqrt{v_t}$ 调整的是参数**之间**的相对步长（A 走 3 步、B 走 1 步），但所有参数的**绝对步长**由全局 $\eta$ 控制。训练后期参数接近最优，需要把**所有参数的步长都缩小**来精调，否则会在最优点附近震荡。Adam 做不了这件事——它只管比例，不管整体大小。Cosine decay 就是那个全局音量旋钮。

#### MiniMind 的具体配置

MiniMind 用了简化版——没有 warmup，直接从 lr 开始 cosine decay：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
# step=0:   lr × 1.0（满学习率）
# step=T/2: lr × 0.55
# step=T:   lr × 0.1（最低学习率 = 10% × lr）
```

| 阶段 | 初始 lr | min_lr | warmup | 调度策略 |
|------|---------|--------|--------|---------|
| MiniMind 预训练 | 5e-4 | 5e-5 (= 0.1 $\times$ lr) | 无 | Cosine |
| MiniMind SFT | 1e-5 | 1e-6 (= 0.1 $\times$ lr) | 无 | Cosine |
| LLaMA-3 | 1.5e-4 | 1.5e-5 | 8000 steps | Warmup + Cosine |
| DeepSeek-V3 | 2.2e-4 | 2.2e-5 | 2000 steps | WSD（warmup-stable-decay 三段式） |

MiniMind 为什么能省 warmup？64M 参数的小模型训练稳定性问题不大，随机初始化的梯度幅度有限。大模型（1B+）不加 warmup，训练初期 loss 爆炸的风险很高。

> **面试话术**："LLM 训练的标准 lr 调度是 Warmup + Cosine Decay。Warmup 解决训练初期参数随机、梯度不稳定的问题——用小 lr 给 Adam 的二阶矩时间积累稳定的梯度统计。Cosine Decay 让 lr 在后期衰减更慢，给模型更多精调时间。MiniMind 用了简化版省掉 warmup，64M 的小模型没问题，但 1B+ 模型不加 warmup 训练初期大概率 loss 爆炸。"

### 6.9 梯度消失与梯度爆炸 ⭐

反向传播中，loss 对第 1 层参数的梯度是每层局部梯度的**连乘**：

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h_n} \cdot \prod_{i=2}^{n} \frac{\partial h_i}{\partial h_{i-1}} \cdot \frac{\partial h_1}{\partial w_1}
$$

每个因子 $< 1$ → 连乘指数衰减 → **梯度消失**（浅层参数几乎不更新）；每个因子 $> 1$ → 连乘指数增长 → **梯度爆炸**（参数剧烈跳动，loss 飙升或 NaN）。

| 问题 | 问题原因 | 解决方案 | 解决原理 |
|------|---------|---------|---------|
| 梯度消失 | Sigmoid/Tanh 激活函数导数远小于 1（Sigmoid 最大 0.25），每层梯度至少缩小 4 倍 | ReLU / SiLU 激活函数 | ReLU 正区间导数恒为 1，连乘不衰减；SiLU 正区间导数 $\approx 1$，接近不衰减 |
| 梯度消失 | 网络太深且无直连路径，梯度必须逐层穿过非线性变换，层层缩小 | **残差连接**（Residual） | $h_{l+1} = h_l + F(h_l)$，梯度 $= 1 + \partial F/\partial h$，$+1$ 保证永远有一条梯度直达通道 |
| 梯度消失 | Attention score 未缩放，内积随 $d_k$ 增大使 softmax 进入饱和区，梯度趋近 0 | $\sqrt{d_k}$ 缩放 | 把 $QK^T$ 除以 $\sqrt{d_k}$，将方差拉回 1.0，softmax 不饱和，梯度正常回传 |
| 梯度消失 | FP16 精度不足，最小正数 $\approx 6 \times 10^{-8}$，小梯度直接 underflow 为 0 | AMP Loss Scaling | 前向把 loss 乘大数（如 1024）放大梯度到 FP16 可表示范围；step 前再除回 |
| 梯度爆炸 | 异常 batch 产生巨大梯度（gradient spike） | **梯度裁剪**（Gradient Clipping） | $\|g\| >$ max_norm 时等比缩小：$g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|}$，保持方向不变 |
| 梯度爆炸 | 学习率太大，大梯度 × 大 lr → 参数剧烈跳动 → 下一步梯度更大 → 正反馈循环 | Warmup + Cosine Decay | Warmup 用极小 lr 度过不稳定的初始阶段；Cosine Decay 后期逐步缩小 lr 防止震荡 |
| 梯度爆炸 | 大模型中 $Q$、$K$ 向量范数随层数/训练进展失控，attention logit 不稳定 | QK Norm | 计算 $QK^T$ 前对 $Q$、$K$ 做 RMSNorm，稳定 attention score 数值范围 |
| 梯度爆炸 | 权重持续增长 → 每层输出放大 → 梯度指数增长 | Weight Decay（L2 正则） | 每步对权重施加衰减 $w \leftarrow (1 - \lambda)w$，约束权重范数不至于无限增长 |
| 两者都可能 | 权重初始化不当：方差太小 → 输出逐层缩小 → 消失；方差太大 → 逐层放大 → 爆炸 | Xavier / Kaiming 初始化 | 根据层的输入/输出维度设定权重方差，使每层输出方差 $\approx$ 输入方差 |
| 两者都可能 | 缺少归一化层，各层输入分布不一致，尺度逐层漂移 | RMSNorm / LayerNorm（Pre-Norm） | 每层计算前将输入归一化到稳定分布，消除层间尺度漂移 |
| 两者都可能 | RNN 循环矩阵连乘，序列长度 $T$ 可达数百步，谱半径 $> 1$ 爆炸、$< 1$ 消失 | Self-Attention 替代 RNN | 每个 token 直接 attend 到所有 token，无逐步连乘；配合残差连接，梯度不受序列长度影响 |

> **MiniMind 用了哪些？** SiLU 激活 + 残差连接 + RMSNorm（Pre-Norm）+ $\sqrt{d_k}$ 缩放 + QK Norm + 梯度裁剪 + Warmup + Cosine Decay + Weight Decay + AMP Loss Scaling。11 项方案中用了 10 项（只有"Self-Attention 替代 RNN"不算，因为本身就是 Transformer）。

#### 三种"归一化"的区别

上表有三处都和"归一化"有关，容易混淆。它们作用的对象、时机、解决的问题完全不同：

```
                    作用在哪里              作用在什么上          解决什么
                    ─────────             ──────────          ─────────
① sqrt(d_k) 缩放    Attention 内部         Q·K^T 的点积结果     梯度消失
                    scores = Q·K^T / √d_k                    （softmax 饱和）

② QK Norm           Attention 内部         Q 和 K 向量本身      梯度爆炸
                    计算点积之前            （对向量做 RMSNorm） （Q/K 范数失控）

③ Pre-Norm           每层 Block 入口        整个 hidden_states   两者都防
                    Attention/FFN 之前     （对输入做 RMSNorm）  （层间尺度漂移）
```

**为什么 ① 防消失、② 防爆炸？**

① 的问题是 $QK^T$ 的值**太大**（$d_k=96$ 时方差会放大 96 倍），softmax 进入饱和区（参见 [05 章激活函数导数图](05-FeedForward&激活函数.md)），导数趋近 0 → 梯度消失。除以 $\sqrt{d_k}$ 把方差拉回 1.0，softmax 工作在导数最大的区间。

② 的问题是训练过程中 $Q$、$K$ 向量的**范数逐渐变大**（特别是深层和长训练），就算除了 $\sqrt{d_k}$，点积结果还是可能暴涨 → attention logit 不稳定 → 梯度爆炸。QK Norm 在点积之前把 $Q$、$K$ 向量的范数归一化，从根源上控制住点积的数值范围。

③ 是全局性的——不管 Attention 还是 FFN，每层计算前都先把输入归一化到稳定分布。没有 ③，即使 ①② 都做了，隐藏状态的数值也会逐层漂移（越来越大或越来越小），最终导致梯度问题。

**一句话总结**：① 管 softmax 不饱和，② 管 Q/K 不失控，③ 管整体不漂移。三者互补，不是重复。

---

## 七、概率图模型与采样（进阶）

### 7.1 蒙特卡洛采样

**要解决什么问题**：在机器学习中，我们经常需要计算某个分布下的期望 $E_{p(x)}[f(x)] = \int f(x) p(x) dx$。但当 $x$ 是高维向量时（比如 LLM 中 vocab size = 64000 的离散分布，或者连续潜变量模型中的积分），解析求解不可能，数值积分也因为维度灾难而不可行。蒙特卡洛方法的核心思想：**不去算积分，而是从分布中抽样，用样本均值逼近期望**。

**公式**：

从目标分布 $p(x)$ 中独立采样 $N$ 个样本 $x_1, x_2, \dots, x_N \sim p(x)$，用样本均值估计期望：

$$
E_{p(x)}[f(x)] \approx \hat{\mu}_N = \frac{1}{N} \sum_{i=1}^{N} f(x_i)
$$

**为什么这样做是对的**：大数定律保证，当 $N \to \infty$ 时，$\hat{\mu}_N \to E_{p(x)}[f(x)]$（几乎必然收敛）。中心极限定理进一步告诉我们收敛速度：误差的标准差为 $\sigma / \sqrt{N}$，其中 $\sigma^2 = \text{Var}_{p(x)}[f(x)]$。注意这个收敛速度 $O(1/\sqrt{N})$ 与维度无关——这是蒙特卡洛方法在高维问题上的核心优势。

**和 LLM 的关系**：

- LLM 生成 token 时，模型输出 logits 经过 softmax 得到下一个 token 的概率分布 $p(x_t | x_{<t})$，然后从这个分布中采样一个 token——这就是蒙特卡洛采样的直接应用
- 每次生成（每个 sample）就是一次采样；多次生成同一个 prompt（如 pass@k 评估），就是用多个样本来估计模型能力的期望
- RLHF 训练时，需要估计策略 $\pi_\theta$ 下的期望 reward $E_{\pi_\theta}[R(x)]$，也是通过让模型生成多个 response 再取平均来近似

**面试考法**：

- "蒙特卡洛采样的收敛速度和维度有什么关系？" → $O(1/\sqrt{N})$，与维度无关，这是它能处理高维问题的根本原因
- "LLM 推理时的采样和蒙特卡洛有什么关系？" → 每次从 softmax 分布采样 token 就是蒙特卡洛；pass@k 是用 k 个样本估计"至少一个正确"的概率

### 7.2 重要性采样

**要解决什么问题**：有时候我们想计算分布 $p(x)$ 下的期望，但存在两种困难：(1) 无法直接从 $p(x)$ 采样（比如 $p(x)$ 是一个未归一化的后验分布）；(2) 即使能采样，$p(x)$ 的高概率区域和 $f(x)$ 的大值区域不重合，导致采样效率极低（大量样本的 $f(x) \approx 0$，浪费计算）。重要性采样的解决方案：**从另一个容易采样的分布 $q(x)$ 出发，通过重新加权来修正分布偏差**。

**公式推导**：

从恒等变换出发：

$$
E_{p(x)}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = E_{q(x)}\left[\frac{p(x)}{q(x)} f(x)\right]
$$

其中 $w(x) = \frac{p(x)}{q(x)}$ 称为**重要性权重**。于是从 $q(x)$ 采样 $x_1, \dots, x_N \sim q(x)$，有：

$$
E_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} w(x_i) f(x_i) = \frac{1}{N} \sum_{i=1}^{N} \frac{p(x_i)}{q(x_i)} f(x_i)
$$

**直觉**：$q(x)$ 在某处采样过多（相对于 $p(x)$），重要性权重 $p(x)/q(x) < 1$ 会降低这些样本的贡献；反之，$q(x)$ 在某处采样不足，权重 $p(x)/q(x) > 1$ 会放大这些样本的贡献。通过加权修正，最终估计的是 $p(x)$ 下的期望。

**重要性权重的方差问题**：如果 $p(x)$ 和 $q(x)$ 差异过大，少数样本会获得极大的权重 $w(x)$，导致估计的方差爆炸，几乎所有的"有效样本"集中在一两个点上。衡量这种退化的指标是**有效样本量**（Effective Sample Size, ESS）：

$$
\text{ESS} = \frac{\left(\sum_{i=1}^{N} w_i\right)^2}{\sum_{i=1}^{N} w_i^2}
$$

ESS 越小，说明权重越集中，估计越不可靠。

**和 LLM/RLHF 的关系（PPO 的核心）**：

PPO 训练时，用**旧策略** $\pi_{\theta_{\text{old}}}$ 采集的 rollout 数据来优化**新策略** $\pi_\theta$。这正是重要性采样的场景——数据是从 $q = \pi_{\theta_{\text{old}}}$ 采的，但目标是估计 $p = \pi_\theta$ 下的期望 reward：

$$
E_{\pi_\theta}\left[\hat{A}_t\right] \approx E_{\pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t\right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 就是重要性采样比率。PPO 的 clipping 机制直接解决了权重过大的问题：

$$
L^{\text{CLIP}}(\theta) = E_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right)\right]
$$

把 $r_t(\theta)$ 裁剪到 $[1-\epsilon, 1+\epsilon]$（通常 $\epsilon = 0.2$），防止新策略偏离旧策略太远导致权重爆炸。

**面试考法**：

- "PPO 为什么要用 clipping？" → 因为 PPO 本质是重要性采样，新旧策略差异过大时重要性权重 $r_t(\theta)$ 会爆炸，clipping 限制了权重范围，等价于限制了新旧策略的偏离程度
- "重要性采样和 off-policy 是什么关系？" → off-policy 学习（用行为策略采集数据训练目标策略）天然需要重要性采样来修正分布偏差；PPO 的 clipping 就是在控制 off-policy 的程度
- "为什么不直接用新策略采样？" → 每次更新 $\theta$ 后重新 rollout 代价太大（需要和环境交互）；重用旧数据做多步更新更高效，但必须用重要性采样修正

### 7.3 温度采样

**要解决什么问题**：LLM 推理时，softmax 输出的概率分布决定了生成内容的多样性。有些场景需要确定性输出（代码生成、数学推理），有些场景需要多样性（创意写作、brainstorming）。温度参数 $T$ 提供了一个统一的旋钮来控制分布的"尖锐程度"。

**公式**：

给定模型输出的 logits $z_1, z_2, \dots, z_V$（$V$ 是 vocab size），温度采样的 softmax 为：

$$
P(x_i) = \frac{e^{z_i / T}}{\sum_{j=1}^{V} e^{z_j / T}}
$$

**温度 $T$ 的效果**：

- $T = 1$：标准 softmax，分布不变
- $T < 1$：logits 被放大（除以小于 1 的数），概率分布变得更尖锐，高概率 token 更突出
- $T > 1$：logits 被缩小（除以大于 1 的数），概率分布变得更平坦，低概率 token 获得更多机会
- $T \to 0$：退化为 argmax，完全确定性（greedy decoding）
- $T \to \infty$：退化为均匀分布，完全随机

**具体数字举例**：

设 logits $z = [2.0, 1.0, 0.5]$，对应三个 token A, B, C：

| 温度 $T$ | $z/T$ | $e^{z/T}$ | 概率 $P$ | 分布特点 |
|----------|--------|-----------|----------|---------|
| $0.5$ | $[4.0, 2.0, 1.0]$ | $[54.6, 7.39, 2.72]$ | $[0.844, 0.114, 0.042]$ | 尖锐：A 占 84% |
| $1.0$ | $[2.0, 1.0, 0.5]$ | $[7.39, 2.72, 1.65]$ | $[0.627, 0.231, 0.140]$ | 标准 |
| $2.0$ | $[1.0, 0.5, 0.25]$ | $[2.72, 1.65, 1.28]$ | $[0.481, 0.292, 0.227]$ | 平坦：接近均匀 |

可以看到：$T=0.5$ 时 A 的概率从 62.7% 跃升到 84.4%，$T=2.0$ 时降到 48.1%，三个 token 的概率趋于均匀。

**一句话总结**：greedy 是"选最大的"，sampling 是"按概率抽签"，Top-k / Top-p 是"先把不靠谱的候选踢掉，再抽签"。

**Top-k 采样**：

先按概率排序，只保留概率最高的 $k$ 个 token，其余概率置零后重新归一化，然后**从归一化后的分布中随机采样**一个 token。

> **数字例子**（$k=3$）：假设 softmax 后概率为：
>
> | token | "的" | "了" | "是" | "在" | "吗" |
> |-------|------|------|------|------|------|
> | 原始概率 | 0.40 | 0.25 | 0.20 | 0.10 | 0.05 |
> | Top-3 截断 | 0.40 | 0.25 | 0.20 | ~~砍掉~~ | ~~砍掉~~ |
> | 归一化 | 0.47 | 0.29 | 0.24 | 0 | 0 |
>
> 最后从 {的:47%, 了:29%, 是:24%} 中**抽签**选一个 → 不是"选最大的"，而是按概率随机采样。保留 3 个不是为了选 3 个，是为了把尾部垃圾候选踢掉，让抽签只在靠谱的候选里进行。

问题是 $k$ 是固定的——当分布本身就集中在 2 个 token 时，$k=50$ 会引入太多噪声 token；当分布很平坦时，$k=50$ 又可能截断有意义的选项。

**Top-p（Nucleus）采样**：

不固定 token 数量，而是动态选择最小的 token 集合使其累积概率 $\geq p$：

$$
\text{Top-p}(p) = \min \{S \subseteq V : \sum_{x \in S} P(x) \geq p\}
$$

其中 $S$ 按概率降序填充。典型 $p = 0.9$ 或 $0.95$。相比 Top-k 的优势在于自适应——当分布尖锐时自动选少量 token，分布平坦时自动选更多。

> **数字例子**（$p=0.85$）：同样的概率分布，从大到小累加：
>
> | 步骤 | token | 概率 | 累积概率 | 是否保留 |
> |------|-------|------|---------|---------|
> | 1 | "的" | 0.40 | 0.40 | ✓ |
> | 2 | "了" | 0.25 | 0.65 | ✓ |
> | 3 | "是" | 0.20 | 0.85 | ✓ 达到 $p$，停！ |
> | 4 | "在" | 0.10 | — | ✗ |
> | 5 | "吗" | 0.05 | — | ✗ |
>
> 保留 {"的", "了", "是"}，归一化后抽签。如果分布更尖锐（"的" 占 0.90），Top-p 会只保留 1 个；如果分布很平（五个都是 0.20），Top-p 会保留 5 个——这就是"自适应"。

**实际生成流程（以 Hugging Face 为例）**：

```
logits → 除以 T → softmax → Top-k 截断 → Top-p 截断 → 重新归一化 → 采样
```

温度、Top-k、Top-p 通常组合使用。典型配置：

| 场景 | 温度 | Top-p | Top-k |
|------|------|-------|-------|
| 代码生成 / 数学推理 | 0.0（greedy）或 0.2 | 0.95 | - |
| 通用对话 | 0.7 | 0.9 | - |
| 创意写作 | 1.0 ~ 1.2 | 0.95 | 50 |

**面试考法**：

- "温度为什么是除以 T 而不是乘以 T？" → 温度来自统计物理的 Boltzmann 分布 $P \propto e^{-E/kT}$，$T$ 在分母位置；$T$ 越高"热运动"越剧烈，分布越均匀
- "Top-p 和 Top-k 的核心区别？" → Top-k 固定候选数量，Top-p 固定累积概率——Top-p 能自适应分布形状
- "temperature = 0 的实现和 temperature 很小有什么区别？" → $T=0$ 通常直接实现为 argmax（不走 softmax），而 $T$ 很小但非零仍然是概率采样，极端情况下可能出现数值溢出
- "temperature = 0 时多次推理结果一定相同吗？" → 理论上 argmax 是确定性操作，结果一定相同。但 GPU 矩阵乘法的并行归约存在浮点舍入误差，同一输入跑两次 logits 可能在小数点后 6-7 位不同；多卡张量并行时 all-reduce 的合并顺序也会引入类似误差。当两个 token 的 logit 极其接近时，微小抖动可能翻转排名。实际几乎观察不到，但严格 100% 确定性需要 `torch.use_deterministic_algorithms(True)`

**面试话术**：

> 温度采样本质上是对 logits 做缩放后再 softmax——$T<1$ 放大差异让分布变尖锐，$T>1$ 缩小差异让分布变平坦。实际推理时通常和 Top-p 组合使用：先用温度控制分布形状，再用 Top-p 动态截断长尾。我在复现 MiniMind 的 generate 函数时实现过这套流程，温度影响的是多样性和一致性之间的 trade-off。

---

## 八、数值计算技巧 🔥 ❌

### 8.1 数值稳定性

#### 8.1.1 Log-Sum-Exp 技巧

**动机**：Softmax 需要计算 $e^{x_i}$，但当 logits 值很大时，$e^x$ 会直接溢出到 `inf`；当 logits 值很小（大负数）时，$e^x$ 会下溢到 0，导致 $\log(0) = -\inf$。训练 LLM 时 logits 动辄上千维，数值范围不受控，必须有稳定的计算方式。

**公式**：

Softmax 的定义：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

取对数得到 log-softmax：

$$
\log \text{softmax}(x_i) = x_i - \log \sum_j e^{x_j}
$$

问题出在 $\log \sum_j e^{x_j}$ 这一项。Log-sum-exp 技巧：令 $c = \max_j x_j$，则：

$$
\log \sum_j e^{x_j} = c + \log \sum_j e^{x_j - c}
$$

证明很简单——把 $e^c$ 从求和中提出来：

$$
\log \sum_j e^{x_j} = \log \left( e^c \cdot \sum_j e^{x_j - c} \right) = c + \log \sum_j e^{x_j - c}
$$

减去 $c$ 之后，指数的最大值变成 $e^0 = 1$，其余都是 $e^{\text{负数}} \in (0, 1]$，不会溢出。

**具体数字举例**：

假设 logits = $[1000, 1001, 999]$：

- 直接计算：$e^{1000} \approx 10^{434}$，FP32 最大值约 $3.4 \times 10^{38}$，直接返回 `inf`
- 用 LSE 技巧：$c = \max = 1001$，减去后变成 $[-1, 0, -2]$
  - $e^{-1} \approx 0.368$，$e^{0} = 1$，$e^{-2} \approx 0.135$
  - $\sum = 1.503$，$\log(1.503) \approx 0.407$
  - $\text{LSE} = 1001 + 0.407 = 1001.407$

PyTorch 中 `F.cross_entropy` 和 `F.log_softmax` 内部都自动使用了这个技巧，不需要手动实现。但理解原理很重要——面试经常问"softmax 如何保证数值稳定"。

**和 LLM 的关系**：LLM 的输出层是 vocab_size 维的 logits（MiniMind 中 vocab_size=6400），每个 token 都要算一次 softmax。如果不做 log-sum-exp，训练初期 logits 未归一化时极易溢出。

**面试考法**：
- "softmax 计算中如何避免数值溢出？" → 减去最大值（log-sum-exp 技巧），使指数项的最大值为 1
- "为什么 PyTorch 推荐用 `F.cross_entropy` 而不是先 softmax 再取 log 再 nll_loss？" → `F.cross_entropy` 内部走 log-softmax 一步到位，数值稳定且避免中间变量

#### 8.1.2 浮点精度：FP32 / FP16 / BF16

**动机**：LLM 参数量巨大（MiniMind 64M，GPT-3 175B），用 FP32 存储和计算太耗显存。降低精度可以：(1) 显存减半，(2) Tensor Core 加速计算。但降低精度会引入数值问题，不同格式的取舍不同。

**IEEE 754 浮点数结构**：

一个浮点数由三部分组成：$(-1)^s \times 2^{E - \text{bias}} \times (1 + M)$

- **符号位 (s)**：1 bit，0 正 1 负
- **指数位 (E)**：决定数值范围（能表示多大/多小的数）
- **尾数位 (M)**：决定精度（相邻两个可表示数之间的间隔）

| 格式 | 总位数 | 符号 | 指数位 | 尾数位 | 最大值 | 最小正数（非零） | 精度（有效十进制位） |
|------|--------|------|--------|--------|--------|-----------------|-------------------|
| FP32 | 32 | 1 | 8 | 23 | $\approx 3.4 \times 10^{38}$ | $\approx 1.2 \times 10^{-38}$ | ~7 位 |
| FP16 | 16 | 1 | 5 | 10 | $65504$ | $\approx 6.1 \times 10^{-5}$ | ~3 位 |
| BF16 | 16 | 1 | 8 | 7 | $\approx 3.4 \times 10^{38}$ | $\approx 1.2 \times 10^{-38}$ | ~2 位 |

**直觉解释**：

- **FP16** 指数位只有 5 bit，最大值 65504。训练中 loss 或中间激活值稍大就溢出为 `inf`，然后梯度变 `nan`，训练崩掉。
- **BF16** 指数位和 FP32 一样都是 8 bit，所以数值范围完全相同（$10^{-38}$ 到 $10^{38}$），几乎不会溢出。代价是尾数只有 7 bit（FP32 有 23 bit），精度低——但实验表明 LLM 训练对精度不敏感，对范围更敏感。
- 所以 **BF16 = 保留了 FP32 的范围，砍掉了精度**，是 LLM 训练的最优选择。

**具体数字举例**：

假设某层的梯度值为 $0.00003$（$3 \times 10^{-5}$）：

- FP32：正常表示，23 bit 尾数足够
- FP16：最小正数 $\approx 6.1 \times 10^{-5}$，$3 \times 10^{-5}$ 低于此值，**直接变成 0**（梯度下溢）
- BF16：最小正数 $\approx 1.2 \times 10^{-38}$，$3 \times 10^{-5}$ 远大于此值，正常表示（虽然精度只有 2 位有效数字）

这就是 FP16 训练需要 GradScaler 而 BF16 通常不需要的原因。

**和 LLM 的关系**：

- LLM 训练标配 BF16（需要 Ampere 及以上 GPU：A100、H100、RTX 3090/4060）
- MiniMind 在 `train_pretrain.py` 中使用 `torch.cuda.amp.autocast(dtype=torch.bfloat16)`
- DeepSeek、Qwen、LLaMA 预训练全部使用 BF16
- BF16 可以直接替代 FP32 的 master weight（因为范围相同），简化训练流程

**面试考法**：
- "FP16 和 BF16 的区别？为什么 LLM 训练选 BF16？" → FP16 指数位少（5 bit），范围只到 65504，容易溢出；BF16 指数位和 FP32 一样（8 bit），范围相同，训练稳定
- "BF16 精度低不会影响训练效果吗？" → LLM 训练对数值范围敏感（溢出直接崩），对精度不敏感（梯度本身就有噪声）

#### 8.1.3 梯度下溢与 GradScaler

**动机**：用 FP16 训练时，梯度值经常小于 FP16 的最小正数（$6.1 \times 10^{-5}$），导致梯度变成 0——这叫**梯度下溢 (gradient underflow)**。梯度为 0 意味着参数不更新，模型停止学习。

**GradScaler 的原理**：

核心思路：在 FP16 能表示的范围内"放大"梯度，更新参数时再"缩小"回来。

1. **前向传播**：用 FP16 计算，得到 loss
2. **放大 loss**：$\text{scaled\_loss} = \text{loss} \times S$（$S$ 是 scale factor，初始值通常 $2^{16} = 65536$）
3. **反向传播**：因为 loss 被放大了 $S$ 倍，根据链式法则，所有梯度也被放大 $S$ 倍
4. **检查 inf/nan**：如果放大后梯度溢出了，则跳过本次更新，并将 $S$ 减半
5. **缩小梯度**：$g_{\text{real}} = g_{\text{scaled}} / S$，恢复真实梯度值
6. **用 FP32 更新参数**：optimizer 维护 FP32 的 master weight，用真实梯度更新

**具体数字举例**：

假设某个梯度的真实值为 $10^{-6}$（FP16 下溢为 0）：

- Scale factor $S = 65536 = 2^{16}$
- 放大后的梯度：$10^{-6} \times 65536 = 0.065536$，FP16 可以正常表示
- 反向传播完成后除以 $S$：$0.065536 / 65536 = 10^{-6}$，恢复真实值
- 用 FP32 的 optimizer 完成参数更新

Scale factor 的动态调整策略：

- 连续 $N$ 步（默认 2000 步）没有 inf/nan → $S$ 翻倍（尝试更大的放大倍数）
- 出现 inf/nan → $S$ 减半，跳过本次更新

```python
# PyTorch GradScaler 使用示例
scaler = torch.cuda.amp.GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        loss = model(batch)
    scaler.scale(loss).backward()   # 放大 loss 再反传
    scaler.step(optimizer)          # 内部：unscale → 检查 inf → 更新
    scaler.update()                 # 动态调整 scale factor
```

**和 LLM 的关系**：

- FP16 训练必须配合 GradScaler，否则小梯度全部下溢
- BF16 训练通常不需要 GradScaler，因为 BF16 的范围和 FP32 一样，不存在下溢问题
- 现代 LLM（DeepSeek、Qwen、LLaMA）直接用 BF16，跳过 GradScaler 这一步
- MiniMind 的训练脚本中，如果用 BF16 则不启用 GradScaler

**面试考法**：
- "FP16 训练为什么需要 GradScaler？BF16 需要吗？" → FP16 最小正数 $6.1 \times 10^{-5}$，小梯度下溢为 0；GradScaler 先放大 loss 间接放大梯度，更新时再缩回。BF16 范围和 FP32 一样，不需要
- "GradScaler 的 scale factor 是固定的吗？" → 不是，动态调整：连续无 inf 则翻倍，出现 inf 则减半并跳过本步更新

### 8.2 梯度裁剪

> 梯度裁剪的公式、数字例子和训练循环中的位置见 **6.7 节**；梯度消失/爆炸的完整原因-方案对照表见 **6.9 节**。

**面试考法**：
- "梯度裁剪的原理是什么？" → 计算全局梯度 L2 范数，超过阈值就等比例缩小到阈值，方向不变
- "梯度裁剪能解决梯度消失吗？" → 不能。裁剪只设了上限，没设下限。梯度消失靠残差连接和归一化解决
- "为什么是全局裁剪而不是逐参数裁剪？" → 全局裁剪保持不同参数梯度之间的相对比例不变，逐参数裁剪会破坏方向信息

---

## 九、面试高频推导题汇总

| 题目 | 考察知识 | 难度 |
|------|---------|------|
| 推导 Softmax + 交叉熵的梯度 | 链式法则、Softmax Jacobian | ⭐⭐ |
| Attention 反传（$QK^T$ 部分） | 矩阵乘法反传 | ⭐⭐ |
| 推导 BatchNorm 的反传 | 链式法则、复合函数 | ⭐⭐⭐ |
| 为什么 Attention 除以 $\sqrt{d}$ | 方差计算、数值稳定性 | ⭐ |
| L1 vs L2 正则化的几何解释 | 范数、凸优化 | ⭐ |
| Adam 的偏差修正推导 | 期望、几何级数 | ⭐⭐ |
| 推导 Transformer 的参数量 | 矩阵维度计算 | ⭐ |
| LoRA 为什么有效 | 低秩近似、SVD | ⭐ |
| KL 散度的非对称性解释 | 信息论 | ⭐ |
| MLE 与交叉熵的等价关系 | 概率、信息论 | ⭐⭐ |

---

## 学习路线建议

```
第一周：线性代数（1.1-1.5）+ 微积分（2.1-2.3）
        → 能看懂矩阵运算和求导

第二周：矩阵微积分（3.1-3.6）+ 常见层反传
        → 能手推 Attention 反传

第三周：概率论（4.1-4.5）+ 信息论（5.1-5.3）
        → 能解释损失函数和采样

第四周：优化（6.1-6.6）+ 数值计算（8.1-8.2）
        → 能回答训练相关问题

贯穿：每天做一道推导题（第九节的列表）
```

import torch
from torch import nn
import math
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def norm(self, x):
        # x.pow(2).mean(-1, keepdim=True) → 沿 hidden_dim 求 x² 的均值
        # torch.rsqrt → 1/sqrt，比 sqrt+除法更快（GPU 上一条指令）
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先转 float32 算 norm（输入可能是 bf16/fp16，低精度算 rsqrt 数值不稳定）
        # 算完 .type_as(x) 转回原精度
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim, end=32768, rope_base=1e6):
    # 1. 每个维度对的基础频率: 1/base^(2i/dim)
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))
    # shape: [dim//2] = [48]，从高频到低频

    # 2. 外积: 位置 × 频率 = 角度矩阵
    t = torch.arange(end)              # [0, 1, ..., 32767]
    freqs = torch.outer(t, freqs)      # [32768, 48]

    # 3. cos/sin，cat 两份拼成 head_dim 长度（给 rotate_half 用）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)  # [32768, 96]
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)  # [32768, 96]
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        # [x1,...,x48, x49,...,x96] → [-x49,...,-x96, x1,...,x48]
        # 等价于每对二维坐标的旋转矩阵乘法的展开形式
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)

    # cos/sin: [S, head_dim] → [S, 1, head_dim]，广播到所有 head
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    # 旋转公式: x' = x * cosθ + rotate_half(x) * sinθ
    q_embed = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_embed = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x, n_rep):
    B, S, n_kv, d = x.shape
    if n_rep == 1: return x  # MHA 直接返回
    return x[:, :, :, None, :] \
            .expand(B, S, n_kv, n_rep, d) \
            .reshape(B, S, n_kv * n_rep, d)
    # expand 是虚拟复制（共享内存，零开销）
    # K₀ K₁ K₂ K₃ → K₀ K₀ K₁ K₁ K₂ K₂ K₃ K₃

class Attention(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8, n_kv_heads=4, head_dim=96):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = head_dim

        self.wq = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape

        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)  # [B, n_heads, S, head_dim]
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.full((S, S), float("-inf"), device=x.device).triu(1)
        scores = scores + mask
        output = F.softmax(scores.float(), dim=-1).type_as(xq) @ xv

        output = output.transpose(1, 2).reshape(B, S, -1)
        return self.wo(output)
    
class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=2432):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  # Step 3: 门控
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)  # Step 3: 候选特征
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)  # Step 3: 降维

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    
vocab_size = 6400    # 词表大小：能认多少个"字"
hidden_size = 768    # 每个字映射成多长的向量

embed_tokens = nn.Embedding(vocab_size, hidden_size)
# 本质：一个 [6400, 768] 的可学习矩阵
# 每一行是一个 token 的向量表示，初始随机，训练中更新
input_ids = torch.tensor([[1, 42, 7, 100]])  # [batch=1, seq_len=4]
hidden_states = embed_tokens(input_ids)       # [1, 4, 768]
dropout = nn.Dropout(0.0)  # MiniMind 默认 0.0，但架构里留了这个口子
hidden_states = dropout(embed_tokens(input_ids))
# [batch, seq_len, 768]

lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
# 权重矩阵: [768, 6400]（Linear 内部转置存储）
embed_tokens.weight = lm_head.weight  # 共享权重，省 ~490 万参数

class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=2432):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=2432,
                 num_experts=4, num_experts_per_tok=1,
                 norm_topk_prob=True, router_aux_loss_coef=5e-4):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(
            scores, k=self.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * self.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(B, S, D)

import torch
from torch import nn
import math
import torch.nn.functional as F


# ---- Chapter 02: RMSNorm ----

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


# ---- Chapter 03: RoPE ----

def precompute_freqs_cis(dim, end=32768, rope_base=1e6):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos + rotate_half(q) * sin).to(q.dtype)
    k_embed = (k * cos + rotate_half(k) * sin).to(k.dtype)
    return q_embed, k_embed


# ---- Chapter 04: Attention (GQA + QK Norm + KV Cache + SDPA) ----

def repeat_kv(x, n_rep):
    B, S, n_kv, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(B, S, n_kv, n_rep, d).reshape(B, S, n_kv * n_rep, d)


class Attention(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8, n_kv_heads=4, head_dim=96,
                 dropout=0.0, flash_attn=True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.flash = hasattr(F, 'scaled_dot_product_attention') and flash_attn

    def forward(self, x, position_embeddings, past_key_value=None,
                use_cache=False, attention_mask=None):
        B, S, _ = x.shape
        xq = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV Cache: 把历史 KV 和当前新 token 的 KV 沿 seq 维拼接
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)                         # [B, heads, S, dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)  # GQA repeat
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        if self.flash and S > 1 and past_key_value is None and \
                (attention_mask is None or torch.all(attention_mask == 1)):
            # SDPA 路径（训练时走这里）
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # 手写路径（带 KV Cache 的推理、有 padding mask 时走这里）
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -S:] += torch.full((S, S), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask[:, None, None, :]) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        output = output.transpose(1, 2).reshape(B, S, -1)
        return self.resid_dropout(self.wo(output)), past_kv


# ---- Chapter 05: FeedForward (SwiGLU) ----

class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=2432):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---- Chapter 06: MOEFeedForward ----

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
                y.index_add_(0, token_idx,
                             (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * self.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(B, S, D)


# ---- Chapter 07: MiniMindBlock ----

class MiniMindBlock(nn.Module):
    def __init__(self, hidden_size=768, n_heads=8, n_kv_heads=4,
                 head_dim=96, intermediate_size=2432, use_moe=False,
                 dropout=0.0, flash_attn=True):
        super().__init__()
        self.self_attn = Attention(hidden_size, n_heads, n_kv_heads, head_dim,
                                   dropout, flash_attn)
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = MOEFeedForward(hidden_size, intermediate_size) if use_moe else FeedForward(hidden_size, intermediate_size)

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None):
        # Pre-Norm + Attention + 残差
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings, past_key_value, use_cache, attention_mask,
        )
        hidden_states = hidden_states + residual

        # Pre-Norm + FFN + 残差
        residual = hidden_states
        hidden_states = self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        hidden_states = hidden_states + residual

        return hidden_states, present_key_value


# ---- 验证 ----

if __name__ == '__main__':
    # ---- Embedding + LM Head ----
    vocab_size = 6400
    hidden_size = 768
    embed_tokens = nn.Embedding(vocab_size, hidden_size)
    input_ids = torch.randint(0, vocab_size, (2, 128))
    hidden_states = embed_tokens(input_ids)
    dropout = nn.Dropout(0.0)
    hidden_states = dropout(hidden_states)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    embed_tokens.weight = lm_head.weight
    assert hidden_states.shape == (2, 128, 768)
    print("Embedding + LM Head verified!")

    # ---- RMSNorm ----
    norm = RMSNorm(768)
    assert norm(hidden_states).shape == hidden_states.shape
    print("RMSNorm verified!")

    # ---- RoPE ----
    freqs_cos, freqs_sin = precompute_freqs_cis(dim=96, end=512)
    assert freqs_cos.shape == (512, 96)
    print("RoPE verified!")

    # ---- Attention (含 KV Cache) ----
    attn = Attention()
    cos, sin = freqs_cos[:128], freqs_sin[:128]
    out, kv = attn(hidden_states, (cos, sin), use_cache=True)
    assert out.shape == hidden_states.shape
    assert kv[0].shape == (2, 128, 4, 96)  # KV Cache: [B, S, n_kv_heads, head_dim]
    # KV Cache 推理：只传 1 个新 token，复用历史 cache
    new_token = torch.randn(2, 1, 768)
    cos2, sin2 = freqs_cos[128:129], freqs_sin[128:129]
    out2, kv2 = attn(new_token, (cos2, sin2), past_key_value=kv, use_cache=True)
    assert out2.shape == (2, 1, 768)
    assert kv2[0].shape == (2, 129, 4, 96)  # cache 增长了 1
    print("Attention (GQA + QK Norm + KV Cache + SDPA) verified!")

    # ---- FeedForward ----
    ffn = FeedForward()
    x = torch.randn(2, 128, 768)
    assert ffn(x).shape == x.shape
    print("FeedForward verified!")

    # ---- MOEFeedForward ----
    moe = MOEFeedForward()
    x = torch.randn(2, 128, 768)
    assert moe(x).shape == x.shape
    total = sum(p.numel() for p in moe.parameters())
    expert_params = sum(p.numel() for p in moe.experts.parameters())
    router_params = sum(p.numel() for p in moe.gate.parameters())
    print(f"MoE total params: {total:,}")
    print(f"  Expert params:  {expert_params:,}")
    print(f"  Router params:  {router_params:,}")
    print("MOEFeedForward verified!")

    # ---- MiniMindBlock ----
    block = MiniMindBlock()
    x = torch.randn(2, 128, 768)
    cos, sin = freqs_cos[:128], freqs_sin[:128]
    out, kv = block(x, (cos, sin), use_cache=True)
    assert out.shape == x.shape
    assert kv[0].shape == (2, 128, 4, 96)
    print(f"Block params: {sum(p.numel() for p in block.parameters()):,}")
    print("MiniMindBlock verified!")

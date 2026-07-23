import torch
from torch import nn
import math
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


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


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # --- 模型尺寸 ---
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", hidden_size // self.num_attention_heads)
        self.vocab_size = kwargs.get("vocab_size", 6400)

        # --- FFN ---
        self.hidden_act = kwargs.get("hidden_act", "silu")
        # intermediate_size 的经验公式：ceil(hidden_size × π / 64) × 64
        # 8/3 ≈ π 是 SwiGLU 的常用扩展比，对齐到 64 的倍数方便 GPU 计算
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )

        # --- 位置编码 ---
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rope_theta = kwargs.get("rope_theta", 1e6)

        # --- 归一化 & 正则化 ---
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.dropout = kwargs.get("dropout", 0.0)

        # --- 权重共享 & Flash Attention ---
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.flash_attn = kwargs.get("flash_attn", True)

        # --- 特殊 token ---
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)

        # --- MoE（use_moe=True 时生效） ---
        self.use_moe = use_moe
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


# ---- Chapter 08: MiniMindModel ----

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            MiniMindBlock(
                hidden_size=config.hidden_size,
                n_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.moe_intermediate_size if config.use_moe else config.intermediate_size,
                use_moe=config.use_moe,
                dropout=config.dropout,
                flash_attn=config.flash_attn,
            )
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        B, S = input_ids.shape

        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + S],
            self.freqs_sin[start_pos:start_pos + S],
        )

        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_kv, use_cache=use_cache, attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss


# ---- Chapter 08: MiniMindForCausalLM ----

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache
        )
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        # MoeCausalLMOutputWithPast(...) 是在创建一个对象，括号里的 loss=loss, logits=logits 是传给构造函数的参数——告诉它"把 loss 存到 .loss 属性，把 logits 存到 .logits 属性"。创建完的对象通过 return 返回给调用方。
        return MoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits,
            past_key_values=past_key_values, hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens=8192, temperature=0.85, top_p=0.85,
                 top_k=50, eos_token_id=2, use_cache=True, do_sample=True,
                 repetition_penalty=1.0):
        input_ids = inputs
        past_key_values = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[:, past_len:], use_cache=use_cache, past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            if top_k > 0:
                threshold = torch.topk(logits, top_k)[0][..., -1, None]
                logits[logits < threshold] = -float('inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if finished.all():
                break

        return input_ids


# ---- 验证 ----

if __name__ == '__main__':
    # ---- Ch01-07: 各组件独立验证 ----
    freqs_cos, freqs_sin = precompute_freqs_cis(dim=96, end=512)
    assert freqs_cos.shape == (512, 96)
    print("RoPE verified!")

    attn = Attention()
    x = torch.randn(2, 128, 768)
    cos, sin = freqs_cos[:128], freqs_sin[:128]
    out, kv = attn(x, (cos, sin), use_cache=True)
    assert out.shape == x.shape
    assert kv[0].shape == (2, 128, 4, 96)
    print("Attention verified!")

    block = MiniMindBlock()
    out, kv = block(x, (cos, sin), use_cache=True)
    assert out.shape == x.shape
    print("MiniMindBlock verified!")

    # ---- Ch08: 完整模型 ----
    config = MiniMindConfig()
    model = MiniMindForCausalLM(config)
    print(f"Dense params: {sum(p.numel() for p in model.parameters()):,}")

    input_ids = torch.randint(0, 6400, (2, 128))
    result = model(input_ids, labels=input_ids)
    print(f"Logits: {result.logits.shape}")
    print(f"Loss: {result.loss.item():.4f}")
    result.loss.backward()
    print("Forward + backward verified!")

    # KV Cache: Prefill → Decode
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, 6400, (1, 10))
        r1 = model(prompt, use_cache=True)
        r2 = model(torch.randint(0, 6400, (1, 1)),
                   past_key_values=r1.past_key_values, use_cache=True)
        assert r2.logits.shape == (1, 1, 6400)
        assert r2.past_key_values[0][0].shape == (1, 11, 4, 96)
    print("KV Cache (Prefill → Decode) verified!")

    # Generate
    generated = model.generate(prompt, max_new_tokens=5, do_sample=False)
    assert generated.shape == (1, 15)
    print(f"Generate: {prompt.shape} → {generated.shape}")

    # MoE 版本
    moe_model = MiniMindForCausalLM(MiniMindConfig(use_moe=True))
    print(f"MoE params: {sum(p.numel() for p in moe_model.parameters()):,}")
    moe_r = moe_model(input_ids, labels=input_ids)
    (moe_r.loss + moe_r.aux_loss).backward()
    print("MoE (loss + aux_loss) backward verified!")

    print("\nFull model (Dense + MoE + KV Cache + Generate) all verified!")

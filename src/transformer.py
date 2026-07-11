"""The Transformer: SwiGLU FFN, GQA attention, and the GPT model that stacks them."""
from __future__ import annotations
import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config
from src.layers import RMSNorm, build_rope_cache, apply_rope


class SwiGLU(nn.Module):
    """Gated FFN: down(silu(gate(x)) * up(x)). Three matrices, not two."""
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = cfg.ffn_hidden()
        self.w_gate = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_up   = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Attention(nn.Module):
    """Causal self-attention with RoPE and optional grouped-query attention (GQA)."""
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads   # how many q-heads share each kv-head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)   # rotate q, k by position (not v)

        if self.n_rep > 1:                  # GQA: broadcast kv-heads up to match q-heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # FlashAttention kernel; is_causal handles the mask, no need to build one
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class Block(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + ffn(norm(x))."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight   # share input/output embeddings

        head_dim = cfg.dim // cfg.n_heads
        cos, sin = build_rope_cache(head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init on the residual-output projections (GPT-2 trick).
        # NOTE: these suffixes must match YOUR layer names (out_proj, w_down).
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"seq len {T} > max {self.cfg.max_seq_len}"
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss
        # inference: only the last position's logits are needed
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        # 2D tensors (matmuls, embeddings) get weight decay; 1D (norms) do not.
        decay, no_decay = [], []
        for p in self.parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8,
                                 **({"fused": True} if use_fused else {}))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200, eos_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]           # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")     # keep only top-k
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx

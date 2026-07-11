"""Core layers: RMSNorm and RoPE. (SwiGLU + Attention live in transformer.py)"""
from __future__ import annotations
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Pre-norm without mean-centering. Computed in fp32 for stability, cast back."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.gain.float()).to(out_dtype)


def build_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute (cos, sin) of shape (T, head_dim) — Llama rotate-half convention."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_seq_len).float()
    freqs = torch.outer(pos, inv_freq)          # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)     # (T, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(q, k, cos, sin):
    """Rotate q, k by position. q,k: (B, H, T, Dh); cos,sin: (T, Dh)."""
    def rotate_half(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)
    T = q.size(-2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

"""Single config object for the model + training run, loadable from YAML."""
from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Optional
import yaml


@dataclass
class Config:
    # --- model ---
    vocab_size: int = 8000
    dim: int = 640
    n_layers: int = 10
    n_heads: int = 10
    n_kv_heads: Optional[int] = None      # None -> full multi-head attention
    max_seq_len: int = 1024
    ffn_mult: float = 8 / 3               # SwiGLU keeps ~2/3 of 4x dense FFN
    ffn_multiple_of: int = 128
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    # --- training ---
    mix: dict = None                      # e.g. {"fineweb_edu": 0.5, "pubmed": 0.5}
    batch_size: int = 16
    grad_accum: int = 8
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    max_steps: int = 4000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    dtype: str = "bfloat16"

    def ffn_hidden(self) -> int:
        h = int(self.ffn_mult * self.dim)
        m = self.ffn_multiple_of
        return m * ((h + m - 1) // m)     # round up to a kernel-friendly multiple

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            print(f"[config] ignoring unknown keys: {unknown}")
        return cls(**{k: v for k, v in raw.items() if k in known})

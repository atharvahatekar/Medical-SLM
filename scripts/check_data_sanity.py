"""Sanity-check the real data: shapes, next-token shift, token range, and — most
importantly — that decoded batches read as real text (proves .bin <-> tokenizer agree)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer
from src.dataset import TokenMixLoader

MIX = {"fineweb_edu": 0.5, "pubmed": 0.5}
tok = Tokenizer.from_file("tokenizer/tokenizer.json")
vocab = tok.get_vocab_size()

loader = TokenMixLoader(
    data_dir="data", mix=MIX,
    seq_len=128, batch_size=8, device="cuda", split="train",
)
x, y = loader.next_batch()

# 1) shapes
print("x", tuple(x.shape), "y", tuple(y.shape))          # (8, 128) (8, 128)
assert x.shape == (8, 128) and y.shape == (8, 128)

# 2) next-token shift: y is x shifted left by one
assert torch.equal(x[:, 1:], y[:, :-1]), "shift is wrong!"
print("shift check: OK")

# 3) all token ids are inside the vocab
assert x.min() >= 0 and x.max() < vocab, f"token id out of range (vocab={vocab})"
print(f"token range: OK  (min={x.min().item()}, max={x.max().item()}, vocab={vocab})")

# 4) THE important one: decode row 0 back to text
print("\n--- decoded sample (row 0) ---")
print(tok.decode(x[0].tolist()))

# 5) val split loads too
vloader = TokenMixLoader(data_dir="data", mix=MIX, seq_len=128,
                        batch_size=4, device="cuda", split="val")
vx, vy = vloader.next_batch()
print("\nval batch:", tuple(vx.shape), "OK")

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))


import numpy as np, os, torch
from src.dataset import TokenMixLoader

os.makedirs("data", exist_ok=True)
# fake corpora: 100k random tokens each
np.arange(100_000, dtype=np.uint16).tofile("data/fineweb_edu.train.bin")
(np.arange(100_000, dtype=np.uint16) % 8000).astype(np.uint16).tofile("data/pubmed.train.bin")

loader = TokenMixLoader(
    data_dir="data",
    mix={"fineweb_edu": 0.5, "pubmed": 0.5},
    seq_len=64, batch_size=8, device="cpu", split="train",
)
x, y = loader.next_batch()
print("x", x.shape, "y", y.shape)          # (8, 64) (8, 64)
assert torch.equal(x[:, 1:], y[:, :-1])    # y is x shifted by one
print("shift check passed")

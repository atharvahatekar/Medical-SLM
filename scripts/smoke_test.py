from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

import torch, math
from src.config import Config
from src.transformer import GPT

# 1) forward + loss
cfg = Config(vocab_size=8000, dim=128, n_layers=2, n_heads=4, max_seq_len=64)
m = GPT(cfg)
x = torch.randint(0, 8000, (2, 64))
logits, loss = m(x, targets=x)
print(f"{m.num_params()/1e6:.1f}M params  loss={loss.item():.2f}")   # loss ~9 (=ln 8000)



# 2) GQA path actually runs (2 kv-heads shared across 4 q-heads)
gqa = GPT(Config(vocab_size=8000, dim=128, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64))
gqa(x, targets=x)

# 3) generation produces new tokens
out = m.generate(x[:, :5], max_new_tokens=10)
print("generated shape:", out.shape)   # (2, 15)





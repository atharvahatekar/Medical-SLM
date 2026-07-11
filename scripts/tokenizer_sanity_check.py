from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer/tokenizer.json")
s = "Aspirin inhibits cyclooxygenase to reduce prostaglandin synthesis."
ids = tok.encode(s).ids
print(len(ids), "tokens ->", tok.decode(ids))
for t in ["<|endoftext|>", "<|user|>", "<|assistant|>"]:
    print(t, tok.token_to_id(t))    # all should be non-None (ids 0..4)
    
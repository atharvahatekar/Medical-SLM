"""Instruction-tune a base checkpoint. Chat format:
    <|user|>\n{q}<|assistant|>\n{a}<|endoftext|>
Loss is masked to the assistant's response tokens only."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

import json
import argparse
import math
import torch
from tokenizers import Tokenizer
from src.config import Config
from src.transformer import GPT


def build_examples(path, tok, max_len):
    u_id = tok.token_to_id("<|user|>")
    a_id = tok.token_to_id("<|assistant|>")
    eot = tok.token_to_id("<|endoftext|>")
    examples = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        prompt = [u_id] + tok.encode("\n" + r["user"]).ids + [a_id]
        resp = tok.encode("\n" + r["assistant"]).ids + [eot]
        full = (prompt + resp)[:max_len]
        labels = ([-1] * len(prompt) + resp)[:max_len]   # mask the prompt
        if len(full) < 4:
            continue
        examples.append((full, labels))
    return examples, eot


def collate(batch, pad_id, device):
    m = max(len(f) for f, _ in batch)
    X, Y = [], []
    for full, labels in batch:
        pad = m - len(full)
        X.append(full[:-1] + [pad_id] * pad)      # inputs
        Y.append(labels[1:] + [-1] * pad)         # shifted targets, pad masked
    x = torch.tensor(X, dtype=torch.long, device=device)
    y = torch.tensor(Y, dtype=torch.long, device=device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/local30m/best.pt")
    ap.add_argument("--data", default="data/sft_train.jsonl")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out", default="runs/local30m/best_sft.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id("<|pad|>")

    ck = torch.load(args.base, map_location=device)
    cfg = Config(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.train()

    examples, _ = build_examples(args.data, tok, cfg.max_seq_len)
    print(f"[sft] {len(examples)} examples, {args.epochs} epochs")

    opt = model.configure_optimizers(0.1, args.lr, (0.9, 0.95), device)
    use_fp16 = cfg.dtype == "float16"
    autocast = (torch.amp.autocast(device_type="cuda",
                dtype=torch.float16 if use_fp16 else torch.bfloat16)
                if device == "cuda" else torch.autocast("cpu", enabled=False))
    scaler = torch.amp.GradScaler(enabled=use_fp16)

    import random
    random.seed(0)
    total = args.epochs * math.ceil(len(examples) / args.batch_size)
    step = 0
    for ep in range(args.epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            x, y = collate(batch, pad_id, device)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / total))  # cosine to 0
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with autocast:
                _, loss = model(x, targets=y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 50 == 0:
                print(f"epoch {ep+1} step {step}/{total} | loss {loss.item():.3f} | lr {lr:.2e}")

    torch.save({"model": model.state_dict(), "cfg": ck["cfg"]}, args.out)
    print(f"[sft] saved -> {args.out}")


if __name__ == "__main__":
    main()
"""python src/sft.py --base runs/local30m/best.pt --out runs/local30m/best_sft.pt --epochs 3"""
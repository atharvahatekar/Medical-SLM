from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

"""Zero-shot multiple-choice eval by answer-likelihood.
Scores each option by the model's length-normalized log-prob; picks the argmax."""
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from datasets import load_dataset
from src.config import Config
from src.transformer import GPT


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    cfg = Config(**ck["cfg"])
    m = GPT(cfg).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, cfg


@torch.no_grad()
def option_score(model, tok, context, option, device, max_len):
    ctx = tok.encode(context).ids
    opt = tok.encode(" " + option).ids
    ids = (ctx + opt)[-max_len:]
    n_opt = len(opt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(inp, targets=inp)                # targets forces full-logit path
    logp = F.log_softmax(logits[0].float(), dim=-1)
    L = len(ids)
    total = sum(logp[j - 1, ids[j]].item() for j in range(L - n_opt, L))
    return total / max(n_opt, 1)                       # length-normalized


def eval_medmcqa(model, tok, device, max_len, limit):
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    keys, correct, total = ["opa", "opb", "opc", "opd"], 0, 0
    for ex in ds:
        opts = [ex[k] for k in keys]
        if not all(opts) or ex["cop"] is None:
            continue
        ctx = f"Question: {ex['question']}\nAnswer:"
        scores = [option_score(model, tok, ctx, o, device, max_len) for o in opts]
        correct += int(max(range(4), key=lambda i: scores[i]) == ex["cop"])
        total += 1
        if total >= limit:
            break
    return correct / total, total


def eval_pubmedqa(model, tok, device, max_len, limit):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    opts, correct, total = ["yes", "no", "maybe"], 0, 0
    for ex in ds:
        gold = (ex.get("final_decision") or "").strip().lower()
        if gold not in opts:
            continue
        ctx = f"Question: {ex['question']}\nAnswer:"
        scores = [option_score(model, tok, ctx, o, device, max_len) for o in opts]
        correct += int(opts[max(range(3), key=lambda i: scores[i])] == gold)
        total += 1
        if total >= limit:
            break
    return correct / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/local30m/best_sft.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--tasks", default="medmcqa,pubmedqa")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(args.tokenizer)
    model, cfg = load_model(args.ckpt, device)
    for task in args.tasks.split(","):
        if task == "medmcqa":
            acc, n = eval_medmcqa(model, tok, device, cfg.max_seq_len, args.limit)
            print(f"MedMCQA : {acc:.3f}  (n={n}, chance=0.250)")
        elif task == "pubmedqa":
            acc, n = eval_pubmedqa(model, tok, device, cfg.max_seq_len, args.limit)
            print(f"PubMedQA: {acc:.3f}  (n={n}, chance~0.333)")


if __name__ == "__main__":
    main()

"""
python src/evaluate.py --ckpt runs/local30m/best.pt --tasks medmcqa,pubmedqa --limit 500
python src/evaluate.py --ckpt runs/local30m/best_sft.pt --tasks medmcqa,pubmedqa --limit 500
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))



"""Generate text from a trained checkpoint. Use --chat for the SFT'd model."""
import argparse
import torch
from tokenizers import Tokenizer
from src.config import Config
from src.transformer import GPT


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ck["cfg"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/local30m/best_sft.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--chat", action="store_true", help="wrap prompt in the SFT chat template")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(args.tokenizer)
    model, cfg = load_model(args.ckpt, device)
    eot = tok.token_to_id("<|endoftext|>")

    if args.chat:
        u, a = tok.token_to_id("<|user|>"), tok.token_to_id("<|assistant|>")
        ids = [u] + tok.encode("\n" + args.prompt).ids + [a] + tok.encode("\n").ids
    else:
        ids = tok.encode(args.prompt).ids

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k, eos_id=eot)
    new_ids = out[0].tolist()[len(ids):] if args.chat else out[0].tolist()
    print(tok.decode(new_ids))


if __name__ == "__main__":
    main()


"""
Some Examples sentence completion tests before SFT:
python src/generate.py --ckpt runs/local30m/best.pt --prompt "The mechanism of action of aspirin is"
python src/generate.py --ckpt runs/local30m/best.pt --prompt "Common side effects of metformin include"
python src/generate.py --ckpt runs/local30m/best.pt --prompt "Warfarin is an anticoagulant that"

Some examples of SFT chat prompts:
python src/generate.py --ckpt runs/local30m/best_sft.pt --chat --prompt "What is the mechanism of action of aspirin?"
python src/generate.py --ckpt runs/local30m/best_sft.pt --chat --prompt "What are the common side effects of warfarin?"
"""
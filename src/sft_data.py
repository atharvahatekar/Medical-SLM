"""Build a small instruction-tuning set from medical Q&A + some general instructions.
Writes JSONL lines: {"user": ..., "assistant": ...}"""
from __future__ import annotations
import json
import argparse
from datasets import load_dataset

LETTERS = ["A", "B", "C", "D"]


def medmcqa(n):
    ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=True)
    out = []
    for ex in ds:
        opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        cop = ex["cop"]
        if not all(opts) or cop is None:
            continue
        q = ex["question"].strip()
        body = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, opts))
        ans = f"The correct answer is {LETTERS[cop]}. {opts[cop]}."
        if ex.get("exp"):
            ans += f" {ex['exp'].strip()}"
        out.append({"user": f"{q}\n{body}", "assistant": ans})
        if len(out) >= n:
            break
    return out


def pubmedqa(n):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", streaming=True)
    out = []
    for ex in ds:
        q = (ex.get("question") or "").strip()
        dec = (ex.get("final_decision") or "").strip()
        long = (ex.get("long_answer") or "").strip()
        if not q or not long:
            continue
        ans = f"{dec.capitalize()}. {long}" if dec else long
        out.append({"user": q, "assistant": ans})
        if len(out) >= n:
            break
    return out


def alpaca(n):
    ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    out = []
    for ex in ds:
        instr, inp, outp = ex["instruction"].strip(), ex.get("input", "").strip(), ex["output"].strip()
        if not instr or not outp:
            continue
        user = f"{instr}\n{inp}" if inp else instr
        out.append({"user": user, "assistant": outp})
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_train.jsonl")
    ap.add_argument("--n-medmcqa", type=int, default=6000)
    ap.add_argument("--n-pubmedqa", type=int, default=4000)
    ap.add_argument("--n-general", type=int, default=2000)
    args = ap.parse_args()

    rows = medmcqa(args.n_medmcqa) + pubmedqa(args.n_pubmedqa) + alpaca(args.n_general)
    import random
    random.seed(0)
    random.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[sft_data] wrote {len(rows)} examples -> {args.out}")


if __name__ == "__main__":
    main()
"""python src/sft_data.py --n-medmcqa 6000 --n-pubmedqa 4000 --n-general 2000"""
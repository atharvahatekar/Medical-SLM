"""Build the tokenizer and the training .bin files for a small from-scratch run.

Two sources, streamed straight from the HuggingFace Hub (no GPU, no Modal):
    fineweb_edu -> general English fluency
    pubmed      -> PubMed titles + abstracts (the biomedical/pharma domain)

Pipeline:
    1. train a byte-level BPE on a blend of both sources
    2. tokenize a token budget of each source into <data_dir>/<name>.{train,val}.bin
            (uint16, since vocab <= 65536), holding out a small validation tail

Example:
    python scripts/build_data.py --vocab-size 8000 \
        --general-budget 150000000 --domain-budget 150000000
"""
from __future__ import annotations
import os
import argparse
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import ByteLevel as ByteLevelProcessor

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>"]

# name -> how to open its streaming dataset + which field holds the text
SOURCES = {
    "fineweb_edu": dict(
        load=lambda: load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                                split="train", streaming=True),
        text_key="text",
    ),
    "pubmed": dict(
        load=lambda: load_dataset("casinca/PUBMED_title_abstracts_2019_baseline",
                                split="train", streaming=True),
        text_key="text",
    ),
}


def stream_texts(name, max_docs=None):
    """Yield non-empty text documents from a source's streaming dataset."""
    spec = SOURCES[name]
    key = spec["text_key"]
    n = 0
    for ex in spec["load"]():
        txt = ex.get(key) or next((v for v in ex.values()
                                if isinstance(v, str) and len(v) > 20), None)
        if txt and txt.strip():
            yield txt
            n += 1
            if max_docs and n >= max_docs:
                return


# --------------------------------------------------------------------------- #
# 1) tokenizer
# --------------------------------------------------------------------------- #
def train_tokenizer(out_path, vocab_size, docs_per_source):
    """Train a byte-level BPE on a round-robin blend of the sources."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = ByteLevelProcessor(trim_offsets=False)

    def blend():
        # round-robin so the sample reflects the mix, not just the first source.
        # cap each doc at 5000 chars: BPE only needs local context, and un-truncated
        # FineWeb docs blow up the trainer's word table.
        streams = {n: stream_texts(n, docs_per_source) for n in SOURCES}
        active = dict(streams)
        while active:
            for name in list(active):
                try:
                    yield next(active[name])[:5000]
                except StopIteration:
                    del active[name]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(blend(), trainer=trainer)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tok.save(out_path)

    # fertility sanity check on a dense pharma string (lower = better)
    sample = ("Pharmacokinetics of acetaminophen: hepatic glucuronidation and "
            "cytochrome P450-mediated metabolism in hepatic impairment.")
    ids = tok.encode(sample).ids
    print(f"[tok] saved {out_path}  vocab={tok.get_vocab_size()}  "
        f"fertility={len(ids)/len(sample.split()):.2f} tok/word")
    return tok


# --------------------------------------------------------------------------- #
# 2) tokenize each source -> .bin
# --------------------------------------------------------------------------- #
def tokenize_source(name, tok, data_dir, token_budget, val_frac=0.005):
    train_path = os.path.join(data_dir, f"{name}.train.bin")
    if os.path.exists(train_path) and os.path.getsize(train_path) > 0:
        print(f"[data] {name}: {train_path} exists -- skipping (delete to rebuild)")
        return

    eot = tok.token_to_id("<|endoftext|>")
    buf, total = [], 0
    for txt in stream_texts(name):
        buf.append(np.asarray(tok.encode(txt).ids + [eot], dtype=np.uint16))
        total += len(buf[-1])
        if total >= token_budget:
            break
        if total % 5_000_000 < len(buf[-1]):
            print(f"[data] {name}: {total/1e6:.0f}M / {token_budget/1e6:.0f}M tokens", flush=True)

    toks = np.concatenate(buf)[:token_budget]
    n_val = int(len(toks) * val_frac)
    toks[n_val:].tofile(train_path)                                   # train tail
    toks[:n_val].tofile(os.path.join(data_dir, f"{name}.val.bin"))    # val head
    print(f"[data] {name}: wrote {len(toks)/1e6:.1f}M tokens "
        f"({n_val/1e6:.2f}M val) -> {data_dir}/{name}.*.bin")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokenizer-path", default="tokenizer/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--tok-docs", type=int, default=50_000,
                    help="docs per source used to TRAIN the tokenizer")
    ap.add_argument("--general-budget", type=int, default=150_000_000)
    ap.add_argument("--domain-budget", type=int, default=150_000_000)
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if os.path.exists(args.tokenizer_path):
        print(f"[tok] {args.tokenizer_path} exists -- loading (delete to retrain)")
        tok = Tokenizer.from_file(args.tokenizer_path)
    else:
        tok = train_tokenizer(args.tokenizer_path, args.vocab_size, args.tok_docs)

    tokenize_source("fineweb_edu", tok, args.data_dir, args.general_budget)
    tokenize_source("pubmed",      tok, args.data_dir, args.domain_budget)

    print("\n[done] tokenizer + data ready. Next: train with configs/small.yaml")


if __name__ == "__main__":
    main()

"""Plot pretraining curves from a run's metrics.jsonl.

Produces one figure: average val-loss, per-source val-loss (domain vs general),
and train-loss vs. step -- the single picture that shows the model learned.

    python scripts/plot_metrics.py --run runs/local30m --out assets/loss_curve.png
"""
from __future__ import annotations
import os
import json
import argparse
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/local30m")
    ap.add_argument("--out", default="assets/loss_curve.png")
    args = ap.parse_args()

    rows = load(os.path.join(args.run, "metrics.jsonl"))
    steps = [r["step"] for r in rows]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["val_avg"] for r in rows], label="val (avg)", lw=2.2)
    if "val_pubmed" in rows[0]:
        plt.plot(steps, [r["val_pubmed"] for r in rows],
                 label="val — PubMed (domain)", lw=2)
    if "val_fineweb_edu" in rows[0]:
        plt.plot(steps, [r["val_fineweb_edu"] for r in rows],
                 label="val — FineWeb-Edu (general)", lw=2)
    if "train_loss" in rows[0]:
        plt.plot(steps, [r["train_loss"] for r in rows],
                 label="train", ls="--", alpha=0.5)

    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title("Medical-SLM (~30M) — pretraining loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

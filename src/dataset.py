"""Mixing token loader.

Each source lives in a flat uint16 file: <data_dir>/<name>.<split>.bin
At batch time we pick a source per sequence (weighted by `mix`), then read a
random contiguous window from it via memmap. Mix ratio is therefore a runtime
knob -- no re-tokenizing needed to change the general/domain balance.
"""
from __future__ import annotations
import os
import numpy as np
import torch


class TokenMixLoader:
    def __init__(self, data_dir, mix, seq_len, batch_size, device,
                split="train", seed=1337):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.on_cuda = "cuda" in str(device)

        names, weights, self.lengths = [], [], []
        for name, w in mix.items():
            path = self._resolve(name, split)
            if path is None:
                print(f"[data] dropping '{name}' (no usable {split} file)")
                continue
            n = len(np.memmap(path, dtype=np.uint16, mode="r"))
            if n <= seq_len + 1:
                print(f"[data] dropping '{name}' (only {n} tokens)")
                continue
            names.append((name, path))
            weights.append(float(w))
            self.lengths.append(n)

        if not names:
            raise RuntimeError(f"No usable sources in {data_dir} (split={split}, mix={mix})")

        self.sources = names
        w = np.asarray(weights, dtype=np.float64)
        self.weights = w / w.sum()                       # normalize so they sum to 1
        # separate RNG per split so train/val windows don't move together
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 99))

        total = sum(self.lengths)
        summary = ", ".join(
            f"{n}:{L/1e6:.1f}M(w={ww:.2f})"
            for (n, _), L, ww in zip(self.sources, self.lengths, self.weights)
        )
        print(f"[data] split={split}  {summary}  total={total/1e9:.3f}B toks")

    def _resolve(self, name, split):
        """Prefer the requested split; fall back to train for tiny sources with no val."""
        path = os.path.join(self.data_dir, f"{name}.{split}.bin")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        alt = os.path.join(self.data_dir, f"{name}.train.bin")
        if os.path.exists(alt) and os.path.getsize(alt) > 0:
            return alt
        return None

    def next_batch(self):
        # choose a source independently for each sequence in the batch
        picks = self.rng.choice(len(self.sources), size=self.batch_size, p=self.weights)
        xs, ys = [], []
        for pi in picks:
            _, path = self.sources[pi]
            n = self.lengths[pi]
            # re-open the memmap each time -- avoids the known memmap memory leak
            data = np.memmap(path, dtype=np.uint16, mode="r")
            start = int(self.rng.integers(0, n - self.seq_len - 1))
            window = data[start:start + self.seq_len + 1].astype(np.int64)
            xs.append(torch.from_numpy(window[:-1]))     # inputs
            ys.append(torch.from_numpy(window[1:]))      # targets shifted by one
        x = torch.stack(xs)
        y = torch.stack(ys)
        if self.on_cuda:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

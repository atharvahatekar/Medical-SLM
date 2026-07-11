"""Trainer: cosine-LR loop with warmup, gradient accumulation, per-source validation,
checkpointing, and resume-from-latest. Works on CPU or a single CUDA GPU."""
from __future__ import annotations
import os
import json
import time
import math
from contextlib import nullcontext
from dataclasses import asdict

import torch

from src.config import Config
from src.transformer import GPT
from src.dataset import TokenMixLoader

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
LOG_EVERY = 10


class Trainer:
    def __init__(self, cfg: Config, data_dir="data", out_dir="runs/small",
                resume=True, device=None):
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dev_type = "cuda" if "cuda" in self.device else "cpu"
        ptdtype = _DTYPES[cfg.dtype]
        # autocast only on CUDA; fp16 needs a grad scaler, bf16/fp32 do not
        self.autocast = (torch.amp.autocast(device_type="cuda", dtype=ptdtype)
                        if self.dev_type == "cuda" else nullcontext())
        self.scaler = torch.amp.GradScaler(enabled=(cfg.dtype == "float16"))

        self.model = GPT(cfg).to(self.device)
        print(f"[trainer] {self.model.num_params()/1e6:.1f}M params on {self.device} "
            f"({cfg.dtype})")
        self.opt = self.model.configure_optimizers(
            cfg.weight_decay, cfg.lr, betas=(0.9, 0.95), device_type=self.dev_type)

        # one mixed train loader; one single-source val loader per source (per-source loss)
        self.train_loader = TokenMixLoader(
            data_dir, cfg.mix, cfg.max_seq_len, cfg.batch_size, self.device, "train")
        self.val_loaders = {
            src: TokenMixLoader(data_dir, {src: 1.0}, cfg.max_seq_len,
                                cfg.batch_size, self.device, "val")
            for src in cfg.mix
        }

        self.step = 0
        self.best_val = float("inf")
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        if resume:
            self._maybe_resume()

    # --- learning-rate schedule: linear warmup, then cosine decay to min_lr --------
    def lr_at(self, step: int) -> float:
        c = self.cfg
        if step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        if step >= c.max_steps:
            return c.min_lr
        ratio = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))   # 1 -> 0
        return c.min_lr + coeff * (c.lr - c.min_lr)

    # --- validation: mean loss per source -----------------------------------------
    @torch.no_grad()
    def evaluate(self, n_batches=50) -> dict:
        self.model.eval()
        out = {}
        for src, loader in self.val_loaders.items():
            tot = 0.0
            for _ in range(n_batches):
                x, y = loader.next_batch()
                with self.autocast:
                    _, loss = self.model(x, targets=y)
                tot += loss.item()
            out[src] = tot / n_batches
        self.model.train()
        return out

    # --- the training loop --------------------------------------------------------
    def fit(self):
        c = self.cfg
        tok_per_step = c.batch_size * c.grad_accum * c.max_seq_len
        self.model.train()
        t0 = time.time()
        while self.step < c.max_steps:
            lr = self.lr_at(self.step)
            for g in self.opt.param_groups:
                g["lr"] = lr

            self.opt.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(c.grad_accum):
                x, y = self.train_loader.next_batch()
                with self.autocast:
                    _, loss = self.model(x, targets=y)
                    loss = loss / c.grad_accum
                self.scaler.scale(loss).backward()
                loss_accum += loss.item()

            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.step += 1

            if self.step % LOG_EVERY == 0:
                dt = time.time() - t0
                tps = tok_per_step * LOG_EVERY / dt
                print(f"step {self.step:>5} | loss {loss_accum:.3f} | lr {lr:.2e} "
                    f"| {tps/1e3:.0f}k tok/s")
                t0 = time.time()

            if self.step % c.eval_every == 0 or self.step == c.max_steps:
                val = self.evaluate()
                avg = sum(val.values()) / len(val)
                self._log({"step": self.step, "train_loss": loss_accum, "lr": lr,
                           "val_avg": avg, **{f"val_{k}": v for k, v in val.items()}})
                srcs = "  ".join(f"{k}={v:.3f}" for k, v in val.items())
                star = ""
                if avg < self.best_val:
                    self.best_val = avg
                    self._save("best.pt")
                    star = "  <- best"
                self._save("latest.pt")
                print(f"  [eval] step {self.step} | val_avg {avg:.3f} | {srcs}{star}")

        self._save("final.pt")
        print(f"[trainer] done. best val_avg={self.best_val:.3f}. ckpts in {self.out_dir}/")

    # --- checkpoint / resume / logging --------------------------------------------
    def _save(self, name):
        torch.save({
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step,
            "best_val": self.best_val,
            "cfg": asdict(self.cfg),
        }, os.path.join(self.out_dir, name))

    def _maybe_resume(self):
        path = os.path.join(self.out_dir, "latest.pt")
        if not os.path.exists(path):
            return
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.step = ck["step"]
        self.best_val = ck["best_val"]
        print(f"[trainer] resumed from {path} at step {self.step}")

    def _log(self, row: dict):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

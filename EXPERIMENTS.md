# Experiments — Medical-SLM (from scratch)

## Setup
- **Model:** ~30M params — decoder-only Transformer, dim 512, 8 layers, 8 heads,
  512 context. RoPE, RMSNorm (pre-norm), SwiGLU, weight tying. Hand-written PyTorch.
- **Tokenizer:** custom 8k byte-level BPE, trained on a general+pharma blend.
  Fertility ~2.75 tok/word on dense pharma text.
- **Data:** FineWeb-Edu (general) + PubMed abstracts (domain), 50/50 mix,
  ~298M tokens tokenized (149M each), memmap mixing loader (mix ratio = runtime knob).
- **Hardware:** single RTX 2060 (6 GB), fp16, local. Cost: $0.

## Pretraining (6000 steps, ~197M tokens seen, ~0.7 epoch)
- val_avg 4.94 → **3.08**; val_pubmed 4.67 → **2.69**; val_fineweb_edu 5.20 → **3.46**.
- train_loss (3.04) ≈ val_avg (3.08): **no overfitting → data/capacity-bound.**
- Note: domain loss < general loss (PubMed abstracts are more formulaic).

## Generation (base model)
- Fluent, correct medical *register*; facts unreliable, invents plausible compounds
  (e.g. "dipropionulphoxide"). Illustrates fluency ≠ correctness at small scale.

## SFT (12k examples: 6k MedMCQA, 4k PubMedQA, 2k Alpaca; 3 epochs)
- Response-masked loss 3.0 → ~2.0. Model shifts from completing text to answering.

## Evaluation (zero-shot, answer-likelihood, n=500)
| Task | Base | SFT | Chance |
|---|---|---|---|
| MedMCQA (4-opt) | 0.258 | 0.260 | 0.250 |
| PubMedQA (y/n/maybe) | 0.464 | 0.514 | 0.333 |

## Findings
1. **Capacity-bound on hard MCQ:** MedMCQA at chance, unmoved by SFT.
2. **Real signal on simpler tasks:** PubMedQA above chance, SFT +0.05.
3. **Fluent domain writer** with unreliable facts — the fluency/reasoning split.
4. **Next step to fix it:** scale params (30M → 60M+) and tokens; pipeline supports it.

## Ablations to run next
- [Ongoing]  Learning rate: 3e-4 vs 8e-4 vs 1e-3 (compare best val_pubmed).
- [Future-Work]  Mix ratio: 30/70 vs 50/50 vs 70/30 domain (compare val_pubmed vs val_fineweb).
- [Future-Work]  Scale: rent an L4, train 60M, check whether MedMCQA finally moves off chance.

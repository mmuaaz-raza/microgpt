# MicroGPT

A **from-scratch, NumPy-only decoder-only Transformer** trained as a character-level language model on Tiny Shakespeare.  
No PyTorch, no JAX, no autograd — every forward pass, backward pass, and Adam update is implemented by hand with pure NumPy array operations.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/backend-NumPy-013243.svg)](https://numpy.org/)

---

## Key Features

- **Pure NumPy** — zero deep-learning framework dependencies
- **Manual backpropagation** — full analytical gradients for multi-head attention, GELU FFN, LayerNorm, embeddings, and unembedding
- **Adam with schedule** — linear warmup + inverse-time decay, implemented from the original equations
- **Complete math write-ups** — two companion documents deriving every forward and backward equation
- **Reproducible training** — loss curves, checkpoints, and qualitative generation samples included

---

## Architecture

```
Token Embedding  (vocab_size × d_model)
        ↓  + Sinusoidal Positional Encoding
[ LayerNorm → Multi-Head Causal Attention → Residual ]  × L
[ LayerNorm → Feed-Forward (GELU)         → Residual ]  × L
        ↓
Final LayerNorm → Unembedding → Softmax
```

| Component            | Detail                                              |
|----------------------|-----------------------------------------------------|
| Embedding            | Learned character embedding table                   |
| Positional encoding  | Fixed sinusoidal (Vaswani et al., 2017)             |
| Attention            | Multi-head causal self-attention with **Pre-LN**    |
| FFN activation       | GELU (tanh approximation, GPT-2 style)              |
| Optimiser            | Adam with linear warmup + inverse-time LR decay     |
| Loss                 | Cross-entropy (next-character prediction)           |

### Hyperparameters

| Parameter       | Value | Description                          |
|-----------------|-------|--------------------------------------|
| `d_model` (ed)  | 128   | Embedding / hidden dimension         |
| `heads`         | 4     | Parallel attention heads             |
| `d_k` / `d_v`   | 32    | Query-key / value dim per head       |
| `layers`        | 4     | Stacked Transformer blocks           |
| `ffn_wd`        | 512   | FFN hidden width (4 × d_model)       |
| `block_size`    | 64    | Context window (sequence length)     |
| `batch_size`    | 32    | Sequences per gradient step          |
| `base_alpha`    | 0.001 | Peak learning rate                   |
| `warmup_steps`  | 500   | Linear LR warmup                     |
| `beta1`, `beta2`| 0.9, 0.999 | Adam moment coefficients        |


---

## Mathematical Documentation

The two companion documents derive every equation used in the implementation:

| Document | Content |
|----------|---------|
| [**Part 1 — Forward Pass**](docs/Transformer-I.docx) | Architecture, design decisions (Pre-LN, causal mask, per-head projections), and full forward-pass mathematics with NumPy implementations |
| [**Part 2 — Backward Pass & Training**](docs/Transformer-II.docx) | Analytical gradients for Softmax, LayerNorm, GELU, attention, FFN, embeddings; Adam derivation; loss curves and generation samples |

Both documents use a unified notation and the same hyperparameter set as the code.

---

## Project Structure

```
microgpt/
├── model.py              # Transformer class (forward, backward, Adam, save/load)
├── Tmodel.py             # Dataclasses for parameters, dimensions, runtime caches
├── process_data.py       # Data loading, vocabulary, batch sampling, activations
├── train.py              # Training loop entry point
├── sample.py             # Autoregressive text generation from a checkpoint
├── stats.py              # Loss-curve plotting utility
├── input.txt             # Tiny Shakespeare corpus
├── loss_curve.png        # Train / validation loss plot
├── docs/
│   ├── Transformer_Part1_Forward_Pass.docx
│   └── Transformer_Part2_Backward_Pass.docx
├── pyproject.toml
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or plain pip
pip install numpy matplotlib
```

### 2. Train

```bash
python train.py
```

Training runs for **3 000 steps** by default with Adam.  
Checkpoints are saved periodically; loss is logged to JSON.

### 3. Generate text

```bash
python sample.py
```

Loads the latest checkpoint and streams characters to stdout (temperature = 0.8).  
Full output is also written to `output.txt`.

### 4. Plot the loss curve

```bash
python stats.py
```

Reads the loss history and saves/displays `loss_curve.png`.

---

## Training Progress

![Training and validation loss](loss_curve.png)

**Key observations**

- **Rapid initial descent** — loss drops from ~3.25 to ~1.60 in the first 1 000 steps as the model learns unigrams and frequent bigrams.
- **Syntactic formation** — by step 4 000 the loss reaches ~1.35; valid English words, punctuation and dialogue format appear.
- **Asymptotic behaviour** — training loss continues to ~1.22 while validation loss stabilises near 1.58. The ~0.36 gap is expected for a small character-level model trained without dropout or weight decay.


## Implementation Notes

### Why NumPy only?

This project is a learning exercise in understanding Transformer internals. Implementing every operation explicitly — including the LayerNorm backward pass, the Softmax Jacobian, and the multi-head attention gradient — makes the mechanics transparent and debuggable without relying on autograd.

### Gradient flow (high level)

Back-propagation is performed in `Transformer.backward()`:

1. **Final block** — combined Softmax + cross-entropy gradient, then unembedding and final LayerNorm.
2. **Transformer blocks (reverse order)** — FFN backward (GELU → two linears → LayerNorm) followed by attention backward (output projection → head split → Softmax → Q/K/V projections → LayerNorm).
3. **Embedding table** — scatter-add of the positional gradient into the rows that were looked up.

Full derivations are in [Part 2](docs/Transformer-II.docx).

### Weight initialisation

| Weights              | Scheme                          |
|----------------------|---------------------------------|
| Embedding `W_emb`    | \(\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})\) |
| Q, K, V              | \(\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})\) |
| Output proj `W_O`    | \(\mathcal{N}(0, 1/\sqrt{d_v})\) |
| FFN `W_0`            | \(\mathcal{N}(0, \sqrt{2/d_{\text{model}}})\) (He) |
| FFN `W_1`            | \(\mathcal{N}(0, \sqrt{2/\text{ffn_wd}})\) |
| Unembedding `W_U`    | \(\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})\) |
| LayerNorm \(\gamma\) | Ones                            |
| LayerNorm \(\beta\)  | Zeros                           |

---

## License

MIT

---

## Acknowledgements

Architecture follows the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy, 2023) and the original Transformer (Vaswani et al., 2017). All gradients and the Adam update are derived and implemented from first principles.

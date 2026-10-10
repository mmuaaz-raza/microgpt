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
- **Adam with schedule** — linear warmup + inverse-time decay, implemented from Vthe original equations
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
| `d_k/qk_d` / `d_v`   | 32    | Query-key / value dim per head       |
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
├── model.py                  # Transformer class (forward, backward, Adam, save/load)
├── Tmodel.py                 # Dataclasses for parameters, dimensions, runtime caches
├── process_data.py           # Data loading, vocabulary, batch sampling, activations
├── train.py                  # Training loop entry point with periodic sampling
├── sample.py                 # Autoregressive text generation from a checkpoint
├── stats.py                  # Loss-curve plotting utility
├── input.txt                 # Tiny Shakespeare corpus
├── loss_curve_shakespare.png # Train / validation loss plot
├── shakespareADAM.json       # Logged training history
├── ShakespareOutput*.txt     # Periodic qualitative generation samples
├── docs/
│   ├── Transformer-I.docx    # Part 1: Architecture & Forward Pass
│   └── Transformer-II.docx   # Part 2: Backward Pass & Training
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

Training runs for **4 000 steps** by default with Adam (`base_alpha = 0.001`, `warmup_steps = 500`, `decay_rate = 1e-3`).  
Checkpoints and qualitative generation samples (`ShakespareOutput{step}.txt`) are saved periodically, and loss history is logged to `shakespareADAM.json`.

### 3. Generate text

```bash
python sample.py
```

Loads the latest checkpoint (`shakespare.pkl`) and streams characters to stdout (temperature = 0.8).  
Full output is also written to `output.txt`.

### 4. Plot the loss curve

```bash
python stats.py
```

Reads the loss history and saves/displays `loss_curve_shakespare.png`.

---

## Training Progress

![Training and validation loss](/history/loss_curve_shakespare.png)

### Loss Trajectory Analysis

- **Initial rapid descent (Steps 0 – 1 000):** Cross-entropy loss drops sharply from ~3.60 to ~1.95 as the model quickly captures character unigram frequencies (whitespace, line breaks, vowels) and common bigram transitions.
- **Syntactic consolidation (Steps 1 000 – 2 500):** Loss drops from ~1.95 to ~1.55. The network begins assembling valid English syllables, words, punctuation, and character name headers.
- **Asymptotic convergence (Steps 2 500 – 4 000):** Training loss stabilizes around ~1.48 with validation loss near ~1.65. The ~0.17 generalization gap reflects healthy convergence on Tiny Shakespeare with minimal overfitting.

### Generation Progression Across Training

Periodic generation samples captured during training (`ShakespareOutput*.txt`, prompt `\n`, temperature = 0.8) demonstrate the linguistic evolution of the model:

#### Step 600 (`train_loss ≈ 2.26`, `test_loss ≈ 2.23`)
*Whitespace, capitalization, and syllable fragments begin to form:*
```text
IOCTTINTEH:
For cat sar thot thall the ord the tto'l fiss tit
O dir hom, bus and kly reaingl.

AUCANK:
Sly the llathin wilen theere thin thon me
```

#### Step 1 800 (`train_loss ≈ 1.63`, `test_loss ≈ 1.88`)
*Common words, pronouns, and basic clause structure appear:*
```text
With and the you not all, man'd the news times
The is taken loved and and by for shat seech.

AUTOLY:
For thee my sight of he
thus baring on that will my well me be take
Than that is our the the main; it edain mure...
```

#### Step 3 000 (`train_loss ≈ 1.50`, `test_loss ≈ 1.64`)
*Distinct Shakespearean character names, archaic inflections, and poetic line breaks:*
```text
FRIAR LAURENCE:
It soon, good when is he's bief
To the feear'st, yet to parting the one weep;
'Tis so not suppert, which you the wide foolicuticy,
Richar compon our agest.
O, what the hastisure her brow well...
```

#### Step 4 000 (`train_loss ≈ 1.49`, `test_loss ≈ 1.66`)
*Full multi-speaker theatrical turn-taking with dialogue headers and dramatic cadence:*
```text
GLOUCESTER:
And there I'll apprison him every in the id
What tain his worse helpt thief his prisonmage
To have than lates.
There not holy are to much shame! but you are!

AUTOLYCUS:
Pray?

RATCLIOF:
Why, will I beseech is a couse at pardon,
Hath will with my thought most be husband.

LUCIO:
It make you shall we may that faith to say
Does both of a grap in their main...
```

---

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
| Embedding `W_emb`    | $\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$ |
| Q, K, V              | $\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$ |
| Output proj `W_O`    | $\mathcal{N}(0, 1/\sqrt{d_v})$ |
| FFN `W_0`            | $\mathcal{N}(0, \sqrt{2/d_{\text{model}}})$ (He) |
| FFN `W_1`            | $\mathcal{N}(0, \sqrt{2/\text{ffn\_wd}})$ |
| Unembedding `W_U`    | $\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$ |
| LayerNorm $\gamma$   | Ones                            |
| LayerNorm $\beta$    | Zeros                           |

---

## License

MIT

---

## Acknowledgements

Architecture follows the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy, 2023) and the original Transformer (Vaswani et al., 2017). All gradients and the Adam update are derived and implemented from first principles.

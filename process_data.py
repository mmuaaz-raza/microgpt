"""
process_data.py
---------------
Data loading, vocabulary construction, batch sampling, and all activation
functions (softmax, layer-norm, ReLU, GELU) used throughout the model.
"""

import numpy as np

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
DATA_FILE = "input.txt"


def load_vocabulary(data: str):
    """Return (itos, stoi) character-level lookup dicts for the corpus."""
    vocab = sorted(set(data))
    itos  = {i: ch for i, ch in enumerate(vocab)}
    stoi  = {ch: i for i, ch in enumerate(vocab)}
    return itos, stoi


def load_essentials():
    """
    Read the corpus, build the vocabulary, and split 90/10 train/test.

    Returns
    -------
    train_set, test_set : np.ndarray[int64]  – encoded token arrays.
    itos, stoi          : vocabulary lookup dicts.
    encode, decode      : str <-> list[int] helpers.
    """
    with open(DATA_FILE, encoding="utf-8") as f:
        data = f.read()

    split      = int(0.9 * len(data))
    itos, stoi = load_vocabulary(data)

    encode = lambda text: [stoi[ch] for ch in text]
    decode = lambda indices: "".join(itos[i] for i in indices)

    train_set = np.array(encode(data[:split]), dtype=np.int64)
    test_set  = np.array(encode(data[split:]), dtype=np.int64)

    return train_set, test_set, itos, stoi, encode, decode


def get_target_labels(block_size: int, data: np.ndarray, batch_size: int):
    """
    Sample a random mini-batch.
    x = data[i:i+block_size],  y = data[i+1:i+block_size+1]  (next-token targets).
    """
    rng     = np.random.default_rng()
    indices = rng.integers(low=0, high=len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i     : i + block_size    ] for i in indices])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in indices])
    return x, y


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x     = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm_cal(x: np.ndarray, epsilon: float):
    """
    Normalise each row independently. Returns (x_hat, mean, var);
    mean and var are cached for the backward pass.
    """
    mean = np.mean(x, axis=1, keepdims=True)
    var  = np.var(x,  axis=1, keepdims=True)
    return (x - mean) / (var + epsilon) ** 0.5, mean, var


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def drelu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation — tanh approximation used in GPT-2."""
    inner = (2 / np.pi) ** 0.5 * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1 + np.tanh(inner))


def dgelu(x: np.ndarray) -> np.ndarray:
    """Derivative of GELU w.r.t. x (used in FFN backward pass)."""
    inner      = (2 / np.pi) ** 0.5 * (x + 0.044715 * x ** 3)
    tanh_inner = np.tanh(inner)
    sech2      = 1 / np.cosh(inner) ** 2
    d_inner    = (2 / np.pi) ** 0.5 * (1 + 3 * 0.044715 * x ** 2)
    # chain rule: d/dx [0.5*x*(1+tanh(inner))]
    return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * d_inner

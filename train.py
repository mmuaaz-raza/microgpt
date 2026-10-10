"""
train.py
--------
Training loop for the MicroGPT character-level language model using Adam with
linear warmup and inverse-time decay.
"""

import json
from model import (
    Transformer,
    block_size,
    batch_size,
    beta1,
    beta2,
    ed,
    ffn_wd,
    heads,
    layers,
    modelFilename,
    qk_d,
    v_d,
)
from process_data import get_target_labels

LOSS_HISTORY_FILE = "shakespareADAM.json"

tinygpt = Transformer(
    ed=ed,
    heads=heads,
    layers=layers,
    qk_d=qk_d,
    v_d=v_d,
    seq_len=block_size,
    ffn_wd=ffn_wd,
    ModelFileName=modelFilename,
)

loss_history = []


def train_network(
    number_iter: int,
    decay_rate: float,
    checkpoint_rate: float,
    tracking_rate: float,
    base_alpha: float,
    warmup_steps: int = 500,
) -> None:
    log_every        = max(1, int(number_iter * tracking_rate))
    checkpoint_every = max(1, int(number_iter * checkpoint_rate))

    for step in range(1, number_iter + 1):
        if step < warmup_steps:
            alpha = base_alpha * (step + 1) / warmup_steps
        else:
            alpha = base_alpha / (1.0 + decay_rate * (step - warmup_steps))

        x, y = get_target_labels(
            batch_size=batch_size,
            data=tinygpt.train_set,
            block_size=block_size,
        )
        for inp, out in zip(x, y):
            tinygpt.backward(inp, out)

        tinygpt.updateWeightsAdam(alpha, batch_size, step, beta1, beta2)

        if step == number_iter or step % checkpoint_every == 0:
            tinygpt.save(step)
            with open(LOSS_HISTORY_FILE, "w") as f:
                json.dump(loss_history, f)

        if step % log_every == 0:
            train_loss = tinygpt.calculateBatchLoss(x, y)

            tx, ty = get_target_labels(
                batch_size=batch_size,
                data=tinygpt.test_set,
                block_size=block_size,
            )
            test_loss = tinygpt.calculateBatchLoss(tx, ty)

            loss_history.append((step, train_loss, test_loss))
            print(
                f"step {step}/{number_iter}  "
                f"lr={alpha:.6f}  "
                f"train_loss={train_loss:.4f}  "
                f"test_loss={test_loss:.4f}"
            )


train_network(
    number_iter=3000,
    decay_rate=1e-5,
    checkpoint_rate=0.05,
    tracking_rate=0.01,
    base_alpha=0.001,
    warmup_steps=500,
)

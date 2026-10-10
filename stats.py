"""
stats.py
--------
Plot train and test loss curves from training history JSON and save to PNG.
"""

import os
import json
import matplotlib.pyplot as plt

DEFAULT_FILES = ["loss_history.json"]
OUTPUT_IMAGE_FILE = "loss_curve.png"


def plot_losses(filename: str = None) -> None:
    if filename is None:
        for f in DEFAULT_FILES:
            if os.path.exists(f):
                filename = f
                break

    if filename is None or not os.path.exists(filename):
        print(f"Loss history file not found. Checked: {DEFAULT_FILES}")
        return

    with open(filename, "r") as f:
        loss_history = json.load(f)

    steps        = [record[0] for record in loss_history]
    train_losses = [record[1] for record in loss_history]
    test_losses  = [record[2] for record in loss_history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, train_losses, label="Train Loss", color="blue", linewidth=1.5)
    ax.plot(steps, test_losses,  label="Test Loss",  color="orange", linewidth=1.5)

    ax.set_title("Transformer Loss over Training Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300)
    print(f"Loss curve saved to '{OUTPUT_IMAGE_FILE}' (from '{filename}').")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    plot_losses()

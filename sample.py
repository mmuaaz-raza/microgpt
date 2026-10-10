"""
sample.py
---------
Autoregressive text generation from a trained MicroGPT checkpoint.
"""

import numpy as np

from model import (
    Transformer,
    block_size,
    ed,
    ffn_wd,
    heads,
    layers,
    modelFilename,
    qk_d,
    v_d,
)

temperature  = 0.8
num_samples  = 500
start        = "\n"
output_file  = "output.txt"
save_to_file = True



def sample_token(probs: np.ndarray, temperature: float = 1.0,) -> int:
    """Sample a token from the distribution with temperature scaling."""
    if temperature == 0:
        return int(np.argmax(probs))

    probs      = np.clip(probs, 1e-10, 1.0)
    logits     = np.log(probs) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    scaled     = exp_logits / np.sum(exp_logits)

    return int(np.random.choice(len(probs), p=scaled))


def generate(tinygpt: Transformer,isSave: bool = False, isPrint: bool = False, fileName:str="output.txt") -> None:
    context     = tinygpt.encode(start)
    predictions = list(context)
    if isPrint:
        print(start, end="")

    for _ in range(num_samples):
        output = tinygpt.forward(np.array(context))
        if output is None:
            break

        next_token = sample_token(output[-1], temperature)
        context.append(next_token)

        if isPrint:
            print(tinygpt.decode([next_token]), end="", flush=True)

        if isSave:
            predictions.append(next_token)

        if len(context) > block_size:
            context = context[1:]

    if isSave:
        generated_text = tinygpt.decode(predictions)
        with open(fileName, "w", encoding="utf-8") as f:
            f.write(generated_text)
        if isPrint:
            print(f"\n\nOutput saved to '{fileName}'.")


if __name__ == "__main__":
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
    generate(tinygpt,save_to_file, False, output_file)

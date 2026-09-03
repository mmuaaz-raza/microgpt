

**1. Validate gradients before touching real data**
Implement numerical gradient checking: pick a single small weight (e.g., one entry of `Wq[0][0]`), perturb it by a tiny `ε` (like `1e-5`), rerun the forward pass, measure the loss change, and compare `(loss_plus - loss_minus)/(2ε)` against your analytical gradient for that same weight. Do this for at least one weight in every module (embedding, Q/K/V, Wp, W0/W1, gama/beta for each LayerNorm, Wun). If any mismatch, stop here — training on broken gradients wastes time and produces meaningless results.

**2. Overfit a tiny toy sequence**
Before the full dataset, train on a handful of characters repeated (e.g., just `"hello world"` looped) with a tiny model. If your backward pass is correct, loss should crash toward near-zero within a few hundred steps. This isolates whether your training loop itself (not just individual gradients) is wired correctly — order of operations, parameter updates actually applying, etc.


**4. Batch the forward+backward pass properly**
Your current code processes examples one at a time in a list comprehension (`[tinygpt.forward(x[i]) for i in range(batch_size)]`). For real training, decide: are you averaging gradients across the batch before the update step, or updating after every single example (pure SGD, no batching)? If averaging, you need to accumulate gradients across the batch loop and divide by `batch_size` before applying the update — decide this explicitly rather than letting it happen by accident.

**6. Write the actual training loop**
Structure: for each step, sample a batch → forward pass (get probs) → compute loss → backward pass (get all gradients) → apply parameter updates → log the loss. Wrap this in a loop for however many steps/epochs you want, printing loss periodically (e.g., every 50-100 steps) so you can watch it decrease.

**7. Add loss logging for later plotting**
Store `(step, loss)` pairs in a list or array as training runs — you'll need this for the loss-curve figure in your writeup. Don't just print and discard.

**8. Run a short training session first (sanity budget)**
Before committing to a long run, do maybe 200-500 steps and check: is loss actually trending down, even noisily? If it's flat or increasing, debug before spending more compute — check learning rate first (most common culprit), then re-check gradient checking results.

**9. Add basic text generation/sampling**
Implement a simple greedy or temperature-sampled generation function: feed a seed sequence, get next-token probabilities, sample, append, repeat. This gives you qualitative checkpoints ("does the output start looking Shakespeare-like") independent of the loss number.

**10. Full training run + checkpointing**
Once steps 1-9 are all verified working, run the real, longer training session. Periodically save your `self.params` (e.g., via `np.savez`) so you don't lose progress and can compare checkpoints later for your writeup.

**11. Only after this — move to ablations**
Once you have one clean, working baseline training run with a real loss curve and sensible generated text, that's your baseline for the ablation studies from before (head count, positional encoding on/off, pre-LN vs post-LN, etc.) — each ablation just repeats steps 6-10 with one thing changed.

Want to start with step 1 (the gradient-check harness) — want help designing that function for your specific `self.params` structure?
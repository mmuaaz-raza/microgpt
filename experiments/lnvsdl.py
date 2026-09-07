"""
Demo: Why pooling statistics across non-identically-distributed groups
distorts normalization, compared to normalizing each group with its own stats.

Run: python normalization_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# --- Simulate two "positions" with genuinely different distributions ---
# (stand-in for, e.g., position 1 vs position 50 in a sequence)
n = 1000
groupA = np.random.normal(loc=5, scale=1, size=n)   # e.g. position 1
groupB = np.random.normal(loc=-5, scale=1, size=n)  # e.g. position 50
pooled = np.concatenate([groupA, groupB])

pooled_mean, pooled_std = pooled.mean(), pooled.std()

# Normalize using POOLED (contaminated) stats -- mimics BatchNorm pooling
A_pooled_norm = (groupA - pooled_mean) / pooled_std
B_pooled_norm = (groupB - pooled_mean) / pooled_std

# Normalize using OWN per-group stats -- mimics LayerNorm's per-token behavior
A_own_norm = (groupA - groupA.mean()) / groupA.std()
B_own_norm = (groupB - groupB.mean()) / groupB.std()

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

bins = np.linspace(-8, 8, 60)

axes[0].hist(groupA, bins=bins, alpha=0.6, label="Group A (pos 1)", color="tab:blue")
axes[0].hist(groupB, bins=bins, alpha=0.6, label="Group B (pos 50)", color="tab:orange")
axes[0].set_title("Raw data\n(before any normalization)")
axes[0].legend()

bins2 = np.linspace(-3, 3, 60)
axes[1].hist(A_pooled_norm, bins=bins2, alpha=0.6, label="Group A", color="tab:blue")
axes[1].hist(B_pooled_norm, bins=bins2, alpha=0.6, label="Group B", color="tab:orange")
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_title("Normalized with POOLED stats\n(mimics BatchNorm across positions)")
axes[1].legend()

axes[2].hist(A_own_norm, bins=bins2, alpha=0.6, label="Group A", color="tab:blue")
axes[2].hist(B_own_norm, bins=bins2, alpha=0.6, label="Group B", color="tab:orange")
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_title("Normalized with OWN stats\n(mimics LayerNorm per-token)")
axes[2].legend()

for ax in axes:
    ax.set_xlabel("value")
    ax.set_ylabel("count")

plt.tight_layout()
plt.savefig("normalization_demo.png", dpi=150)
print("Saved plot to normalization_demo.png")

# --- Print the numeric summary too ---
print()
print(f"{'':20s} {'mean':>10s} {'std':>10s}")
print(f"{'Group A (pooled)':20s} {A_pooled_norm.mean():10.3f} {A_pooled_norm.std():10.3f}")
print(f"{'Group B (pooled)':20s} {B_pooled_norm.mean():10.3f} {B_pooled_norm.std():10.3f}")
print(f"{'Group A (own)':20s} {A_own_norm.mean():10.3f} {A_own_norm.std():10.3f}")
print(f"{'Group B (own)':20s} {B_own_norm.mean():10.3f} {B_own_norm.std():10.3f}")

plt.show()
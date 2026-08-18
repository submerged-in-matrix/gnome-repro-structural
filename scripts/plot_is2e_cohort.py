"""
Generate the hero figure comparing EMA-GNN against other IS2E direct-prediction
models on the Matbench Discovery leaderboard.

Data below is pulled directly from each model's public YAML in the
Matbench Discovery repository (models/<family>/<model>.yml, full_test_set
block) on 2026-08-13. Values are static since they only change if a model
is re-ingested; re-pull from the repo before reuse if reproducing later.

Filter applied: test_task == "IS2E" and targets == "E" (direct energy
prediction, no relaxation step at inference).

Usage:
    python plot_is2e_cohort.py
Output:
    hero.png in the current directory.
"""

import matplotlib.pyplot as plt
import numpy as np

# name -> (F1, MAE eV/atom, R2)
DATA = {
    "ESNet":       (0.568, 0.107, -0.148),
    "EMA-GNN":     (0.566, 0.084,  0.387),
    "ALIGNN":      (0.565, 0.092,  0.274),
    "MEGNet":      (0.513, 0.128, -0.277),
    "CGCNN":       (0.510, 0.135, -0.624),
    "Voronoi RF":  (0.344, 0.141, -0.316),
}

HIGHLIGHT = "EMA-GNN"
HIGHLIGHT_COLOR = "#d62728"
BASE_COLOR = "#4c72b0"

names = list(DATA.keys())
f1 = [DATA[n][0] for n in names]
mae = [DATA[n][1] for n in names]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

colors = [HIGHLIGHT_COLOR if n == HIGHLIGHT else BASE_COLOR for n in names]

# --- F1 panel (higher is better) ---
ax = axes[0]
bars = ax.bar(names, f1, color=colors)
ax.set_ylabel("Discovery F1")
ax.set_title("F1 (higher is better)")
ax.set_ylim(0, max(f1) * 1.15)
ax.tick_params(axis="x", rotation=35)
for b, v in zip(bars, f1):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
             ha="center", va="bottom", fontsize=9)

# --- MAE panel (lower is better) ---
ax = axes[1]
bars = ax.bar(names, mae, color=colors)
ax.set_ylabel("MAE (eV/atom)")
ax.set_title("MAE (lower is better)")
ax.set_ylim(0, max(mae) * 1.15)
ax.tick_params(axis="x", rotation=35)
for b, v in zip(bars, mae):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
             ha="center", va="bottom", fontsize=9)

fig.suptitle("IS2E direct-prediction models — Matbench Discovery leaderboard",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("hero.png", dpi=200, bbox_inches="tight")
print("Saved hero.png")

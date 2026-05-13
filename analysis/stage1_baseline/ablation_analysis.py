"""Stage 1 — Ablation Analysis.

Reads history.json and best.pt test_mae from all ablation runs and
the baseline, then produces:

    Plot 1 : MAE vs Epoch — all runs overlaid (convergence comparison)
    Plot 2 : Bar chart — best MAE per run + delta vs baseline
    Plot 3 : Last-50-epoch zoom — all runs overlaid
    results/stage1_baseline/03_ablation_summary_table.csv

Ablation runs expected at:
    runs/default/            baseline (seed=0)
    runs/ablation_no_norm/   B1: use_adj_norm=False
    runs/ablation_small/     B2: hidden_dim=128
    runs/ablation_shallow/   B3: n_layers=2

Public API
----------
    run(repo_root, show=False)
    main()
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]

# ---------------------------------------------------------------------------
# Run registry — label, run directory, plot style
# ---------------------------------------------------------------------------
RUNS = [
    {
        "label"  : "Baseline (hidden=256, layers=3, adj_norm=True)",
        "run_dir": "runs/default",
        "color"  : "#4C72B0",
        "lw"     : 2.5,
        "ls"     : "-",
    },
    {
        "label"  : "B1: no adj norm",
        "run_dir": "runs/ablation_no_norm",
        "color"  : "#DD8452",
        "lw"     : 1.8,
        "ls"     : "--",
    },
    {
        "label"  : "B2: hidden_dim=128",
        "run_dir": "runs/ablation_small",
        "color"  : "#55A868",
        "lw"     : 1.8,
        "ls"     : "-.",
    },
    {
        "label"  : "B3: n_layers=2",
        "run_dir": "runs/ablation_shallow",
        "color"  : "#C44E52",
        "lw"     : 1.8,
        "ls"     : ":",
    },
]

GREY = "#8C8C8C"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 120,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _load_run(root: Path, run_dir: str) -> dict | None:
    """Load history.json for one run. Returns None if not found."""
    hist_path = root / run_dir / "history.json"
    if not hist_path.exists():
        print(f"  Warning: history.json not found at {hist_path} — skipping.")
        return None
    with open(hist_path) as f:
        history = json.load(f)
    df       = pd.DataFrame(history)
    epochs   = df["epoch"].values
    mae_mev  = df["test_mae_eV_per_atom"].values * 1000
    best_idx = int(np.argmin(mae_mev))
    return {
        "epochs"    : epochs,
        "mae_mev"   : mae_mev,
        "mae_smooth": _smooth(mae_mev),
        "best_mae"  : float(mae_mev[best_idx]),
        "best_epoch": int(epochs[best_idx]),
        "final_mae" : float(mae_mev[-1]),
    }


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run(repo_root: Path | None = None, show: bool = False) -> dict:
    """Run ablation analysis. Returns summary dict keyed by run label."""

    root    = Path(repo_root) if repo_root else _REPO_ROOT
    out_dir = root / "results" / "stage1_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    loaded = []
    for cfg in RUNS:
        data = _load_run(root, cfg["run_dir"])
        if data is not None:
            loaded.append({**cfg, **data})

    if len(loaded) < 2:
        print("ERROR: fewer than 2 runs loaded — check run directory names.")
        return {}

    baseline_mae = loaded[0]["best_mae"]   # baseline is always first

    # -------------------------------------------------------------------
    # Plot 1 — MAE vs Epoch, all runs overlaid (smoothed)
    # -------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for r in loaded:
        ax1.plot(r["epochs"], r["mae_smooth"],
                 color=r["color"], lw=r["lw"], ls=r["ls"],
                 label=f"{r['label']}  (best {r['best_mae']:.1f} meV)")
        ax1.scatter([r["best_epoch"]], [r["best_mae"]],
                    color=r["color"], zorder=5, s=40)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test MAE  (meV/atom)")
    ax1.set_title("Stage 1 — Ablation Comparison: MAE vs Epoch")
    ax1.legend(fontsize=8, loc="upper right")
    fig1.tight_layout()
    p1 = out_dir / "03_ablation_mae_curves.png"
    fig1.savefig(p1, dpi=150)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------
    # Plot 2 — Best MAE bar chart + delta vs baseline
    # -------------------------------------------------------------------
    labels    = [r["label"].split(":")[0].strip() for r in loaded]
    best_maes = [r["best_mae"] for r in loaded]
    deltas    = [m - baseline_mae for m in best_maes]
    colors    = [r["color"] for r in loaded]

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 4))

    # Absolute MAE
    bars = ax2a.bar(labels, best_maes, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, best_maes):
        ax2a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.1,
                  f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax2a.set_ylabel("Best MAE  (meV/atom)")
    ax2a.set_title("Best MAE per run")
    ax2a.tick_params(axis="x", rotation=15)

    # Delta vs baseline
    delta_colors = [GREY if d == 0 else ("#C44E52" if d > 0 else "#55A868")
                    for d in deltas]
    bars2 = ax2b.bar(labels, deltas, color=delta_colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars2, deltas):
        ax2b.text(bar.get_x() + bar.get_width() / 2,
                  val + (0.02 if val >= 0 else -0.08),
                  f"{val:+.2f}", ha="center", va="bottom", fontsize=9)
    ax2b.axhline(0, color=GREY, lw=1)
    ax2b.set_ylabel("Delta MAE vs baseline  (meV/atom)")
    ax2b.set_title("Delta vs Baseline")
    ax2b.tick_params(axis="x", rotation=15)

    fig2.tight_layout()
    p2 = out_dir / "03_ablation_bar_comparison.png"
    fig2.savefig(p2, dpi=150)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------
    # Plot 3 — Last-50-epoch zoom, all runs overlaid
    # -------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for r in loaded:
        zoom_n  = min(50, len(r["epochs"]))
        z_ep    = r["epochs"][-zoom_n:]
        z_mae   = r["mae_smooth"][-zoom_n:]
        ax3.plot(z_ep, z_mae,
                 color=r["color"], lw=r["lw"], ls=r["ls"],
                 label=f"{r['label']}  (final {r['final_mae']:.1f} meV)")

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Test MAE  (meV/atom)")
    ax3.set_title("Stage 1 — Ablation Comparison: Last 50 Epochs")
    ax3.legend(fontsize=8, loc="upper right")
    fig3.tight_layout()
    p3 = out_dir / "03_ablation_last50_zoom.png"
    fig3.savefig(p3, dpi=150)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    summary = {}
    for r in loaded:
        summary[r["label"]] = {
            "best_mae_meV"  : round(r["best_mae"],  2),
            "best_epoch"    : r["best_epoch"],
            "final_mae_meV" : round(r["final_mae"], 2),
            "delta_vs_baseline_meV": round(r["best_mae"] - baseline_mae, 2),
        }

    csv_path = out_dir / "03_ablation_summary_table.csv"
    pd.DataFrame(summary).T.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # Console summary
    print(f"\n{'='*60}")
    print(f"  Stage 1 — Ablation Summary")
    print(f"  {'Run':<40} {'Best MAE':>9}  {'Delta':>7}  {'Best Ep':>8}")
    print(f"  {'-'*40} {'-'*9}  {'-'*7}  {'-'*8}")
    for r in loaded:
        delta = r["best_mae"] - baseline_mae
        print(f"  {r['label']:<40} "
              f"{r['best_mae']:>8.2f}  "
              f"{delta:>+7.2f}  "
              f"{r['best_epoch']:>8}")
    print(f"{'='*60}\n")

    if show:
        plt.show()
    else:
        plt.close("all")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(repo_root=args.repo_root, show=args.show)


if __name__ == "__main__":
    main()
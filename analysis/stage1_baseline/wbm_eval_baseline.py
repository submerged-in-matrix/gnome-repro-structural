"""Stage 1 Baseline — WBM Evaluation Summary.

Reads:
    runs/default/metrics_wbm.json    MAE / RMSE / bias
    runs/default/f1_wbm.json         F1 / precision / recall

Produces:
    Plot 1 : bar chart — MAE, RMSE, bias (meV/atom)
    Plot 2 : confusion matrix heatmap (raw strategy)
    Plot 3 : precision / recall / F1 grouped bar (raw vs bias-corrected)
    results/stage1_baseline/02_wbm_summary_table.csv

Public API
----------
    run(repo_root, show=False)
    main()
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
GREY   = "#8C8C8C"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 120,
})


def run(repo_root: Path | None = None, show: bool = False) -> dict:
    """Run WBM evaluation summary. Returns combined metrics dict."""

    root    = Path(repo_root) if repo_root else _REPO_ROOT
    run_dir = root / "runs" / "default"
    out_dir = root / "results" / "stage1_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load metrics ---
    with open(run_dir / "metrics_wbm.json") as f:
        reg = json.load(f)
    with open(run_dir / "f1_wbm.json") as f:
        cls = json.load(f)

    mae_mev  = reg["mae_meV_per_atom"]
    rmse_mev = reg["rmse_meV_per_atom"]
    bias_mev = reg["bias_meV_per_atom"]

    raw  = cls["strategy_raw"]
    corr = cls.get("strategy_bias_corrected", None)

    # -------------------------------------------------------------------
    # Plot 1 — Regression metrics bar chart
    # -------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    labels = ["MAE", "RMSE", "Bias"]
    values = [mae_mev, rmse_mev, bias_mev]
    colors = [BLUE, ORANGE, RED]
    bars   = ax1.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("meV/atom")
    ax1.set_title("Stage 1 Baseline — WBM Regression Metrics\n(seed=0, 20-pt TTA)")
    ax1.axhline(21, color=GREY, ls="--", lw=1,
                label="GNoME pre-AL MP test MAE on relaxed MP structures, not unrelaxed wbm structures")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    p1 = out_dir / "02_wbm_regression_metrics.png"
    fig1.savefig(p1, dpi=150)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------
    # Plot 2 — Confusion matrix heatmap (raw strategy)
    # -------------------------------------------------------------------
    cm = np.array([[raw["tp"], raw["fn"]],
                   [raw["fp"], raw["tn"]]])
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Pred Stable", "Pred Unstable"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["True Stable", "True Unstable"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                     fontsize=11,
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax2.set_title("Stage 1 Baseline — Confusion Matrix\n(raw predictions)")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    p2 = out_dir / "02_wbm_confusion_matrix.png"
    fig2.savefig(p2, dpi=150)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------
    # Plot 3 — Precision / Recall / F1 grouped bar
    # -------------------------------------------------------------------
    metrics_names = ["Precision", "Recall", "F1"]
    raw_vals  = [raw["precision"],  raw["recall"],  raw["f1"]]

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    x     = np.arange(len(metrics_names))
    width = 0.3

    ax3.bar(x - width / 2, raw_vals, width,
            color=BLUE, label="Raw predictions", edgecolor="white")

    if corr is not None:
        corr_vals = [corr["precision"], corr["recall"], corr["f1"]]
        ax3.bar(x + width / 2, corr_vals, width,
                color=ORANGE, label="Bias-corrected", edgecolor="white")

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Score")
    ax3.set_title("Stage 1 Baseline — Stability Classification\n(seed=0, 20-pt TTA)")
    ax3.axhline(0.5, color=GREY, ls="--", lw=1, alpha=0.6,
                label="F1=0.5 reference")
    ax3.legend(fontsize=9)

    # Annotate values
    for rect in ax3.patches:
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width() / 2, h + 0.01,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    fig3.tight_layout()
    p3 = out_dir / "02_wbm_classification_metrics.png"
    fig3.savefig(p3, dpi=150)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    summary = {
        "n_structures"       : reg["n_structures"],
        "mae_meV"            : mae_mev,
        "rmse_meV"           : rmse_mev,
        "bias_meV"           : bias_mev,
        "f1_raw"             : raw["f1"],
        "precision_raw"      : raw["precision"],
        "recall_raw"         : raw["recall"],
        "f1_bias_corrected"  : corr["f1"]       if corr else None,
        "prec_bias_corrected": corr["precision"] if corr else None,
        "rec_bias_corrected" : corr["recall"]    if corr else None,
        "n_true_stable"      : cls["n_true_stable"],
        "tp_raw"             : raw["tp"],
        "fp_raw"             : raw["fp"],
        "fn_raw"             : raw["fn"],
        "tn_raw"             : raw["tn"],
    }
    csv_path = out_dir / "02_wbm_summary_table.csv"
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # Console summary
    print(f"\n{'='*52}")
    print(f"  Stage 1 Baseline — WBM Evaluation Summary")
    print(f"{'='*52}")
    print(f"  N structures  : {reg['n_structures']:,}")
    print(f"  MAE           : {mae_mev:.2f} meV/atom")
    print(f"  RMSE          : {rmse_mev:.2f} meV/atom")
    print(f"  Bias          : {bias_mev:.2f} meV/atom")
    print(f"  F1  (raw)     : {raw['f1']:.4f}")
    print(f"  Prec(raw)     : {raw['precision']:.4f}")
    print(f"  Rec (raw)     : {raw['recall']:.4f}")
    if corr:
        print(f"  F1  (corr)    : {corr['f1']:.4f}")
    print(f"{'='*52}\n")

    if show:
        plt.show()
    else:
        plt.close("all")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(repo_root=args.repo_root, show=args.show)


if __name__ == "__main__":
    main()

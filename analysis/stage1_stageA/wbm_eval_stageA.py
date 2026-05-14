"""Stage A — WBM Evaluation Summary.

Reads:
    runs/stage_a/metrics_wbm.json    MAE / RMSE / bias
    runs/stage_a/f1_wbm.json         F1 / precision / recall

Produces:
    Plot 1 : bar chart — MAE, RMSE, bias (meV/atom) — Stage A vs Baseline
    Plot 2 : confusion matrix heatmap (raw strategy)
    Plot 3 : precision / recall / F1 grouped bar (raw vs bias-corrected)
    results/stage1_stageA/02_wbm_summary_table.csv

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

# Baseline numbers for comparison (seed=0, 200ep)
_BASELINE = {
    "mae_meV"  : 168.8,
    "rmse_meV" : 253.9,
    "bias_meV" : 122.5,
    "f1_raw"   : 0.365,
    "precision": 0.225,
    "recall"   : 0.964,
}


def run(repo_root: Path | None = None, show: bool = False) -> dict:
    """Run Stage A WBM evaluation summary. Returns combined metrics dict."""

    root    = Path(repo_root) if repo_root else _REPO_ROOT
    run_dir = root / "runs" / "stage_a"
    out_dir = root / "results" / "stage1_stageA"
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
    # Plot 1 — Regression metrics: Stage A vs Baseline
    # -------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    labels     = ["MAE", "RMSE", "Bias"]
    stageA_vals   = [mae_mev, rmse_mev, bias_mev]
    baseline_vals = [_BASELINE["mae_meV"], _BASELINE["rmse_meV"], _BASELINE["bias_meV"]]

    x     = np.arange(len(labels))
    width = 0.3

    bars_b = ax1.bar(x - width / 2, baseline_vals, width,
                     color=GREY,   label="Baseline (200ep)", edgecolor="white")
    bars_a = ax1.bar(x + width / 2, stageA_vals,   width,
                     color=BLUE,   label="Stage A (500ep+EMA)", edgecolor="white")

    for bar, val in zip(list(bars_b) + list(bars_a),
                        baseline_vals + stageA_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("meV/atom")
    ax1.set_title("Stage A vs Baseline — WBM Regression Metrics\n(20-pt TTA)")
    ax1.legend(fontsize=9)
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
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Pred Stable", "Pred Unstable"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["True Stable", "True Unstable"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                     fontsize=11,
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax2.set_title("Stage A — Confusion Matrix\n(raw predictions)")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    p2 = out_dir / "02_wbm_confusion_matrix.png"
    fig2.savefig(p2, dpi=150)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------
    # Plot 3 — Precision / Recall / F1: Stage A vs Baseline
    # -------------------------------------------------------------------
    metrics_names = ["Precision", "Recall", "F1"]
    stageA_cls    = [raw["precision"], raw["recall"], raw["f1"]]
    baseline_cls  = [_BASELINE["precision"], _BASELINE["recall"], _BASELINE["f1_raw"]]

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    x     = np.arange(len(metrics_names))
    width = 0.25

    ax3.bar(x - width, baseline_cls, width,
            color=GREY,   label="Baseline (200ep)", edgecolor="white")
    ax3.bar(x,          stageA_cls,  width,
            color=BLUE,   label="Stage A raw (500ep+EMA)", edgecolor="white")

    if corr is not None:
        corr_vals = [corr["precision"], corr["recall"], corr["f1"]]
        ax3.bar(x + width, corr_vals, width,
                color=ORANGE, label="Stage A bias-corrected", edgecolor="white")

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Score")
    ax3.set_title("Stage A vs Baseline — Stability Classification\n(20-pt TTA)")
    ax3.axhline(0.5, color=GREY, ls="--", lw=1, alpha=0.6, label="F1=0.5 reference")
    ax3.legend(fontsize=8)

    for rect in ax3.patches:
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width() / 2, h + 0.01,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7)

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
        # Deltas vs baseline
        "delta_mae_meV"      : round(mae_mev  - _BASELINE["mae_meV"],  2),
        "delta_bias_meV"     : round(bias_mev - _BASELINE["bias_meV"], 2),
        "delta_f1_raw"       : round(raw["f1"] - _BASELINE["f1_raw"],  4),
    }
    csv_path = out_dir / "02_wbm_summary_table.csv"
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # Console summary
    print(f"\n{'='*52}")
    print(f"  Stage A — WBM Evaluation Summary")
    print(f"{'='*52}")
    print(f"  N structures  : {reg['n_structures']:,}")
    print(f"  MAE           : {mae_mev:.2f} meV/atom  "
          f"(Δ {mae_mev - _BASELINE['mae_meV']:+.1f} vs baseline)")
    print(f"  RMSE          : {rmse_mev:.2f} meV/atom")
    print(f"  Bias          : {bias_mev:.2f} meV/atom  "
          f"(Δ {bias_mev - _BASELINE['bias_meV']:+.1f} vs baseline)")
    print(f"  F1  (raw)     : {raw['f1']:.4f}  "
          f"(Δ {raw['f1'] - _BASELINE['f1_raw']:+.4f} vs baseline)")
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
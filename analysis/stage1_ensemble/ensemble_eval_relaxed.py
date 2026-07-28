"""Stage C — Ensemble WBM Evaluation Summary.

Parametrized to support both the original unrelaxed ensemble and the
MLIP-relaxed (mace/chgnet) ensembles, with two baselines:
  Baseline A: single-model min-TTA (unrelaxed)
  Baseline B: 6-seed ensemble, unrelaxed (isolates the relaxation effect)
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
TEAL   = "#4EAAA1"
PURPLE = "#8172B2"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 120,
})

_BASELINE_MIN_TTA = {
    "label"    : "Single model (EMA-0.999, min-TTA)",
    "mae_meV"  : 109.4013,
    "rmse_meV" : 169.9810,
    "bias_meV" : -47.6474,
    "f1_raw"   : 0.3432,
    "precision": 0.2074,
    "recall"   : 0.993,
    "color"    : GREY,
}


def _load_ensemble_baseline(root: Path) -> dict:
    """Baseline B: read the existing unrelaxed 6-seed ensemble result
    directly from disk (runs/ensemble/), not from hardcoded numbers.
    """
    run_dir = root / "runs" / "ensemble"
    with open(run_dir / "metrics_wbm_ensemble.json") as f:
        reg_all = json.load(f)
    with open(run_dir / "f1_wbm.json") as f:
        cls = json.load(f)
    reg  = reg_all["ensemble"]
    raw  = cls["strategy_raw"]
    corr = cls.get("strategy_bias_corrected", None)
    return {
        "label"    : "Ensemble (6 seeds, unrelaxed)",
        "mae_meV"  : reg["mae_meV_per_atom"],
        "rmse_meV" : reg["rmse_meV_per_atom"],
        "bias_meV" : reg["bias_meV_per_atom"],
        "f1_raw"   : raw["f1"],
        "precision": raw["precision"],
        "recall"   : raw["recall"],
        "f1_corr"  : corr["f1"]        if corr else None,
        "prec_corr": corr["precision"] if corr else None,
        "rec_corr" : corr["recall"]    if corr else None,
        "color"    : PURPLE,
    }


def run(
    repo_root:    Path | None = None,
    show:         bool        = False,
    run_dir_name: str         = "ensemble",
    out_subdir:   str         = "stage1_ensemble",
    tag:          str | None  = None,
) -> dict:
    root    = Path(repo_root) if repo_root else _REPO_ROOT
    run_dir = root / "runs" / run_dir_name
    out_dir = root / "results" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics_wbm_ensemble.json") as f:
        reg_all = json.load(f)
    with open(run_dir / "f1_wbm.json") as f:
        cls = json.load(f)

    reg   = reg_all["ensemble"]
    seeds = reg_all["seeds"]

    mae_mev  = reg["mae_meV_per_atom"]
    rmse_mev = reg["rmse_meV_per_atom"]
    bias_mev = reg["bias_meV_per_atom"]

    raw  = cls["strategy_raw"]
    corr = cls.get("strategy_bias_corrected", None)

    baseline_a = _BASELINE_MIN_TTA
    baseline_b = _load_ensemble_baseline(root)
    tag        = tag or f"Ensemble ({run_dir_name}, median, min-TTA)"

    # -------------------------------------------------------------------
    # Plot 1 — Regression metrics: baseline A vs baseline B vs new result
    # -------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    labels   = ["MAE", "RMSE", "Bias"]
    a_vals   = [baseline_a["mae_meV"], baseline_a["rmse_meV"], baseline_a["bias_meV"]]
    b_vals   = [baseline_b["mae_meV"], baseline_b["rmse_meV"], baseline_b["bias_meV"]]
    ens_vals = [mae_mev, rmse_mev, bias_mev]

    x     = np.arange(len(labels))
    width = 0.25

    bars_a = ax1.bar(x - width, a_vals, width, color=baseline_a["color"], label=baseline_a["label"], edgecolor="white")
    bars_b = ax1.bar(x, b_vals, width, color=baseline_b["color"], label=baseline_b["label"], edgecolor="white")
    bars_e = ax1.bar(x + width, ens_vals, width, color=ORANGE, label=tag, edgecolor="white")

    for bar, val in zip(list(bars_a) + list(bars_b) + list(bars_e), a_vals + b_vals + ens_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (2 if val >= 0 else -6),
                 f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)

    ax1.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("meV/atom")
    ax1.set_title(f"{tag} — WBM Regression Metrics\n(20-pt min-TTA)")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    p1 = out_dir / "03_ensemble_regression_metrics.png"
    fig1.savefig(p1, dpi=150)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------
    # Plot 2 — Confusion matrix (new result only)
    # -------------------------------------------------------------------
    cm = np.array([[raw["tp"], raw["fn"]],
                   [raw["fp"], raw["tn"]]])
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Pred Stable", "Pred Unstable"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["True Stable", "True Unstable"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=11,
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax2.set_title(f"{tag} — Confusion Matrix\n(raw predictions)")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    p2 = out_dir / "03_ensemble_confusion_matrix.png"
    fig2.savefig(p2, dpi=150)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------
    # Plot 3 — Precision/Recall/F1: baseline A, baseline B, new raw, new corrected
    # -------------------------------------------------------------------
    metrics_names = ["Precision", "Recall", "F1"]
    ens_cls_raw   = [raw["precision"], raw["recall"], raw["f1"]]

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    x     = np.arange(len(metrics_names))
    width = 0.2

    ax3.bar(x - 1.5*width, [baseline_a["precision"], baseline_a["recall"], baseline_a["f1_raw"]],
            width, color=baseline_a["color"], label=baseline_a["label"], edgecolor="white")
    ax3.bar(x - 0.5*width, [baseline_b["precision"], baseline_b["recall"], baseline_b["f1_raw"]],
            width, color=baseline_b["color"], label=f"{baseline_b['label']} (raw)", edgecolor="white")
    ax3.bar(x + 0.5*width, ens_cls_raw, width, color=ORANGE, label=f"{tag} (raw)", edgecolor="white")

    if corr is not None:
        corr_vals = [corr["precision"], corr["recall"], corr["f1"]]
        ax3.bar(x + 1.5*width, corr_vals, width, color=TEAL, label=f"{tag} (bias-corrected)", edgecolor="white")
        if baseline_b.get("f1_corr") is not None:
            ax3.axhline(baseline_b["f1_corr"], color=baseline_b["color"], ls=":", lw=1.5,
                        label=f"{baseline_b['label']} F1 (corrected) = {baseline_b['f1_corr']:.3f}")

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Score")
    ax3.set_title(f"{tag} — Stability Classification")
    ax3.axhline(0.5, color=GREY, ls="--", lw=1, alpha=0.4)
    ax3.legend(fontsize=7)

    for rect in ax3.patches:
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width() / 2, h + 0.01,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=6)

    fig3.tight_layout()
    p3 = out_dir / "03_ensemble_classification_metrics.png"
    fig3.savefig(p3, dpi=150)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------
    # Plot 4 — Per-seed MAE vs this run's own ensemble (unchanged)
    # -------------------------------------------------------------------
    seed_names = list(seeds.keys())
    seed_maes  = [seeds[s]["mae_meV_per_atom"] for s in seed_names]

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    x4 = np.arange(len(seed_names))
    ax4.bar(x4, seed_maes, color=GREY, edgecolor="white", label="Individual seeds")
    ax4.axhline(mae_mev, color=ORANGE, lw=2, label=f"Ensemble (median) = {mae_mev:.2f} meV")
    ax4.set_xticks(x4)
    ax4.set_xticklabels([s.replace("seed_", "seed ") for s in seed_names])
    ax4.set_ylabel("MAE (meV/atom)")
    ax4.set_title(f"{tag} — Per-Seed MAE vs Ensemble")
    ax4.legend(fontsize=8)
    fig4.tight_layout()
    p4 = out_dir / "03_ensemble_per_seed_mae.png"
    fig4.savefig(p4, dpi=150)
    print(f"Saved: {p4}")

    # -------------------------------------------------------------------
    # Summary dict / CSV — deltas vs BOTH baselines
    # -------------------------------------------------------------------
    summary = {
        "run_name"                  : run_dir_name,
        "aggregation"                : reg_all.get("tta_aggregation"),
        "n_ensemble_members"         : reg_all.get("n_ensemble_members"),
        "n_structures"               : reg["n_structures"],
        "mae_meV"                    : mae_mev,
        "rmse_meV"                   : rmse_mev,
        "bias_meV"                   : bias_mev,
        "f1_raw"                     : raw["f1"],
        "precision_raw"              : raw["precision"],
        "recall_raw"                 : raw["recall"],
        "f1_bias_corrected"          : corr["f1"]        if corr else None,
        "prec_bias_corrected"        : corr["precision"] if corr else None,
        "rec_bias_corrected"         : corr["recall"]    if corr else None,
        "n_true_stable"              : cls["n_true_stable"],
        "tp_raw": raw["tp"], "fp_raw": raw["fp"], "fn_raw": raw["fn"], "tn_raw": raw["tn"],
        "delta_mae_meV_vs_single"    : round(mae_mev  - baseline_a["mae_meV"],  2),
        "delta_bias_meV_vs_single"   : round(bias_mev - baseline_a["bias_meV"], 2),
        "delta_f1raw_vs_single"      : round(raw["f1"] - baseline_a["f1_raw"], 4),
        "delta_mae_meV_vs_ens_unrel" : round(mae_mev  - baseline_b["mae_meV"],  2),
        "delta_bias_meV_vs_ens_unrel": round(bias_mev - baseline_b["bias_meV"], 2),
        "delta_f1raw_vs_ens_unrel"   : round(raw["f1"] - baseline_b["f1_raw"], 4),
        "delta_f1corr_vs_ens_unrel"  : (round(corr["f1"] - baseline_b["f1_corr"], 4)
                                         if corr and baseline_b.get("f1_corr") else None),
    }
    csv_path = out_dir / "03_ensemble_summary_table.csv"
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(csv_path)
    print(f"Saved: {csv_path}")

    print(f"\n{'='*58}\n  {tag} — WBM Evaluation Summary\n{'='*58}")
    print(f"  N structures  : {reg['n_structures']:,}")
    print(f"  MAE           : {mae_mev:.2f}  (Δ {mae_mev-baseline_a['mae_meV']:+.2f} vs single, "
          f"Δ {mae_mev-baseline_b['mae_meV']:+.2f} vs unrelaxed ensemble)")
    print(f"  Bias          : {bias_mev:.2f}  (Δ {bias_mev-baseline_a['bias_meV']:+.2f} vs single, "
          f"Δ {bias_mev-baseline_b['bias_meV']:+.2f} vs unrelaxed ensemble)")
    print(f"  F1 (raw)      : {raw['f1']:.4f}  (Δ {raw['f1']-baseline_a['f1_raw']:+.4f} vs single, "
          f"Δ {raw['f1']-baseline_b['f1_raw']:+.4f} vs unrelaxed ensemble)")
    if corr and baseline_b.get("f1_corr"):
        print(f"  F1 (corrected): {corr['f1']:.4f}  (Δ {corr['f1']-baseline_b['f1_corr']:+.4f} vs unrelaxed ensemble corrected)")
    print(f"{'='*58}\n")

    if show:
        plt.show()
    else:
        plt.close("all")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble WBM evaluation summary")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--run-dir-name", type=str, default="ensemble")
    parser.add_argument("--out-subdir", type=str, default="stage1_ensemble")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(repo_root=args.repo_root, run_dir_name=args.run_dir_name, out_subdir=args.out_subdir, show=args.show)


if __name__ == "__main__":
    main()
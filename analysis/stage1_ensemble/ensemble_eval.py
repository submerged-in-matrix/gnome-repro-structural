"""Stage C — Ensemble WBM Evaluation Summary.

Reads:
    runs/ensemble/metrics_wbm_ensemble.json   (per-seed + ensemble MAE/RMSE/bias)
    runs/ensemble/f1_wbm.json                 (ensemble classification: raw + bias-corrected)

Baseline for comparison: single-model Stage A EMA-0.999 min-TTA
(protocol-matched — same min-TTA aggregation, isolates the pure
ensemble effect of 6 models vs 1, rather than mixing in a TTA-strategy
change as well).

Produces:
    results/stage1_ensemble/03_ensemble_regression_metrics.png
    results/stage1_ensemble/03_ensemble_confusion_matrix.png
    results/stage1_ensemble/03_ensemble_classification_metrics.png
    results/stage1_ensemble/03_ensemble_per_seed_mae.png
    results/stage1_ensemble/03_ensemble_summary_table.csv

Public API
----------
    run(repo_root, show)
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

# -------------------------------------------------------------------
# Reference baseline — single-model Stage A EMA-0.999, min-TTA
# (protocol-matched: same min-TTA aggregation as the ensemble)
# Regression: runs/stage_a/min_tta/metrics_wbm.json
# Classification: runs/stage_a/min_tta/f1_wbm.json
# -------------------------------------------------------------------
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


def run(
    repo_root: Path | None = None,
    show:      bool        = False,
) -> dict:
    """Run WBM evaluation summary for the 6-seed ensemble (median aggregation).

    Baseline comparison is the single-model min-TTA result (option A),
    not the mean-TTA Stage A baseline, since both use min-TTA and the
    only variable that changes is ensemble size (1 -> 6).
    """
    root    = Path(repo_root) if repo_root else _REPO_ROOT
    run_dir = root / "runs" / "ensemble"
    out_dir = root / "results" / "stage1_ensemble"
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

    baseline = _BASELINE_MIN_TTA
    tag      = "Ensemble (6 seeds, median, min-TTA)"

    # -------------------------------------------------------------------
    # Plot 1 — Regression metrics vs baseline
    # -------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    labels        = ["MAE", "RMSE", "Bias"]
    ens_vals      = [mae_mev, rmse_mev, bias_mev]
    baseline_vals = [baseline["mae_meV"], baseline["rmse_meV"], baseline["bias_meV"]]

    x     = np.arange(len(labels))
    width = 0.3

    bars_b = ax1.bar(x - width / 2, baseline_vals, width,
                     color=baseline["color"], label=baseline["label"], edgecolor="white")
    bars_e = ax1.bar(x + width / 2, ens_vals, width,
                     color=ORANGE, label=tag, edgecolor="white")

    for bar, val in zip(list(bars_b) + list(bars_e), baseline_vals + ens_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (2 if val >= 0 else -6),
                 f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    ax1.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("meV/atom")
    ax1.set_title(f"{tag} vs {baseline['label']} — WBM Regression Metrics\n(20-pt min-TTA)")
    ax1.legend(fontsize=9)
    fig1.tight_layout()
    p1 = out_dir / "03_ensemble_regression_metrics.png"
    fig1.savefig(p1, dpi=150)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------
    # Plot 2 — Confusion matrix
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
    ax2.set_title(f"{tag} — Confusion Matrix\n(raw predictions)")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    p2 = out_dir / "03_ensemble_confusion_matrix.png"
    fig2.savefig(p2, dpi=150)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------
    # Plot 3 — Precision / Recall / F1 (baseline vs ensemble raw vs
    # ensemble bias-corrected). Baseline precision/recall may be
    # unavailable (None) — skipped from the bar group if so.
    # -------------------------------------------------------------------
    metrics_names = ["Precision", "Recall", "F1"]
    ens_cls       = [raw["precision"], raw["recall"], raw["f1"]]

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    x     = np.arange(len(metrics_names))
    width = 0.25

    has_baseline_prf = baseline["precision"] is not None and baseline["recall"] is not None
    if has_baseline_prf:
        baseline_cls = [baseline["precision"], baseline["recall"], baseline["f1_raw"]]
        ax3.bar(x - width, baseline_cls, width,
                color=baseline["color"], label=baseline["label"], edgecolor="white")
    else:
        # Only F1 confirmed for baseline — show as a single reference bar/line.
        ax3.axhline(baseline["f1_raw"], color=baseline["color"], ls=":", lw=1.5,
                    label=f"{baseline['label']} F1 = {baseline['f1_raw']:.3f}")

    ax3.bar(x, ens_cls, width, color=ORANGE, label=f"{tag} raw", edgecolor="white")

    if corr is not None:
        corr_vals = [corr["precision"], corr["recall"], corr["f1"]]
        ax3.bar(x + width, corr_vals, width,
                color=TEAL, label=f"{tag} bias-corrected", edgecolor="white")

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Score")
    ax3.set_title(f"{tag} — Stability Classification\n(vs {baseline['label']})")
    ax3.axhline(0.5, color=GREY, ls="--", lw=1, alpha=0.4)
    ax3.legend(fontsize=8)

    for rect in ax3.patches:
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width() / 2, h + 0.01,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    fig3.tight_layout()
    p3 = out_dir / "03_ensemble_classification_metrics.png"
    fig3.savefig(p3, dpi=150)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------
    # Plot 4 — Per-seed MAE vs ensemble (shows ensemble beats every seed)
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
    ax4.set_title("Per-Seed MAE vs Ensemble (median aggregation)")
    ax4.legend(fontsize=8)
    fig4.tight_layout()
    p4 = out_dir / "03_ensemble_per_seed_mae.png"
    fig4.savefig(p4, dpi=150)
    print(f"Saved: {p4}")

    # -------------------------------------------------------------------
    # Summary dict / CSV
    # -------------------------------------------------------------------
    summary = {
        "run_name"           : "ensemble",
        "aggregation"        : reg_all.get("tta_aggregation"),
        "n_ensemble_members" : reg_all.get("n_ensemble_members"),
        "n_structures"       : reg["n_structures"],
        "mae_meV"            : mae_mev,
        "rmse_meV"           : rmse_mev,
        "bias_meV"           : bias_mev,
        "f1_raw"             : raw["f1"],
        "precision_raw"      : raw["precision"],
        "recall_raw"         : raw["recall"],
        "f1_bias_corrected"  : corr["f1"]        if corr else None,
        "prec_bias_corrected": corr["precision"] if corr else None,
        "rec_bias_corrected" : corr["recall"]    if corr else None,
        "n_true_stable"      : cls["n_true_stable"],
        "tp_raw"             : raw["tp"],
        "fp_raw"             : raw["fp"],
        "fn_raw"             : raw["fn"],
        "tn_raw"             : raw["tn"],
        "delta_mae_meV"      : round(mae_mev  - baseline["mae_meV"],  2),
        "delta_bias_meV"     : round(bias_mev - baseline["bias_meV"], 2),
        "delta_f1_raw"       : round(raw["f1"] - baseline["f1_raw"], 4),
    }
    csv_path = out_dir / "03_ensemble_summary_table.csv"
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(csv_path)
    print(f"Saved: {csv_path}")

    print(f"\n{'='*58}")
    print(f"  {tag} — WBM Evaluation Summary")
    print(f"{'='*58}")
    print(f"  N structures  : {reg['n_structures']:,}")
    print(f"  Aggregation   : {reg_all.get('tta_aggregation')}")
    print(f"  MAE           : {mae_mev:.2f} meV/atom  "
          f"(Δ {mae_mev - baseline['mae_meV']:+.2f} vs {baseline['label']})")
    print(f"  RMSE          : {rmse_mev:.2f} meV/atom  "
          f"(Δ {rmse_mev - baseline['rmse_meV']:+.2f} vs {baseline['label']})")
    print(f"  Bias          : {bias_mev:.2f} meV/atom  "
          f"(Δ {bias_mev - baseline['bias_meV']:+.2f} vs {baseline['label']})")
    print(f"  F1  (raw)     : {raw['f1']:.4f}  "
          f"(Δ {raw['f1'] - baseline['f1_raw']:+.4f} vs {baseline['label']})")
    print(f"  Prec(raw)     : {raw['precision']:.4f}")
    print(f"  Rec (raw)     : {raw['recall']:.4f}")
    if corr:
        print(f"  F1  (corr)    : {corr['f1']:.4f}")
    print(f"{'='*58}\n")

    if show:
        plt.show()
    else:
        plt.close("all")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ensemble WBM evaluation summary (median aggregation)")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(repo_root=args.repo_root, show=args.show)


if __name__ == "__main__":
    main()
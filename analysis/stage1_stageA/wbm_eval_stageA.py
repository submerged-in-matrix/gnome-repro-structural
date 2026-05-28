"""Stage A — WBM Evaluation Summary.

Supports both EMA variants and both TTA strategies via ``run_name``:

    run(run_name="stage_a",         ema_label="EMA-0.999")          # mean-TTA
    run(run_name="stage_a_ema99",   ema_label="EMA-0.99")           # mean-TTA
    run(run_name="stage_a", sub_dir="min_tta", ema_label="EMA-0.999 min-TTA")  # min-TTA

Reads:
    runs/<run_name>/metrics_wbm.json
    runs/<run_name>/f1_wbm.json

Produces per-run:
    results/stage1_<run_name>/02_wbm_regression_metrics.png
    results/stage1_<run_name>/02_wbm_confusion_matrix.png
    results/stage1_<run_name>/02_wbm_classification_metrics.png
    results/stage1_<run_name>/02_wbm_summary_table.csv

Cross-variant comparison (call after all desired runs complete):
    compare_ema_variants(summary_999, summary_99, summary_999_min, repo_root)
    -> results/stage1_stageA_compare/02_ema_wbm_comparison.png

Public API
----------
    run(repo_root, run_name, ema_label, show)
    compare_ema_variants(summary_999, summary_99, summary_999_min, repo_root, show)
    main()
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
GREY   = "#8C8C8C"
TEAL   = "#4EAAA1"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 120,
})

# -------------------------------------------------------------------
# Reference baselines
# -------------------------------------------------------------------

# 200-epoch seed-0 baseline — used when run_name == "stage_a"
# Regression: runs/default/metrics_wbm.json (mean-TTA)
# Classification: runs/default/f1_wbm.json
_BASELINE_200EP = {
    "label"    : "Baseline (200ep, mean-TTA)",
    "mae_meV"  : 168.7566,
    "rmse_meV" : 253.9444,
    "bias_meV" : 122.4786,
    "f1_raw"   : 0.365,
    "precision": 0.225,
    "recall"   : 0.964,
    "color"    : GREY,
}

# EMA-0.999 Stage A — used when run_name == "stage_a_ema99"
# Regression: runs/stage_a/metrics_wbm.json (mean-TTA)
# Classification: runs/stage_a/f1_wbm.json
_BASELINE_EMA999 = {
    "label"    : "Stage A (EMA-0.999, mean-TTA)",
    "mae_meV"  : 159.9269,
    "rmse_meV" : 244.8783,
    "bias_meV" : 110.8082,
    "f1_raw"   : 0.363,
    "precision": 0.212,
    "recall"   : 0.980,
    "color"    : BLUE,
}

_BASELINE_MAP = {
    "stage_a"       : _BASELINE_200EP,
    "stage_a_ema99" : _BASELINE_EMA999,
}


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------

def run(
    repo_root: Path | None = None,
    run_name:  str         = "stage_a",
    ema_label: str         = "EMA-0.999",
    sub_dir:   str | None  = None,
    show:      bool        = False,
) -> dict:
    """Run WBM evaluation summary for one Stage A variant.

    Parameters
    ----------
    run_name : str
        Sub-folder under ``runs/``.  e.g. ``"stage_a"``, ``"stage_a_ema99"``.
    sub_dir : str or None
        Optional sub-folder inside the run directory where JSON files live.
        Use ``"min_tta"`` for results at ``runs/<run_name>/min_tta/metrics_wbm.json``.
    ema_label : str
        Human-readable label for plots.
    """
    root    = Path(repo_root) if repo_root else _REPO_ROOT
    run_dir = root / "runs" / run_name / sub_dir if sub_dir else root / "runs" / run_name
    out_tag = f"{run_name}_{sub_dir}" if sub_dir else run_name
    out_dir = root / "results" / f"stage1_{out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics_wbm.json") as f:
        reg = json.load(f)
    with open(run_dir / "f1_wbm.json") as f:
        cls = json.load(f)

    mae_mev  = reg["mae_meV_per_atom"]
    rmse_mev = reg["rmse_meV_per_atom"]
    bias_mev = reg["bias_meV_per_atom"]

    raw  = cls["strategy_raw"]
    corr = cls.get("strategy_bias_corrected", None)

    baseline = _BASELINE_MAP.get(run_name, _BASELINE_200EP)
    tag      = f"Stage A ({ema_label})"

    # -------------------------------------------------------------------
    # Plot 1 — Regression metrics vs baseline
    # -------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    labels        = ["MAE", "RMSE", "Bias"]
    stageA_vals   = [mae_mev, rmse_mev, bias_mev]
    baseline_vals = [baseline["mae_meV"], baseline["rmse_meV"], baseline["bias_meV"]]

    x     = np.arange(len(labels))
    width = 0.3

    bars_b = ax1.bar(x - width / 2, baseline_vals, width,
                     color=baseline["color"], label=baseline["label"], edgecolor="white")
    bars_a = ax1.bar(x + width / 2, stageA_vals,   width,
                     color=ORANGE, label=tag, edgecolor="white")

    for bar, val in zip(list(bars_b) + list(bars_a), baseline_vals + stageA_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("meV/atom")
    ax1.set_title(f"{tag} vs {baseline['label']} — WBM Regression Metrics\n(20-pt TTA)")
    ax1.legend(fontsize=9)
    fig1.tight_layout()
    p1 = out_dir / "02_wbm_regression_metrics.png"
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
    p2 = out_dir / "02_wbm_confusion_matrix.png"
    fig2.savefig(p2, dpi=150)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------
    # Plot 3 — Precision / Recall / F1
    # -------------------------------------------------------------------
    metrics_names = ["Precision", "Recall", "F1"]
    stageA_cls    = [raw["precision"], raw["recall"], raw["f1"]]
    baseline_cls  = [baseline["precision"], baseline["recall"], baseline["f1_raw"]]

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    x     = np.arange(len(metrics_names))
    width = 0.25

    ax3.bar(x - width, baseline_cls, width,
            color=baseline["color"], label=baseline["label"], edgecolor="white")
    ax3.bar(x,          stageA_cls,  width,
            color=ORANGE,            label=f"{tag} raw",      edgecolor="white")

    if corr is not None:
        corr_vals = [corr["precision"], corr["recall"], corr["f1"]]
        ax3.bar(x + width, corr_vals, width,
                color=TEAL, label=f"{tag} bias-corrected", edgecolor="white")

    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Score")
    ax3.set_title(f"{tag} vs {baseline['label']} — Stability Classification\n(20-pt TTA)")
    ax3.axhline(0.5, color=GREY, ls="--", lw=1, alpha=0.6, label="F1 = 0.5 reference")
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
    # Summary dict / CSV
    # -------------------------------------------------------------------
    summary = {
        "run_name"           : run_name,
        "ema_label"          : ema_label,
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
        "delta_mae_meV"      : round(mae_mev  - baseline["mae_meV"],  2),
        "delta_bias_meV"     : round(bias_mev - baseline["bias_meV"], 2),
        "delta_f1_raw"       : round(raw["f1"] - baseline["f1_raw"],  4),
    }
    csv_path = out_dir / "02_wbm_summary_table.csv"
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(csv_path)
    print(f"Saved: {csv_path}")

    print(f"\n{'='*52}")
    print(f"  {tag} — WBM Evaluation Summary")
    print(f"{'='*52}")
    print(f"  N structures  : {reg['n_structures']:,}")
    print(f"  MAE           : {mae_mev:.2f} meV/atom  "
          f"(Δ {mae_mev - baseline['mae_meV']:+.1f} vs {baseline['label']})")
    print(f"  RMSE          : {rmse_mev:.2f} meV/atom")
    print(f"  Bias          : {bias_mev:.2f} meV/atom  "
          f"(Δ {bias_mev - baseline['bias_meV']:+.1f} vs {baseline['label']})")
    print(f"  F1  (raw)     : {raw['f1']:.4f}  "
          f"(Δ {raw['f1'] - baseline['f1_raw']:+.4f} vs {baseline['label']})")
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


# ---------------------------------------------------------------------------
# Cross-variant WBM comparison
# ---------------------------------------------------------------------------

def compare_ema_variants(
    summary_999:     dict,
    summary_99:      dict,
    summary_999_min: dict | None = None,
    repo_root:       Path | None = None,
    show:            bool        = False,
) -> None:
    """Side-by-side WBM metric comparison across all TTA strategies.

    Parameters
    ----------
    summary_999     : returned by run(run_name="stage_a",       ema_label="EMA-0.999")  [mean-TTA]
    summary_99      : returned by run(run_name="stage_a_ema99", ema_label="EMA-0.99")   [mean-TTA]
    summary_999_min : returned by run(run_name="stage_a", sub_dir="min_tta", ema_label="EMA-0.999 min-TTA")
                      Pass None to omit this series.

    Produces:
        results/stage1_stageA_compare/02_ema_wbm_comparison.png
    """
    root    = Path(repo_root) if repo_root else _REPO_ROOT
    out_dir = root / "results" / "stage1_stageA_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build series list dynamically so min-TTA is optional
    series = [
        ("Baseline\n(200ep)",          _BASELINE_200EP,  GREY),
        ("Stage A\n(EMA-0.999\nmean)", summary_999,      BLUE),
        ("Stage A\n(EMA-0.99\nmean)",  summary_99,       ORANGE),
    ]
    if summary_999_min is not None:
        series.append(("Stage A\n(EMA-0.999\nmin)", summary_999_min, TEAL))

    run_labels = [s[0] for s in series]
    summaries  = [s[1] for s in series]
    colors     = [s[2] for s in series]
    n          = len(series)

    def _bars(ax, values, title, ylabel):
        """Draw bars with labels that handle negative values correctly."""
        x    = np.arange(n)
        bars = ax.bar(x, values, color=colors, edgecolor="white", width=0.5)
        yrange = max(abs(v) for v in values) if values else 1
        offset = yrange * 0.02
        for bar, val in zip(bars, values):
            # place label above bar if positive, below if negative
            va   = "bottom" if val >= 0 else "top"
            y    = val + offset if val >= 0 else val - offset
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f"{val:.2f}", ha="center", va=va, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # extend y-axis so labels aren't clipped
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - yrange * 0.12, hi + yrange * 0.12)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35)

    # Panel A — MAE
    _bars(
        fig.add_subplot(gs[0, 0]),
        [s["mae_meV"] for s in summaries],
        "WBM MAE (mean-TTA)", "meV/atom",
    )

    # Panel B — Bias  (min-TTA flips sign — important to show)
    ax_b = fig.add_subplot(gs[0, 1])
    _bars(
        ax_b,
        [s["bias_meV"] for s in summaries],
        "WBM Bias (systematic shift)", "meV/atom",
    )
    ax_b.axhline(0, color=GREY, lw=0.8, ls="--", alpha=0.6)

    # Panel C — F1 (raw)
    ax_c = fig.add_subplot(gs[1, 0])
    _bars(
        ax_c,
        [s["f1_raw"] for s in summaries],
        "Stability F1 (raw strategy)", "F1 score",
    )
    ax_c.axhline(0.5, color=GREY, ls="--", lw=1, alpha=0.5, label="F1 = 0.5 ref")
    ax_c.legend(fontsize=7)

    # Panel D — Precision / Recall grouped bars
    ax_d   = fig.add_subplot(gs[1, 1])
    mnames = ["Precision", "Recall"]
    x      = np.arange(len(mnames))
    width  = 0.8 / n   # auto-scale width to number of series
    for i, (s, c, lbl) in enumerate(zip(summaries, colors, run_labels)):
        prec = s.get("precision", s.get("precision_raw"))
        rec  = s.get("recall",    s.get("recall_raw"))
        vals = [prec, rec]
        offset_x = (i - (n - 1) / 2) * width
        bars = ax_d.bar(x + offset_x, vals, width,
                        color=c, label=lbl.replace("\n", " "), edgecolor="white")
        for bar, val in zip(bars, vals):
            ax_d.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.01,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=6)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(mnames)
    ax_d.set_ylim(0, 1.15)
    ax_d.set_ylabel("Score")
    ax_d.set_title("Precision & Recall Breakdown")
    ax_d.legend(fontsize=6)

    fig.suptitle(
        "Stage A — TTA Strategy & EMA Variant Comparison  (WBM evaluation)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    p = out_dir / "02_ema_wbm_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")

    # Console delta table
    col_w = 12
    print(f"\n{'='*72}")
    print(f"  TTA Strategy & EMA Variant — WBM Evaluation")
    print(f"{'='*72}")
    hdrs  = ["Metric"] + [lbl.replace("\n", " ") for lbl in run_labels]
    print("  " + "  ".join(f"{h:<{col_w}}" for h in hdrs))
    print("  " + "  ".join("-" * col_w for _ in hdrs))
    metric_rows = [
        ("MAE (meV)",     [s["mae_meV"]                                   for s in summaries], ".2f"),
        ("RMSE (meV)",    [s["rmse_meV"]                                  for s in summaries], ".2f"),
        ("Bias (meV)",    [s["bias_meV"]                                  for s in summaries], ".2f"),
        ("F1 raw",        [s["f1_raw"]                                    for s in summaries], ".4f"),
        ("Precision raw", [s.get("precision", s.get("precision_raw"))     for s in summaries], ".4f"),
        ("Recall raw",    [s.get("recall",    s.get("recall_raw"))        for s in summaries], ".4f"),
    ]
    for name, vals, fmt in metric_rows:
        row = f"  {name:<{col_w}}  " + "  ".join(f"{v:{fmt}>{col_w}}" for v in vals)
        print(row)
    print(f"{'='*72}\n")

    if show:
        plt.show()
    else:
        plt.close("all")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--run-name",  type=str,  default="stage_a")
    parser.add_argument("--ema-label", type=str,  default="EMA-0.999")
    parser.add_argument("--compare",   action="store_true",
                        help="Run mean-TTA variants and produce comparison plot.")
    parser.add_argument("--compare-min-tta", action="store_true",
                        help="Include min-TTA series in comparison (requires runs/stage_a/min_tta/).")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.compare:
        s999 = run(repo_root=args.repo_root, run_name="stage_a",
                   ema_label="EMA-0.999", show=False)
        s99  = run(repo_root=args.repo_root, run_name="stage_a_ema99",
                   ema_label="EMA-0.99",  show=False)
        s_min = None
        if args.compare_min_tta:
            s_min = run(repo_root=args.repo_root, run_name="stage_a",
                        sub_dir="min_tta", ema_label="EMA-0.999 min-TTA", show=False)
        compare_ema_variants(s999, s99, s_min,
                             repo_root=args.repo_root, show=args.show)
    else:
        run(repo_root=args.repo_root,
            run_name=args.run_name,
            ema_label=args.ema_label,
            show=args.show)


if __name__ == "__main__":
    main()
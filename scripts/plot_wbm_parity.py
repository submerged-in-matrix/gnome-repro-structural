"""
Generate the WBM parity figure: predicted vs. true formation energy per atom,
colored by stability classification outcome (TP/FP/TN/FN).

Uses the same schema and scoring logic as eval_discovery_mbd.py:
  - each_true  = e_above_hull_mp2020_corrected_ppd_mp   (from wbm-summary)
  - e_form_dft = e_form_per_atom_mp2020_corrected        (from wbm-summary)
  - each_pred  = each_true + e_form_pred - e_form_dft    (hull-displacement)
  - predictions with |e_form_pred - e_form_dft| > 5 eV/atom masked as outliers,
    treated as predicted-unstable (matches stable_metrics fillna=True default)

Run this on the FINAL 6-seed ensemble predictions (the ones actually scored on
the leaderboard), not intermediate baseline/stageA single-seed runs — those use
different columns and the superseded naive threshold and will not reproduce the
submitted metrics.

Usage:
    python plot_wbm_parity.py \
        --predictions runs/ensemble/predictions_wbm.csv \
        --wbm-summary data/raw/wbm-summary.csv.gz
Output:
    fig1.png in the current directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]

_EACH_TRUE = "e_above_hull_mp2020_corrected_ppd_mp"
_E_FORM_DFT = "e_form_per_atom_mp2020_corrected"
_MAT_ID = "material_id"
_PRED_COL = "e_form_pred"
_MAX_ERROR = 5.0  # same outlier convention as eval_discovery_mbd.py

COLORS = {
    "TP": "#2ca02c",  # true positive  — correctly predicted stable
    "FP": "#d62728",  # false positive — predicted stable, actually not
    "FN": "#ff7f0e",  # false negative — predicted unstable, actually stable
    "TN": "#7f7f7f",  # true negative  — correctly predicted unstable (majority)
}
# TN is the majority class and would dominate visually — subsample it for plotting.
TN_SAMPLE_MAX = 15_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path,
        default=_REPO_ROOT / "runs" / "ensemble" / "predictions_wbm.csv",
    )
    parser.add_argument(
        "--wbm-summary", type=Path,
        default=_REPO_ROOT / "data" / "raw" / "wbm-summary.csv.gz",
    )
    parser.add_argument("--seed", type=int, default=0, help="TN subsample seed")
    args = parser.parse_args()

    summary = pd.read_csv(args.wbm_summary).set_index(_MAT_ID)
    preds = pd.read_csv(args.predictions).set_index(_MAT_ID)

    common = summary.index.intersection(preds.index)
    ref = summary.loc[common]
    e_form_pred = preds.loc[common, _PRED_COL].values
    e_form_dft = ref[_E_FORM_DFT].values
    each_true = ref[_EACH_TRUE].values

    error = np.abs(e_form_pred - e_form_dft)
    outlier_mask = error > _MAX_ERROR

    each_pred = each_true + e_form_pred - e_form_dft
    each_pred_scored = each_pred.copy()
    each_pred_scored[outlier_mask] = np.nan  # scored as predicted-unstable

    true_stable = each_true <= 0
    pred_stable = np.where(np.isnan(each_pred_scored), False, each_pred_scored <= 0)

    tp = true_stable & pred_stable
    fp = ~true_stable & pred_stable
    fn = true_stable & ~pred_stable
    tn = ~true_stable & ~pred_stable

    print(f"matched: {len(common):,}  outliers masked: {int(outlier_mask.sum()):,}")
    print(f"TP={tp.sum():,}  FP={fp.sum():,}  FN={fn.sum():,}  TN={tn.sum():,}")

    rng = np.random.default_rng(args.seed)
    tn_idx = np.where(tn)[0]
    if len(tn_idx) > TN_SAMPLE_MAX:
        tn_idx = rng.choice(tn_idx, size=TN_SAMPLE_MAX, replace=False)
    tn_plot_mask = np.zeros_like(tn)
    tn_plot_mask[tn_idx] = True

    fig, ax = plt.subplots(figsize=(6.2, 6))

    for mask, label, alpha, size, z in [
        (tn_plot_mask, "TN", 0.25, 6, 1),
        (fn, "FN", 0.55, 8, 2),
        (fp, "FP", 0.45, 8, 3),
        (tp, "TP", 0.55, 8, 4),
    ]:
        ax.scatter(
            e_form_dft[mask], e_form_pred[mask],
            s=size, alpha=alpha, color=COLORS[label], label=label,
            linewidths=0, zorder=z,
        )

    lo = min(e_form_dft.min(), e_form_pred.min())
    hi = max(e_form_dft.max(), e_form_pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", zorder=5, label="y = x")

    ax.set_xlabel("True formation energy (eV/atom, MP2020-corrected)")
    ax.set_ylabel("Predicted formation energy (eV/atom)")
    ax.set_title("EMA-GNN — WBM test set, full test set")
    leg = ax.legend(loc="upper left", markerscale=3, frameon=False)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig("fig1.png", dpi=200, bbox_inches="tight")
    print("Saved fig1.png")


if __name__ == "__main__":
    main()

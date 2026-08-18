"""
Generate the false-positive-rate-vs-chemical-complexity figure: FP rate among
predicted-stable materials, binned by number of distinct elements.

Uses the same schema and scoring logic as eval_discovery_mbd.py (see that file
for the full rationale):
  - each_true  = e_above_hull_mp2020_corrected_ppd_mp   (from wbm-summary)
  - e_form_dft = e_form_per_atom_mp2020_corrected        (from wbm-summary)
  - each_pred  = each_true + e_form_pred - e_form_dft    (hull-displacement)
  - predictions with |e_form_pred - e_form_dft| > 5 eV/atom masked as outliers,
    treated as predicted-unstable

Element counting reuses the approach from wbm_FP_diagnosis.ipynb (pymatgen
Composition parsed from the formula column), but classification here is fixed
to the corrected method — the notebook version used e_above_hull_wbm (WBM-native,
not MP2020-corrected) and a naive e_form_pred <= 0 threshold, both superseded.

Run this on the FINAL 6-seed ensemble predictions, not intermediate
baseline/stageA single-seed runs.

Usage:
    python plot_fp_vs_elements.py \
        --predictions runs/ensemble/predictions_wbm.csv \
        --wbm-summary data/raw/wbm-summary.csv.gz
Output:
    fig2.png in the current directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Composition

_REPO_ROOT = Path(__file__).resolve().parents[1]

_EACH_TRUE = "e_above_hull_mp2020_corrected_ppd_mp"
_E_FORM_DFT = "e_form_per_atom_mp2020_corrected"
_MAT_ID = "material_id"
_FORMULA_COL = "formula"
_PRED_COL = "e_form_pred"
_MAX_ERROR = 5.0

# Bucket 5+ elements together — WBM has very few quinary-and-above entries,
# and splitting further makes the tail bars statistically meaningless.
MAX_BUCKET = 5


def n_elements(formula_str: str) -> float:
    try:
        return len(Composition(formula_str).elements)
    except Exception:
        return np.nan


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
    each_pred[outlier_mask] = np.nan

    true_stable = each_true <= 0
    pred_stable = np.where(np.isnan(each_pred), False, each_pred <= 0)

    fp = ~true_stable & pred_stable
    tn = ~true_stable & ~pred_stable

    print("computing element counts from formula (pymatgen) — this can take a minute")
    n_el = ref[_FORMULA_COL].apply(n_elements).values
    n_el_bucketed = np.clip(n_el, None, MAX_BUCKET)

    df = pd.DataFrame({
        "n_el": n_el_bucketed,
        "fp": fp,
        "tn": tn,
        "not_true_stable": ~true_stable,
    }).dropna(subset=["n_el"])

    grouped = df.groupby("n_el").agg(
        n_fp=("fp", "sum"),
        n_not_stable=("not_true_stable", "sum"),
    )
    grouped["fp_rate"] = grouped["n_fp"] / grouped["n_not_stable"]
    print(grouped)

    labels = [
        f"{int(i)}" if i < MAX_BUCKET else f"{MAX_BUCKET}+"
        for i in grouped.index
    ]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(labels, grouped["fp_rate"] * 100, color="#d62728", alpha=0.85)
    for b, (n_fp, n) in zip(bars, zip(grouped["n_fp"], grouped["n_not_stable"])):
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
            f"n={int(n):,}", ha="center", va="bottom", fontsize=8, color="#444",
        )

    ax.set_xlabel("Number of distinct elements")
    ax.set_ylabel("False-positive rate among unstable materials (%)")
    ax.set_title("EMA-GNN — FP rate vs. chemical complexity")
    ax.set_ylim(0, max(grouped["fp_rate"]) * 100 * 1.25)

    fig.tight_layout()
    fig.savefig("fig2.png", dpi=200, bbox_inches="tight")
    print("Saved fig2.png")


if __name__ == "__main__":
    main()

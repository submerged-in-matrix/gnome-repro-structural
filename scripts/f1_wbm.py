"""Stage D (part 2): Stability classification F1 on WBM predictions.

Loads predictions_wbm.csv (from eval_wbm.py) and wbm-summary.csv.gz,
then computes F1, precision, and recall for stability classification.

Definition of stable (ground truth):
    e_above_hull_wbm <= 0  (structure lies on or below the convex hull)

Two classification strategies are evaluated:
    1. Raw threshold      : predicted e_form <= e_form_hull_cutoff (default 0)
    2. Bias-corrected     : same but predictions shifted by -bias before thresholding
       Rationale: the single-model has a known +122 meV/atom systematic bias
       from predicting on unrelaxed structures. Bias correction gives a fairer
       picture of ranking quality independent of the offset.

The Matbench Discovery leaderboard uses strategy 1 with a fixed threshold.
Strategy 2 is diagnostic only — it shows how much of the error is pure bias
vs. genuine ranking failure.

Outputs
-------
f1_wbm.json    precision, recall, F1 for both strategies + confusion matrix counts

Usage
-----
    python scripts/f1_wbm.py
    python scripts/f1_wbm.py --predictions runs/default/predictions_wbm.csv
    python scripts/f1_wbm.py --threshold 0.05   # 50 meV/atom above-hull cutoff
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from counts. Returns (P, R, F1)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def classify_and_score(
    e_pred: np.ndarray,
    e_true_above_hull: np.ndarray,
    pred_threshold: float,
    hull_threshold: float,
) -> dict:
    """Classify stability and compute metrics.

    A structure is predicted stable if e_pred <= pred_threshold.
    A structure is truly stable if e_true_above_hull <= hull_threshold.

    Returns a dict with tp, fp, fn, tn, precision, recall, f1.
    """
    pred_stable = e_pred <= pred_threshold
    true_stable = e_true_above_hull <= hull_threshold

    tp = int(( pred_stable &  true_stable).sum())
    fp = int(( pred_stable & ~true_stable).sum())
    fn = int((~pred_stable &  true_stable).sum())
    tn = int((~pred_stable & ~true_stable).sum())

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1"       : round(f1,        4),
        "n_pred_stable": int(pred_stable.sum()),
        "n_true_stable": int(true_stable.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WBM stability classification F1")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_REPO_ROOT / "runs" / "default" / "predictions_wbm.csv",
        help="Path to predictions_wbm.csv",
    )
    parser.add_argument(
        "--wbm-summary",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "wbm-summary.csv.gz",
        help="Path to wbm-summary.csv.gz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "runs" / "default",
        help="Directory to write f1_wbm.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Hull distance threshold in eV/atom for true-stable label (default: 0.0)",
    )
    args = parser.parse_args()

    # --- Load predictions ---
    if not args.predictions.exists():
        print(f"ERROR: predictions not found at {args.predictions}")
        return
    preds = pd.read_csv(args.predictions)
    print(f"Predictions loaded: {len(preds):,} rows")

    # --- Load WBM summary ---
    if not args.wbm_summary.exists():
        print(f"ERROR: wbm-summary not found at {args.wbm_summary}")
        return
    summary = pd.read_csv(args.wbm_summary)
    print(f"WBM summary loaded: {len(summary):,} rows")
    print(f"  columns: {list(summary.columns)}")

    # Identify the hull distance column.
    hull_col_candidates = [
        "e_above_hull_wbm",
        "e_above_hull",
        "decomposition_energy",
    ]
    hull_col = None
    for c in hull_col_candidates:
        if c in summary.columns:
            hull_col = c
            break

    if hull_col is None:
        print("ERROR: no hull distance column found in wbm-summary.")
        print("  Update hull_col_candidates with the correct column name.")
        return
    print(f"  Hull distance column: '{hull_col}'")

    # Identify the formation energy column (for bias computation).
    e_col_candidates = [
        "e_form_per_atom_wbm",
        "e_form_per_atom",
        "formation_energy_per_atom",
    ]
    e_col = None
    for c in e_col_candidates:
        if c in summary.columns:
            e_col = c
            break

    # --- Merge ---
    cols_to_merge = ["material_id", hull_col]
    if e_col:
        cols_to_merge.append(e_col)

    merged = preds.merge(summary[cols_to_merge], on="material_id", how="inner")
    merged = merged.dropna(subset=["e_form_pred", hull_col])
    print(f"\nMatched {len(merged):,} structures for F1 computation.")

    e_pred       = merged["e_form_pred"].values
    e_above_hull = merged[hull_col].values

    # --- Compute bias ---
    bias = float(np.mean(e_pred - merged[e_col].values)) if e_col else None
    if bias is not None:
        print(f"  Prediction bias (pred - true): {bias*1000:.2f} meV/atom")

    # --- Strategy 1: raw threshold on predicted formation energy ---
    # The predicted formation energy is used directly as a proxy for hull distance.
    # Threshold of 0.0 eV/atom on e_form_pred approximates "on or below the hull".
    raw = classify_and_score(
        e_pred            = e_pred,
        e_true_above_hull = e_above_hull,
        pred_threshold    = args.threshold,
        hull_threshold    = args.threshold,
    )

    # --- Strategy 2: bias-corrected predictions ---
    corrected = None
    if bias is not None:
        e_pred_corrected = e_pred - bias
        corrected = classify_and_score(
            e_pred            = e_pred_corrected,
            e_true_above_hull = e_above_hull,
            pred_threshold    = args.threshold,
            hull_threshold    = args.threshold,
        )

    # --- Print results ---
    print(f"\n{'='*50}")
    print(f"  Stability Classification — WBM (seed=0 baseline + 20-pt TTA)")
    print(f"  Hull threshold : {args.threshold*1000:.0f} meV/atom")
    print(f"  N structures   : {len(merged):,}")
    print(f"  N truly stable : {raw['n_true_stable']:,} "
          f"({100*raw['n_true_stable']/len(merged):.1f}%)")
    print()
    print(f"  --- Strategy 1: raw predictions ---")
    print(f"  Predicted stable : {raw['n_pred_stable']:,}")
    print(f"  Precision        : {raw['precision']:.4f}")
    print(f"  Recall           : {raw['recall']:.4f}")
    print(f"  F1               : {raw['f1']:.4f}")
    print(f"  TP/FP/FN/TN      : {raw['tp']}/{raw['fp']}/{raw['fn']}/{raw['tn']}")

    if corrected is not None:
        print()
        print(f"  --- Strategy 2: bias-corrected predictions ---")
        print(f"  Bias removed     : {bias*1000:.2f} meV/atom")
        print(f"  Predicted stable : {corrected['n_pred_stable']:,}")
        print(f"  Precision        : {corrected['precision']:.4f}")
        print(f"  Recall           : {corrected['recall']:.4f}")
        print(f"  F1               : {corrected['f1']:.4f}")
        print(f"  TP/FP/FN/TN      : "
              f"{corrected['tp']}/{corrected['fp']}/{corrected['fn']}/{corrected['tn']}")
    print(f"{'='*50}\n")

    # --- Save results ---
    results = {
        "n_structures"  : len(merged),
        "n_true_stable" : raw["n_true_stable"],
        "hull_threshold_eV": args.threshold,
        "strategy_raw"  : raw,
    }
    if corrected is not None:
        results["bias_meV_per_atom"]    = round(bias * 1000, 4)
        results["strategy_bias_corrected"] = corrected

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "f1_wbm.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
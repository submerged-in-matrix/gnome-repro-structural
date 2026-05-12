"""Compute WBM metrics from an existing predictions_wbm.csv.

Reads pre-computed predictions (from eval_wbm.py) and evaluates them
against the WBM ground truth in wbm-summary.csv.gz.  No model or
checkpoint is required.

Usage
-----
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py --predictions runs/default/predictions_wbm.csv
    python scripts/compute_metrics.py --out-dir runs/default
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

TTA_N_POINTS  = 20
TTA_SCALE_MIN = 0.80
TTA_SCALE_MAX = 1.20


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WBM metrics from predictions CSV")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_REPO_ROOT / "runs" / "default" / "predictions_wbm.csv",
        help="Path to predictions_wbm.csv (default: runs/default/predictions_wbm.csv)",
    )
    parser.add_argument(
        "--wbm-summary",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "wbm-summary.csv.gz",
        help="Path to wbm-summary.csv.gz (default: data/raw/wbm-summary.csv.gz)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write metrics_wbm.json (default: same directory as predictions)",
    )
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"ERROR: predictions not found at {args.predictions}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading predictions from {args.predictions} ...")
    pred_df = pd.read_csv(args.predictions)
    print(f"  {len(pred_df):,} rows loaded.")

    if not args.wbm_summary.exists():
        print(f"wbm-summary.csv.gz not found at {args.wbm_summary} — skipping metrics.")
        return

    print(f"\nLoading WBM ground truth from {args.wbm_summary} ...")
    summary = pd.read_csv(args.wbm_summary)

    # Columns are printed to allow verification of the correct column name.
    print(f"  wbm-summary columns: {list(summary.columns)}")

    # Common column name variants for WBM formation energy — first match is used.
    e_col_candidates = [
        "e_form_per_atom_wbm",
        "e_form_per_atom",
        "e_above_hull_wbm",
        "formation_energy_per_atom",
    ]
    e_col = None
    for candidate in e_col_candidates:
        if candidate in summary.columns:
            e_col = candidate
            break

    if e_col is None:
        print("WARNING: No formation energy column found in wbm-summary.")
        print("  Check the printed columns above and update e_col_candidates.")
        return

    print(f"  Using ground-truth column: '{e_col}'")

    # Merge predictions with ground truth on material_id.
    merged = pred_df.merge(
        summary[["material_id", e_col]].rename(columns={e_col: "e_form_true"}),
        on="material_id",
        how="inner",
    )
    merged = merged.dropna(subset=["e_form_pred", "e_form_true"])
    print(f"  Matched {len(merged):,} structures for metric computation.")

    mae  = float((merged["e_form_pred"] - merged["e_form_true"]).abs().mean())
    rmse = float(((merged["e_form_pred"] - merged["e_form_true"]) ** 2).mean() ** 0.5)
    bias = float((merged["e_form_pred"] - merged["e_form_true"]).mean())

    print(f"\n{'='*40}")
    print(f"  WBM Eval Results (seed=0 baseline + 20-pt TTA)")
    print(f"  N structures : {len(merged):,}")
    print(f"  MAE          : {mae*1000:.2f} meV/atom")
    print(f"  RMSE         : {rmse*1000:.2f} meV/atom")
    print(f"  Bias         : {bias*1000:.2f} meV/atom")
    print(f"{'='*40}\n")

    metrics = {
        "n_structures"      : len(merged),
        "mae_meV_per_atom"  : round(mae  * 1000, 4),
        "rmse_meV_per_atom" : round(rmse * 1000, 4),
        "bias_meV_per_atom" : round(bias * 1000, 4),
        "ground_truth_column": e_col,
        "tta_n_points"      : TTA_N_POINTS,
        "tta_scale_range"   : [TTA_SCALE_MIN, TTA_SCALE_MAX],
    }

    out_dir = args.out_dir if args.out_dir is not None else args.predictions.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_wbm.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
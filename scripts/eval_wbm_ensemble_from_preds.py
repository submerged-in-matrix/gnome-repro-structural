"""Stage D (ensemble): re-aggregate WBM ensemble predictions from existing
per-seed CSVs, using median across models (paper-faithful) instead of mean.

This is a fast-path companion to eval_wbm_ensemble.py. It does NOT rebuild
TTA graphs or run inference  and just redoes the aggregation + metrics step.


Outputs (under --ensemble-dir, default runs/ensemble)
------------------------------------------------------
predictions_wbm.csv              ensemble predictions (median of 6 seeds)
metrics_wbm_ensemble.json        MAE/RMSE/bias for each seed AND ensemble

Usage
-----
    python scripts/eval_wbm_ensemble_from_preds.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

SEEDS = list(range(6))


def compute_metrics(pred_df: pd.DataFrame, summary: pd.DataFrame, e_col: str) -> dict:
    """Same MAE/RMSE/bias computation as eval_wbm.py / eval_wbm_ensemble.py,
    inlined here to avoid importing eval_wbm_ensemble.py (which pulls in
    torch, torch_geometric, and gnome.graphs at module level).
    """
    merged = pred_df.merge(
        summary[["material_id", e_col]].rename(columns={e_col: "e_form_true"}),
        on="material_id", how="inner",
    )
    merged = merged.dropna(subset=["e_form_pred", "e_form_true"])
    mae  = float((merged["e_form_pred"] - merged["e_form_true"]).abs().mean())
    rmse = float(((merged["e_form_pred"] - merged["e_form_true"]) ** 2).mean() ** 0.5)
    bias = float((merged["e_form_pred"] - merged["e_form_true"]).mean())
    return {
        "n_structures"      : len(merged),
        "mae_meV_per_atom"  : round(mae  * 1000, 4),
        "rmse_meV_per_atom" : round(rmse * 1000, 4),
        "bias_meV_per_atom" : round(bias * 1000, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-aggregate ensemble WBM predictions via median, from existing per-seed CSVs"
    )
    parser.add_argument("--ensemble-dir", type=Path, default=_REPO_ROOT / "runs" / "ensemble",
                         help="Directory containing seed_0 .. seed_5 subfolders with predictions_wbm.csv")
    parser.add_argument("--wbm-summary", type=Path,
                         default=_REPO_ROOT / "data" / "raw" / "wbm-summary.csv.gz")
    args = parser.parse_args()

    # --- Load all 6 per-seed prediction CSVs ---
    per_seed_dfs = []
    for seed in SEEDS:
        pred_path = args.ensemble_dir / f"seed_{seed}" / "predictions_wbm.csv"
        if not pred_path.exists():
            print(f"ERROR: missing predictions {pred_path}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(pred_path)
        per_seed_dfs.append(df.rename(columns={"e_form_pred": f"e_form_pred_seed{seed}"}))
    print(f"Loaded {len(per_seed_dfs)} per-seed prediction CSVs.\n")

    # --- Explicit join on material_id across all 6, immune to row-order differences ---
    merged = per_seed_dfs[0]
    for df in per_seed_dfs[1:]:
        merged = merged.merge(df, on="material_id", how="inner")

    n_each = [len(df) for df in per_seed_dfs]
    if len(set(n_each)) > 1:
        print(f"WARNING: per-seed CSVs have differing row counts {n_each}; "
              f"inner join reduced to {len(merged):,} common structures.")

    pred_cols = [f"e_form_pred_seed{seed}" for seed in SEEDS]
    median_vals = merged[pred_cols].median(axis=1, skipna=True)

    ens_df = pd.DataFrame({
        "material_id": merged["material_id"],
        "e_form_pred": median_vals,
    })

    args.ensemble_dir.mkdir(parents=True, exist_ok=True)
    ens_path = args.ensemble_dir / "predictions_wbm.csv"
    ens_df.to_csv(ens_path, index=False)
    print(f"Ensemble (median) predictions -> {ens_path}  ({len(ens_df):,} structures)")

    # --- Metrics against WBM ground truth ---
    if not args.wbm_summary.exists():
        print(f"\nwbm-summary.csv.gz not found at {args.wbm_summary} — skipping metrics.")
        return

    summary = pd.read_csv(args.wbm_summary)
    e_col_candidates = [
        "e_form_per_atom_wbm", "e_form_per_atom",
        "e_above_hull_wbm", "formation_energy_per_atom",
    ]
    e_col = next((c for c in e_col_candidates if c in summary.columns), None)
    if e_col is None:
        print("WARNING: no formation energy column found in wbm-summary; skipping metrics.")
        return
    print(f"\nUsing ground-truth column: '{e_col}'")

    metrics = {"seeds": {}, "ensemble": None,
               "ground_truth_column": e_col,
               "tta_aggregation": "min_per_model_then_median_across_ensemble",
               "n_ensemble_members": len(SEEDS)}

    for seed, df in zip(SEEDS, per_seed_dfs):
        seed_df = df.rename(columns={f"e_form_pred_seed{seed}": "e_form_pred"})
        m = compute_metrics(seed_df, summary, e_col)
        metrics["seeds"][f"seed_{seed}"] = m
        print(f"  seed_{seed}: MAE {m['mae_meV_per_atom']:.2f} meV  "
              f"bias {m['bias_meV_per_atom']:+.2f} meV")

    metrics["ensemble"] = compute_metrics(ens_df, summary, e_col)
    print(f"  ENSEMBLE (median): MAE {metrics['ensemble']['mae_meV_per_atom']:.2f} meV  "
          f"bias {metrics['ensemble']['bias_meV_per_atom']:+.2f} meV")

    metrics_path = args.ensemble_dir / "metrics_wbm_ensemble.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")
    print("\nFor F1, run f1_wbm.py unchanged against either CSV, e.g.:")
    print(f"  python scripts/f1_wbm.py --predictions {ens_path} --out-dir {args.ensemble_dir}")


if __name__ == "__main__":
    main()
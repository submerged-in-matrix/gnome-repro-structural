"""Score WBM formation-energy predictions with the official Matbench Discovery code.

Supersedes scripts/f1_wbm.py, which thresholded raw predicted formation energy
(e_form_pred <= 0) as a stand-in for hull distance. That criterion flags ~88% of
WBM as stable while only ~16.7% lies on or below the hull, producing near-unity
recall and ~0.2 precision regardless of model quality.

The benchmark instead holds the convex hull fixed at its DFT reference values and
lets the formation-energy error displace each material across the stability line:

    each_pred = each_true + e_form_pred - e_form_dft

Metrics are then computed by matbench_discovery.metrics.stable_metrics, so the
numbers produced here are the numbers the leaderboard would produce.

Reference columns are the MP2020-corrected ones, not the WBM-native ones:
    each_true  -> e_above_hull_mp2020_corrected_ppd_mp
    e_form_dft -> e_form_per_atom_mp2020_corrected

Requires: pip install matbench-discovery

Outputs: discovery_metrics_mbd.json

Usage
-----
    python scripts/eval_discovery_mbd.py \
        --predictions runs/ensemble/predictions_wbm.csv \
        --wbm-summary data/raw/wbm-summary.csv.gz \
        --out-dir runs/ensemble --label unrelaxed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from matbench_discovery.metrics import stable_metrics

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Column names — hardcoded, no fallback list.  Absence is a hard error.
_EACH_TRUE = "e_above_hull_mp2020_corrected_ppd_mp"
_E_FORM_DFT = "e_form_per_atom_mp2020_corrected"
_UNIQ_PROTO = "unique_prototype"
_REQUIRED_COLS = (_EACH_TRUE, _E_FORM_DFT, _UNIQ_PROTO)

_MAT_ID = "material_id"
_PRED_COL = "e_form_pred"

# Predictions with abs error > this are masked as outliers (leaderboard convention).
_MAX_ERROR = 5.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score WBM predictions with official Matbench Discovery metrics"
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=_REPO_ROOT / "runs" / "ensemble" / "predictions_wbm.csv",
    )
    parser.add_argument(
        "--wbm-summary", type=Path,
        default=_REPO_ROOT / "data" / "raw" / "wbm-summary.csv.gz",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.predictions.parent

    # --- Load reference ---
    summary = pd.read_csv(args.wbm_summary)
    missing = [c for c in _REQUIRED_COLS if c not in summary.columns]
    if missing:
        raise KeyError(f"WBM summary lacks required columns {missing}")
    summary = summary.set_index(_MAT_ID)

    # --- Load predictions ---
    preds = pd.read_csv(args.predictions).set_index(_MAT_ID)
    if _PRED_COL not in preds.columns:
        raise KeyError(f"{_PRED_COL!r} absent from predictions CSV")

    # --- Align ---
    common = summary.index.intersection(preds.index)
    print(f"reference : {len(summary):,}")
    print(f"predictions: {len(preds):,}")
    print(f"matched    : {len(common):,}")
    if len(common) == 0:
        raise ValueError("no material_id overlap")

    ref = summary.loc[common]
    e_form_pred = preds.loc[common, _PRED_COL].values
    e_form_dft = ref[_E_FORM_DFT].values
    each_true = ref[_EACH_TRUE].values

    # --- Mask outliers (leaderboard convention) ---
    error = np.abs(e_form_pred - e_form_dft)
    outlier_mask = error > _MAX_ERROR
    n_outliers = int(outlier_mask.sum())
    print(f"masked (>{_MAX_ERROR} eV/atom error): {n_outliers:,}")

    # --- Convert formation energy prediction -> hull distance prediction ---
    # The convex hull is fixed from DFT; prediction error shifts materials
    # across the stability line.
    each_pred = each_true + e_form_pred - e_form_dft

    # Apply outlier mask: set outlier predictions to NaN so stable_metrics
    # treats them as "predicted unstable" (fillna=True default).
    each_pred[outlier_mask] = np.nan

    # --- Score: full test set ---
    full_metrics = stable_metrics(each_true, each_pred)

    # --- Score: unique prototypes only ---
    uniq_mask = ref[_UNIQ_PROTO].astype(bool).values
    uniq_metrics = stable_metrics(each_true[uniq_mask], each_pred[uniq_mask])

    # --- Print ---
    results = {}
    for subset_name, metrics in [
        ("full_test_set", full_metrics),
        ("unique_prototypes", uniq_metrics),
    ]:
        print(f"\n--- {subset_name} ---")
        for key in ("F1", "DAF", "Precision", "Recall", "Accuracy",
                     "MAE", "RMSE", "R2"):
            print(f"  {key:10s}: {metrics[key]:.4f}")
        results[subset_name] = {k: round(float(v), 4) for k, v in metrics.items()}

    if args.label:
        results["label"] = args.label
    results["predictions_file"] = str(args.predictions)
    results["n_matched"] = len(common)
    results["n_outliers_masked"] = n_outliers

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "discovery_metrics_mbd.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
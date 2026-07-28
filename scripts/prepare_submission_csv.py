"""Prepare the Matbench Discovery submission prediction file from ensemble predictions.

Matbench Discovery expects a gzipped CSV named:
    <yyyy-mm-dd>-wbm-IS2E.csv.gz

with exactly two columns:
    material_id                       — WBM material ID
    e_form_per_atom_ema_gnn           — predicted formation energy (eV/atom)

The prediction column name must match the model_key declared in the YAML:
    e_form_per_atom_<model_key>

Input: runs/ensemble/predictions_wbm.csv (material_id, e_form_pred)
Output: models/ema-gnn/2026-07-28-wbm-IS2E.csv.gz

Usage
-----
    python scripts/prepare_submission_csv.py
    python scripts/prepare_submission_csv.py --predictions runs/ensemble/predictions_wbm.csv
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_KEY = "ema-gnn"
_PRED_COL_IN = "e_form_pred"
_PRED_COL_OUT = f"e_form_per_atom_{_MODEL_KEY.replace('-', '_')}"
_DATE = "2026-07-28"
_OUT_NAME = f"{_DATE}-wbm-IS2E.csv.gz"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Matbench Discovery submission prediction file"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_REPO_ROOT / "runs" / "ensemble" / "predictions_wbm.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "models" / "ema-gnn",
    )
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions)

    for col in ("material_id", _PRED_COL_IN):
        if col not in preds.columns:
            raise KeyError(
                f"{col!r} not found in {args.predictions}. "
                f"Available columns: {list(preds.columns)}"
            )

    n_before = len(preds)
    preds = preds.dropna(subset=[_PRED_COL_IN])
    n_dropped = n_before - len(preds)
    if n_dropped:
        print(f"dropped {n_dropped} rows with NaN predictions")

    out = preds[["material_id"]].copy()
    out[_PRED_COL_OUT] = preds[_PRED_COL_IN].values

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / _OUT_NAME

    out.to_csv(out_path, index=False, compression="gzip")

    file_size = os.path.getsize(out_path)
    file_md5 = md5(out_path)

    print(f"rows written : {len(out):,}")
    print(f"output       : {out_path}")
    print(f"size (bytes) : {file_size:,}")
    print(f"md5          : {file_md5}")
    print()
    print("Paste these into ema-gnn.yml under metrics.discovery.pred_file:")
    print(f"  size: {file_size}")
    print(f"  md5: {file_md5}")
    print()
    print("Column name for YAML pred_col field:")
    print(f"  {_PRED_COL_OUT}")


if __name__ == "__main__":
    main()

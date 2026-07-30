"""Format WBM predictions as a Matbench Discovery submission file.

The benchmark expects a gzipped CSV at

    models/<arch-name>/<model-variant>/<yyyy-mm-dd>-discovery.csv.gz

holding material IDs matching the WBM test set and predicted formation energies
per atom in eV/atom. The energy column must be named exactly ``e_form_per_atom``:
``matbench_discovery.data.load_df_wbm_with_preds`` indexes that literal string and
raises if it is absent. The model key appears in the directory path, not in the
column name, and the older ``pred_col`` YAML field is no longer part of the schema.

Also emits the size and md5 checksum required by the metrics.discovery.pred_file
block of the model YAML.

Usage
-----
    python scripts/build_submission_preds.py \
        --predictions runs/ensemble/predictions_wbm.csv \
        --out-dir models/ema-gnn/ema-gnn
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

_MAT_ID = "material_id"
_PRED_COL = "e_form_pred"
_ENERGY_COL = "e_form_per_atom"


def md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex md5 digest of a file, read in chunks to bound memory use."""
    digest = hashlib.md5()  # noqa: S324 - checksum for integrity, not security
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format WBM predictions for Matbench Discovery submission"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=_REPO_ROOT / "runs" / "ensemble" / "predictions_wbm.csv",
        help="Source predictions CSV with material_id and e_form_pred columns",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "models" / "ema-gnn" / "ema-gnn",
        help="Directory to write the submission file into; the benchmark expects "
        "models/<arch-name>/<model-variant>/",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date stamp for the filename as YYYY-MM-DD (defaults to today, UTC)",
    )
    args = parser.parse_args()

    date_stamp = args.date or datetime.now(UTC).strftime("%Y-%m-%d")

    preds = pd.read_csv(args.predictions)
    for col in (_MAT_ID, _PRED_COL):
        if col not in preds.columns:
            raise KeyError(
                f"{col!r} absent from {args.predictions}. "
                f"Present columns: {sorted(preds.columns)}"
            )

    submission = preds[[_MAT_ID, _PRED_COL]].rename(columns={_PRED_COL: _ENERGY_COL})

    n_missing = int(submission[_ENERGY_COL].isna().sum())
    n_duplicated = int(submission[_MAT_ID].duplicated().sum())
    print(f"rows      : {len(submission):,}")
    print(f"missing   : {n_missing:,}")
    print(f"duplicated: {n_duplicated:,}")
    if n_duplicated:
        raise ValueError("duplicate material_id values in predictions")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{date_stamp}-discovery.csv.gz"
    submission.to_csv(out_path, index=False, compression="gzip")

    size_bytes = out_path.stat().st_size
    checksum = md5_of(out_path)

    # the YAML records the path relative to the Matbench Discovery repo root
    try:
        yaml_name = out_path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        yaml_name = out_path.as_posix()

    print(f"\nwrote -> {out_path}")
    print("\npaste into the model YAML under metrics.discovery.pred_file:")
    print(f"      name: {yaml_name}")
    print("      url: <fill in after uploading to Zenodo>")
    print(f"      size: {size_bytes}")
    print(f"      md5: {checksum}")


if __name__ == "__main__":
    main()
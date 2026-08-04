# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "gnome-repro[ml] @ git+https://github.com/submerged-in-matrix/gnome-repro-structural",
#   "matbench-discovery>=1.3.1",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "tqdm>=4.66",
# ]
#
# [tool.uv.sources]
# matbench-discovery = { path = "../../../", editable = true }
# ///
"""Reproduce EMA-GNN discovery predictions on the WBM test set.

Loads 6 EMA-averaged checkpoints from Figshare, runs 20-point volume
test-time augmentation (TTA) on each WBM unrelaxed structure, takes the
min-TTA prediction per model, then the median across the 6 models.
Writes a discovery CSV and prints metrics.

Usage:
    uv run models/ema-gnn/ema-gnn/test_ema_gnn_discovery.py
    uv run models/ema-gnn/ema-gnn/test_ema_gnn_discovery.py --limit 100
    uv run models/ema-gnn/ema-gnn/test_ema_gnn_discovery.py --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import zipfile
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data
from tqdm import tqdm

from gnome.graphs import structure_to_graph
from gnome.model import GNoMEStructural

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS = 6
N_TTA = 20
V_MIN, V_MAX = 0.8, 1.2
TTA_SCALES = np.linspace(V_MIN, V_MAX, N_TTA) ** (1 / 3)  # linear in volume

# NOTE: figshare.com/ndownloader/... is behind an AWS WAF challenge that
# rejects non-browser clients. The legacy ndownloader.figshare.com host
# returns a direct redirect to S3 and works with plain urllib.
CHECKPOINT_URL = "https://ndownloader.figshare.com/files/67085774"
CHECKPOINT_MD5 = None  # add after verifying the zip md5
CACHE_DIR = Path.home() / ".cache" / "ema-gnn"

# WBM initial structures — Figshare hosted by Matbench Discovery.
# Columnar JSON: keys 'material_id', 'formula_from_cse', 'initial_structure'.
WBM_STRUCTS_URL = "https://ndownloader.figshare.com/files/40344466"
WBM_STRUCTS_FILENAME = "2022-10-19-wbm-init-structs.json.bz2"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path, label: str = "") -> None:
    if dest.exists():
        print(f"Skipping {label or url} (already exists at {dest})")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {label or url}")
    tmp = dest.with_suffix(".tmp")

    # Prefer curl.exe on Windows to bypass the PowerShell `curl` alias.
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError(
            "curl is not available; install curl or download the file manually"
        )

    # -L follows the Figshare 302 -> S3 presigned redirect.
    # --retry handles transient S3 errors; -f fails on HTTP errors.
    # NOTE: do NOT send a browser User-Agent. The WAF checks UA/TLS-fingerprint
    # consistency: a Chrome UA with curl's TLS fingerprint triggers HTTP 202 +
    # empty body, while curl's default UA passes.
    cmd = [curl, "-L", "-f", "--retry", "3", "--retry-delay", "2", "-o", str(tmp), url]

    last_err: Exception | None = None
    for attempt in range(4):
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            last_err = exc
            tmp.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
            continue
        if tmp.stat().st_size > 0:
            break
        tmp.unlink(missing_ok=True)
        last_err = RuntimeError(f"curl downloaded 0 bytes from {url}")
        time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"curl failed to download {url}: {last_err}") from last_err

    tmp.rename(dest)
    print(f"  saved to {dest}")


def download_checkpoints() -> Path:
    """Download and unzip the 6-seed checkpoint archive. Returns cache dir."""
    zip_path = CACHE_DIR / "ema-gnn-checkpoints.tar.gz"
    print(f"Downloading checkpoints to {zip_path} ...")
    _download(CHECKPOINT_URL, zip_path, label="ema-gnn-checkpoints.tar.gz")

    # Unzip if seed_0/best.pt is missing.
    marker = CACHE_DIR / "seed_0" / "best.pt"
    if not marker.exists():
        print("Now... Extracting checkpoints")
        with tarfile.open(zip_path, "r:gz") as tf:
            tf.extractall(CACHE_DIR)
        assert marker.exists(), f"Expected {marker} after extraction"
        print(f"  extracted to {CACHE_DIR}")

    return CACHE_DIR


def download_wbm_structures() -> Path:
    """Download the WBM initial-structures JSON. Returns local path."""
    dest = CACHE_DIR / WBM_STRUCTS_FILENAME
    _download(WBM_STRUCTS_URL, dest, label=WBM_STRUCTS_FILENAME)
    return dest


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models(
    cache_dir: Path, device: torch.device
) -> tuple[list[torch.nn.Module], dict]:
    """Load 6 seed checkpoints. Returns (models_list, stats_dict)."""
    models = []
    stats = None

    for seed in range(N_SEEDS):
        ckpt_path = cache_dir / f"seed_{seed}" / "best.pt"
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]

        if stats is None:
            stats = ckpt["stats"]

        model = GNoMEStructural(
            avg_adjacency=stats["avg_adjacency"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            use_adj_norm=cfg.get("use_adj_norm", True),
        ).to(device)

        # The checkpoints were saved with an older model version that did not
        # register the `avg_adjacency` buffers. The current model initializes
        # them correctly from `stats["avg_adjacency"]` in the constructor, so
        # load with strict=False to ignore the missing buffer keys.
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} models on {device}")
    print(f"  label_mean={stats['label_mean']:.4f}, "
          f"label_std={stats['label_std']:.4f}")
    return models, stats


# ---------------------------------------------------------------------------
# WBM structure loading
# ---------------------------------------------------------------------------
def load_wbm_structures(
    path: Path, limit: int | None = None
) -> list[tuple[str, Structure]]:
    """Load WBM initial structures as (material_id, Structure) pairs.

    The JSON is columnar: wbm['material_id']['0'], wbm['initial_structure']['0'], etc.
    For .json.bz2, pandas handles decompression transparently.
    """
    import bz2
    import json

    open_fn = bz2.open if str(path).endswith(".bz2") else open
    with open_fn(path, "rt") as f:
        wbm = json.load(f)

    ids = wbm["material_id"]
    structs = wbm["initial_structure"]
    n = len(ids)

    if limit is not None:
        n = min(n, limit)

    result = []
    for i in range(n):
        key = str(i)
        mid = ids[key]
        struct = Structure.from_dict(structs[key])
        result.append((mid, struct))

    print(f"Loaded {len(result)} WBM structures")
    return result


# ---------------------------------------------------------------------------
# TTA + ensemble inference
# ---------------------------------------------------------------------------
def rescale_structure(structure: Structure, scale: float) -> Structure:
    """Isotropically rescale lattice vectors by `scale`."""
    s = deepcopy(structure)
    s.lattice = s.lattice.scale(s.lattice.volume * scale**3)
    return s


def build_tta_graphs(
    structure: Structure,
) -> list[Data]:
    """Build 20 TTA-scaled graphs for one structure.

    Returns list of PyG Data objects (may be <20 if some scales fail).
    Graph construction is the CPU bottleneck — build once, reuse across models.
    """
    graphs = []
    for scale in TTA_SCALES:
        rs = rescale_structure(structure, scale)
        g = structure_to_graph(rs, e_form_per_atom=0.0)
        if g is not None:
            # Add batch index for single-graph forward pass.
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            graphs.append(g)
    return graphs


@torch.no_grad()
def predict_single_model(
    model: torch.nn.Module,
    graphs: list[Data],
    device: torch.device,
    mu: float,
    sigma: float,
) -> float:
    """Predict on TTA graphs with one model, return min energy (eV/atom)."""
    energies = []
    for g in graphs:
        g_dev = g.to(device)
        pred_norm = model(g_dev).item()
        pred_ev = pred_norm * sigma + mu  # unstandardize
        energies.append(pred_ev)
    return min(energies) if energies else float("nan")


def run_ensemble(
    models: list[torch.nn.Module],
    structures: list[tuple[str, Structure]],
    stats: dict,
    device: torch.device,
) -> pd.DataFrame:
    """Run full TTA-ensemble inference. Returns DataFrame with predictions."""
    mu = stats["label_mean"]
    sigma = stats["label_std"]

    records = []
    for mid, struct in tqdm(structures, desc="Predicting"):
        graphs = build_tta_graphs(struct)

        if not graphs:
            records.append({"material_id": mid, "e_form_per_atom": float("nan")})
            continue

        seed_preds = []
        for model in models:
            e = predict_single_model(model, graphs, device, mu, sigma)
            seed_preds.append(e)

        # Median across seeds (paper-faithful: resists OOD-poisoned model).
        ensemble_pred = float(np.median(seed_preds))
        records.append({"material_id": mid, "e_form_per_atom": ensemble_pred})

    df = pd.DataFrame(records).set_index("material_id")
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(df_pred: pd.DataFrame) -> dict:
    """Compute Matbench Discovery metrics against WBM ground truth.

    Uses the corrected MP2020 reference energies and convex hull distances.
    Formula: e_above_hull_pred = e_above_hull_true + (e_form_pred - e_form_dft)
    """
    # Pre-download via curl to bypass the Figshare WAF that blocks urllib
    _download(
        "https://ndownloader.figshare.com/files/44225498",
        Path.home() / ".cache" / "matbench-discovery" / "wbm" / "2023-12-13-wbm-summary.csv.gz",
        label="wbm-summary.csv.gz",
    )
    from matbench_discovery.data import df_wbm
    from matbench_discovery.metrics import stable_metrics

    # Align predictions with ground truth by material_id.
    df = df_wbm.join(df_pred, how="inner")
    n_missing = len(df_wbm) - len(df)
    print(f"  aligned {len(df)} predictions ({n_missing} missing)")

    # Corrected reference columns.
    col_hull = "e_above_hull_mp2020_corrected_ppd_mp"
    col_eform_dft = "e_form_per_atom_mp2020_corrected"

    e_above_hull_true = df[col_hull].values
    e_form_pred = df["e_form_per_atom"].values
    e_form_dft = df[col_eform_dft].values

    # Convert formation-energy prediction to hull-distance prediction.
    e_above_hull_pred = e_above_hull_true + (e_form_pred - e_form_dft)

    # Classification: stable = below hull (pred <= 0).
    true_stable = e_above_hull_true <= 0
    pred_stable = e_above_hull_pred <= 0

    tp = int(np.sum(true_stable & pred_stable))
    fp = int(np.sum(~true_stable & pred_stable))
    fn = int(np.sum(true_stable & ~pred_stable))
    tn = int(np.sum(~true_stable & ~pred_stable))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Regression metrics on formation energy.
    errors = e_form_pred - e_form_dft
    mae = float(np.nanmean(np.abs(errors)))
    rmse = float(np.sqrt(np.nanmean(errors**2)))

    ss_res = np.nansum(errors**2)
    ss_tot = np.nansum((e_form_dft - np.nanmean(e_form_dft)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    results = {
        "n_predictions": len(df),
        "n_missing": n_missing,
        "MAE_eV_per_atom": round(mae, 4),
        "RMSE_eV_per_atom": round(rmse, 4),
        "R2": round(r2, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: cuda if available)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N structures (for testing)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output CSV path (default: ./<date>-discovery.csv.gz)",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    # # 1. Download checkpoints and WBM structures.
    # cache_dir = download_checkpoints()
    # wbm_path = download_wbm_structures()

    # # 2. Load models.
    # models, stats = load_models(cache_dir, device)

    # # 3. Load WBM structures.
    # structures = load_wbm_structures(wbm_path, limit=args.limit)

    # # 4. Run inference.
    # df_pred = run_ensemble(models, structures, stats, device)
    # n_valid = df_pred["e_form_per_atom"].notna().sum()
    # n_nan = df_pred["e_form_per_atom"].isna().sum()
    # print(f"\nPredictions: {n_valid} valid, {n_nan} failed")

    # # 5. Write CSV.
    # from matbench_discovery import today

    # out_path = args.out or f"{today}-discovery.csv.gz"
    # df_pred.to_csv(out_path)
    # size_mb = os.path.getsize(out_path) / 1e6
    # print(f"Wrote {out_path} ({size_mb:.1f} MB)")

    # 6. Metrics (skip for --limit runs).
    current_path = "2026-08-03-discovery.csv.gz"
    df_pred = pd.read_csv(current_path, index_col="material_id")

    if args.limit is None:
        print("\nComputing metrics ...")
        metrics = compute_metrics(df_pred)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print(f"\nSkipping metrics (--limit {args.limit} active)")


if __name__ == "__main__":
    main()
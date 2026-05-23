"""Stage D: WBM evaluation with 20-point volume TTA.

Loads the seed=0 baseline checkpoint (runs/default/best.pt) and runs
inference on all 256,963 WBM initial (unrelaxed) structures.

Volume TTA: for each structure, the lattice is scaled isotropically at
20 factors linearly spaced from 0.80 to 1.20 of the reference volume
(per GNoME paper Methods). Inference is run on each scaled variant and
predictions are averaged. This compensates for the train-on-relaxed /
predict-on-unrelaxed distribution shift.

Outputs
-------
predictions_wbm.csv   material_id, e_form_pred (eV/atom)
metrics_wbm.json      MAE vs WBM ground truth (if wbm-summary.csv.gz present)

Usage
-----
    python scripts/eval_wbm.py                        # full 256K run
    python scripts/eval_wbm.py --limit 1000           # quick sanity check
    python scripts/eval_wbm.py --checkpoint runs/default/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Batch
from tqdm import tqdm

# Ensure src/ is on path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gnome.graphs import structure_to_graph
from gnome.model import GNoMEStructural


# ---------------------------------------------------------------------------
# TTA configuration — matches GNoME paper Methods exactly.
# ---------------------------------------------------------------------------
TTA_N_POINTS = 20
TTA_SCALE_MIN = 0.80   # 80% of reference volume
TTA_SCALE_MAX = 1.20   # 120% of reference volume

# Lattice scale factors: volume scales as a^3, so lattice scale = vol_scale^(1/3)
_vol_scales = np.linspace(TTA_SCALE_MIN, TTA_SCALE_MAX, TTA_N_POINTS)
TTA_LATTICE_SCALES = _vol_scales ** (1.0 / 3.0)  # shape (20,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    stats = ckpt["stats"]

    model = GNoMEStructural(
        avg_adjacency=stats["avg_adjacency"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        use_adj_norm=cfg.get("use_adj_norm", True),
    )

    state = ckpt["model_state"]

    # EMA shadow only contains parameters (requires_grad=True).
    # Buffers like avg_adjacency are missing — fill from the freshly
    # initialised model which already has them set correctly.
    model_state = model.state_dict()
    for key in model_state:
        if key not in state:
            state[key] = model_state[key]   # restore missing buffers

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    mu    = stats["label_mean"]
    sigma = stats["label_std"]
    print(f"Model loaded from {checkpoint_path}")
    print(f"  hidden_dim={cfg['hidden_dim']}, n_layers={cfg['n_layers']}, "
          f"use_adj_norm={cfg.get('use_adj_norm', True)}")
    print(f"  label stats: mu={mu:.4f}, sigma={sigma:.4f}")
    return model, mu, sigma

def scale_structure(structure: Structure, lattice_scale: float) -> Structure:
    """Return a copy of the input structure with the lattice uniformly scaled."""
    new_lattice = structure.lattice.matrix * lattice_scale
    return Structure(
        new_lattice,
        [site.specie for site in structure],
        [site.frac_coords for site in structure],
        coords_are_cartesian=False,
    )

def predict_with_tta(
    structure: Structure,
    model: GNoMEStructural,
    mu: float,
    sigma: float,
    device: torch.device,
    aggregator: str = "mean", 
) -> float | None:
    """Trial with min: Run 20-pt volume TTA for one structure. Returns mean prediction in eV/atom.
    Returns None if all 20 graph-builds fail (e.g. no edges within the 4 A cutoff at any volume scale).
    
    Paper version:Run 20-pt volume TTA for one structure. Returns aggregated prediction in eV/atom.
    # Aggregator: 'min' (paper-faithful) picks the lowest energy across volume scales,
    # approximating a 1D variable-cell relaxation. 'mean' averages all scales (original).
    """
    graphs = []
    for ls in TTA_LATTICE_SCALES:
        scaled = scale_structure(structure, ls)
        # Dummy label 0.0 satisfies the structure_to_graph signature;
        # the value is not used during inference.
        g = structure_to_graph(scaled, 0.0)
        if g is not None:
            graphs.append(g)

    if not graphs:
        return None

    # All valid TTA variants are batched into a single forward pass.
    batch = Batch.from_data_list(graphs).to(device)
    with torch.no_grad():
        pred_norm = model(batch)       # (n_valid_tta,) normalized
        pred = pred_norm * sigma + mu  # de-normalize to eV/atom

    # return float(pred.mean().item())
    return float(pred.min().item()) if aggregator == "min" else float(pred.mean().item())


def load_wbm_structures(json_path: Path) -> tuple[list[str], list[Structure]]:
    """Load WBM initial structures from the columnar JSON file.

    Format: data[field][str_index] = value
    Fields: 'material_id', 'formula_from_cse', 'initial_structure'
    """
    print(f"Loading WBM initial structures from {json_path} ...")
    with open(json_path) as f:
        data = json.load(f)

    n = len(data["material_id"])
    print(f"  {n:,} entries found.")

    ids        = []
    structures = []
    n_failed   = 0

    for idx_str in tqdm(data["material_id"], desc="parsing structures", total=n):
        mid         = data["material_id"][idx_str]
        struct_dict = data["initial_structure"][idx_str]
        try:
            s = Structure.from_dict(struct_dict)
            ids.append(mid)
            structures.append(s)
        except Exception:
            n_failed += 1

    if n_failed:
        print(f"  Warning: {n_failed} structures failed to parse and were skipped.")

    return ids, structures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WBM eval with volume TTA")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_ROOT / "runs" / "default" / "best.pt",
        help="Path to best.pt checkpoint (default: runs/default/best.pt)",
    )
    parser.add_argument(
        "--wbm-structs",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "2022-10-19-wbm-init-structs.json",
        help="Path to WBM initial structures JSON",
    )
    parser.add_argument(
        "--wbm-summary",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "wbm-summary.csv.gz",
        help="Path to wbm-summary.csv.gz (for computing metrics)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "runs" / "default",
        help="Directory to write output files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate first N structures (for quick sanity checks)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--batch-report-every",
        type=int,
        default=1000,
        help="Print progress every N structures",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="mean",
        choices=["mean", "min"],
        help="TTA aggregator: 'mean' (original) or 'min' (paper-faithful minimum reduction)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Load model ---
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    model, mu, sigma = load_model(args.checkpoint, device)

    # --- Load WBM structures ---
    if not args.wbm_structs.exists():
        print(f"ERROR: WBM structures not found at {args.wbm_structs}", file=sys.stderr)
        sys.exit(1)
    ids, structures = load_wbm_structures(args.wbm_structs)

    if args.limit is not None:
        ids        = ids[:args.limit]
        structures = structures[:args.limit]
        print(f"  Limiting to first {args.limit} structures (--limit flag).")

    # --- Run TTA inference ---
    print(f"\nRunning 20-pt volume TTA on {len(structures):,} structures ...")
    print(f"TTA lattice scales: {TTA_LATTICE_SCALES[0]:.4f} -> {TTA_LATTICE_SCALES[-1]:.4f}")

    preds    = []
    n_failed = 0
    t0       = time.time()

    for i, (mid, struct) in enumerate(zip(ids, structures)):
        # pred = predict_with_tta(struct, model, mu, sigma, device) # original mean aggregator 
        pred = predict_with_tta(struct, model, mu, sigma, device, args.aggregator)
        if pred is None:
            preds.append(float("nan"))
            n_failed += 1
        else:
            preds.append(pred)

        if (i + 1) % args.batch_report_every == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (len(structures) - i - 1) / rate
            print(f"  [{i+1:>7,}/{len(structures):,}]  "
                  f"{rate:.1f} structs/s  "
                  f"ETA {remaining/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\nDone. {len(structures):,} structures in {elapsed/60:.1f} min "
          f"({len(structures)/elapsed:.1f} structs/s)")
    if n_failed:
        print(f"Warning: {n_failed} structures produced no valid graphs (NaN prediction).")

    # --- Save predictions ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_df   = pd.DataFrame({"material_id": ids, "e_form_pred": preds})
    pred_path = args.out_dir / "predictions_wbm.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved -> {pred_path}")

    # --- Compute metrics against WBM ground truth ---
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
        "checkpoint"        : str(args.checkpoint),
        "tta_n_points"      : TTA_N_POINTS,
        "tta_scale_range"   : [TTA_SCALE_MIN, TTA_SCALE_MAX],
        "tta_aggregator"    : args.aggregator,   # "mean" or "min"
    }
    metrics_path = args.out_dir / "metrics_wbm.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()
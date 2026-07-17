"""Stage D (ensemble): WBM evaluation for the 6-seed ensemble.

Paper-faithful aggregation: min-TTA per model, then median across the 6
models. Reuses load_model / scale_structure / load_wbm_structures from
eval_wbm.py unchanged; no TTA or metric logic is duplicated here. However, the ensemble
was first inferenced using mean. Then with a seperate script i will load the predictions 
and compute the metrics using median.

Efficiency note: the 20 volume-scaled graphs for a structure do not
depend on which model evaluates them. They are built once per structure
and the same batch is passed through all 6 models, so graph construction
(the CPU-bound cost) is paid once, not six times.

Outputs (under --ensemble-dir, default runs/ensemble)
------------------------------------------------------
seed_N/predictions_wbm.csv       per-seed min-TTA predictions (schema
                                  matches eval_wbm.py's output, so
                                  f1_wbm.py runs on it unchanged)
predictions_wbm.csv              ensemble predictions (median of 6 mins)
metrics_wbm_ensemble.json        MAE/RMSE/bias for each seed AND ensemble

Usage
-----
    python scripts/eval_wbm_ensemble.py
    python scripts/eval_wbm_ensemble.py --limit 1000      
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
from torch_geometric.data import Batch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for eval_wbm import

from gnome.graphs import structure_to_graph
from eval_wbm import load_model, scale_structure, load_wbm_structures, TTA_LATTICE_SCALES

SEEDS = list(range(6))


def predict_tta_all_models(structure, models_stats, device) -> list[float | None]:
    """Build the 20 TTA graphs once, then min-reduce per model over that
    single batch. Returns one value per model; None where no TTA variant
    produced a valid graph (identical across models, since graph validity
    does not depend on model weights).
    """
    graphs = []
    for ls in TTA_LATTICE_SCALES:
        scaled = scale_structure(structure, ls)
        g = structure_to_graph(scaled, 0.0)
        if g is not None:
            graphs.append(g)

    if not graphs:
        return [None] * len(models_stats)

    batch = Batch.from_data_list(graphs).to(device)
    results = []
    for model, mu, sigma in models_stats:
        with torch.no_grad():
            pred_norm = model(batch)
            pred = pred_norm * sigma + mu
        results.append(float(pred.min().item()))
    return results


def compute_metrics(pred_df: pd.DataFrame, summary: pd.DataFrame, e_col: str) -> dict:
    """Same MAE/RMSE/bias computation as eval_wbm.py, factored out for reuse
    across 6 seeds and the ensemble.
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
    parser = argparse.ArgumentParser(description="Ensemble WBM eval, min-TTA per model then median")
    parser.add_argument("--ensemble-dir", type=Path, default=_REPO_ROOT / "runs" / "ensemble",
                         help="Directory containing seed_0 .. seed_5 subfolders with best.pt")
    parser.add_argument("--wbm-structs", type=Path,
                         default=_REPO_ROOT / "data" / "raw" / "2022-10-19-wbm-init-structs.json")
    parser.add_argument("--wbm-summary", type=Path,
                         default=_REPO_ROOT / "data" / "raw" / "wbm-summary.csv.gz")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only evaluate first N structures (sanity check)")
    parser.add_argument("--device", type=str,
                         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-report-every", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Load all 6 checkpoints upfront ---
    models_stats = []
    for seed in SEEDS:
        ckpt_path = args.ensemble_dir / f"seed_{seed}" / "best.pt"
        if not ckpt_path.exists():
            print(f"ERROR: missing checkpoint {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        model, mu, sigma = load_model(ckpt_path, device)
        models_stats.append((model, mu, sigma))
    print(f"Loaded {len(models_stats)} ensemble members.\n")

    # --- Load WBM structures once ---
    if not args.wbm_structs.exists():
        print(f"ERROR: WBM structures not found at {args.wbm_structs}", file=sys.stderr)
        sys.exit(1)
    ids, structures = load_wbm_structures(args.wbm_structs)
    if args.limit is not None:
        ids        = ids[:args.limit]
        structures = structures[:args.limit]
        print(f"  Limiting to first {args.limit} structures (--limit flag).")

    n = len(structures)
    per_seed_preds: list[list[float]] = [[] for _ in SEEDS]
    ensemble_preds: list[float] = []

    print(f"Running ensemble min-TTA on {n:,} structures "
          f"(graphs built once per structure, reused across all 6 models) ...")
    t0 = time.time()
    for i, (mid, struct) in enumerate(zip(ids, structures)):
        seed_vals = predict_tta_all_models(struct, models_stats, device)
        for s_idx, v in enumerate(seed_vals):
            per_seed_preds[s_idx].append(v if v is not None else float("nan"))
        valid = [v for v in seed_vals if v is not None]
        ensemble_preds.append(float(np.median(valid)) if valid else float("nan"))

        if (i + 1) % args.batch_report_every == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (n - i - 1) / rate
            print(f"  [{i+1:>7,}/{n:,}]  {rate:.1f} structs/s  ETA {remaining/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\nDone. {n:,} structures in {elapsed/60:.1f} min ({n/elapsed:.1f} structs/s)")

    # --- Save per-seed prediction CSVs (schema matches eval_wbm.py output) ---
    args.ensemble_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dfs = []
    for s_idx, seed in enumerate(SEEDS):
        out_dir = args.ensemble_dir / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"material_id": ids, "e_form_pred": per_seed_preds[s_idx]})
        df.to_csv(out_dir / "predictions_wbm.csv", index=False)
        per_seed_dfs.append(df)
        print(f"  seed_{seed} predictions -> {out_dir / 'predictions_wbm.csv'}")

    ens_df = pd.DataFrame({"material_id": ids, "e_form_pred": ensemble_preds})
    ens_path = args.ensemble_dir / "predictions_wbm.csv"
    ens_df.to_csv(ens_path, index=False)
    print(f"  ensemble predictions -> {ens_path}")

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

    for s_idx, seed in enumerate(SEEDS):
        m = compute_metrics(per_seed_dfs[s_idx], summary, e_col)
        metrics["seeds"][f"seed_{seed}"] = m
        print(f"  seed_{seed}: MAE {m['mae_meV_per_atom']:.2f} meV  "
              f"bias {m['bias_meV_per_atom']:+.2f} meV")

    metrics["ensemble"] = compute_metrics(ens_df, summary, e_col)
    print(f"  ENSEMBLE: MAE {metrics['ensemble']['mae_meV_per_atom']:.2f} meV  "
          f"bias {metrics['ensemble']['bias_meV_per_atom']:+.2f} meV")

    metrics_path = args.ensemble_dir / "metrics_wbm_ensemble.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")
    print("\nFor F1, run f1_wbm.py unchanged against either CSV, e.g.:")
    print(f"  python scripts/f1_wbm.py --predictions {ens_path} --out-dir {args.ensemble_dir}")


if __name__ == "__main__":
    main()

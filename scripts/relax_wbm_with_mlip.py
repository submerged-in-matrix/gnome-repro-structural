"""Pre-relax WBM initial structures with a pretrained MLIP (CHGNet or
MACE-MP-0) before scoring with the GNoME structural GNN ensemble.

Motivation: Model A (the structural GNN) was trained only on relaxed MP
structures (RC1 in project notes). WBM's initial_structure entries are
unrelaxed, causing a systematic domain-shift bias. Instead of retraining
Model A, this script relaxes the input first with a pretrained, open MLIP,
then hands the RELAXED structure to the existing, already-trained ensemble
unchanged.

This is a pipeline pre-processing step ONLY. It does not modify, retrain,
or touch Model A in any way.

Two MLIPs supported via --mlip {chgnet,mace}. Each gets its own tagged
output paths (relaxed structures, fail log, eval run dirs), so running
both never collides and each resumes independently.

NOTE on MACE-MP-0: the calculator-loading call below follows the commonly
documented mace-torch pattern (mace.calculators.mace_mp), but this has NOT
been verified against the specific mace-torch version you install — the
API has changed across releases. Check the actual installed package's
docs/examples if this call errors, rather than assuming it is correct.

Checkpointing: relaxed structures are written incrementally to --out.
On restart, structures already present in --out are skipped, so an
interrupted run (e.g. a Kaggle session timeout) can be resumed by
re-running the same command.

Chaining: when --limit is NOT set (i.e. a full run), this script
automatically calls, as subprocesses, after relaxation completes:
    1. eval_wbm_ensemble.py --wbm-structs <relaxed>   (6-seed median ensemble)
    2. eval_wbm.py --checkpoint runs/ensemble/seed_0/best.pt
                    --wbm-structs <relaxed> --aggregator min
       (seed_0 chosen because it has the lowest individual WBM MAE in
       metrics_wbm_ensemble.json: 106.5891 meV, vs 109-113 for others)
    3. f1_wbm.py against both resulting predictions_wbm.csv files
When --limit IS set (sanity-check runs), chaining never happens —
inspect the relaxed output manually.

Usage
-----
    # Quick sanity check, no chaining
    python scripts/relax_wbm_with_mlip.py --mlip chgnet --limit 15
    python scripts/relax_wbm_with_mlip.py --mlip mace   --limit 15

    # Full run (resumable), auto-chains to eval + F1 scripts at the end
    python scripts/relax_wbm_with_mlip.py --mlip chgnet
    python scripts/relax_wbm_with_mlip.py --mlip mace

    # Full run, skip the auto-chain (relax only)
    python scripts/relax_wbm_with_mlip.py --mlip chgnet --no-chain-eval
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for eval_wbm import

from eval_wbm import load_wbm_structures  # reused unchanged, no duplication

# Best single seed by WBM MAE (see metrics_wbm_ensemble.json: seed_0 = 106.5891 meV,

BEST_SEED_CHECKPOINT = _REPO_ROOT / "runs" / "ensemble" / "seed_0" / "best.pt"


def load_calculator(mlip: str, device: str):
    """Load the requested pretrained MLIP as an ASE calculator."""
    if mlip == "chgnet":
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        chgnet = CHGNet.load()
        calc = CHGNetCalculator(model=chgnet, use_device=device)
        print("CHGNet loaded (pretrained, MPtrj).")
        return calc
    elif mlip == "mace":
        # UNVERIFIED against your installed mace-torch version — check this
        # call against the actual package docs/examples if it errors.
        from mace.calculators import mace_mp
        calc = mace_mp(model="medium", device=device, default_dtype="float32")
        print("MACE-MP-0 loaded (pretrained, medium).")
        return calc
    else:
        raise ValueError(f"Unknown --mlip: {mlip}")


def relax_one(structure: Structure, calc, fmax: float, max_steps: int) -> tuple[Structure | None, bool]:
    """Relax a single pymatgen Structure with the given ASE calculator.

    Returns (relaxed_structure, converged). relaxed_structure is None if
    conversion or relaxation raised an exception (skipped, not silently
    substituted with the original).
    """
    from ase.optimize import FIRE

    try:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = calc
        dyn = FIRE(atoms, logfile=None)
        converged = dyn.run(fmax=fmax, steps=max_steps)
        relaxed = AseAtomsAdaptor.get_structure(atoms)
        return relaxed, bool(converged)
    except Exception as e:
        print(f"  WARNING: relaxation failed ({e})", file=sys.stderr)
        return None, False


def load_existing_output(out_path: Path) -> tuple[dict, set[str]]:
    """Load an existing partial output file, if present, for resume support."""
    if not out_path.exists():
        return {"material_id": {}, "initial_structure": {}}, set()

    with open(out_path) as f:
        data = json.load(f)
    done_ids = set(data["material_id"].values())
    print(f"Resuming: {len(done_ids):,} structures already relaxed in {out_path}")
    return data, done_ids


def _json_default(o):
    """Fallback for json.dump: convert numpy scalar types to native Python
    types. Only triggered for types the default encoder cannot already
    handle (e.g. numpy.float32 from CHGNet's final_magmom) -- harmless and
    inert on environments where this type mismatch does not occur."""
    if type(o).__name__ in ("float32", "float64"):
        return float(o)
    if type(o).__name__ in ("int32", "int64"):
        return int(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def save_output(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, default=_json_default)
    tmp_path.replace(out_path)  # atomic-ish replace, avoids truncated file on crash mid-write


def main():
    parser = argparse.ArgumentParser(description="Pre-relax WBM structures with a pretrained MLIP")
    parser.add_argument("--mlip", type=str, default="chgnet", choices=["chgnet", "mace"],
                         help="Which pretrained MLIP to use for relaxation")
    parser.add_argument("--wbm-structs", type=Path,
                         default=_REPO_ROOT / "data" / "raw" / "2022-10-19-wbm-init-structs.json")
    parser.add_argument("--out", type=Path, default=None,
                         help="Default: data/raw/wbm-init-structs-<mlip>-relaxed.json")
    parser.add_argument("--fail-log", type=Path, default=None,
                         help="Default: data/raw/wbm-<mlip>-relax-failures.json")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only relax first N structures (sanity check; disables auto-chain)")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold, eV/A")
    parser.add_argument("--max-steps", type=int, default=200, help="Max relaxation steps per structure")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                         help="Flush partial output to disk every N structures")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-chain-eval", action="store_true",
                         help="Skip auto-chaining to eval scripts even on a full run")
    parser.add_argument("--ensemble-out-dir", type=Path, default=None,
                         help="Default: runs/ensemble_<mlip>_relaxed")
    parser.add_argument("--seed0-out-dir", type=Path, default=None,
                         help="Default: runs/seed0_<mlip>_relaxed")
    parser.add_argument("--wbm-summary", type=Path,
                         default=_REPO_ROOT / "data" / "raw" / "wbm-summary.csv.gz")
    args = parser.parse_args()

    # --- Resolve MLIP-tagged default paths (keeps chgnet/mace runs fully independent) ---
    if args.out is None:
        args.out = _REPO_ROOT / "data" / "raw" / f"wbm-init-structs-{args.mlip}-relaxed.json"
    if args.fail_log is None:
        args.fail_log = _REPO_ROOT / "data" / "raw" / f"wbm-{args.mlip}-relax-failures.json"
    if args.ensemble_out_dir is None:
        args.ensemble_out_dir = _REPO_ROOT / "runs" / f"ensemble_{args.mlip}_relaxed"
    if args.seed0_out_dir is None:
        args.seed0_out_dir = _REPO_ROOT / "runs" / f"seed0_{args.mlip}_relaxed"

    device = args.device
    print(f"Device: {device}")
    print(f"MLIP: {args.mlip}")

    # --- Load MLIP calculator ---
    calc = load_calculator(args.mlip, device)

    # --- Load WBM initial structures ---
    ids, structures = load_wbm_structures(args.wbm_structs)
    if args.limit is not None:
        ids        = ids[:args.limit]
        structures = structures[:args.limit]
        print(f"  Limiting to first {args.limit} structures (--limit flag). Auto-chain disabled.")

    # --- Resume support ---
    out_data, done_ids = load_existing_output(args.out)
    fail_log: list[dict] = []
    if args.fail_log.exists():
        with open(args.fail_log) as f:
            fail_log = json.load(f)

    remaining = [(mid, s) for mid, s in zip(ids, structures) if mid not in done_ids]
    print(f"  {len(done_ids):,} already done, {len(remaining):,} remaining.")

    n_converged = 0
    n_failed    = 0
    n_nonconverged = 0
    t0 = time.time()

    next_idx = len(out_data["material_id"])  # continue string-index numbering

    for i, (mid, struct) in enumerate(remaining):
        relaxed, converged = relax_one(struct, calc, args.fmax, args.max_steps)

        if relaxed is None:
            n_failed += 1
            fail_log.append({"material_id": mid, "reason": "exception"})
            continue

        if not converged:
            n_nonconverged += 1
            fail_log.append({"material_id": mid, "reason": "max_steps_reached"})
        else:
            n_converged += 1

        idx_str = str(next_idx)
        out_data["material_id"][idx_str] = mid
        out_data["initial_structure"][idx_str] = relaxed.as_dict()
        next_idx += 1

        if (i + 1) % args.checkpoint_every == 0:
            save_output(out_data, args.out)
            with open(args.fail_log, "w") as f:
                json.dump(fail_log, f, indent=2)
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            eta       = (len(remaining) - i - 1) / rate
            print(f"  [{i+1:>7,}/{len(remaining):,}]  {rate:.2f} structs/s  "
                  f"ETA {eta/60:.1f} min  "
                  f"(converged={n_converged}, non-converged={n_nonconverged}, failed={n_failed})")

    # Final flush
    save_output(out_data, args.out)
    with open(args.fail_log, "w") as f:
        json.dump(fail_log, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone. {len(remaining):,} structures processed in {elapsed/60:.1f} min.")
    print(f"  Converged      : {n_converged:,}")
    print(f"  Non-converged  : {n_nonconverged:,} (hit --max-steps={args.max_steps})")
    print(f"  Failed         : {n_failed:,}")
    print(f"Relaxed structures -> {args.out}")
    print(f"Failure log        -> {args.fail_log}")

    # --- Auto-chain to eval scripts (full run only) ---
    if args.limit is not None:
        print("\n--limit was set: skipping auto-chain. Inspect the relaxed output manually.")
        return
    if args.no_chain_eval:
        print("\n--no-chain-eval set: skipping auto-chain.")
        return

    print(f"\n{'='*60}")
    print("  Auto-chaining to eval scripts (full run detected)")
    print(f"{'='*60}")

    # 1. Full 6-seed median ensemble
    print("\n[1/3] Running 6-seed median ensemble eval on relaxed structures ...")
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parent / "eval_wbm_ensemble.py"),
        "--wbm-structs", str(args.out),
        "--wbm-summary", str(args.wbm_summary),
        "--ensemble-dir", str(args.ensemble_out_dir),
        "--device", device,
    ], check=True)

    # 2. Best single seed (seed_0, lowest individual WBM MAE), min-TTA
    print("\n[2/3] Running best-single-seed (seed_0, min-TTA) eval on relaxed structures ...")
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parent / "eval_wbm.py"),
        "--checkpoint", str(BEST_SEED_CHECKPOINT),
        "--wbm-structs", str(args.out),
        "--wbm-summary", str(args.wbm_summary),
        "--out-dir", str(args.seed0_out_dir),
        "--aggregator", "min",
        "--device", device,
    ], check=True)

    # 3. F1 scoring for both (using f1_wbm.py's real CLI: --predictions, --wbm-summary, --out-dir)
    print("\n[3/3] Running F1 scoring on both relaxed-ensemble and relaxed-seed0 predictions ...")
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parent / "f1_wbm.py"),
        "--predictions", str(args.ensemble_out_dir / "predictions_wbm.csv"),
        "--wbm-summary", str(args.wbm_summary),
        "--out-dir", str(args.ensemble_out_dir),
    ], check=True)
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parent / "f1_wbm.py"),
        "--predictions", str(args.seed0_out_dir / "predictions_wbm.csv"),
        "--wbm-summary", str(args.wbm_summary),
        "--out-dir", str(args.seed0_out_dir),
    ], check=True)

    # --- Side-by-side summary ---
    print(f"\n{'='*66}")
    print(f"  Summary: unrelaxed vs {args.mlip}-relaxed")
    print(f"{'='*66}")

    def _load_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    unrelaxed_ens_metrics = _load_json(_REPO_ROOT / "runs" / "ensemble" / "metrics_wbm_ensemble.json")
    unrelaxed_ens_f1      = _load_json(_REPO_ROOT / "runs" / "ensemble" / "f1_wbm.json")
    relaxed_ens_metrics   = _load_json(args.ensemble_out_dir / "metrics_wbm_ensemble.json")
    relaxed_ens_f1        = _load_json(args.ensemble_out_dir / "f1_wbm.json")
    relaxed_seed0_metrics = _load_json(args.seed0_out_dir / "metrics_wbm.json")
    relaxed_seed0_f1      = _load_json(args.seed0_out_dir / "f1_wbm.json")

    rows = [
        ("Ensemble (unrelaxed, median)",
         unrelaxed_ens_metrics["ensemble"] if unrelaxed_ens_metrics else None,
         unrelaxed_ens_f1["strategy_raw"] if unrelaxed_ens_f1 else None),
        (f"Ensemble ({args.mlip}-relaxed, median)",
         relaxed_ens_metrics["ensemble"] if relaxed_ens_metrics else None,
         relaxed_ens_f1["strategy_raw"] if relaxed_ens_f1 else None),
        (f"Seed_0 ({args.mlip}-relaxed, min-TTA)",
         relaxed_seed0_metrics,
         relaxed_seed0_f1["strategy_raw"] if relaxed_seed0_f1 else None),
    ]
    for label, m, f1 in rows:
        if m is None:
            print(f"  {label:<38} not available")
            continue
        f1_str = f"F1={f1['f1']:.4f} P={f1['precision']:.4f} R={f1['recall']:.4f}" if f1 else "F1=n/a"
        print(f"  {label:<38} MAE={m['mae_meV_per_atom']:.2f}  "
              f"RMSE={m['rmse_meV_per_atom']:.2f}  bias={m['bias_meV_per_atom']:+.2f}  {f1_str}")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
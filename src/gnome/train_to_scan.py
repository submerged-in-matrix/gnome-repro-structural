"""Shared harness for the anchored coordinate hyperparameter search.

Provides a no-EMA training function over the stratified subset and the ledger
logic that anchors each axis. Each axis launcher (scan_1_depth.py, etc.) is a
thin wrapper that calls run_axis(); all training, anchoring, and bookkeeping
live here so the four launchers stay free of duplicated logic.

Search strategy (anchored coordinate search):
- One axis is scanned at a time in a fixed order.
- The base value of every axis is always carried as an anchor; a new winner is
  added alongside the base rather than replacing it, so dependencies between
  axes are not hidden by a greedy collapse.
- The anchor frontier for an axis is the Cartesian product of the {base, winner}
  sets of all already-resolved axes; a winner equal to its base adds no second
  anchor, which keeps the frontier from doubling when an axis does not help.

Decision metric: best stratified-test MAE over the run (lower is better). WBM is
not used during the search; it is reserved for the locked architecture only.

DDP launch (torchrun):
    torchrun --nproc_per_node=2 scripts/scan_N_<axis>.py
Single-GPU fallback (no dist.init_process_group required):
    python scripts/scan_N_<axis>.py
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from gnome.model_to_scan import GNoMEStructural


# Stratified subset on Kaggle; read-only input mount. All three files
# (train/test/stats_stratified) live in the same dataset folder.
DATA_DIR = Path("/kaggle/input/datasets/saidul1991/gnome-stratified")
TRAIN_PT = DATA_DIR / "train_stratified.pt"
TEST_PT = DATA_DIR / "test_stratified.pt"
STATS_PT = DATA_DIR / "stats_stratified.pt"

# Writable Kaggle scratch holds per-run outputs and the persistent ledger.
RUNS_DIR = Path("/kaggle/working/scan_runs")
LEDGER_PATH = RUNS_DIR / "scan_ledger.json"

# Fixed search order; an axis may run only after all earlier axes are resolved.
ORDER = ["n_hidden", "activation", "n_layers", "lr"]

# Base value of every axis reproduces the current production architecture, so
# the base anchor is always the existing model and the search builds outward.
BASE_DEFAULTS = {
    "n_hidden": 1,
    "activation": "silu",
    "n_layers": 3,
    "lr": 5.5e-4,
}

# Canonical value list per axis; every list includes its own base value so the
# base anchor is re-evaluated within each axis as a fair comparison point.
AXIS_VALUES = {
    "n_hidden": [1, 2, 3, 4, 6],
    "activation": ["silu", "gelu", "mish"],
    "n_layers": [3, 4, 5],
    "lr": [1e-4, 3e-4, 5.5e-4, 1e-3],
}


# ── DDP helpers ─────────────────────────────────────────────────────────────────

def _is_main() -> bool:
    """True when not in DDP or when this process is rank-0."""
    return not dist.is_initialized() or dist.get_rank() == 0


def _local_rank() -> int:
    """Local GPU index; falls back to 0 when not launched via torchrun."""
    return int(os.environ.get("LOCAL_RANK", 0))


# ── Reproducibility ─────────────────────────────────────────────────────────────

def _seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices).

    cudnn.deterministic eliminates non-deterministic scatter/gather kernels
    used in PyG message passing. cudnn.benchmark is disabled so the same
    convolution algorithm is selected on every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Config ──────────────────────────────────────────────────────────────────────

@dataclass
class ScanConfig:
    """One training run's full hyperparameter set and fixed search settings."""

    # Searchable axes; defaults are the base anchor.
    n_hidden: int = BASE_DEFAULTS["n_hidden"]
    activation: str = BASE_DEFAULTS["activation"]
    n_layers: int = BASE_DEFAULTS["n_layers"]
    lr: float = BASE_DEFAULTS["lr"]

    # Fixed across the whole search; hidden_dim stays at the paper value.
    hidden_dim: int = 256
    use_adj_norm: bool = True

    # Fixed training settings per the locked plan: no EMA, seed 0.
    # Epochs cut from 500 to 200 (2026-06-17): Stage B ablation reached
    # MAE=24.4 meV/atom at 200 epochs; 500 is excess compute for search.
    # Effective batch = batch_size * accum_steps = 256, matching Stage A.
    epochs: int = 200
    batch_size: int = 128
    accum_steps: int = 2
    lr_end_factor: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0

    # System.
    num_workers: int = 0
    log_every: int = 25
    # Resume checkpoint saved every resume_every epochs; a crash loses at most
    # resume_every epochs of work. summary.json handles fully-completed runs;
    # resume.pt handles mid-run restarts.
    resume_every: int = 10

    run_name: str = "scan_run"
    runs_dir: str = str(RUNS_DIR)


# ── Training ────────────────────────────────────────────────────────────────────

def train_one(cfg: ScanConfig) -> dict:
    """Train one model with no EMA and return its summary including best MAE.

    Mirrors the no-EMA loop in train.py but builds the parameterised model,
    reads the stratified subset, and uses gradient accumulation for an effective
    batch of 256. Early stopping is omitted so every run sees the full epoch
    budget and best-MAE numbers are comparable across runs.

    Compatible with single-GPU (python) and multi-GPU DDP (torchrun) launches.
    In DDP mode each rank trains on a DistributedSampler slice; metrics are
    all-reduced across ranks so logged values are global. Only rank-0 writes
    checkpoints, history, and summary.
    """
    _seed_all(cfg.seed)
    ddp = dist.is_initialized()
    rank = dist.get_rank() if ddp else 0
    world_size = dist.get_world_size() if ddp else 1
    device = torch.device(
        f"cuda:{_local_rank()}" if torch.cuda.is_available() else "cpu"
    )

    run_dir = Path(cfg.runs_dir) / cfg.run_name
    if _is_main():
        run_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        # All ranks wait for rank-0 to create the run directory before proceeding.
        dist.barrier()

    train_data = torch.load(TRAIN_PT, weights_only=False)
    test_data  = torch.load(TEST_PT,  weights_only=False)
    stats      = torch.load(STATS_PT, weights_only=False)

    if ddp:
        train_sampler = DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True,
        )
        test_sampler = DistributedSampler(
            test_data, num_replicas=world_size, rank=rank, shuffle=False,
        )
    else:
        train_sampler = test_sampler = None

    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False,
        sampler=test_sampler,
        num_workers=cfg.num_workers,
    )

    model = GNoMEStructural(
        avg_adjacency=stats["avg_adjacency"],
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        use_adj_norm=cfg.use_adj_norm,
        n_hidden=cfg.n_hidden,
        activation=cfg.activation,
    ).to(device)
    # n_params computed before DDP wrapping so it reflects the actual model size.
    n_params = sum(p.numel() for p in model.parameters())

    if ddp:
        model = DDP(model, device_ids=[_local_rank()])

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=cfg.lr_end_factor,
        total_iters=cfg.epochs,
    )

    mu    = torch.tensor(stats["label_mean"], device=device)
    sigma = torch.tensor(stats["label_std"],  device=device)

    best_mae  = float("inf")
    history   = []
    start_epoch = 0

    # Resume from a mid-run checkpoint if one exists. rank-0 wrote it; all
    # ranks load from the same path (shared filesystem on a single machine).
    resume_path = run_dir / "resume.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model = model.module if ddp else model
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_mae    = ckpt["best_mae"]
        history     = ckpt["history"]
        if _is_main():
            print(f"  [{cfg.run_name}] resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        # DistributedSampler seeds its shuffle from the epoch index; must be
        # updated each epoch so successive epochs see different orderings.
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss = _train_epoch(
            model, train_loader, optimizer, mu, sigma, device,
            cfg.grad_clip, cfg.accum_steps, ddp,
        )
        test_mae = _eval_epoch(model, test_loader, mu, sigma, device, ddp)
        scheduler.step()
        wall = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss_norm": train_loss,
            "test_mae_eV_per_atom": test_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "wall_seconds": wall,
        })

        if _is_main():
            # Log at every log_every epoch and always at the final epoch so
            # the last data point is never silently missing from the output.
            if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
                print(
                    f"  [{cfg.run_name}] epoch {epoch:>3d}  "
                    f"test_MAE {test_mae * 1000:>6.1f} meV/atom  "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}  ({wall:.1f}s)"
                )

            # Best-MAE checkpoint; state stripped of DDP wrapper so it loads
            # directly into a plain GNoMEStructural without key-prefix surgery.
            if test_mae < best_mae:
                best_mae = test_mae
                state = model.module.state_dict() if ddp else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state": state,
                    "config": asdict(cfg),
                    "stats": stats,
                    "test_mae": test_mae,
                }, run_dir / "best.pt")

            # Resume checkpoint written periodically; overwritten each time to
            # keep disk use constant. Captures full training state so a restart
            # is identical to an uninterrupted run from this epoch forward.
            if (epoch + 1) % cfg.resume_every == 0:
                state = model.module.state_dict() if ddp else model.state_dict()
                torch.save({
                    "epoch": epoch,
                    "model_state": state,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_mae": best_mae,
                    "history": history,
                }, resume_path)

    # All ranks must finish training before rank-0 writes summary.json,
    # which is the completion marker used by run_axis to skip finished runs.
    if ddp:
        dist.barrier()

    if _is_main():
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        summary = {
            "run_name":           cfg.run_name,
            "n_hidden":           cfg.n_hidden,
            "activation":         cfg.activation,
            "n_layers":           cfg.n_layers,
            "lr":                 cfg.lr,
            "best_test_mae_meV":  best_mae * 1000,
            "n_params":           n_params,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Remove resume.pt once the run completes; summary.json is now the
        # completion marker and resume.pt is no longer needed.
        if resume_path.exists():
            resume_path.unlink()

    # All ranks wait for rank-0 to write summary.json before moving to the
    # next run in run_axis, so every rank sees a consistent filesystem state.
    if ddp:
        dist.barrier()

    # Constructed on all ranks; only rank-0 uses it in run_axis, but the
    # values are identical because test_mae was all-reduced during training.
    return {
        "run_name":           cfg.run_name,
        "n_hidden":           cfg.n_hidden,
        "activation":         cfg.activation,
        "n_layers":           cfg.n_layers,
        "lr":                 cfg.lr,
        "best_test_mae_meV":  best_mae * 1000,
        "n_params":           n_params,
    }


def _train_epoch(
    model, loader, optimizer, mu, sigma, device,
    grad_clip, accum_steps, ddp=False,
):
    """One training epoch with gradient accumulation; returns mean normalised loss.

    Loss is divided by accum_steps so accumulated gradients equal a single pass
    over the full effective batch; the optimiser steps every accum_steps batches.
    In DDP mode, local loss sums are all-reduced for accurate global logging.
    """
    model.train()
    total, count = 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        target_norm = (batch.y - mu) / sigma
        pred_norm   = model(batch)
        loss = (pred_norm - target_norm).abs().mean() / accum_steps
        loss.backward()

        total += loss.item() * accum_steps * batch.num_graphs
        count += batch.num_graphs

        # Step on the accum boundary or at the final batch so no gradient is lost.
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

    if ddp:
        # Reduce local sums to produce the global training loss for logging.
        t = torch.tensor([total, float(count)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t[0] / t[1]).item()
    return total / count


def _eval_epoch(model, loader, mu, sigma, device, ddp=False):
    """One evaluation pass; returns MAE in eV/atom on the stratified test set.

    Each rank evaluates its DistributedSampler slice; absolute errors are
    all-reduced to produce the global MAE, matching the single-GPU metric.
    """
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch) * sigma + mu
            total += (pred - batch.y).abs().sum().item()
            count += batch.num_graphs

    if ddp:
        t = torch.tensor([total, float(count)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t[0] / t[1]).item()
    return total / count


# ── Ledger ──────────────────────────────────────────────────────────────────────

def load_ledger() -> dict:
    """Return the persistent ledger, initialising a fresh one if none exists."""
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            return json.load(f)
    return {"resolved": {}, "pending": list(ORDER), "runs": []}


def save_ledger(ledger: dict) -> None:
    """Persist the ledger to writable scratch so the next axis can read it."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def _anchor_configs(ledger: dict) -> list[dict]:
    """Build the anchor frontier as the product of resolved {base, winner} sets.

    Axes not yet resolved take their base value; the result is the list of
    partial hyperparameter dicts onto which the current axis is swept.
    """
    resolved = ledger["resolved"]
    keys = [k for k in ORDER if k in resolved]
    if not keys:
        return [dict(BASE_DEFAULTS)]

    anchors = []
    for combo in product(*(resolved[k] for k in keys)):
        anchor = dict(BASE_DEFAULTS)
        anchor.update(dict(zip(keys, combo)))
        anchors.append(anchor)
    return anchors


def _winner_of(axis: str, results: list[dict]) -> object:
    """Return the axis value from the single lowest-MAE run."""
    best = min(results, key=lambda r: r["best_test_mae_meV"])
    return best[axis]


def run_axis(axis: str) -> dict:
    """Scan one axis across the current anchor frontier and update the ledger.

    Trains anchors x AXIS_VALUES[axis] runs, records every run, selects the
    winning axis value, and appends {base, winner} to the resolved set (the base
    alone if no improvement). Returns the updated ledger.

    DDP-safe: I/O is gated by _is_main(); all ranks participate in each
    train_one call; barriers inside train_one prevent rank interleaving.
    All ranks share the same filesystem so skip/run decisions are identical
    across ranks without any broadcast.
    """
    if axis not in AXIS_VALUES:
        raise ValueError(f"unknown axis '{axis}'; choose from {list(AXIS_VALUES)}")

    ledger = load_ledger()

    earlier = ORDER[: ORDER.index(axis)]
    missing = [a for a in earlier if a not in ledger["resolved"]]
    if missing:
        raise RuntimeError(
            f"axis '{axis}' cannot run before resolving {missing}; "
            f"run the earlier axis launchers first"
        )
    if axis in ledger["resolved"]:
        raise RuntimeError(f"axis '{axis}' already resolved; nothing to do")

    anchors = _anchor_configs(ledger)
    values  = AXIS_VALUES[axis]
    if _is_main():
        print(
            f"=== axis '{axis}': {len(anchors)} anchor(s) x {len(values)} value(s) "
            f"= {len(anchors) * len(values)} run(s) ==="
        )

    axis_results = []
    for anchor in anchors:
        for val in values:
            cfg_kwargs         = dict(anchor)
            cfg_kwargs[axis]   = val
            # Run name encodes only the hyperparameter config values, not the
            # scanning axis or anchor index; the four values are jointly unique
            # across the whole search so identical configs reuse the same folder.
            run_name = (
                f"h{cfg_kwargs['n_hidden']}"
                f"_{cfg_kwargs['activation']}"
                f"_L{cfg_kwargs['n_layers']}"
                f"_lr{cfg_kwargs['lr']:g}"
            )
            cfg          = ScanConfig(run_name=run_name, **cfg_kwargs)
            summary_path = Path(cfg.runs_dir) / run_name / "summary.json"

            # All ranks see the same filesystem; decision is identical without
            # broadcasting. Reuse completed run if summary.json is valid.
            if summary_path.exists():
                try:
                    cached = json.load(open(summary_path))
                    if "best_test_mae_meV" in cached:
                        if _is_main():
                            print(f"--- {run_name}: reuse cached result ---")
                        axis_results.append(cached)
                        continue
                except (json.JSONDecodeError, OSError):
                    pass

            if _is_main():
                print(f"--- {run_name} ---")
            summary = train_one(cfg)
            axis_results.append(summary)

    base             = BASE_DEFAULTS[axis]
    winner           = _winner_of(axis, axis_results)
    resolved_values  = [base] if winner == base else [base, winner]

    if _is_main():
        ledger["resolved"][axis] = resolved_values
        ledger["pending"]        = [a for a in ledger["pending"] if a != axis]
        ledger["runs"].extend(axis_results)
        save_ledger(ledger)
        print(
            f"=== axis '{axis}' done: winner={winner} "
            f"(base={base}) -> resolved={resolved_values} ==="
        )
        _print_axis_table(axis, axis_results)

    return ledger


def _print_axis_table(axis: str, results: list[dict]) -> None:
    """Print a compact MAE table for the axis, sorted best first."""
    print(f"\n{axis} results (best MAE first):")
    print(f"{'run_name':<40} {'best_MAE_meV':>12}")
    for r in sorted(results, key=lambda r: r["best_test_mae_meV"]):
        print(f"{r['run_name']:<40} {r['best_test_mae_meV']:>12.2f}")
    print()
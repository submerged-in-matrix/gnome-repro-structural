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
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel as PyGDataParallel

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
    # Epochs cut from 500 to 200 (2026-06-17): Stage B ablation already reached
    # MAE=24.4 meV/atom at 200 epochs, so 500 is excess compute for search purposes.
    # Effective batch = batch_size * accum_steps = 256, matching Stage A.
    epochs: int = 200
    batch_size: int = 128
    accum_steps: int = 2
    lr_end_factor: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0

    # System.
    device: str = "cuda"
    num_workers: int = 0
    log_every: int = 25
    # When True and >1 GPU is visible, uses PyG DataParallel across all devices.
    # Mathematically equivalent to single-GPU (no batch-dependent layers in the
    # model; loss is a per-graph mean), verified 2026-06-17.
    parallel: bool = True

    run_name: str = "scan_run"
    runs_dir: str = str(RUNS_DIR)


def train_one(cfg: ScanConfig) -> dict:
    """Train one model with no EMA and return its summary including best MAE.

    Mirrors the no-EMA loop in train.py but builds the parameterised model,
    reads the stratified subset, and uses gradient accumulation for an effective
    batch of 256; early stopping is omitted so every run sees the full epoch
    budget and the best-MAE numbers stay comparable across runs.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    parallel = cfg.parallel and torch.cuda.device_count() > 1

    run_dir = Path(cfg.runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Stratified stats are required so normalisation matches the subset; using
    # full-dataset stats here would reintroduce the known normalisation bug.
    train_data = torch.load(TRAIN_PT, weights_only=False)
    test_data = torch.load(TEST_PT, weights_only=False)
    stats = torch.load(STATS_PT, weights_only=False)

    if parallel:
        # DataListLoader keeps each Data object separate so PyG DataParallel
        # can split the list across GPUs by graph, not by a pre-batched tensor.
        train_loader = DataListLoader(
            train_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers,
        )
        test_loader = DataListLoader(
            test_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers,
        )
        test_loader = DataLoader(
            test_data, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers,
        )

    # Parameterised model carries the two new axes plus the existing n_layers.
    model = GNoMEStructural(
        avg_adjacency=stats["avg_adjacency"],
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        use_adj_norm=cfg.use_adj_norm,
        n_hidden=cfg.n_hidden,
        activation=cfg.activation,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if parallel:
        model = PyGDataParallel(model)

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=cfg.lr_end_factor,
        total_iters=cfg.epochs,
    )

    mu = torch.tensor(stats["label_mean"], device=device)
    sigma = torch.tensor(stats["label_std"], device=device)

    best_mae = float("inf")
    history = []

    for epoch in range(cfg.epochs):
        t0 = time.time()
        if parallel:
            train_loss = _train_epoch_parallel(
                model, train_loader, optimizer, mu, sigma,
                cfg.grad_clip, cfg.accum_steps,
            )
            test_mae = _eval_epoch_parallel(model, test_loader, mu, sigma)
        else:
            train_loss = _train_epoch(
                model, train_loader, optimizer, mu, sigma, device,
                cfg.grad_clip, cfg.accum_steps,
            )
            test_mae = _eval_epoch(model, test_loader, mu, sigma, device)
        scheduler.step()
        wall = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss_norm": train_loss,
            "test_mae_eV_per_atom": test_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "wall_seconds": wall,
        })

        if epoch % cfg.log_every == 0:
            print(
                f"  [{cfg.run_name}] epoch {epoch:>3d}  "
                f"test_MAE {test_mae * 1000:>6.1f} meV/atom  "
                f"lr {optimizer.param_groups[0]['lr']:.2e}  ({wall:.1f}s)"
            )

        # Best test MAE over the run is the search decision metric; the best
        # checkpoint is kept so a winning architecture can be reloaded later.
        # model.module is used under DataParallel so the saved state_dict
        # loads directly into a plain GNoMEStructural later, with no "module."
        # key prefix to strip.
        if test_mae < best_mae:
            best_mae = test_mae
            state = model.module.state_dict() if parallel else model.state_dict()
            torch.save({
                "epoch": epoch,
                "model_state": state,
                "config": asdict(cfg),
                "stats": stats,
                "test_mae": test_mae,
            }, run_dir / "best.pt")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "run_name": cfg.run_name,
        "n_hidden": cfg.n_hidden,
        "activation": cfg.activation,
        "n_layers": cfg.n_layers,
        "lr": cfg.lr,
        "best_test_mae_meV": best_mae * 1000,
        "n_params": n_params,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _train_epoch(model, loader, optimizer, mu, sigma, device, grad_clip, accum_steps):
    """One training epoch with gradient accumulation; returns mean normalised loss.

    Loss is divided by accum_steps so accumulated gradients equal a single pass
    over the full effective batch; the optimiser steps every accum_steps batches.
    """
    model.train()
    total, count = 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        target_norm = (batch.y - mu) / sigma

        pred_norm = model(batch)
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

    return total / count


def _train_epoch_parallel(model, loader, optimizer, mu, sigma, grad_clip, accum_steps):
    """Multi-GPU train epoch using PyG DataParallel.

    DataParallel gathers outputs back to the primary device in the same order
    as the input data_list, so targets are obtained by concatenating data.y
    from that same list directly; no manual re-splitting is needed.
    """
    model.train()
    total, count = 0.0, 0
    optimizer.zero_grad()

    for step, data_list in enumerate(loader):
        target = torch.cat([d.y for d in data_list])
        pred_norm = model(data_list)
        target_norm = (target.to(pred_norm.device) - mu) / sigma
        loss = (pred_norm - target_norm).abs().mean() / accum_steps
        loss.backward()

        n_graphs = len(data_list)
        total += loss.item() * accum_steps * n_graphs
        count += n_graphs

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

    return total / count


def _eval_epoch_parallel(model, loader, mu, sigma):
    """Multi-GPU eval epoch; returns MAE in eV/atom, mirrors _eval_epoch."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for data_list in loader:
            target = torch.cat([d.y for d in data_list])
            pred = model(data_list) * sigma + mu
            total += (pred - target.to(pred.device)).abs().sum().item()
            count += len(data_list)
    return total / count


def _eval_epoch(model, loader, mu, sigma, device):
    """One evaluation pass; returns MAE in eV/atom on the stratified test set."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # De-normalise to physical units before MAE so the metric is in
            # eV/atom and directly comparable across runs.
            pred = model(batch) * sigma + mu
            total += (pred - batch.y).abs().sum().item()
            count += batch.num_graphs
    return total / count


# ── Ledger ──────────────────────────────────────────────────────────────────────

def load_ledger() -> dict:
    """Return the persistent ledger, initialising a fresh one if none exists."""
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH) as f:
            return json.load(f)
    # A fresh ledger has nothing resolved and every axis pending in fixed order.
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
        # No axis resolved yet, so the only anchor is the full base model.
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
    """
    if axis not in AXIS_VALUES:
        raise ValueError(f"unknown axis '{axis}'; choose from {list(AXIS_VALUES)}")

    ledger = load_ledger()

    # Ordering guard: every earlier axis must be resolved so anchors are complete.
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
    values = AXIS_VALUES[axis]
    print(
        f"=== axis '{axis}': {len(anchors)} anchor(s) x {len(values)} value(s) "
        f"= {len(anchors) * len(values)} run(s) ==="
    )

    axis_results = []
    for ai, anchor in enumerate(anchors):
        for val in values:
            cfg_kwargs = dict(anchor)
            cfg_kwargs[axis] = val
            run_name = (
                f"{axis}__h{cfg_kwargs['n_hidden']}"
                f"_{cfg_kwargs['activation']}"
                f"_L{cfg_kwargs['n_layers']}"
                f"_lr{cfg_kwargs['lr']:g}"
                f"__a{ai}"
            )
            cfg = ScanConfig(run_name=run_name, **cfg_kwargs)
            summary_path = Path(cfg.runs_dir) / run_name / "summary.json"
            # Reuse a completed run so an interrupted axis resumes without
            # repeating finished work; a missing or unreadable file retrains.
            if summary_path.exists():
                try:
                    cached = json.load(open(summary_path))
                    if "best_test_mae_meV" in cached:
                        print(f"--- {run_name}: reuse cached result ---")
                        axis_results.append(cached)
                        continue
                except (json.JSONDecodeError, OSError):
                    pass
            print(f"--- {run_name} ---")
            summary = train_one(cfg)
            axis_results.append(summary)

    # Winner is the axis value of the lowest-MAE run; base is always carried so
    # the resolved set is [base] or [base, winner] with no duplication.
    base = BASE_DEFAULTS[axis]
    winner = _winner_of(axis, axis_results)
    resolved_values = [base] if winner == base else [base, winner]

    ledger["resolved"][axis] = resolved_values
    ledger["pending"] = [a for a in ledger["pending"] if a != axis]
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
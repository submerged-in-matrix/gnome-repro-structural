"""Training loop for GNoME structural GNN.

Reads cached PyG Data objects from data/processed/, normalizes labels
using cached training-set statistics, trains with Adam + linear LR decay,
saves best-val checkpoint and per-epoch metrics.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch_geometric.loader import DataLoader

from gnome.model import GNoMEStructural


@dataclass
class TrainConfig:
    # Data paths
    data_dir: str = "./data"
    runs_dir: str = "./runs"
    run_name: str = "default"

    # Architecture
    hidden_dim: int = 256
    n_layers: int = 3

    # Optimization
    epochs: int = 200
    batch_size: int = 128
    lr: float = 5.5e-4
    lr_end_factor: float = 0.1   # final lr = initial * lr_end_factor
    grad_clip: float = 1.0
    early_stop_patience: int = 30

    # Subset for smoke testing; None = full dataset
    train_subset: int | None = None
    test_subset: int | None = None

    # System
    device: str = "cuda"
    seed: int = 0
    num_workers: int = 0   # PyG on Windows: 0 is safest
    log_every: int = 1


def fit(cfg: TrainConfig) -> dict:
    """Train one model end-to-end. Returns final summary."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Resolve paths.
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(cfg.data_dir)
    if not data_dir.is_absolute():
        data_dir = (repo_root / data_dir).resolve()
    runs_dir = Path(cfg.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = (repo_root / runs_dir).resolve()
    run_dir = runs_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load cached data.
    print(f"Loading cached dataset from {data_dir / 'processed'}")
    train_data = torch.load(data_dir / "processed" / "train.pt",
                             weights_only=False)
    test_data = torch.load(data_dir / "processed" / "test.pt",
                             weights_only=False)
    stats = torch.load(data_dir / "processed" / "stats.pt",
                        weights_only=False)

    if cfg.train_subset is not None:
        train_data = train_data[:cfg.train_subset]
    if cfg.test_subset is not None:
        test_data = test_data[:cfg.test_subset]

    print(f"  train: {len(train_data):,}   test: {len(test_data):,}")
    print(f"  avg adjacency: {stats['avg_adjacency']:.3f}")
    print(f"  label mean ± std: {stats['label_mean']:+.4f} ± "
          f"{stats['label_std']:.4f} eV/atom")

    # Loaders.
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size,
                                shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size,
                               shuffle=False, num_workers=cfg.num_workers)

    # Model.
    model = GNoMEStructural(
        avg_adjacency=stats["avg_adjacency"],
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = LinearLR(optimizer, start_factor=1.0,
                          end_factor=cfg.lr_end_factor, total_iters=cfg.epochs)

    # Cache normalization constants on device.
    mu = torch.tensor(stats["label_mean"], device=device)
    sigma = torch.tensor(stats["label_std"], device=device)

    # Training state.
    best_mae = float("inf")
    epochs_since_improvement = 0
    history = []

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss = _run_epoch(model, train_loader, optimizer, mu, sigma,
                                 device, cfg.grad_clip, train=True)
        test_mae = _run_epoch(model, test_loader, None, mu, sigma,
                                device, None, train=False)
        scheduler.step()

        wall = time.time() - t0
        log = {
            "epoch": epoch,
            "train_loss_norm": train_loss,
            "test_mae_eV_per_atom": test_mae,
            "lr": optimizer.param_groups[0]["lr"],
            "wall_seconds": wall,
        }
        history.append(log)

        if epoch % cfg.log_every == 0:
            print(f"epoch {epoch:>3d}  loss(norm) {train_loss:.4f}  "
                  f"test_MAE {test_mae*1000:>6.1f} meV/atom  "
                  f"lr {log['lr']:.2e}  ({wall:.1f}s)")

        # Save best.
        if test_mae < best_mae:
            best_mae = test_mae
            epochs_since_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": asdict(cfg),
                "stats": stats,
                "test_mae": test_mae,
            }, run_dir / "best.pt")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= cfg.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {cfg.early_stop_patience} epochs).")
                break

    # Save full history.
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final summary.
    summary = {
        "run_name": cfg.run_name,
        "best_test_mae_eV_per_atom": best_mae,
        "best_test_mae_meV_per_atom": best_mae * 1000,
        "final_epoch": history[-1]["epoch"],
        "n_train": len(train_data),
        "n_test": len(test_data),
        "n_params": n_params,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. best test MAE: {best_mae*1000:.1f} meV/atom")
    print(f"Run dir: {run_dir}")
    return summary


def _run_epoch(model, loader, optimizer, mu, sigma, device,
                grad_clip, train: bool) -> float:
    """Run one epoch. Returns average loss (train) or MAE in eV/atom (eval)."""
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    count = 0

    for batch in loader:
        batch = batch.to(device)
        target_norm = (batch.y - mu) / sigma

        if train:
            optimizer.zero_grad()
            pred_norm = model(batch)
            loss = (pred_norm - target_norm).abs().mean()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs
        else:
            with torch.no_grad():
                pred_norm = model(batch)
                # De-normalize to physical units (eV/atom) before MAE.
                pred = pred_norm * sigma + mu
                abs_err = (pred - batch.y).abs().sum().item()
                total += abs_err
                count += batch.num_graphs

    return total / count
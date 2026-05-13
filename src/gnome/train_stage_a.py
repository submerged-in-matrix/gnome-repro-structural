"""Stage A training loop — 500 epochs, EMA, effective batch=256.

Differences from train.py / fit():
- EMA (decay=0.999): shadow weights tracked; val MAE logged for both
  raw and EMA weights every epoch.
- Effective batch size = 256 via gradient accumulation (2 × 128).
- Linear LR decay to 0.1× over all epochs.
- Early stopping DISABLED — EMA keeps improving past raw plateau.
- Checkpoint saves EMA weights (not raw) as best.pt.
- history.json logs both val_mae_raw and val_mae_ema each epoch.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch_geometric.loader import DataLoader

from gnome.model import GNoMEStructural
from gnome.ema import EMA


@dataclass
class StageAConfig:
    # Data paths
    data_dir: str = "./data"
    runs_dir: str = "./runs"
    run_name: str = "stage_a"

    # Architecture
    hidden_dim: int = 256
    n_layers: int = 3
    use_adj_norm: bool = True

    # Optimization
    epochs: int = 500
    batch_size: int = 128          # physical batch per forward pass
    accum_steps: int = 2           # effective batch = batch_size * accum_steps
    lr: float = 5.5e-4
    lr_end_factor: float = 0.1
    grad_clip: float = 1.0

    # EMA
    ema_decay: float = 0.999

    # System
    device: str = "cuda"
    seed: int = 0
    num_workers: int = 0
    log_every: int = 1


def fit_stage_a(cfg: StageAConfig) -> dict:
    """Train one Stage A model. Returns final summary dict."""
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
    test_data  = torch.load(data_dir / "processed" / "test.pt",
                            weights_only=False)
    stats      = torch.load(data_dir / "processed" / "stats.pt",
                            weights_only=False)

    print(f"  train: {len(train_data):,}   test: {len(test_data):,}")
    print(f"  avg adjacency:      {stats['avg_adjacency']:.3f}")
    print(f"  label mean ± std:   {stats['label_mean']:+.4f} ± "
          f"{stats['label_std']:.4f} eV/atom")
    print(f"  effective batch:    {cfg.batch_size * cfg.accum_steps} "
          f"({cfg.batch_size} × {cfg.accum_steps} accum steps)")

    # Loaders.
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_data,  batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)

    # Model.
    model = GNoMEStructural(
        avg_adjacency=stats["avg_adjacency"],
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        use_adj_norm=cfg.use_adj_norm,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = LinearLR(optimizer, start_factor=1.0,
                         end_factor=cfg.lr_end_factor,
                         total_iters=cfg.epochs)

    # Normalisation constants on device.
    mu    = torch.tensor(stats["label_mean"], device=device)
    sigma = torch.tensor(stats["label_std"],  device=device)

    # EMA tracker.
    ema = EMA(model, decay=cfg.ema_decay)

    best_mae_ema = float("inf")
    history = []

    for epoch in range(cfg.epochs):
        t0 = time.time()

        # ── Train with gradient accumulation ──────────────────────────
        train_loss = _train_epoch(
            model, train_loader, optimizer, ema,
            mu, sigma, device, cfg.grad_clip, cfg.accum_steps,
        )

        # ── Evaluate raw weights ───────────────────────────────────────
        val_mae_raw = _eval_epoch(model, test_loader, mu, sigma, device)

        # ── Evaluate EMA weights (context manager swaps weights) ───────
        with ema.apply(model):
            val_mae_ema = _eval_epoch(model, test_loader, mu, sigma, device)

        scheduler.step()
        wall = time.time() - t0

        log = {
            "epoch":               epoch,
            "train_loss_norm":     train_loss,
            "val_mae_raw_meV":     val_mae_raw  * 1000,
            "val_mae_ema_meV":     val_mae_ema  * 1000,
            "ema_gain_meV":        (val_mae_raw - val_mae_ema) * 1000,
            "lr":                  optimizer.param_groups[0]["lr"],
            "wall_seconds":        wall,
        }
        history.append(log)

        if epoch % cfg.log_every == 0:
            print(
                f"epoch {epoch:>3d}  "
                f"loss {train_loss:.4f}  "
                f"MAE_raw {val_mae_raw*1000:>6.1f}  "
                f"MAE_ema {val_mae_ema*1000:>6.1f}  "
                f"gain {log['ema_gain_meV']:>+5.1f} meV  "
                f"lr {log['lr']:.2e}  ({wall:.1f}s)"
            )

        # Checkpoint on EMA MAE improvement.
        if val_mae_ema < best_mae_ema:
            best_mae_ema = val_mae_ema
            torch.save({
                "epoch":        epoch,
                "model_state":  ema.shadow,   # save EMA weights
                "config":       asdict(cfg),
                "stats":        stats,
                "val_mae_ema":  val_mae_ema,
                "val_mae_raw":  val_mae_raw,
            }, run_dir / "best.pt")

    # Save full history.
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "run_name":               cfg.run_name,
        "best_val_mae_ema_meV":   best_mae_ema * 1000,
        "final_epoch":            history[-1]["epoch"],
        "n_train":                len(train_data),
        "n_test":                 len(test_data),
        "n_params":               n_params,
        "effective_batch":        cfg.batch_size * cfg.accum_steps,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone.  best EMA val MAE: {best_mae_ema*1000:.1f} meV/atom")
    print(f"Run dir: {run_dir}")
    return summary


# ── Epoch helpers ──────────────────────────────────────────────────────────────

def _train_epoch(
    model, loader, optimizer, ema,
    mu, sigma, device, grad_clip, accum_steps,
) -> float:
    """One training epoch with gradient accumulation. Returns mean normalised loss."""
    model.train()
    total, count = 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        target_norm = (batch.y - mu) / sigma

        pred_norm = model(batch)
        # Scale loss by accum_steps so gradients are equivalent to a
        # single forward pass over the full effective batch.
        loss = (pred_norm - target_norm).abs().mean() / accum_steps
        loss.backward()

        total += loss.item() * accum_steps * batch.num_graphs
        count += batch.num_graphs

        # Optimizer step every accum_steps mini-batches (or at end of epoch).
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)          # update shadow weights after each real step
            optimizer.zero_grad()

    return total / count


def _eval_epoch(model, loader, mu, sigma, device) -> float:
    """One evaluation pass. Returns MAE in eV/atom."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch) * sigma + mu
            total += (pred - batch.y).abs().sum().item()
            count += batch.num_graphs
    return total / count

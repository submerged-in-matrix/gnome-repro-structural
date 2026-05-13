"""Stage 1 Baseline — Learning Curve Analysis.

Reads runs/default/history.json and produces:
    Plot 1 : test MAE vs epoch (raw + smoothed)
    Plot 2 : normalised train loss + learning rate vs epoch
    Plot 3 : last-100-epoch zoom + exponential trend + extrapolation
    results/stage1_baseline/01_summary_table.csv

All plots saved to results/stage1_baseline/. However, they will be gitignored.

Public API
----------
    run(repo_root, show=False)   called by the master analysis notebook
    main()                       CLI entry point
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file's location)
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
GREY   = "#8C8C8C"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 120,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    """Rolling mean with reflection padding to avoid edge shrinkage."""
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _exp_decay(x, a, b, c):
    """Exponential decay model: a * exp(-b * x) + c."""
    return a * np.exp(-b * x) + c


def _fit_and_project(epochs: np.ndarray, mae: np.ndarray,
                     targets: list[int]) -> tuple[dict, object]:
    """Fit exponential decay; return (projections dict, fitted params or None)."""
    try:
        p0      = (mae[0] - mae[-1], 0.005, mae[-1])
        bounds  = ([0, 0, 0], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(_exp_decay, epochs, mae,
                            p0=p0, bounds=bounds, maxfev=10_000)
        proj = {t: float(_exp_decay(t, *popt)) for t in targets}
        return proj, popt
    except Exception:
        return {t: float("nan") for t in targets}, None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run(repo_root: Path | None = None, show: bool = False) -> dict:
    """Run full learning curve analysis. Returns summary dict."""

    root     = Path(repo_root) if repo_root else _REPO_ROOT
    hist_path= root / "runs" / "default" / "history.json"
    out_dir  = root / "results" / "stage1_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load history ---
    with open(hist_path) as f:
        history = json.load(f)

    df         = pd.DataFrame(history)
    epochs     = df["epoch"].values
    mae_mev    = df["test_mae_eV_per_atom"].values * 1000
    train_loss = df["train_loss_norm"].values
    lr         = df["lr"].values
    wall_s     = df["wall_seconds"].values

    n_epochs   = len(epochs)
    best_idx   = int(np.argmin(mae_mev))
    best_epoch = int(epochs[best_idx])
    best_mae   = float(mae_mev[best_idx])
    final_mae  = float(mae_mev[-1])
    avg_wall   = float(np.mean(wall_s))
    mae_smooth = _smooth(mae_mev)

    # Last-50-epoch slope
    slope = float(np.polyfit(epochs[-50:].astype(float), mae_mev[-50:], 1)[0])

    # Convergence epoch: first epoch within 2% of best MAE
    candidates = np.where(mae_mev <= best_mae * 1.02)[0]
    conv_epoch = int(epochs[candidates[0]]) if len(candidates) else best_epoch

    # Extrapolation from last-100-epoch window
    zoom_n      = min(100, n_epochs)
    zoom_epochs = epochs[-zoom_n:].astype(float)
    zoom_mae    = mae_mev[-zoom_n:]
    proj, popt  = _fit_and_project(zoom_epochs, zoom_mae, [300, 500, 1000])
    proj_time   = {e: round(e * avg_wall / 3600, 1) for e in [300, 500, 1000]}

    # -------------------------------------------------------------------
    # Plot 1 — Test MAE vs Epoch
    # -------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(epochs, mae_mev,    color=BLUE,   alpha=0.35, lw=1,
             label="Test MAE (raw)")
    ax1.plot(epochs, mae_smooth, color=BLUE,   alpha=1.0,  lw=2,
             label="Test MAE (smoothed)")
    ax1.axvline(best_epoch, color=ORANGE, ls="--", lw=1.5,
                label=f"Best epoch {best_epoch}  ({best_mae:.1f} meV/atom)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test MAE  (meV/atom)")
    ax1.set_title("Stage 1 Baseline — Test MAE vs Epoch")
    ax1.legend(fontsize=9)
    fig1.tight_layout()
    p1 = out_dir / "01_learning_curve_mae.png"
    fig1.savefig(p1, dpi=150)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------
    # Plot 2 — Normalised Train Loss + LR
    # -------------------------------------------------------------------
    fig2, ax2a = plt.subplots(figsize=(9, 4))
    ax2a.plot(epochs, train_loss, color=GREEN, lw=1.5,
              label="Train loss (normalised)")
    ax2b = ax2a.twinx()
    ax2b.plot(epochs, lr, color=GREY, lw=1, ls=":", label="Learning rate")
    ax2b.set_ylabel("Learning rate", color=GREY)
    ax2b.tick_params(axis="y", colors=GREY)
    ax2b.spines["right"].set_visible(True)
    ax2b.spines["right"].set_color(GREY)
    ax2a.set_xlabel("Epoch")
    ax2a.set_ylabel("Normalised train loss")
    ax2a.set_title("Stage 1 Baseline — Train Loss vs Epoch")
    lines  = ax2a.get_legend_handles_labels()
    lines2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=9)
    fig2.tight_layout()
    p2 = out_dir / "01_learning_curve_loss.png"
    fig2.savefig(p2, dpi=150)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------
    # Plot 3 — Last-100-epoch zoom + extrapolation
    # -------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(zoom_epochs, zoom_mae, color=BLUE, lw=2,
             label=f"Test MAE (last {zoom_n} epochs)")

    if popt is not None:
        ext_ep  = np.linspace(zoom_epochs[0], 1000, 400)
        ext_mae = _exp_decay(ext_ep, *popt)
        ax3.plot(ext_ep, ext_mae, color=ORANGE, lw=1.5, ls="--",
                 label="Extrapolated trend")
        colors = {300: GREEN, 500: GREY, 1000: "red"}
        for t, c in colors.items():
            if t > zoom_epochs[-1]:
                val = float(_exp_decay(t, *popt))
                ax3.axvline(t, color=c, ls=":", lw=1, alpha=0.6)
                ax3.scatter([t], [val], color=c, zorder=5,
                            label=f"Epoch {t}: ~{val:.1f} meV")

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Test MAE  (meV/atom)")
    ax3.set_title("Stage 1 Baseline — Last 100 Epochs + Extrapolation")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    p3 = out_dir / "01_learning_curve_zoom.png"
    fig3.savefig(p3, dpi=150)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    summary = {
        "best_epoch"              : best_epoch,
        "best_mae_meV"            : round(best_mae,  2),
        "final_mae_meV"           : round(final_mae, 2),
        "convergence_epoch_2pct"  : conv_epoch,
        "last50_slope_meV_per_ep" : round(slope, 4),
        "avg_wall_s_per_epoch"    : round(avg_wall,  1),
        "proj_mae_ep300_meV"      : round(proj[300],  2),
        "proj_mae_ep500_meV"      : round(proj[500],  2),
        "proj_mae_ep1000_meV"     : round(proj[1000], 2),
        "proj_time_ep300_h"       : proj_time[300],
        "proj_time_ep500_h"       : proj_time[500],
        "proj_time_ep1000_h"      : proj_time[1000],
    }
    csv_path = out_dir / "01_summary_table.csv"
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # Console summary
    print(f"\n{'='*52}")
    print(f"  Stage 1 Baseline — Learning Curve Summary")
    print(f"{'='*52}")
    print(f"  Best epoch         : {best_epoch}  ({best_mae:.2f} meV/atom)")
    print(f"  Final epoch MAE    : {final_mae:.2f} meV/atom")
    print(f"  Convergence epoch  : {conv_epoch}  (within 2% of best)")
    print(f"  Last-50 slope      : {slope:.4f} meV/epoch")
    print(f"  Avg wall/epoch     : {avg_wall:.1f} s")
    print(f"  --- Projections (extrapolated) ---")
    print(f"  Epoch  300 : ~{proj[300]:.2f} meV   ({proj_time[300]} h)")
    print(f"  Epoch  500 : ~{proj[500]:.2f} meV   ({proj_time[500]} h)")
    print(f"  Epoch 1000 : ~{proj[1000]:.2f} meV  ({proj_time[1000]} h)")
    print(f"{'='*52}\n")

    if show:
        plt.show()
    else:
        plt.close("all")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(repo_root=args.repo_root, show=args.show)


if __name__ == "__main__":
    main()

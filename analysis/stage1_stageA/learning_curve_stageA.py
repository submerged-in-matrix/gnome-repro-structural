"""Stage A — Learning Curve Analysis.

Reads runs/stage_a/history.json and produces:
    Plot 1 : val MAE raw vs EMA vs epoch
    Plot 2 : normalised train loss + learning rate vs epoch
    Plot 3 : last-100-epoch zoom + extrapolation (EMA MAE)
    Plot 4 : EMA gain (raw - EMA) vs epoch
    results/stage1_stageA/01_summary_table.csv

All plots saved to results/stage1_stageA/.

Public API
----------
    run(repo_root, show=False)   called by master analysis notebook
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
# Paths
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]

# ---------------------------------------------------------------------------
# Style constants  (consistent with baseline)
# ---------------------------------------------------------------------------
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
GREY   = "#8C8C8C"
RED    = "#C44E52"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 120,
})


# ---------------------------------------------------------------------------
# Helpers  (identical to baseline)
# ---------------------------------------------------------------------------

def _smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def _fit_and_project(epochs: np.ndarray, mae: np.ndarray,
                     targets: list[int]) -> tuple[dict, object]:
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
    """Run full Stage A learning curve analysis. Returns summary dict."""

    root      = Path(repo_root) if repo_root else _REPO_ROOT
    hist_path = root / "runs" / "stage_a" / "history.json"
    out_dir   = root / "results" / "stage1_stageA"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load history ---
    with open(hist_path) as f:
        history = json.load(f)

    df         = pd.DataFrame(history)
    epochs     = df["epoch"].values
    mae_raw    = df["val_mae_raw_meV"].values
    mae_ema    = df["val_mae_ema_meV"].values
    ema_gain   = df["ema_gain_meV"].values
    train_loss = df["train_loss_norm"].values
    lr         = df["lr"].values
    wall_s     = df["wall_seconds"].values

    n_epochs       = len(epochs)
    avg_wall       = float(np.mean(wall_s))

    # Best by EMA MAE
    best_idx_ema   = int(np.argmin(mae_ema))
    best_epoch_ema = int(epochs[best_idx_ema])
    best_mae_ema   = float(mae_ema[best_idx_ema])
    final_mae_ema  = float(mae_ema[-1])
    final_mae_raw  = float(mae_raw[-1])

    # Last-50-epoch slope on EMA MAE
    slope = float(np.polyfit(epochs[-50:].astype(float), mae_ema[-50:], 1)[0])

    # Convergence epoch: first epoch within 2% of best EMA MAE
    candidates = np.where(mae_ema <= best_mae_ema * 1.02)[0]
    conv_epoch = int(epochs[candidates[0]]) if len(candidates) else best_epoch_ema

    # Extrapolation from last-100-epoch EMA window
    zoom_n      = min(100, n_epochs)
    zoom_epochs = epochs[-zoom_n:].astype(float)
    zoom_mae    = mae_ema[-zoom_n:]
    proj, popt  = _fit_and_project(zoom_epochs, zoom_mae, [500, 700, 1000])
    proj_time   = {e: round(e * avg_wall / 3600, 1) for e in [500, 700, 1000]}

    # -------------------------------------------------------------------
    # Plot 1 — Raw vs EMA MAE
    # -------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(epochs, mae_raw, color=BLUE,   alpha=0.4, lw=1,
             label="Val MAE — raw weights")
    ax1.plot(epochs, mae_ema, color=ORANGE, alpha=1.0, lw=2,
             label="Val MAE — EMA weights")
    ax1.axvline(best_epoch_ema, color=GREEN, ls="--", lw=1.5,
                label=f"Best EMA epoch {best_epoch_ema}  ({best_mae_ema:.1f} meV/atom)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val MAE  (meV/atom)")
    ax1.set_title("Stage A — Val MAE: Raw vs EMA")
    ax1.legend(fontsize=9)
    fig1.tight_layout()
    p1 = out_dir / "01_learning_curve_mae.png"
    fig1.savefig(p1, dpi=150)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------
    # Plot 2 — Train loss + LR
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
    ax2a.set_title("Stage A — Train Loss vs Epoch")
    lines  = ax2a.get_legend_handles_labels()
    lines2 = ax2b.get_legend_handles_labels()
    ax2a.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=9)
    fig2.tight_layout()
    p2 = out_dir / "01_learning_curve_loss.png"
    fig2.savefig(p2, dpi=150)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------
    # Plot 3 — Last-100-epoch zoom + extrapolation (EMA)
    # -------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(zoom_epochs, zoom_mae, color=ORANGE, lw=2,
             label=f"EMA MAE (last {zoom_n} epochs)")

    if popt is not None:
        ext_ep  = np.linspace(zoom_epochs[0], 1000, 400)
        ext_mae = _exp_decay(ext_ep, *popt)
        ax3.plot(ext_ep, ext_mae, color=BLUE, lw=1.5, ls="--",
                 label="Extrapolated trend")
        colors = {500: GREEN, 700: GREY, 1000: RED}
        for t, c in colors.items():
            if t > zoom_epochs[-1]:
                val = float(_exp_decay(t, *popt))
                ax3.axvline(t, color=c, ls=":", lw=1, alpha=0.6)
                ax3.scatter([t], [val], color=c, zorder=5,
                            label=f"Epoch {t}: ~{val:.1f} meV")

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Val MAE  (meV/atom)")
    ax3.set_title("Stage A — Last 100 Epochs + Extrapolation (EMA)")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    p3 = out_dir / "01_learning_curve_zoom.png"
    fig3.savefig(p3, dpi=150)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------
    # Plot 4 — EMA gain vs epoch
    # -------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.plot(epochs, ema_gain, color=RED, lw=1.5,
             label="EMA gain  (raw − EMA MAE)")
    ax4.axhline(0, color=GREY, lw=0.8, ls="--")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("MAE improvement from EMA  (meV/atom)")
    ax4.set_title("Stage A — EMA Contribution Over Training")
    ax4.legend(fontsize=9)
    fig4.tight_layout()
    p4 = out_dir / "01_ema_gain.png"
    fig4.savefig(p4, dpi=150)
    print(f"Saved: {p4}")

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    summary = {
        "best_epoch_ema"          : best_epoch_ema,
        "best_mae_ema_meV"        : round(best_mae_ema,  2),
        "final_mae_ema_meV"       : round(final_mae_ema, 2),
        "final_mae_raw_meV"       : round(final_mae_raw, 2),
        "ema_gain_final_meV"      : round(final_mae_raw - final_mae_ema, 2),
        "convergence_epoch_2pct"  : conv_epoch,
        "last50_slope_meV_per_ep" : round(slope, 4),
        "avg_wall_s_per_epoch"    : round(avg_wall, 1),
        "proj_mae_ep500_meV"      : round(proj[500],  2),
        "proj_mae_ep700_meV"      : round(proj[700],  2),
        "proj_mae_ep1000_meV"     : round(proj[1000], 2),
        "proj_time_ep500_h"       : proj_time[500],
        "proj_time_ep700_h"       : proj_time[700],
        "proj_time_ep1000_h"      : proj_time[1000],
    }
    csv_path = out_dir / "01_summary_table.csv"
    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # Console summary
    print(f"\n{'='*52}")
    print(f"  Stage A — Learning Curve Summary")
    print(f"{'='*52}")
    print(f"  Best EMA epoch     : {best_epoch_ema}  ({best_mae_ema:.2f} meV/atom)")
    print(f"  Final EMA MAE      : {final_mae_ema:.2f} meV/atom")
    print(f"  Final raw MAE      : {final_mae_raw:.2f} meV/atom")
    print(f"  EMA gain (final)   : {final_mae_raw - final_mae_ema:.2f} meV/atom")
    print(f"  Convergence epoch  : {conv_epoch}  (within 2% of best)")
    print(f"  Last-50 slope      : {slope:.4f} meV/epoch")
    print(f"  Avg wall/epoch     : {avg_wall:.1f} s")
    print(f"  --- Projections (EMA, extrapolated) ---")
    print(f"  Epoch  500 : ~{proj[500]:.2f} meV   ({proj_time[500]} h)")
    print(f"  Epoch  700 : ~{proj[700]:.2f} meV   ({proj_time[700]} h)")
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
"""Stage A — Learning Curve Analysis.

Supports both EMA variants by accepting a ``run_name`` parameter:

    run(run_name="stage_a",       ema_label="EMA-0.999")   # original
    run(run_name="stage_a_ema99", ema_label="EMA-0.99")    # new variant

Reads:
    runs/<run_name>/history.json

Produces per-run:
    results/stage1_<run_name>/01_learning_curve_mae.png
    results/stage1_<run_name>/01_learning_curve_loss.png
    results/stage1_<run_name>/01_learning_curve_zoom.png
    results/stage1_<run_name>/01_ema_gain.png
    results/stage1_<run_name>/01_summary_table.csv

Cross-variant comparison (call after both runs):
    compare_ema_variants(summary_999, summary_99, repo_root)
    -> results/stage1_stageA_compare/01_ema_variant_comparison.png

Public API
----------
    run(repo_root, run_name, ema_label, show)
    compare_ema_variants(summary_999, summary_99, repo_root, show)
    main()
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

_HERE      = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
GREY   = "#8C8C8C"
RED    = "#C44E52"
TEAL   = "#4EAAA1"

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
# Per-run analysis
# ---------------------------------------------------------------------------

def run(
    repo_root: Path | None = None,
    run_name:  str         = "stage_a",
    ema_label: str         = "EMA-0.999",
    show:      bool        = False,
) -> dict:
    """Run learning curve analysis for one Stage A variant.

    Parameters
    ----------
    run_name : str
        Sub-folder under ``runs/``.  Use ``"stage_a"`` for the EMA-0.999
        run and ``"stage_a_ema99"`` for the EMA-0.99 run.
    ema_label : str
        Human-readable label used in plot titles / legends.
    """
    root      = Path(repo_root) if repo_root else _REPO_ROOT
    hist_path = root / "runs" / run_name / "history.json"
    out_dir   = root / "results" / f"stage1_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    n_epochs   = len(epochs)
    avg_wall   = float(np.mean(wall_s))

    best_idx_ema   = int(np.argmin(mae_ema))
    best_epoch_ema = int(epochs[best_idx_ema])
    best_mae_ema   = float(mae_ema[best_idx_ema])
    final_mae_ema  = float(mae_ema[-1])
    final_mae_raw  = float(mae_raw[-1])

    slope = float(np.polyfit(epochs[-50:].astype(float), mae_ema[-50:], 1)[0])

    candidates = np.where(mae_ema <= best_mae_ema * 1.02)[0]
    conv_epoch = int(epochs[candidates[0]]) if len(candidates) else best_epoch_ema

    zoom_n      = min(100, n_epochs)
    zoom_epochs = epochs[-zoom_n:].astype(float)
    zoom_mae    = mae_ema[-zoom_n:]
    proj, popt  = _fit_and_project(zoom_epochs, zoom_mae, [500, 700, 1000])
    proj_time   = {e: round(e * avg_wall / 3600, 1) for e in [500, 700, 1000]}

    tag = f"Stage A ({ema_label})"

    # -------------------------------------------------------------------
    # Plot 1 — Raw vs EMA MAE
    # -------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(epochs, mae_raw, color=BLUE,   alpha=0.4, lw=1,
             label="Val MAE — raw weights")
    ax1.plot(epochs, mae_ema, color=ORANGE, alpha=1.0, lw=2,
             label=f"Val MAE — {ema_label} weights")
    ax1.axvline(best_epoch_ema, color=GREEN, ls="--", lw=1.5,
                label=f"Best EMA epoch {best_epoch_ema}  ({best_mae_ema:.1f} meV/atom)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val MAE  (meV/atom)")
    ax1.set_title(f"{tag} — Val MAE: Raw vs EMA")
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
    ax2a.set_title(f"{tag} — Train Loss vs Epoch")
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
    ax3.plot(zoom_epochs, zoom_mae, color=ORANGE, lw=2,
             label=f"{ema_label} MAE (last {zoom_n} epochs)")
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
    ax3.set_title(f"{tag} — Last 100 Epochs + Extrapolation")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    p3 = out_dir / "01_learning_curve_zoom.png"
    fig3.savefig(p3, dpi=150)
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------
    # Plot 4 — EMA gain
    # -------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.plot(epochs, ema_gain, color=RED, lw=1.5,
             label=f"{ema_label} gain  (raw − EMA MAE)")
    ax4.axhline(0, color=GREY, lw=0.8, ls="--")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("MAE improvement from EMA  (meV/atom)")
    ax4.set_title(f"{tag} — EMA Contribution Over Training")
    ax4.legend(fontsize=9)
    fig4.tight_layout()
    p4 = out_dir / "01_ema_gain.png"
    fig4.savefig(p4, dpi=150)
    print(f"Saved: {p4}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    summary = {
        "run_name"                : run_name,
        "ema_label"               : ema_label,
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

    print(f"\n{'='*52}")
    print(f"  {tag} — Learning Curve Summary")
    print(f"{'='*52}")
    print(f"  Best EMA epoch     : {best_epoch_ema}  ({best_mae_ema:.2f} meV/atom)")
    print(f"  Final EMA MAE      : {final_mae_ema:.2f} meV/atom")
    print(f"  Final raw MAE      : {final_mae_raw:.2f} meV/atom")
    print(f"  EMA gain (final)   : {final_mae_raw - final_mae_ema:.2f} meV/atom")
    print(f"  Convergence epoch  : {conv_epoch}  (within 2% of best)")
    print(f"  Last-50 slope      : {slope:.4f} meV/epoch")
    print(f"  Avg wall/epoch     : {avg_wall:.1f} s")
    print(f"  --- Projections ---")
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
# Cross-variant comparison
# ---------------------------------------------------------------------------

def compare_ema_variants(
    summary_999: dict,
    summary_99:  dict,
    repo_root:   Path | None = None,
    show:        bool        = False,
) -> None:
    """Overlay learning curves of EMA-0.999 vs EMA-0.99 and compare key metrics.

    Produces:
        results/stage1_stageA_compare/01_ema_variant_comparison.png

    Parameters
    ----------
    summary_999 : dict   returned by run(run_name="stage_a",       ema_label="EMA-0.999")
    summary_99  : dict   returned by run(run_name="stage_a_ema99", ema_label="EMA-0.99")
    """
    root    = Path(repo_root) if repo_root else _REPO_ROOT
    out_dir = root / "results" / "stage1_stageA_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load both histories
    def _load(run_name):
        with open(root / "runs" / run_name / "history.json") as f:
            h = json.load(f)
        df = pd.DataFrame(h)
        return df["epoch"].values, df["val_mae_ema_meV"].values, df["ema_gain_meV"].values

    ep999, ema999, gain999 = _load(summary_999["run_name"])
    ep99,  ema99,  gain99  = _load(summary_99["run_name"])

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # --- Panel A: EMA MAE overlay ---
    ax_a = fig.add_subplot(gs[0, :])   # full-width top
    ax_a.plot(ep999, ema999, color=BLUE,   lw=2, label="EMA-0.999 (stage_a)")
    ax_a.plot(ep99,  ema99,  color=ORANGE, lw=2, label="EMA-0.99  (stage_a_ema99)", ls="--")
    ax_a.axvline(summary_999["best_epoch_ema"], color=BLUE,   ls=":", lw=1, alpha=0.7)
    ax_a.axvline(summary_99["best_epoch_ema"],  color=ORANGE, ls=":", lw=1, alpha=0.7)
    ax_a.annotate(
        f"best {summary_999['best_mae_ema_meV']:.2f} meV\n@ep{summary_999['best_epoch_ema']}",
        xy=(summary_999["best_epoch_ema"], summary_999["best_mae_ema_meV"]),
        xytext=(summary_999["best_epoch_ema"] + 15, summary_999["best_mae_ema_meV"] + 1.5),
        fontsize=8, color=BLUE,
    )
    ax_a.annotate(
        f"best {summary_99['best_mae_ema_meV']:.2f} meV\n@ep{summary_99['best_epoch_ema']}",
        xy=(summary_99["best_epoch_ema"], summary_99["best_mae_ema_meV"]),
        xytext=(summary_99["best_epoch_ema"] + 15, summary_99["best_mae_ema_meV"] + 3.0),
        fontsize=8, color=ORANGE,
    )
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Val MAE  (meV/atom)")
    ax_a.set_title("EMA-0.999 vs EMA-0.99 — Validation MAE (EMA weights)")
    ax_a.legend(fontsize=9)

    # --- Panel B: EMA gain overlay ---
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.plot(ep999, gain999, color=BLUE,   lw=1.5, label="EMA-0.999 gain")
    ax_b.plot(ep99,  gain99,  color=ORANGE, lw=1.5, label="EMA-0.99 gain", ls="--")
    ax_b.axhline(0, color=GREY, lw=0.8, ls="--")
    ax_b.set_xlabel("Epoch")
    ax_b.set_ylabel("EMA gain  (meV/atom)")
    ax_b.set_title("EMA Contribution")
    ax_b.legend(fontsize=8)

    # --- Panel C: Key metric bar comparison ---
    ax_c = fig.add_subplot(gs[1, 1])
    metric_keys   = ["best_mae_ema_meV", "final_mae_ema_meV", "ema_gain_final_meV"]
    metric_labels = ["Best EMA MAE", "Final EMA MAE", "EMA gain (final)"]
    vals_999 = [summary_999[k] for k in metric_keys]
    vals_99  = [summary_99[k]  for k in metric_keys]

    x     = np.arange(len(metric_labels))
    width = 0.3
    bars1 = ax_c.bar(x - width / 2, vals_999, width, color=BLUE,   label="EMA-0.999", edgecolor="white")
    bars2 = ax_c.bar(x + width / 2, vals_99,  width, color=ORANGE, label="EMA-0.99",  edgecolor="white")
    for bar, val in zip(list(bars1) + list(bars2), vals_999 + vals_99):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=7,
        )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(metric_labels, fontsize=8)
    ax_c.set_ylabel("meV/atom")
    ax_c.set_title("Learning Curve Metrics")
    ax_c.legend(fontsize=8)

    fig.suptitle(
        "Stage A — EMA Variant Comparison  (learning curves)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    p = out_dir / "01_ema_variant_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")

    # Console delta summary
    delta_best  = summary_99["best_mae_ema_meV"]  - summary_999["best_mae_ema_meV"]
    delta_final = summary_99["final_mae_ema_meV"] - summary_999["final_mae_ema_meV"]
    print(f"\n{'='*52}")
    print(f"  EMA Variant — Learning Curve Delta")
    print(f"{'='*52}")
    print(f"  {'Metric':<28} {'EMA-0.999':>9} {'EMA-0.99':>9} {'Δ':>8}")
    print(f"  {'-'*28}  {'-'*9}  {'-'*9}  {'-'*8}")
    print(f"  {'Best EMA MAE (meV)':<28} {summary_999['best_mae_ema_meV']:>9.2f} {summary_99['best_mae_ema_meV']:>9.2f} {delta_best:>+8.2f}")
    print(f"  {'Final EMA MAE (meV)':<28} {summary_999['final_mae_ema_meV']:>9.2f} {summary_99['final_mae_ema_meV']:>9.2f} {delta_final:>+8.2f}")
    print(f"  {'Best epoch':<28} {summary_999['best_epoch_ema']:>9d} {summary_99['best_epoch_ema']:>9d}")
    print(f"  {'Convergence epoch (2%)':<28} {summary_999['convergence_epoch_2pct']:>9d} {summary_99['convergence_epoch_2pct']:>9d}")
    print(f"  {'Last-50 slope (meV/ep)':<28} {summary_999['last50_slope_meV_per_ep']:>9.4f} {summary_99['last50_slope_meV_per_ep']:>9.4f}")
    print(f"{'='*52}\n")

    if show:
        plt.show()
    else:
        plt.close("all")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--run-name",  type=str,  default="stage_a")
    parser.add_argument("--ema-label", type=str,  default="EMA-0.999")
    parser.add_argument("--compare",   action="store_true",
                        help="Run both variants and produce comparison plot.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.compare:
        s999 = run(repo_root=args.repo_root, run_name="stage_a",
                   ema_label="EMA-0.999", show=False)
        s99  = run(repo_root=args.repo_root, run_name="stage_a_ema99",
                   ema_label="EMA-0.99",  show=False)
        compare_ema_variants(s999, s99, repo_root=args.repo_root, show=args.show)
    else:
        run(repo_root=args.repo_root,
            run_name=args.run_name,
            ema_label=args.ema_label,
            show=args.show)


if __name__ == "__main__":
    main()
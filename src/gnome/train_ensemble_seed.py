"""Train one ensemble member via the locked Stage A recipe, parameterized by seed.

One seed is one self-contained job: writes runs/ensemble/seed_<N>/{best.pt,
history.json, summary.json}. The orchestrator loops this entrypoint; nothing
here is aware of sibling seeds, keeping the atomic unit resource-agnostic.
"""
from __future__ import annotations

import argparse

from gnome.train_stage_a import StageAConfig, fit_stage_a


# Paper Methods learning rate; set explicitly so the final ensemble matches the
# published value rather than the 5.5e-4 search-era StageAConfig default.
PAPER_LR = 5.55e-4


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one ensemble seed.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--runs-dir", type=str, default="./runs")
    args = parser.parse_args()

    # Architecture and EMA fields are omitted on purpose: StageAConfig defaults
    # already equal the locked config (hidden=256, n_layers=3, adj_norm=True,
    # ema=0.999, batch 128x2), so only per-seed identity, epochs, and the paper
    # LR are set here to prevent silent drift from the locked recipe.
    cfg = StageAConfig(
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        run_name=f"ensemble/seed_{args.seed}",
        seed=args.seed,
        epochs=args.epochs,
        lr=PAPER_LR,
    )
    fit_stage_a(cfg)


if __name__ == "__main__":
    main()

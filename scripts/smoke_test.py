"""Smoke test: 5 epochs on small data subsets.

Verifies the entire training pipeline runs end-to-end on the GPU
before committing to a long full-dataset run. Should complete in ~3-5 min.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gnome.train import TrainConfig, fit


cfg = TrainConfig(
    run_name="smoke",
    epochs=5,
    batch_size=64,
    train_subset=2000,
    test_subset=500,
    early_stop_patience=999,   # No early stop in smoke
    log_every=1,
)


if __name__ == "__main__":
    summary = fit(cfg)
    print("\n=== Smoke test passed ===")
    print(f"Loss should have decreased over 5 epochs.")
    print(f"Final test MAE: {summary['best_test_mae_meV_per_atom']:.1f} meV/atom")
    print(f"(Note: meaningless on 2000 structures; just checking the loop runs.)")
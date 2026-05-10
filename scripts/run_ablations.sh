#!/usr/bin/env bash
# Run all three Stage B ablations sequentially.
# Launch once, walk away.
set -euo pipefail   # stop immediately if any run fails

echo "=== Stage B Ablations ==="
echo "Started: $(date)"

python scripts/train_full.py --config configs/ablation_no_norm.yaml
echo "B1 done: $(date)"

python scripts/train_full.py --config configs/ablation_small.yaml
echo "B2 done: $(date)"

python scripts/train_full.py --config configs/ablation_shallow.yaml
echo "B3 done: $(date)"

echo "=== All ablations complete: $(date) ==="
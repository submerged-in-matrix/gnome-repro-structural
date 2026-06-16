#!/usr/bin/env bash
# Sequential anchored coordinate search: depth -> activation -> n_layers -> lr.
#
# Restartable on two levels:
#   - Each launcher skips its axis if already resolved in the ledger.
#   - run_axis reuses any completed per-run summary.json.
# Re-running this script after an interruption therefore resumes from the first
# unfinished work rather than starting over. set -e stops on a real failure so a
# crash does not silently advance to the next axis on stale anchors.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== axis 1/4: MLP depth ==="
python "${SCRIPT_DIR}/scan_1_depth.py"

echo "=== axis 2/4: activation ==="
python "${SCRIPT_DIR}/scan_2_activation.py"

echo "=== axis 3/4: n_layers ==="
python "${SCRIPT_DIR}/scan_3_nlayers.py"

echo "=== axis 4/4: learning rate ==="
python "${SCRIPT_DIR}/scan_4_lr.py"

echo "=== search complete ==="

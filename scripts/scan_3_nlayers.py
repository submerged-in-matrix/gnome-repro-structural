"""Axis 3 launcher: message-passing rounds (n_layers).

Idempotent: if the axis is already resolved in the ledger the run is skipped,
so run_scan.sh can restart after an interruption without repeating the axis.
"""
from gnome.train_to_scan import run_axis, load_ledger

AXIS = "n_layers"

if AXIS in load_ledger()["resolved"]:
    print(f"axis '{AXIS}' already resolved; skipping")
else:
    run_axis(AXIS)

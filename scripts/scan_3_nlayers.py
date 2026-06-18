"""Axis 3 launcher: message-passing rounds (n_layers).

Idempotent: if the axis is already resolved in the ledger the run is skipped,
so run_scan.sh can restart after an interruption without repeating the axis.

Launch with:
    torchrun --nproc_per_node=2 scripts/scan_3_nlayers.py   # T4x2 DDP
    python scripts/scan_3_nlayers.py                         # single GPU
"""
import os
import torch
import torch.distributed as dist
from gnome.train_to_scan import run_axis, load_ledger

AXIS = "n_layers"

if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if AXIS in load_ledger()["resolved"]:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"axis '{AXIS}' already resolved; skipping")
    else:
        run_axis(AXIS)

    if dist.is_initialized():
        dist.destroy_process_group()

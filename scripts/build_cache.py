"""Convert all MP entries to PyG Data objects and cache to disk.

Output:
    data/processed/train.pt    # list of Data objects, train split
    data/processed/test.pt     # list of Data objects, test split
    data/processed/stats.pt    # dataset stats (avg adjacency, label mean/std)

One-time cost: ~30-45 min single-threaded on a modern CPU.
After cache exists, training scripts load in seconds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Ensure project src/ is on path even when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gnome.data import iter_mp_entries, assign_split
from gnome.graphs import structure_to_graph


DATA_DIR = Path(os.environ.get("GNOME_DATA_DIR", "./data"))
if not DATA_DIR.is_absolute():
    DATA_DIR = (_REPO_ROOT / DATA_DIR).resolve()
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    train_data: list = []
    test_data: list = []
    n_skipped_graph_failed = 0
    n_skipped_too_large = 0
    MAX_ATOMS = 200    # Hard cap; structures above this are exotic supercells.

    print(f"Building cache → {PROCESSED_DIR}")
    print(f"Cap: max {MAX_ATOMS} atoms per structure (rare exceedances skipped).\n")

    for entry in tqdm(iter_mp_entries(), total=160_000,
                       desc="building graphs"):
        n_atoms = len(entry["structure"])
        if n_atoms > MAX_ATOMS:
            n_skipped_too_large += 1
            continue

        graph = structure_to_graph(entry["structure"], entry["e_form_per_atom"])
        if graph is None:
            n_skipped_graph_failed += 1
            continue

        # Tag with material_id so we can trace failures later if needed.
        graph.material_id = entry["material_id"]

        if assign_split(entry["formula_pretty"]) == "train":
            train_data.append(graph)
        else:
            test_data.append(graph)

    print(f"\nFinal counts:")
    print(f"  train: {len(train_data):,}")
    print(f"  test:  {len(test_data):,}")
    print(f"  skipped (n_atoms > {MAX_ATOMS}): {n_skipped_too_large}")
    print(f"  skipped (graph build failed):    {n_skipped_graph_failed}")

    # Compute dataset-average adjacency over training set only.
    # Average of (edges per atom) across all training graphs.
    total_edges = sum(g.edge_index.size(1) for g in train_data)
    total_atoms = sum(g.num_atoms.item() for g in train_data)
    avg_adjacency = total_edges / total_atoms

    # Compute label statistics for normalization in the training loop.
    train_labels = torch.tensor([g.y.item() for g in train_data])
    label_mean = float(train_labels.mean())
    label_std = float(train_labels.std())

    print(f"\nTraining-set statistics:")
    print(f"  avg adjacency:  {avg_adjacency:.3f} edges per atom")
    print(f"  label mean:     {label_mean:+.4f} eV/atom")
    print(f"  label std:      {label_std:.4f} eV/atom")

    # Save.
    torch.save(train_data, PROCESSED_DIR / "train.pt")
    torch.save(test_data, PROCESSED_DIR / "test.pt")
    torch.save({
        "avg_adjacency": avg_adjacency,
        "label_mean": label_mean,
        "label_std": label_std,
        "n_train": len(train_data),
        "n_test": len(test_data),
    }, PROCESSED_DIR / "stats.pt")

    print(f"\nWrote:")
    print(f"  {PROCESSED_DIR / 'train.pt'}")
    print(f"  {PROCESSED_DIR / 'test.pt'}")
    print(f"  {PROCESSED_DIR / 'stats.pt'}")


if __name__ == "__main__":
    main()
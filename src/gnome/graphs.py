"""Graph construction: pymatgen Structure -> PyG Data.

Promoted from notebooks/01_data_check.ipynb (Cell 1).
Single source of truth for graph features so training and inference use
identical conventions.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure


# GNoME paper specifies 4.0 Å cutoff for the structural property GNN.
EDGE_CUTOFF_ANGSTROMS = 4.0

# 100 covers all elements up to Fermium with margin for periodic-table
# edge cases beyond what MP contains.
NUM_ELEMENTS = 100

# Gaussian basis size. 64 sits between CGCNN's 41 and MEGNet's 100,
# matched to a 256-dim hidden layer downstream.
NUM_RBF = 64

# r_max exceeds EDGE_CUTOFF so the last Gaussian center is not at the
# truncation boundary. r_min=0 covers minimum bond length.
RBF_R_MIN = 0.0
RBF_R_MAX = 5.0


def expand_gaussians(distances: torch.Tensor, num_gaussians: int = NUM_RBF,
                      r_min: float = RBF_R_MIN,
                      r_max: float = RBF_R_MAX) -> torch.Tensor:
    """Expand scalar distances into a basis of equally-spaced Gaussians.

    Input shape:  (N,)
    Output shape: (N, num_gaussians)
    """
    centers = torch.linspace(r_min, r_max, num_gaussians,
                              device=distances.device)
    sigma = (r_max - r_min) / (num_gaussians - 1)
    diff = distances.unsqueeze(-1) - centers
    return torch.exp(-0.5 * (diff / sigma) ** 2)


def structure_to_graph(structure: Structure,
                        e_form_per_atom: float) -> Data | None:
    """Convert a pymatgen Structure to a PyG Data object.

    Returns None for isolated atoms or structures with no edges within
    cutoff. Caller filters Nones.
    """
    Z = np.array([site.specie.Z for site in structure])
    if Z.max() > NUM_ELEMENTS:
        return None
    x = torch.zeros(len(Z), NUM_ELEMENTS, dtype=torch.float32)
    x[torch.arange(len(Z)), torch.from_numpy(Z) - 1] = 1.0

    centers, neighbors, offsets, distances = structure.get_neighbor_list(
        r=EDGE_CUTOFF_ANGSTROMS)
    if len(centers) == 0:
        return None

    edge_index = torch.from_numpy(np.stack([centers, neighbors])).long()
    edge_attr = expand_gaussians(torch.from_numpy(distances).float())

    pos = torch.from_numpy(structure.cart_coords).float()
    y = torch.tensor([e_form_per_atom], dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=y,
        num_atoms=torch.tensor([len(Z)], dtype=torch.long),
    )
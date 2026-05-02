"""GNoME-style structural GNN for crystal formation-energy prediction.

Architecture: 3 message-passing layers operating on node, edge, and global
features. Edge-to-node messages are normalized by the dataset-average
adjacency to stabilize training across structures of varying coordination.
Readout: linear projection of the final global feature to scalar energy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


# Architectural defaults match the GNoME paper Methods section:
# hidden_dim=256, n_layers=3, edge_dim=64 (Gaussian basis from graphs.py).
DEFAULT_HIDDEN_DIM = 256
DEFAULT_N_LAYERS = 3
DEFAULT_NUM_ELEMENTS = 100
DEFAULT_EDGE_DIM = 64


class MLP(nn.Module):
    """Two-layer perceptron: Linear -> SiLU -> Linear.

    Used as the update function inside each message-passing block.
    SiLU (a.k.a. swish) matches GNoME's choice; it is smooth and
    self-gating, helpful when stacked.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GNoMEBlock(nn.Module):
    """One round of message passing: edge update -> node update -> global update.

    Mirrors the GraphNetwork formulation of Battaglia et al. 2018,
    minus per-graph batching subtleties handled outside.
    """

    def __init__(self, hidden_dim: int, avg_adjacency: float):
        super().__init__()
        # Edge update: takes (src_node, dst_node, edge) -> new edge.
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim)
        # Node update: takes (current_node, aggregated_edges, global) -> new node.
        self.node_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim)
        # Global update: takes (current_global, aggregated_nodes) -> new global.
        self.global_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim)
        # Stored as a buffer (not a parameter); it is a fixed dataset statistic.
        self.register_buffer(
            "avg_adjacency", torch.tensor(avg_adjacency, dtype=torch.float32)
        )

    def forward(
        self,
        x: torch.Tensor,           # (N_atoms_total, hidden_dim)
        edge_index: torch.Tensor,  # (2, N_edges_total)
        edge_attr: torch.Tensor,   # (N_edges_total, hidden_dim)
        u: torch.Tensor,           # (N_graphs, hidden_dim)
        batch: torch.Tensor,       # (N_atoms_total,) - graph index per atom
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index

        # Edge update: concatenate source-node, destination-node, current-edge,
        # pass through MLP. Result is the updated edge feature.
        edge_inputs = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_attr_new = self.edge_mlp(edge_inputs)

        # Aggregate updated edges into destination nodes via summed scatter,
        # then divide by dataset-average adjacency. This is the GNoME trick:
        # without it, message magnitudes scale with coordination number and
        # destabilize training across heterogeneously coordinated structures.
        edge_messages = scatter(
            edge_attr_new, dst, dim=0, dim_size=x.size(0), reduce="sum"
        )
        edge_messages = edge_messages / self.avg_adjacency

        # Node update: concatenate current node, aggregated edge messages,
        # and the per-graph global feature broadcast to each atom.
        node_inputs = torch.cat([x, edge_messages, u[batch]], dim=-1)
        x_new = self.node_mlp(node_inputs)

        # Global update: aggregate updated nodes per graph (mean) and combine
        # with current global feature.
        node_per_graph = scatter(
            x_new, batch, dim=0, dim_size=u.size(0), reduce="mean"
        )
        global_inputs = torch.cat([u, node_per_graph], dim=-1)
        u_new = self.global_mlp(global_inputs)

        return x_new, edge_attr_new, u_new


class GNoMEStructural(nn.Module):
    """Full GNoME structural GNN.

    Embeds one-hot atoms and Gaussian-basis edges into hidden_dim space,
    iterates n_layers GNoME blocks, reads out scalar energy per graph
    from the final global feature.
    """

    def __init__(
        self,
        avg_adjacency: float,
        n_elements: int = DEFAULT_NUM_ELEMENTS,
        edge_dim: int = DEFAULT_EDGE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_layers: int = DEFAULT_N_LAYERS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project one-hot atoms into hidden space. Linear layer with no bias
        # is equivalent to a learned embedding table of shape (n_elements, hidden).
        self.node_embed = nn.Linear(n_elements, hidden_dim, bias=False)
        # Project Gaussian-expanded distances into hidden space.
        self.edge_embed = nn.Linear(edge_dim, hidden_dim, bias=False)
        # Initial global feature: a single learned vector, broadcast per graph.
        self.global_init = nn.Parameter(torch.zeros(hidden_dim))
        # Stack of message-passing blocks.
        self.blocks = nn.ModuleList(
            [GNoMEBlock(hidden_dim, avg_adjacency) for _ in range(n_layers)]
        )
        # Readout: scalar prediction from final global feature.
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        # data is a torch_geometric.data.Batch (or Data, single-graph case).
        # data.batch is the per-atom graph index; for a single graph it is
        # absent, so synthesize zeros.
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                data.x.size(0), dtype=torch.long, device=data.x.device
            )
        n_graphs = int(batch.max().item()) + 1

        # Embed inputs into hidden space.
        x = self.node_embed(data.x)
        e = self.edge_embed(data.edge_attr)
        # Replicate the learned initial global vector once per graph in batch.
        u = self.global_init.unsqueeze(0).expand(n_graphs, -1).contiguous()

        # Iterate message-passing blocks.
        for block in self.blocks:
            x, e, u = block(x, data.edge_index, e, u, batch)

        # Project final global feature to scalar energy per graph.
        return self.readout(u).squeeze(-1)
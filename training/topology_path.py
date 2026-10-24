from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import base_path as base

def canonical_topology_variant(variant: str) -> str:
    allowed = {'no_graph', 'contact_only', 'esm_map_only', 'identity_graph_control', 'dual_graph'}
    if variant not in allowed:
        raise ValueError(f'unsupported topology-path variant: {variant}')
    return variant

def directed_high_attention_graph(attention_matrix: np.ndarray, k: int, symmetric: bool=True) -> np.ndarray:
    del symmetric
    return base.attention_to_knn_matrix(attention_matrix, k, symmetric=False, weighted=True)

class StandardGCNIILayer(nn.Module):

    def __init__(self, hidden_size: int, alpha: float=0.1, lam: float=0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.lam = lam
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, initial: torch.Tensor, layer_id: int, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.bmm(adjacency, hidden)
        support = (1.0 - self.alpha) * propagated + self.alpha * initial
        beta = math.log(self.lam / float(layer_id) + 1.0)
        identity_path = (1.0 - beta) * support
        learned_path = beta * self.linear(support)
        return self.norm(torch.nn.functional.leaky_relu(identity_path + learned_path))

class ContactGCNII(nn.Module):

    def __init__(self, hidden_size: int=512, num_layers: int=4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([StandardGCNIILayer(hidden_size) for _ in range(num_layers)])
        self.final = nn.Linear(hidden_size, hidden_size // 2, bias=False)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        initial = hidden
        for layer_id, layer in enumerate(self.layers, start=1):
            hidden = layer(hidden, initial, layer_id, adjacency)
        return self.final(hidden)

class DirectedAttentionGAT(nn.Module):

    def __init__(self, hidden_size: int=512, num_layers: int=4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([base.BaseGAT(hidden_size, dropout=0.0) for _ in range(num_layers)])
        self.final = nn.Linear(hidden_size, hidden_size // 2, bias=False)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        return self.final(hidden)

def identity_adjacency(hidden: torch.Tensor, node_mask: Optional[torch.Tensor]) -> torch.Tensor:
    batch, nodes, _ = hidden.shape
    if node_mask is None:
        diagonal = torch.ones((batch, nodes), dtype=hidden.dtype, device=hidden.device)
    else:
        diagonal = node_mask.squeeze(-1).to(dtype=hidden.dtype)
    return torch.diag_embed(diagonal)

class TopologyGraphBlock(nn.Module):

    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = canonical_topology_variant(variant)
        self.contact_gcn2 = ContactGCNII(512)
        self.attention_gat = DirectedAttentionGAT(512)
        self.mixer = nn.Linear(512, 512, bias=False)
        nn.init.eye_(self.mixer.weight)
        self.branch_log_scales = nn.Parameter(torch.full((2,), -2.0))
        self.dropout = nn.Dropout(0.2)

    def forward(self, hidden: torch.Tensor, contact_adj: torch.Tensor, attention_adj: torch.Tensor, node_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if self.variant == 'no_graph':
            return hidden
        identity = identity_adjacency(hidden, node_mask)
        if self.variant == 'identity_graph_control':
            contact_adj = identity
            attention_adj = identity
        zeros = torch.zeros((*hidden.shape[:2], 256), dtype=hidden.dtype, device=hidden.device)
        if self.variant in {'contact_only', 'dual_graph', 'identity_graph_control'}:
            contact_delta = self.contact_gcn2(hidden, contact_adj) - self.contact_gcn2(hidden, identity)
        else:
            contact_delta = zeros
        if self.variant in {'esm_map_only', 'dual_graph', 'identity_graph_control'}:
            attention_delta = self.attention_gat(hidden, attention_adj) - self.attention_gat(hidden, identity)
        else:
            attention_delta = zeros
        scales = torch.sigmoid(self.branch_log_scales)
        update = torch.cat([scales[0] * contact_delta, scales[1] * attention_delta], dim=-1)
        update = self.mixer(update)
        if node_mask is not None:
            update = update * node_mask
        return hidden + self.dropout(update)

def install_topology_path() -> None:
    base.canonical_variant = canonical_topology_variant
    base.distance_to_knn_matrix = directed_high_attention_graph
    base.GraphBlock = TopologyGraphBlock
if __name__ == '__main__':
    install_topology_path()
    base.main()

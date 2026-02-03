"""Topology GNN (GraphVAE-style) for fullerene adjacency generation.

This model learns a distribution over 3-regular planar graphs from the
training dataset. It can be used to generate topologies for C values that
are missing from the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from synthetic_topology import generate_planar_3_regular_graph


def build_adj_matrix(edge_index: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    adj = torch.zeros((n, n), device=device, dtype=torch.float32)
    row, col = edge_index
    adj[row, col] = 1.0
    adj[col, row] = 1.0
    adj.fill_diagonal_(0.0)
    return adj


def topology_loss(
    logits: torch.Tensor,
    adj: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    lambda_degree: float = 0.1,
    lambda_kl: float = 0.01,
) -> torch.Tensor:
    # BCE on upper triangle
    n = adj.size(0)
    mask = torch.triu(torch.ones((n, n), device=adj.device, dtype=torch.bool), diagonal=1)
    bce = F.binary_cross_entropy_with_logits(logits[mask], adj[mask])

    # Degree regularization (soft degree from sigmoid)
    prob = torch.sigmoid(logits)
    degree = prob.sum(dim=1)
    degree_loss = ((degree - 3.0) ** 2).mean()

    # KL divergence
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + lambda_degree * degree_loss + lambda_kl * kl


class TopologyGNN(nn.Module):
    """GraphVAE-style topology generator.

    Encoder uses true adjacency; decoder samples adjacency from a latent code.
    """

    def __init__(self, *, max_nodes: int = 720, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.node_emb = nn.Embedding(max_nodes, hidden_dim)
        self.C_emb = nn.Embedding(max_nodes + 1, hidden_dim)

        # Encoder
        self.enc_lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_lin1 = nn.Linear(latent_dim + hidden_dim, hidden_dim)
        self.dec_lin2 = nn.Linear(hidden_dim, hidden_dim)

    def _encode(self, edge_index: torch.Tensor, C_value: int, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.arange(num_nodes, device=edge_index.device)
        h = self.node_emb(idx) + self.C_emb(torch.tensor([C_value], device=edge_index.device)).expand(num_nodes, -1)

        # Simple neighbor aggregation using adjacency
        row, col = edge_index
        agg = torch.zeros_like(h)
        agg.index_add_(0, row, h[col])
        h = F.relu(self.enc_lin1(h + agg))

        agg = torch.zeros_like(h)
        agg.index_add_(0, row, h[col])
        h = F.relu(self.enc_lin2(h + agg))

        g = h.mean(dim=0, keepdim=True)
        return self.mu(g), self.logvar(g)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z: torch.Tensor, C_value: int, num_nodes: int) -> torch.Tensor:
        idx = torch.arange(num_nodes, device=z.device)
        c_emb = self.C_emb(torch.tensor([C_value], device=z.device)).expand(num_nodes, -1)
        z_expand = z.expand(num_nodes, -1)
        h = torch.cat([z_expand, c_emb], dim=-1)
        h = F.relu(self.dec_lin1(h))
        h = self.dec_lin2(h)
        logits = h @ h.t()
        logits.fill_diagonal_(float("-inf"))
        return logits

    def forward(self, *, edge_index: torch.Tensor, C_value: int, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self._encode(edge_index, C_value, num_nodes)
        z = self._reparameterize(mu, logvar)
        logits = self._decode(z, C_value, num_nodes)
        return logits, mu, logvar

    def generate_edge_index(
        self,
        *,
        C_value: int,
        device: torch.device,
        max_tries: int = 50,
        enforce_planar: bool = True,
    ) -> torch.Tensor:
        if C_value < 20 or C_value % 2 != 0:
            raise ValueError("C must be >=20 and even for fullerene topologies.")
        if C_value > self.max_nodes:
            raise ValueError(f"C={C_value} exceeds max_nodes={self.max_nodes}.")

        for _ in range(max_tries):
            z = torch.randn(1, self.latent_dim, device=device)
            logits = self._decode(z, C_value, C_value)
            prob = torch.sigmoid(logits)

            # 3-regular via top-3 neighbors
            edges = []
            for i in range(C_value):
                vals, idxs = torch.topk(prob[i], k=3)
                for j in idxs.tolist():
                    if i != j:
                        edges.append((i, j))
                        edges.append((j, i))

            # Build graph and verify
            G = nx.Graph()
            G.add_nodes_from(range(C_value))
            G.add_edges_from({(min(u, v), max(u, v)) for u, v in edges})

            if not nx.is_connected(G):
                continue

            if enforce_planar:
                is_planar, _ = nx.check_planarity(G)
                if not is_planar:
                    continue

            # Ensure 3-regular
            if all(d == 3 for _, d in G.degree()):
                edge_index = torch.tensor(list(G.edges()), dtype=torch.long)
                edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0).t().contiguous()
                return edge_index.to(device)

        # Fallback: algorithmic planar 3-regular graph
        G_fallback = generate_planar_3_regular_graph(C_value, max_tries=2000)
        if G_fallback is None:
            raise RuntimeError("Failed to generate planar 3-regular graph for fallback.")
        edge_index = torch.tensor(list(G_fallback.edges()), dtype=torch.long)
        edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0).t().contiguous()
        return edge_index.to(device)

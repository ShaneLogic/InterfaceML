"""Train the topology GNN (graph autoencoder) used to generate fullerene adjacencies.

This model is trained **before** diffusion so it can provide synthetic
3-regular planar topologies when dataset templates are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import FullereneDataset
from topology_gnn import TopologyGNN, topology_loss, build_adj_matrix

logger = logging.getLogger(__name__)


def _load_graphs(config: dict, split: str) -> List[tuple[torch.Tensor, int, int]]:
    data_cfg = config.get("data", {})
    dataset = FullereneDataset(
        xyz_dir=data_cfg["xyz_dir"],
        split_csv=data_cfg["split_csv"],
        split=split,
        C_range=(data_cfg.get("C_min", 20), data_cfg.get("C_max", 70)),
        center=True,
        normalize_scale=True,
    )

    graphs: List[tuple[torch.Tensor, int, int]] = []
    for item in dataset:
        graphs.append((item.edge_index, int(item.C.item()), int(item.num_nodes)))
    return graphs


def train_topology_gnn(config: dict, device: torch.device) -> Path:
    topo_cfg = config.get("topology_gnn", {})
    if not topo_cfg.get("enabled", False):
        return Path(topo_cfg.get("checkpoint_path", "checkpoints/topology_gnn.pt"))

    max_C = int(topo_cfg.get("max_C", 720))
    hidden_dim = int(topo_cfg.get("hidden_dim", 64))
    latent_dim = int(topo_cfg.get("latent_dim", 32))
    epochs = int(topo_cfg.get("epochs", 50))
    lr = float(topo_cfg.get("learning_rate", 1e-3))
    weight_decay = float(topo_cfg.get("weight_decay", 1e-5))
    lambda_degree = float(topo_cfg.get("lambda_degree", 0.1))
    lambda_kl = float(topo_cfg.get("lambda_kl", 0.01))

    ckpt_path = Path(topo_cfg.get("checkpoint_path", "checkpoints/topology_gnn.pt"))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_graphs = _load_graphs(config, "train")
    val_graphs = _load_graphs(config, "val")

    model = TopologyGNN(max_nodes=max_C, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for edge_index, C, n in train_graphs:
            edge_index = edge_index.to(device)
            optimizer.zero_grad()
            logits, mu, logvar = model(edge_index=edge_index, C_value=C, num_nodes=n)
            adj = build_adj_matrix(edge_index, n, device=device)
            loss = topology_loss(logits, adj, mu, logvar, lambda_degree=lambda_degree, lambda_kl=lambda_kl)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("[TopologyGNN] Skipping NaN/Inf loss for C=%d", C)
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for edge_index, C, n in val_graphs:
                edge_index = edge_index.to(device)
                logits, mu, logvar = model(edge_index=edge_index, C_value=C, num_nodes=n)
                adj = build_adj_matrix(edge_index, n, device=device)
                loss = topology_loss(logits, adj, mu, logvar, lambda_degree=lambda_degree, lambda_kl=lambda_kl)
                val_loss += float(loss.item())

        val_loss /= max(1, len(val_graphs))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": topo_cfg,
                    "best_val": best_val,
                },
                ckpt_path,
            )

        avg_train = total_loss / max(1, len(train_graphs))
        logger.info(
            "[TopologyGNN] Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
            epoch, epochs, avg_train, val_loss, scheduler.get_last_lr()[0],
        )

    logger.info("[TopologyGNN] Training complete. Best val loss: %.6f", best_val)
    return ckpt_path


def main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Train topology GNN for fullerene graphs")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to config
    config_dir = Path(args.config).resolve().parent
    data_cfg = config.get("data", {})
    for key in ["xyz_dir", "analysis_dir", "split_csv"]:
        path_val = data_cfg.get(key)
        if path_val and not Path(path_val).is_absolute():
            data_cfg[key] = str((config_dir / path_val).resolve())
    config["data"] = data_cfg

    topo_cfg = config.get("topology_gnn", {})
    ckpt_path = topo_cfg.get("checkpoint_path")
    if ckpt_path and not Path(ckpt_path).is_absolute():
        topo_cfg["checkpoint_path"] = str((config_dir / ckpt_path).resolve())
        config["topology_gnn"] = topo_cfg

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu")
    train_topology_gnn(config, device)


if __name__ == "__main__":
    main()

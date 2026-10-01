"""Diagnose the interface generation pipeline bug."""
import sys
import torch
import numpy as np

sys.path.insert(0, ".")
from api import FullereneAPI
from units import denormalize_positions

api = FullereneAPI("checkpoints/best_model.pt")

# Simulate exactly what InterfaceAIGenerator does
results = api.generate(num_carbon=60, num_samples=1, ddim=False)
sample = results[0]
positions_from_api = np.asarray(sample["positions"], dtype=float)

print("=== From API (already denormalized to Angstroms) ===")
print(f"Shape: {positions_from_api.shape}")
print(f"NaN count: {np.isnan(positions_from_api).sum()}")
norms = np.linalg.norm(positions_from_api, axis=1)
print(f"Radius: mean={np.nanmean(norms):.4f}  min={np.nanmin(norms):.4f}  max={np.nanmax(norms):.4f}")
print(f"Expected C60 radius ~3.49 Angstrom")

# InterfaceAIGenerator calls _denormalize_positions AGAIN
pos_tensor = torch.tensor(positions_from_api, dtype=torch.float32)
pos_double_denorm = denormalize_positions(pos_tensor, 60).detach().numpy()
norms2 = np.linalg.norm(pos_double_denorm, axis=1)
print()
print("=== After DOUBLE denormalization (bug in interface_generator.py) ===")
print(f"Radius: mean={np.nanmean(norms2):.4f}  min={np.nanmin(norms2):.4f}  max={np.nanmax(norms2):.4f}")
print(f"Scale factor: {np.nanmean(norms2) / max(np.nanmean(norms), 1e-8):.2f}x")
print()
print("The double denorm makes positions ~3.5x too large!")
print("This blows up auto_supercell_xy, making the fullerene tiny relative to the slab.")

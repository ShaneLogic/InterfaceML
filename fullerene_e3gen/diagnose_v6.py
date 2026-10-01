"""Diagnose why noise_loss is stuck at 1.0 even with pure denoising."""
import torch
import sys
import glob
import yaml

sys.path.insert(0, '.')

print('=== DIAGNOSING WHY NOISE_LOSS STUCK AT 1.0 ===')
print()

# Theory
print('THEORY: The model output_head maps INVARIANT features h → 3D noise vectors.')
print('  But h depends only on pairwise DISTANCES (SE(3) invariant).')
print('  Rotating molecule: distances unchanged → h unchanged → same noise_pred.')
print('  But target noise SHOULD rotate with molecule → CONTRADICTION!')
print('  Best invariant predictor for random noise: predict ZERO.')
print('  MSE(zero, N(0,1)) = Var(N(0,1)) = 1.0  ← explains stuck noise_loss!')
print()

# Empirical verification
noise = torch.randn(10000, 3)
zero_pred = torch.zeros_like(noise)
mse = torch.nn.functional.mse_loss(zero_pred, noise)
print(f'Empirical: MSE(0, N(0,1)) = {mse:.6f}  (should be ~1.0)')
print()

# Load model and check outputs
from model import FullereneDiffusionModel

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

mc = cfg['model']
model = FullereneDiffusionModel(
    hidden_dim=mc['hidden_dim'],
    num_layers=mc['num_layers'],
    edge_dim=mc['edge_dim'],
    C_embed_dim=mc['C_embed_dim'],
    time_embed_dim=mc['time_embed_dim'],
    max_C=mc.get('max_C', 720),
    continuous_C_embed=mc.get('continuous_C_embed', False),
    C_fourier_features=mc.get('C_fourier_features', 16),
    use_hierarchical=mc.get('use_hierarchical', False),
    hierarchical_threshold=mc.get('hierarchical_threshold', 80),
    hierarchical_layers=mc.get('hierarchical_layers', 2),
)

ckpts = sorted(glob.glob('checkpoints/checkpoint_epoch_*.pt'))
if ckpts:
    ckpt = torch.load(ckpts[-1], map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded: {ckpts[-1]}')
else:
    print('No checkpoint found, using random weights')

model.eval()
with torch.no_grad():
    N = 60
    pos = torch.randn(N, 3)
    # Simple ring edges for testing
    src = list(range(N))
    dst = [(i + 1) % N for i in range(N)]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    t = torch.tensor([500])
    C = torch.tensor([N])
    batch = torch.zeros(N, dtype=torch.long)
    
    out = model(pos, edge_index, t, C, batch)
    
    print(f'Model output stats (should be near zero if bug confirmed):')
    print(f'  mean:          {out.mean():.6f}')
    print(f'  std:           {out.std():.6f}')
    print(f'  abs max:       {out.abs().max():.6f}')
    print(f'  L2/atom:       {out.norm(dim=-1).mean():.6f}')
    print(f'  Expected noise: ~1.0 per component, L2/atom ~1.73')
    print()
    
    # Rotation test: same molecule rotated should give same output (if invariant bug)
    theta = torch.tensor(1.2)
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    pos_rot = pos @ R.T  # Rotate positions
    out_rot = model(pos_rot, edge_index, t, C, batch)
    
    # If output is invariant (BUG), out ≈ out_rot
    # If output is equivariant (CORRECT), out_rot ≈ R @ out
    diff_invariant = (out - out_rot).norm()
    diff_equivariant = (out_rot - (out @ R.T)).norm()
    
    print(f'Rotation test (θ=1.2 rad):')
    print(f'  ||out - out_rot|| = {diff_invariant:.6f}  (if ~0: output is INVARIANT = BUG)')
    print(f'  ||out_rot - R*out|| = {diff_equivariant:.6f}  (if ~0: output is EQUIVARIANT = OK)')
    
    if diff_invariant < diff_equivariant:
        print(f'  → Output is INVARIANT (NOT equivariant) → CONFIRMED BUG!')
    else:
        print(f'  → Output appears equivariant')

print()
print('FIX: Replace output_head(h) with coordinate displacement (pos_final - pos_initial)')
print('The EGNN coordinate updates ARE equivariant by construction.')

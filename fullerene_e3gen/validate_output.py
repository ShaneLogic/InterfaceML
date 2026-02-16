"""Quick validation of generated structures."""
import numpy as np
import glob
import sys
import os

output_dir = sys.argv[1] if len(sys.argv) > 1 else "generated/test_v4"
files = sorted(glob.glob(os.path.join(output_dir, "*.xyz")))

if not files:
    print(f"No .xyz files found in {output_dir}")
    sys.exit(1)

print(f"Found {len(files)} structures in {output_dir}\n")

for fname in files:
    with open(fname) as f:
        n = int(f.readline().strip())
        f.readline()  # comment
        coords = []
        for _ in range(n):
            parts = f.readline().split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    coords = np.array(coords)
    has_nan = np.isnan(coords).any()
    
    # Pairwise distances
    dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    np.fill_diagonal(dists, 999)
    min_d = dists.min()
    
    # Radial analysis (sphericity)
    center = coords.mean(axis=0)
    radii = np.linalg.norm(coords - center, axis=1)
    
    # Bond count (C-C bonds typically 1.2-1.8 Å)
    bonds = (dists < 1.8).sum() // 2
    expected_bonds_c60 = 90  # C60 has 90 bonds
    
    name = os.path.basename(fname)
    status = "OK" if not has_nan and min_d > 0.8 else "BAD"
    print(f"[{status}] {name}: atoms={n}, NaN={has_nan}, "
          f"min_dist={min_d:.3f}A, bonds(<1.8A)={bonds}, "
          f"radius={radii.mean():.3f}+-{radii.std():.3f}A")

# C60 reference: radius ~3.55 Å, min C-C dist ~1.40 Å
print(f"\nReference C60: radius~3.55A, C-C~1.40A, 90 bonds")

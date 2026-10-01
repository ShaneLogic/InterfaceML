#!/usr/bin/env python
"""Validate v7 generated structures."""
import numpy as np
import os, glob
from scipy.spatial.distance import pdist

import sys
pattern = sys.argv[1] if len(sys.argv) > 1 else 'generated/test_v7/*.xyz'
files = sorted(glob.glob(pattern))
print(f'Found {len(files)} files')
print(f'{"File":<35} {"Atoms":>5} {"Radius":>10} {"MinDist":>10} {"MaxDist":>10} {"Bonds<1.8":>10}')
print('-' * 85)

for f in files:
    lines = open(f).readlines()
    n = int(lines[0].strip())
    coords = []
    for line in lines[2:2+n]:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords = np.array(coords)

    center = coords.mean(axis=0)
    dists_from_center = np.linalg.norm(coords - center, axis=1)
    radius = dists_from_center.mean()

    pw = pdist(coords)
    min_d = pw.min()
    max_d = pw.max()
    bonds = int(np.sum(pw < 1.8))

    print(f'{os.path.basename(f):<35} {n:>5} {radius:>10.3f} {min_d:>10.3f} {max_d:>10.3f} {bonds:>10d}')

print()
print('Expected C60: radius ~3.55A, min_dist ~1.40A, bonds ~90')

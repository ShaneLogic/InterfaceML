"""Analyze generated fullerene structures."""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
from pathlib import Path

def analyze_structure(xyz_file):
    """Analyze a single structure."""
    coords = []
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
        n_atoms = int(lines[0].strip())
        for line in lines[2:2+n_atoms]:
            parts = line.split()
            if len(parts) >= 4:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    coords = np.array(coords)
    centered = coords - coords.mean(axis=0)
    
    # Pairwise distances
    dists = pdist(centered)
    
    # Nearest neighbors (bonds)
    dist_mat = distance_matrix(centered, centered)
    np.fill_diagonal(dist_mat, np.inf)
    nearest_3 = np.sort(dist_mat, axis=1)[:, :3]
    
    # Radii
    radii = np.linalg.norm(centered, axis=1)
    
    return {
        'coord_range': (coords.min(), coords.max()),
        'coord_std': coords.std(),
        'centered_std': centered.std(),
        'all_dist_min': dists.min(),
        'all_dist_mean': dists.mean(),
        'all_dist_max': dists.max(),
        'bond_mean': nearest_3.mean(),
        'bond_std': nearest_3.std(),
        'bond_min': nearest_3.min(),
        'bond_max': nearest_3.max(),
        'reasonable_bonds_pct': 100 * np.sum((nearest_3 > 1.35) & (nearest_3 < 1.50)) / nearest_3.size,
        'radius_mean': radii.mean(),
        'radius_std': radii.std(),
        'radius_cv': radii.std() / radii.mean(),
    }

if __name__ == '__main__':
    gen_dir = Path('generated')
    xyz_files = sorted(gen_dir.glob('C60_sample_*.xyz'))[:5]  # Analyze first 5
    
    print("=" * 80)
    print("STRUCTURE QUALITY ANALYSIS")
    print("=" * 80)
    
    for xyz_file in xyz_files:
        print(f"\n{xyz_file.name}:")
        metrics = analyze_structure(xyz_file)
        
        print(f"  Coordinate range: [{metrics['coord_range'][0]:.2f}, {metrics['coord_range'][1]:.2f}]")
        print(f"  Coordinate std: {metrics['coord_std']:.3f}")
        print(f"  Centered std: {metrics['centered_std']:.3f}")
        print(f"  All distances: min={metrics['all_dist_min']:.3f}, mean={metrics['all_dist_mean']:.3f}, max={metrics['all_dist_max']:.3f}")
        print(f"  Bond lengths (3 nearest):")
        print(f"    Mean: {metrics['bond_mean']:.3f} Å (should be ~1.42)")
        print(f"    Std: {metrics['bond_std']:.3f} Å (should be <0.05)")
        print(f"    Range: [{metrics['bond_min']:.3f}, {metrics['bond_max']:.3f}]")
        print(f"    Reasonable (1.35-1.50): {metrics['reasonable_bonds_pct']:.1f}%")
        print(f"  Sphericity:")
        print(f"    Radius: {metrics['radius_mean']:.3f} ± {metrics['radius_std']:.3f} Å")
        print(f"    CV: {metrics['radius_cv']:.3f} (should be <0.1)")
        
        # Diagnose problems
        problems = []
        if metrics['bond_mean'] < 1.0 or metrics['bond_mean'] > 2.0:
            problems.append("❌ Bond lengths unrealistic")
        if metrics['bond_std'] > 0.5:
            problems.append("❌ Bond length variance too high")
        if metrics['reasonable_bonds_pct'] < 50:
            problems.append(f"❌ Only {metrics['reasonable_bonds_pct']:.0f}% bonds in valid range")
        if metrics['radius_cv'] > 0.15:
            problems.append("❌ Not spherical (high radius variation)")
        if metrics['all_dist_min'] < 0.8:
            problems.append("❌ Atoms too close (overlap)")
        
        if problems:
            print(f"  Problems:")
            for p in problems:
                print(f"    {p}")
        else:
            print(f"  ✅ Structure looks reasonable")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Aggregate analysis
    all_metrics = [analyze_structure(f) for f in xyz_files]
    avg_bond = np.mean([m['bond_mean'] for m in all_metrics])
    avg_bond_std = np.mean([m['bond_std'] for m in all_metrics])
    avg_reasonable = np.mean([m['reasonable_bonds_pct'] for m in all_metrics])
    avg_cv = np.mean([m['radius_cv'] for m in all_metrics])
    
    print(f"\nAverage bond length: {avg_bond:.3f} Å (target: 1.42 Å)")
    print(f"Average bond std: {avg_bond_std:.3f} Å (target: <0.05 Å)")
    print(f"Average reasonable bonds: {avg_reasonable:.1f}% (target: >90%)")
    print(f"Average radius CV: {avg_cv:.3f} (target: <0.1)")
    
    print("\nModel issues:")
    if avg_bond < 1.0:
        print("  ⚠️  Bonds too short - structures are compressed")
        print("     → Increase data normalization scale")
        print("     → Check dataset statistics")
    elif avg_bond > 1.8:
        print("  ⚠️  Bonds too long - structures are stretched")
    
    if avg_bond_std > 0.5:
        print("  ⚠️  High bond variance - model not learning connectivity")
        print("     → Increase lambda_bond from 0.5 to 2.0")
        print("     → Train longer (100+ epochs)")
    
    if avg_reasonable < 50:
        print("  ⚠️  Most bonds invalid - model failing")
        print("     → Check edge_index in dataset")
        print("     → Verify bond_length_loss implementation")
    
    if avg_cv > 0.15:
        print("  ⚠️  Not spherical - collapse or elongation")
        print("     → Increase lambda_sphere from 0.2 to 1.0")

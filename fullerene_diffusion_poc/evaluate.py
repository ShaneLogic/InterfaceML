"""
Evaluation script for generated fullerene structures.

Computes comprehensive metrics for publication-quality assessment:
- Bond length statistics and distribution
- Connectivity validation  
- Sphericity and shape metrics
- Radial Distribution Function (RDF)
- Angular Distribution Function (ADF)
- Gyration tensor eigenvalue distribution
- Energy landscape analysis
- Multi-panel publication-quality visualizations

Usage:
    python evaluate.py --generated_dir generated --reference_csv dataset/fullerenes/fullerene_xyz/analysis_20260122/structures.csv

Author: InterfaceML Project
Date: 2026-01-30 (Enhanced)
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import torch


def parse_xyz_file(filepath: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Parse extended XYZ file.
    
    Args:
        filepath: Path to XYZ file
    
    Returns:
        positions: [N, 3] array of coordinates
        edges: List of (src, dst) edge tuples
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0].strip())
    positions = []
    edges = []
    
    for i in range(2, 2 + num_atoms):
        parts = lines[i].strip().split()
        # Format: element x y z index nb1 nb2 nb3
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        positions.append([x, y, z])
        
        if len(parts) >= 8:
            idx = int(parts[4])
            for nb in [int(parts[5]), int(parts[6]), int(parts[7])]:
                if nb >= 0:  # Valid neighbor
                    edges.append((idx, nb))
    
    return np.array(positions), edges


def compute_bond_lengths(positions: np.ndarray, edges: List[Tuple[int, int]]) -> np.ndarray:
    """Compute bond lengths from positions and edges."""
    bond_lengths = []
    seen = set()
    
    for src, dst in edges:
        # Avoid double counting (undirected edges)
        if (min(src, dst), max(src, dst)) in seen:
            continue
        seen.add((min(src, dst), max(src, dst)))
        
        r_src = positions[src]
        r_dst = positions[dst]
        dist = np.linalg.norm(r_src - r_dst)
        bond_lengths.append(dist)
    
    return np.array(bond_lengths)


def compute_sphericity(positions: np.ndarray) -> dict:
    """
    Compute sphericity metrics.
    
    Returns:
        Dictionary with:
        - radius_gyration: Radius of gyration
        - asphericity: Asphericity parameter (0=sphere, higher=elongated)
        - eigenvalues: Gyration tensor eigenvalues [lambda1, lambda2, lambda3]
        - acylindricity: Acylindricity parameter
        - shape_anisotropy: Relative shape anisotropy
    """
    # Center positions
    center = positions.mean(axis=0)
    centered = positions - center
    
    # Radius of gyration
    r_g = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    
    # Gyration tensor
    S = np.dot(centered.T, centered) / len(positions)
    
    # Eigenvalues
    eigvals = np.linalg.eigvalsh(S)
    eigvals = np.sort(eigvals)[::-1]  # Descending order
    
    # Asphericity
    lambda1, lambda2, lambda3 = eigvals
    asphericity = lambda1 - 0.5 * (lambda2 + lambda3)
    
    # Acylindricity
    acylindricity = lambda2 - lambda3
    
    # Relative shape anisotropy (0=sphere, 1=rod)
    lambda_mean = eigvals.mean()
    if lambda_mean > 1e-8:
        shape_anisotropy = np.sum((eigvals - lambda_mean)**2) / (3 * lambda_mean**2)
    else:
        shape_anisotropy = 0.0
    
    return {
        'radius_gyration': r_g,
        'asphericity': asphericity,
        'acylindricity': acylindricity,
        'shape_anisotropy': shape_anisotropy,
        'eigenvalues': eigvals,
    }


def compute_rdf(positions: np.ndarray, r_max: float = 15.0, n_bins: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Radial Distribution Function (RDF) g(r).
    
    Args:
        positions: [N, 3] atomic positions
        r_max: Maximum radius (Angstroms)
        n_bins: Number of bins
    
    Returns:
        r: Radial distances
        g_r: Radial distribution function
    """
    N = len(positions)
    
    # Compute all pairwise distances
    distances = pdist(positions)
    
    # Histogram
    bins = np.linspace(0, r_max, n_bins + 1)
    hist, _ = np.histogram(distances, bins=bins)
    
    # Volume normalization
    r = (bins[:-1] + bins[1:]) / 2
    dr = bins[1] - bins[0]
    
    # Ideal gas normalization: 4πr²ρdr with ρ = N/V
    # For molecular structure, use simple pair normalization
    num_pairs = N * (N - 1) / 2
    shell_volume = 4 * np.pi * r**2 * dr
    shell_volume[shell_volume < 1e-10] = 1e-10  # Avoid division by zero
    
    g_r = hist / (num_pairs / len(distances)) / shell_volume * np.sum(shell_volume)
    
    return r, g_r


def compute_adf(positions: np.ndarray, edges: List[Tuple[int, int]], n_bins: int = 180) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Angular Distribution Function (ADF) for bond angles.
    
    Args:
        positions: [N, 3] atomic positions
        edges: List of edges
        n_bins: Number of angle bins
    
    Returns:
        angles_deg: Angles in degrees
        adf: Angular distribution function (normalized histogram)
    """
    # Build adjacency list
    adjacency = defaultdict(list)
    for src, dst in edges:
        adjacency[src].append(dst)
    
    # Compute all bond angles
    angles = []
    for atom_i, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue
        
        pos_i = positions[atom_i]
        
        # For each pair of neighbors, compute angle
        for idx1 in range(len(neighbors)):
            for idx2 in range(idx1 + 1, len(neighbors)):
                j = neighbors[idx1]
                k = neighbors[idx2]
                
                # Vectors from i to j and i to k
                v1 = positions[j] - pos_i
                v2 = positions[k] - pos_i
                
                # Compute angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(np.degrees(angle))
    
    if len(angles) == 0:
        return np.linspace(0, 180, n_bins), np.zeros(n_bins)
    
    angles = np.array(angles)
    
    # Histogram
    hist, bin_edges = np.histogram(angles, bins=n_bins, range=(0, 180), density=True)
    angles_deg = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return angles_deg, hist


def compute_structure_energy(positions: np.ndarray, edges: List[Tuple[int, int]], 
                             k_bond: float = 500.0, r0: float = 1.39) -> dict:
    """
    Compute simple harmonic energy landscape metrics.
    
    E = Σ k_bond * (r - r0)²
    
    Args:
        positions: Atomic positions
        edges: Bond connectivity
        k_bond: Harmonic force constant (kcal/mol/Å²)
        r0: Equilibrium bond length (Å)
    
    Returns:
        Dictionary with energy metrics
    """
    bond_lengths = compute_bond_lengths(positions, edges)
    
    # Harmonic bond energy
    bond_energies = k_bond * (bond_lengths - r0)**2
    total_energy = np.sum(bond_energies)
    
    return {
        'total_energy': total_energy,
        'mean_bond_energy': total_energy / len(bond_energies) if len(bond_energies) > 0 else 0.0,
        'max_bond_energy': np.max(bond_energies) if len(bond_energies) > 0 else 0.0,
    }


def compute_wasserstein_distance(samples1: np.ndarray, samples2: np.ndarray) -> float:
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance) between two distributions.
    
    Args:
        samples1: First sample set
        samples2: Second sample set
    
    Returns:
        Wasserstein distance
    """
    return stats.wasserstein_distance(samples1, samples2)


def validate_connectivity(edges: List[Tuple[int, int]], num_atoms: int) -> dict:
    """
    Validate that structure has correct connectivity.
    
    For fullerenes, each atom should have exactly 3 neighbors.
    
    Returns:
        Dictionary with validation results
    """
    degrees = [0] * num_atoms
    
    for src, dst in edges:
        degrees[src] += 1
    
    degrees = np.array(degrees)
    
    return {
        'mean_degree': degrees.mean(),
        'std_degree': degrees.std(),
        'all_degree_3': np.all(degrees == 3),
        'min_degree': degrees.min(),
        'max_degree': degrees.max(),
    }


class StructureEvaluator:
    """Evaluates generated structures against reference dataset with comprehensive metrics."""
    
    def __init__(self, reference_csv: Path, load_reference_structures: bool = False):
        """
        Args:
            reference_csv: Path to structures.csv from analysis
            load_reference_structures: If True, load full reference structures for RDF/ADF
        """
        self.reference_stats = self._load_reference(reference_csv)
        self.reference_csv_path = reference_csv.parent
        self.load_reference_structures = load_reference_structures
        
        if load_reference_structures:
            self.reference_structures = self._load_reference_structures()
        else:
            self.reference_structures = None
    
    def _load_reference(self, csv_path: Path) -> dict:
        """Load reference statistics from dataset."""
        bond_lengths = []
        radii = []
        asphericities = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bond_lengths.append(float(row['bond_mean']))
                radii.append(float(row['rg']))
                asphericities.append(float(row['asphericity']))
        
        return {
            'bond_lengths': {
                'mean': np.mean(bond_lengths),
                'std': np.std(bond_lengths),
                'min': np.min(bond_lengths),
                'max': np.max(bond_lengths),
                'all': np.array(bond_lengths),
            },
            'radii': {
                'mean': np.mean(radii),
                'std': np.std(radii),
                'all': np.array(radii),
            },
            'asphericities': {
                'mean': np.mean(asphericities),
                'std': np.std(asphericities),
                'all': np.array(asphericities),
            },
        }
    
    def _load_reference_structures(self) -> List[Tuple[np.ndarray, List]]:
        """Load a subset of reference structures for comparison."""
        # This would require access to original XYZ files
        # For now, return empty list
        return []
    
    def evaluate_structures(self, generated_dir: Path) -> dict:
        """Evaluate all structures in directory with comprehensive metrics."""
        xyz_files = sorted(generated_dir.glob('*.xyz'))
        
        if not xyz_files:
            print(f"No XYZ files found in {generated_dir}")
            return {}
        
        print(f"Evaluating {len(xyz_files)} structures with comprehensive metrics...")
        
        # Collect metrics
        all_bond_lengths = []
        all_radii = []
        all_asphericities = []
        all_acylindricities = []
        all_shape_anisotropies = []
        all_eigenvalues = []
        connectivity_results = []
        all_energies = []
        all_angles = []
        
        # For RDF/ADF - accumulate from multiple structures
        all_rdfs = []
        all_adfs = []
        
        for xyz_file in xyz_files:
            try:
                positions, edges = parse_xyz_file(xyz_file)
                
                # Bond lengths
                bonds = compute_bond_lengths(positions, edges)
                all_bond_lengths.extend(bonds.tolist())
                
                # Sphericity metrics
                sphere_metrics = compute_sphericity(positions)
                all_radii.append(sphere_metrics['radius_gyration'])
                all_asphericities.append(sphere_metrics['asphericity'])
                all_acylindricities.append(sphere_metrics['acylindricity'])
                all_shape_anisotropies.append(sphere_metrics['shape_anisotropy'])
                all_eigenvalues.append(sphere_metrics['eigenvalues'])
                
                # Connectivity
                conn = validate_connectivity(edges, len(positions))
                connectivity_results.append(conn['all_degree_3'])
                
                # Energy
                energy_metrics = compute_structure_energy(positions, edges)
                all_energies.append(energy_metrics['total_energy'])
                
                # RDF
                r, g_r = compute_rdf(positions)
                all_rdfs.append((r, g_r))
                
                # ADF
                angles_deg, adf = compute_adf(positions, edges)
                all_adfs.append((angles_deg, adf))
                all_angles.extend(angles_deg.tolist())
                
            except Exception as e:
                print(f"  Error processing {xyz_file.name}: {e}")
        
        # Aggregate statistics
        all_bond_lengths = np.array(all_bond_lengths)
        all_radii = np.array(all_radii)
        all_asphericities = np.array(all_asphericities)
        all_acylindricities = np.array(all_acylindricities)
        all_shape_anisotropies = np.array(all_shape_anisotropies)
        all_energies = np.array(all_energies)
        all_eigenvalues = np.array(all_eigenvalues)  # Shape: [N_structures, 3]
        
        # Average RDF and ADF
        if all_rdfs:
            r_common = all_rdfs[0][0]
            g_r_avg = np.mean([g_r for _, g_r in all_rdfs], axis=0)
            g_r_std = np.std([g_r for _, g_r in all_rdfs], axis=0)
        else:
            r_common, g_r_avg, g_r_std = None, None, None
        
        if all_adfs:
            angles_common = all_adfs[0][0]
            adf_avg = np.mean([adf for _, adf in all_adfs], axis=0)
            adf_std = np.std([adf for _, adf in all_adfs], axis=0)
        else:
            angles_common, adf_avg, adf_std = None, None, None
        
        # Compute Wasserstein distances
        w_dist_bonds = compute_wasserstein_distance(all_bond_lengths, self.reference_stats['bond_lengths']['all'])
        w_dist_radii = compute_wasserstein_distance(all_radii, self.reference_stats['radii']['all'])
        w_dist_asph = compute_wasserstein_distance(all_asphericities, self.reference_stats['asphericities']['all'])
        
        results = {
            'num_structures': len(xyz_files),
            'bond_lengths': {
                'mean': all_bond_lengths.mean(),
                'std': all_bond_lengths.std(),
                'min': all_bond_lengths.min(),
                'max': all_bond_lengths.max(),
                'median': np.median(all_bond_lengths),
                'q25': np.percentile(all_bond_lengths, 25),
                'q75': np.percentile(all_bond_lengths, 75),
            },
            'radii': {
                'mean': all_radii.mean(),
                'std': all_radii.std(),
                'median': np.median(all_radii),
            },
            'asphericities': {
                'mean': all_asphericities.mean(),
                'std': all_asphericities.std(),
                'median': np.median(all_asphericities),
            },
            'acylindricities': {
                'mean': all_acylindricities.mean(),
                'std': all_acylindricities.std(),
            },
            'shape_anisotropies': {
                'mean': all_shape_anisotropies.mean(),
                'std': all_shape_anisotropies.std(),
            },
            'eigenvalues': {
                'lambda1': {'mean': all_eigenvalues[:, 0].mean(), 'std': all_eigenvalues[:, 0].std()},
                'lambda2': {'mean': all_eigenvalues[:, 1].mean(), 'std': all_eigenvalues[:, 1].std()},
                'lambda3': {'mean': all_eigenvalues[:, 2].mean(), 'std': all_eigenvalues[:, 2].std()},
            },
            'connectivity': {
                'fraction_valid': np.mean(connectivity_results),
            },
            'energies': {
                'mean': all_energies.mean(),
                'std': all_energies.std(),
                'min': all_energies.min(),
                'max': all_energies.max(),
            },
            'wasserstein_distances': {
                'bond_lengths': w_dist_bonds,
                'radii': w_dist_radii,
                'asphericities': w_dist_asph,
            },
            # Raw data for plotting
            '_raw': {
                'bond_lengths': all_bond_lengths,
                'radii': all_radii,
                'asphericities': all_asphericities,
                'acylindricities': all_acylindricities,
                'shape_anisotropies': all_shape_anisotropies,
                'energies': all_energies,
                'eigenvalues': all_eigenvalues,
                'rdf': (r_common, g_r_avg, g_r_std) if r_common is not None else None,
                'adf': (angles_common, adf_avg, adf_std) if angles_common is not None else None,
            }
        }
        
        return results
    
    def print_comparison(self, generated_results: dict):
        """Print comprehensive comparison between generated and reference."""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nNumber of structures: {generated_results['num_structures']}")
        
        # Bond lengths
        print("\n" + "-"*80)
        print("BOND LENGTH STATISTICS (Å):")
        print(f"  Reference:  {self.reference_stats['bond_lengths']['mean']:.4f} ± {self.reference_stats['bond_lengths']['std']:.4f}")
        print(f"  Generated:  {generated_results['bond_lengths']['mean']:.4f} ± {generated_results['bond_lengths']['std']:.4f}")
        print(f"  Generated Median: {generated_results['bond_lengths']['median']:.4f}")
        print(f"  Generated IQR:    [{generated_results['bond_lengths']['q25']:.4f}, {generated_results['bond_lengths']['q75']:.4f}]")
        print(f"  Wasserstein Distance: {generated_results['wasserstein_distances']['bond_lengths']:.6f}")
        
        # Check if within reasonable range (1.39 ± 0.10 Å for C-C bonds)
        gen_mean = generated_results['bond_lengths']['mean']
        if 1.29 <= gen_mean <= 1.49:
            print(f"  ✓ Within physical range [1.29, 1.49] Å")
        else:
            print(f"  ✗ Outside physical range [1.29, 1.49] Å")
        
        # Radii
        print("\n" + "-"*80)
        print("RADIUS OF GYRATION (Å):")
        print(f"  Reference:  {self.reference_stats['radii']['mean']:.4f} ± {self.reference_stats['radii']['std']:.4f}")
        print(f"  Generated:  {generated_results['radii']['mean']:.4f} ± {generated_results['radii']['std']:.4f}")
        print(f"  Generated Median: {generated_results['radii']['median']:.4f}")
        print(f"  Wasserstein Distance: {generated_results['wasserstein_distances']['radii']:.6f}")
        
        # Shape metrics
        print("\n" + "-"*80)
        print("SHAPE DESCRIPTORS:")
        print(f"  Asphericity:")
        print(f"    Reference:  {self.reference_stats['asphericities']['mean']:.4f} ± {self.reference_stats['asphericities']['std']:.4f}")
        print(f"    Generated:  {generated_results['asphericities']['mean']:.4f} ± {generated_results['asphericities']['std']:.4f}")
        print(f"    Wasserstein Distance: {generated_results['wasserstein_distances']['asphericities']:.6f}")
        
        print(f"  Acylindricity:    {generated_results['acylindricities']['mean']:.4f} ± {generated_results['acylindricities']['std']:.4f}")
        print(f"  Shape Anisotropy: {generated_results['shape_anisotropies']['mean']:.4f} ± {generated_results['shape_anisotropies']['std']:.4f}")
        
        # Eigenvalues
        print("\n  Gyration Tensor Eigenvalues:")
        print(f"    λ₁: {generated_results['eigenvalues']['lambda1']['mean']:.4f} ± {generated_results['eigenvalues']['lambda1']['std']:.4f}")
        print(f"    λ₂: {generated_results['eigenvalues']['lambda2']['mean']:.4f} ± {generated_results['eigenvalues']['lambda2']['std']:.4f}")
        print(f"    λ₃: {generated_results['eigenvalues']['lambda3']['mean']:.4f} ± {generated_results['eigenvalues']['lambda3']['std']:.4f}")
        
        # Connectivity
        print("\n" + "-"*80)
        print("CONNECTIVITY:")
        print(f"  Fraction valid (all degree=3): {generated_results['connectivity']['fraction_valid']:.2%}")
        if generated_results['connectivity']['fraction_valid'] >= 0.95:
            print("  ✓ Excellent connectivity")
        elif generated_results['connectivity']['fraction_valid'] >= 0.80:
            print("  ⚠ Good connectivity")
        else:
            print("  ✗ Poor connectivity")
        
        # Energy
        print("\n" + "-"*80)
        print("HARMONIC BOND ENERGY (kcal/mol):")
        print(f"  Mean:   {generated_results['energies']['mean']:.2f} ± {generated_results['energies']['std']:.2f}")
        print(f"  Range:  [{generated_results['energies']['min']:.2f}, {generated_results['energies']['max']:.2f}]")
        
        print("\n" + "="*80)
    
    def plot_comparison(self, generated_results: dict, output_path: Path):
        """Create publication-quality multi-panel comparison plots."""
        
        # Set publication style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2.0,
        })
        
        # Create comprehensive figure with GridSpec
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Color scheme
        color_gen = '#2E86AB'  # Blue for generated
        color_ref = '#A23B72'  # Purple for reference
        color_fill = '#F18F01'  # Orange for fills
        
        # 1. Bond Length Distribution (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(generated_results['_raw']['bond_lengths'], bins=40, alpha=0.7, 
                color=color_gen, label='Generated', density=True, edgecolor='black', linewidth=0.5)
        ax1.axvline(self.reference_stats['bond_lengths']['mean'], color=color_ref, 
                   linestyle='--', linewidth=2.5, label='Reference Mean')
        ax1.axvspan(
            self.reference_stats['bond_lengths']['mean'] - self.reference_stats['bond_lengths']['std'],
            self.reference_stats['bond_lengths']['mean'] + self.reference_stats['bond_lengths']['std'],
            alpha=0.2, color=color_ref, label='Reference ±1σ'
        )
        ax1.set_xlabel('Bond Length (Å)', fontweight='bold')
        ax1.set_ylabel('Probability Density', fontweight='bold')
        ax1.set_title('(a) Bond Length Distribution', fontweight='bold', loc='left')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(alpha=0.3)
        
        # 2. Radius of Gyration Distribution (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(generated_results['_raw']['radii'], bins=25, alpha=0.7, 
                color=color_gen, label='Generated', density=True, edgecolor='black', linewidth=0.5)
        ax2.axvline(self.reference_stats['radii']['mean'], color=color_ref, 
                   linestyle='--', linewidth=2.5, label='Reference Mean')
        ax2.axvspan(
            self.reference_stats['radii']['mean'] - self.reference_stats['radii']['std'],
            self.reference_stats['radii']['mean'] + self.reference_stats['radii']['std'],
            alpha=0.2, color=color_ref, label='Reference ±1σ'
        )
        ax2.set_xlabel('Radius of Gyration (Å)', fontweight='bold')
        ax2.set_ylabel('Probability Density', fontweight='bold')
        ax2.set_title('(b) Radius Distribution', fontweight='bold', loc='left')
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.grid(alpha=0.3)
        
        # 3. Asphericity Distribution (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(generated_results['_raw']['asphericities'], bins=25, alpha=0.7, 
                color=color_gen, label='Generated', density=True, edgecolor='black', linewidth=0.5)
        ax3.axvline(self.reference_stats['asphericities']['mean'], color=color_ref, 
                   linestyle='--', linewidth=2.5, label='Reference Mean')
        ax3.set_xlabel('Asphericity', fontweight='bold')
        ax3.set_ylabel('Probability Density', fontweight='bold')
        ax3.set_title('(c) Asphericity Distribution', fontweight='bold', loc='left')
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        ax3.grid(alpha=0.3)
        
        # 4. Box plots comparison (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        metrics_data = [
            generated_results['_raw']['bond_lengths'],
            self.reference_stats['bond_lengths']['all']
        ]
        bp = ax4.boxplot(metrics_data, labels=['Generated', 'Reference'], patch_artist=True,
                        widths=0.6, showmeans=True, meanline=True)
        for patch, color in zip(bp['boxes'], [color_gen, color_ref]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax4.set_ylabel('Bond Length (Å)', fontweight='bold')
        ax4.set_title('(d) Bond Length Comparison', fontweight='bold', loc='left')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Eigenvalue Distribution (middle-center)
        ax5 = fig.add_subplot(gs[1, 1])
        eigenvals = generated_results['_raw']['eigenvalues']
        positions_eigen = [1, 2, 3]
        bp_eigen = ax5.boxplot([eigenvals[:, 0], eigenvals[:, 1], eigenvals[:, 2]], 
                               labels=['λ₁', 'λ₂', 'λ₃'], patch_artist=True, widths=0.6)
        for patch in bp_eigen['boxes']:
            patch.set_facecolor(color_gen)
            patch.set_alpha(0.7)
        ax5.set_ylabel('Eigenvalue', fontweight='bold')
        ax5.set_title('(e) Gyration Tensor Eigenvalues', fontweight='bold', loc='left')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Energy Distribution (middle-right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(generated_results['_raw']['energies'], bins=25, alpha=0.7, 
                color=color_fill, edgecolor='black', linewidth=0.5)
        ax6.axvline(generated_results['energies']['mean'], color='darkred', 
                   linestyle='--', linewidth=2.5, label=f"Mean: {generated_results['energies']['mean']:.1f}")
        ax6.set_xlabel('Total Bond Energy (kcal/mol)', fontweight='bold')
        ax6.set_ylabel('Count', fontweight='bold')
        ax6.set_title('(f) Harmonic Energy Distribution', fontweight='bold', loc='left')
        ax6.legend(frameon=True, fancybox=True, shadow=True)
        ax6.grid(alpha=0.3)
        
        # 7. RDF Plot (bottom-left)
        ax7 = fig.add_subplot(gs[2, 0])
        if generated_results['_raw']['rdf'] is not None:
            r, g_r_avg, g_r_std = generated_results['_raw']['rdf']
            ax7.plot(r, g_r_avg, color=color_gen, linewidth=2.5, label='Generated RDF')
            ax7.fill_between(r, g_r_avg - g_r_std, g_r_avg + g_r_std, 
                            alpha=0.3, color=color_gen)
            ax7.set_xlabel('Distance r (Å)', fontweight='bold')
            ax7.set_ylabel('g(r)', fontweight='bold')
            ax7.set_title('(g) Radial Distribution Function', fontweight='bold', loc='left')
            ax7.legend(frameon=True, fancybox=True, shadow=True)
            ax7.grid(alpha=0.3)
            ax7.set_xlim(0, 15)
        
        # 8. ADF Plot (bottom-center)
        ax8 = fig.add_subplot(gs[2, 1])
        if generated_results['_raw']['adf'] is not None:
            angles, adf_avg, adf_std = generated_results['_raw']['adf']
            ax8.plot(angles, adf_avg, color=color_gen, linewidth=2.5, label='Generated ADF')
            ax8.fill_between(angles, adf_avg - adf_std, adf_avg + adf_std, 
                            alpha=0.3, color=color_gen)
            # Mark expected fullerene angles (120° for hexagons, 108° for pentagons)
            ax8.axvline(120, color='red', linestyle=':', linewidth=1.5, label='Hexagon (120°)')
            ax8.axvline(108, color='orange', linestyle=':', linewidth=1.5, label='Pentagon (108°)')
            ax8.set_xlabel('Bond Angle (degrees)', fontweight='bold')
            ax8.set_ylabel('Probability Density', fontweight='bold')
            ax8.set_title('(h) Angular Distribution Function', fontweight='bold', loc='left')
            ax8.legend(frameon=True, fancybox=True, shadow=True)
            ax8.grid(alpha=0.3)
        
        # 9. Wasserstein Distance Bar Chart (bottom-right)
        ax9 = fig.add_subplot(gs[2, 2])
        w_distances = generated_results['wasserstein_distances']
        metrics_names = ['Bond\nLength', 'Radius', 'Asphericity']
        w_values = [w_distances['bond_lengths'], w_distances['radii'], w_distances['asphericities']]
        bars = ax9.bar(metrics_names, w_values, color=[color_gen, color_ref, color_fill], 
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax9.set_ylabel('Wasserstein Distance', fontweight='bold')
        ax9.set_title('(i) Distribution Similarity', fontweight='bold', loc='left')
        ax9.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, w_values):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Overall title
        fig.suptitle('Comprehensive Fullerene Generation Evaluation', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved publication-quality plot: {output_path}")
        plt.close()
        
    def save_detailed_report(self, generated_results: dict, output_path: Path):
        """Save detailed text report with all metrics."""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE FULLERENE EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated Structures: {generated_results['num_structures']}\n\n")
            
            f.write("BOND LENGTH STATISTICS (Angstrom)\n")
            f.write("-"*80 + "\n")
            f.write(f"Reference: {self.reference_stats['bond_lengths']['mean']:.4f} ± {self.reference_stats['bond_lengths']['std']:.4f}\n")
            f.write(f"Generated: {generated_results['bond_lengths']['mean']:.4f} ± {generated_results['bond_lengths']['std']:.4f}\n")
            f.write(f"Median:    {generated_results['bond_lengths']['median']:.4f}\n")
            f.write(f"IQR:       [{generated_results['bond_lengths']['q25']:.4f}, {generated_results['bond_lengths']['q75']:.4f}]\n")
            f.write(f"Wasserstein Distance: {generated_results['wasserstein_distances']['bond_lengths']:.6f}\n\n")
            
            f.write("GEOMETRIC PROPERTIES\n")
            f.write("-"*80 + "\n")
            f.write(f"Radius of Gyration:  {generated_results['radii']['mean']:.4f} ± {generated_results['radii']['std']:.4f} Å\n")
            f.write(f"Asphericity:         {generated_results['asphericities']['mean']:.4f} ± {generated_results['asphericities']['std']:.4f}\n")
            f.write(f"Acylindricity:       {generated_results['acylindricities']['mean']:.4f} ± {generated_results['acylindricities']['std']:.4f}\n")
            f.write(f"Shape Anisotropy:    {generated_results['shape_anisotropies']['mean']:.4f} ± {generated_results['shape_anisotropies']['std']:.4f}\n\n")
            
            f.write("GYRATION TENSOR EIGENVALUES\n")
            f.write("-"*80 + "\n")
            f.write(f"λ₁: {generated_results['eigenvalues']['lambda1']['mean']:.4f} ± {generated_results['eigenvalues']['lambda1']['std']:.4f}\n")
            f.write(f"λ₂: {generated_results['eigenvalues']['lambda2']['mean']:.4f} ± {generated_results['eigenvalues']['lambda2']['std']:.4f}\n")
            f.write(f"λ₃: {generated_results['eigenvalues']['lambda3']['mean']:.4f} ± {generated_results['eigenvalues']['lambda3']['std']:.4f}\n\n")
            
            f.write("CONNECTIVITY\n")
            f.write("-"*80 + "\n")
            f.write(f"Valid Structures (degree=3): {generated_results['connectivity']['fraction_valid']:.2%}\n\n")
            
            f.write("ENERGY LANDSCAPE\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean Energy:  {generated_results['energies']['mean']:.2f} ± {generated_results['energies']['std']:.2f} kcal/mol\n")
            f.write(f"Energy Range: [{generated_results['energies']['min']:.2f}, {generated_results['energies']['max']:.2f}] kcal/mol\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Saved detailed report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation of Generated Fullerenes')
    parser.add_argument('--generated_dir', type=str, required=True, help='Directory with generated XYZ files')
    parser.add_argument('--reference_csv', type=str, required=True, help='Reference structures.csv')
    parser.add_argument('--output_plot', type=str, default='evaluation_comprehensive.png', 
                       help='Output plot filename')
    parser.add_argument('--output_report', type=str, default='evaluation_report.txt', 
                       help='Output text report filename')
    args = parser.parse_args()
    
    generated_dir = Path(args.generated_dir)
    reference_csv = Path(args.reference_csv)
    
    if not generated_dir.exists():
        print(f"Error: {generated_dir} does not exist")
        return
    
    if not reference_csv.exists():
        print(f"Error: {reference_csv} does not exist")
        return
    
    print("="*80)
    print("COMPREHENSIVE FULLERENE EVALUATION")
    print("="*80)
    
    # Create evaluator
    evaluator = StructureEvaluator(reference_csv, load_reference_structures=False)
    
    # Evaluate generated structures
    results = evaluator.evaluate_structures(generated_dir)
    
    if not results:
        return
    
    # Print comparison
    evaluator.print_comparison(results)
    
    # Create publication-quality plots
    output_plot = generated_dir / args.output_plot
    evaluator.plot_comparison(results, output_plot)
    
    # Save detailed report
    output_report = generated_dir / args.output_report
    evaluator.save_detailed_report(results, output_report)
    
    print("\n" + "="*80)
    print("✓ Evaluation complete!")
    print(f"  - Plot saved to: {output_plot}")
    print(f"  - Report saved to: {output_report}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

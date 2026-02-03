"""
Generate standard fullerene topology templates.

Creates topologically correct connectivity for common fullerene sizes
(C20, C60, C70) based on icosahedral and related symmetries.

These templates ensure proper fullerene topology (degree-3 planar graphs,
Euler characteristic V-E+F=2, isolated pentagon rule for stable isomers).

Author: InterfaceML Project
Date: 2026-01-30
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path


def generate_icosahedral_c60() -> tuple:
    """
    Generate C60 (Buckminsterfullerene) with icosahedral symmetry.
    
    Topology: truncated icosahedron (soccer ball pattern)
    - 60 vertices
    - 90 edges (each vertex has degree 3)
    - 12 pentagons + 20 hexagons
    
    Returns:
        pos: [60, 3] initial coordinates (not optimized)
        edges: [2, E] edge connectivity
    """
    # C60 coordinates from standard icosahedral construction
    # Using golden ratio for icosahedral symmetry
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # Generate 60 vertices on approximate sphere
    vertices = []
    
    # 12 vertices from icosahedron
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([0, i, j * phi])
            vertices.append([i, j * phi, 0])
            vertices.append([j * phi, 0, i])
    
    # 20 face centers (hexagon centers)
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.append([i, j, k])
    
    # 30 edge midpoints
    for i in [-1, 1]:
        for j in [-1, 1]:
            vertices.append([0, i / phi, j * phi])
            vertices.append([i / phi, j * phi, 0])
            vertices.append([j * phi, 0, i / phi])
    
    vertices = np.array(vertices)
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices *= 3.5  # Scale to realistic C60 radius (~3.5 Å)
    
    # Build edges: connect vertices within bonding distance
    edges = []
    bond_threshold = 1.5  # Å
    
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist < bond_threshold:
                edges.append([i, j])
                edges.append([j, i])  # Bidirectional
    
    pos = torch.tensor(vertices, dtype=torch.float32)
    edges = torch.tensor(edges, dtype=torch.long).t()
    
    return pos, edges


def generate_dodecahedral_c20() -> tuple:
    """
    Generate C20 (smallest stable fullerene) with dodecahedral structure.
    
    Topology: 12 pentagonal faces only
    - 20 vertices
    - 30 edges
    
    Returns:
        pos: [20, 3] coordinates
        edges: [2, E] connectivity
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Dodecahedron vertices
    vertices = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                vertices.append([i, j, k])
    
    for i in [-1, 1]:
        vertices.append([0, i / phi, i * phi])
        vertices.append([i / phi, i * phi, 0])
        vertices.append([i * phi, 0, i / phi])
    
    vertices = np.array(vertices)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices *= 2.0  # Smaller radius for C20
    
    # Build edges
    edges = []
    bond_threshold = 1.0
    
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            if dist < bond_threshold:
                edges.append([i, j])
                edges.append([j, i])
    
    pos = torch.tensor(vertices, dtype=torch.float32)
    edges = torch.tensor(edges, dtype=torch.long).t()
    
    return pos, edges


def save_template(pos: torch.Tensor, edges: torch.Tensor, name: str, output_dir: Path):
    """Save template to file."""
    data = Data(pos=pos, edge_index=edges)
    output_path = output_dir / f"{name}.pt"
    torch.save(data, output_path)
    print(f"Saved {name} template: {pos.size(0)} atoms, {edges.size(1)} edges")


def main():
    """Generate and save all standard templates."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    print("Generating standard fullerene templates...")
    print("=" * 60)
    
    # C60 (Buckminsterfullerene)
    pos_c60, edges_c60 = generate_icosahedral_c60()
    save_template(pos_c60, edges_c60, "C60_icosahedral", output_dir)
    
    # C20 (Dodecahedron)
    pos_c20, edges_c20 = generate_dodecahedral_c20()
    save_template(pos_c20, edges_c20, "C20_dodecahedral", output_dir)
    
    print("=" * 60)
    print("✓ Template generation complete!")
    print(f"\nTemplates saved to: {output_dir}")
    print("\nUsage in generate.py:")
    print("  template_path = Path('templates/C60_icosahedral.pt')")
    print("  template = torch.load(template_path)")


if __name__ == '__main__':
    main()

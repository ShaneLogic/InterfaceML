"""
Geometric utility functions for InterfaceML.

This module provides basic vector operations and geometry calculations
that are used throughout the package.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute the angle between two vectors in degrees.

    Parameters
    ----------
    vec_a, vec_b
        Input vectors (any dimension).

    Returns
    -------
    angle
        Angle in degrees, in the range [0, 180].
    """
    dot_val = np.dot(vec_a, vec_b)
    norms = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)

    if norms < 1e-8:
        return 0.0

    cos_theta = np.clip(dot_val / norms, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    vec
        Input vector.

    Returns
    -------
    normalized
        Unit vector in the same direction, or zero vector if input is zero.
    """
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros_like(vec)
    return vec / norm


def compute_distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:
    """
    Compute Euclidean distance between two points.

    Parameters
    ----------
    point_a, point_b
        Coordinate arrays (same dimension).

    Returns
    -------
    distance
        Euclidean distance.
    """
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))


def estimate_molecule_diameter(coords: np.ndarray) -> float:
    """
    Estimate the diameter of a molecule from its coordinates.

    This computes the maximum pairwise distance between atoms, which
    provides an upper bound on the molecular size. Useful for determining
    required supercell dimensions.

    Parameters
    ----------
    coords
        Cartesian coordinates, shape (N, 3).

    Returns
    -------
    diameter
        Maximum pairwise distance in Angstroms.
        
    Notes
    -----
    For large molecules (>100 atoms), consider using scipy.spatial.distance.pdist
    for better performance. This implementation prioritizes simplicity.
    """
    coords = np.asarray(coords, dtype=float)
    
    if len(coords) == 0:
        return 0.0
    if len(coords) == 1:
        return 0.0

    # For molecules with many atoms, vectorized approach is more efficient
    if len(coords) > 50:
        # Compute all pairwise distances at once using broadcasting
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        # Use nanmax to be robust against NaN values from diffusion models
        result = float(np.nanmax(distances))
        return result if np.isfinite(result) else 0.0
    
    # For small molecules, simple loop is fine and more memory-efficient
    max_dist = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist > max_dist:
                max_dist = dist

    return max_dist


def center_of_mass(coords: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the center of mass (or geometric center if masses not provided).

    Parameters
    ----------
    coords
        Cartesian coordinates, shape (N, 3).
    masses
        Atomic masses, shape (N,). If None, uses geometric center (equal weights).

    Returns
    -------
    com
        Center of mass, shape (3,).
    """
    if len(coords) == 0:
        return np.array([0.0, 0.0, 0.0])

    if masses is None:
        return np.mean(coords, axis=0)

    total_mass = np.sum(masses)
    if total_mass < 1e-12:
        return np.mean(coords, axis=0)

    return np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass


def project_onto_plane(
    point: np.ndarray,
    plane_normal: np.ndarray,
    plane_point: np.ndarray | None = None,
) -> np.ndarray:
    """
    Project a point onto a plane defined by a normal vector and a point.

    Parameters
    ----------
    point
        Point to project, shape (3,).
    plane_normal
        Normal vector to the plane, shape (3,).
    plane_point
        A point on the plane. If None, the plane passes through the origin.

    Returns
    -------
    projected
        Projected point on the plane, shape (3,).
    """
    if plane_point is None:
        plane_point = np.zeros(3)

    n = normalize_vector(plane_normal)
    v = point - plane_point
    distance = np.dot(v, n)
    return point - distance * n

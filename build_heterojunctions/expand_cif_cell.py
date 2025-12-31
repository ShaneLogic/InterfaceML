#!/usr/bin/env python3
"""
expand_cif_cell.py

Update CIF unit-cell parameters (a/b/c and volume), with two coordinate modes:

- fractional (default): keep fractional coordinates unchanged. This preserves the
  structure's relative position inside the unit cell.
- cartesian: keep Cartesian coordinates unchanged by recomputing fractional coords
  under the new cell. This is useful for "adding vacuum" around an isolated molecule.

This script is intentionally lightweight and does NOT require external libraries.
It supports typical CIFs where atomic positions are provided in fractional coordinates
inside an atom_site loop.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple


def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def _cell_vectors_from_abc_angles(
    a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Build lattice vectors (a_vec, b_vec, c_vec) in Cartesian coordinates (Å) from
    cell lengths and angles. Uses a standard crystallography convention:
      - a along x
      - b in xy-plane
      - c with positive z
    """
    alpha = _deg2rad(alpha_deg)
    beta = _deg2rad(beta_deg)
    gamma = _deg2rad(gamma_deg)

    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)
    if abs(sg) < 1e-12:
        raise ValueError("Invalid cell: gamma is too close to 0 or 180 degrees.")

    a_vec = (a, 0.0, 0.0)
    b_vec = (b * cg, b * sg, 0.0)

    c_x = c * cb
    c_y = c * (ca - cb * cg) / sg
    c_z_sq = c * c - c_x * c_x - c_y * c_y
    c_z = math.sqrt(max(c_z_sq, 0.0))
    c_vec = (c_x, c_y, c_z)
    return a_vec, b_vec, c_vec


def _mat_det_3x3(m: Sequence[Sequence[float]]) -> float:
    (a, b, c), (d, e, f), (g, h, i) = m
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _mat_inv_3x3(m: Sequence[Sequence[float]]) -> List[List[float]]:
    (a, b, c), (d, e, f), (g, h, i) = m
    det = _mat_det_3x3(m)
    if abs(det) < 1e-15:
        raise ValueError("Cell matrix is singular; cannot invert.")

    # Cofactor matrix (transposed for adjugate).
    inv = [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]
    return inv


def _mat_vec_mul(m: Sequence[Sequence[float]], v: Sequence[float]) -> Tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _frac_to_cart(
    frac: Tuple[float, float, float],
    a_vec: Tuple[float, float, float],
    b_vec: Tuple[float, float, float],
    c_vec: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Convert fractional to Cartesian: r = x*a + y*b + z*c."""
    x, y, z = frac
    return (
        x * a_vec[0] + y * b_vec[0] + z * c_vec[0],
        x * a_vec[1] + y * b_vec[1] + z * c_vec[1],
        x * a_vec[2] + y * b_vec[2] + z * c_vec[2],
    )


def _cart_to_frac(
    cart: Tuple[float, float, float],
    a_vec: Tuple[float, float, float],
    b_vec: Tuple[float, float, float],
    c_vec: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Convert Cartesian to fractional by solving:
      [a b c] * f = r
    where columns are lattice vectors.
    """
    # Build matrix with column vectors.
    m = [
        [a_vec[0], b_vec[0], c_vec[0]],
        [a_vec[1], b_vec[1], c_vec[1]],
        [a_vec[2], b_vec[2], c_vec[2]],
    ]
    inv = _mat_inv_3x3(m)
    fx, fy, fz = _mat_vec_mul(inv, cart)
    return fx, fy, fz


def _parse_key_value_float(line: str) -> Optional[Tuple[str, float]]:
    s = line.strip()
    if not s.startswith("_"):
        return None
    parts = s.split()
    if len(parts) < 2:
        return None
    key = parts[0]
    try:
        val = float(parts[1])
    except ValueError:
        return None
    return key, val


def _format_cif_float(x: float) -> str:
    # CIFs in this repo often use 6 or 8 decimals; 8 can be noisy. Use 6 for readability.
    return f"{x:.6f}"


def _find_atom_loop(lines: List[str]) -> Optional[Dict[str, object]]:
    """
    Find the first loop_ that contains atom fractional coordinate tags.
    Returns a dict with:
      - loop_start: int
      - tags_start: int
      - tags_end: int (exclusive)
      - data_start: int
      - data_end: int (exclusive)
      - tags: List[str] (stripped)
    """
    n = len(lines)
    i = 0
    while i < n:
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        tags: List[str] = []
        while j < n and lines[j].lstrip().startswith("_"):
            tags.append(lines[j].strip())
            j += 1

        tagset = set(tags)
        if (
            "_atom_site_fract_x" in tagset
            and "_atom_site_fract_y" in tagset
            and "_atom_site_fract_z" in tagset
        ):
            data_start = j
            data_end = data_start
            while data_end < n:
                s = lines[data_end].strip()
                if s == "" or s.startswith("loop_") or s.startswith("data_") or s.startswith("_"):
                    break
                data_end += 1
            return {
                "loop_start": i,
                "tags_start": i + 1,
                "tags_end": j,
                "data_start": data_start,
                "data_end": data_end,
                "tags": tags,
            }
        i = j
    return None


def expand_cif_cell_keep_cartesian(
    cif_text: str,
    new_a: float,
    new_b: float,
    new_c: float,
    *,
    mode: Literal["fractional", "cartesian"] = "fractional",
    center: bool = False,
    wrap_fractional: bool = False,
) -> str:
    """
    Update a CIF content string:
      - set _cell_length_{a,b,c} to new values
      - update _cell_volume
      - optionally update coordinates depending on mode
    """
    lines = cif_text.splitlines(keepends=True)

    cell: Dict[str, float] = {}
    cell_line_idx: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        parsed = _parse_key_value_float(line)
        if not parsed:
            continue
        key, val = parsed
        if key in {
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
            "_cell_volume",
        }:
            cell[key] = val
            cell_line_idx[key] = idx

    # Minimal required parameters for coordinate transforms.
    for req in [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    ]:
        if req not in cell:
            raise ValueError(f"Missing required CIF cell parameter: {req}")

    old_a = float(cell["_cell_length_a"])
    old_b = float(cell["_cell_length_b"])
    old_c = float(cell["_cell_length_c"])
    alpha = float(cell["_cell_angle_alpha"])
    beta = float(cell["_cell_angle_beta"])
    gamma = float(cell["_cell_angle_gamma"])

    if mode not in ("fractional", "cartesian"):
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == "fractional" and center:
        raise ValueError("--center is only meaningful in --mode cartesian.")

    old_vecs = _cell_vectors_from_abc_angles(old_a, old_b, old_c, alpha, beta, gamma)
    new_vecs = _cell_vectors_from_abc_angles(new_a, new_b, new_c, alpha, beta, gamma)

    # Update cell lengths.
    if "_cell_length_a" in cell_line_idx:
        lines[cell_line_idx["_cell_length_a"]] = f"_cell_length_a   {_format_cif_float(new_a)}\n"
    if "_cell_length_b" in cell_line_idx:
        lines[cell_line_idx["_cell_length_b"]] = f"_cell_length_b   {_format_cif_float(new_b)}\n"
    if "_cell_length_c" in cell_line_idx:
        lines[cell_line_idx["_cell_length_c"]] = f"_cell_length_c   {_format_cif_float(new_c)}\n"

    # Update or insert cell volume.
    new_cell_matrix = [
        [new_vecs[0][0], new_vecs[1][0], new_vecs[2][0]],
        [new_vecs[0][1], new_vecs[1][1], new_vecs[2][1]],
        [new_vecs[0][2], new_vecs[1][2], new_vecs[2][2]],
    ]
    new_vol = _mat_det_3x3(new_cell_matrix)
    if "_cell_volume" in cell_line_idx:
        lines[cell_line_idx["_cell_volume"]] = f"_cell_volume   {_format_cif_float(new_vol)}\n"
    else:
        # Insert after _cell_angle_gamma if present, else near top.
        insert_at = cell_line_idx.get("_cell_angle_gamma", 0) + 1
        lines.insert(insert_at, f"_cell_volume   {_format_cif_float(new_vol)}\n")

    loop_info = _find_atom_loop(lines)
    if loop_info is None:
        raise ValueError("Could not find an atom_site loop with fractional coordinates in CIF.")

    tags: List[str] = loop_info["tags"]  # type: ignore[assignment]
    tag_to_idx = {t: k for k, t in enumerate(tags)}
    ix = tag_to_idx["_atom_site_fract_x"]
    iy = tag_to_idx["_atom_site_fract_y"]
    iz = tag_to_idx["_atom_site_fract_z"]

    data_start = int(loop_info["data_start"])  # type: ignore[arg-type]
    data_end = int(loop_info["data_end"])  # type: ignore[arg-type]

    # Mode: keep fractional coordinates unchanged (preserve relative positions in the cell).
    if mode == "fractional":
        if wrap_fractional:
            # Optional cleanup: wrap existing fractional coords into [0, 1).
            for row_idx in range(data_start, data_end):
                raw = lines[row_idx].strip()
                if raw == "":
                    continue
                parts = raw.split()
                if len(parts) < len(tags):
                    raise ValueError(
                        f"Atom row has fewer columns than tags (line {row_idx+1}). "
                        f"cols={len(parts)} tags={len(tags)}"
                    )
                try:
                    fx0 = float(parts[ix]) % 1.0
                    fy0 = float(parts[iy]) % 1.0
                    fz0 = float(parts[iz]) % 1.0
                except ValueError as e:
                    raise ValueError(f"Failed to parse fractional coords at line {row_idx+1}: {e}") from e
                parts[ix] = _format_cif_float(fx0)
                parts[iy] = _format_cif_float(fy0)
                parts[iz] = _format_cif_float(fz0)
                lines[row_idx] = "  " + "  ".join(parts) + "\n"
        return "".join(lines)

    # Collect old Cartesian positions first so we can optionally translate the whole structure.
    old_carts: List[Tuple[float, float, float]] = []
    atom_rows: List[Tuple[int, List[str]]] = []

    for row_idx in range(data_start, data_end):
        raw = lines[row_idx].strip()
        if raw == "":
            continue
        parts = raw.split()
        if len(parts) < len(tags):
            raise ValueError(
                f"Atom row has fewer columns than tags (line {row_idx+1}). "
                f"cols={len(parts)} tags={len(tags)}"
            )

        try:
            fx0 = float(parts[ix])
            fy0 = float(parts[iy])
            fz0 = float(parts[iz])
        except ValueError as e:
            raise ValueError(f"Failed to parse fractional coords at line {row_idx+1}: {e}") from e

        cart = _frac_to_cart((fx0, fy0, fz0), *old_vecs)
        old_carts.append(cart)
        atom_rows.append((row_idx, parts))

    # Optional: translate the whole structure so its geometric center is at the center of the new cell.
    dx = dy = dz = 0.0
    if center and old_carts:
        cx = sum(p[0] for p in old_carts) / len(old_carts)
        cy = sum(p[1] for p in old_carts) / len(old_carts)
        cz = sum(p[2] for p in old_carts) / len(old_carts)

        # Cell center relative to origin is (a + b + c) / 2 for the chosen lattice-vector convention.
        new_center = (
            0.5 * (new_vecs[0][0] + new_vecs[1][0] + new_vecs[2][0]),
            0.5 * (new_vecs[0][1] + new_vecs[1][1] + new_vecs[2][1]),
            0.5 * (new_vecs[0][2] + new_vecs[1][2] + new_vecs[2][2]),
        )
        dx, dy, dz = new_center[0] - cx, new_center[1] - cy, new_center[2] - cz

    # Write updated fractional coordinates.
    for cart, (row_idx, parts) in zip(old_carts, atom_rows):
        cart2 = (cart[0] + dx, cart[1] + dy, cart[2] + dz)
        fx1, fy1, fz1 = _cart_to_frac(cart2, *new_vecs)

        if wrap_fractional:
            fx1 %= 1.0
            fy1 %= 1.0
            fz1 %= 1.0

        parts[ix] = _format_cif_float(fx1)
        parts[iy] = _format_cif_float(fy1)
        parts[iz] = _format_cif_float(fz1)
        lines[row_idx] = "  " + "  ".join(parts) + "\n"

    return "".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Update CIF unit cell lengths (a/b/c) and volume.\n"
            "Default mode keeps fractional coordinates unchanged (relative positions preserved)."
        )
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="Input CIF file(s).",
    )
    p.add_argument(
        "--target",
        type=float,
        default=20.0,
        help="Convenience: set a=b=c=target (default: 20.0). Overridden by --a/--b/--c if provided.",
    )
    p.add_argument("--a", type=float, default=None, help="Set new cell length a in Å (optional).")
    p.add_argument("--b", type=float, default=None, help="Set new cell length b in Å (optional).")
    p.add_argument("--c", type=float, default=None, help="Set new cell length c in Å (optional).")
    p.add_argument(
        "--mode",
        choices=["fractional", "cartesian"],
        default="fractional",
        help=(
            "Coordinate handling mode. "
            "'fractional' keeps fractional coordinates unchanged (preserve relative positions). "
            "'cartesian' keeps Cartesian coordinates unchanged by recomputing fractional coords."
        ),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory. If omitted, writes next to each input file.",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="_cell20",
        help="Suffix to append to output filename stem (default: _cell20).",
    )
    p.add_argument(
        "--center",
        action="store_true",
        help="Translate the structure so its geometric center is at the center of the new unit cell.",
    )
    p.add_argument(
        "--wrap",
        action="store_true",
        help="Wrap fractional coordinates back to [0, 1) after transforming/translating.",
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    target = float(args.target)

    for in_path_str in args.inputs:
        in_path = Path(in_path_str).expanduser().resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input CIF not found: {in_path}")

        cif_text = in_path.read_text(encoding="utf-8", errors="replace")
        # Determine new cell lengths. By default use --target for all; allow per-axis override.
        new_a = float(args.a) if args.a is not None else target
        new_b = float(args.b) if args.b is not None else target
        new_c = float(args.c) if args.c is not None else target
        out_text = expand_cif_cell_keep_cartesian(
            cif_text,
            new_a,
            new_b,
            new_c,
            mode=str(args.mode),
            center=bool(args.center),
            wrap_fractional=bool(args.wrap),
        )

        dest_dir = out_dir if out_dir is not None else in_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / f"{in_path.stem}{args.suffix}{in_path.suffix}"
        out_path.write_text(out_text, encoding="utf-8")

        print(f"Wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



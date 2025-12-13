"""Compute difference charge density from CP2K Gaussian cube files.

Formula (heterojunctions):
    Δρ(r) = ρ_interface − ρ_layerA − ρ_layerB

Outputs a Gaussian .cube file that can be visualized in VESTA.

This implementation is streaming (constant-memory) and only assumes that the
three inputs share the same volumetric grid definition (origin + 3 axis lines).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO, Tuple


@dataclass(frozen=True)
class CubeGrid:
    natoms_raw: int
    origin: Tuple[float, float, float]
    nvox: Tuple[int, int, int]
    axis: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


@dataclass
class CubeHeader:
    # raw header lines to preserve formatting for output
    raw_lines: list[str]
    grid: CubeGrid


def _parse_grid_from_header_lines(line3: str, axis_lines: list[str]) -> CubeGrid:
    p3 = line3.split()
    if len(p3) < 4:
        raise ValueError(f"Invalid cube header line (natoms+origin): {line3!r}")

    natoms_raw = int(p3[0])
    origin = (float(p3[1]), float(p3[2]), float(p3[3]))

    nvox = []
    axis = []
    if len(axis_lines) != 3:
        raise ValueError("Cube header must contain 3 axis lines")

    for ln in axis_lines:
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid cube axis line: {ln!r}")
        nvox.append(int(parts[0]))
        axis.append((float(parts[1]), float(parts[2]), float(parts[3])))

    return CubeGrid(
        natoms_raw=natoms_raw,
        origin=origin,
        nvox=(nvox[0], nvox[1], nvox[2]),
        axis=(axis[0], axis[1], axis[2]),
    )


def read_cube_header(fp: TextIO) -> CubeHeader:
    comment1 = fp.readline()
    comment2 = fp.readline()
    if not comment1 or not comment2:
        raise ValueError("Cube file ended before comment lines")

    line3 = fp.readline()
    if not line3:
        raise ValueError("Cube file ended before natoms+origin line")

    axis_lines = [fp.readline(), fp.readline(), fp.readline()]
    if any(not ln for ln in axis_lines):
        raise ValueError("Cube file ended before 3 axis lines")

    grid = _parse_grid_from_header_lines(line3, axis_lines)
    natoms = abs(grid.natoms_raw)

    atom_lines = []
    for _ in range(natoms):
        ln = fp.readline()
        if not ln:
            raise ValueError("Cube file ended before atom block finished")
        atom_lines.append(ln)

    raw_lines = [comment1, comment2, line3, *axis_lines, *atom_lines]
    return CubeHeader(raw_lines=raw_lines, grid=grid)


def iter_cube_values(fp: TextIO) -> Iterator[float]:
    for line in fp:
        s = line.strip()
        if not s:
            continue
        for tok in s.split():
            yield float(tok)


def _close_enough(a: float, b: float, atol: float = 1e-5, rtol: float = 1e-6) -> bool:
    # CP2K cube headers are often printed with limited decimals; allow small rounding differences.
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))


def _assert_same_grid(
    ref: CubeGrid, other: CubeGrid, label: str, *, atol: float = 1e-5, rtol: float = 1e-6
) -> None:
    if ref.nvox != other.nvox:
        raise ValueError(f"Grid size mismatch vs interface for {label}: {other.nvox} != {ref.nvox}")

    for i in range(3):
        if not _close_enough(ref.origin[i], other.origin[i], atol=atol, rtol=rtol):
            raise ValueError(
                f"Origin mismatch vs interface for {label}: {other.origin} != {ref.origin}"
            )

    for ax in range(3):
        for j in range(3):
            if not _close_enough(ref.axis[ax][j], other.axis[ax][j], atol=atol, rtol=rtol):
                raise ValueError(
                    f"Axis vector mismatch vs interface for {label} (axis {ax}): {other.axis[ax]} != {ref.axis[ax]}"
                )


def write_delta_density_cube(
    interface_path: Path,
    layer_a_path: Path,
    layer_b_path: Path,
    output_path: Path,
    values_per_line: int = 6,
) -> None:
    with interface_path.open("r") as f_int, layer_a_path.open("r") as f_a, layer_b_path.open("r") as f_b:
        h_int = read_cube_header(f_int)
        h_a = read_cube_header(f_a)
        h_b = read_cube_header(f_b)

        _assert_same_grid(h_int.grid, h_a.grid, label=str(layer_a_path.name))
        _assert_same_grid(h_int.grid, h_b.grid, label=str(layer_b_path.name))

        nx, ny, nz = h_int.grid.nvox
        npts = nx * ny * nz

        it_int = iter_cube_values(f_int)
        it_a = iter_cube_values(f_a)
        it_b = iter_cube_values(f_b)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as out:
            out.writelines(h_int.raw_lines)

            for i in range(npts):
                try:
                    v = next(it_int) - next(it_a) - next(it_b)
                except StopIteration as e:
                    raise ValueError(
                        "One of the cube files ended before providing all grid values. "
                        "Check that all three files have the same grid and are complete."
                    ) from e

                # Gaussian cube commonly uses 6 values per line with scientific notation.
                out.write(f" {v:13.5E}")
                if (i + 1) % values_per_line == 0:
                    out.write("\n")

            if npts % values_per_line != 0:
                out.write("\n")


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Compute difference charge density cube: Δρ = ρ_interface − ρ_layerA − ρ_layerB (streaming)."
        )
    )
    p.add_argument(
        "--interface",
        required=True,
        type=Path,
        help="Interface electron density cube (e.g., MAPbI3/TiO2).",
    )
    p.add_argument(
        "--layerA",
        required=True,
        type=Path,
        help="Layer A electron density cube (e.g., MAPbI3).",
    )
    p.add_argument(
        "--layerB",
        required=True,
        type=Path,
        help="Layer B electron density cube (e.g., TiO2).",
    )
    p.add_argument(
        "-o",
        "--output",
        default=Path("delta-density.cube"),
        type=Path,
        help="Output cube filename (default: delta-density.cube).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    write_delta_density_cube(
        interface_path=args.interface,
        layer_a_path=args.layerA,
        layer_b_path=args.layerB,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

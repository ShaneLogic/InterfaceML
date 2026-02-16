#!/usr/bin/env python3
"""
High-throughput interface builder for perovskite/adsorbate stacks.

This script iterates over perovskite and fullerene structure folders and
builds interface models using interfaceml.core.adsorbate utilities.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shlex
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from interfaceml.core import io
from interfaceml.core import adsorbate as adsorbate_core

logger = logging.getLogger(__name__)


def _parse_miller(value: str) -> Tuple[int, int, int]:
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"Invalid miller index: {value}")
    return tuple(int(p) for p in parts)


def _parse_supercell(value: str) -> Optional[Tuple[int, int]]:
    if value.lower() == "auto":
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"Invalid supercell: {value}")
    return int(parts[0]), int(parts[1])


def _parse_xy(value: str) -> Tuple[float, float]:
    parts = value.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"Invalid xy: {value}")
    return float(parts[0]), float(parts[1])


def _collect_files(path: Path, exts: Sequence[str]) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for ext in exts:
        files.extend(path.rglob(f"*{ext}"))
    return sorted({p for p in files if p.is_file()})


def _sample(items: List[Path], limit: Optional[int], rng: random.Random) -> List[Path]:
    if limit is None or limit <= 0 or len(items) <= limit:
        return items
    return rng.sample(items, limit)


def _write_record(fp, record: dict) -> None:
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    fp.flush()


def _min_interlayer_distance(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    *,
    chunk: int = 2048,
) -> float:
    if coords_a.size == 0 or coords_b.size == 0:
        return float("inf")
    min_d2 = float("inf")
    for i in range(0, len(coords_a), chunk):
        block = coords_a[i : i + chunk]
        diff = block[:, None, :] - coords_b[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        local_min = float(np.min(d2))
        if local_min < min_d2:
            min_d2 = local_min
    return float(np.sqrt(min_d2))


def _parse_energy(output: str) -> Optional[float]:
    matches = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", output)
    if not matches:
        return None
    return float(matches[0])


def _rotation_matrices(
    mode: str,
    count: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    if count <= 1:
        return [np.eye(3)]

    mode = mode.lower()
    if mode == "z":
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        return [adsorbate_core.rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), a) for a in angles]

    if mode == "random3d":
        mats: List[np.ndarray] = []
        for _ in range(count):
            u1, u2, u3 = rng.random(3)
            q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
            q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
            q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
            q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
            rot = np.array(
                [
                    [1 - 2 * (q3 ** 2 + q4 ** 2), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
                    [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 ** 2 + q4 ** 2), 2 * (q3 * q4 - q1 * q2)],
                    [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2)],
                ],
                dtype=float,
            )
            mats.append(rot)
        return mats

    raise ValueError(f"Unknown orientation mode: {mode}")


def _xy_positions(
    *,
    xy_single: Tuple[float, float],
    xy_grid: Optional[Tuple[int, int]],
    xy_samples: int,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    if xy_grid:
        nx, ny = xy_grid
        xs = [(i + 0.5) / nx for i in range(nx)]
        ys = [(j + 0.5) / ny for j in range(ny)]
        return [(x, y) for x in xs for y in ys]

    if xy_samples and xy_samples > 0:
        return [(float(rng.random()), float(rng.random())) for _ in range(xy_samples)]

    return [xy_single]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch build perovskite/adsorbate interfaces")
    parser.add_argument("--perovskite_dir", required=True, help="Folder (or file) for perovskite structures")
    parser.add_argument("--fullerene_dir", required=True, help="Folder (or file) for fullerene structures")
    parser.add_argument("--output_dir", required=True, help="Output folder for interface structures")
    parser.add_argument("--metadata", default=None, help="Path to JSONL metadata output (default: <output_dir>/metadata.jsonl)")

    parser.add_argument("--miller", default="0,0,1", help="Miller index for perovskite slab (e.g., 0,0,1)")
    parser.add_argument("--slab_thickness", type=float, default=18.0, help="Perovskite slab thickness (A)")
    parser.add_argument("--vacuum", type=float, default=20.0, help="Vacuum thickness (A)")
    parser.add_argument("--distance", type=float, default=3.2, help="Initial separation between slab and adsorbate (A)")
    parser.add_argument("--supercell", default="auto", help="In-plane supercell (nx,ny) or 'auto'")
    parser.add_argument("--buffer", type=float, default=10.0, help="Buffer for auto supercell sizing (A)")
    parser.add_argument("--xy", default="0.5,0.5", help="Adsorbate XY position in fractional coords (fx,fy)")
    parser.add_argument("--xy_grid", default=None, help="Grid for XY sampling (nx,ny)")
    parser.add_argument("--xy_samples", type=int, default=0, help="Number of random XY samples (if no grid)")
    parser.add_argument("--orientation_samples", type=int, default=1, help="Number of orientations per pair")
    parser.add_argument("--orientation_mode", default="z", choices=["z", "random3d"], help="Orientation sampling mode")
    parser.add_argument("--termination", default="auto", help="Termination selection (auto/PbI/AI)")
    parser.add_argument("--layer_tol", type=float, default=1.5, help="Top/bottom layer tolerance (A)")

    parser.add_argument("--relax_cmd", default=None, help="Optional relaxation command template with {input} {output}")
    parser.add_argument("--relax_output_dir", default=None, help="Output folder for relaxed structures")
    parser.add_argument("--min_interlayer_dist", type=float, default=None, help="Minimum allowed distance between layers (A)")
    parser.add_argument("--keep_overlap", action="store_true", help="Keep structures that fail overlap filter")
    parser.add_argument("--energy_cmd", default=None, help="Optional energy command template with {input}")
    parser.add_argument("--energy_threshold", type=float, default=None, help="Filter if energy exceeds this threshold")
    parser.add_argument("--energy_use_relaxed", action="store_true", help="Use relaxed structure for energy if available")

    parser.add_argument("--max_perovskites", type=int, default=None, help="Limit number of perovskite inputs")
    parser.add_argument("--max_fullerenes", type=int, default=None, help="Limit number of fullerene inputs")
    parser.add_argument("--max_pairs", type=int, default=None, help="Limit total number of pairs (random sample)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for sampling")
    parser.add_argument("--skip_existing", action="store_true", help="Skip outputs that already exist")
    parser.add_argument("--dry_run", action="store_true", help="Only print planned jobs")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    perovskite_dir = Path(args.perovskite_dir)
    fullerene_dir = Path(args.fullerene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata) if args.metadata else output_dir / "metadata.jsonl"

    perovskites = _collect_files(perovskite_dir, [".cif", ".vasp", ".poscar"])
    fullerenes = _collect_files(fullerene_dir, [".xyz", ".cif", ".vasp", ".poscar"])

    perovskites = _sample(perovskites, args.max_perovskites, rng)
    fullerenes = _sample(fullerenes, args.max_fullerenes, rng)

    pairs = list(product(perovskites, fullerenes))
    if args.max_pairs is not None and args.max_pairs > 0 and len(pairs) > args.max_pairs:
        pairs = rng.sample(pairs, args.max_pairs)

    miller = _parse_miller(args.miller)
    supercell = _parse_supercell(args.supercell)
    xy_frac = _parse_xy(args.xy)
    xy_grid = _parse_supercell(args.xy_grid) if args.xy_grid else None

    if args.dry_run:
        logger.info("Planned jobs: %d", len(pairs))
        for idx, (p_path, f_path) in enumerate(pairs[:10], 1):
            logger.info("%04d: %s + %s", idx, p_path.name, f_path.name)
        if len(pairs) > 10:
            logger.info("... (truncated)")
        return

    orientation_mats = _rotation_matrices(args.orientation_mode, args.orientation_samples, np_rng)
    xy_positions = _xy_positions(
        xy_single=xy_frac,
        xy_grid=xy_grid,
        xy_samples=args.xy_samples,
        rng=np_rng,
    )

    relax_output_dir = None
    if args.relax_cmd:
        relax_output_dir = Path(args.relax_output_dir) if args.relax_output_dir else (output_dir / "relaxed")
        relax_output_dir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("a", encoding="utf-8") as meta_fp:
        for idx, (p_path, f_path) in enumerate(pairs, 1):
            record_base = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "perovskite_path": str(p_path),
                "fullerene_path": str(f_path),
                "miller": list(miller),
                "slab_thickness": args.slab_thickness,
                "vacuum": args.vacuum,
                "distance": args.distance,
                "supercell": None,
                "xy_frac": list(xy_frac),
                "termination": args.termination,
                "output_path": None,
                "n_atoms": None,
                "status": "pending",
                "error": None,
            }

            try:
                perovskite = io.load_structure(p_path)
                fullerene = io.load_structure(f_path)

                for o_idx, rot in enumerate(orientation_mats, 1):
                    for x_idx, xy_pos in enumerate(xy_positions, 1):
                        bottom, top, combined, choice = adsorbate_core.build_adsorbate_interface(
                            perovskite,
                            fullerene,
                            miller=miller,
                            slab_thickness=args.slab_thickness,
                            vacuum=args.vacuum,
                            separation=args.distance,
                            supercell_xy=supercell,
                            buffer=args.buffer,
                            xy_frac=xy_pos,
                            termination=args.termination,
                            layer_tol=args.layer_tol,
                            rotation=rot,
                        )

                        nx = choice.nx
                        ny = choice.ny
                        out_name = (
                            f"{p_path.stem}__{f_path.stem}_m{miller[0]}{miller[1]}{miller[2]}"
                            f"_sc{nx}x{ny}_d{args.distance:.2f}"
                            f"_o{o_idx:02d}_x{x_idx:02d}.vasp"
                        )
                        out_path = output_dir / out_name

                        record = dict(record_base)
                        record.update(
                            {
                                "supercell": [nx, ny],
                                "xy_frac": [float(xy_pos[0]), float(xy_pos[1])],
                                "orientation_index": o_idx,
                                "xy_index": x_idx,
                                "termination_selected": choice.termination,
                                "termination_top": list(choice.termination_top),
                                "termination_bottom": list(choice.termination_bottom),
                            }
                        )

                        min_dist = _min_interlayer_distance(
                            np.asarray(bottom.cart_coords, dtype=float),
                            np.asarray(top.cart_coords, dtype=float),
                        )
                        record["min_interlayer_dist"] = min_dist

                        if args.min_interlayer_dist is not None and min_dist < args.min_interlayer_dist:
                            record["status"] = "filtered_overlap"
                            record["filter_reason"] = f"min_dist<{args.min_interlayer_dist}"
                            if args.keep_overlap:
                                record["note"] = "kept_overlap"
                            else:
                                _write_record(meta_fp, record)
                                continue

                        if args.skip_existing and out_path.exists():
                            record.update(
                                {
                                    "status": "skipped",
                                    "output_path": str(out_path),
                                    "n_atoms": int(len(combined)),
                                }
                            )
                            _write_record(meta_fp, record)
                            continue

                        io.write_poscar(
                            combined,
                            out_path,
                            comment=(
                                f"Interface: {p_path.name} + {f_path.name} | "
                                f"supercell={nx}x{ny} | d={args.distance:.2f}A | "
                                f"o={o_idx} | xy={xy_pos[0]:.3f},{xy_pos[1]:.3f}"
                            ),
                        )

                        record.update(
                            {
                                "status": "success",
                                "output_path": str(out_path),
                                "n_atoms": int(len(combined)),
                            }
                        )

                        if args.relax_cmd:
                            relaxed_path = relax_output_dir / out_path.name
                            cmd = args.relax_cmd.format(input=str(out_path), output=str(relaxed_path))
                            try:
                                subprocess.run(shlex.split(cmd), check=True)
                                record["relax_output_path"] = str(relaxed_path)
                                record["relax_status"] = "success"
                            except Exception as exc:
                                record["relax_status"] = "failed"
                                record["relax_error"] = str(exc)

                        if args.energy_cmd:
                            energy_input = out_path
                            if args.energy_use_relaxed and record.get("relax_status") == "success":
                                energy_input = Path(record["relax_output_path"])
                            cmd = args.energy_cmd.format(input=str(energy_input))
                            try:
                                result = subprocess.run(
                                    shlex.split(cmd),
                                    check=True,
                                    capture_output=True,
                                    text=True,
                                )
                                energy = _parse_energy(result.stdout.strip())
                                if energy is None:
                                    raise RuntimeError("Energy parse failed")
                                record["energy"] = energy
                                record["energy_status"] = "success"
                                if args.energy_threshold is not None and energy > args.energy_threshold:
                                    record["status"] = "filtered_energy"
                                    record["filter_reason"] = f"energy>{args.energy_threshold}"
                            except Exception as exc:
                                record["energy_status"] = "failed"
                                record["energy_error"] = str(exc)

                        _write_record(meta_fp, record)

            except Exception as exc:
                record_base.update({"status": "failed", "error": str(exc)})
                _write_record(meta_fp, record_base)
                logger.error("[%d/%d] Failed: %s + %s -> %s", idx, len(pairs), p_path.name, f_path.name, exc)
                continue


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

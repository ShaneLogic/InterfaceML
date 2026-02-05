"""
Routes for interface building, layer fixing, and splitting.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from interfaceml.web.utils import compress_ranges, format_ranges


bp = Blueprint("interface", __name__)


@bp.route('/api/build-interface', methods=['POST'])
def build_interface():
    """Build a heterojunction interface (placeholder)."""
    data = request.json
    _ = data
    return jsonify({
        'status': 'not_implemented',
        'message': 'Interface building functionality coming soon'
    }), 501


@bp.route('/api/build-adsorbate', methods=['POST'])
def build_adsorbate():
    """Build an adsorbate model (placeholder)."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({'error': 'Core modules not available'}), 500

    data = request.json
    _ = data
    return jsonify({
        'status': 'not_implemented',
        'message': 'Adsorbate building functionality coming soon'
    }), 501


@bp.route('/api/fix-layers', methods=['POST'])
def fix_layers():
    """Add selective dynamics to a structure."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({'error': 'Core modules not available'}), 500

    data = request.json
    structure_file = data.get('structure_file')

    if not structure_file or not Path(structure_file).exists():
        return jsonify({'error': 'Invalid structure file'}), 400

    try:
        structure = core_status.io.load_structure(structure_file)
        mode = data.get('mode', 'by_z_layers')
        n_fix = data.get('n_fix_layers', 3)
        include_mols = data.get('include_molecules', True)

        if mode == 'by_z_layers':
            layers, _ = core_status.layering.split_layers_by_z(structure, gap_cut=True)

            if n_fix > len(layers):
                return jsonify({
                    'error': f'Only {len(layers)} layers found, cannot fix {n_fix}'
                }), 400

            fixed_indices = []
            for layer in layers[:n_fix]:
                fixed_indices.extend(layer)

            if include_mols:
                fixed_indices = core_status.layering.include_whole_molecules(
                    structure,
                    fixed_indices
                )

            selective_dynamics = []
            fixed_set = set(fixed_indices)
            for i in range(len(structure)):
                if i in fixed_set:
                    selective_dynamics.append((False, False, False))
                else:
                    selective_dynamics.append((True, True, True))

            output_filename = Path(structure_file).stem + "_fixed.vasp"
            output_path = Path(current_app.config['UPLOAD_FOLDER']) / output_filename
            core_status.io.write_poscar(structure, output_path, selective_dynamics=selective_dynamics)

            return jsonify({
                'status': 'success',
                'n_layers_total': len(layers),
                'n_layers_fixed': n_fix,
                'n_atoms_fixed': len(fixed_indices),
                'output_file': str(output_path),
                'download_url': f'/api/download/{output_filename}',
                'fixed_indices': sorted(fixed_indices)
            })

        return jsonify({'error': f'Unknown mode: {mode}'}), 400

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@bp.route('/api/split-layers', methods=['POST'])
def split_layers():
    """Split a multi-layer structure into separate layer files."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available or not core_status.splitting_available:
        return jsonify({'error': 'Core splitting module not available'}), 500

    data = request.json
    structure_file = data.get('structure_file')
    n_interfaces = data.get('n_interfaces', 1)

    if not structure_file or not Path(structure_file).exists():
        return jsonify({'error': 'Invalid structure file'}), 400

    try:
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"

        structure = core_status.io.load_structure(structure_file)

        input_hash = hashlib.sha256(Path(structure_file).read_bytes()).hexdigest()[:12]

        min_gap = float(data.get('min_gap', 1.5))
        use_smart = data.get('use_smart_detection', True)

        layers = core_status.splitting.split_structure_into_layers(
            structure,
            n_interfaces=n_interfaces,
            min_gap=min_gap,
            use_smart_detection=use_smart
        )

        base_name = Path(structure_file).stem
        layer_info = []

        normal = core_status.layering.interface_normal_unit(structure)
        all_heights = np.dot(structure.cart_coords, normal)

        for i, layer_struct in enumerate(layers, start=1):
            layer_type = core_status.splitting.identify_layer_type(layer_struct)
            output_filename = f"{base_name}_{run_id}_layer{i}_{layer_type.replace(' ', '_')}.vasp"
            output_path = Path(current_app.config['UPLOAD_FOLDER']) / output_filename

            layer_indices = []
            for site_idx, site in enumerate(structure):
                if any(np.allclose(site.frac_coords, layer_struct[j].frac_coords)
                       for j in range(len(layer_struct))):
                    layer_indices.append(site_idx)

            layer_heights = all_heights[layer_indices] if layer_indices else [0]
            z_min = float(np.min(layer_heights))
            z_max = float(np.max(layer_heights))

            core_status.io.write_poscar(
                layer_struct,
                output_path,
                comment=f"{base_name} - Layer {i}/{len(layers)}: {layer_type}"
            )

            output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()[:12]

            layer_info.append({
                'layer_number': i,
                'n_atoms': len(layer_struct),
                'composition': layer_struct.composition.formula,
                'layer_type': layer_type,
                'z_range': f"{z_min:.2f} - {z_max:.2f} Å",
                'filename': output_filename,
                'download_url': f'/api/download/{output_filename}?v={run_id}',
                'file_hash': output_hash,
            })

        splitting_file = getattr(core_status.splitting, '__file__', None)
        splitting_hash = None
        try:
            if splitting_file and Path(splitting_file).exists():
                splitting_hash = hashlib.sha256(Path(splitting_file).read_bytes()).hexdigest()[:12]
        except Exception:
            splitting_hash = None

        return jsonify({
            'status': 'success',
            'run_id': run_id,
            'input_file': str(structure_file),
            'input_hash': input_hash,
            'splitting_file': splitting_file,
            'splitting_hash': splitting_hash,
            'n_interfaces': n_interfaces,
            'n_layers': len(layers),
            'layers': layer_info,
            'message': f'Successfully split into {len(layers)} layers'
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@bp.route('/api/pdos-layers', methods=['POST'])
def pdos_layers():
    """Generate layer-resolved atom indices for CP2K PDOS input."""
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({'error': 'Core modules not available'}), 500

    if not core_status.splitting_available:
        return jsonify({'error': 'Core splitting module not available'}), 500

    data = request.json or {}
    structure_file = data.get('structure_file')

    if not structure_file or not Path(structure_file).exists():
        return jsonify({'error': 'Invalid structure file'}), 400

    try:
        structure = core_status.io.load_structure(structure_file)
        n_interfaces = int(data.get('n_interfaces', 1))
        min_gap = float(data.get('min_gap', 1.5))
        use_smart = bool(data.get('use_smart_detection', True))

        layers_struct = core_status.splitting.split_structure_into_layers(
            structure,
            n_interfaces=n_interfaces,
            min_gap=min_gap,
            use_smart_detection=use_smart,
        )

        heights = np.dot(
            np.asarray(structure.cart_coords, dtype=float),
            core_status.layering.interface_normal_unit(structure),
        )

        layers: list[list[int]] = []
        for layer_struct in layers_struct:
            layer_indices = []
            for site_idx, site in enumerate(structure):
                if any(
                    np.allclose(site.frac_coords, layer_struct[j].frac_coords)
                    for j in range(len(layer_struct))
                ):
                    layer_indices.append(site_idx)
            layers.append(sorted(set(layer_indices)))

        pdos_filename = data.get('pdos_filename') or f"{Path(structure_file).stem}_PDOS"
        nlumo = int(data.get('nlumo', 250))

        layer_info = []
        ldos_blocks = []
        for idx, layer in enumerate(layers, start=1):
            indices_0 = sorted(int(i) for i in layer)
            indices_1 = [i + 1 for i in indices_0]
            ranges_1 = format_ranges(compress_ranges(indices_1))
            z_min = float(np.min(heights[indices_0])) if indices_0 else 0.0
            z_max = float(np.max(heights[indices_0])) if indices_0 else 0.0

            layer_info.append({
                'layer_number': idx,
                'n_atoms': len(indices_0),
                'indices_0': indices_0,
                'indices_1': indices_1,
                'ranges_1': ranges_1,
                'z_range': f"{z_min:.2f} - {z_max:.2f} Å",
            })

            ldos_blocks.append(
                "\n".join([
                    "&LDOS",
                    f"  LIST {ranges_1}",
                    "&END LDOS",
                ])
            )

        pdos_block = "\n".join(
            [
                "&PDOS",
                f"  FILENAME {pdos_filename}",
                f"  NLUMO {nlumo}",
                "  COMPONENTS",
                *ldos_blocks,
                "&END PDOS",
            ]
        )

        return jsonify({
            'status': 'success',
            'input_file': str(structure_file),
            'n_layers': len(layers),
            'n_interfaces': n_interfaces,
            'min_gap': min_gap,
            'use_smart_detection': use_smart,
            'layers': layer_info,
            'cp2k_pdos': pdos_block,
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

"""
Flask web application for InterfaceML.

This module provides a browser-based interface for heterojunction modeling,
allowing users to upload structure files and interactively configure
modeling parameters.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import secrets
import hashlib
import re
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import core functionality (will be implemented)
try:
    from interfaceml.core import io, layering
    try:
        from interfaceml.core import splitting
        SPLITTING_AVAILABLE = True
    except ImportError:
        SPLITTING_AVAILABLE = False
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    SPLITTING_AVAILABLE = False

# Import Fullerene AI module
fullerene_diffusion_path = Path(__file__).parent.parent.parent / 'fullerene_diffusion_poc'
if fullerene_diffusion_path.exists():
    sys.path.insert(0, str(fullerene_diffusion_path))
    try:
        from api import FullereneAPI
        AI_MODULE_AVAILABLE = True
        # Initialize AI API with checkpoint
        checkpoint_path = fullerene_diffusion_path / 'checkpoints' / 'best_model.pt'
        if checkpoint_path.exists():
            fullerene_api = FullereneAPI(str(checkpoint_path))
            print(f"✓ Fullerene AI module loaded: {checkpoint_path}")
        else:
            AI_MODULE_AVAILABLE = False
            fullerene_api = None
            print(f"⚠ Fullerene checkpoint not found: {checkpoint_path}")
    except Exception as e:
        AI_MODULE_AVAILABLE = False
        fullerene_api = None
        print(f"⚠ Failed to load Fullerene AI: {e}")
else:
    AI_MODULE_AVAILABLE = False
    fullerene_api = None
    print("⚠ Fullerene diffusion module not found")

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix='interfaceml_')
app.config['ALLOWED_EXTENSIONS'] = {'cif', 'vasp', 'poscar', 'xyz', 'dos', 'pdos'}

# Ensure upload folder exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def compress_ranges(indices: list[int]) -> list[tuple[int, int]]:
    """Compress sorted indices into inclusive ranges."""
    if not indices:
        return []

    sorted_idx = sorted(set(indices))
    ranges: list[tuple[int, int]] = []
    start = sorted_idx[0]
    prev = sorted_idx[0]

    for value in sorted_idx[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = value
        prev = value

    ranges.append((start, prev))
    return ranges


def format_ranges(ranges: list[tuple[int, int]]) -> str:
    """Format ranges for CP2K LIST syntax (e.g., 1..12 14 18..25)."""
    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(f"{start}")
        else:
            parts.append(f"{start}..{end}")
    return " ".join(parts)


def read_cp2k_dos_like(
    filepath: Path,
) -> tuple[np.ndarray, np.ndarray, float | None, str, str]:
    """
    Read CP2K .dos/.pdos files and extract Fermi energy when available.

    Parameters
    ----------
    filepath
        Path to the DOS/PDOS file.

    Returns
    -------
    energy
        Energy axis in eV.
    data
        Numeric columns after energy.
    ef
        Fermi energy in eV if present in header, otherwise None.
    unit
        Original energy unit detected in the header ("a.u." or "eV").
    header_text
        Full header text for label extraction.
    """
    ef = None
    header_lines: list[str] = []

    with open(filepath, "r", errors="ignore") as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                header_lines.append(line.strip())
            else:
                break

    header_text = "\n".join(header_lines)
    patterns = [
        r"Fermi\s*energy.*?([-\d\.Ee+]+)",
        r"E\s*\(\s*Fermi\s*\)\s*=?\s*([-\d\.Ee+]+)",
        r"Efermi\s*=?\s*([-\d\.Ee+]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, header_text, flags=re.IGNORECASE)
        if match:
            try:
                ef = float(match.group(1))
                break
            except ValueError:
                continue

    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    header_text = "\n".join(header_lines)
    unit = "eV"
    if "a.u." in header_text or "au" in header_text.lower():
        unit = "a.u."

    # CP2K PDOS format often includes an index column followed by energy and occupation.
    has_index_column = False
    if data.shape[1] >= 3:
        idx_col = data[:, 0]
        if np.allclose(idx_col, np.round(idx_col), atol=1e-6):
            has_index_column = True

    if has_index_column:
        energy = data[:, 1]
        rest = data[:, 3:] if data.shape[1] > 3 else np.zeros((len(energy), 0))
    else:
        energy = data[:, 0]
        rest = data[:, 1:] if data.shape[1] > 1 else np.zeros((len(energy), 0))

    if unit == "a.u.":
        hartree_to_ev = 27.211386245988
        energy = energy * hartree_to_ev
        if ef is not None:
            ef = ef * hartree_to_ev

    return energy, rest, ef, unit, header_text


def extract_pdos_meta(header_text: str, fallback: str) -> tuple[str, int | None]:
    """Extract PDOS label and atom count from CP2K header text."""
    match = re.search(
        r"list\s+(\d+)\s+of\s+(\d+)\s+atoms",
        header_text,
        flags=re.IGNORECASE,
    )
    if match:
        return f"List {match.group(1)} ({match.group(2)} atoms)", int(match.group(2))
    return fallback, None


def gaussian_broaden(
    energies: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Apply Gaussian broadening to discrete DOS data."""
    if sigma <= 0:
        return np.interp(grid, energies, weights, left=0.0, right=0.0)

    coeff = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    broadened = np.zeros_like(grid, dtype=float)
    for energy, weight in zip(energies, weights):
        broadened += weight * np.exp(-0.5 * ((grid - energy) / sigma) ** 2)
    return coeff * broadened


def merge_overlapping_layers(
    layers: list[list[int]],
    heights: np.ndarray,
) -> list[list[int]]:
    """Merge overlapping layers and return sorted layers by mean height."""
    pending = [set(layer) for layer in layers if layer]
    merged: list[set[int]] = []

    while pending:
        current = pending.pop()
        changed = True
        while changed:
            changed = False
            for other in list(pending):
                if current & other:
                    current.update(other)
                    pending.remove(other)
                    changed = True
        merged.append(current)

    merged_lists = [sorted(layer) for layer in merged]
    merged_lists.sort(
        key=lambda g: float(np.mean(heights[np.array(g, dtype=int)])) if g else 0.0
    )
    return merged_lists


@app.route('/')
def index():
    """Render the main web interface."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    splitting_file = None
    splitting_hash = None
    if SPLITTING_AVAILABLE:
        try:
            import hashlib

            splitting_file = getattr(splitting, '__file__', None)
            if splitting_file and Path(splitting_file).exists():
                data = Path(splitting_file).read_bytes()
                splitting_hash = hashlib.sha256(data).hexdigest()[:12]
        except Exception:
            splitting_file = splitting_file or None
            splitting_hash = splitting_hash or None

    return jsonify({
        'status': 'ok',
        'core_available': CORE_AVAILABLE,
        'splitting_available': SPLITTING_AVAILABLE,
        'splitting_file': splitting_file,
        'splitting_hash': splitting_hash,
        'version': '1.0.0'
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload.

    Expected POST data:
        file: The structure file
        file_type: Type of file (base, adsorbate, etc.)
        
    Returns
    -------
    JSON response with file info or error message.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_type = request.form.get('file_type', 'structure')

    if not file.filename or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        allowed_exts = ', '.join(app.config['ALLOWED_EXTENSIONS'])
        return jsonify({
            'error': f'Invalid file type. Allowed: {allowed_exts}'
        }), 400

    try:
        # Save file securely with sanitized filename
        filename = secure_filename(file.filename)
        if not filename:  # Ensure filename is not empty after sanitization
            return jsonify({'error': 'Invalid filename'}), 400
            
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(filepath))

        # Compute hash to help users confirm they're operating on the expected file.
        file_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()[:12]

        # Try to parse the structure and extract basic info
        info: Dict[str, Any] = {
            'filename': filename,
            'filepath': str(filepath),
            'file_type': file_type,
            'file_hash': file_hash,
        }

        # Skip structure parsing for DOS/PDOS files (identified by type prefix or extension)
        is_dos_file = (
            file_type in {"tdos", "dos"} or 
            file_type.startswith("pdos") or
            filename.lower().endswith(('.dos', '.pdos'))
        )
        
        if CORE_AVAILABLE and not is_dos_file:
            try:
                structure = io.load_structure(filepath)
                info['n_atoms'] = len(structure)
                info['composition'] = structure.composition.formula
                info['lattice_abc'] = [
                    round(structure.lattice.a, 3),
                    round(structure.lattice.b, 3),
                    round(structure.lattice.c, 3),
                ]
                info['lattice_angles'] = [
                    round(structure.lattice.alpha, 2),
                    round(structure.lattice.beta, 2),
                    round(structure.lattice.gamma, 2),
                ]
            except Exception as e:
                info['parse_error'] = str(e)
                # Don't return error, just include parse error in response

        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': f'File upload failed: {str(e)}'}), 500


@app.route('/api/build-interface', methods=['POST'])
def build_interface():
    """
    Build a heterojunction interface.

    Expected JSON data:
        {
            "base_file": "path/to/base.cif",
            "film_file": "path/to/film.cif",
            "miller_base": [0, 0, 1],
            "miller_film": [0, 0, 1],
            "max_area": 400,
            "strain_mode": "bidirectional"
        }
    """
    if not CORE_AVAILABLE:
        return jsonify({'error': 'Core modules not available'}), 500

    data = request.json
    # TODO: Implement interface building logic
    return jsonify({
        'status': 'not_implemented',
        'message': 'Interface building functionality coming soon'
    }), 501


@app.route('/api/build-adsorbate', methods=['POST'])
def build_adsorbate():
    """
    Build an adsorbate model (molecule on slab).

    Expected JSON data:
        {
            "base_file": "path/to/perovskite.cif",
            "adsorbate_files": ["path/to/c60.cif", "path/to/c70.cif"],
            "termination": "PbI",
            "miller": [0, 0, 1],
            "distances": [2.5, 2.0],
            "supercell": "auto"
        }
    """
    if not CORE_AVAILABLE:
        return jsonify({'error': 'Core modules not available'}), 500

    data = request.json
    # TODO: Implement adsorbate building logic
    return jsonify({
        'status': 'not_implemented',
        'message': 'Adsorbate building functionality coming soon'
    }), 501


@app.route('/api/fix-layers', methods=['POST'])
def fix_layers():
    """
    Add selective dynamics to a structure.

    Expected JSON data:
        {
            "structure_file": "path/to/structure.vasp",
            "mode": "by_z_layers",
            "n_fix_layers": 3,
            "include_molecules": true
        }
    """
    if not CORE_AVAILABLE:
        return jsonify({'error': 'Core modules not available'}), 500

    data = request.json
    structure_file = data.get('structure_file')

    if not structure_file or not Path(structure_file).exists():
        return jsonify({'error': 'Invalid structure file'}), 400

    try:
        structure = io.load_structure(structure_file)
        mode = data.get('mode', 'by_z_layers')
        n_fix = data.get('n_fix_layers', 3)
        include_mols = data.get('include_molecules', True)

        if mode == 'by_z_layers':
            layers, tol = layering.split_layers_by_z(structure, gap_cut=True)

            if n_fix > len(layers):
                return jsonify({
                    'error': f'Only {len(layers)} layers found, cannot fix {n_fix}'
                }), 400

            fixed_indices = []
            for layer in layers[:n_fix]:
                fixed_indices.extend(layer)

            if include_mols:
                fixed_indices = layering.include_whole_molecules(
                    structure,
                    fixed_indices
                )

            # Create output with selective dynamics
            selective_dynamics = []
            fixed_set = set(fixed_indices)
            for i in range(len(structure)):
                if i in fixed_set:
                    selective_dynamics.append((False, False, False))  # Fixed
                else:
                    selective_dynamics.append((True, True, True))  # Relaxed

            # Write output POSCAR
            output_filename = Path(structure_file).stem + "_fixed.vasp"
            output_path = Path(app.config['UPLOAD_FOLDER']) / output_filename
            io.write_poscar(structure, output_path, selective_dynamics=selective_dynamics)

            return jsonify({
                'status': 'success',
                'n_layers_total': len(layers),
                'n_layers_fixed': n_fix,
                'n_atoms_fixed': len(fixed_indices),
                'output_file': str(output_path),
                'download_url': f'/api/download/{output_filename}',
                'fixed_indices': sorted(fixed_indices)
            })

        else:
            return jsonify({'error': f'Unknown mode: {mode}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/split-layers', methods=['POST'])
def split_layers():
    """
    Split a multi-layer structure into separate layer files.

    Expected JSON data:
        {
            "structure_file": "path/to/structure.vasp",
            "n_interfaces": 2,
            "min_gap": 0.0
        }
    
    Returns:
        JSON with layer information and download links
    """
    if not CORE_AVAILABLE or not SPLITTING_AVAILABLE:
        return jsonify({'error': 'Core splitting module not available'}), 500

    data = request.json
    structure_file = data.get('structure_file')
    n_interfaces = data.get('n_interfaces', 1)

    if not structure_file or not Path(structure_file).exists():
        return jsonify({'error': 'Invalid structure file'}), 400

    try:
        from interfaceml.core import io, splitting

        # Unique run id to prevent confusion from same-name downloads.
        # Example: 20260105-001234-ab12cd34
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
        
        # Load structure
        structure = io.load_structure(structure_file)

        input_hash = hashlib.sha256(Path(structure_file).read_bytes()).hexdigest()[:12]
        
        # Get parameters
        min_gap = float(data.get('min_gap', 1.5))  # Default 1.5 Å for smart detection
        use_smart = data.get('use_smart_detection', True)
        
        # Split into layers using smart composition-aware algorithm
        layers = splitting.split_structure_into_layers(
            structure,
            n_interfaces=n_interfaces,
            min_gap=min_gap,
            use_smart_detection=use_smart
        )
        
        # Save each layer with enhanced information
        base_name = Path(structure_file).stem
        output_files = []
        download_urls = []
        layer_info = []
        
        for i, layer_struct in enumerate(layers, start=1):
            # Identify layer type
            layer_type = splitting.identify_layer_type(layer_struct)
            
            # Create descriptive filename
            output_filename = f"{base_name}_{run_id}_layer{i}_{layer_type.replace(' ', '_')}.vasp"
            output_path = Path(app.config['UPLOAD_FOLDER']) / output_filename
            
            # Get height range for this layer
            from interfaceml.core.layering import interface_normal_unit
            import numpy as np
            normal = interface_normal_unit(structure)
            all_heights = np.dot(structure.cart_coords, normal)
            
            # Find indices for this layer in original structure
            layer_indices = []
            for site_idx, site in enumerate(structure):
                if any(np.allclose(site.frac_coords, layer_struct[j].frac_coords) 
                       for j in range(len(layer_struct))):
                    layer_indices.append(site_idx)
            
            layer_heights = all_heights[layer_indices] if layer_indices else [0]
            z_min = float(np.min(layer_heights))
            z_max = float(np.max(layer_heights))
            
            # Write structure
            io.write_poscar(
                layer_struct,
                output_path,
                comment=f"{base_name} - Layer {i}/{len(layers)}: {layer_type}"
            )

            output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()[:12]
            
            output_files.append(str(output_path))
            download_urls.append(f'/api/download/{output_filename}?v={run_id}')
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
        
        splitting_file = getattr(splitting, '__file__', None)
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
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pdos-layers', methods=['POST'])
def pdos_layers():
    """
    Generate layer-resolved atom indices for CP2K PDOS input.

    Expected JSON data:
        {
            "structure_file": "path/to/structure.cif",
            "n_interfaces": 2,
            "min_gap": 1.5,
            "use_smart_detection": true,
            "pdos_filename": "my_pdos",
            "nlumo": 250
        }
    """
    if not CORE_AVAILABLE:
        return jsonify({'error': 'Core modules not available'}), 500

    if not SPLITTING_AVAILABLE:
        return jsonify({'error': 'Core splitting module not available'}), 500

    data = request.json or {}
    structure_file = data.get('structure_file')

    if not structure_file or not Path(structure_file).exists():
        return jsonify({'error': 'Invalid structure file'}), 400

    try:
        structure = io.load_structure(structure_file)
        n_interfaces = int(data.get('n_interfaces', 1))
        min_gap = float(data.get('min_gap', 1.5))
        use_smart = bool(data.get('use_smart_detection', True))

        layers_struct = splitting.split_structure_into_layers(
            structure,
            n_interfaces=n_interfaces,
            min_gap=min_gap,
            use_smart_detection=use_smart,
        )

        heights = np.dot(
            np.asarray(structure.cart_coords, dtype=float),
            layering.interface_normal_unit(structure),
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

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plot-dos', methods=['POST'])
def plot_dos():
    """
    Plot TDOS and PDOS curves into a single PNG.

    Expected JSON data:
        {
            "tdos_file": "path/to/tdos.dos",
            "pdos_files": ["path/to/pdos1.pdos", "path/to/pdos2.pdos"],
            "title": "Custom plot title",
            "output_name": "dos_plot",
            "sigma": 0.12,
            "grid_step": 0.02,
            "normalize": false,
            "x_min": -10,
            "x_max": 6,
            "pdos_scale": "per_atom",
            "tdos_scale": "none",
            "tdos_total_atoms": 240
        }
    """
    data = request.json or {}
    tdos_file = data.get('tdos_file')
    pdos_files = data.get('pdos_files', [])

    if not tdos_file or not Path(tdos_file).exists():
        return jsonify({'error': 'Invalid TDOS file'}), 400

    valid_pdos = [Path(p) for p in pdos_files if p and Path(p).exists()]
    if not valid_pdos:
        return jsonify({'error': 'No valid PDOS files provided'}), 400

    try:
        energy_t, tcols, ef_t, unit_t, header_t = read_cp2k_dos_like(Path(tdos_file))
        if tcols.shape[1] < 1:
            return jsonify({'error': 'TDOS file contains no numeric columns'}), 400

        # TDOS files in CP2K can be ambiguous:
        # - Some builds output an actual DOS density column.
        # - Others output an "Occupation" count per energy bin.
        #   In that case, the physically comparable DOS is occupation / ΔE.
        tdos_source = (data.get('tdos_source') or 'auto').lower()
        has_occupation_col = (tcols.shape[1] >= 2) and (re.search(r"\bOccupation\b", header_t, flags=re.IGNORECASE) is not None)

        # Ensure increasing energy for interpolation/smoothing.
        order = np.argsort(energy_t)
        energy_t = energy_t[order]
        tcols = tcols[order]

        if tdos_source not in {'auto', 'density', 'occupation'}:
            return jsonify({'error': 'Invalid tdos_source (use auto|density|occupation)'}), 400

        # Auto mode heuristic: if an Occupation column exists and looks like counts,
        # prefer deriving DOS from Occupation/ΔE.
        if tdos_source == 'auto':
            if has_occupation_col:
                occ_max = float(np.nanmax(tcols[:, 1]))
                dens_max = float(np.nanmax(tcols[:, 0]))
                if occ_max > 1.0 and dens_max < 10.0:
                    tdos_source = 'occupation'
                else:
                    tdos_source = 'density'
            else:
                tdos_source = 'density'

        if tdos_source == 'occupation' and not has_occupation_col:
            return jsonify({'error': 'TDOS occupation column not found in header/data'}), 400

        if tdos_source == 'occupation':
            # Convert counts-per-bin into DOS density (states/eV) via ΔE.
            # Problem: Occupation column is 0 for unoccupied states (E > E_F).
            # Solution: Calculate scaling factor from occupied region, then apply to Density column.
            diffs = np.diff(energy_t)
            delta_e = float(np.median(np.abs(diffs[diffs != 0]))) if diffs.size else 0.0
            if delta_e <= 0:
                return jsonify({'error': 'Cannot infer TDOS energy bin width (ΔE)'}), 400

            occupation_col = tcols[:, 1]
            density_col = tcols[:, 0]

            # Find occupied region where Occupation > 0 and Density > 0
            occupied_mask = (occupation_col > 0.5) & (density_col > 1e-6)
            if np.sum(occupied_mask) > 5:
                # Calculate DOS from Occupation in occupied region
                dos_from_occ = occupation_col[occupied_mask] / delta_e
                # Calculate scaling factor: DOS_true / Density
                scale_factors = dos_from_occ / density_col[occupied_mask]
                scale_factor = float(np.median(scale_factors))
                # Apply scaling to entire Density column (works for both occupied and unoccupied)
                tdos = density_col * scale_factor
            else:
                # Fallback: use Occupation/ΔE directly (will be 0 for unoccupied states)
                tdos = occupation_col / delta_e
        else:
            tdos = tcols[:, 0]

        pdos_series = []
        ef_candidates = [ef_t]
        for pdos_path in valid_pdos:
            energy_p, cols_p, ef_p, unit_p, header_p = read_cp2k_dos_like(pdos_path)
            ef_candidates.append(ef_p)
            if cols_p.size == 0:
                continue
            label, atom_count = extract_pdos_meta(header_p, pdos_path.stem)
            pdos_series.append({
                'energy': energy_p,
                'dos': np.sum(cols_p, axis=1),
                'label': label,
                'atom_count': atom_count,
            })

        if not pdos_series:
            return jsonify({'error': 'PDOS files contain no DOS columns'}), 400

        ef = next((value for value in ef_candidates if value is not None), None)

        if ef is not None:
            x_t = energy_t - ef
            x_label = "Energy (E − E_F) [eV]"
        else:
            x_t = energy_t
            x_label = "Energy [eV]"

        sigma = float(data.get('sigma', 0.12))
        grid_step = float(data.get('grid_step', 0.02))
        normalize = bool(data.get('normalize', False))
        pdos_scale = data.get('pdos_scale', 'per_atom') or 'per_atom'
        tdos_scale = data.get('tdos_scale', 'none') or 'none'
        tdos_total_atoms = data.get('tdos_total_atoms')
        tdos_total_atoms = int(tdos_total_atoms) if tdos_total_atoms not in (None, "") else None
        x_min = data.get('x_min')
        x_max = data.get('x_max')

        x_min = float(x_min) if x_min not in (None, "") else float(np.min(x_t))
        x_max = float(x_max) if x_max not in (None, "") else float(np.max(x_t))
        if x_max <= x_min:
            return jsonify({'error': 'Invalid energy range (x_max must be > x_min)'}), 400

        grid = np.arange(x_min, x_max + grid_step * 0.5, grid_step)
        tdos_values = tdos
        if tdos_scale == 'per_atom':
            if tdos_total_atoms is None:
                pdos_atom_sum = sum(
                    series.get('atom_count') or 0 for series in pdos_series
                )
                if pdos_atom_sum > 0:
                    tdos_total_atoms = pdos_atom_sum
            if tdos_total_atoms:
                tdos_values = tdos_values / float(tdos_total_atoms)

        # For TDOS (already on an energy grid), use interpolation + optional smoothing.
        tdos_interp = np.interp(grid, x_t, tdos_values, left=0.0, right=0.0)
        if sigma > 0:
            radius = int(max(3, round((4.0 * sigma) / grid_step)))
            kernel_x = np.arange(-radius, radius + 1) * grid_step
            kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
            kernel = kernel / np.sum(kernel)
            tdos_smooth = np.convolve(tdos_interp, kernel, mode='same')
        else:
            tdos_smooth = tdos_interp
        if normalize and np.max(tdos_smooth) > 0:
            tdos_smooth = tdos_smooth / np.max(tdos_smooth)

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9 if len(pdos_series) > 5 else 10,
        })

        # Adjust figure width for many PDOS curves
        fig_width = 7.6 if len(pdos_series) <= 5 else 8.5
        fig, ax = plt.subplots(figsize=(fig_width, 4.8))
        ax.plot(grid, tdos_smooth, label="TDOS", linewidth=1.8, color="#111827")

        # Extended color palette for many PDOS files
        palette = [
            "#2563eb",  # Blue
            "#f97316",  # Orange
            "#16a34a",  # Green
            "#7c3aed",  # Purple
            "#dc2626",  # Red
            "#0891b2",  # Cyan
            "#ca8a04",  # Yellow
            "#be185d",  # Pink
            "#4f46e5",  # Indigo
            "#059669",  # Emerald
            "#9333ea",  # Violet
            "#ea580c",  # Deep Orange
        ]
        for idx, series in enumerate(pdos_series):
            x_p = series['energy'] - ef if ef is not None else series['energy']
            pdos_values = series['dos']
            if pdos_scale == 'per_atom' and series.get('atom_count'):
                pdos_values = pdos_values / float(series['atom_count'])
            pdos_smooth = gaussian_broaden(x_p, pdos_values, grid, sigma)
            if pdos_scale == 'match_tdos' and np.max(pdos_smooth) > 0:
                pdos_smooth = pdos_smooth * (np.max(tdos_smooth) / np.max(pdos_smooth))
            if normalize and np.max(pdos_smooth) > 0:
                pdos_smooth = pdos_smooth / np.max(pdos_smooth)
            ax.plot(
                grid,
                pdos_smooth,
                label=series['label'],
                linewidth=1.4,
                color=palette[idx % len(palette)],
                alpha=0.9,
            )

        if ef is not None:
            ax.axvline(0.0, linewidth=0.9, linestyle="--", color="#334155")

        title = data.get('title') or "TDOS + PDOS"
        ax.set_title(title)
        ax.set_xlabel(x_label)
        y_label = "DOS (states/eV)"
        if tdos_scale == 'per_atom':
            y_label = "DOS (states/eV/atom)"
        ax.set_ylabel(y_label)
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(x_min, x_max)
        fig.tight_layout()

        output_name = data.get('output_name') or "dos_tdos_pdos_overlay"
        output_filename = secure_filename(f"{output_name}.png")
        output_path = Path(app.config['UPLOAD_FOLDER']) / output_filename
        fig.savefig(output_path, dpi=220)
        plt.close(fig)

        # Collect debug info for transparency
        tdos_peak = float(np.max(tdos_smooth)) if tdos_smooth.size else 0.0
        pdos_peaks = {
            series['label']: float(np.max(gaussian_broaden(
                series['energy'] - ef if ef else series['energy'],
                series['dos'] / (float(series['atom_count']) if pdos_scale == 'per_atom' and series.get('atom_count') else 1.0),
                grid, sigma
            ))) for series in pdos_series
        }

        return jsonify({
            'status': 'success',
            'ef': ef,
            'tdos_source_used': tdos_source,
            'tdos_total_atoms_used': tdos_total_atoms,
            'tdos_peak': round(tdos_peak, 2),
            'pdos_peaks': {k: round(v, 2) for k, v in pdos_peaks.items()},
            'image_file': str(output_path),
            'image_url': f"/api/download/{output_filename}",
            'download_url': f"/api/download/{output_filename}",
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/generate', methods=['POST'])
def ai_generate_structures():
    """AI-powered fullerene structure generation endpoint."""
    if not AI_MODULE_AVAILABLE or fullerene_api is None:
        return jsonify({
            'status': 'error',
            'error': 'AI module not available. Please ensure fullerene_diffusion_poc is properly configured.'
        }), 503
    
    try:
        data = request.get_json() or {}
        num_atoms = int(data.get('num_atoms', 60))
        num_samples = int(data.get('num_samples', 5))
        output_format = data.get('output_format', 'xyz')
        
        # Validate parameters
        if num_atoms < 20 or num_atoms > 240:
            return jsonify({'status': 'error', 'error': 'num_atoms must be between 20 and 240'}), 400
        if num_samples < 1 or num_samples > 50:
            return jsonify({'status': 'error', 'error': 'num_samples must be between 1 and 50'}), 400
        
        # Create output directory
        output_dir = Path(app.config['UPLOAD_FOLDER']) / f'ai_generated_{secrets.token_hex(8)}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate structures
        start_time = time.time()
        results = fullerene_api.generate(
            num_atoms=num_atoms,
            num_samples=num_samples,
            output_dir=str(output_dir),
            output_format=output_format
        )
        generation_time = time.time() - start_time
        
        # Read generated structures for return
        structures = []
        if output_format == 'xyz' and results.get('generated_files'):
            for filepath in results['generated_files'][:10]:  # Limit to first 10 for response
                try:
                    with open(filepath, 'r') as f:
                        structures.append({
                            'filename': Path(filepath).name,
                            'content': f.read(),
                            'download_url': f"/api/download/{Path(filepath).name}"
                        })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
        
        return jsonify({
            'status': 'success',
            'num_generated': results.get('num_generated', 0),
            'success_rate': results.get('success_rate', 0.0),
            'generation_time': round(generation_time, 2),
            'output_dir': str(output_dir.relative_to(app.config['UPLOAD_FOLDER'])),
            'structures': structures,
            'message': f"Generated {results.get('num_generated', 0)} {output_format.upper()} structures in {generation_time:.1f}s"
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/ai/evaluate', methods=['POST'])
def ai_evaluate_structures():
    """AI evaluation of generated structures."""
    if not AI_MODULE_AVAILABLE or fullerene_api is None:
        return jsonify({
            'status': 'error',
            'error': 'AI module not available'
        }), 503
    
    try:
        data = request.get_json() or {}
        generated_dir = data.get('generated_dir')
        
        if not generated_dir:
            return jsonify({'status': 'error', 'error': 'generated_dir is required'}), 400
        
        # Resolve path relative to upload folder
        gen_path = Path(app.config['UPLOAD_FOLDER']) / generated_dir
        if not gen_path.exists():
            return jsonify({'status': 'error', 'error': 'Generated directory not found'}), 404
        
        # Reference directory (optional)
        ref_dir = fullerene_diffusion_path / 'dataset' / 'fullerenes' / 'fullerene_xyz'
        if not ref_dir.exists():
            ref_dir = None
        
        # Output directory for evaluation results
        eval_output = Path(app.config['UPLOAD_FOLDER']) / f'evaluation_{secrets.token_hex(8)}'
        eval_output.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation
        results = fullerene_api.evaluate(
            generated_dir=str(gen_path),
            reference_dir=str(ref_dir) if ref_dir else None,
            output_dir=str(eval_output)
        )
        
        # Find generated plot
        plot_files = list(eval_output.glob('evaluation_*.png'))
        plot_url = f"/api/download/{plot_files[0].name}" if plot_files else None
        
        return jsonify({
            'status': 'success',
            'metrics': results.get('metrics', {}),
            'plot_url': plot_url,
            'evaluation_dir': str(eval_output.relative_to(app.config['UPLOAD_FOLDER']))
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/ai/info', methods=['GET'])
def ai_model_info():
    """Get AI model information."""
    if not AI_MODULE_AVAILABLE or fullerene_api is None:
        return jsonify({
            'status': 'error',
            'available': False,
            'error': 'AI module not available'
        }), 503
    
    try:
        info = fullerene_api.get_model_info()
        info['available'] = True
        return jsonify(info)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'available': False,
            'error': str(e)
        }), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download a generated file."""
    filepath = Path(app.config['UPLOAD_FOLDER']) / secure_filename(filename)
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    # Prevent browsers from caching downloads with the same URL/filename across runs.
    # This is critical because we often overwrite files with identical names.
    resp = send_file(str(filepath), as_attachment=True)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


def main():
    """Main entry point for the web application."""
    import argparse
    parser = argparse.ArgumentParser(description="InterfaceML Web Server")
    parser.add_argument('--port', type=int, default=5000, help='Port to run on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  InterfaceML - Professional Heterojunction Modeling Platform")
    print("="*60)
    print(f"\n📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"⚙️  Core modules: {'✓ Available' if CORE_AVAILABLE else '✗ Not available'}")
    print(f"🤖 AI Generation: {'✓ Available' if AI_MODULE_AVAILABLE else '✗ Not available'}")
    print(f"\n🌐 Server: http://localhost:{args.port}")
    print("\n" + "="*60 + "\n")
    app.run(debug=True, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

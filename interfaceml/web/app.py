"""
Flask web application for InterfaceML.

This module provides a browser-based interface for heterojunction modeling,
allowing users to upload structure files and interactively configure
modeling parameters.
"""

from __future__ import annotations

import os
import tempfile
import time
import secrets
import hashlib
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

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

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix='interfaceml_')
app.config['ALLOWED_EXTENSIONS'] = {'cif', 'vasp', 'poscar', 'xyz'}

# Ensure upload folder exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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

        if CORE_AVAILABLE:
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
    
    print("Starting InterfaceML Web Server...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Core modules available: {CORE_AVAILABLE}")
    print(f"\nOpen your browser and navigate to: http://localhost:{args.port}")
    app.run(debug=True, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

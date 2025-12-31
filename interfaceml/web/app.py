"""
Flask web application for InterfaceML.

This module provides a browser-based interface for heterojunction modeling,
allowing users to upload structure files and interactively configure
modeling parameters.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import core functionality (will be implemented)
try:
    from interfaceml.core import io, layering
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

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
    return jsonify({
        'status': 'ok',
        'core_available': CORE_AVAILABLE,
        'version': '1.0.0'
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload.

    Expected POST data:
        file: The structure file
        file_type: Type of file (base, adsorbate, etc.)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_type = request.form.get('file_type', 'structure')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save file securely
    filename = secure_filename(file.filename)
    filepath = Path(app.config['UPLOAD_FOLDER']) / filename
    file.save(str(filepath))

    # Try to parse the structure and extract basic info
    info: Dict[str, Any] = {
        'filename': filename,
        'filepath': str(filepath),
        'file_type': file_type,
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

    return jsonify(info)


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


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download a generated file."""
    filepath = Path(app.config['UPLOAD_FOLDER']) / secure_filename(filename)
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_file(str(filepath), as_attachment=True)


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

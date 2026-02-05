"""
Routes for AI-powered structure generation.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from interfaceml.ai.interface_generator import InterfaceAIGenerator


bp = Blueprint("ai", __name__)


@bp.route('/api/ai/generate', methods=['POST'])
def ai_generate_structures():
    """AI-powered fullerene structure generation endpoint."""
    ai_status = current_app.extensions.get("ai_status")
    if not ai_status or not ai_status.available or ai_status.api is None:
        detail = None
        if ai_status and ai_status.error:
            detail = ai_status.error
        return jsonify({
            'status': 'error',
            'error': detail or 'AI module not available. Please ensure fullerene_diffusion_poc is properly configured.'
        }), 503

    try:
        data = request.get_json() or {}
        num_atoms = int(data.get('num_atoms', 60))
        num_samples = int(data.get('num_samples', 5))
        output_format = str(data.get('output_format', 'xyz')).lower()

        if num_atoms < 20 or num_atoms > 240:
            return jsonify({'status': 'error', 'error': 'num_atoms must be between 20 and 240'}), 400
        if num_samples < 1 or num_samples > 50:
            return jsonify({'status': 'error', 'error': 'num_samples must be between 1 and 50'}), 400
        if output_format not in {'xyz', 'json'}:
            return jsonify({'status': 'error', 'error': 'output_format must be xyz or json'}), 400

        output_dir = Path(current_app.config['UPLOAD_FOLDER'])
        run_id = secrets.token_hex(8)

        start_time = time.time()
        results = ai_status.api.generate(num_carbon=num_atoms, num_samples=num_samples)
        generation_time = time.time() - start_time

        structures = []
        for idx, result in enumerate(results):
            filename = f"ai_{run_id}_C{num_atoms}_sample_{idx:03d}.{output_format}"
            filepath = output_dir / filename
            ai_status.api.save_structure(
                result['positions'],
                result['edges'],
                str(filepath),
                format=output_format
            )

            if len(structures) < 10:
                content = None
                if output_format == 'xyz':
                    try:
                        content = filepath.read_text()
                    except Exception as exc:
                        print(f"Error reading {filepath}: {exc}")
                structures.append({
                    'filename': filename,
                    'content': content,
                    'download_url': f"/api/download/{filename}"
                })
        num_generated = len(results)
        success_rate = (num_generated / num_samples) if num_samples else 0.0

        return jsonify({
            'status': 'success',
            'num_generated': num_generated,
            'success_rate': success_rate,
            'generation_time': round(generation_time, 2),
            'output_dir': '.',
            'structures': structures,
            'message': f"Generated {num_generated} {output_format.upper()} structures in {generation_time:.1f}s"
        })

    except Exception as exc:
        return jsonify({
            'status': 'error',
            'error': str(exc)
        }), 500


@bp.route('/api/ai/generate-interface', methods=['POST'])
def ai_generate_interface():
    """Generate an interface using EGNN (global) + optional local GNN refinement."""
    ai_status = current_app.extensions.get("ai_status")
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({'status': 'error', 'error': 'Core modules not available'}), 500
    if not ai_status or not ai_status.available or ai_status.api is None:
        detail = None
        if ai_status and ai_status.error:
            detail = ai_status.error
        return jsonify({
            'status': 'error',
            'error': detail or 'AI module not available. Please ensure fullerene_diffusion_poc is properly configured.'
        }), 503

    data = request.get_json() or {}
    base_filename = data.get('base_filename') or data.get('base_file')
    if not base_filename:
        return jsonify({'status': 'error', 'error': 'base_filename is required'}), 400

    upload_folder = Path(current_app.config['UPLOAD_FOLDER'])
    safe_name = secure_filename(str(base_filename))
    base_path = (upload_folder / safe_name).resolve()
    try:
        base_path.relative_to(upload_folder.resolve())
    except ValueError:
        return jsonify({'status': 'error', 'error': 'Invalid base_filename path'}), 400

    if not base_path.exists():
        return jsonify({'status': 'error', 'error': 'Base structure file not found'}), 404

    try:
        base_structure = core_status.io.load_structure(base_path)
    except Exception as exc:
        return jsonify({'status': 'error', 'error': f'Failed to load base structure: {exc}'}), 400

    num_atoms = int(data.get('num_atoms', 60))
    if num_atoms < 20 or num_atoms > 240 or num_atoms % 2 != 0:
        return jsonify({'status': 'error', 'error': 'num_atoms must be even and between 20 and 240'}), 400

    miller = _parse_int_tuple(data.get('miller'), default=(0, 0, 1))
    xy_frac = _parse_float_tuple(data.get('xy_frac'), default=(0.5, 0.5))
    supercell_xy = _parse_int_tuple(data.get('supercell_xy'), length=2, default=None)

    termination = data.get('termination')
    slab_thickness = float(data.get('slab_thickness', 18.0))
    vacuum = float(data.get('vacuum', 20.0))
    separation = float(data.get('separation', 3.2))
    buffer = float(data.get('buffer', 10.0))
    layer_tol = float(data.get('layer_tol', 1.5))
    ddim = bool(data.get('ddim', False))

    refine_scope = str(data.get('refine_scope', 'adsorbate')).lower()
    if refine_scope not in {'adsorbate', 'interface', 'all'}:
        refine_scope = 'adsorbate'
    local_refine = bool(data.get('local_refine', True))

    generator = InterfaceAIGenerator(
        fullerene_api=ai_status.api,
        local_refiner=ai_status.local_refiner if local_refine else None,
    )

    start_time = time.time()
    try:
        result = generator.generate_interface(
            base_structure=base_structure,
            num_atoms=num_atoms,
            num_samples=1,
            ddim=ddim,
            miller=miller,
            slab_thickness=slab_thickness,
            vacuum=vacuum,
            separation=separation,
            supercell_xy=supercell_xy,
            buffer=buffer,
            xy_frac=xy_frac,
            termination=termination,
            layer_tol=layer_tol,
            refine_scope=refine_scope,
        )
    except Exception as exc:
        return jsonify({'status': 'error', 'error': f'Interface generation failed: {exc}'}), 500

    generation_time = time.time() - start_time
    run_id = secrets.token_hex(6)
    output_filename = f"ai_interface_{run_id}.vasp"
    output_path = upload_folder / output_filename
    try:
        core_status.io.write_poscar(result.combined, output_path)
    except Exception as exc:
        return jsonify({'status': 'error', 'error': f'Failed to write output: {exc}'}), 500

    return jsonify({
        'status': 'success',
        'output_file': str(output_path),
        'download_url': f"/api/download/{output_filename}",
        'generation_time': round(generation_time, 2),
        'metadata': result.metadata,
        'n_atoms': len(result.combined),
        'base_filename': safe_name,
        'local_refiner_error': getattr(ai_status, 'local_error', None),
    })


@bp.route('/api/ai/evaluate', methods=['POST'])
def ai_evaluate_structures():
    """AI evaluation of generated structures."""
    ai_status = current_app.extensions.get("ai_status")
    if not ai_status or not ai_status.available or ai_status.api is None:
        return jsonify({
            'status': 'error',
            'error': 'AI module not available'
        }), 503

    try:
        data = request.get_json() or {}
        positions = data.get('positions')
        edges = data.get('edges')
        compute_advanced = bool(data.get('compute_advanced_metrics', True))

        if positions is not None and edges is not None:
            metrics = ai_status.api.evaluate(
                positions=np.array(positions),
                edges=edges,
                compute_advanced_metrics=compute_advanced
            )
            return jsonify({
                'status': 'success',
                'metrics': metrics,
                'plot_url': None
            })

        generated_dir = data.get('generated_dir')
        if not generated_dir:
            return jsonify({'status': 'error', 'error': 'generated_dir or positions/edges is required'}), 400

        gen_path = Path(current_app.config['UPLOAD_FOLDER']) / generated_dir
        if not gen_path.exists():
            return jsonify({'status': 'error', 'error': 'Generated directory not found'}), 404

        try:
            from evaluate import parse_xyz_file  # type: ignore
        except Exception as exc:
            return jsonify({'status': 'error', 'error': f'Could not import evaluator: {exc}'}), 500

        xyz_files = sorted(gen_path.glob('*.xyz'))
        if not xyz_files:
            return jsonify({'status': 'error', 'error': 'No XYZ files found'}), 400

        metrics_by_file = []
        for filepath in xyz_files:
            positions, edges = parse_xyz_file(filepath)
            metrics = ai_status.api.evaluate(
                positions=positions,
                edges=edges,
                compute_advanced_metrics=compute_advanced
            )
            metrics_by_file.append({
                'filename': filepath.name,
                'metrics': metrics
            })

        return jsonify({
            'status': 'success',
            'metrics': metrics_by_file,
            'plot_url': None
        })

    except Exception as exc:
        return jsonify({
            'status': 'error',
            'error': str(exc)
        }), 500


@bp.route('/api/ai/info', methods=['GET'])
def ai_model_info():
    """Get AI model information."""
    ai_status = current_app.extensions.get("ai_status")
    if not ai_status or not ai_status.available or ai_status.api is None:
        detail = None
        checkpoint_path = None
        base_path = None
        if ai_status:
            detail = ai_status.error
            checkpoint_path = str(ai_status.checkpoint_path) if ai_status.checkpoint_path else None
            base_path = str(ai_status.base_path) if ai_status.base_path else None
        return jsonify({
            'status': 'error',
            'available': False,
            'error': detail or 'AI module not available',
            'checkpoint_path': checkpoint_path,
            'base_path': base_path
        }), 503

    try:
        info = ai_status.api.get_model_info()
        info['available'] = True
        return jsonify(info)
    except Exception as exc:
        return jsonify({
            'status': 'error',
            'available': False,
            'error': str(exc)
        }), 500


def _parse_int_tuple(value, *, length: int = 3, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == length:
        try:
            return tuple(int(x) for x in value)
        except Exception:
            return default
    return default


def _parse_float_tuple(value, *, length: int = 2, default=None):
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == length:
        try:
            return tuple(float(x) for x in value)
        except Exception:
            return default
    return default

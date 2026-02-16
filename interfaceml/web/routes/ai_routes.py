"""
Routes for AI-powered structure generation.
"""

from __future__ import annotations

import logging
import secrets
import time
from pathlib import Path

import numpy as np
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from interfaceml.ai.interface_generator import InterfaceAIGenerator

logger = logging.getLogger(__name__)


bp = Blueprint("ai", __name__)


@bp.route('/api/ai/models', methods=['GET'])
def ai_list_models():
    """List available model backends and their status."""
    ai_status = current_app.extensions.get("ai_status")
    if not ai_status:
        return jsonify({'status': 'error', 'models': []}), 503

    models_out = []
    for key, entry in ai_status.models.items():
        models_out.append({
            'key': entry.key,
            'label': entry.label,
            'architecture': entry.architecture,
            'diffusion_type': entry.diffusion_type,
            'available': entry.available,
            'error': entry.error,
            'checkpoint_path': str(entry.checkpoint_path) if entry.checkpoint_path else None,
            'is_default': key == ai_status.default_model,
        })

    return jsonify({
        'status': 'success',
        'default_model': ai_status.default_model,
        'models': models_out,
    })


@bp.route('/api/ai/generate', methods=['POST'])
def ai_generate_structures():
    """AI-powered fullerene structure generation endpoint."""
    ai_status = current_app.extensions.get("ai_status")
    if not ai_status or not ai_status.available:
        detail = None
        if ai_status and ai_status.error:
            detail = ai_status.error
        return jsonify({
            'status': 'error',
            'error': detail or 'AI module not available. Please ensure fullerene_e3gen is properly configured.'
        }), 503

    try:
        data = request.get_json() or {}
        model_key = data.get('model') or ai_status.default_model
        api = ai_status.get_api(model_key)
        if api is None:
            return jsonify({
                'status': 'error',
                'error': f'Model "{model_key}" is not available.'
            }), 503
        model_entry = ai_status.get_model_entry(model_key)

        num_atoms = int(data.get('num_atoms', 60))
        num_samples = int(data.get('num_samples', 5))
        output_format = str(data.get('output_format', 'xyz')).lower()

        if num_atoms < 20 or num_atoms > 720:
            return jsonify({'status': 'error', 'error': 'num_atoms must be between 20 and 720'}), 400
        if num_samples < 1 or num_samples > 50:
            return jsonify({'status': 'error', 'error': 'num_samples must be between 1 and 50'}), 400
        if output_format not in {'xyz', 'json'}:
            return jsonify({'status': 'error', 'error': 'output_format must be xyz or json'}), 400

        output_dir = Path(current_app.config['UPLOAD_FOLDER'])
        run_id = secrets.token_hex(8)

        start_time = time.time()
        results = api.generate(num_carbon=num_atoms, num_samples=num_samples)
        generation_time = time.time() - start_time

        structures = []
        for idx, result in enumerate(results):
            filename = f"ai_{run_id}_C{num_atoms}_sample_{idx:03d}.{output_format}"
            filepath = output_dir / filename
            api.save_structure(
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
                        logger.error("Error reading %s: %s", filepath, exc)
                # Send positions for 3D viewer (list of [x,y,z])
                pos_arr = result['positions']
                if hasattr(pos_arr, 'tolist'):
                    # Replace NaN/Inf with 0 before serialization
                    pos_arr = np.nan_to_num(pos_arr, nan=0.0, posinf=0.0, neginf=0.0)
                    positions_list = pos_arr.tolist()
                else:
                    positions_list = pos_arr
                edges_list = result.get('edges', [])
                if hasattr(edges_list, 'tolist'):
                    edges_list = edges_list.tolist()
                elif edges_list and isinstance(edges_list[0], tuple):
                    edges_list = [list(e) for e in edges_list]
                structures.append({
                    'filename': filename,
                    'content': content,
                    'download_url': f"/api/download/{filename}",
                    'positions': positions_list,
                    'edges': edges_list,
                    'num_atoms': len(positions_list),
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
            'model': model_key,
            'model_label': model_entry.label if model_entry else model_key,
            'message': f"Generated {num_generated} {output_format.upper()} structures in {generation_time:.1f}s (model: {model_entry.label if model_entry else model_key})"
        })

    except Exception as exc:
        return jsonify({
            'status': 'error',
            'error': str(exc)
        }), 500


@bp.route('/api/ai/generate-interface', methods=['POST'])
def ai_generate_interface():
    """Generate an interface using selected model + optional local GNN refinement."""
    ai_status = current_app.extensions.get("ai_status")
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({'status': 'error', 'error': 'Core modules not available'}), 500
    if not ai_status or not ai_status.available:
        detail = None
        if ai_status and ai_status.error:
            detail = ai_status.error
        return jsonify({
            'status': 'error',
            'error': detail or 'AI module not available. Please ensure fullerene_e3gen is properly configured.'
        }), 503

    data = request.get_json() or {}
    model_key = data.get('model') or ai_status.default_model
    selected_api = ai_status.get_api(model_key)
    if selected_api is None:
        return jsonify({'status': 'error', 'error': f'Model "{model_key}" is not available.'}), 503
    model_entry = ai_status.get_model_entry(model_key)
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
    if num_atoms < 20 or num_atoms > 720 or num_atoms % 2 != 0:
        return jsonify({'status': 'error', 'error': 'num_atoms must be even and between 20 and 720'}), 400

    num_samples = int(data.get('num_samples', 1))
    if num_samples < 1 or num_samples > 20:
        return jsonify({'status': 'error', 'error': 'num_samples must be between 1 and 20'}), 400

    miller = _parse_int_tuple(data.get('miller'), default=(0, 0, 1))
    xy_frac = _parse_float_tuple(data.get('xy_frac'), default=(0.5, 0.5))
    supercell_xy = _parse_int_tuple(data.get('supercell_xy'), length=2, default=None)

    termination = data.get('termination')
    slab_thickness = float(data.get('slab_thickness', 18.0))
    vacuum = float(data.get('vacuum', 20.0))
    separation = float(data.get('separation', 3.2))
    buffer = float(data.get('buffer', 10.0))
    layer_tol = float(data.get('layer_tol', 1.5))
    ddim = bool(data.get('ddim', True))

    refine_scope = str(data.get('refine_scope', 'adsorbate')).lower()
    if refine_scope not in {'adsorbate', 'interface', 'all'}:
        refine_scope = 'adsorbate'
    local_refine = bool(data.get('local_refine', True))

    generator = InterfaceAIGenerator(
        fullerene_api=selected_api,
        local_refiner=ai_status.local_refiner if local_refine else None,
    )

    start_time = time.time()
    try:
        results = generator.generate_interface(
            base_structure=base_structure,
            num_atoms=num_atoms,
            num_samples=num_samples,
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

    # Write each sample to its own file
    structures_out = []
    for idx, result in enumerate(results):
        suffix = f"_{idx:03d}" if len(results) > 1 else ""
        output_filename = f"ai_interface_{run_id}{suffix}.vasp"
        output_path = upload_folder / output_filename
        try:
            core_status.io.write_poscar(result.combined, output_path)
        except Exception as exc:
            logger.error("Failed to write sample %d: %s", idx, exc)
            continue

        structures_out.append({
            'sample_index': idx,
            'output_file': str(output_path),
            'download_url': f"/api/download/{output_filename}",
            'n_atoms': len(result.combined),
            'metadata': result.metadata,
        })

    if not structures_out:
        return jsonify({'status': 'error', 'error': 'All samples failed to write'}), 500

    # For backward compatibility, expose first result's fields at top level
    first = structures_out[0]
    return jsonify({
        'status': 'success',
        'output_file': first['output_file'],
        'download_url': first['download_url'],
        'generation_time': round(generation_time, 2),
        'metadata': first['metadata'],
        'n_atoms': first['n_atoms'],
        'num_samples': len(structures_out),
        'samples': structures_out,
        'base_filename': safe_name,
        'local_refiner_error': getattr(ai_status, 'local_error', None),
        'model': model_key,
        'model_label': model_entry.label if model_entry else model_key,
    })


@bp.route('/api/ai/evaluate', methods=['POST'])
def ai_evaluate_structures():
    """AI evaluation of generated structures."""
    ai_status = current_app.extensions.get("ai_status")
    if not ai_status or not ai_status.available:
        return jsonify({
            'status': 'error',
            'error': 'AI module not available'
        }), 503

    try:
        data = request.get_json() or {}
        model_key = data.get('model') or ai_status.default_model
        api = ai_status.get_api(model_key)
        if api is None:
            return jsonify({'status': 'error', 'error': f'Model "{model_key}" not available'}), 503

        positions = data.get('positions')
        edges = data.get('edges')
        compute_advanced = bool(data.get('compute_advanced_metrics', True))

        if positions is not None and edges is not None:
            metrics = api.evaluate(
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
            metrics = api.evaluate(
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
    """Get AI model information (multi-model aware)."""
    ai_status = current_app.extensions.get("ai_status")
    if not ai_status or not ai_status.available:
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
        primary_api = ai_status.get_api()
        info = primary_api.get_model_info() if primary_api else {}
        info['available'] = True

        # Multi-model summary
        info['available_models'] = []
        for key, entry in ai_status.models.items():
            model_info = {
                'key': entry.key,
                'label': entry.label,
                'architecture': entry.architecture,
                'diffusion_type': entry.diffusion_type,
                'available': entry.available,
                'error': entry.error,
                'is_default': key == ai_status.default_model,
            }
            if entry.available and entry.api:
                try:
                    mi = entry.api.get_model_info()
                    model_info['num_parameters'] = mi.get('num_parameters')
                    model_info['epoch'] = (mi.get('checkpoint_info') or {}).get('epoch')
                except Exception:
                    pass
            info['available_models'].append(model_info)

        info['default_model'] = ai_status.default_model

        # Include local GNN refiner status
        info['local_refiner_available'] = ai_status.local_refiner is not None
        info['local_refiner_trained'] = bool(
            ai_status.local_refiner and getattr(ai_status.local_refiner, 'trained', False)
        )
        info['local_refiner_error'] = getattr(ai_status, 'local_error', None)
        # SE(3) equivariant GNN info
        info['local_refiner_equivariant'] = True
        info['pbc_aware'] = True
        info['batch_generation'] = True
        # Async task backend info
        try:
            from interfaceml.web.tasks import CELERY_AVAILABLE, get_task_backend
            backend = get_task_backend()
            info['async_backend'] = type(backend).__name__
            info['celery_available'] = CELERY_AVAILABLE
        except Exception:
            info['async_backend'] = 'none'
            info['celery_available'] = False
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


# -------------------------------------------------------------------------
# Async interface generation (Celery / thread-pool fallback)
# -------------------------------------------------------------------------

@bp.route('/api/ai/generate-interface-async', methods=['POST'])
def ai_generate_interface_async():
    """Submit an interface generation task asynchronously.

    Returns a ``task_id`` that can be polled via ``/api/ai/task-status/<task_id>``.
    """
    ai_status = current_app.extensions.get("ai_status")
    core_status = current_app.extensions.get("core")
    if not core_status or not core_status.available:
        return jsonify({'status': 'error', 'error': 'Core modules not available'}), 500
    if not ai_status or not ai_status.available:
        detail = ai_status.error if ai_status else None
        return jsonify({
            'status': 'error',
            'error': detail or 'AI module not available.',
        }), 503

    data = request.get_json() or {}
    model_key = data.get('model') or ai_status.default_model
    selected_api = ai_status.get_api(model_key)
    if selected_api is None:
        return jsonify({'status': 'error', 'error': f'Model "{model_key}" is not available.'}), 503

    data = request.get_json() or {}
    base_filename = data.get('base_filename') or data.get('base_file')
    if not base_filename:
        return jsonify({'status': 'error', 'error': 'base_filename is required'}), 400

    upload_folder = str(Path(current_app.config['UPLOAD_FOLDER']))

    from interfaceml.web.tasks import get_task_backend

    backend = get_task_backend()
    task_id = backend.submit_interface_generation(
        payload=data,
        ai_status=ai_status,
        core_status=core_status,
        upload_folder=upload_folder,
    )

    return jsonify({
        'status': 'accepted',
        'task_id': task_id,
        'poll_url': f'/api/ai/task-status/{task_id}',
    }), 202


@bp.route('/api/ai/task-status/<task_id>', methods=['GET'])
def ai_task_status(task_id: str):
    """Poll the status of an async generation task.

    Possible states: PENDING, STARTED, SUCCESS, FAILURE, NOT_FOUND.
    When state == SUCCESS, the full generation result is in ``result``.
    """
    from interfaceml.web.tasks import get_task_backend

    backend = get_task_backend()
    info = backend.get_result(task_id)

    status_code = 200
    if info["state"] == "NOT_FOUND":
        status_code = 404
    elif info["state"] in ("PENDING", "STARTED"):
        status_code = 202

    return jsonify(info), status_code

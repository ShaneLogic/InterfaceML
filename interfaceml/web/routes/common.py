"""
Common (non-domain specific) routes for InterfaceML.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, current_app, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from interfaceml import __version__
from interfaceml.web.utils import allowed_file


bp = Blueprint("common", __name__)
REPO_ROOT = Path(__file__).resolve().parents[3]


@bp.route('/')
def index():
    """Render the documentation homepage."""
    return render_template('docs.html')


@bp.route('/app')
def app_home():
    """Render the main web interface."""
    return render_template('index.html')


@bp.route('/docs')
def docs():
    """Render the documentation page."""
    return render_template('docs.html')


@bp.route('/docs/file/<path:doc_path>')
def docs_file(doc_path):
    """Serve markdown documentation files from the repo."""
    target = (REPO_ROOT / doc_path).resolve()
    try:
        target.relative_to(REPO_ROOT)
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 400
    if not target.exists() or not target.is_file():
        return jsonify({'error': 'File not found'}), 404
    if target.suffix.lower() != '.md':
        return jsonify({'error': 'Unsupported file type'}), 400
    return send_file(str(target), mimetype='text/markdown')


@bp.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    core_status = current_app.extensions.get("core")
    splitting_file = None
    splitting_hash = None
    if core_status and core_status.splitting_available and core_status.splitting:
        try:
            splitting_file = getattr(core_status.splitting, '__file__', None)
            if splitting_file and Path(splitting_file).exists():
                data = Path(splitting_file).read_bytes()
                splitting_hash = hashlib.sha256(data).hexdigest()[:12]
        except Exception:
            splitting_file = splitting_file or None
            splitting_hash = splitting_hash or None

    return jsonify({
        'status': 'ok',
        'core_available': bool(core_status and core_status.available),
        'splitting_available': bool(core_status and core_status.splitting_available),
        'splitting_file': splitting_file,
        'splitting_hash': splitting_hash,
        'version': __version__,
    })


@bp.route('/api/upload', methods=['POST'])
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

    if not file.filename or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        allowed_exts = ', '.join(current_app.config['ALLOWED_EXTENSIONS'])
        return jsonify({
            'error': f'Invalid file type. Allowed: {allowed_exts}'
        }), 400

    try:
        # Save file securely with sanitized filename
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400

        upload_folder = Path(current_app.config['UPLOAD_FOLDER'])
        filepath = upload_folder / filename
        file.save(str(filepath))

        # Compute hash to help users confirm they're operating on the expected file.
        file_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()[:12]

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

        core_status = current_app.extensions.get("core")
        if core_status and core_status.available and not is_dos_file:
            try:
                structure = core_status.io.load_structure(filepath)
                info['n_atoms'] = len(structure)
                info['composition'] = structure.composition.formula
                info['lattice'] = {
                    'a': round(structure.lattice.a, 3),
                    'b': round(structure.lattice.b, 3),
                    'c': round(structure.lattice.c, 3),
                    'alpha': round(structure.lattice.alpha, 2),
                    'beta': round(structure.lattice.beta, 2),
                    'gamma': round(structure.lattice.gamma, 2),
                }
            except Exception as exc:
                info['parse_warning'] = f"Could not parse structure: {exc}"

        return jsonify({
            'status': 'success',
            'file_info': info
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@bp.route('/api/download/<filename>')
def download_file(filename):
    """Download a generated file."""
    upload_folder = Path(current_app.config['UPLOAD_FOLDER'])
    filepath = upload_folder / secure_filename(filename)
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    # Prevent browsers from caching downloads with the same URL/filename across runs.
    resp = send_file(str(filepath), as_attachment=True)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

"""
Flask web application for InterfaceML.

This module provides a browser-based interface for heterojunction modeling,
allowing users to upload structure files and interactively configure
modeling parameters.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask
from flask_cors import CORS

from interfaceml.web.ai import load_fullerene_api
from interfaceml.web.core import load_core_modules
from interfaceml.web.routes.ai_routes import bp as ai_bp
from interfaceml.web.routes.common import bp as common_bp
from interfaceml.web.routes.dos_routes import bp as dos_bp
from interfaceml.web.routes.interface import bp as interface_bp


def create_app() -> Flask:
    """Application factory for InterfaceML web server."""
    app = Flask(__name__)
    CORS(app)

    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix='interfaceml_')
    app.config['ALLOWED_EXTENSIONS'] = {'cif', 'vasp', 'poscar', 'xyz', 'dos', 'pdos'}

    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

    core_status = load_core_modules()
    app.extensions['core'] = core_status

    ai_status = load_fullerene_api()
    app.extensions['ai_status'] = ai_status
    if ai_status.available:
        print(f"✓ Fullerene AI module loaded: {ai_status.checkpoint_path}")
    else:
        reason = ai_status.error or "Unknown reason"
        print(f"⚠ Fullerene AI unavailable: {reason}")

    app.register_blueprint(common_bp)
    app.register_blueprint(interface_bp)
    app.register_blueprint(dos_bp)
    app.register_blueprint(ai_bp)

    return app


app = create_app()


def main() -> None:
    """Main entry point for the web application."""
    import argparse

    parser = argparse.ArgumentParser(description="InterfaceML Web Server")
    parser.add_argument('--port', type=int, default=5000, help='Port to run on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()

    core_status = app.extensions.get("core")
    ai_status = app.extensions.get("ai_status")

    print("\n" + "=" * 60)
    print("  InterfaceML - Professional Heterojunction Modeling Platform")
    print("=" * 60)
    print(f"\n📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"⚙️  Core modules: {'✓ Available' if core_status and core_status.available else '✗ Not available'}")
    print(f"🤖 AI Generation: {'✓ Available' if ai_status and ai_status.available else '✗ Not available'}")
    print(f"\n🌐 Server: http://localhost:{args.port}")
    print("\n" + "=" * 60 + "\n")

    app.run(debug=True, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

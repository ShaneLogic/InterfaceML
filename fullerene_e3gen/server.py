"""
REST API Server for Fullerene Generation

Provides HTTP endpoints for frontend integration.
Uses Flask for lightweight deployment.

Endpoints:
    POST /api/generate - Generate fullerene structures
    POST /api/evaluate - Evaluate a structure
    GET  /api/model/info - Get model information
    GET  /api/health - Health check

Author: InterfaceML Project
Date: 2026-01-30
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np

from api import FullereneAPI


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global API instance
api_instance = None
CONFIG = {
    'checkpoint_path': None,
    'max_samples_per_request': 100,
    'allowed_carbon_range': (20, 720)
}


def init_api(checkpoint_path: str):
    """Initialize the API with model checkpoint."""
    global api_instance
    try:
        api_instance = FullereneAPI(checkpoint_path)
        CONFIG['checkpoint_path'] = checkpoint_path
        logger.info(f"API initialized with checkpoint: {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}")
        return False


def error_response(message: str, status_code: int = 400) -> tuple:
    """Create error response."""
    return jsonify({'error': message, 'success': False}), status_code


def success_response(data: Dict[str, Any]) -> tuple:
    """Create success response."""
    data['success'] = True
    return jsonify(data), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON with server status
    """
    status = {
        'status': 'healthy',
        'model_loaded': api_instance is not None,
        'checkpoint': CONFIG['checkpoint_path']
    }
    return success_response(status)


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """
    Get model information.
    
    Returns:
        JSON with model configuration and status
    """
    if api_instance is None:
        return error_response("Model not loaded", 503)
    
    try:
        info = api_instance.get_model_info()
        return success_response(info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return error_response(str(e), 500)


@app.route('/api/generate', methods=['POST'])
def generate_structures():
    """
    Generate fullerene structures.
    
    Request JSON:
        {
            "num_carbon": int,
            "num_samples": int (optional, default=1),
            "ddim": bool (optional, default=false),
            "return_trajectory": bool (optional, default=false),
            "output_format": str (optional, "json"|"xyz", default="json")
        }
    
    Returns:
        JSON with generated structures or file download
    """
    if api_instance is None:
        return error_response("Model not loaded", 503)
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return error_response("No JSON data provided")
        
        num_carbon = data.get('num_carbon')
        if num_carbon is None:
            return error_response("Missing required field: num_carbon")
        
        num_samples = data.get('num_samples', 1)
        ddim = data.get('ddim', False)
        return_trajectory = data.get('return_trajectory', False)
        output_format = data.get('output_format', 'json')
        
        # Validate parameters
        if num_carbon < CONFIG['allowed_carbon_range'][0] or \
           num_carbon > CONFIG['allowed_carbon_range'][1]:
            return error_response(
                f"num_carbon must be in range {CONFIG['allowed_carbon_range']}"
            )
        
        if num_carbon % 2 != 0:
            return error_response("num_carbon must be even")
        
        if num_samples < 1 or num_samples > CONFIG['max_samples_per_request']:
            return error_response(
                f"num_samples must be in [1, {CONFIG['max_samples_per_request']}]"
            )
        
        # Generate structures
        logger.info(f"Generating {num_samples} C{num_carbon} structures")
        results = api_instance.generate(
            num_carbon=num_carbon,
            num_samples=num_samples,
            ddim=ddim,
            return_trajectory=return_trajectory
        )
        
        # Format response
        if output_format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for result in results:
                json_result = {
                    'positions': result['positions'].tolist(),
                    'edges': result['edges'],
                    'metadata': result['metadata']
                }
                if 'trajectory' in result:
                    json_result['trajectory'] = [t.tolist() for t in result['trajectory']]
                json_results.append(json_result)
            
            return success_response({
                'structures': json_results,
                'count': len(json_results)
            })
        
        else:
            return error_response(f"Unsupported output format: {output_format}")
        
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Generation error: {str(e)}\n{traceback.format_exc()}")
        return error_response(f"Generation failed: {str(e)}", 500)


@app.route('/api/evaluate', methods=['POST'])
def evaluate_structure():
    """
    Evaluate a structure.
    
    Request JSON:
        {
            "positions": [[x, y, z], ...],
            "edges": [[src, dst], ...],
            "compute_advanced_metrics": bool (optional, default=true)
        }
    
    Returns:
        JSON with evaluation metrics
    """
    if api_instance is None:
        return error_response("Model not loaded", 503)
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return error_response("No JSON data provided")
        
        positions = data.get('positions')
        edges = data.get('edges')
        
        if positions is None or edges is None:
            return error_response("Missing required fields: positions, edges")
        
        # Convert to numpy
        positions = np.array(positions, dtype=np.float32)
        edges = [tuple(e) for e in edges]
        
        compute_advanced = data.get('compute_advanced_metrics', True)
        
        # Evaluate
        logger.info(f"Evaluating structure with {len(positions)} atoms")
        metrics = api_instance.evaluate(
            positions=positions,
            edges=edges,
            compute_advanced_metrics=compute_advanced
        )
        
        return success_response({'metrics': metrics})
        
    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}\n{traceback.format_exc()}")
        return error_response(f"Evaluation failed: {str(e)}", 500)


@app.route('/api/save', methods=['POST'])
def save_structure():
    """
    Save structure to file.
    
    Request JSON:
        {
            "positions": [[x, y, z], ...],
            "edges": [[src, dst], ...],
            "filename": str,
            "format": str (optional, "xyz"|"json", default="xyz")
        }
    
    Returns:
        File download
    """
    if api_instance is None:
        return error_response("Model not loaded", 503)
    
    try:
        data = request.get_json()
        if not data:
            return error_response("No JSON data provided")
        
        positions = np.array(data.get('positions'), dtype=np.float32)
        edges = [tuple(e) for e in data.get('edges')]
        filename = data.get('filename', 'structure.xyz')
        format_type = data.get('format', 'xyz')
        
        # Sanitize filename to prevent path traversal
        filename = os.path.basename(filename)
        if not filename:
            return error_response("Invalid filename")
        
        # Save to temporary file
        temp_path = Path(f"/tmp/{filename}")
        api_instance.save_structure(positions, edges, str(temp_path), format_type)
        
        # Send file
        return send_file(
            str(temp_path),
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Save error: {str(e)}")
        return error_response(f"Save failed: {str(e)}", 500)


@app.route('/api/batch_generate', methods=['POST'])
def batch_generate():
    """
    Generate multiple carbon sizes in one request.
    
    Request JSON:
        {
            "carbon_sizes": [50, 52, 54, 60],
            "samples_per_size": int (default=1),
            "ddim": bool (optional, default=false)
        }
    
    Returns:
        JSON with all generated structures grouped by carbon size
    """
    if api_instance is None:
        return error_response("Model not loaded", 503)
    
    try:
        data = request.get_json()
        if not data:
            return error_response("No JSON data provided")
        
        carbon_sizes = data.get('carbon_sizes', [])
        samples_per_size = data.get('samples_per_size', 1)
        ddim = data.get('ddim', False)
        
        if not carbon_sizes:
            return error_response("carbon_sizes cannot be empty")
        
        if len(carbon_sizes) * samples_per_size > CONFIG['max_samples_per_request']:
            return error_response(
                f"Total samples exceeds limit: {CONFIG['max_samples_per_request']}"
            )
        
        # Generate for each size
        all_results = {}
        for num_carbon in carbon_sizes:
            logger.info(f"Batch generating {samples_per_size} C{num_carbon} structures")
            results = api_instance.generate(
                num_carbon=num_carbon,
                num_samples=samples_per_size,
                ddim=ddim
            )
            
            all_results[f"C{num_carbon}"] = [
                {
                    'positions': r['positions'].tolist(),
                    'edges': r['edges'],
                    'metadata': r['metadata']
                }
                for r in results
            ]
        
        return success_response({
            'structures': all_results,
            'total_count': sum(len(v) for v in all_results.values())
        })
        
    except Exception as e:
        logger.error(f"Batch generation error: {str(e)}")
        return error_response(f"Batch generation failed: {str(e)}", 500)


def create_app(checkpoint_path: str, host: str = '0.0.0.0', port: int = 5000):
    """
    Create and configure the Flask app.
    
    Args:
        checkpoint_path: Path to model checkpoint
        host: Host address
        port: Port number
        
    Returns:
        Configured Flask app
    """
    if not init_api(checkpoint_path):
        raise RuntimeError("Failed to initialize API")
    
    logger.info(f"Server starting on {host}:{port}")
    return app


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fullerene REST API Server')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port number (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    args = parser.parse_args()
    
    # Initialize and run
    app_instance = create_app(args.checkpoint, args.host, args.port)
    app_instance.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

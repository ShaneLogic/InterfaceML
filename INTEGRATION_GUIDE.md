# InterfaceML and AI Structure Generation Integration Guide

## Overview

This guide describes the integration of the AI structure generation module with
InterfaceML. The integration adds fullerene generation and evaluation to the web
interface and exposes new API endpoints for batch workflows.

## Features

- AI-based fullerene generation for C60, C70, C80, C84, C100, and custom sizes
- Batch generation with metadata export (XYZ and JSON)
- Web UI integration with status checks and validation
- Unified REST API endpoints for traditional and AI features

## API Endpoints

Traditional features:
- `/api/upload` - File uploads
- `/api/build-interface` - Interface builder
- `/api/fix-layers` - Layer management
- `/api/compute-density` - Density analysis
- `/api/plot-dos` - DOS visualization

AI features:
- `/api/ai/generate` - Structure generation
- `/api/ai/evaluate` - Quality metrics
- `/api/ai/info` - Model information

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- InterfaceML dependencies
- PyTorch 2.0 or higher
- PyTorch Geometric 2.0 or higher

### Quick Start

1. Ensure the AI module is present:

```bash
cd /path/to/InterfaceML
ls fullerene_e3gen/checkpoints/best_model.pt
```

2. Optional environment overrides:

```bash
# Override the AI module location
export INTERFACEML_FULLERENE_PATH=/path/to/fullerene_e3gen

# Override the checkpoint path
export INTERFACEML_FULLERENE_CHECKPOINT=/path/to/best_model.pt
```

3. Start the integrated server:

```bash
cd interfaceml/web
python app.py --port 5000
```

Expected output includes the upload folder, module availability, and server URL.

4. Open a browser at `http://localhost:5000` and select the AI Structure
   Generation tab.

## Usage Examples

### 1. Generate C60 Fullerenes (Web)

1. Select "C60 (Buckminsterfullerene)" from the dropdown
2. Set the number of samples
3. Click "Generate Structures"
4. Download results

### 2. Generate C60 Fullerenes (API)

```bash
curl -X POST http://localhost:5000/api/ai/generate \
  -H "Content-Type: application/json" \
  -d '{
    "num_atoms": 60,
    "num_samples": 5,
    "output_format": "xyz"
  }'
```

### 3. Evaluate Generated Structures (API)

```bash
curl -X POST http://localhost:5000/api/ai/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "generated_dir": "ai_generated_abc123"
  }'
```

### 4. Batch Generation (Web)

1. Select multiple carbon counts
2. Set samples per size
3. Click "Batch Generate"

### 5. Custom Fullerenes (Web)

1. Select "Custom" from the dropdown
2. Enter an even atom count between 20 and 240
3. Generate

## Architecture

### Backend Integration

Key files:
- `interfaceml/web/app.py` - app factory and blueprint registration
- `interfaceml/web/routes/` - Flask blueprints by domain
- `interfaceml/web/ai.py` - AI module discovery and loading
- `interfaceml/web/core.py` - core module availability
- `interfaceml/web/dos.py` - DOS and PDOS helpers
- `interfaceml/web/utils.py` - shared utilities

```python
from interfaceml.web.ai import load_fullerene_api
ai_status = load_fullerene_api()

app.register_blueprint(common_bp)
app.register_blueprint(interface_bp)
app.register_blueprint(dos_bp)
app.register_blueprint(ai_bp)
```

### Frontend Integration

Key files:
- `templates/index.html` - AI tab UI
- `static/css/style.css` - AI styles
- `static/js/app.js` - AI interactions

## Configuration

The AI module loads from the following layout:

```
fullerene_e3gen/
|-- checkpoints/
|   `-- best_model.pt          # Main checkpoint (required)
|-- api.py                      # Python API
`-- config.yaml                 # Model config
```

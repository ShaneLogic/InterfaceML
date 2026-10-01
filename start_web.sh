#!/bin/bash
#
# Quick script to start InterfaceML web server
# This script handles common issues automatically
#

echo "🚀 Starting InterfaceML Web Server"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "interfaceml/web/app.py" ]; then
    echo "❌ Error: Please run this script from the InterfaceML root directory"
    exit 1
fi

# Add current directory to PYTHONPATH so interfaceml can be imported
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "✓ PYTHONPATH set to: $PWD"

# Choose a Python interpreter that has the required web dependencies.
# Prefer an explicit override via INTERFACEML_PYTHON, then the current `python`,
# then fall back to the known conda env used in this repo.
PYTHON_BIN="${INTERFACEML_PYTHON:-python}"

check_web_deps() {
    "$1" -c "import flask; import flask_cors" >/dev/null 2>&1
}

if ! check_web_deps "$PYTHON_BIN"; then
    # Try EGNNs env first (has torch + flask), then interface env (flask only)
    EGNNS_FALLBACK="/Users/shane/Applications/anaconda3/envs/EGNNs/bin/python"
    CONDA_FALLBACK="/Users/shane/Applications/anaconda3/envs/interface/bin/python"
    if [ -x "$EGNNS_FALLBACK" ] && check_web_deps "$EGNNS_FALLBACK"; then
        PYTHON_BIN="$EGNNS_FALLBACK"
        echo "✓ Using Python: $PYTHON_BIN (EGNNs conda env)"
    elif [ -x "$CONDA_FALLBACK" ] && check_web_deps "$CONDA_FALLBACK"; then
        PYTHON_BIN="$CONDA_FALLBACK"
        echo "✓ Using Python: $PYTHON_BIN (interface conda env fallback)"
    else
        echo "❌ Error: cannot start web server because required packages are missing in: $PYTHON_BIN"
        echo "   Missing: flask and/or flask_cors"
        echo ""
        echo "💡 Fix options:"
        echo "   1) Activate the correct environment (recommended)"
        echo "   2) Or install deps: pip install -r requirements.txt"
        echo "   3) Or set INTERFACEML_PYTHON to a Python that has deps installed"
        exit 1
    fi
fi

# Check optional AI dependencies and assets (warn only).
AI_CHECK_OUTPUT="$("$PYTHON_BIN" - <<'PY'
import importlib
import os
import sys
from pathlib import Path

missing = []
for mod in ("torch", "torch_geometric"):
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {exc}")

base = Path(os.getenv("INTERFACEML_FULLERENE_PATH") or (Path.cwd() / "fullerene_e3gen"))
if not base.exists():
    missing.append(f"fullerene_e3gen path not found: {base}")

ckpt = Path(os.getenv("INTERFACEML_FULLERENE_CHECKPOINT") or (base / "checkpoints" / "best_model.pt"))
if not ckpt.exists():
    missing.append(f"checkpoint not found: {ckpt}")

if missing:
    print("\n".join(missing))
    sys.exit(1)
PY
)"

if [ $? -ne 0 ]; then
    echo "⚠️  AI module will be unavailable with the current Python:"
    echo "$PYTHON_BIN"
    echo ""
    echo "Missing items:"
    echo "$AI_CHECK_OUTPUT"
    echo ""
    echo "💡 To enable AI generation:"
    echo "   1) Use a Python version supported by PyTorch (e.g., 3.10 or 3.11)"
    echo "   2) Install AI deps: pip install -e \".[ai]\""
    echo "      (or pip install -r fullerene_e3gen/requirements.txt)"
    echo "   3) Ensure the checkpoint exists (or set INTERFACEML_FULLERENE_CHECKPOINT)"
    echo ""
fi

# Try different ports in case 5000 is occupied
PORTS=(5000 8000 8080 5001 3000)
PORT_FOUND=false

for PORT in "${PORTS[@]}"; do
    # Check if port is available
    if ! lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        PORT_FOUND=true
        echo "✓ Using port: $PORT"
        echo ""
        echo "📱 Open your browser to: http://localhost:$PORT"
        echo ""
        echo "Press Ctrl+C to stop the server"
        echo "=================================="
        echo ""
        
        # Start the server
        "$PYTHON_BIN" interfaceml/web/app.py --port $PORT
        exit $?
    fi
done

if [ "$PORT_FOUND" = false ]; then
    echo "❌ Error: All common ports (5000, 8000, 8080, 5001, 3000) are in use"
    echo ""
    echo "💡 Try manually:"
    echo "   $PYTHON_BIN interfaceml/web/app.py --port 9000"
    exit 1
fi

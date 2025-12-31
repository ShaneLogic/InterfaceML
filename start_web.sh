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
        python interfaceml/web/app.py --port $PORT
        exit $?
    fi
done

if [ "$PORT_FOUND" = false ]; then
    echo "❌ Error: All common ports (5000, 8000, 8080, 5001, 3000) are in use"
    echo ""
    echo "💡 Try manually:"
    echo "   python interfaceml/web/app.py --port 9000"
    exit 1
fi

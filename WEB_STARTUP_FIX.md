# Web Server Startup Troubleshooting Guide

## 🔴 Issues Identified

Based on the terminal output, two main issues were detected:

1. **Core modules unavailable** (`Core modules available: False`)
   - InterfaceML package not properly installed in the new environment
   
2. **Port already in use** (`Port 5000 is in use`)
   - Port 5000 is occupied by another program (likely AirPlay Receiver)

---

## ✅ Quick Solutions

### Solution 1: Use Startup Script (Recommended)

```bash
# From InterfaceML root directory
./start_web.sh
```

This script will automatically:
- Set PYTHONPATH for module imports
- Try multiple ports (5000, 8000, 8080, 5001, 3000)
- Display the available URL

### Solution 2: Manual Startup (Recommended)

```bash
# 1. Ensure you're in the project root directory
cd /Users/shane/Library/CloudStorage/OneDrive-HKUST\(Guangzhou\)/EGNNs-DEV/InterfaceML

# 2. Set PYTHONPATH (Critical!)
export PYTHONPATH=$PWD:$PYTHONPATH

# 3. Start with alternative port (avoid 5000)
python interfaceml/web/app.py --port 8000
```

Then open in browser: **http://localhost:8000**

### Solution 3: Close Program Using Port 5000

**For macOS AirPlay Receiver:**
```bash
# Check which program is using the port
lsof -i :5000

# Disable AirPlay Receiver in macOS:
# System Settings → General → AirDrop & Handoff → 
# Uncheck "AirPlay Receiver"
```

Then use port 5000:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python interfaceml/web/app.py
```

---

## 🔧 Core Modules Issue Explained

### Root Cause

You created a new `interface` environment and installed dependencies:
```bash
conda create -n interface python=3.8
conda activate interface
pip install -r requirements.txt
```

However, **the InterfaceML package itself was not installed**, so Python cannot find the `interfaceml.core` module at runtime.

### Resolution Methods

Three ways to make Python find the `interfaceml` module:

#### Method 1: Set PYTHONPATH (Temporary, good for testing)

```bash
# Run before each startup
export PYTHONPATH=/Users/shane/Library/CloudStorage/OneDrive-HKUST\(Guangzhou\)/EGNNs-DEV/InterfaceML:$PYTHONPATH
python interfaceml/web/app.py --port 8000
```

#### Method 2: Install as Development Package (Permanent, recommended)

```bash
# From project root directory
cd /Users/shane/Library/CloudStorage/OneDrive-HKUST\(Guangzhou\)/EGNNs-DEV/InterfaceML

# Install in editable mode
pip install -e .
```

After installation, `Core modules available` will show `True`.

#### Method 3: Change Startup Method (Simplest)

Instead of running `python interfaceml/web/app.py`, use:

```bash
# Ensure you're in project root
cd /Users/shane/Library/CloudStorage/OneDrive-HKUST\(Guangzhou\)/EGNNs-DEV/InterfaceML

# Run as module (automatically handles paths)
python -m interfaceml.web.app --port 8000
```

---

## 📋 Complete Startup Steps (Copy & Paste)

```bash
# 1. Activate environment
conda activate interface

# 2. Navigate to project directory
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/EGNNs-DEV/InterfaceML"

# 3. Option A: Use script (simplest)
./start_web.sh

# OR Option B: Manual startup
export PYTHONPATH=$PWD:$PYTHONPATH
python interfaceml/web/app.py --port 8000

# OR Option C: Run as module (recommended)
python -m interfaceml.web.app --port 8000
```

---

## ✅ Success Indicators

You should see the following output when successful:

```
Starting InterfaceML Web Server...
Upload folder: /var/folders/.../interfaceml_xxxxx
Core modules available: True  ← This should be True

Open your browser and navigate to: http://localhost:8000
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server...
 * Running on http://0.0.0.0:8000
```

**Key indicator: `Core modules available: True`**

---

## 🧪 Verification

After successful startup, test in browser:

1. **Open homepage**: http://localhost:8000
   - Should see beautiful gradient background
   - Should see 4 functional tabs

2. **Test API**:
```bash
curl http://localhost:8000/api/health
```

Should return:
```json
{
  "status": "ok",
  "core_available": true,
  "version": "1.0.0"
}
```

---

## 💡 Additional Tips

### Permanent Solution

For future convenience:

1. **Install package**:
```bash
pip install -e .
```

2. **Create alias** (add to `~/.zshrc`):
```bash
alias interfaceml-web='cd /path/to/InterfaceML && python -m interfaceml.web.app --port 8000'
```

Then simply run:
```bash
interfaceml-web
```

### Find Process Using Port

```bash
# Check which program is using port 5000
lsof -i :5000

# Kill the process (use with caution)
kill -9 <PID>
```

### Run in Background

```bash
# Run in background
nohup python -m interfaceml.web.app --port 8000 > webserver.log 2>&1 &

# View logs
tail -f webserver.log

# Stop service
pkill -f "interfaceml.web.app"
```

---

## 🆘 Still Not Working?

If you still have issues, run diagnostics:

```bash
# 1. Check Python environment
which python
python --version

# 2. Check import capability
python -c "import sys; print('\\n'.join(sys.path))"
python -c "from interfaceml.core import io; print('✓ Can import!')"

# 3. Check ports
lsof -i :5000
lsof -i :8000

# 4. View full error output
python interfaceml/web/app.py --port 8000 2>&1 | tee error.log
```

Share the error message for further assistance!

---

## 📝 Summary

**Fastest solution:**

```bash
cd "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/EGNNs-DEV/InterfaceML"
export PYTHONPATH=$PWD:$PYTHONPATH
python -m interfaceml.web.app --port 8000
```

Open in browser: **http://localhost:8000** 🎉

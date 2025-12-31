# Quick Testing Guide

**Get started testing InterfaceML in 5 minutes!**

## Step-by-Step Testing Instructions

### 1. Basic Verification (1 minute)

```bash
# Navigate to project
cd /path/to/InterfaceML

# Quick check - core modules
python -c "from interfaceml.core import io, layering; print('✓ Core modules working')"

# Quick check - basic tests
python tests/test_basic.py
```

**Expected:** All 4 tests should pass ✓

---

### 2. Run Automated Test Suite (2 minutes)

```bash
# Make test script executable (first time only)
chmod +x run_all_tests.sh

# Run all automated tests
./run_all_tests.sh
```

**Expected Output:**
```
╔════════════════════════════════════════════════════════════╗
║        InterfaceML Automated Test Suite                   ║
╚════════════════════════════════════════════════════════════╝

[1] Core Module Imports              ✓ PASSED
[2] I/O Module - CIF Loading         ✓ PASSED
[3] Layering Module                  ✓ PASSED
[4] Geometry Utils                   ✓ PASSED
...

╔════════════════════════════════════════════════════════════╗
║                    Test Summary                            ║
╠════════════════════════════════════════════════════════════╣
║  Total Tests:  10                                          ║
║  Passed:       8-10                                        ║
║  Failed:       0-2                                         ║
╚════════════════════════════════════════════════════════════╝
```

> **Note:** Some tests may be skipped if example structure files are missing. Core module tests should all pass.

---

### 3. Test Web Interface (1 minute)

```bash
# Terminal 1: Start web server
python -m interfaceml.web.app

# Terminal 2: Test health endpoint
curl http://localhost:5000/api/health
```

**Open browser to:** http://localhost:5000

**Quick checks:**
- [ ] Page loads with nice gradient background
- [ ] Can switch between tabs (Adsorbate, Interface, Layers, Density)
- [ ] File upload areas are visible
- [ ] No JavaScript errors in browser console (F12)

---

### 4. Test Command-Line Tools (1 minute)

```bash
# Test help messages
python build_heterojunctions/interface_builder.py --help | head -20
python build_heterojunctions/fix_interface_layers.py --help | head -20

# Quick build test (if you have structure files)
python build_heterojunctions/interface_builder.py \
    --adsorbate_mode \
    --base structures/perovskites/fapbi3-1.cif \
    --adsorbate structures/etl/C60-Ih.cif \
    --termination PbI \
    --distance 2.5 \
    --output test_output/quick_test.vasp

# Check output
ls -lh test_output/quick_test.vasp
```

---

## Test Results Checklist

After running tests, verify:

### ✅ Core Functionality
- [ ] `interfaceml.core.io` imports successfully
- [ ] `interfaceml.core.layering` imports successfully
- [ ] `interfaceml.utils.geometry` imports successfully
- [ ] Basic tests pass (4/4)

### ✅ CLI Tools
- [ ] `interface_builder.py` shows help message
- [ ] `fix_interface_layers.py` shows help message
- [ ] Can build a simple interface (if structure files available)

### ✅ Web Interface
- [ ] Web server starts without errors
- [ ] Health check returns `{"status": "ok"}`
- [ ] Frontend loads in browser
- [ ] Tabs switch correctly

### ✅ Python API
- [ ] Can import core modules in Python
- [ ] Can load structure files
- [ ] Can detect layers
- [ ] Can write POSCAR output

---

## Troubleshooting

### Problem: Import errors

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Problem: Structure files not found

```bash
# Solution: Verify you're in project root
pwd
ls structures/perovskites/

# If files are missing, the core tests will still pass
# Only integration tests may fail
```

### Problem: Web server port conflict

```bash
# Solution: Use different port
python -m interfaceml.web.app --port 8000
```

### Problem: Permission denied for test script

```bash
# Solution: Make executable
chmod +x run_all_tests.sh
```

---

## What to Test When Adding New Features

When you add a new module, test:

1. **Module imports**: `python -c "from interfaceml.core import your_module"`
2. **Basic functionality**: Create `tests/test_your_module.py`
3. **CLI tool**: Run with `--help` and test basic usage
4. **Integration**: Add to `run_all_tests.sh`

---

## Detailed Testing Guide

For comprehensive testing instructions, see:
- [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) - Complete testing procedures
- [EXTENSIBILITY_GUIDE.md](docs/EXTENSIBILITY_GUIDE.md) - How to test new features

---

## Quick Test Commands Reference

```bash
# Core module test
python tests/test_basic.py

# Full automated suite
./run_all_tests.sh

# Web interface
python -m interfaceml.web.app

# CLI tool help
python build_heterojunctions/interface_builder.py --help

# Python API test
python -c "from interfaceml.core import io, layering; print('OK')"

# Create test output directory
mkdir -p test_output

# Clean test outputs
rm -rf test_output/*
```

---

## Success Criteria

**Minimum for "Working Installation":**
- ✓ Core module tests pass (4/4)
- ✓ Can import `interfaceml.core.io` and `interfaceml.core.layering`
- ✓ Web app starts without errors

**Full Functionality:**
- ✓ All automated tests pass (or only skip tests with missing files)
- ✓ CLI tools run successfully
- ✓ Web interface loads and responds
- ✓ Can process example structures

---

**Time to complete: ~5 minutes**

**Next steps:**
1. Read [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for detailed tests
2. Review [EXTENSIBILITY_GUIDE.md](docs/EXTENSIBILITY_GUIDE.md) to add features
3. Start building interfaces! 🚀

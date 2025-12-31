# InterfaceML Development Guide

**Complete guide for testing and extending InterfaceML**

## Table of Contents
1. [Testing Overview](#testing-overview)
2. [Extensibility Overview](#extensibility-overview)
3. [Quick Start](#quick-start)
4. [Adding New Features](#adding-new-features)
5. [Architecture Patterns](#architecture-patterns)
6. [Best Practices](#best-practices)

---

## Testing Overview

### Test Levels

```
┌─────────────────────────────────────────────┐
│  Level 1: Core Module Tests (Required)     │
│  ✓ Import tests                             │
│  ✓ Function unit tests                      │
│  ✓ Edge case validation                     │
├─────────────────────────────────────────────┤
│  Level 2: Integration Tests                │
│  ✓ CLI tool execution                       │
│  ✓ End-to-end workflows                     │
│  ✓ File I/O validation                      │
├─────────────────────────────────────────────┤
│  Level 3: Web Interface Tests              │
│  ✓ API endpoint responses                   │
│  ✓ Frontend functionality                   │
│  ✓ File upload/download                     │
└─────────────────────────────────────────────┘
```

### Quick Test Commands

```bash
# 1. Core modules (30 seconds)
python tests/test_basic.py

# 2. Full suite (2-5 minutes)
./run_all_tests.sh

# 3. Web interface (manual, 2 minutes)
python -m interfaceml.web.app
# Open: http://localhost:5000

# 4. CLI tools (1 minute each)
python build_heterojunctions/interface_builder.py --help
python build_heterojunctions/fix_interface_layers.py --help
```

### Test Documentation

- **Quick Start**: `TESTING_QUICKSTART.md` (5 min read)
- **Detailed Guide**: `docs/TESTING_GUIDE.md` (30 min read)
- **Automated Script**: `run_all_tests.sh` (auto-run)

---

## Extensibility Overview

### Module Types

InterfaceML has **3 extension points**:

#### 1. Core Modules (Scientific Logic)
**Location**: `interfaceml/core/`
**Purpose**: Algorithms, analysis, data processing
**Example**: Strain calculation, bond analysis, surface reconstruction

```python
# interfaceml/core/your_analysis.py
from pymatgen.core import Structure

def analyze_feature(structure: Structure, param: float) -> dict:
    """Your scientific algorithm here."""
    results = {}
    # Implementation
    return results
```

#### 2. CLI Tools (User Interface)
**Location**: `build_heterojunctions/` or `interfaceml/cli/`
**Purpose**: Command-line access to core functionality
**Example**: Analysis tools, batch processors, file converters

```bash
# Usage pattern
python build_heterojunctions/your_tool.py --input file.cif --param 2.5
```

#### 3. Web Endpoints (API)
**Location**: `interfaceml/web/app.py`
**Purpose**: Browser-based access, REST API
**Example**: Interactive parameter tuning, visualization

```python
# API pattern
@app.route('/api/your-feature', methods=['POST'])
def your_feature():
    # Process and return JSON
    return jsonify(results)
```

---

## Quick Start

### Testing in 5 Minutes

1. **Verify installation** (30 sec):
```bash
cd InterfaceML
python -c "from interfaceml.core import io, layering; print('✓ Working')"
```

2. **Run basic tests** (1 min):
```bash
python tests/test_basic.py
```

3. **Test CLI tools** (2 min):
```bash
python build_heterojunctions/interface_builder.py --help
python build_heterojunctions/fix_interface_layers.py --help
```

4. **Test web interface** (1 min):
```bash
python -m interfaceml.web.app &
curl http://localhost:5000/api/health
```

### Adding a Feature in 10 Minutes

1. **Copy template** (30 sec):
```bash
cp templates/new_core_module_template.py interfaceml/core/my_feature.py
```

2. **Edit basics** (3 min):
```python
# Update docstring and function name
def analyze_my_feature(structure, threshold=2.5):
    """Analyze some feature."""
    # Your logic
    return results, metadata
```

3. **Add test** (2 min):
```bash
python -c "from interfaceml.core import my_feature; print('✓ Imports')"
```

4. **Create CLI wrapper** (3 min):
```bash
cp templates/new_cli_tool_template.py build_heterojunctions/my_tool.py
# Edit main() to call my_feature.analyze_my_feature()
```

5. **Test it** (1 min):
```bash
python build_heterojunctions/my_tool.py --help
```

---

## Adding New Features

### Pattern 1: Scientific Analysis Module

**Goal**: Add strain analysis functionality

**Steps**:

1. **Create core module**:
```python
# interfaceml/core/strain.py
def calculate_strain(structure, reference_spacing):
    """Calculate strain distribution."""
    # Algorithm here
    return strain_values, statistics
```

2. **Add CLI tool**:
```bash
cp templates/new_cli_tool_template.py build_heterojunctions/analyze_strain.py
```

3. **Add web endpoint** (optional):
```python
# interfaceml/web/app.py
@app.route('/api/analyze-strain', methods=['POST'])
def analyze_strain():
    from interfaceml.core import strain
    # Implementation
```

4. **Write tests**:
```python
# tests/test_strain.py
def test_strain_calculation():
    # Test code
```

5. **Document**:
```markdown
# README.md
### Strain Analysis
- Calculate layer strain
- CLI: python analyze_strain.py
```

### Pattern 2: Data Processing Tool

**Goal**: Add structure conversion utility

**Steps**:

1. **Create utility** (if shared):
```python
# interfaceml/utils/converters.py
def convert_format(input_file, output_format):
    """Convert between structure formats."""
```

2. **Create CLI**:
```python
# build_heterojunctions/convert_structure.py
# Simple wrapper around utility
```

3. **Add to README**:
```markdown
### Structure Conversion
Convert between CIF, VASP, XYZ formats.
```

### Pattern 3: Visualization Feature

**Goal**: Add 3D structure viewer to web interface

**Steps**:

1. **Add JavaScript library** (e.g., 3Dmol.js):
```html
<!-- interfaceml/web/templates/index.html -->
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
```

2. **Add view endpoint**:
```python
@app.route('/api/view-structure', methods=['POST'])
def view_structure():
    # Return structure data in JSON
```

3. **Add frontend UI**:
```html
<div id="viewer-3d"></div>
```

```javascript
// static/js/app.js
function displayStructure(structureData) {
    let viewer = $3Dmol.createViewer('viewer-3d');
    viewer.addModel(structureData, 'xyz');
    viewer.render();
}
```

---

## Architecture Patterns

### Layered Architecture

```
User Interfaces (CLI, Web, API)
        ↓
Core Modules (Business Logic)
        ↓
Utilities (Helper Functions)
        ↓
External Libraries (pymatgen, numpy)
```

**Rules**:
1. Core modules are **interface-independent**
2. CLI/Web **depend on** core, not vice versa
3. Utilities are **stateless helpers**
4. No circular dependencies

### Data Flow Pattern

```python
# Input → Load → Process → Output

# 1. Load structure
structure = io.load_structure(input_file)

# 2. Process with core module
results, metadata = your_module.analyze(structure, params)

# 3. Create output
modified_structure = apply_results(structure, results)

# 4. Write output
io.write_poscar(modified_structure, output_file)
```

### Error Handling Pattern

```python
def your_function(structure, param):
    # 1. Validate inputs
    if len(structure) == 0:
        raise ValueError("Empty structure")
    
    # 2. Try main logic
    try:
        result = complex_calculation(structure)
    except Exception as e:
        # 3. Provide context
        raise RuntimeError(f"Calculation failed: {e}") from e
    
    # 4. Validate outputs
    if result is None:
        raise RuntimeError("No results found")
    
    return result
```

---

## Best Practices

### Code Style

1. **Follow PEP 8**:
```bash
pip install flake8 black
flake8 interfaceml/
black interfaceml/
```

2. **Use type hints**:
```python
from typing import List, Dict, Optional
def func(x: float, y: Optional[str] = None) -> List[int]:
    pass
```

3. **Write docstrings** (NumPy style):
```python
def function(param1: float) -> int:
    """
    Brief description.
    
    Parameters
    ----------
    param1 : float
        Description
    
    Returns
    -------
    int
        Description
    """
```

### Testing Strategy

1. **Test during development**:
```bash
# After each change
python -c "from interfaceml.core import your_module"
```

2. **Write unit tests first** (TDD):
```python
def test_new_feature():
    # Write test before implementation
    result = your_module.new_feature(input_data)
    assert result == expected
```

3. **Test edge cases**:
```python
def test_edge_cases():
    # Empty input
    with pytest.raises(ValueError):
        your_module.function([])
    
    # Invalid parameters
    with pytest.raises(ValueError):
        your_module.function(data, param=-1)
```

### Documentation

1. **Update README.md** with new features
2. **Add examples** in docstrings
3. **Create tutorials** for complex features
4. **Keep changelog** of modifications

### Version Control

```bash
# Feature branch workflow
git checkout -b feature/strain-analysis

# Logical commits
git commit -m "Add strain calculation core module"
git commit -m "Add CLI tool for strain analysis"
git commit -m "Add tests for strain module"
git commit -m "Update documentation"

# Merge when complete
git checkout main
git merge feature/strain-analysis
```

---

## Common Development Tasks

### Task 1: Add New Analysis Algorithm

```bash
# 1. Create module from template
cp templates/new_core_module_template.py interfaceml/core/analysis.py

# 2. Implement algorithm
# Edit interfaceml/core/analysis.py

# 3. Test interactively
python -c "from interfaceml.core import analysis; print(dir(analysis))"

# 4. Add to package
# Edit interfaceml/core/__init__.py
# Add "analysis" to __all__

# 5. Write tests
cp tests/test_basic.py tests/test_analysis.py
# Edit tests/test_analysis.py

# 6. Run tests
python tests/test_analysis.py

# 7. Document
# Update README.md
```

### Task 2: Modify Web Interface

```bash
# 1. Add backend endpoint
# Edit interfaceml/web/app.py
@app.route('/api/new-feature', methods=['POST'])
def new_feature():
    # Implementation

# 2. Add frontend UI
# Edit interfaceml/web/templates/index.html
# Add new tab or section

# 3. Add JavaScript handler
# Edit interfaceml/web/static/js/app.js
async function handleNewFeature() {
    // Implementation
}

# 4. Test
python -m interfaceml.web.app
# Open browser and test manually
```

### Task 3: Create Batch Processing Script

```python
#!/usr/bin/env python3
"""Batch process multiple structures."""

from pathlib import Path
from interfaceml.core import io, your_module

def process_directory(input_dir, output_dir, params):
    """Process all CIF files in directory."""
    for cif_file in Path(input_dir).glob("*.cif"):
        structure = io.load_structure(cif_file)
        results, _ = your_module.analyze(structure, **params)
        
        output_file = Path(output_dir) / f"{cif_file.stem}_processed.vasp"
        io.write_poscar(results, output_file)
        print(f"✓ Processed: {cif_file.name}")

if __name__ == "__main__":
    process_directory("input/", "output/", {"param1": 2.5})
```

---

## Resources

### Documentation
- `README.md` - Project overview
- `QUICK_START.md` - 5-minute guide
- `TESTING_QUICKSTART.md` - Quick testing
- `docs/TESTING_GUIDE.md` - Detailed testing
- `docs/EXTENSIBILITY_GUIDE.md` - Extension guide
- `docs/GETTING_STARTED.md` - Tutorial
- `PROJECT_STRUCTURE.md` - Architecture

### Templates
- `templates/new_core_module_template.py` - Core module
- `templates/new_cli_tool_template.py` - CLI tool
- `templates/README_templates.md` - Template guide

### Scripts
- `run_all_tests.sh` - Automated testing
- `examples/example_workflow.sh` - Usage example

### Examples
- `interfaceml/core/io.py` - I/O patterns
- `interfaceml/core/layering.py` - Algorithm patterns
- `build_heterojunctions/interface_builder.py` - Complex CLI
- `interfaceml/web/app.py` - Web API patterns

---

## Getting Help

### Workflow Issues
1. Check `TESTING_QUICKSTART.md`
2. Run `./run_all_tests.sh`
3. Review error messages
4. Check `docs/TESTING_GUIDE.md` troubleshooting

### Extension Questions
1. Check `docs/EXTENSIBILITY_GUIDE.md`
2. Review existing modules for patterns
3. Use templates as starting points
4. Test frequently during development

### Bug Reports
- Create issue on GitHub
- Include test case
- Provide environment details
- Show error messages

---

## Development Workflow Summary

```
1. Plan Feature
   ↓
2. Copy Template
   ↓
3. Implement Logic
   ↓
4. Write Tests
   ↓
5. Test Locally
   ↓
6. Add CLI/Web (optional)
   ↓
7. Update Documentation
   ↓
8. Run Full Test Suite
   ↓
9. Commit Changes
   ↓
10. Share/Deploy
```

---

**Ready to develop and extend InterfaceML! 🚀**

For questions or contributions, see:
- GitHub: https://github.com/yourusername/InterfaceML
- Issues: https://github.com/yourusername/InterfaceML/issues
- Discussions: https://github.com/yourusername/InterfaceML/discussions

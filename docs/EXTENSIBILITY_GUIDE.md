# InterfaceML Extensibility Guide

This guide explains how to extend InterfaceML with new functionality modules while maintaining clean architecture and backward compatibility.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Adding Core Modules](#adding-core-modules)
3. [Adding CLI Tools](#adding-cli-tools)
4. [Adding Web Endpoints](#adding-web-endpoints)
5. [Integration Patterns](#integration-patterns)
6. [Best Practices](#best-practices)
7. [Example: Adding a New Module](#example-adding-a-new-module)

---

## Architecture Overview

InterfaceML follows a **modular, layered architecture**:

```
┌─────────────────────────────────────────────────┐
│         User Interfaces (Multiple Entry Points) │
├──────────────┬──────────────┬───────────────────┤
│  CLI Tools   │  Web App     │  Python API       │
├──────────────┴──────────────┴───────────────────┤
│         Core Modules (Business Logic)           │
├─────────────────────────────────────────────────┤
│         Utilities (Helper Functions)            │
├─────────────────────────────────────────────────┤
│         External Dependencies (pymatgen, etc.)  │
└─────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Core logic is independent of interface
2. **DRY (Don't Repeat Yourself)**: Shared code in utilities
3. **Backward Compatibility**: Existing APIs remain stable
4. **Modularity**: New features don't break existing ones
5. **Testability**: Each module can be tested independently

---

## Adding Core Modules

Core modules contain the business logic and should be **interface-independent**.

### Step 1: Create Module File

```bash
# Create new module in interfaceml/core/
touch interfaceml/core/your_module.py
```

### Step 2: Module Template

```python
"""
your_module.py

Brief description of what this module does.

This module provides functionality for [specific task], including:
- Feature 1
- Feature 2
- Feature 3

Author: Your Name
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import numpy as np
from pymatgen.core import Structure

# Import from other core modules
from interfaceml.core.io import load_structure, get_element_symbols
from interfaceml.core.layering import interface_normal_unit


def your_main_function(
    structure: Structure,
    *,
    parameter1: float,
    parameter2: Optional[str] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Brief one-line description.

    Detailed description of what this function does, including any
    important assumptions, limitations, or usage notes.

    Parameters
    ----------
    structure
        Input pymatgen Structure object.
    parameter1
        Description of parameter1.
    parameter2
        Optional parameter with default None.

    Returns
    -------
    result_indices
        List of atom indices that satisfy some condition.
    metadata
        Dictionary containing additional information about the result.

    Raises
    ------
    ValueError
        If structure is invalid or parameters are out of range.

    Examples
    --------
    >>> from interfaceml.core import io, your_module
    >>> struct = io.load_structure("file.cif")
    >>> indices, meta = your_module.your_main_function(struct, parameter1=2.5)
    """
    # Input validation
    if len(structure) == 0:
        raise ValueError("Structure cannot be empty")
    
    # Implementation
    result_indices = []
    metadata = {}
    
    # Your algorithm here
    # ...
    
    return result_indices, metadata


def _helper_function(data: np.ndarray) -> float:
    """
    Private helper function (prefix with underscore).

    This won't be exported in __all__ but can be used internally.
    """
    # Helper implementation
    return float(np.mean(data))


# Public API - explicitly list what should be importable
__all__ = [
    "your_main_function",
    # Add other public functions here
]
```

### Step 3: Add to __init__.py

```python
# Edit interfaceml/core/__init__.py
"""
Core functionality modules for InterfaceML.
...
- your_module: Description of your module
"""

__all__ = ["io", "layering", "your_module"]  # Add your module
```

### Step 4: Write Tests

Create `tests/test_your_module.py`:

```python
"""
Tests for your_module.

Run with: python tests/test_your_module.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaceml.core import io, your_module
import numpy as np


def test_your_main_function():
    """Test the main function with known input."""
    print("Testing your_main_function...")
    
    try:
        # Load test structure
        struct = io.load_structure("structures/perovskites/fapbi3-1.cif")
        
        # Call your function
        indices, meta = your_module.your_main_function(
            struct,
            parameter1=2.5
        )
        
        # Assertions
        assert len(indices) > 0, "Should return non-empty indices"
        assert isinstance(meta, dict), "Should return metadata dict"
        
        print(f"✓ Function returned {len(indices)} indices")
        print(f"✓ Metadata: {meta}")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    # Test with empty structure (should raise error)
    # Test with invalid parameters
    # Test with extreme values
    # ...
    
    print("✓ Edge cases handled correctly")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Your Module")
    print("=" * 60)
    
    tests = [
        test_your_main_function,
        test_edge_cases,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    sys.exit(0 if all(results) else 1)
```

---

## Adding CLI Tools

CLI tools provide command-line access to core functionality.

### Step 1: Create CLI Script

```bash
# Option A: Create in build_heterojunctions/ (for backward compatibility)
touch build_heterojunctions/your_tool.py

# Option B: Create wrapper in interfaceml/cli/
touch interfaceml/cli/your_tool.py
```

### Step 2: CLI Template

```python
#!/usr/bin/env python3
"""
your_tool.py

Command-line tool for [specific functionality].

Usage:
    python your_tool.py --input file.vasp --param1 value --output result.vasp

Examples:
    # Basic usage
    python your_tool.py --input structure.cif --output modified.vasp
    
    # With parameters
    python your_tool.py --input file.cif --param1 2.5 --param2 auto
"""

import argparse
import sys
from pathlib import Path

# Import core functionality
from interfaceml.core import io, your_module


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Your tool description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Example 1
  %(prog)s --input structure.cif --output result.vasp
  
  # Example 2
  %(prog)s --input file.vasp --param1 2.5 --verbose
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input structure file (CIF, VASP, or XYZ)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path"
    )
    
    # Functionality-specific arguments
    parser.add_argument(
        "--param1",
        type=float,
        default=2.5,
        help="Description of parameter 1 (default: 2.5)"
    )
    
    parser.add_argument(
        "--param2",
        type=str,
        choices=["auto", "manual"],
        default="auto",
        help="Description of parameter 2 (default: auto)"
    )
    
    # Optional flags
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Print parameters if verbose
    if args.verbose:
        print("=" * 60)
        print("Your Tool - Parameters")
        print("=" * 60)
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Param1: {args.param1}")
        print(f"Param2: {args.param2}")
        print("=" * 60)
    
    try:
        # Load structure
        if args.verbose:
            print(f"\nLoading structure from {args.input}...")
        
        structure = io.load_structure(args.input)
        
        if args.verbose:
            print(f"✓ Loaded {len(structure)} atoms")
            print(f"  Composition: {structure.composition.formula}")
        
        # Call core functionality
        if args.verbose:
            print(f"\nProcessing with your_module...")
        
        result_indices, metadata = your_module.your_main_function(
            structure,
            parameter1=args.param1,
            parameter2=args.param2 if args.param2 != "auto" else None
        )
        
        if args.verbose:
            print(f"✓ Found {len(result_indices)} results")
            print(f"  Metadata: {metadata}")
        
        # Modify structure or create output
        # (This depends on what your tool does)
        
        # Write output
        if args.verbose:
            print(f"\nWriting output to {args.output}...")
        
        io.write_poscar(structure, args.output)
        
        if args.verbose:
            print(f"✓ Output written successfully")
        
        print(f"\n✓ Processing complete: {args.output}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 3: Add to setup.py Entry Points

```python
# Edit setup.py
entry_points={
    "console_scripts": [
        "interfaceml-build=interfaceml.cli.build_interface:main",
        "interfaceml-fix=interfaceml.cli.fix_layers:main",
        "interfaceml-your-tool=interfaceml.cli.your_tool:main",  # Add this
    ],
},
```

---

## Adding Web Endpoints

### Step 1: Add API Endpoint

Edit `interfaceml/web/app.py`:

```python
@app.route('/api/your-endpoint', methods=['POST'])
def your_endpoint():
    """
    API endpoint for your functionality.

    Expected JSON:
        {
            "input_file": "path/to/file.vasp",
            "param1": 2.5,
            "param2": "value"
        }

    Returns:
        JSON with results or error message
    """
    if not CORE_AVAILABLE:
        return jsonify({'error': 'Core modules not available'}), 500
    
    data = request.json
    input_file = data.get('input_file')
    
    if not input_file or not Path(input_file).exists():
        return jsonify({'error': 'Invalid input file'}), 400
    
    try:
        from interfaceml.core import io, your_module
        
        # Load structure
        structure = io.load_structure(input_file)
        
        # Process with your module
        result_indices, metadata = your_module.your_main_function(
            structure,
            parameter1=float(data.get('param1', 2.5)),
            parameter2=data.get('param2')
        )
        
        # Generate output
        output_filename = Path(input_file).stem + "_processed.vasp"
        output_path = Path(app.config['UPLOAD_FOLDER']) / output_filename
        io.write_poscar(structure, output_path)
        
        return jsonify({
            'status': 'success',
            'n_results': len(result_indices),
            'metadata': metadata,
            'output_file': str(output_path),
            'download_url': f'/api/download/{output_filename}',
            'result_indices': result_indices
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Step 2: Add Frontend UI

Edit `interfaceml/web/templates/index.html` to add a new tab:

```html
<!-- Add tab button -->
<button class="tab-btn" data-tab="your-feature">
    <span class="tab-icon">🔧</span>
    Your Feature Name
</button>

<!-- Add tab content -->
<div id="your-feature" class="tab-content">
    <div class="panel">
        <h2 class="panel-title">Your Feature Title</h2>
        <p class="panel-description">
            Description of what this feature does.
        </p>

        <!-- File upload -->
        <div class="form-section">
            <h3>Input Structure</h3>
            <div class="file-upload-area" id="your-feature-upload">
                <input type="file" id="your-feature-file" accept=".cif,.vasp,.xyz" hidden>
                <label for="your-feature-file" class="file-upload-label">
                    <span class="upload-icon">📁</span>
                    <span class="upload-text">Upload structure</span>
                </label>
            </div>
            <div id="your-feature-info" class="file-info hidden"></div>
        </div>

        <!-- Parameters -->
        <div class="form-section">
            <h3>Parameters</h3>
            <div class="param-grid">
                <div class="param-item">
                    <label for="your-param1">Parameter 1</label>
                    <input type="number" id="your-param1" class="form-control" 
                           value="2.5" step="0.1">
                </div>
                <div class="param-item">
                    <label for="your-param2">Parameter 2</label>
                    <select id="your-param2" class="form-control">
                        <option value="auto">Auto</option>
                        <option value="manual">Manual</option>
                    </select>
                </div>
            </div>
        </div>

        <!-- Action button -->
        <div class="action-bar">
            <button id="your-feature-btn" class="btn btn-primary">
                <span class="btn-icon">🚀</span>
                Process
            </button>
        </div>

        <!-- Results -->
        <div id="your-feature-result" class="result-panel hidden"></div>
    </div>
</div>
```

### Step 3: Add Frontend JavaScript

Edit `interfaceml/web/static/js/app.js`:

```javascript
// Add to initializeFileUploads()
setupFileUpload('your-feature-file', 'your_feature', displayFileInfo);

// Add to initializeButtons()
const yourFeatureBtn = document.getElementById('your-feature-btn');
if (yourFeatureBtn) {
    yourFeatureBtn.addEventListener('click', processYourFeature);
}

// Add processing function
async function processYourFeature() {
    if (!state.uploadedFiles.your_feature) {
        showMessage('Please upload a structure file', 'error');
        return;
    }
    
    const data = {
        input_file: state.uploadedFiles.your_feature.filepath,
        param1: parseFloat(document.getElementById('your-param1').value),
        param2: document.getElementById('your-param2').value
    };
    
    try {
        showLoading(true);
        const response = await fetch('/api/your-endpoint', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayYourFeatureResult(result);
        } else {
            displayResult('your-feature-result', result, false);
        }
    } catch (error) {
        displayResult('your-feature-result', { error: error.message }, false);
    } finally {
        showLoading(false);
    }
}

function displayYourFeatureResult(result) {
    const resultDiv = document.getElementById('your-feature-result');
    resultDiv.classList.remove('hidden', 'error');
    resultDiv.innerHTML = `
        <h4>✓ Processing Complete</h4>
        <p><strong>Results found:</strong> ${result.n_results}</p>
        <p><strong>Metadata:</strong> ${JSON.stringify(result.metadata)}</p>
        <p><strong>Output:</strong> <a href="${result.download_url}" download>Download File</a></p>
    `;
}
```

---

## Integration Patterns

### Pattern 1: Reusing Existing Core Functions

```python
# In your new module, reuse existing functionality
from interfaceml.core.io import load_structure
from interfaceml.core.layering import interface_normal_unit

def your_new_function(structure: Structure):
    # Leverage existing code
    normal = interface_normal_unit(structure)
    # Add your logic
    # ...
```

### Pattern 2: Creating Utility Functions

```python
# Add to interfaceml/utils/ for shared helpers
# interfaceml/utils/chemistry.py

def get_bond_length(elem1: str, elem2: str) -> float:
    """Get typical bond length between two elements."""
    bond_lengths = {
        ('C', 'C'): 1.54,
        ('C', 'N'): 1.47,
        # ...
    }
    return bond_lengths.get((elem1, elem2)) or bond_lengths.get((elem2, elem1))
```

### Pattern 3: Plugin Architecture

For maximum extensibility, create a plugin system:

```python
# interfaceml/core/plugins.py

class PluginBase:
    """Base class for plugins."""
    
    name: str = "plugin"
    version: str = "1.0.0"
    
    def process(self, structure: Structure, **kwargs):
        """Main processing method to be implemented by plugins."""
        raise NotImplementedError

# Your plugin
class YourPlugin(PluginBase):
    name = "your_plugin"
    
    def process(self, structure, **kwargs):
        # Your implementation
        return result
```

---

## Best Practices

### Code Style

1. **Follow PEP 8**:
```bash
# Check code style
pip install flake8
flake8 interfaceml/core/your_module.py
```

2. **Use type hints**:
```python
def your_function(x: float) -> List[int]:
    """Always use type annotations."""
    pass
```

3. **Write docstrings**:
```python
def your_function(x: float) -> int:
    """
    Use NumPy/Google style docstrings.
    
    Parameters
    ----------
    x : float
        Description
    
    Returns
    -------
    int
        Description
    """
    pass
```

### Testing

1. **Write tests first** (TDD approach)
2. **Test edge cases** (empty inputs, extreme values)
3. **Use assertions** to validate behavior
4. **Add integration tests** for end-to-end workflows

### Documentation

1. **Update README.md** with new features
2. **Add examples** to docs/
3. **Include usage** in docstrings
4. **Create tutorials** for complex features

### Version Control

```bash
# Create feature branch
git checkout -b feature/your-new-module

# Make commits with clear messages
git add interfaceml/core/your_module.py
git commit -m "Add your_module for [specific functionality]"

# Update tests
git add tests/test_your_module.py
git commit -m "Add tests for your_module"

# Update documentation
git add docs/
git commit -m "Update docs for your_module"
```

---

## Example: Adding a New Module

Let's add a **strain analysis** module as a complete example.

### 1. Create Core Module

File: `interfaceml/core/strain.py`

```python
"""
strain.py

Strain analysis module for interfaces and heterostructures.

This module provides tools for calculating and visualizing strain
distributions in interface structures.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from pymatgen.core import Structure

from interfaceml.core.io import get_element_symbols
from interfaceml.core.layering import interface_normal_unit


def calculate_layer_strain(
    structure: Structure,
    reference_spacing: float,
) -> Dict[str, np.ndarray]:
    """
    Calculate strain for each layer relative to reference spacing.
    
    Parameters
    ----------
    structure
        Input structure
    reference_spacing
        Reference interlayer spacing in Angstroms
    
    Returns
    -------
    strain_data
        Dictionary containing strain values and positions
    """
    normal = interface_normal_unit(structure)
    heights = np.dot(structure.cart_coords, normal)
    
    # Calculate local spacing
    spacings = np.diff(np.sort(heights))
    strains = (spacings - reference_spacing) / reference_spacing
    
    return {
        'strains': strains,
        'heights': heights,
        'mean_strain': float(np.mean(strains)),
        'max_strain': float(np.max(np.abs(strains)))
    }


__all__ = ['calculate_layer_strain']
```

### 2. Add CLI Tool

File: `build_heterojunctions/analyze_strain.py`

```python
#!/usr/bin/env python3
"""Strain analysis CLI tool."""

import argparse
from interfaceml.core import io, strain

def main():
    parser = argparse.ArgumentParser(description="Analyze interface strain")
    parser.add_argument("--input", required=True, help="Input structure")
    parser.add_argument("--reference", type=float, default=2.5,
                       help="Reference spacing (Å)")
    args = parser.parse_args()
    
    structure = io.load_structure(args.input)
    result = strain.calculate_layer_strain(structure, args.reference)
    
    print(f"Mean strain: {result['mean_strain']:.3%}")
    print(f"Max strain: {result['max_strain']:.3%}")

if __name__ == "__main__":
    main()
```

### 3. Add Web Endpoint

In `interfaceml/web/app.py`:

```python
@app.route('/api/analyze-strain', methods=['POST'])
def analyze_strain():
    """Analyze strain in uploaded structure."""
    data = request.json
    structure_file = data.get('structure_file')
    reference = float(data.get('reference_spacing', 2.5))
    
    try:
        from interfaceml.core import io, strain
        structure = io.load_structure(structure_file)
        result = strain.calculate_layer_strain(structure, reference)
        
        return jsonify({
            'status': 'success',
            'mean_strain': result['mean_strain'],
            'max_strain': result['max_strain'],
            'strain_distribution': result['strains'].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### 4. Write Tests

File: `tests/test_strain.py`

```python
from interfaceml.core import io, strain

def test_strain_calculation():
    """Test strain calculation."""
    struct = io.load_structure("structures/perovskites/fapbi3-1.cif")
    result = strain.calculate_layer_strain(struct, 2.5)
    
    assert 'mean_strain' in result
    assert 'max_strain' in result
    assert isinstance(result['strains'], np.ndarray)
    
    print("✓ Strain calculation test passed")
    return True

if __name__ == "__main__":
    test_strain_calculation()
```

### 5. Update Documentation

Add to `README.md`:

```markdown
### Strain Analysis
- Calculate layer-by-layer strain
- Compare to reference structures
- Visualize strain distribution
```

---

## Maintenance Checklist

When adding new modules, ensure:

- [ ] Core logic is interface-independent
- [ ] Functions have type hints and docstrings
- [ ] Tests are written and pass
- [ ] CLI tool has --help and examples
- [ ] Web endpoint returns proper JSON
- [ ] Documentation is updated
- [ ] Code follows PEP 8 style
- [ ] No circular imports
- [ ] Backward compatibility maintained
- [ ] Example usage provided

---

## Getting Help

- Review existing modules for patterns
- Check `PROJECT_STRUCTURE.md` for architecture
- Run tests frequently: `python tests/test_basic.py`
- Use git branches for new features
- Ask for code reviews

---

**Ready to extend InterfaceML! 🚀**

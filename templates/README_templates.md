# InterfaceML Code Templates

This directory contains templates to help you quickly add new functionality to InterfaceML.

## Available Templates

### 1. Core Module Template
**File:** `new_core_module_template.py`

**Use when:** Adding new scientific functionality (algorithms, analysis tools)

**Features:**
- Comprehensive docstring format
- Type hints throughout
- Input validation
- Error handling
- Helper functions pattern
- __all__ export declaration
- Command-line test interface

**Quick start:**
```bash
# Copy template
cp templates/new_core_module_template.py interfaceml/core/your_module.py

# Edit the file
# - Replace "your_module_name" with actual name
# - Implement main_function()
# - Add helper functions as needed
# - Update __all__ list

# Test it
python -c "from interfaceml.core import your_module; print('✓ Module imported')"
```

---

### 2. CLI Tool Template
**File:** `new_cli_tool_template.py`

**Use when:** Creating command-line tools

**Features:**
- argparse setup with groups
- Comprehensive help messages
- Input validation
- Error handling with debug mode
- Verbose output option
- Dry-run mode
- Progress reporting
- Exit code handling

**Quick start:**
```bash
# Copy template
cp templates/new_cli_tool_template.py build_heterojunctions/your_tool.py

# Make executable
chmod +x build_heterojunctions/your_tool.py

# Edit and customize
# - Update description
# - Add your parameters
# - Implement main() logic

# Test it
python build_heterojunctions/your_tool.py --help
```

---

## Template Usage Workflow

### Step 1: Choose Template

**Core Module** → For scientific algorithms
**CLI Tool** → For command-line interface

### Step 2: Copy and Rename

```bash
# For core module
cp templates/new_core_module_template.py interfaceml/core/strain_analysis.py

# For CLI tool
cp templates/new_cli_tool_template.py build_heterojunctions/analyze_strain.py
```

### Step 3: Customize

1. **Update docstrings** - Replace placeholder text
2. **Modify parameters** - Add/remove as needed
3. **Implement logic** - Fill in the actual functionality
4. **Update __all__** - List public functions

### Step 4: Test

```bash
# Core module
python -c "from interfaceml.core import your_module"

# CLI tool
python build_heterojunctions/your_tool.py --help
```

### Step 5: Document

- Add to `interfaceml/__init__.py` (core modules)
- Update `README.md` with new feature
- Add examples to `docs/`

---

## Template Features Explained

### Core Module Template

**Structure:**
```python
"""Module docstring"""
# Imports
# Functions with full docstrings
# Helper functions (public and private)
# __all__ declaration
# Optional CLI test code
```

**Key patterns:**
- Functions return `Tuple[data, metadata]`
- Use `Structure` as primary input type
- Private helpers start with `_`
- Type hints for everything
- Comprehensive error checking

### CLI Tool Template

**Structure:**
```python
"""Tool docstring"""
# Imports with availability check
# create_parser() - argument setup
# validate_args() - input validation
# main() - main processing logic
```

**Key patterns:**
- Argument groups for organization
- Verbose and debug modes
- Try-except with informative errors
- Return exit codes (0=success)
- Pretty output formatting

---

## Customization Tips

### Adding Custom Parameters

**Core module:**
```python
def your_function(
    structure: Structure,
    *,
    new_param: float,  # Add here
    optional_param: Optional[str] = None,
) -> Tuple[List[int], Dict]:
    # Implementation
```

**CLI tool:**
```python
parser.add_argument(
    "--new-param",
    type=float,
    default=2.5,
    help="Description of new parameter"
)
```

### Adding Type Hints

```python
from typing import List, Dict, Tuple, Optional, Union, Set
import numpy as np
from pymatgen.core import Structure

def function(
    x: float,
    y: Optional[str] = None,
) -> Tuple[List[int], Dict[str, float]]:
    result: List[int] = []
    meta: Dict[str, float] = {}
    return result, meta
```

### Error Handling Pattern

```python
try:
    # Your code
    if invalid_condition:
        raise ValueError("Clear error message")
    
except ValueError as e:
    print(f"Validation error: {e}")
    if args.debug:
        raise
    sys.exit(1)
    
except Exception as e:
    print(f"Unexpected error: {e}")
    if args.debug:
        import traceback
        traceback.print_exc()
    sys.exit(1)
```

---

## Integration with Existing Code

### Reusing Core Functions

```python
# In your new module
from interfaceml.core.io import load_structure
from interfaceml.core.layering import interface_normal_unit

def your_function(structure: Structure):
    # Leverage existing functionality
    normal = interface_normal_unit(structure)
    # Your new logic
    return results
```

### Calling from CLI

```python
# In your CLI tool
from interfaceml.core import io, your_module

structure = io.load_structure(args.input)
results, metadata = your_module.your_function(
    structure,
    parameter1=args.param1
)
```

### Adding to Web API

```python
# In interfaceml/web/app.py
@app.route('/api/your-endpoint', methods=['POST'])
def your_endpoint():
    data = request.json
    structure = io.load_structure(data['input_file'])
    results = your_module.your_function(structure, **params)
    return jsonify(results)
```

---

## Testing Your New Code

### 1. Unit Test Template

```python
# tests/test_your_module.py
from interfaceml.core import io, your_module

def test_basic_functionality():
    struct = io.load_structure("structures/perovskites/fapbi3-1.cif")
    result, meta = your_module.your_function(struct, parameter1=2.5)
    assert len(result) > 0
    assert 'key' in meta
    print("✓ Test passed")

if __name__ == "__main__":
    test_basic_functionality()
```

### 2. Integration Test

Add to `run_all_tests.sh`:
```bash
run_test "Your New Feature" \
    "python build_heterojunctions/your_tool.py --input test.cif --output test.vasp"
```

---

## Best Practices Checklist

When using templates:

- [ ] Update all placeholder text
- [ ] Add comprehensive docstrings
- [ ] Include type hints
- [ ] Validate inputs
- [ ] Handle errors gracefully
- [ ] Write unit tests
- [ ] Update documentation
- [ ] Test with real data
- [ ] Check code style (flake8)
- [ ] Add to __all__ exports

---

## Examples of Real Modules

Study these for patterns:

**Core modules:**
- `interfaceml/core/io.py` - File I/O
- `interfaceml/core/layering.py` - Algorithmic logic
- `interfaceml/utils/geometry.py` - Helper functions

**CLI tools:**
- `build_heterojunctions/interface_builder.py` - Complex tool
- `build_heterojunctions/fix_interface_layers.py` - Multiple modes

---

## Quick Reference

### Copy Template
```bash
cp templates/new_core_module_template.py interfaceml/core/NAME.py
```

### Edit Key Sections
1. Module docstring (top)
2. Function signatures
3. Implementation logic
4. __all__ list

### Test
```bash
python -c "from interfaceml.core import NAME"
python tests/test_NAME.py
```

### Integrate
1. Add to `interfaceml/core/__init__.py`
2. Update `README.md`
3. Add to test suite

---

## Getting Help

- **Templates not working?** Check [EXTENSIBILITY_GUIDE.md](../docs/EXTENSIBILITY_GUIDE.md)
- **Need examples?** See existing modules in `interfaceml/core/`
- **Questions?** Open an issue on GitHub

---

**Happy coding! 🚀**

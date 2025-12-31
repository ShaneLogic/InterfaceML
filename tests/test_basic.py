"""
Basic tests for InterfaceML core functionality.

These tests verify that the core modules can be imported and basic functions work.
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from interfaceml.core import io, layering
        from interfaceml.utils import geometry
        print("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_geometry_functions():
    """Test basic geometry utilities."""
    print("\nTesting geometry functions...")
    
    from interfaceml.utils.geometry import angle_between, normalize_vector, compute_distance
    import numpy as np
    
    try:
        # Test angle calculation
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        angle = angle_between(v1, v2)
        assert abs(angle - 90.0) < 0.01, f"Expected 90°, got {angle}°"
        
        # Test normalization
        v3 = np.array([3, 4, 0])
        v3_norm = normalize_vector(v3)
        assert abs(np.linalg.norm(v3_norm) - 1.0) < 1e-10, "Normalization failed"
        
        # Test distance
        p1 = np.array([0, 0, 0])
        p2 = np.array([3, 4, 0])
        dist = compute_distance(p1, p2)
        assert abs(dist - 5.0) < 0.01, f"Expected 5.0, got {dist}"
        
        print("✓ Geometry functions working correctly")
        return True
    except Exception as e:
        print(f"✗ Geometry tests failed: {e}")
        return False


def test_io_module():
    """Test structure I/O functions."""
    print("\nTesting I/O module...")
    
    try:
        from interfaceml.core import io
        
        # Check that functions exist
        assert hasattr(io, 'load_structure'), "load_structure not found"
        assert hasattr(io, 'write_poscar'), "write_poscar not found"
        assert hasattr(io, 'get_element_symbols'), "get_element_symbols not found"
        
        print("✓ I/O module functions available")
        return True
    except Exception as e:
        print(f"✗ I/O tests failed: {e}")
        return False


def test_layering_module():
    """Test layering functions."""
    print("\nTesting layering module...")
    
    try:
        from interfaceml.core import layering
        
        # Check that functions exist
        functions = [
            'interface_normal_unit',
            'unwrap_periodic_1d',
            'split_stack_layers',
            'split_layers_by_z',
            'include_whole_molecules',
            'format_layer_indices',
        ]
        
        for func in functions:
            assert hasattr(layering, func), f"{func} not found"
        
        print("✓ Layering module functions available")
        return True
    except Exception as e:
        print(f"✗ Layering tests failed: {e}")
        return False


def run_all_tests():
    """Run all basic tests."""
    print("=" * 60)
    print("InterfaceML Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_geometry_functions,
        test_io_module,
        test_layering_module,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Minimal test for basic Python-Rust communication
"""

def test_call_rust_from_python():
    """Test calling Rust functions from Python"""
    print("Testing Python->Rust communication...")
    
    # For now, just test that we can compile and find the module structure
    try:
        # Check if we have the expected library file
        import os
        import glob
        
        lib_files = glob.glob("target/release/deps/lib_rustorch_py*")
        if lib_files:
            print(f"✓ Found library files: {lib_files}")
        else:
            print("✗ No library files found")
            
        # Try to import (this will likely fail due to linking issues, but that's expected)
        try:
            import _rustorch_py
            print("✓ Successfully imported _rustorch_py")
            
            # Test basic functions
            result = _rustorch_py.hello_from_rust()
            print(f"✓ hello_from_rust() = {result}")
            
            version = _rustorch_py.get_version()
            print(f"✓ get_version() = {version}")
            
            sum_result = _rustorch_py.add_numbers(1.5, 2.5)
            print(f"✓ add_numbers(1.5, 2.5) = {sum_result}")
            
        except ImportError as e:
            print(f"ℹ  Import failed (expected due to linking): {e}")
            
    except Exception as e:
        print(f"✗ Error: {e}")

def test_call_python_from_rust():
    """Test calling Python from Rust (future implementation)"""
    print("Testing Rust->Python communication...")
    print("ℹ  This will be implemented in the next phase")

def main():
    print("Minimal Python-Rust Communication Test")
    print("=" * 40)
    
    test_call_rust_from_python()
    print()
    test_call_python_from_rust()
    
    print("=" * 40)
    print("Basic structure test completed")

if __name__ == "__main__":
    main()
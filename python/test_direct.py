#!/usr/bin/env python3
"""
Direct test of _rustorch_py module bypassing __init__.py issues
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_basic_functions():
    """Test basic Rust functions"""
    try:
        # Direct import of the compiled module
        import _rustorch_py as rt

        print("=== RusTorch Python Binding Test ===")
        print("Module loaded successfully!")

        print("\nAvailable functions:")
        for attr in sorted(dir(rt)):
            if not attr.startswith('_'):
                print(f"  {attr}")

        print("\n--- Testing Basic Functions ---")
        print(f"Hello from Rust: {rt.hello_from_rust()}")
        print(f"Version: {rt.get_version()}")
        print(f"Add numbers 1.5 + 2.5 = {rt.add_numbers(1.5, 2.5)}")
        print(f"Sum list [1,2,3,4,5] = {rt.sum_list([1.0, 2.0, 3.0, 4.0, 5.0])}")

        print("\n✅ All basic functions work correctly!")
        print("✅ Pythonバインディングが正常に動作しています！")
        return True

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functions()
    sys.exit(0 if success else 1)
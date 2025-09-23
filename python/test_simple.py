#!/usr/bin/env python3
"""
Simple test for RusTorch Python bindings Phase 1
"""

try:
    import rustorch
    print("✓ Successfully imported rustorch")

    # Test tensor creation
    print("\n=== Testing Tensor Creation ===")

    # Test zeros
    t1 = rustorch.zeros([2, 3])
    print(f"✓ zeros([2, 3]): {t1}")
    print(f"  shape: {t1.shape}")
    print(f"  numel: {t1.numel}")
    print(f"  ndim: {t1.ndim}")

    # Test ones
    t2 = rustorch.ones([2, 3])
    print(f"✓ ones([2, 3]): {t2}")

    # Test tensor from list
    t3 = rustorch.tensor([1.0, 2.0, 3.0])
    print(f"✓ tensor([1.0, 2.0, 3.0]): {t3}")
    print(f"  shape: {t3.shape}")

    print("\n=== Testing Tensor Operations ===")

    # Test addition
    result = t1 + t2
    print(f"✓ Addition (zeros + ones): {result}")

    # Test multiplication
    result = t2 * t2
    print(f"✓ Multiplication (ones * ones): {result}")

    print("\n🎉 All tests passed!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to build with: maturin develop")

except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
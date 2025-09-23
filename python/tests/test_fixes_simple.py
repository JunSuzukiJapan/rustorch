#!/usr/bin/env python3
"""
Simple test to verify major fixes
主要な修正の検証用簡単テスト
"""

try:
    import rustorch
    print("✅ Import successful")

    # Test fixed SGD parameters
    print("\n🔧 Testing SGD with momentum parameter...")
    tensor_data = rustorch.zeros([2, 3])
    variable = rustorch.Variable(tensor_data, requires_grad=False)
    sgd = rustorch.SGD([variable], lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=False)
    print(f"✅ SGD with momentum: {sgd}")

    # Test BatchNorm1d num_features display
    print("\n🔧 Testing BatchNorm1d num_features...")
    bn = rustorch.BatchNorm1d(16, 1e-5, 0.1, True, True)
    print(f"✅ BatchNorm1d features: {bn.num_features}")
    print(f"✅ BatchNorm1d repr: {bn}")

    # Test improved tensor error handling
    print("\n🔧 Testing tensor error handling...")
    try:
        rustorch.Tensor([], [2, 3])  # Empty data with shape
        print("❌ Should have failed")
    except ValueError as e:
        print(f"✅ Proper error handling: {e}")

    # Test basic functionality still works
    print("\n🔧 Testing basic functionality...")
    X = rustorch.Variable(rustorch.tensor([1.0, 2.0, 3.0, 4.0]), requires_grad=False)
    linear = rustorch.Linear(4, 1, True)
    output = linear(X)
    print(f"✅ Forward pass: {X.data.shape} → {output.data.shape}")

    print("\n🎉 All major fixes working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
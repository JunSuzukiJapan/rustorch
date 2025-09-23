#!/usr/bin/env python3
"""
Refactored Structure Test
リファクタリング済み構造テスト
"""

import sys
import traceback

def test_refactored_imports():
    """Test that all modules can be imported with the refactored structure"""
    try:
        # Test current working implementation
        import rustorch

        print("✅ Core imports successful")

        # Test all major components exist
        components = [
            # Core
            'Tensor', 'Variable',
            # Layers
            'Linear', 'Conv2d', 'MaxPool2d', 'BatchNorm1d', 'BatchNorm2d', 'Dropout', 'Flatten',
            # Activations
            'ReLU', 'Sigmoid', 'Tanh',
            # Loss functions
            'MSELoss', 'CrossEntropyLoss',
            # Optimizers
            'SGD', 'Adam',
            # Tensor functions
            'zeros', 'ones', 'tensor'
        ]

        missing = []
        for component in components:
            if not hasattr(rustorch, component):
                missing.append(component)

        if missing:
            print(f"⚠️ Missing components: {missing}")
        else:
            print("✅ All components available")

        return len(missing) == 0

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_component_creation():
    """Test that components can be created successfully"""
    try:
        import rustorch

        print("\n🧪 Testing component creation...")

        # Test tensor creation
        t = rustorch.zeros([2, 3])
        print(f"✅ Tensor creation: {t}")

        # Test variable creation
        v = rustorch.Variable(t, requires_grad=False)
        print(f"✅ Variable creation: {v}")

        # Test layer creation
        linear = rustorch.Linear(3, 5, True)
        print(f"✅ Linear layer: {linear}")

        conv = rustorch.Conv2d(1, 8, (3, 3), (1, 1), (1, 1), True)
        print(f"✅ Conv2d layer: {conv}")

        bn = rustorch.BatchNorm1d(5, 1e-5, 0.1, True, True)
        print(f"✅ BatchNorm1d layer: {bn}")

        dropout = rustorch.Dropout(0.5, False)
        print(f"✅ Dropout layer: {dropout}")

        # Test activation functions
        relu = rustorch.ReLU()
        print(f"✅ ReLU activation: {relu}")

        # Test loss functions
        mse = rustorch.MSELoss()
        ce = rustorch.CrossEntropyLoss()
        print(f"✅ Loss functions: {mse}, {ce}")

        # Test optimizers
        sgd = rustorch.SGD([v], lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=False)
        print(f"✅ SGD optimizer: {sgd}")

        return True

    except Exception as e:
        print(f"❌ Component creation error: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test improved error handling"""
    try:
        import rustorch

        print("\n🔍 Testing error handling...")

        # Test invalid tensor creation
        try:
            rustorch.Tensor([], [2, 3])  # Empty data with non-empty shape
            print("❌ Expected error for invalid tensor creation")
            return False
        except Exception as e:
            print(f"✅ Proper error for invalid tensor: {type(e).__name__}")

        # Test invalid layer parameters
        try:
            rustorch.Linear(0, 5, True)  # Invalid input size
            print("❌ Expected error for invalid Linear parameters")
            return False
        except (ValueError, Exception) as e:
            print(f"✅ Proper error for invalid Linear: {type(e).__name__}")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality still works after refactoring"""
    try:
        import rustorch

        print("\n⚡ Testing basic functionality...")

        # Create simple neural network
        X = rustorch.Variable(rustorch.tensor([1.0, 2.0, 3.0, 4.0]), requires_grad=False)
        y = rustorch.Variable(rustorch.tensor([1.0]), requires_grad=False)

        # Create model
        linear = rustorch.Linear(4, 1, True)
        relu = rustorch.ReLU()

        # Forward pass
        output = linear(X)
        activated = relu(output)

        print(f"✅ Forward pass: {X.data.shape} → {output.data.shape} → {activated.data.shape}")

        # Test loss
        mse = rustorch.MSELoss()
        loss = mse(activated, y)

        print(f"✅ Loss computation: {loss.data.shape}")

        # Test optimizer
        optimizer = rustorch.Adam([linear.weight, linear.bias], lr=0.001,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)

        print(f"✅ Optimizer creation: {optimizer}")

        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🔧 RusTorch Refactored Structure Test")
    print("=" * 50)

    tests = [
        ("Import Test", test_refactored_imports),
        ("Component Creation", test_component_creation),
        ("Error Handling", test_error_handling),
        ("Basic Functionality", test_basic_functionality),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)

        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Refactoring structure is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
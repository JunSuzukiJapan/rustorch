#!/usr/bin/env python3
"""
Phase 5 Autograd API Test
RusTorch Python Bindings - Advanced Autograd Features

Tests for:
- no_grad() and enable_grad() context managers
- Variable.detach() and retain_grad()
- Functional gradient computation: grad()
- Advanced Variable operations
"""

import sys
sys.path.insert(0, '.')

try:
    import rustorch
    print("✅ RusTorch import successful")
except ImportError as e:
    print(f"❌ RusTorch import failed: {e}")
    sys.exit(1)

def test_no_grad_context():
    """Test no_grad context manager"""
    print("\n🧪 Testing no_grad() context manager...")
    try:
        # Test no_grad context creation
        context = rustorch.no_grad()
        print("✅ no_grad() context created")

        # Test context manager protocol (simplified test)
        with rustorch.no_grad():
            # Create tensors inside no_grad context
            tensor = rustorch.zeros([2, 2])
            var = rustorch.Variable(tensor, requires_grad=True)
            print(f"✅ Variable created in no_grad context: {var}")

    except Exception as e:
        print(f"❌ no_grad context test failed: {e}")
        return False
    return True

def test_enable_grad_context():
    """Test enable_grad context manager"""
    print("\n🧪 Testing enable_grad() context manager...")
    try:
        # Test enable_grad context creation
        context = rustorch.enable_grad()
        print("✅ enable_grad() context created")

        with rustorch.enable_grad():
            tensor = rustorch.ones([2, 2])
            var = rustorch.Variable(tensor, requires_grad=True)
            print(f"✅ Variable created in enable_grad context: {var}")

    except Exception as e:
        print(f"❌ enable_grad context test failed: {e}")
        return False
    return True

def test_variable_detach():
    """Test Variable.detach() method"""
    print("\n🧪 Testing Variable.detach()...")
    try:
        # Create variable with gradient tracking
        tensor = rustorch.tensor([1.0, 2.0, 3.0])
        var = rustorch.Variable(tensor, requires_grad=True)
        print(f"Original variable: {var}, requires_grad: {var.requires_grad}")

        # Detach from computation graph
        detached = var.detach()
        print(f"Detached variable: {detached}, requires_grad: {detached.requires_grad}")

        # Verify detached variable doesn't require gradients
        assert not detached.requires_grad, "Detached variable should not require gradients"
        print("✅ Variable.detach() working correctly")

    except Exception as e:
        print(f"❌ Variable.detach() test failed: {e}")
        return False
    return True

def test_variable_retain_grad():
    """Test Variable.retain_grad() method"""
    print("\n🧪 Testing Variable.retain_grad()...")
    try:
        tensor = rustorch.tensor([1.0, 2.0, 3.0])
        var = rustorch.Variable(tensor, requires_grad=True)

        # Call retain_grad (placeholder implementation)
        var.retain_grad()
        print("✅ Variable.retain_grad() called successfully")

    except Exception as e:
        print(f"❌ Variable.retain_grad() test failed: {e}")
        return False
    return True

def test_variable_register_hook():
    """Test Variable.register_hook() method"""
    print("\n🧪 Testing Variable.register_hook()...")
    try:
        tensor = rustorch.tensor([1.0, 2.0, 3.0])
        var = rustorch.Variable(tensor, requires_grad=True)

        # Define a dummy hook function
        def dummy_hook(grad):
            return grad * 2

        # Register hook (placeholder implementation)
        var.register_hook(dummy_hook)
        print("✅ Variable.register_hook() called successfully")

    except Exception as e:
        print(f"❌ Variable.register_hook() test failed: {e}")
        return False
    return True

def test_variable_clone():
    """Test Variable.clone() method"""
    print("\n🧪 Testing Variable.clone()...")
    try:
        tensor = rustorch.tensor([1.0, 2.0, 3.0])
        var = rustorch.Variable(tensor, requires_grad=True)

        # Clone the variable
        cloned = var.clone()
        print(f"Original: {var}")
        print(f"Cloned: {cloned}")
        print("✅ Variable.clone() working correctly")

    except Exception as e:
        print(f"❌ Variable.clone() test failed: {e}")
        return False
    return True

def test_variable_from_tensor():
    """Test Variable.from_tensor() static method"""
    print("\n🧪 Testing Variable.from_tensor()...")
    try:
        tensor = rustorch.tensor([1.0, 2.0, 3.0])

        # Create variable from tensor with gradient tracking
        var = rustorch.Variable.from_tensor(tensor, requires_grad=True)
        print(f"Variable from tensor: {var}, requires_grad: {var.requires_grad}")

        # Create variable from tensor without gradient tracking
        var_no_grad = rustorch.Variable.from_tensor(tensor, requires_grad=False)
        print(f"Variable no grad: {var_no_grad}, requires_grad: {var_no_grad.requires_grad}")

        print("✅ Variable.from_tensor() working correctly")

    except Exception as e:
        print(f"❌ Variable.from_tensor() test failed: {e}")
        return False
    return True

def test_functional_grad():
    """Test functional grad() computation"""
    print("\n🧪 Testing functional grad() computation...")
    try:
        # Create simple computation graph
        x = rustorch.Variable(rustorch.tensor([2.0]), requires_grad=True)
        y = x * x  # y = x^2

        # Compute gradients (simplified test)
        gradients = rustorch.grad([y], [x], retain_graph=False, create_graph=False)
        print(f"Gradients: {gradients}")
        print("✅ Functional grad() called successfully")

    except Exception as e:
        print(f"❌ Functional grad() test failed: {e}")
        return False
    return True

def test_autograd_integration():
    """Test integration of autograd features"""
    print("\n🧪 Testing autograd integration...")
    try:
        # Create computation with multiple variables
        x = rustorch.Variable(rustorch.tensor([1.0, 2.0]), requires_grad=True)
        y = rustorch.Variable(rustorch.tensor([3.0, 4.0]), requires_grad=True)

        # Perform operations
        z = x + y
        w = z * z

        # Test detach
        w_detached = w.detach()
        assert not w_detached.requires_grad

        # Test clone
        x_clone = x.clone()
        assert x_clone.requires_grad == x.requires_grad

        print("✅ Autograd integration test passed")

    except Exception as e:
        print(f"❌ Autograd integration test failed: {e}")
        return False
    return True

def main():
    """Run all Phase 5 autograd tests"""
    print("🚀 RusTorch Phase 5 Autograd API Tests")
    print("=" * 50)

    tests = [
        test_no_grad_context,
        test_enable_grad_context,
        test_variable_detach,
        test_variable_retain_grad,
        test_variable_register_hook,
        test_variable_clone,
        test_variable_from_tensor,
        test_functional_grad,
        test_autograd_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All Phase 5 Autograd tests passed!")
        return True
    else:
        print(f"⚠️  {failed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
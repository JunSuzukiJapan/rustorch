#!/usr/bin/env python3
"""
Basic test for RusTorch Python bindings
"""

import sys
import os

# Add current directory to path to import the built module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_import():
    """Test if we can import the basic rustorch module"""
    try:
        # Try to import from the built extension
        import rustorch._rustorch_py as rt_native
        print("✓ Successfully imported rustorch._rustorch_py")
        return True
    except ImportError as e:
        print(f"✗ Failed to import rustorch._rustorch_py: {e}")
        return False

def test_tensor_creation():
    """Test basic tensor creation"""
    try:
        import rustorch._rustorch_py as rt
        
        # Test creating a tensor
        tensor = rt.tensor([1.0, 2.0, 3.0])
        print(f"✓ Created tensor: shape={tensor.shape()}")
        
        # Test zeros
        zeros = rt.zeros([2, 3])
        print(f"✓ Created zeros tensor: shape={zeros.shape()}")
        
        # Test ones  
        ones = rt.ones([2, 3])
        print(f"✓ Created ones tensor: shape={ones.shape()}")
        
        return True
    except Exception as e:
        print(f"✗ Tensor creation failed: {e}")
        return False

def test_variable_creation():
    """Test Variable creation and autograd"""
    try:
        import rustorch._rustorch_py as rt
        
        # Create a tensor
        tensor = rt.tensor([1.0, 2.0, 3.0])
        
        # Create a Variable
        var = rt.Variable(tensor, requires_grad=True)
        print(f"✓ Created Variable: requires_grad={var.requires_grad()}")
        
        return True
    except Exception as e:
        print(f"✗ Variable creation failed: {e}")
        return False

def test_linear_layer():
    """Test Linear layer"""
    try:
        import rustorch._rustorch_py as rt
        
        # Create Linear layer
        linear = rt.Linear(3, 2)
        print("✓ Created Linear layer")
        
        # Test forward pass
        input_tensor = rt.tensor([[1.0, 2.0, 3.0]])
        input_var = rt.Variable(input_tensor, requires_grad=False)
        
        output = linear.forward(input_var)
        print(f"✓ Linear forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Linear layer test failed: {e}")
        return False

def test_optimizer():
    """Test optimizer"""
    try:
        import rustorch._rustorch_py as rt
        
        # Create a linear layer
        linear = rt.Linear(3, 2)
        
        # Get parameters 
        params = linear.parameters()
        print(f"✓ Retrieved {len(params)} parameters")
        
        # Create optimizer
        optimizer = rt.SGD(params, lr=0.01)
        print("✓ Created SGD optimizer")
        
        return True
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing RusTorch Python bindings...")
    print("=" * 40)
    
    tests = [
        test_basic_import,
        test_tensor_creation,
        test_variable_creation,
        test_linear_layer,
        test_optimizer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
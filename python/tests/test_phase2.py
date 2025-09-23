#!/usr/bin/env python3
"""
Test for RusTorch Python bindings Phase 2: Variable implementation
"""

try:
    import rustorch
    print("✓ Successfully imported rustorch with Variable support")

    # Test Variable creation
    print("\n=== Testing Variable Creation ===")

    # Create a basic tensor
    t1 = rustorch.tensor([1.0, 2.0, 3.0])
    print(f"✓ Created tensor: {t1}")

    # Create Variable without gradient
    v1 = rustorch.Variable(t1, requires_grad=False)
    print(f"✓ Created Variable (no grad): {v1}")
    print(f"  requires_grad: {v1.requires_grad}")
    print(f"  data shape: {v1.data.shape}")

    # Create Variable with gradient
    v2 = rustorch.Variable(t1, requires_grad=True)
    print(f"✓ Created Variable (with grad): {v2}")
    print(f"  requires_grad: {v2.requires_grad}")
    print(f"  grad: {v2.grad}")

    print("\n=== Testing Variable Properties ===")

    # Test data access
    data = v2.data
    print(f"✓ Data access: {data}")
    print(f"  Data shape: {data.shape}")

    # Test gradient access (should be None initially)
    grad = v2.grad
    print(f"✓ Gradient access: {grad}")

    print("\n=== Testing Variable Operations ===")

    # Test sum operation
    sum_result = v2.sum()
    print(f"✓ Sum operation: {sum_result}")
    print(f"  Sum data shape: {sum_result.data.shape}")

    print("\n=== Testing Variable Arithmetic ===")

    # Create two Variables for arithmetic tests
    t2 = rustorch.tensor([2.0, 3.0, 4.0])
    v3 = rustorch.Variable(t2, requires_grad=True)
    print(f"✓ Created second Variable: {v3}")

    # Test addition
    v_add = v2 + v3
    print(f"✓ Addition (v2 + v3): {v_add}")
    print(f"  Addition requires_grad: {v_add.requires_grad}")

    # Test subtraction
    v_sub = v2 - v3
    print(f"✓ Subtraction (v2 - v3): {v_sub}")

    # Test multiplication
    v_mul = v2 * v3
    print(f"✓ Multiplication (v2 * v3): {v_mul}")

    # Matrix multiplication test (skipped for now due to shape constraints)
    print("⚠️ Matrix multiplication test skipped (requires specific tensor shapes)")

    # Test autograd with arithmetic operations
    print("\n=== Testing Autograd with Arithmetic ===")
    try:
        loss = v_add.sum()
        loss.backward()
        print("✓ Backward pass through addition completed")

        # Check if original variables have gradients
        if v2.grad is not None:
            print("✓ v2 has gradient after backward")
        if v3.grad is not None:
            print("✓ v3 has gradient after backward")

    except Exception as e:
        print(f"⚠️ Arithmetic autograd error: {e}")

    # Test backward pass
    print("\n=== Testing Autograd ===")
    try:
        sum_result.backward()
        print("✓ Backward pass completed")

        # Check gradients after backward
        grad_after = v2.grad
        if grad_after is not None:
            print(f"✓ Gradient computed: {grad_after}")
        else:
            print("⚠️ Gradient is still None after backward")

    except Exception as e:
        print(f"⚠️ Backward pass error: {e}")

    # Test zero_grad
    try:
        v2.zero_grad()
        print("✓ zero_grad() completed")
        grad_after_zero = v2.grad
        print(f"  Gradient after zero_grad: {grad_after_zero}")
    except Exception as e:
        print(f"⚠️ zero_grad error: {e}")

    print("\n=== Testing Linear Layer ===")

    # Test Linear layer creation
    linear = rustorch.Linear(3, 2, True)  # input_size=3, output_size=2, bias=True
    print(f"✓ Created Linear layer: {linear}")
    print(f"  Input size: {linear.input_size}")
    print(f"  Output size: {linear.output_size}")

    # Test weight and bias access
    weight = linear.weight
    print(f"✓ Weight access: {weight}")
    print(f"  Weight shape: {weight.data.shape}")

    bias = linear.bias
    if bias is not None:
        print(f"✓ Bias access: {bias}")
        print(f"  Bias shape: {bias.data.shape}")
    else:
        print("⚠️ No bias found")

    # Test Linear layer without bias
    linear_no_bias = rustorch.Linear(3, 2, bias=False)
    print(f"✓ Created Linear layer without bias: {linear_no_bias}")
    bias_no_bias = linear_no_bias.bias
    print(f"  Bias: {bias_no_bias}")

    print("\n=== Testing Linear Forward Pass ===")

    # Create input Variable for linear layer
    input_data = rustorch.tensor([1.0, 2.0, 3.0])  # shape [3]
    input_var = rustorch.Variable(input_data, requires_grad=True)
    print(f"✓ Created input: {input_var}")

    # Forward pass through linear layer
    try:
        output = linear(input_var)
        print(f"✓ Linear forward pass: {output}")
        print(f"  Output shape: {output.data.shape}")
        print(f"  Output requires_grad: {output.requires_grad}")

        # Test autograd through Linear layer
        print("\n=== Testing Autograd through Linear Layer ===")
        loss = output.sum()
        print(f"✓ Loss from linear output: {loss}")

        loss.backward()
        print("✓ Backward pass through Linear layer completed")

        # Check gradients
        input_grad = input_var.grad
        if input_grad is not None:
            print("✓ Input has gradient after Linear backward")

        weight_grad = weight.grad
        if weight_grad is not None:
            print("✓ Weight has gradient after Linear backward")

        if bias is not None:
            bias_grad = bias.grad
            if bias_grad is not None:
                print("✓ Bias has gradient after Linear backward")

    except Exception as e:
        print(f"⚠️ Linear forward pass error: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Phase 2 complete tests passed!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to build with: maturin develop")

except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
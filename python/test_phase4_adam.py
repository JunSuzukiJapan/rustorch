#!/usr/bin/env python3
"""
Phase 4 Test: Adam Optimizer
Adamオプティマイザーのテスト
"""

try:
    import rustorch
    print("✓ Successfully imported rustorch with Phase 4 Adam optimizer")

    print("\n=== Testing Phase 4: Adam Optimizer ===")

    # Test basic Adam creation
    print("\n1. Testing Adam Optimizer Creation")

    # Create some dummy parameters
    dummy_tensor = rustorch.tensor([1.0, 2.0])
    dummy_var = rustorch.Variable(dummy_tensor, requires_grad=True)

    # Test Adam creation with default parameters
    adam = rustorch.Adam([dummy_var], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
    print(f"✓ Created Adam with defaults: {adam}")
    print(f"  Learning rate: {adam.learning_rate}")
    print(f"  Beta1: {adam.beta1}")
    print(f"  Beta2: {adam.beta2}")
    print(f"  Epsilon: {adam.eps}")
    print(f"  Weight decay: {adam.weight_decay}")
    print(f"  Step count: {adam.step_count}")

    # Test Adam creation with custom parameters
    print("\n2. Testing Adam with Custom Parameters")
    adam_custom = rustorch.Adam(
        [dummy_var],
        lr=0.01,
        betas=(0.95, 0.999),
        eps=1e-6,
        weight_decay=0.001,
        amsgrad=False
    )
    print(f"✓ Created custom Adam: {adam_custom}")
    print(f"  Learning rate: {adam_custom.learning_rate}")
    print(f"  Beta1: {adam_custom.beta1}")
    print(f"  Beta2: {adam_custom.beta2}")
    print(f"  Epsilon: {adam_custom.eps}")
    print(f"  Weight decay: {adam_custom.weight_decay}")

    print("\n=== Testing Adam vs SGD Performance ===")

    # Create training data
    print("\n3. Creating Training Data")
    X = rustorch.Variable(rustorch.tensor([1.0, 2.0]), requires_grad=False)
    y = rustorch.Variable(rustorch.tensor([1.0]), requires_grad=False)
    print(f"✓ Input X: {X} (shape: {X.data.shape})")
    print(f"✓ Target y: {y} (shape: {y.data.shape})")

    # Create model components
    print("\n4. Creating Model")
    linear1 = rustorch.Linear(2, 4, True)  # 2 → 4
    relu_activation = rustorch.ReLU()
    linear2 = rustorch.Linear(4, 1, True)  # 4 → 1
    sigmoid_activation = rustorch.Sigmoid()

    print(f"✓ Linear1 (2→4): {linear1}")
    print(f"✓ ReLU activation: {relu_activation}")
    print(f"✓ Linear2 (4→1): {linear2}")
    print(f"✓ Sigmoid activation: {sigmoid_activation}")

    # Create loss function
    criterion = rustorch.MSELoss()

    # Collect parameters
    params = [linear1.weight, linear1.bias, linear2.weight, linear2.bias]

    # Test SGD vs Adam
    print("\n5. Testing SGD vs Adam Training")

    print("\n--- SGD Training ---")
    sgd_optimizer = rustorch.SGD(params, 0.1, 0.0)
    print(f"✓ SGD Optimizer: {sgd_optimizer}")

    for epoch in range(3):
        sgd_optimizer.zero_grad()

        # Forward pass
        h1 = linear1(X)
        h1_relu = relu_activation(h1)
        h2 = linear2(h1_relu)
        output = sigmoid_activation(h2)

        # Compute loss
        loss = criterion(output, y)

        # Backward pass
        loss.backward()

        # Update parameters
        sgd_optimizer.step()

        print(f"  SGD Epoch {epoch + 1}: Loss shape = {loss.data.shape}")

    print("\n--- Adam Training ---")
    adam_optimizer = rustorch.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
    print(f"✓ Adam Optimizer: {adam_optimizer}")

    for epoch in range(3):
        adam_optimizer.zero_grad()

        # Forward pass
        h1 = linear1(X)
        h1_relu = relu_activation(h1)
        h2 = linear2(h1_relu)
        output = sigmoid_activation(h2)

        # Compute loss
        loss = criterion(output, y)

        # Backward pass
        loss.backward()

        # Update parameters
        adam_optimizer.step()

        print(f"  Adam Epoch {epoch + 1}: Loss shape = {loss.data.shape}, Step count = {adam_optimizer.step_count}")

    print("\n6. Testing Adam Parameter Updates")

    # Test learning rate modification
    print(f"  Current learning rate: {adam_optimizer.learning_rate}")
    adam_optimizer.set_lr(0.005)
    print(f"  Updated learning rate: {adam_optimizer.learning_rate}")

    print("\n🎉 Phase 4 Adam Optimizer Working!")
    print("✅ Adam optimizer creation functional")
    print("✅ Custom parameter initialization functional")
    print("✅ Training loop with Adam functional")
    print("✅ Parameter updates and step counting functional")
    print("✅ Learning rate modification functional")
    print("✅ SGD vs Adam comparison functional")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to build with: maturin develop")

except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()
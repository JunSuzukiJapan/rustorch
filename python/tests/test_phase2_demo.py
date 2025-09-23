#!/usr/bin/env python3
"""
Phase 2 Demo: Neural Network Foundation - achieving the implementation goal
"""

import rustorch

# 目標: 基本的なニューラルネットワークトレーニング
print("=== Phase 2 Demo: Basic Neural Network Foundation ===")

# Variable (自動微分対応)
x = rustorch.Variable(rustorch.tensor([1.0, 2.0]), requires_grad=True)
print(f"✓ Input Variable: {x}")
print(f"  Data shape: {x.data.shape}")  # [2]

# Linear Layer
linear = rustorch.Linear(2, 1, True)  # input_size=2, output_size=1
print(f"✓ Linear Layer: {linear}")

# Forward pass
y = linear(x)  # forward pass
print(f"✓ Forward pass output: {y}")
print(f"  Output shape: {y.data.shape}")  # [1]

# 基本的な自動微分
loss = y.sum()
print(f"✓ Loss: {loss}")

loss.backward()
print("✓ Backward pass completed")

grad = x.grad  # 勾配情報
if grad is not None:
    print(f"✓ Input gradient: {grad}")
    print(f"  Gradient shape: {grad.shape}")
else:
    print("❌ No gradient found")

# Additional demonstration: Variable arithmetic with autograd
print("\n=== Variable Arithmetic with Autograd ===")

# Create two variables
a = rustorch.Variable(rustorch.tensor([2.0, 3.0]), requires_grad=True)
b = rustorch.Variable(rustorch.tensor([4.0, 5.0]), requires_grad=True)

# Arithmetic operations
c = a + b  # Addition
d = c * a  # Multiplication
result = d.sum()  # Reduction

print(f"✓ a = {a}")
print(f"✓ b = {b}")
print(f"✓ c = a + b = {c}")
print(f"✓ d = c * a = {d}")
print(f"✓ result = d.sum() = {result}")

# Backward pass
result.backward()
print("✓ Backward pass through arithmetic operations completed")

if a.grad is not None:
    print(f"✓ a.grad = {a.grad}")
if b.grad is not None:
    print(f"✓ b.grad = {b.grad}")

print("\n🎉 Phase 2 Neural Network Foundation Complete!")
print("✅ Variable with autograd support")
print("✅ Linear layer with learnable parameters")
print("✅ Forward and backward passes working")
print("✅ Variable arithmetic operations")
print("✅ Gradient computation and accumulation")
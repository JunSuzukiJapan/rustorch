# RusTorch Quick Start Guide

## Installation

### 1. Prerequisites
```bash
# Rust 1.70 or later
rustc --version

# Python 3.8 or later
python --version

# Install required dependencies
pip install maturin numpy matplotlib
```

### 2. Build and Install RusTorch
```bash
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Create Python virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Build and install in development mode
maturin develop --release
```

## Basic Usage Examples

### 1. Tensor Creation and Basic Operations

```python
import rustorch
import numpy as np

# Tensor creation
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Tensor x:\n{x}")
print(f"Shape: {x.shape()}")  # [2, 2]

# Zero matrices and identity matrices
zeros = rustorch.zeros([3, 3])
ones = rustorch.ones([2, 2])
identity = rustorch.eye(3)

print(f"Zeros:\n{zeros}")
print(f"Ones:\n{ones}")
print(f"Identity:\n{identity}")

# Random tensors
random_normal = rustorch.randn([2, 3])
random_uniform = rustorch.rand([2, 3])

print(f"Random normal:\n{random_normal}")
print(f"Random uniform:\n{random_uniform}")

# NumPy integration
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
tensor_from_numpy = rustorch.from_numpy(np_array)
print(f"From NumPy:\n{tensor_from_numpy}")

# Convert back to NumPy
back_to_numpy = tensor_from_numpy.to_numpy()
print(f"Back to NumPy:\n{back_to_numpy}")
```

### 2. Arithmetic Operations

```python
# Basic arithmetic operations
a = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = rustorch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Element-wise operations
add_result = a.add(b)  # a + b
sub_result = a.sub(b)  # a - b
mul_result = a.mul(b)  # a * b (element-wise)
div_result = a.div(b)  # a / b (element-wise)

print(f"Addition:\n{add_result}")
print(f"Subtraction:\n{sub_result}")
print(f"Multiplication:\n{mul_result}")
print(f"Division:\n{div_result}")

# Scalar operations
scalar_add = a.add(2.0)
scalar_mul = a.mul(3.0)

print(f"Scalar addition (+2):\n{scalar_add}")
print(f"Scalar multiplication (*3):\n{scalar_mul}")

# Matrix multiplication
matmul_result = a.matmul(b)
print(f"Matrix multiplication:\n{matmul_result}")

# Mathematical functions
sqrt_result = a.sqrt()
exp_result = a.exp()
log_result = a.log()

print(f"Square root:\n{sqrt_result}")
print(f"Exponential:\n{exp_result}")
print(f"Natural log:\n{log_result}")
```

### 3. Tensor Shape Manipulation

```python
# Shape manipulation examples
original = rustorch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(f"Original shape: {original.shape()}")  # [2, 4]

# Reshape
reshaped = original.reshape([4, 2])
print(f"Reshaped [4, 2]:\n{reshaped}")

# Transpose
transposed = original.transpose(0, 1)
print(f"Transposed:\n{transposed}")

# Dimension addition/removal
squeezed = rustorch.tensor([[[1], [2], [3]]])
print(f"Before squeeze: {squeezed.shape()}")  # [1, 3, 1]

unsqueezed = squeezed.squeeze()
print(f"After squeeze: {unsqueezed.shape()}")  # [3]

expanded = unsqueezed.unsqueeze(0)
print(f"After unsqueeze: {expanded.shape()}")  # [1, 3]
```

### 4. Statistical Operations

```python
# Statistical functions
data = rustorch.randn([3, 4])
print(f"Data:\n{data}")

# Basic statistics
mean_val = data.mean()
sum_val = data.sum()
std_val = data.std()
var_val = data.var()
max_val = data.max()
min_val = data.min()

print(f"Mean: {mean_val.item():.4f}")
print(f"Sum: {sum_val.item():.4f}")
print(f"Std: {std_val.item():.4f}")
print(f"Var: {var_val.item():.4f}")
print(f"Max: {max_val.item():.4f}")
print(f"Min: {min_val.item():.4f}")

# Statistics along specific dimensions
row_mean = data.mean(dim=1)  # Mean of each row
col_sum = data.sum(dim=0)    # Sum of each column

print(f"Row means: {row_mean}")
print(f"Column sums: {col_sum}")
```

## Automatic Differentiation Basics

### 1. Gradient Computation

```python
# Automatic differentiation example
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
print(f"Input tensor: {x}")

# Create Variable
var_x = rustorch.autograd.Variable(x)

# Build computation graph
y = var_x.pow(2).sum()  # y = sum(x^2)
print(f"Output: {y.data().item()}")

# Backward propagation
y.backward()

# Get gradient
grad = var_x.grad()
print(f"Gradient: {grad}")  # dy/dx = 2x = [2, 4]
```

### 2. Complex Computation Graphs

```python
# More complex example
x = rustorch.tensor([[2.0, 3.0]], requires_grad=True)
var_x = rustorch.autograd.Variable(x)

# Complex function: z = sum((x^2 + 3x) * exp(x))
y = var_x.pow(2).add(var_x.mul(3))  # x^2 + 3x
z = y.mul(var_x.exp()).sum()        # (x^2 + 3x) * exp(x), then sum

print(f"Result: {z.data().item():.4f}")

# Backward propagation
z.backward()
grad = var_x.grad()
print(f"Gradient: {grad}")
```

## Neural Network Basics

### 1. Simple Linear Layer

```python
# Create linear layer
linear_layer = rustorch.nn.Linear(3, 1)  # 3 inputs -> 1 output

# Random input
input_data = rustorch.randn([2, 3])  # Batch size 2, 3 features
print(f"Input: {input_data}")

# Forward pass
output = linear_layer.forward(input_data)
print(f"Output: {output}")

# Check parameters
weight = linear_layer.weight()
bias = linear_layer.bias()
print(f"Weight shape: {weight.shape()}")
print(f"Weight: {weight}")
if bias is not None:
    print(f"Bias: {bias}")
```

### 2. Activation Functions

```python
# Various activation functions
x = rustorch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])

# ReLU
relu = rustorch.nn.ReLU()
relu_output = relu.forward(x)
print(f"ReLU: {relu_output}")

# Sigmoid
sigmoid = rustorch.nn.Sigmoid()
sigmoid_output = sigmoid.forward(x)
print(f"Sigmoid: {sigmoid_output}")

# Tanh
tanh = rustorch.nn.Tanh()
tanh_output = tanh.forward(x)
print(f"Tanh: {tanh_output}")
```

### 3. Loss Functions

```python
# Loss function usage examples
predictions = rustorch.tensor([[2.0, 1.0], [0.5, 1.5]])
targets = rustorch.tensor([[1.8, 0.9], [0.6, 1.4]])

# Mean squared error
mse_loss = rustorch.nn.MSELoss()
loss_value = mse_loss.forward(predictions, targets)
print(f"MSE Loss: {loss_value.item():.6f}")

# Cross-entropy (for classification)
logits = rustorch.tensor([[1.0, 2.0, 0.5], [0.2, 0.8, 2.1]])
labels = rustorch.tensor([1, 2], dtype="int64")  # Class indices

ce_loss = rustorch.nn.CrossEntropyLoss()
ce_loss_value = ce_loss.forward(logits, labels)
print(f"Cross Entropy Loss: {ce_loss_value.item():.6f}")
```

## Data Processing

### 1. Datasets and DataLoaders

```python
# Create dataset
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 4).astype(np.float32)  # 100 samples, 4 features
y = np.random.randint(0, 3, (100,)).astype(np.int64)  # 3-class classification

# Convert to tensors
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y.reshape(-1, 1).astype(np.float32))

# Create dataset
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
print(f"Dataset size: {len(dataset)}")

# Create dataloader
dataloader = rustorch.data.DataLoader(
    dataset, 
    batch_size=10, 
    shuffle=True
)

# Get batches from dataloader
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 3:  # Show only first 3 batches
        break
    
    if len(batch) >= 2:
        inputs, targets = batch[0], batch[1]
        print(f"Batch {batch_idx}: Input shape {inputs.shape()}, Target shape {targets.shape()}")
```

### 2. Data Transforms

```python
# Data transformation examples
data = rustorch.randn([10, 10])
print(f"Original data mean: {data.mean().item():.4f}")
print(f"Original data std: {data.std().item():.4f}")

# Normalization transform
normalize_transform = rustorch.data.transforms.normalize(mean=0.0, std=1.0)
normalized_data = normalize_transform(data)
print(f"Normalized data mean: {normalized_data.mean().item():.4f}")
print(f"Normalized data std: {normalized_data.std().item():.4f}")
```

## Complete Training Example

### Linear Regression

```python
# Complete linear regression example
import numpy as np

# Generate data
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# Convert to tensors
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Create dataset and dataloader
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Define model
model = rustorch.nn.Linear(1, 1)  # 1 input -> 1 output

# Loss function and optimizer
criterion = rustorch.nn.MSELoss()
optimizer = rustorch.optim.SGD([model.weight(), model.bias()], lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0.0
    batch_count = 0
    
    dataloader.reset()
    while True:
        batch = dataloader.next_batch()
        if batch is None:
            break
        
        if len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model.forward(inputs)
            loss = criterion.forward(predictions, targets)
            
            # Backpropagation (simplified)
            epoch_loss += loss.item()
            batch_count += 1
    
    if batch_count > 0:
        avg_loss = epoch_loss / batch_count
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

print("Training completed!")

# Final parameters
final_weight = model.weight()
final_bias = model.bias()
print(f"Learned weight: {final_weight.item():.4f} (true: 2.0)")
if final_bias is not None:
    print(f"Learned bias: {final_bias.item():.4f} (true: 1.0)")
```

## Troubleshooting

### Common Issues and Solutions

1. **Installation Problems**
```bash
# If maturin is not found
pip install --upgrade maturin

# If Rust is outdated
rustup update

# Python environment issues
python -m pip install --upgrade pip
```

2. **Runtime Errors**
```python
# Check tensor shapes
print(f"Tensor shape: {tensor.shape()}")
print(f"Tensor dtype: {tensor.dtype()}")

# Be careful with data types in NumPy conversion
np_array = np.array(data, dtype=np.float32)  # Explicit float32
```

3. **Performance Optimization**
```python
# Build in release mode
# maturin develop --release

# Adjust batch size
dataloader = rustorch.data.DataLoader(dataset, batch_size=64)  # Larger batch
```

## Next Steps

1. **Try Advanced Examples**: Check examples in `docs/examples/neural_networks/`
2. **Use Keras-style API**: `rustorch.training.Model` for easier model building
3. **Visualization Features**: `rustorch.visualization` for training progress visualization
4. **Distributed Training**: `rustorch.distributed` for parallel processing

Detailed Documentation:
- [Python API Reference](../en/python_api_reference.md)
- [Overview Documentation](../en/python_bindings_overview.md)
- [Example Collection](../examples/)
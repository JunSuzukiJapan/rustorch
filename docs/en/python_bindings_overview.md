# RusTorch Python Bindings Overview

## Overview

RusTorch is a high-performance deep learning framework implemented in Rust, providing PyTorch-like APIs while leveraging Rust's safety and performance benefits. Through Python bindings, you can access RusTorch functionality directly from Python.

## Key Features

### 🚀 **High Performance**
- **Rust Core**: Achieves C++-level performance while guaranteeing memory safety
- **SIMD Support**: Automatic vectorization for optimized numerical computations
- **Parallel Processing**: Efficient parallel computation using rayon
- **Zero-Copy**: Minimal data copying between NumPy and RusTorch

### 🛡️ **Safety**
- **Memory Safety**: Prevents memory leaks and data races through Rust's ownership system
- **Type Safety**: Compile-time type checking reduces runtime errors
- **Error Handling**: Comprehensive error handling with automatic conversion to Python exceptions

### 🎯 **Ease of Use**
- **PyTorch-Compatible API**: Easy migration from existing PyTorch code
- **Keras-Style High-Level API**: Intuitive interfaces like model.fit()
- **NumPy Integration**: Bidirectional conversion with NumPy arrays

## Architecture

RusTorch's Python bindings consist of 10 modules:

### 1. **tensor** - Tensor Operations
```python
import rustorch

# Tensor creation
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = rustorch.zeros((3, 3))
z = rustorch.randn((2, 2))

# NumPy integration
import numpy as np
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
torch_tensor = rustorch.from_numpy(np_array)
```

### 2. **autograd** - Automatic Differentiation
```python
# Gradient computation
x = rustorch.tensor([[1.0, 2.0]], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(x.grad)  # Get gradients
```

### 3. **nn** - Neural Networks
```python
# Layer creation
linear = rustorch.nn.Linear(10, 1)
conv2d = rustorch.nn.Conv2d(3, 64, kernel_size=3)
relu = rustorch.nn.ReLU()

# Loss functions
mse_loss = rustorch.nn.MSELoss()
cross_entropy = rustorch.nn.CrossEntropyLoss()
```

### 4. **optim** - Optimizers
```python
# Optimizers
optimizer = rustorch.optim.Adam(model.parameters(), lr=0.001)
sgd = rustorch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Learning rate schedulers
scheduler = rustorch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
```

### 5. **data** - Data Loading
```python
# Dataset creation
dataset = rustorch.data.TensorDataset(data, targets)
dataloader = rustorch.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Data transforms
transform = rustorch.data.transforms.Normalize(mean=0.5, std=0.2)
```

### 6. **training** - High-Level Training API
```python
# Keras-style API
model = rustorch.Model()
model.add("Dense(64, activation=relu)")
model.add("Dense(10, activation=softmax)")
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Training execution
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

### 7. **distributed** - Distributed Training
```python
# Distributed training setup
config = rustorch.distributed.DistributedConfig(
    backend="nccl", world_size=4, rank=0
)

# Data parallel
model = rustorch.distributed.DistributedDataParallel(model)
```

### 8. **visualization** - Visualization
```python
# Plot training history
plotter = rustorch.visualization.Plotter()
plotter.plot_training_history(history, save_path="training.png")

# Tensor visualization
plotter.plot_tensor_as_image(tensor, title="Feature Map")
```

### 9. **utils** - Utilities
```python
# Model save/load
rustorch.utils.save_model(model, "model.rustorch")
loaded_model = rustorch.utils.load_model("model.rustorch")

# Profiling
profiler = rustorch.utils.Profiler()
with profiler.profile():
    output = model(input_data)
```

## Installation

### Prerequisites
- Python 3.8+
- Rust 1.70+
- CUDA 11.8+ (for GPU usage)

### Build and Install
```bash
# Clone repository
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install maturin numpy

# Build and install
maturin develop --release

# Or install from PyPI (planned for future)
# pip install rustorch
```

## Quick Start

### 1. Basic Tensor Operations
```python
import rustorch
import numpy as np

# Tensor creation
x = rustorch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"Shape: {x.shape()}")  # Shape: [2, 2]

# Mathematical operations
y = x + 2.0
z = x.matmul(y.transpose(0, 1))
print(f"Result: {z.to_numpy()}")
```

### 2. Linear Regression Example
```python
import rustorch
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(100, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

# Convert to tensors
X_tensor = rustorch.from_numpy(X)
y_tensor = rustorch.from_numpy(y)

# Define model
model = rustorch.Model()
model.add("Dense(1)")
model.compile(optimizer="sgd", loss="mse")

# Create dataset
dataset = rustorch.data.TensorDataset(X_tensor, y_tensor)
dataloader = rustorch.data.DataLoader(dataset, batch_size=10)

# Train
history = model.fit(dataloader, epochs=100, verbose=True)

# Display results
print(f"Final loss: {history.train_loss()[-1]:.4f}")
```

### 3. Neural Network Classification
```python
import rustorch

# Prepare data
train_dataset = rustorch.data.TensorDataset(train_X, train_y)
train_loader = rustorch.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)

# Build model
model = rustorch.Model("ClassificationNet")
model.add("Dense(128, activation=relu)")
model.add("Dropout(0.3)")
model.add("Dense(64, activation=relu)")  
model.add("Dense(10, activation=softmax)")

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Training configuration
config = rustorch.training.TrainerConfig(
    epochs=50,
    learning_rate=0.001,
    validation_frequency=5
)
trainer = rustorch.training.Trainer(config)

# Train
history = trainer.train(model, train_loader, val_loader)

# Evaluate
metrics = model.evaluate(test_loader)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
```

## Performance Optimization

### SIMD Utilization
```python
# Enable SIMD optimization during build
# Cargo.toml: target-features = "+avx2,+fma"

x = rustorch.randn((1000, 1000))
y = x.sqrt()  # SIMD-optimized computation
```

### GPU Usage
```python
# CUDA usage (planned for future)
device = rustorch.cuda.device(0)
x = rustorch.randn((1000, 1000)).to(device)
y = x.matmul(x.transpose(0, 1))  # GPU computation
```

### Parallel Data Loading
```python
dataloader = rustorch.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4  # Number of parallel workers
)
```

## Best Practices

### 1. Memory Efficiency
```python
# Utilize zero-copy conversion
np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = rustorch.from_numpy(np_array)  # No copying

# Use in-place operations
tensor.add_(1.0)  # Memory efficient
```

### 2. Error Handling
```python
try:
    result = model(invalid_input)
except rustorch.RusTorchError as e:
    print(f"RusTorch error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 3. Debugging and Profiling
```python
# Use profiler
profiler = rustorch.utils.Profiler()
profiler.start()

# Execute computation
output = model(input_data)

profiler.stop()
print(profiler.summary())
```

## Limitations

### Current Limitations
- **GPU Support**: CUDA/ROCm support under development
- **Dynamic Graphs**: Currently supports static graphs only
- **Distributed Training**: Basic functionality implemented only

### Future Extensions
- GPU acceleration (CUDA, Metal, ROCm)
- Dynamic computation graph support
- More neural network layers
- Model quantization and pruning
- ONNX export functionality

## Contributing

### Development Participation
```bash
# Setup development environment
git clone https://github.com/JunSuzukiJapan/RusTorch
cd RusTorch/rustorch
pip install -e .[dev]

# Run tests
cargo test
python -m pytest tests/

# Code quality checks
cargo clippy
cargo fmt
```

### Community
- GitHub Issues: Bug reports and feature requests
- Discussions: Questions and discussions
- Discord: Real-time support

## License

RusTorch is released under the MIT License. Free to use for both commercial and non-commercial purposes.

## Related Links

- [GitHub Repository](https://github.com/JunSuzukiJapan/RusTorch)
- [API Documentation](./api_documentation.md)
- [Examples and Tutorials](../examples/)
- [Performance Benchmarks](./benchmarks.md)
# RusTorch API Documentation

## 📚 Complete API Reference

This document provides comprehensive API documentation for RusTorch, organized by module and functionality.

## ⚠️ v0.5.0 Breaking Changes & Migration / v0.5.0 破壊的変更と移行

### Method Consolidation Impact / メソッド統合の影響

**Important**: v0.5.0 introduces breaking changes due to method consolidation. All `_v2` methods have been unified with their base versions.
**重要**: v0.5.0はメソッド統合により破壊的変更を導入します。すべての`_v2`メソッドは基本版と統合されました。

#### Affected APIs / 影響を受けるAPI
- All tensor operations with `_v2` suffixes → unified to base method names
- すべての`_v2`接尾辞付きテンソル演算 → 基本メソッド名に統合

#### Migration Examples / 移行例
```rust
// Before v0.5.0 / v0.5.0以前
tensor.add_v2(&other);     // ❌ Removed
tensor.mul_v2(&other);     // ❌ Removed
tensor.matmul_v2(&other);  // ❌ Removed

// v0.5.0+ / v0.5.0以降  
tensor.add(&other);        // ✅ Unified optimized version
tensor.mul(&other);        // ✅ Unified optimized version
tensor.matmul(&other);     // ✅ Unified optimized version
```

#### Benefits / 利点
- **Simplified API**: Single method per operation, no version confusion
- **Performance**: All methods now use optimized implementations
- **Consistency**: Uniform naming across all tensor operations

For complete migration instructions, see the [Migration Guide](#migration-guide) section below.

## 🏗️ Core Architecture

### Module Structure

```
rustorch/
├── tensor/              # Core tensor operations
│   ├── core.rs         # Tensor data structure
│   └── ops/            # Tensor operations (v0.5.0+)
│       ├── arithmetic.rs    # Basic arithmetic operations
│       ├── mathematical.rs  # Mathematical functions (exp, ln, sin, cos, etc.)
│       ├── operators.rs     # Operator overloads (+, -, *, /, +=, -=)
│       ├── matrix.rs        # Matrix operations (matmul, transpose, etc.)
│       ├── statistical.rs   # Statistical operations
│       └── utilities.rs     # Utility operations
├── nn/                  # Neural network layers  
├── autograd/            # Automatic differentiation
├── optim/               # Optimizers (SGD, Adam)
├── gpu/                 # GPU acceleration
├── visualization/       # Plotting and visualization
├── models/              # High-level model APIs
├── training/            # Training utilities
├── data/                # Data loading and processing
└── wasm/                # WebAssembly bindings
```

## 📊 Tensor Operations

### Core Tensor API

```rust
// Creation
let tensor = Tensor::new(vec![2, 3, 4]);           // Shape-based
let tensor = Tensor::from_vec(data, shape);        // From data
let tensor = Tensor::zeros(vec![10, 10]);          // Zero-filled
let tensor = Tensor::ones(vec![5, 5]);             // One-filled
let tensor = Tensor::randn(vec![3, 3]);            // Normal distribution

// Basic Operations
let result = &a + &b;                              // Addition
let result = &a - &b;                              // Subtraction  
let result = &a * &b;                              // Element-wise multiplication
let result = a.matmul(&b);                         // Matrix multiplication
let result = a.transpose();                        // Matrix transpose

// Mathematical Functions (v0.5.0+)
let result = tensor.exp();                         // Exponential function
let result = tensor.ln();                          // Natural logarithm  
let result = tensor.sin();                         // Sine function
let result = tensor.cos();                         // Cosine function
let result = tensor.tan();                         // Tangent function
let result = tensor.sqrt();                        // Square root
let result = tensor.abs();                         // Absolute value
let result = tensor.pow(2.0);                      // Power function

// Enhanced Operator Overloads (v0.5.0+)
let result = &a + &b;                              // Tensor + Tensor
let result = &a - &b;                              // Tensor - Tensor
let result = &a * &b;                              // Tensor * Tensor (element-wise)
let result = &a / &b;                              // Tensor / Tensor (element-wise)

// Scalar Operations
let result = &a + 10.0;                            // Tensor + Scalar
let result = &a - 5.0;                             // Tensor - Scalar
let result = &a * 2.0;                             // Tensor * Scalar
let result = &a / 3.0;                             // Tensor / Scalar

// In-place Operations  
let mut a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
a += &b;                                           // In-place addition
a -= &b;                                           // In-place subtraction

// Aggregation
let sum = tensor.sum();                            // Sum all elements
let mean = tensor.mean();                          // Average
let max_val = tensor.max();                        // Maximum value
let min_val = tensor.min();                        // Minimum value

// Shape Manipulation
let reshaped = tensor.reshape(vec![6, 4]);         // Reshape
let squeezed = tensor.squeeze();                   // Remove size-1 dims
let expanded = tensor.unsqueeze(1);                // Add dimension
```

### Advanced Tensor Features

```rust
// Broadcasting
let a = Tensor::new(vec![3, 1]);
let b = Tensor::new(vec![1, 4]);
let result = &a + &b;                              // Broadcasts to [3, 4]

// Indexing and Slicing
let slice = tensor.slice(0, 1, 3);                 // Slice along dimension
let indexed = tensor.index(&[1, 2]);               // Multi-dimensional index

// Memory Management
let view = tensor.view();                          // Create view (no copy)
let cloned = tensor.clone();                       // Deep copy
let detached = tensor.detach();                    // Remove from computation graph
```

## 🧠 Neural Network Layers

### Linear Layers

```rust
use rustorch::nn::Linear;

let linear = Linear::new(784, 128);                // Input: 784, Output: 128
let output = linear.forward(&input);               // Forward pass

// With bias control
let linear = Linear::with_bias(784, 128, false);   // No bias term
```

### Convolutional Layers

```rust
use rustorch::nn::Conv2d;

// Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
let conv = Conv2d::new(3, 64, (3, 3), Some((1, 1)), Some((1, 1)), None);
let output = conv.forward(&input);                 // Input: [N, C, H, W]
```

### Activation Functions

```rust
use rustorch::nn::*;

// Built-in activations
let relu_out = relu(&input);                       // ReLU activation
let sigmoid_out = sigmoid(&input);                 // Sigmoid activation
let tanh_out = tanh(&input);                       // Tanh activation
let softmax_out = softmax(&input, 1);              // Softmax along dim 1

// Parameterized activations
let leaky_relu = LeakyReLU::new(0.01);
let output = leaky_relu.forward(&input);
```

### Normalization Layers

```rust
use rustorch::nn::{BatchNorm2d, BatchNorm1d};

let bn2d = BatchNorm2d::new(64, None, None, None); // 64 channels
let normalized = bn2d.forward(&input);

let bn1d = BatchNorm1d::new(128, None, None, None);// 128 features
let normalized = bn1d.forward(&input);
```

## ⚡ Automatic Differentiation

### Variable and Gradient Computation

```rust
use rustorch::autograd::Variable;

// Create variable with gradient tracking
let var = Variable::new(tensor, true);             // requires_grad = true

// Forward pass
let output = some_function(&var);

// Backward pass
output.backward();                                 // Compute gradients

// Access gradients
if let Some(grad) = var.grad() {
    println!("Gradient: {:?}", grad.data());
}
```

### Custom Functions

```rust
use rustorch::autograd::Function;

struct CustomFunction;

impl Function<f32> for CustomFunction {
    fn forward(&self, ctx: &mut Context<f32>, input: &Tensor<f32>) -> Tensor<f32> {
        // Custom forward implementation
        input * 2.0
    }
    
    fn backward(&self, ctx: &Context<f32>, grad_output: &Tensor<f32>) -> Tensor<f32> {
        // Custom backward implementation  
        grad_output * 2.0
    }
}
```

## 🔧 Optimizers

### SGD Optimizer

```rust
use rustorch::optim::SGD;

let mut optimizer = SGD::new(parameters, 0.01);    // Learning rate: 0.01

// Training step
let loss = compute_loss(&predictions, &targets);
loss.backward();
optimizer.step();                                  // Update parameters
optimizer.zero_grad();                             // Clear gradients
```

### Adam Optimizer

```rust
use rustorch::optim::Adam;

let mut optimizer = Adam::new(parameters, 0.001, 0.9, 0.999); // lr, beta1, beta2

// Training with Adam
for epoch in 0..num_epochs {
    let loss = training_step(&model, &data);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

## 🎮 GPU Acceleration

### Device Management

```rust
use rustorch::gpu::{DeviceType, GpuContext};

// Check available devices
let devices = GpuContext::available_devices();
println!("Available devices: {:?}", devices);

// Select device
let device = DeviceType::CUDA(0);                  // CUDA device 0
let context = GpuContext::new(device)?;

// Move tensor to GPU
let gpu_tensor = tensor.to_device(&device);
```

### GPU Kernels

```rust
use rustorch::gpu::kernels::{AddKernel, MatMulKernel};

let add_kernel = AddKernel::new();
let result = add_kernel.execute(&gpu_tensor_a, &gpu_tensor_b)?;

let matmul_kernel = MatMulKernel::new();
let result = matmul_kernel.execute(&gpu_tensor_a, &gpu_tensor_b)?;
```

## 📊 Visualization

### Training Curves

```rust
use rustorch::visualization::{TrainingPlotter, PlotConfig};

let plotter = TrainingPlotter::new();
let svg = plotter.plot_training_curves(&training_history)?;

// Save to file
std::fs::write("training_curves.svg", svg)?;
```

### Tensor Visualization

```rust
use rustorch::visualization::{TensorVisualizer, ColorMap};

let visualizer = TensorVisualizer::new();

// Heatmap for 2D tensors
let heatmap = visualizer.plot_heatmap(&tensor_2d)?;

// Bar chart for 1D tensors  
let bar_chart = visualizer.plot_bar_chart(&tensor_1d)?;

// 3D tensor slices
let slices = visualizer.plot_3d_slices(&tensor_3d)?;
```

### Computation Graphs

```rust
use rustorch::visualization::GraphVisualizer;

let mut graph_viz = GraphVisualizer::new();
graph_viz.build_graph(&variable)?;

// Generate SVG
let svg = graph_viz.to_svg()?;

// Generate DOT format for Graphviz
let dot = graph_viz.to_dot()?;
```

## 🚀 High-Level APIs

### Sequential Models

```rust
use rustorch::models::Sequential;
use rustorch::nn::*;

let model = Sequential::new()
    .add(Linear::new(784, 128))
    .add(ReLU::new())
    .add(Dropout::new(0.5))
    .add(Linear::new(128, 64))
    .add(ReLU::new())
    .add(Linear::new(64, 10));

let output = model.forward(&input);
```

### Training Loop

```rust
use rustorch::training::{Trainer, TrainingConfig};
use rustorch::models::TrainingHistory;

let config = TrainingConfig {
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    ..Default::default()
};

let mut trainer = Trainer::new(model, optimizer, loss_fn, config);
let history = trainer.train(&train_data, &val_data)?;
```

## 🌐 WebAssembly Support

### WASM Bindings

```rust
use rustorch::wasm::*;

#[wasm_bindgen]
pub fn create_model() -> WasmModel {
    let model = Sequential::new()
        .add(Linear::new(2, 10))
        .add(ReLU::new())
        .add(Linear::new(10, 1));
    
    WasmModel::new(model)
}

#[wasm_bindgen]
pub fn train_model(model: &mut WasmModel, data: &[f32]) -> f32 {
    model.train_step(data)
}
```

### JavaScript Integration

```javascript
import init, { create_model, train_model } from './pkg/rustorch.js';

async function run() {
    await init();
    
    const model = create_model();
    const trainingData = new Float32Array([1, 2, 3, 4]);
    const loss = train_model(model, trainingData);
    
    console.log(`Training loss: ${loss}`);
}
```

## 📈 Performance Optimization

### SIMD Operations

```rust
// SIMD operations are automatically applied when possible
let result = &large_tensor_a + &large_tensor_b;    // Uses AVX2/SSE4.1 if available
```

### Memory Pool

```rust
use rustorch::memory::MemoryPool;

let pool = MemoryPool::new();
let tensor = pool.allocate_tensor(vec![1000, 1000]); // Pre-allocated memory
```

### Parallel Processing

```rust
use rustorch::parallel::ParallelTensorOps;

// Automatic parallelization for large tensors
let result = tensor.parallel_map(|x| x.powi(2));   // Parallel element-wise operation
```

## 🔍 Error Handling

All RusTorch operations use comprehensive error handling:

```rust
use rustorch::RusTorchError;

match tensor_operation() {
    Ok(result) => println!("Success: {:?}", result),
    Err(RusTorchError::InvalidShape(msg)) => eprintln!("Shape error: {}", msg),
    Err(RusTorchError::ComputationError(msg)) => eprintln!("Computation error: {}", msg),
    Err(RusTorchError::GpuError(msg)) => eprintln!("GPU error: {}", msg),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## 📚 Additional Resources

- [Examples Directory](../examples/): Complete working examples
- [Benchmarks](../benches/): Performance benchmarks and comparisons  
- [GPU Guide](GPU_ACCELERATION_GUIDE.md): Detailed GPU setup and usage
- [Python Integration](../python/): Python bindings and usage
- [WebAssembly Guide](../examples/wasm_basic.html): Browser deployment examples

For the most up-to-date API documentation, run:
```bash
cargo doc --open --no-deps
```
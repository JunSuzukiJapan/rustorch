# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-733%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**A production-ready deep learning library in Rust with PyTorch-like API, GPU acceleration, and enterprise-grade performance**  
**本番環境対応のRust製ディープラーニングライブラリ - PyTorchライクなAPI、GPU加速、エンタープライズグレードパフォーマンス**

RusTorch is a fully functional deep learning library that leverages Rust's safety and performance, providing comprehensive tensor operations, automatic differentiation, neural network layers, transformer architectures, multi-backend GPU acceleration (CUDA/Metal/OpenCL), advanced SIMD optimizations, and enterprise-grade memory management features.

## ✨ Features

- 🔥 **Comprehensive Tensor Operations**: Math operations, broadcasting, indexing, and statistics
- 🤖 **Transformer Architecture**: Complete transformer implementation with multi-head attention
- 🧮 **Matrix Decomposition**: Complete SVD, QR, LU decomposition and eigenvalue solver with PyTorch compatibility
- 🧠 **Automatic Differentiation**: Tape-based computational graph for gradient computation
- 🏗️ **Neural Network Layers**: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout, and more
- ⚡ **SIMD Optimizations**: AVX2/SSE4.1 vectorized operations for high performance
- 🎮 **GPU Integration**: CUDA/Metal/OpenCL support with automatic device selection
- 🌐 **WebAssembly Support**: Browser-compatible WASM bindings for client-side ML
- 📁 **Model Format Support**: Safetensors, ONNX inference, PyTorch state dict compatibility
- ✅ **Production Ready**: 733 tests passing (100% success rate), unified error handling system
- 📈 **Advanced Optimizers**: SGD, Adam, AdamW, RMSprop, AdaGrad with learning rate schedulers

For detailed features, see [Features Documentation](docs/features.md).

## 🚀 Quick Start

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustorch = "0.4.1"

# Optional features
[features]
default = ["linalg"]
linalg = ["rustorch/linalg"]           # Linear algebra operations (SVD, QR, LU, eigenvalue)
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"] 
opencl = ["rustorch/opencl"]
safetensors = ["rustorch/safetensors"]
onnx = ["rustorch/onnx"]

# To disable linalg features (avoid OpenBLAS/LAPACK dependencies):
rustorch = { version = "0.4.1", default-features = false }
```

### Basic Usage

```rust
use rustorch::tensor::Tensor;
use rustorch::optim::{SGD, WarmupScheduler, OneCycleLR, AnnealStrategy};

fn main() {
    // Create tensors
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // Basic operations
    let c = &a + &b;  // Addition
    let d = a.matmul(&b);  // Matrix multiplication
    
    // Advanced optimizers with learning rate scheduling
    let optimizer = SGD::new(0.01);
    let mut scheduler = WarmupScheduler::new(optimizer, 0.1, 5); // Warmup to 0.1 over 5 epochs
    
    // One-cycle learning rate policy
    let optimizer2 = SGD::new(0.01);
    let mut one_cycle = OneCycleLR::new(optimizer2, 1.0, 100, 0.3, AnnealStrategy::Cos);
    
    println!("Shape: {:?}", c.shape());
    println!("Result: {:?}", c.as_slice());
}
```

For more examples, see [Getting Started Guide](docs/getting-started.md).

## 📚 Documentation

- **[Getting Started](docs/getting-started.md)** - Basic usage and examples
- **[Features](docs/features.md)** - Complete feature list and specifications
- **[Performance](docs/performance.md)** - Benchmarks and optimization details
- **[Architecture](docs/architecture.md)** - System design and project structure
- **[Examples](docs/examples.md)** - Comprehensive code examples
- **[API Documentation](https://docs.rs/rustorch)** - Detailed API reference
- **[GPU Acceleration Guide](docs/GPU_ACCELERATION_GUIDE.md)** - GPU setup and usage
- **[Production Guide](docs/PRODUCTION_GUIDE.md)** - Deployment and scaling

## 📊 Performance

**Latest benchmark results:**

| Operation | Performance | Details |
|-----------|-------------|---------|
| **Tensor Addition** | 34K - 2.3M ops/sec | ✅ Broadcasting support |
| **Tensor Sum** | 52M+ ops/sec | ✅ Consistently high performance |
| **Matrix Multiplication** | 0.71 - 0.77 GFLOPS | ✅ Stable scaling |
| **Neural Network Inference** | 15 - 60 inferences/sec | ✅ Batch processing |

For detailed performance analysis, see [Performance Documentation](docs/performance.md).

## 🧪 Testing

**All 733 tests passing** - Production-ready quality assurance with unified error handling system.

```bash
# Run all tests
cargo test

# Run with release optimizations
cargo test --release
```

## 🚀 Production Deployment

### Docker
```bash
# Production deployment
docker build -t rustorch:latest .
docker run -it rustorch:latest

# GPU-enabled deployment
docker build -f Dockerfile.gpu -t rustorch:gpu .
docker run --gpus all -it rustorch:gpu
```

For complete deployment guide, see [Production Guide](docs/PRODUCTION_GUIDE.md).

## 🤝 Contributing

We welcome contributions! See areas where help is especially needed:

- **🎯 Special Functions Precision**: Improve numerical accuracy
- **⚡ Performance Optimization**: SIMD improvements, GPU optimization  
- **🧪 Testing**: More comprehensive test cases
- **📚 Documentation**: Examples, tutorials, improvements
- **🌐 Platform Support**: WebAssembly, mobile platforms

### Development Setup

```bash
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# Run tests
cargo test --all-features

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy --all-targets --all-features
```

## License

Licensed under either of:

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
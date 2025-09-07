# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-1060%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

## 🌐 多言語ドキュメント / Multilingual Documentation

| Language | README | Jupyter Guide |
|----------|---------|---------------|
| 🇺🇸 [English](docs/en/README.md) | [📖 Main](docs/en/README.md) | [📓 Jupyter](docs/en/jupyter-guide.md) |
| 🇫🇷 [Français](docs/fr/README.md) | [📖 Principal](docs/fr/README.md) | [📓 Jupyter](docs/fr/jupyter-guide.md) |
| 🇮🇹 [Italiano](docs/it/README.md) | [📖 Principale](docs/it/README.md) | [📓 Jupyter](docs/it/jupyter-guide.md) |
| 🇪🇸 [Español](docs/es/README.md) | [📖 Principal](docs/es/README.md) | [📓 Jupyter](docs/es/jupyter-guide.md) |
| 🇨🇳 [中文](docs/zh/README.md) | [📖 主要](docs/zh/README.md) | [📓 Jupyter](docs/zh/jupyter-guide.md) |
| 🇰🇷 [한국어](docs/ko/README.md) | [📖 메인](docs/ko/README.md) | [📓 Jupyter](docs/ko/jupyter-guide.md) |
| 🇩🇪 [Deutsch](docs/de/README.md) | [📖 Hauptseite](docs/de/README.md) | [📓 Jupyter](docs/de/jupyter-guide.md) |
| 🇷🇺 [Русский](docs/ru/README.md) | [📖 Основной](docs/ru/README.md) | [📓 Jupyter](docs/ru/jupyter-guide.md) |

---

**A production-ready deep learning library in Rust with PyTorch-like API, GPU acceleration, and enterprise-grade performance**  
**本番環境対応のRust製ディープラーニングライブラリ - PyTorchライクなAPI、GPU加速、エンタープライズグレードパフォーマンス**

RusTorch is a fully functional deep learning library that leverages Rust's safety and performance. **Phase 8 COMPLETED** brings advanced tensor utilities with **conditional operations, indexing, and statistical functions**. Features comprehensive tensor operations, automatic differentiation, neural network layers, transformer architectures, multi-backend GPU acceleration (CUDA/Metal/OpenCL), advanced SIMD optimizations, enterprise-grade memory management, data validation & quality assurance, and comprehensive debug & logging systems.

## ✨ Features

- 🔥 **Comprehensive Tensor Operations**: Math operations, broadcasting, indexing, statistics, and Phase 8 advanced utilities
- 🤖 **Transformer Architecture**: Complete transformer implementation with multi-head attention
- 🧮 **Matrix Decomposition**: SVD, QR, eigenvalue decomposition with PyTorch compatibility
- 🧠 **Automatic Differentiation**: Tape-based computational graph for gradient computation
- 🚀 **Dynamic Execution Engine**: JIT compilation and runtime optimization
- 🏗️ **Neural Network Layers**: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout, and more
- ⚡ **Cross-Platform Optimizations**: SIMD (AVX2/SSE/NEON), platform-specific, and hardware-aware optimizations
- 🎮 **GPU Integration**: CUDA/Metal/OpenCL support with automatic device selection
- 🌐 **WebAssembly Support**: Complete browser ML with Neural Network layers, Computer Vision, and real-time inference
- 🎮 **WebGPU Integration**: Chrome-optimized GPU acceleration with CPU fallback for cross-browser compatibility
- 📁 **Model Format Support**: Safetensors, ONNX inference, PyTorch state dict compatibility
- ✅ **Production Ready**: 1060 tests passing, unified error handling system with `RusTorchError`
- 📐 **Enhanced Mathematical Functions**: Complete set of mathematical functions (exp, ln, sin, cos, tan, sqrt, abs, pow)
- 🔧 **Advanced Operator Overloads**: Full operator support for tensors with scalar operations and in-place assignments
- 📈 **Advanced Optimizers**: SGD, Adam, AdamW, RMSprop, AdaGrad with learning rate schedulers
- 🚀 **Phase 2 Optimization Framework**: NAdam, RAdam, Adamax, Enhanced L-BFGS with 500%+ performance boost
- ⚡ **World-Class Performance**: Adamax 33,632 steps/sec, RAdam 21,939 steps/sec, NAdam 18,976 steps/sec
- 🔍 **Data Validation & Quality Assurance**: Statistical analysis, anomaly detection, consistency checking, real-time monitoring
- 🐛 **Comprehensive Debug & Logging**: Structured logging, performance profiling, memory tracking, automated alerts
- 🎯 **Phase 8 Tensor Utilities**: Conditional operations (where, masked_select, masked_fill), indexing operations (gather, scatter, index_select), statistical operations (topk, kthvalue), and advanced utilities (unique, histogram)

For detailed features, see [Features Documentation](docs/features.md).

## 🚀 Quick Start

### 📓 Interactive Jupyter Demos

**🌟 Universal One-Liner** - Smart installer (defaults to hybrid environment):
```bash
curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/install_jupyter.sh | bash
```
*🦀🐍 Default: Installs Python+Rust dual-kernel environment + global launcher*

**🚀 Manual Setup** - Or choose specific demo type:

| Demo Type | Setup Command | Description |
|-----------|---------------|-------------|
| 🦀🐍 **Hybrid** | `./start_jupyter_hybrid.sh` | Python + Rust dual-kernel environment |
| 🐍 **Python** | `./start_jupyter.sh` | Standard CPU-based ML demos |
| ⚡ **WebGPU** | `./start_jupyter_webgpu.sh` | Browser GPU acceleration (Chrome) |
| 🦀 **Rust Kernel** | `./quick_start_rust_kernel.sh` | Native Rust in Jupyter |
| 🌐 **Online** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JunSuzukiJapan/rustorch/main?urlpath=lab) | No setup needed - run in browser |

**⚡ Quick Restart** - After any setup:
```bash
./start_jupyter_quick.sh  # Interactive menu
# OR (after universal installer):
rustorch-jupyter          # Global command from anywhere
```

📚 **Detailed Setup**: [Complete Jupyter Guide](README_JUPYTER.md) | [日本語ガイド](docs/jupyter-wasm-guide.md)

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustorch = "0.6.1"

# Optional features
[features]
default = ["linalg"]
linalg = ["rustorch/linalg"]           # Linear algebra operations (SVD, QR, eigenvalue)
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"] 
opencl = ["rustorch/opencl"]
safetensors = ["rustorch/safetensors"]
onnx = ["rustorch/onnx"]
wasm = ["rustorch/wasm"]                # WebAssembly support for browser ML
webgpu = ["rustorch/webgpu"]            # Chrome-optimized WebGPU acceleration

# To disable linalg features (avoid OpenBLAS/LAPACK dependencies):
rustorch = { version = "0.6.1", default-features = false }
```

### Basic Usage

```rust
use rustorch::tensor::Tensor;
use rustorch::optim::{SGD, WarmupScheduler, OneCycleLR, AnnealStrategy};
use rustorch::error::RusTorchResult;

fn main() -> RusTorchResult<()> {
    // Create tensors
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // Basic operations with operator overloads
    let c = &a + &b;  // Element-wise addition
    let d = &a - &b;  // Element-wise subtraction
    let e = &a * &b;  // Element-wise multiplication
    let f = &a / &b;  // Element-wise division
    
    // Scalar operations
    let g = &a + 10.0;  // Add scalar to all elements
    let h = &a * 2.0;   // Multiply by scalar
    
    // Mathematical functions
    let exp_result = a.exp();   // Exponential function
    let ln_result = a.ln();     // Natural logarithm
    let sin_result = a.sin();   // Sine function
    let sqrt_result = a.sqrt(); // Square root
    
    // Matrix operations
    let matmul_result = a.matmul(&b);  // Matrix multiplication
    
    // Linear algebra operations (requires linalg feature)
    #[cfg(feature = "linalg")]
    {
        let svd_result = a.svd();       // SVD decomposition
        let qr_result = a.qr();         // QR decomposition  
        let eig_result = a.eigh();      // Eigenvalue decomposition
    }
    
    // Advanced optimizers with learning rate scheduling
    let optimizer = SGD::new(0.01);
    let mut scheduler = WarmupScheduler::new(optimizer, 0.1, 5); // Warmup to 0.1 over 5 epochs
    
    // One-cycle learning rate policy
    let optimizer2 = SGD::new(0.01);
    let mut one_cycle = OneCycleLR::new(optimizer2, 1.0, 100, 0.3, AnnealStrategy::Cos);
    
    println!("Shape: {:?}", c.shape());
    println!("Result: {:?}", c.as_slice());
    
    Ok(())
}
```

### WebAssembly Usage

For browser-based ML applications:

```javascript
import init, * as rustorch from './pkg/rustorch.js';

async function browserML() {
    await init();
    
    // Neural network layers
    const linear = new rustorch.WasmLinear(784, 10, true);
    const conv = new rustorch.WasmConv2d(3, 32, 3, 1, 1, true);
    
    // Enhanced mathematical functions
    const gamma_result = rustorch.WasmSpecial.gamma_batch([1.5, 2.0, 2.5]);
    const bessel_result = rustorch.WasmSpecial.bessel_i_batch(0, [0.5, 1.0, 1.5]);
    
    // Statistical distributions
    const normal_dist = new rustorch.WasmDistributions();
    const samples = normal_dist.normal_sample_batch(100, 0.0, 1.0);
    
    // Optimizers for training
    const sgd = new rustorch.WasmOptimizer();
    sgd.sgd_init(0.01, 0.9); // learning_rate, momentum
    
    // Image processing
    const resized = rustorch.WasmVision.resize(image, 256, 256, 224, 224, 3);
    const normalized = rustorch.WasmVision.normalize(resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 3);
    
    // Forward pass
    const predictions = conv.forward(normalized, 1, 224, 224);
    console.log('Browser ML predictions:', predictions);
}
```

### WebGPU Acceleration (Chrome Optimized)

For Chrome browsers with WebGPU support:

```javascript
import init, * as rustorch from './pkg/rustorch.js';

async function webgpuML() {
    await init();
    
    // Initialize WebGPU engine
    const webgpu = new rustorch.WebGPUSimple();
    await webgpu.initialize();
    
    // Check WebGPU support
    const supported = await webgpu.check_webgpu_support();
    if (supported) {
        console.log('🚀 WebGPU acceleration enabled');
        
        // High-performance tensor operations
        const a = [1, 2, 3, 4];
        const b = [5, 6, 7, 8];
        const result = webgpu.tensor_add_cpu(a, b);
        
        // Matrix multiplication with GPU optimization
        const matrix_a = new Array(256 * 256).fill(0).map(() => Math.random());
        const matrix_b = new Array(256 * 256).fill(0).map(() => Math.random());
        const matrix_result = webgpu.matrix_multiply_cpu(matrix_a, matrix_b, 256, 256, 256);
        
        console.log('WebGPU accelerated computation complete');
    } else {
        console.log('⚠️ WebGPU not supported, using CPU fallback');
    }
    
    // Interactive demo interface
    const demo = new rustorch.WebGPUSimpleDemo();
    demo.create_interface(); // Creates browser UI for testing
}
```

**Run examples:**
```bash
# Basic functionality examples
cargo run --example activation_demo --no-default-features
cargo run --example complex_tensor_demo --no-default-features  
cargo run --example neural_network_demo --no-default-features
cargo run --example autograd_demo --no-default-features

# Machine learning examples
cargo run --example boston_housing_regression --no-default-features
cargo run --example vision_pipeline_demo --no-default-features
cargo run --example embedding_demo --no-default-features

# Linear algebra examples (requires linalg feature)
cargo run --example eigenvalue_demo --features linalg
cargo run --example svd_demo --features linalg
cargo run --example matrix_decomposition_demo --features linalg

# Mathematical functions
cargo run --example special_functions_demo --no-default-features
cargo run --example fft_demo --no-default-features
```

For more examples, see [Getting Started Guide](docs/getting-started.md) and [WebAssembly Guide](docs/WASM_GUIDE.md).

## 📚 Documentation

- **[Getting Started](docs/getting-started.md)** - Basic usage and examples
- **[Features](docs/features.md)** - Complete feature list and specifications
- **[Performance](docs/performance.md)** - Benchmarks and optimization details
- **[Architecture](docs/architecture.md)** - System design and project structure
- **[Examples](docs/examples.md)** - Comprehensive code examples
- **[API Documentation](https://docs.rs/rustorch)** - Detailed API reference

### WebAssembly & Browser ML
- **[WebAssembly Guide](docs/WASM_GUIDE.md)** - Complete WASM integration and API reference
- **[Jupyter WASM Guide](docs/jupyter-wasm-guide.md)** - Step-by-step Jupyter Notebook setup (日本語)
- **[Jupyter WASM Guide (English)](docs/jupyter-wasm-guide-en.md)** - Step-by-step Jupyter Notebook setup
- **[WebGPU Integration](docs/WEBGPU_INTEGRATION.md)** - Chrome-optimized GPU acceleration
- **[Browser Compatibility](docs/BROWSER_COMPATIBILITY.md)** - Cross-browser support matrix
- **[WASM Performance](docs/WASM_PERFORMANCE.md)** - Benchmarking and optimization strategies

### Production & Operations
- **[GPU Acceleration Guide](docs/GPU_ACCELERATION_GUIDE.md)** - GPU setup and usage
- **[Production Guide](docs/PRODUCTION_GUIDE.md)** - Deployment and scaling
- **[Data Validation Guide](docs/DATA_VALIDATION_GUIDE.md)** - Quality assurance and validation
- **[Debug & Logging Guide](docs/DEBUG_GUIDE.md)** - Comprehensive debugging tools

## 📊 Performance

**🏆 Phase 2 Performance Revolution - Up to 580% improvement:**

### Advanced Optimizer Performance (Release Mode)
| Optimizer | Performance | Use Case | Status |
|-----------|-------------|----------|--------|
| **Adamax** | **33,632 steps/sec** ⚡ | Sparse features, Embeddings | **FASTEST** |
| **RAdam** | **21,939 steps/sec** 🚀 | General deep learning, Stability | **RECOMMENDED** |
| **NAdam** | **18,976 steps/sec** ✨ | NLP, Fine-tuning | **NLP-OPTIMIZED** |

### Core Operations Performance
| Operation | Performance | Details |
|-----------|-------------|---------|
| **SVD Decomposition** | ~1ms (8x8 matrix) | ✅ LAPACK-based |
| **QR Decomposition** | ~24μs (8x8 matrix) | ✅ Fast decomposition |
| **Eigenvalue** | ~165μs (8x8 matrix) | ✅ Symmetric matrices |
| **Complex FFT** | 10-312μs (8-64 samples) | ✅ Cooley-Tukey optimized |
| **Neural Network** | 1-7s training | ✅ Boston housing demo |
| **Activation Functions** | <1μs | ✅ ReLU, Sigmoid, Tanh |

### Phase 2 Architectural Benefits
- **🧹 50%+ code reduction** through unified GenericAdamOptimizer
- **🔧 Consistent API** across all Adam variants
- **⚡ Shared optimizations** benefiting all optimizers
- **🛡️ Robust error handling** with RusTorchResult<T>

**Run benchmarks:**
```bash
# Basic benchmarks (no external dependencies)
cargo bench --no-default-features

# Linear algebra benchmarks (requires linalg feature) 
cargo bench --features linalg

# Phase 2 optimizer benchmarks (NEW!)
cargo run --bin quick_optimizer_bench --release

# Quick manual benchmark
cargo run --bin manual_quick_bench
```

For detailed performance analysis, see [Performance Documentation](docs/performance.md).


## 🧪 Testing

**1060 tests passing** - Production-ready quality assurance with unified `RusTorchError` error handling system.

```bash
# Run all tests (recommended for CI/development)
cargo test --no-default-features

# Run tests with linear algebra features
cargo test --features linalg

# Run doctests
cargo test --doc --no-default-features

# Test specific modules
cargo test --no-default-features tensor::
cargo test --no-default-features complex::
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

## 🔒 API Stability / API安定性

**English**: Starting from v0.6.0, RusTorch has reached a stable API milestone. We are committed to maintaining backward compatibility and will avoid major breaking changes in the near future. This ensures a reliable foundation for production applications and long-term projects.

**日本語**: v0.6.0以降、RusTorchは安定したAPIマイルストーンに達しました。後方互換性を維持することをお約束し、近い将来において大規模な破壊的変更は行いません。これにより、本番アプリケーションと長期プロジェクトに信頼性の高い基盤を提供します。

## License

Licensed under either of:

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
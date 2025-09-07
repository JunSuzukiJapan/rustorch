# RusTorch Jupyter Complete Guide

The ultimate guide to using RusTorch in Jupyter environments with Python, Rust, and hybrid setups.

## 📚 Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation Methods](#-installation-methods)
3. [Environment Types](#-environment-types)
4. [Hybrid Environment](#-hybrid-environment)
5. [Usage Examples](#-usage-examples)
6. [Advanced Features](#-advanced-features)
7. [Troubleshooting](#-troubleshooting)
8. [Migration Guide](#-migration-guide)

## 🚀 Quick Start

### Universal One-Liner (Recommended)

The easiest way to get started with RusTorch in Jupyter:

```bash
./install_jupyter.sh
```

**What it does:**
- 🔍 **Auto-detects** your environment (OS, CPU, GPU)
- 🦀🐍 **Installs hybrid** Python+Rust dual-kernel environment by default
- 📦 **Creates global launcher** (`rustorch-jupyter` command)
- ⚡ **Optimizes** for your hardware (CUDA, Metal, WebGPU, CPU)

### Next Time Launch
```bash
rustorch-jupyter          # Global command (after installer)
# OR
./start_jupyter_quick.sh  # Interactive menu
```

## 📦 Installation Methods

### 1. Universal Installer (Recommended)

**Auto-Detection Installation:**
```bash
./install_jupyter.sh
```

**Custom Installation Path:**
```bash
RUSTORCH_INSTALL_PATH=/usr/local/bin ./install_jupyter.sh
```

**Interactive Options:**
- `[1]` Hybrid Environment (Default) - Python + Rust kernels
- `[2]` GPU-Optimized - CUDA/Metal/WebGPU optimized single environment
- `[q]` Cancel

### 2. Manual Setup

Choose specific environment type:

| Environment Type | Command | Use Case |
|-----------------|---------|----------|
| 🦀🐍 **Hybrid** | `./start_jupyter_hybrid.sh` | Both Python and Rust development |
| 🐍 **Python** | `./start_jupyter.sh` | Python-focused ML development |
| ⚡ **WebGPU** | `./start_jupyter_webgpu.sh` | Browser GPU acceleration |
| 🦀 **Rust** | `./quick_start_rust_kernel.sh` | Native Rust development |
| 🌐 **Online** | [Binder](https://mybinder.org/v2/gh/JunSuzukiJapan/rustorch/main?urlpath=lab) | No local setup required |

## 🏗️ Environment Types

### Hybrid Environment (Default)
- **Best for**: Full-stack ML development
- **Features**: Python + Rust kernels, RusTorch bridge, sample notebooks
- **Hardware**: Adapts to available GPU (CUDA/Metal/CPU)

### Python Environment  
- **Best for**: Python developers wanting RusTorch features
- **Features**: Python kernel with RusTorch Python bindings
- **Hardware**: CPU/GPU optimized

### WebGPU Environment
- **Best for**: Browser-based GPU acceleration
- **Features**: WebAssembly + WebGPU, Chrome-optimized
- **Hardware**: Modern browsers with WebGPU support

### Rust Kernel Environment
- **Best for**: Native Rust development
- **Features**: evcxr kernel, direct RusTorch library access
- **Hardware**: Native performance, all features available

## 🦀🐍 Hybrid Environment

The hybrid environment provides the best of both worlds: Python's ecosystem with Rust's performance.

### Architecture

```
Jupyter Lab
├── Python Kernel
│   ├── NumPy, Pandas, Matplotlib
│   ├── RusTorch Python bindings
│   └── rustorch_bridge module
└── Rust Kernel (evcxr)
    ├── Native RusTorch library
    ├── Direct hardware access
    └── Zero-cost abstractions
```

### Setup Process

1. **Environment Detection**
   ```bash
   🔍 Environment Detection
   ==================================
   OS: macos
   CPU: arm64
   GPU: metal
   WebGPU Support: false
   
   🎯 Default: Hybrid Environment
   ```

2. **Installation Steps**
   - Creates Python virtual environment (`.venv-hybrid`)
   - Installs Python packages (jupyter, numpy, matplotlib, etc.)
   - Installs Rust Jupyter kernel (evcxr)
   - Creates RusTorch Python bridge
   - Generates sample notebooks

3. **Global Launcher Setup**
   - Creates `rustorch-jupyter` command in `~/bin/`
   - Adds to PATH automatically
   - Works from any directory

### Usage Examples

**Python Cell with RusTorch:**
```python
import numpy as np
from rustorch_bridge import rust, tensor

# Prepare data in Python
data = np.random.randn(100, 100)

# Process with RusTorch in Rust
result = rust('''
    let tensor = Tensor::randn(&[100, 100]);
    let result = tensor.matmul(&tensor.transpose(0, 1));
    println!("Matrix multiplication completed: {:?}", result.shape());
''')
```

**Native Rust Cell:**
```rust
:dep rustorch = "0.6.2"
extern crate rustorch;

use rustorch::tensor::Tensor;
use rustorch::nn::Linear;

let model = Linear::new(784, 10);
let input = Tensor::<f32>::randn(&[32, 784]);
let output = model.forward(&input);
println!("Neural network output: {:?}", output.shape());
```

## 📊 Usage Examples

### Machine Learning Pipeline

**1. Data Preparation (Python)**
```python
import pandas as pd
import numpy as np

# Load and preprocess data
data = pd.read_csv('dataset.csv')
X = data.drop('target', axis=1).values
y = data['target'].values

print(f"Dataset shape: {X.shape}")
```

**2. Model Training (Rust)**
```rust
use rustorch::nn::{Linear, SGD};
use rustorch::tensor::Tensor;

// Convert to RusTorch tensors
let X_tensor = Tensor::from_vec(X.flatten().to_vec(), vec![X.shape[0], X.shape[1]]);
let y_tensor = Tensor::from_vec(y.to_vec(), vec![y.len()]);

// Create model
let mut model = Linear::new(X.shape[1], 1);
let mut optimizer = SGD::new(model.parameters(), 0.01);

// Training loop
for epoch in 0..100 {
    let output = model.forward(&X_tensor);
    let loss = mse_loss(&output, &y_tensor);
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    
    if epoch % 10 == 0 {
        println!("Epoch {}: Loss = {:.4}", epoch, loss.item());
    }
}
```

**3. Visualization (Python)**
```python
import matplotlib.pyplot as plt

# Get predictions from Rust model
predictions = rust_model_predict(X)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('RusTorch Model Predictions')
plt.show()
```

## ⚙️ Advanced Features

### GPU Acceleration

The installer automatically detects and configures GPU support:

**NVIDIA GPUs (CUDA):**
```bash
# Automatically detected and configured
cargo build --features cuda
```

**Apple Silicon (Metal):**
```bash
# Automatically detected and configured  
cargo build --features metal
```

**WebGPU (Browser):**
```bash
# For Chrome/Chromium with WebGPU support
./start_jupyter_webgpu.sh
```

### Performance Optimization

**Rust Kernel Optimizations:**
- **Compilation Cache**: First run compiles dependencies, subsequent runs use cache
- **Release Mode**: Optimized builds for production performance
- **SIMD Instructions**: Automatic vectorization on supported hardware

**Python Bridge Optimizations:**
- **Zero-Copy**: Direct memory sharing between Python and Rust
- **Batch Processing**: Efficient bulk operations
- **Memory Management**: Automatic cleanup and garbage collection

### Custom Configurations

**Environment Variables:**
```bash
# Custom install path
RUSTORCH_INSTALL_PATH=/opt/rustorch ./install_jupyter.sh

# Python version selection
RUSTORCH_PYTHON=/usr/bin/python3.11 ./install_jupyter.sh

# Feature flags
RUSTORCH_FEATURES="cuda,model-hub" ./install_jupyter.sh
```

## 🔧 Troubleshooting

### Common Issues

**1. Rust Kernel Compilation Errors**
```bash
# Install BLAS/LAPACK dependencies
# Ubuntu/Debian:
sudo apt-get install libblas-dev liblapack-dev libopenblas-dev

# macOS:
brew install openblas lapack

# Clear Rust cache
cargo clean
```

**2. Python Bridge Import Errors**
```bash
# Reinstall Python environment
rm -rf .venv-hybrid
./start_jupyter_hybrid.sh
```

**3. GPU Not Detected**
```bash
# Check GPU support
nvidia-smi  # NVIDIA
system_profiler SPDisplaysDataType  # macOS Metal

# Reinstall with correct features
./install_jupyter.sh  # Re-run auto-detection
```

**4. Permission Issues**
```bash
# Fix launcher permissions
chmod +x ~/bin/rustorch-jupyter

# Add to PATH manually
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Enable debug output
export RUST_LOG=debug
export JUPYTER_LOG_LEVEL=DEBUG

# Run with debug information
rustorch-jupyter
```

## 📈 Performance Benchmarks

### Environment Comparison

| Environment | Setup Time | Startup Time | Execution Speed | Memory Usage |
|-------------|------------|--------------|-----------------|--------------|
| Hybrid | 3-5 min | 10s | Native + Python | Medium |
| Python Only | 1-2 min | 5s | Python speed | Low |
| Rust Only | 2-3 min | 15s | Native | Low |
| WebGPU | 2-3 min | 8s | GPU accelerated | Medium |

### Hardware Optimization

| Hardware | Recommended Environment | Expected Performance |
|----------|------------------------|---------------------|
| Apple Silicon M1/M2/M3 | Hybrid with Metal | 5-10x speedup |
| NVIDIA GPU | Hybrid with CUDA | 10-100x speedup |
| Modern Intel/AMD | Hybrid with SIMD | 2-5x speedup |
| Older Hardware | Python Only | Standard speed |

## 🔄 Migration Guide

### From v0.5.x to v0.6.x

**1. Update Installation:**
```bash
# Remove old installation
rm -rf .venv .venv-*

# Install new hybrid environment
./install_jupyter.sh
```

**2. Update Notebook Dependencies:**
```rust
// Old
:dep rustorch = "0.5.15"

// New
:dep rustorch = "0.6.2"
extern crate rustorch;
extern crate ndarray;
```

**3. API Changes:**
```rust
// Updated tensor creation
let tensor = Tensor::from_vec(data, shape);  // v0.6.x
```

### From Other Libraries

**From PyTorch:**
- Python code remains largely compatible
- Rust provides additional performance
- GPU acceleration automatically configured

**From NumPy/SciPy:**
- Direct integration with NumPy arrays
- Rust tensors for performance-critical operations
- Seamless data exchange

## 🆘 Support

### Getting Help

- **Documentation**: Check this guide and API docs
- **GitHub Issues**: [Report bugs and request features](https://github.com/JunSuzukiJapan/rustorch/issues)
- **Discussions**: [Community support and questions](https://github.com/JunSuzukiJapan/rustorch/discussions)

### Useful Commands

```bash
# Check installation
rustorch-jupyter --help

# Quick launcher menu  
./start_jupyter_quick.sh

# Environment information
./install_jupyter.sh --help

# Update to latest
git pull origin main
./install_jupyter.sh
```

---

**Last Updated**: v0.6.2 - Hybrid Environment Release  
**Maintained by**: RusTorch Team
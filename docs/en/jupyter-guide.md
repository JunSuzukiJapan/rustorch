# RusTorch WASM Jupyter Notebook Guide

A step-by-step guide to easily use RusTorch WASM in Jupyter Notebook, designed for beginners.

## 📚 Table of Contents

1. [Requirements](#requirements)
2. [Setup Instructions](#setup-instructions)
3. [Basic Usage](#basic-usage)
4. [Practical Examples](#practical-examples)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)

## Requirements

### Minimum Requirements
- **Python 3.8+**
- **Jupyter Notebook** or **Jupyter Lab**
- **Node.js 16+** (for WASM builds)
- **Rust** (latest stable version)
- **wasm-pack** (to convert Rust code to WASM)

### Recommended Environment
- Memory: 8GB or more
- Browser: Latest versions of Chrome, Firefox, Safari
- OS: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

## Setup Instructions

### 🚀 Quick Start (Recommended)

**Easiest method**: Launch Jupyter Lab with one command
```bash
./start_jupyter.sh
```

This script automatically:
- Creates and activates virtual environment
- Installs dependencies (numpy, jupyter, matplotlib)
- Builds RusTorch Python bindings
- Launches Jupyter Lab with demo notebook open

### Manual Setup

#### Step 1: Install Basic Tools

```bash
# Check Python version
python --version

# Install Jupyter Lab
pip install jupyterlab

# Install Node.js (macOS with Homebrew)
brew install node

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

#### Step 2: Build RusTorch WASM

```bash
# Clone project
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# Add WASM target
rustup target add wasm32-unknown-unknown

# Build with wasm-pack
wasm-pack build --target web --out-dir pkg
```

#### Step 3: Start Jupyter

```bash
# Start Jupyter Lab
jupyter lab
```

## Basic Usage

### Creating Tensors

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // 1D tensor
    const vec = rt.create_tensor([1, 2, 3, 4, 5]);
    console.log('1D Tensor:', vec.to_array());
    
    // 2D tensor (matrix)
    const matrix = rt.create_tensor(
        [1, 2, 3, 4, 5, 6],
        [2, 3]  // shape: 2 rows, 3 columns
    );
    console.log('2D Tensor shape:', matrix.shape());
});
```

### Basic Operations

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    const a = rt.create_tensor([1, 2, 3, 4], [2, 2]);
    const b = rt.create_tensor([5, 6, 7, 8], [2, 2]);
    
    // Addition
    const sum = a.add(b);
    console.log('A + B =', sum.to_array());
    
    // Matrix multiplication
    const product = a.matmul(b);
    console.log('A × B =', product.to_array());
});
```

### Automatic Differentiation

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // Create tensor with gradient tracking
    const x = rt.create_tensor([2.0], null, true);  // requires_grad=true
    
    // Computation: y = x^2 + 3x + 1
    const y = x.mul(x).add(x.mul_scalar(3.0)).add_scalar(1.0);
    
    // Backpropagation
    y.backward();
    
    // Get gradient (dy/dx = 2x + 3 = 7 when x=2)
    console.log('Gradient:', x.grad().to_array());
});
```

## Practical Examples

### Linear Regression

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // Prepare data
    const X = rt.create_tensor([1, 2, 3, 4, 5]);
    const y = rt.create_tensor([2, 4, 6, 8, 10]);  // y = 2x
    
    // Initialize parameters
    let w = rt.create_tensor([0.5], null, true);
    let b = rt.create_tensor([0.0], null, true);
    
    const lr = 0.01;
    
    // Training loop
    for (let epoch = 0; epoch < 100; epoch++) {
        // Prediction: y_pred = wx + b
        const y_pred = X.mul(w).add(b);
        
        // Loss: MSE = mean((y_pred - y)^2)
        const loss = y_pred.sub(y).pow(2).mean();
        
        // Calculate gradients
        loss.backward();
        
        // Update parameters
        w = w.sub(w.grad().mul_scalar(lr));
        b = b.sub(b.grad().mul_scalar(lr));
        
        // Reset gradients
        w.zero_grad();
        b.zero_grad();
        
        if (epoch % 10 === 0) {
            console.log(`Epoch ${epoch}: Loss = ${loss.item()}`);
        }
    }
    
    console.log(`Final w: ${w.item()}, b: ${b.item()}`);
});
```

## Troubleshooting

### 🚀 Speed Up Rust Kernel (Recommended)
If initial execution is slow, enable caching for significant performance improvement:

```bash
# Create cache directory
mkdir -p ~/.config/evcxr

# Enable 500MB cache
echo ":cache 500" > ~/.config/evcxr/init.evcxr
```

**Effects:**
- First time: Normal compilation time
- Subsequent runs: No recompilation of dependencies (several times faster)
- `rustorch` library is also cached after first use

### Common Errors

#### "RusTorch is not defined" Error
**Solution**: Always wait for RusTorchReady
```javascript
window.RusTorchReady.then((rt) => {
    // Use RusTorch here
});
```

#### "Failed to load WASM module" Error
**Solutions**:
1. Verify `pkg` directory was generated correctly
2. Check browser console for error messages
3. Ensure WASM file paths are correct

#### Memory Shortage Error
**Solutions**:
```javascript
// Free memory explicitly
tensor.free();

// Use smaller batch sizes
const batchSize = 32;  // Use 32 instead of 1000
```

### Performance Tips

1. **Use Batch Processing**: Process data in batches instead of loops
2. **Memory Management**: Explicitly free large tensors
3. **Appropriate Data Types**: Use f32 when high precision isn't needed

## FAQ

### Q: Can I use this in Google Colab?
**A**: Yes, upload the WASM files and use custom JavaScript loaders.

### Q: Can I mix Python and WASM code?
**A**: Yes, use IPython.display.Javascript to pass data between Python and JavaScript.

### Q: How do I debug?
**A**: Use browser developer tools (F12) and check the Console tab for errors.

### Q: What advanced features are available?
**A**: Currently supports basic tensor operations, automatic differentiation, and simple neural networks. CNN and RNN layers are planned.

## Next Steps

1. 📖 [Detailed RusTorch WASM API](../wasm.md)
2. 🔬 [Advanced Examples](../examples/)
3. 🚀 [Performance Optimization Guide](../wasm-memory-optimization.md)

## Community and Support

- GitHub: [RusTorch Repository](https://github.com/JunSuzukiJapan/rustorch)
- Issues: Report bugs and request features on GitHub

---

Happy Learning with RusTorch WASM! 🦀🔥📓
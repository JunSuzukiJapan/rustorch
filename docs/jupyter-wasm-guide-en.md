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

### Step 1: Install Basic Tools

#### 1.1 Verify Python and pip
```bash
# Check Python version
python --version
# or
python3 --version

# Check pip
pip --version
# or
pip3 --version
```

#### 1.2 Install Jupyter Notebook
```bash
# Install Jupyter Notebook
pip install notebook

# Or install Jupyter Lab (more advanced version)
pip install jupyterlab
```

#### 1.3 Install Node.js
```bash
# macOS (using Homebrew)
brew install node

# Windows
# Download and install from https://nodejs.org

# Ubuntu
sudo apt update
sudo apt install nodejs npm

# Verify installation
node --version
npm --version
```

#### 1.4 Install Rust
```bash
# Install Rustup (official recommended method)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add to PATH after installation
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

#### 1.5 Install wasm-pack
```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Or use cargo
cargo install wasm-pack

# Verify installation
wasm-pack --version
```

### Step 2: Prepare RusTorch Project

#### 2.1 Clone RusTorch
```bash
# Create project directory
mkdir ~/rustorch-jupyter
cd ~/rustorch-jupyter

# Clone RusTorch
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch
```

#### 2.2 Build for WASM
```bash
# Add WASM target
rustup target add wasm32-unknown-unknown

# Build with wasm-pack
wasm-pack build --target web --out-dir pkg

# Verify build success (pkg directory should be created)
ls pkg/
# Expected output: rustorch.js  rustorch_bg.wasm  package.json  etc.
```

### Step 3: Configure Jupyter Environment

#### 3.1 Create Jupyter Kernel Extension

Create a new file `jupyter_setup.py`:

```python
# jupyter_setup.py
import os
import shutil
from pathlib import Path

def setup_jupyter_wasm():
    """Setup WASM environment for Jupyter"""
    
    # 1. Check Jupyter configuration directory
    jupyter_dir = Path.home() / '.jupyter'
    jupyter_dir.mkdir(exist_ok=True)
    
    # 2. Create custom directory
    custom_dir = jupyter_dir / 'custom'
    custom_dir.mkdir(exist_ok=True)
    
    # 3. Create custom.js file
    custom_js = custom_dir / 'custom.js'
    
    js_content = """
// RusTorch WASM auto-load configuration
require.config({
    paths: {
        'rustorch': '/files/rustorch/pkg/rustorch'
    }
});

// Make available as global variable
window.RusTorchReady = new Promise((resolve, reject) => {
    require(['rustorch'], function(rustorch) {
        rustorch.default().then(() => {
            window.RusTorch = rustorch;
            console.log('✅ RusTorch WASM loaded successfully!');
            resolve(rustorch);
        }).catch(reject);
    });
});
"""
    
    # 4. Write to file
    with open(custom_js, 'w') as f:
        f.write(js_content)
    
    print(f"✅ Jupyter configuration completed: {custom_js}")
    
    # 5. Create symbolic link (for development)
    notebook_dir = Path.home() / 'rustorch'
    if not notebook_dir.exists():
        current_dir = Path.cwd()
        notebook_dir.symlink_to(current_dir)
        print(f"✅ Created symbolic link: {notebook_dir} -> {current_dir}")

if __name__ == "__main__":
    setup_jupyter_wasm()
```

Run the setup:
```bash
python jupyter_setup.py
```

#### 3.2 Start Jupyter Server

```bash
# Start Jupyter Notebook
jupyter notebook

# Or start Jupyter Lab
jupyter lab
```

### Step 4: Verify Installation

#### 4.1 Open Jupyter in Browser

When you start the Jupyter server, it should automatically open in your browser. If not:

1. **Copy the URL displayed in terminal**
   ```
   [I 12:34:56.789 NotebookApp] Serving notebooks from local directory: /Users/username/rustorch-jupyter
   [I 12:34:56.789 NotebookApp] Jupyter Notebook 6.4.12 is running at:
   [I 12:34:56.789 NotebookApp] Local URL: http://localhost:8888/?token=abc123...
   ```

2. **Manually open in browser**
   - Copy the URL and paste it into your browser's address bar
   - Or navigate to `http://localhost:8888` and enter the token

3. **Recommended Browsers**
   - **Chrome**: Best WASM and WebGPU support ✅
   - **Firefox**: Stable WASM support ✅
   - **Safari**: Basic WASM support ⚠️
   - **Edge**: Good support (Chromium-based) ✅

#### 4.2 Create New Notebook

1. Click "New" → "Python 3" in Jupyter browser interface
2. A new notebook will open

#### 4.3 Initialize RusTorch WASM

Enter the following code in the first cell and execute (Shift + Enter):

```javascript
%%javascript
// Wait for RusTorch WASM to load
window.RusTorchReady.then((rustorch) => {
    console.log('RusTorch is ready!');
    
    // Check version
    const version = rustorch.get_version();
    console.log(`RusTorch version: ${version}`);
    
    // Simple test
    const tensor = rustorch.create_tensor([1, 2, 3, 4], [2, 2]);
    console.log('Created tensor:', tensor);
});
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
    // Create tensors
    const a = rt.create_tensor([1, 2, 3, 4], [2, 2]);
    const b = rt.create_tensor([5, 6, 7, 8], [2, 2]);
    
    // Addition
    const sum = a.add(b);
    console.log('A + B =', sum.to_array());
    
    // Matrix multiplication
    const product = a.matmul(b);
    console.log('A × B =', product.to_array());
    
    // Transpose
    const transposed = a.transpose();
    console.log('A^T =', transposed.to_array());
});
```

### Automatic Differentiation

```javascript
%%javascript
window.RusTorchReady.then((rt) => {
    // Create tensor with gradient tracking enabled
    const x = rt.create_tensor([2.0], null, true);  // requires_grad=true
    
    // Computation: y = x^2 + 3x + 1
    const x_squared = x.mul(x);
    const three_x = x.mul_scalar(3.0);
    const y = x_squared.add(three_x).add_scalar(1.0);
    
    // Backpropagation
    y.backward();
    
    // Get gradient (dy/dx = 2x + 3 = 7 when x=2)
    console.log('Gradient:', x.grad().to_array());
});
```

## Practical Examples

### Example 1: Linear Regression

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // Prepare data
    const X = rt.create_tensor([1, 2, 3, 4, 5]);
    const y = rt.create_tensor([2, 4, 6, 8, 10]);  // y = 2x
    
    // Initialize parameters
    let w = rt.create_tensor([0.5], null, true);
    let b = rt.create_tensor([0.0], null, true);
    
    // Learning rate
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

### Example 2: Neural Network

```javascript
%%javascript
window.RusTorchReady.then(async (rt) => {
    // Simple 2-layer neural network
    class SimpleNN {
        constructor(inputSize, hiddenSize, outputSize) {
            // Weight initialization (Xavier initialization)
            const scale1 = Math.sqrt(2.0 / inputSize);
            const scale2 = Math.sqrt(2.0 / hiddenSize);
            
            this.W1 = rt.randn([inputSize, hiddenSize]).mul_scalar(scale1);
            this.b1 = rt.zeros([hiddenSize]);
            this.W2 = rt.randn([hiddenSize, outputSize]).mul_scalar(scale2);
            this.b2 = rt.zeros([outputSize]);
            
            // Enable gradient tracking
            this.W1.requires_grad_(true);
            this.b1.requires_grad_(true);
            this.W2.requires_grad_(true);
            this.b2.requires_grad_(true);
        }
        
        forward(x) {
            // Layer 1: ReLU activation
            let h = x.matmul(this.W1).add(this.b1);
            h = h.relu();
            
            // Layer 2: Linear
            const output = h.matmul(this.W2).add(this.b2);
            return output;
        }
    }
    
    // Create model
    const model = new SimpleNN(2, 4, 1);
    
    // XOR dataset
    const X = rt.create_tensor([
        0, 0,
        0, 1,
        1, 0,
        1, 1
    ], [4, 2]);
    
    const y = rt.create_tensor([0, 1, 1, 0], [4, 1]);
    
    // Training
    const lr = 0.1;
    for (let epoch = 0; epoch < 1000; epoch++) {
        // Forward pass
        const output = model.forward(X);
        
        // Loss calculation
        const loss = output.sub(y).pow(2).mean();
        
        // Backward pass
        loss.backward();
        
        // Parameter updates
        model.W1 = model.W1.sub(model.W1.grad().mul_scalar(lr));
        model.b1 = model.b1.sub(model.b1.grad().mul_scalar(lr));
        model.W2 = model.W2.sub(model.W2.grad().mul_scalar(lr));
        model.b2 = model.b2.sub(model.b2.grad().mul_scalar(lr));
        
        // Reset gradients
        model.W1.zero_grad();
        model.b1.zero_grad();
        model.W2.zero_grad();
        model.b2.zero_grad();
        
        if (epoch % 100 === 0) {
            console.log(`Epoch ${epoch}: Loss = ${loss.item()}`);
        }
    }
    
    // Test
    const predictions = model.forward(X);
    console.log('Predictions:', predictions.to_array());
});
```

### Example 3: Integration with Data Visualization

```python
# Python cell: visualization with matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML, Javascript

# Generate data with JavaScript and pass to Python
display(Javascript("""
window.RusTorchReady.then((rt) => {
    // Generate data
    const x = rt.linspace(-5, 5, 100);
    const y = x.mul(x);  // y = x^2
    
    // Convert to JSON for Python
    const data = {
        x: x.to_array(),
        y: y.to_array()
    };
    
    // Send to Python using IPython.kernel
    IPython.notebook.kernel.execute(
        `plot_data = ${JSON.stringify(data)}`
    );
});
"""))
```

```python
# Next cell: Plot the data
import json
import time

# Wait a bit for data from JavaScript
time.sleep(1)

# Plot the data
if 'plot_data' in globals():
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data['x'], plot_data['y'])
    plt.title('y = x²')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
```

## Troubleshooting

### Common Errors and Solutions

#### 1. "RusTorch is not defined" Error

**Cause**: WASM module hasn't loaded yet

**Solution**:
```javascript
// Always wait for RusTorchReady
window.RusTorchReady.then((rt) => {
    // Use RusTorch here
});
```

#### 2. "Failed to load WASM module" Error

**Cause**: Incorrect WASM file path

**Solutions**:
1. Verify `pkg` directory was generated correctly
2. Check if `pkg/rustorch_bg.wasm` is visible in Jupyter file browser
3. Check browser console for error messages

#### 3. Memory Shortage Error

**Cause**: Attempting to create large tensors

**Solutions**:
```javascript
// Free memory explicitly
tensor.free();  // Explicitly free unused tensors

// Or use smaller batch sizes
const batchSize = 32;  // Use 32 instead of 1000
```

#### 4. Gradients Not Calculated

**Cause**: `requires_grad` not set

**Solution**:
```javascript
// Specify when creating tensor
const x = rt.create_tensor([1, 2, 3], null, true);  // requires_grad=true

// Or set later
x.requires_grad_(true);
```

### Performance Optimization Tips

#### 1. Use Batch Processing
```javascript
// Bad: Individual processing in loop
for (let i = 0; i < 1000; i++) {
    const result = tensor.mul_scalar(2.0);
}

// Good: Vectorized operations
const batch = rt.create_tensor(data, [1000, 10]);
const result = batch.mul_scalar(2.0);  // Process all at once
```

#### 2. Memory Management
```javascript
// Force garbage collection after large computations
if (typeof gc !== 'undefined') {
    gc();
}

// Explicitly free tensors
largeTensor.free();
```

#### 3. Use Appropriate Data Types
```javascript
// Use f32 when precision is not critical
const tensor_f32 = rt.create_tensor_f32(data);

// Use f64 only when high precision is needed
const tensor_f64 = rt.create_tensor_f64(data);
```

## FAQ

### Q1: Can I use this in Google Colab or Kaggle Notebooks?

**A**: Yes, but the following steps are required:

1. Upload WASM files
2. Configure custom JavaScript loader
3. Be aware of CORS restrictions

Detailed steps:
```python
# Google Colab setup
from google.colab import files
import os

# Upload WASM files
uploaded = files.upload()  # Select rustorch_bg.wasm and rustorch.js

# Display HTML and JavaScript
from IPython.display import HTML

HTML("""
<script type="module">
    import init, * as rustorch from './rustorch.js';
    
    await init();
    window.RusTorch = rustorch;
    console.log('RusTorch loaded in Colab!');
</script>
""")
```

### Q2: Can I mix Python and WASM code?

**A**: Yes, using several methods:

```python
# Prepare data in Python
import numpy as np
data = np.random.randn(100, 10).tolist()

# Pass to JavaScript
from IPython.display import Javascript
Javascript(f"""
window.pythonData = {data};
window.RusTorchReady.then((rt) => {{
    const tensor = rt.create_tensor(window.pythonData, [100, 10]);
    // Process...
}});
""")
```

### Q3: How should I debug?

**A**: Use browser developer tools:

1. **Chrome/Firefox**: Press F12 to open developer tools
2. **Console** tab: Check error messages
3. **Network** tab: Verify WASM file loading
4. **Source** tab: Set breakpoints

Debug helper functions:
```javascript
// Output debug information
function debugTensor(tensor, name) {
    console.log(`=== ${name} ===`);
    console.log('Shape:', tensor.shape());
    console.log('Data:', tensor.to_array());
    console.log('Requires grad:', tensor.requires_grad());
    console.log('Device:', tensor.device());
}
```

### Q4: Can I use advanced features (CNN, RNN, etc.)?

**A**: The current WASM version is limited to basic features:

1. **Available**: Basic tensor operations, automatic differentiation, simple NNs
2. **Limited**: GPU operations, large-scale models
3. **Planned**: CNN layers, RNN layers, optimization algorithms

### Q5: What if I get errors and it doesn't work?

Checklist:

1. ✅ Is Rust installed? `rustc --version`
2. ✅ Is wasm-pack installed? `wasm-pack --version`
3. ✅ Did WASM build succeed? `ls pkg/`
4. ✅ Is Jupyter up to date? `jupyter --version`
5. ✅ Is browser supported? (Chrome/Firefox/Safari recommended)

If still not resolved, create an Issue with the following information:
- OS and version
- Browser and version
- Full error message text
- Command execution history

## Next Steps

1. 📖 [Detailed RusTorch WASM API](./wasm.md)
2. 🔬 [Advanced Example Collection](../examples/)
3. 🚀 [Performance Optimization Guide](./wasm-memory-optimization.md)
4. 🧪 [Testing Methods](./wasm-testing.md)

## Community and Support

- GitHub: [RusTorch Repository](https://github.com/JunSuzukiJapan/rustorch)
- Discord: [RusTorch Community](https://discord.gg/rustorch)
- Stack Overflow: Use tag `rustorch-wasm`

---

Happy Learning with RusTorch WASM! 🦀🔥📓
# RusTorch WebGPU Chrome Demo

WebGPU加速テンソル演算のChrome ブラウザ向けデモンストレーション

## 🎯 Overview

This demo showcases RusTorch's WebGPU acceleration capabilities specifically optimized for Google Chrome browser. It demonstrates GPU-accelerated tensor operations including matrix multiplication, element-wise operations, and activation functions.

## 📋 Requirements

### Browser Requirements
- **Google Chrome 113+** (recommended: latest stable)
- **WebGPU enabled** in Chrome flags
- **Hardware GPU acceleration** enabled
- **Local web server** (not file:// protocol)

### System Requirements
- Modern GPU with compute shader support
- Sufficient GPU memory for tensor operations
- WebGPU-compatible graphics drivers

## 🔧 Setup Instructions

### 1. Enable WebGPU in Chrome

1. Open Chrome and navigate to: `chrome://flags/#enable-unsafe-webgpu`
2. Set "Unsafe WebGPU" to **Enabled**
3. Restart Chrome

### 2. Build the WASM Module

```bash
# Install wasm-pack (if not already installed)
cargo install wasm-pack

# Build RusTorch with WebGPU features for WASM
wasm-pack build --target web --features webgpu

# Alternative: Build specific example
cargo build --example webgpu_chrome_demo --target wasm32-unknown-unknown --features webgpu
```

### 3. Serve the Demo

```bash
# Using Python's built-in server
python3 -m http.server 8000

# Using Node.js http-server (npm install -g http-server)
http-server -p 8000

# Using Rust's basic-http-server (cargo install basic-http-server)
basic-http-server -a 127.0.0.1:8000
```

### 4. Open the Demo

1. Navigate to: `http://localhost:8000/examples/webgpu_demo.html`
2. Open Chrome Developer Console (F12) to see detailed logs
3. Click "Initialize WebGPU Engine" to start

## 🎮 Demo Features

### WebGPU Detection
- Automatic WebGPU support detection
- GPU adapter information display
- Chrome version compatibility check

### Tensor Operations
- **Addition**: Element-wise tensor addition with GPU acceleration
- **Matrix Multiplication**: Optimized matrix multiplication using compute shaders
- **Activation Functions**: ReLU and Sigmoid with GPU compute
- **Memory Management**: Efficient GPU buffer management

### Performance Testing
- Comparative performance analysis (GPU vs CPU estimates)
- Multiple tensor sizes testing (100, 1000, 10000 elements)
- Real-time performance metrics display

### Chrome Optimizations
- Workgroup size optimization for Chrome's WebGPU implementation
- Buffer size optimization for Chrome's memory management
- Chrome-specific GPU adapter selection

## 🚀 Performance Expectations

| Operation | Small Tensors | Large Tensors | Expected Speedup |
|-----------|---------------|---------------|------------------|
| Addition | 1.2x | 2.0x | Moderate |
| Multiplication | 1.2x | 2.0x | Moderate |
| Matrix Multiplication | 1.5x | 10.0x | Significant |
| ReLU | 1.5x | 3.0x | Good |
| Sigmoid | 1.5x | 3.0x | Good |

## 🐛 Troubleshooting

### Common Issues

**"WebGPU not supported"**
- Ensure Chrome 113+ with WebGPU flags enabled
- Check hardware GPU acceleration is available
- Try updating graphics drivers

**"Failed to initialize WebGPU engine"**
- Check GPU memory availability
- Close other GPU-intensive applications
- Try refreshing the page

**"WASM module failed to load"**
- Ensure serving from HTTP server (not file://)
- Check WASM build completed successfully
- Verify all dependencies are available

**"Demo functions not working"**
- Ensure WebGPU engine initialization completed
- Check browser console for detailed error messages
- Try the comprehensive demo function

### Performance Issues

**Slower than expected performance**
- Check if hardware acceleration is actually enabled
- Monitor GPU usage in Chrome Task Manager
- Ensure sufficient GPU memory is available
- Try smaller tensor sizes for testing

## 🔍 Debugging

### Chrome Developer Tools
1. Open DevTools (F12)
2. Go to Console tab for execution logs
3. Check Network tab for WASM loading
4. Use Performance tab for detailed profiling

### GPU Monitoring
1. Chrome Task Manager: `Shift+Esc` → GPU Process
2. Check GPU memory usage and utilization
3. Monitor for GPU compute activity during operations

## 📊 Understanding the Output

The demo provides detailed console output including:
- WebGPU adapter information (GPU name, backend type)
- Tensor operation input/output data
- Performance timing measurements  
- Verification of computational accuracy
- Resource usage statistics

## 🔬 Technical Details

### WGSL Compute Shaders
The demo uses optimized WGSL (WebGPU Shading Language) compute shaders:
- Workgroup size optimization for Chrome
- Memory coalescing for efficient GPU memory access
- Bounds checking for safe GPU computation

### Buffer Management
- Automatic GPU buffer allocation and deallocation
- Efficient CPU-GPU data transfer
- Memory usage optimization for Chrome's WebGPU implementation

### Chrome-Specific Optimizations
- Power preference: High Performance
- Backend selection: Browser WebGPU (not Dawn)
- Memory hints: Performance-optimized
- Workgroup sizing tuned for Chrome's GPU scheduler

## 🆘 Support

If you encounter issues:
1. Check Chrome WebGPU compatibility: `chrome://gpu/`
2. Verify WASM build: Check for `pkg/` directory with generated files
3. Review browser console for detailed error messages
4. Test with smaller tensor sizes first
5. Report issues with system information and Chrome version
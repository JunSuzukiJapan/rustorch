# RusTorch WebAssembly Examples

This directory contains examples demonstrating how to use RusTorch in WebAssembly environments.

## Prerequisites

1. **Install wasm-pack**:
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

2. **Install Node.js** (for Node.js examples):
   - Download from [nodejs.org](https://nodejs.org/)

## Building

Build the WebAssembly module:

```bash
# For web browsers
wasm-pack build --target web --out-dir examples/pkg

# For Node.js
wasm-pack build --target nodejs --out-dir examples/pkg-node
```

## Examples

### 1. Basic Browser Example (`wasm_basic.html`)

A comprehensive HTML page demonstrating all RusTorch WASM features:

- Tensor operations (add, multiply, matrix multiplication)
- Activation functions (ReLU, Sigmoid)
- Neural network construction and forward pass
- Loss function computation
- Performance benchmarking
- Memory management
- System information

**Usage**:
```bash
# Start a local server
cd examples
python -m http.server 8000

# Open in browser
open http://localhost:8000/wasm_basic.html
```

### 2. Neural Network Demo (`wasm_neural_network.js`)

A Node.js script demonstrating:

- XOR neural network creation and evaluation
- Tensor operations
- Performance benchmarking
- Memory management
- System information retrieval

**Usage**:
```bash
cd examples
npm install
npm run demo
```

## API Overview

### Core Classes

#### `WasmTensor`
```javascript
// Create tensor
const tensor = new WasmTensor([1, 2, 3, 4], [2, 2]);

// Operations
const sum = tensor1.add(tensor2);
const product = tensor1.multiply(tensor2);
const result = tensor1.matmul(tensor2);

// Activations
const relu = tensor.relu();
const sigmoid = tensor.sigmoid();

// Statistics
const mean = tensor.mean();
const sum = tensor.sum();
```

#### `WasmModel`
```javascript
// Create model
const model = new WasmModel();
model.add_linear(4, 8, true);  // 4 inputs, 8 outputs, with bias
model.add_relu();
model.add_linear(8, 1, true);

// Forward pass
const output = model.forward(input);
```

#### `JsInterop`
```javascript
const interop = new JsInterop();

// Create tensors from JS arrays
const tensor = interop.tensor_from_array(data, shape);

// Utility functions
const zeros = interop.zeros(shape);
const ones = interop.ones(shape);
const random = interop.random_tensor(shape, min, max);

// Performance
const results = interop.benchmark_operations(1000);
```

#### Memory Management
```javascript
const memory = new JsMemoryManager();

// Check usage
const usage = memory.get_memory_usage_mb();
const utilization = memory.get_memory_utilization();

// Garbage collection
const freed = memory.garbage_collect();

// Recommendations
const advice = memory.get_memory_recommendations();
```

## Performance Considerations

### Memory Management
- WASM has a 256MB default memory limit
- Use `garbage_collect()` to free unused memory
- Monitor memory usage with `JsMemoryManager`

### Optimization Tips
- Batch operations when possible
- Reuse tensors instead of creating new ones
- Use appropriate data types (f32 vs f64)
- Consider tensor shapes for optimal performance

### Browser Compatibility
- Requires modern browsers with WASM support
- SIMD operations require additional browser support
- SharedArrayBuffer needed for threading (if enabled)

## Troubleshooting

### Common Issues

1. **WASM file not found**:
   - Ensure `pkg/` directory exists after building
   - Check file paths in imports

2. **Memory limit exceeded**:
   - Reduce batch sizes
   - Call `garbage_collect()` more frequently
   - Consider model complexity

3. **Performance issues**:
   - Check browser's WASM optimization level
   - Monitor memory fragmentation
   - Use performance profiling tools

### Browser Developer Tools
- Use Console to see RusTorch logs
- Monitor memory in Performance tab
- Check Network tab for WASM loading issues

## Examples Output

### Tensor Operations
```
Tensor1 (ones): [1, 1, 1, 1]
Tensor2 (random): [0.234, 0.678, 0.123, 0.890]
Sum: [1.234, 1.678, 1.123, 1.890]
Product: [0.234, 0.678, 0.123, 0.890]
```

### Neural Network
```
Network architecture: 4 → 8 (ReLU) → 1
Input:  [1.0, 0.5, -0.3, 0.8]
Output: [0.1234]
Layers: 3
```

### Performance
```
Matrix Multiplication (100 iterations): 15.23 ms
Element-wise Addition (100 iterations): 2.45 ms
ReLU Activation (100 iterations): 1.87 ms
Operations/second: 6570
```

## Further Reading

- [WebAssembly Documentation](https://webassembly.org/)
- [wasm-bindgen Book](https://rustwasm.github.io/wasm-bindgen/)
- [RusTorch API Documentation](../docs/)

## Contributing

Feel free to add more examples or improve existing ones. Make sure to:

1. Test in both browser and Node.js environments
2. Include error handling
3. Document any new APIs used
4. Follow the existing code style
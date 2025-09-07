# RusTorch Jupyter Lab Integration
# RusTorch Jupyter Lab統合

Complete Jupyter Lab integration for RusTorch WASM, enabling interactive machine learning development in browser environments.

ブラウザ環境でのインタラクティブな機械学習開発を可能にする、RusTorch WASMの完全なJupyter Lab統合。

## Features / 機能

- 🎓 **Jupyter Lab Native**: Optimized for Jupyter Lab environments
- 🚀 **WebGPU Acceleration**: Chrome-optimized GPU computing with CPU fallback
- 🧮 **Complete Tensor API**: Full tensor operations with visualization
- 📊 **Built-in Benchmarking**: Performance analysis utilities
- 🎨 **Interactive Display**: Tensor visualization in notebook cells
- 🔧 **Auto-initialization**: Handles WASM loading and WebGPU detection
- 🌐 **Cross-Environment**: Works with local Jupyter, JupyterHub, and cloud environments

## Quick Start / クイックスタート

### 🚀 Universal Installer (Recommended) / 万能インストーラー（推奨）

The easiest way to set up any RusTorch Jupyter environment:

最も簡単な方法でRusTorch Jupyter環境をセットアップ：

```bash
curl -sSL https://raw.githubusercontent.com/JunSuzukiJapan/rustorch/main/install_jupyter.sh | bash
```

**What you get / 取得できるもの:**
- 🦀🐍 **Hybrid Environment**: Python + Rust dual-kernel
- 🔍 **Auto-Detection**: Optimal setup for your hardware
- 📦 **Global Command**: `rustorch-jupyter` works from anywhere
- 📓 **Sample Notebooks**: Ready-to-use examples

### 🌐 WebGPU Setup (Advanced) / WebGPUセットアップ（上級者向け）

For browser GPU acceleration specifically:

ブラウザGPU加速専用の場合：

```bash
# Build RusTorch WASM package
cd rustorch
wasm-pack build --target web --features wasm,webgpu

# Copy files to Jupyter directory
cp pkg/* jupyter/
```

### 2. Basic Usage / 基本的な使用方法

```javascript
// Load RusTorch in Jupyter cell
const script = document.createElement('script');
script.src = './rustorch_jupyter.js';
document.head.appendChild(script);
await new Promise(resolve => { script.onload = resolve; });

// Initialize
const rustorch = new RusTorchJupyter();
await rustorch.initialize();

// Create and operate on tensors
const tensor = await rustorch.createTensor([1, 2, 3, 4], [2, 2]);
const utils = await rustorch.utils();
utils.display(tensor, 'My Tensor');
```

### 3. WebGPU Acceleration / WebGPU加速

```javascript
// Check and use WebGPU (Chrome 113+ required)
const webgpu = await rustorch.webgpu();
if (webgpu) {
    const result = webgpu.tensor_add_cpu([1,2,3], [4,5,6]);
    console.log('🚀 WebGPU result:', result);
} else {
    console.log('⚠️ Using CPU fallback');
}
```

## Examples / 例

### Basic Tensor Operations / 基本テンソル演算
- **File**: `examples/basic_tensor_operations.ipynb`
- **Content**: Tensor creation, mathematical operations, display utilities
- **Level**: Beginner
- **Duration**: 10-15 minutes

### Advanced ML Training / 高度なML学習
- **File**: `examples/advanced_ml_training.ipynb`
- **Content**: Neural networks, optimizers, training loops, dataset generation
- **Level**: Intermediate
- **Duration**: 20-30 minutes

### WebGPU Acceleration / WebGPU加速
- **File**: `examples/webgpu_acceleration.ipynb`
- **Content**: WebGPU setup, performance comparison, GPU-accelerated operations
- **Level**: Advanced
- **Duration**: 15-25 minutes

## API Reference / APIリファレンス

### RusTorchJupyter Class

#### Core Methods / コアメソッド

```javascript
// Initialization
await rustorch.initialize()

// Tensor operations
await rustorch.createTensor(data, shape)

// Mathematical functions
const math = await rustorch.mathFunctions()
math.gamma(values)
math.erf(values)
math.bessel_i(n, values)

// Statistical distributions
const dist = await rustorch.distributions()
dist.normal(count, mean, std)
dist.uniform(count, low, high)

// Neural networks
const nn = await rustorch.neuralNetwork()
nn.createLinear(input_size, output_size, bias)
nn.createConv2d(in_channels, out_channels, kernel_size)

// Optimizers
const opt = await rustorch.optimizers()
opt.sgd(learning_rate, momentum)
opt.adam(learning_rate, beta1, beta2, eps)

// WebGPU acceleration
const webgpu = await rustorch.webgpu()
webgpu.tensor_add_cpu(a, b)
webgpu.matrix_multiply_cpu(a, b, m, n, k)

// Utilities
const utils = await rustorch.utils()
utils.display(tensor, title)
utils.benchmark(operation, iterations)
utils.checkCapabilities()
```

## Browser Requirements / ブラウザ要件

### Minimum Requirements / 最小要件
- **Chrome**: 69+ (WASM support)
- **Firefox**: 52+ (WASM support)
- **Safari**: 11+ (WASM support)
- **Edge**: 79+ (Chromium-based)

### WebGPU Requirements / WebGPU要件
- **Chrome**: 113+ with flags enabled
- **Edge**: 113+ (Chromium-based)
- **Firefox**: 113+ (experimental, requires configuration)
- **Safari**: Not supported (CPU fallback automatic)

### Chrome WebGPU Setup / Chrome WebGPU設定

1. Navigate to `chrome://flags/`
2. Enable these flags:
   - `#enable-unsafe-webgpu` → **Enabled**
   - `#enable-webgpu-developer-features` → **Enabled**
3. Restart Chrome
4. Verify: DevTools → Console → `!!navigator.gpu`

## Troubleshooting / トラブルシューティング

### Common Issues / よくある問題

**1. WASM Module Loading Fails**
```javascript
// Check file paths
console.log('Current location:', window.location.href);
// Adjust script.src path accordingly
```

**2. WebGPU Not Available**
```javascript
// Verify browser support
console.log('WebGPU available:', !!navigator.gpu);
// Check Chrome flags or try different browser
```

**3. Performance Issues**
```javascript
// Monitor memory usage
if (performance.memory) {
    console.log('Memory usage:', {
        used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024) + 'MB',
        total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024) + 'MB'
    });
}
```

**4. Tensor Display Issues**
```javascript
// Fallback display method
console.log('Tensor data:', tensor.as_slice());
console.log('Tensor shape:', tensor.shape());
```

## Performance Expectations / パフォーマンス期待値

| Operation | Size | CPU Time | WebGPU Time | Speedup |
|-----------|------|----------|-------------|---------|
| **Tensor Add** | 1K | ~0.2ms | ~0.1ms | 2x |
| **Tensor Add** | 10K | ~1.5ms | ~0.5ms | 3x |
| **Matrix Mul** | 256x256 | ~20ms | ~2ms | 10x |
| **Matrix Mul** | 512x512 | ~160ms | ~8ms | 20x |

*Performance varies by hardware and browser*

## Best Practices / ベストプラクティス

### 1. Resource Management / リソース管理
```javascript
// Always check capabilities first
const capabilities = await utils.checkCapabilities();
console.log('Available features:', capabilities);
```

### 2. Error Handling / エラーハンドリング
```javascript
try {
    const result = await rustorch.createTensor(data, shape);
    utils.display(result);
} catch (error) {
    console.error('Operation failed:', error.message);
}
```

### 3. Performance Optimization / パフォーマンス最適化
```javascript
// Use WebGPU for large operations, CPU for small ones
const size = data.length;
const backend = size > 1000 ? 'webgpu' : 'cpu';
console.log(`Using ${backend} backend for size ${size}`);
```

### 4. Memory Management / メモリ管理
```javascript
// For large datasets, process in batches
const batch_size = 1000;
for (let i = 0; i < data.length; i += batch_size) {
    const batch = data.slice(i, i + batch_size);
    // Process batch...
}
```

## Contributing / 貢献

To contribute to Jupyter Lab integration:

1. Test in different Jupyter environments
2. Report browser compatibility issues
3. Suggest new notebook examples
4. Improve visualization utilities
5. Enhance documentation

---

**🎓 Ready to use RusTorch in Jupyter Lab?**  
**Jupyter LabでRusTorchを使う準備はできましたか？**

Start with `examples/basic_tensor_operations.ipynb` for hands-on experience!  
実践的な体験には`examples/basic_tensor_operations.ipynb`から始めてください！
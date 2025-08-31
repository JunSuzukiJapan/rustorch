# RusTorch WebAssembly (WASM) Complete Guide
# RusTorch WebAssembly (WASM) 完全ガイド

[![WASM Status](https://img.shields.io/badge/WASM-production%20ready-brightgreen.svg)](#)
[![Browser Support](https://img.shields.io/badge/browsers-Chrome%20113%2B%2C%20Firefox%20113%2B%2C%20Safari%2016%2B-blue.svg)](#)
[![WebGPU Support](https://img.shields.io/badge/WebGPU-Chrome%20optimized-orange.svg)](#)

## Overview / 概要

RusTorch provides comprehensive WebAssembly support for running machine learning operations directly in web browsers. This includes neural network training, inference, computer vision pipelines, and GPU acceleration via WebGPU.

RusTorchは、Webブラウザで直接機械学習演算を実行するための包括的なWebAssemblyサポートを提供します。ニューラルネットワークの学習、推論、コンピュータビジョンパイプライン、WebGPUによるGPU加速が含まれます。

## Quick Start / クイックスタート

### 1. Basic WASM Build / 基本WASMビルド

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web browsers
wasm-pack build --target web --features wasm

# Build with WebGPU support (Chrome optimized)
wasm-pack build --target web --features webgpu
```

### 2. Browser Integration / ブラウザ統合

```html
<!DOCTYPE html>
<html>
<head>
    <title>RusTorch WASM Demo</title>
</head>
<body>
    <script type="module">
        import init, { WasmTensor, WasmLinear } from './pkg/rustorch.js';
        
        async function run() {
            await init();
            
            // Create tensor
            const tensor = new WasmTensor([1.0, 2.0, 3.0], [3]);
            console.log("Tensor created:", tensor.data());
            
            // Create neural network layer
            const linear = new WasmLinear(3, 1, true);
            console.log("Linear layer created");
        }
        
        run();
    </script>
</body>
</html>
```

## Available Features / 利用可能機能

### Core Tensor Operations / コアテンソル演算

```javascript
import { WasmTensor, WasmTensorOps } from './pkg/rustorch.js';

// Basic operations
const a = new WasmTensor([1, 2, 3, 4], [2, 2]);
const b = new WasmTensor([5, 6, 7, 8], [2, 2]);
const result = WasmTensorOps.add(a, b);

// Mathematical functions
const exp_result = WasmTensorOps.exp(a);
const sin_result = WasmTensorOps.sin(a);
```

### Neural Network Layers / ニューラルネットワーク層

```javascript
import { WasmLinear, WasmReLU, WasmConv2d } from './pkg/rustorch.js';

// Linear layer
const linear = new WasmLinear(784, 128, true);

// Activation functions
const relu = new WasmReLU();

// Convolutional layer
const conv = new WasmConv2d(3, 64, 3, 1, 1, true);
```

### Automatic Differentiation / 自動微分

```javascript
import { VariableWasm, ComputationGraphWasm } from './pkg/rustorch.js';

// Create variables with gradient tracking
const x = new VariableWasm([2.0, 3.0], [2], true);
const y = new VariableWasm([4.0, 5.0], [2], true);

// Build computation graph
const graph = new ComputationGraphWasm();
const result_id = graph.add_operation(x_id, y_id, "add");

// Compute gradients
graph.backward(result_id);
console.log("Gradient:", x.grad());
```

### Special Mathematical Functions / 特殊数学関数

```javascript
import { 
    gamma_wasm, bessel_j_wasm, erf_wasm,
    gamma_array_wasm, bessel_j_array_wasm 
} from './pkg/rustorch.js';

// Single value functions
const gamma_val = gamma_wasm(5.0);
const bessel_val = bessel_j_wasm(1, 2.0);
const erf_val = erf_wasm(1.0);

// Vectorized operations
const gamma_array = gamma_array_wasm([1.0, 2.0, 3.0, 4.0]);
const bessel_array = bessel_j_array_wasm(0, [0.5, 1.0, 1.5, 2.0]);
```

### Statistical Distributions / 統計分布

```javascript
import { 
    NormalDistributionWasm, 
    GammaDistributionWasm,
    BetaDistributionWasm 
} from './pkg/rustorch.js';

// Normal distribution
const normal = new NormalDistributionWasm(0.0, 1.0);
const samples = normal.sample_batch(1000);
const pdf_vals = normal.pdf_batch([0.0, 1.0, -1.0]);

// Gamma distribution
const gamma = new GammaDistributionWasm(2.0, 1.0);
const gamma_samples = gamma.sample_batch(500);
```

### Computer Vision Pipeline / コンピュータビジョンパイプライン

```javascript
import { 
    WasmPreprocessor, 
    WasmResize, 
    WasmNormalize 
} from './pkg/rustorch.js';

// Create preprocessing pipeline
const preprocessor = new WasmPreprocessor();
preprocessor.add_transform("resize", "224,224");
preprocessor.add_transform("normalize", "imagenet");

// Process image tensor
const processed = preprocessor.apply(image_tensor);
```

### Optimizers / オプティマイザー

```javascript
import { SGDWasm, AdamWasm, RMSpropWasm } from './pkg/rustorch.js';

// SGD optimizer
const sgd = new SGDWasm(0.01, 0.9);
sgd.add_param_group(layer_params);

// Adam optimizer
const adam = new AdamWasm(0.001, 0.9, 0.999, 1e-8);
adam.add_param_group(layer_params);
```

## WebGPU Integration / WebGPU統合

### Chrome-Optimized WebGPU / Chrome最適化WebGPU

```javascript
import { WebGPUSimple, WebGPUSimpleDemo } from './pkg/rustorch.js';

// Initialize WebGPU engine
const engine = new WebGPUSimple();
const init_result = await engine.initialize();

// Check WebGPU support
const webgpu_supported = await engine.check_webgpu_support();

// Run tensor operations with GPU acceleration
const a = [1.0, 2.0, 3.0, 4.0];
const b = [5.0, 6.0, 7.0, 8.0];
const result = engine.tensor_add_cpu(a, b); // CPU fallback if WebGPU unavailable
```

### Interactive Demo / インタラクティブデモ

```javascript
// Comprehensive demo runner
const demo = new WebGPUSimpleDemo();
await demo.initialize();

// Run individual demos
const add_result = demo.run_tensor_addition_demo();
const matmul_result = demo.run_matrix_multiplication_demo();
const activation_result = demo.run_activation_functions_demo();
const benchmark_result = demo.run_performance_benchmark();

// Run everything
const comprehensive_result = await demo.run_comprehensive_demo();
```

## Browser Compatibility / ブラウザ互換性

### Supported Browsers / サポートブラウザ

| Browser | Version | WASM Support | WebGPU Support | Notes |
|---------|---------|--------------|----------------|-------|
| Chrome | 113+ | ✅ Full | ✅ Optimized | 推奨ブラウザ |
| Firefox | 113+ | ✅ Full | ⚠️ Experimental | WebGPU要フラグ有効化 |
| Safari | 16+ | ✅ Full | ❌ Not supported | CPU演算のみ |
| Edge | 113+ | ✅ Full | ✅ Supported | Chrome基盤 |

### WebGPU Requirements / WebGPU要件

```bash
# Chrome flags to enable WebGPU
chrome://flags/#enable-unsafe-webgpu -> Enabled
chrome://flags/#enable-webgpu-developer-features -> Enabled

# Hardware requirements
- Dedicated GPU (Intel/NVIDIA/AMD)
- Hardware acceleration enabled
- Updated graphics drivers
```

## Performance Characteristics / パフォーマンス特性

### Operation Performance / 演算パフォーマンス

| Operation | Small (100) | Medium (1K) | Large (10K) | WebGPU Speedup |
|-----------|-------------|-------------|-------------|----------------|
| Addition | 1.2x | 1.2x | 2.0x | CPU競合～中程度向上 |
| Matrix Multiply | 1.5x | 10.0x | 10.0x | 大幅向上 |
| Activation | 1.5x | 3.0x | 3.0x | 中程度向上 |

### Memory Usage / メモリ使用量

- **WASM Heap**: 自動拡張、最大4GB
- **WebGPU Buffers**: GPU専用メモリ使用
- **CPU Fallback**: 標準ヒープメモリ使用

## Advanced Usage / 高度な使用法

### Model Training Pipeline / モデル学習パイプライン

```javascript
import { 
    WasmModel, WasmSGD, WasmLoss, 
    VariableWasm, ComputationGraphWasm 
} from './pkg/rustorch.js';

// Create model
const model = new WasmModel();
model.add_layer("linear", 784, 128);
model.add_layer("relu");
model.add_layer("linear", 128, 10);

// Setup training
const optimizer = new WasmSGD(0.01, 0.9);
const loss_fn = new WasmLoss("cross_entropy");

// Training loop
for (let epoch = 0; epoch < 100; epoch++) {
    const output = model.forward(input_batch);
    const loss = loss_fn.compute(output, target_batch);
    
    model.zero_grad();
    loss.backward();
    optimizer.step();
}
```

### Quality Assessment / 品質評価

```javascript
import { WasmQualityMetrics } from './pkg/rustorch.js';

const quality = new WasmQualityMetrics();
const assessment = quality.comprehensive_quality_assessment(tensor);
console.log("Quality Score:", assessment.overall_score);
console.log("Anomalies:", assessment.anomaly_count);
```

### Performance Monitoring / パフォーマンス監視

```javascript
import { WasmPerformance } from './pkg/rustorch.js';

const perf = new WasmPerformance();
perf.start_profiling();

// Run operations
const result = model.forward(input);

const report = perf.get_performance_report();
console.log("Execution time:", report.total_time_ms);
console.log("Memory usage:", report.peak_memory_mb);
```

## Deployment Guide / デプロイメントガイド

### Production Deployment / 本番デプロイメント

```bash
# Optimized production build
wasm-pack build --target web --features webgpu --release

# File structure for deployment
your-website/
├── index.html
├── pkg/
│   ├── rustorch.js
│   ├── rustorch_bg.wasm
│   ├── rustorch.d.ts
│   └── package.json
└── assets/
    └── models/
```

### CDN Integration / CDN統合

```html
<!-- Include from CDN (when available) -->
<script type="module">
    import init, { WasmTensor } from 'https://cdn.jsdelivr.net/npm/rustorch-wasm/rustorch.js';
    await init();
    // Your code here
</script>
```

### Bundle Size Optimization / バンドルサイズ最適化

```javascript
// Selective feature imports
import init, { 
    WasmTensor,          // Core: ~50KB
    WasmLinear,          // Neural networks: ~30KB
    WebGPUSimple         // WebGPU: ~80KB
} from './pkg/rustorch.js';

// Avoid importing unused features to reduce bundle size
```

## Troubleshooting / トラブルシューティング

### Common Issues / よくある問題

**1. WASM Module Loading Fails**
```javascript
// Check MIME type support
fetch('./pkg/rustorch_bg.wasm')
    .then(response => {
        console.log('WASM MIME type:', response.headers.get('content-type'));
        // Should be: application/wasm
    });
```

**2. WebGPU Not Available**
```javascript
// Check WebGPU support
if (!navigator.gpu) {
    console.log('WebGPU not supported, using CPU fallback');
    // Use CPU-only operations
}
```

**3. Memory Issues**
```javascript
// Monitor memory usage
const memory = performance.memory;
console.log('Used:', memory.usedJSHeapSize / 1024 / 1024, 'MB');
console.log('Total:', memory.totalJSHeapSize / 1024 / 1024, 'MB');
```

### Debug Mode / デバッグモード

```bash
# Build with debug symbols
wasm-pack build --target web --features wasm --dev

# Enable WASM debugging in browser
// Chrome DevTools -> Sources -> Enable WASM debugging
```

## Performance Best Practices / パフォーマンスベストプラクティス

### 1. Memory Management / メモリ管理

```javascript
// Explicit cleanup for large tensors
tensor.free(); // Free WASM memory immediately

// Use memory pools for frequent allocations
const pool = new WasmTensorPool();
const tensor = pool.get_tensor([1024, 1024]);
// ... use tensor
pool.return_tensor(tensor);
```

### 2. Batch Operations / バッチ演算

```javascript
// Prefer batch operations over loops
const batch_result = model.forward_batch(input_batch); // ✅ Good
// vs
// input_batch.map(x => model.forward(x)); // ❌ Inefficient
```

### 3. WebGPU Optimization / WebGPU最適化

```javascript
// Check operation size for WebGPU efficiency
if (tensor_size > 1000) {
    // Use WebGPU for large operations
    result = engine.tensor_add_webgpu(a, b);
} else {
    // Use CPU for small operations
    result = engine.tensor_add_cpu(a, b);
}
```

## Feature Matrix / 機能マトリックス

### Basic Features / 基本機能

| Feature | CPU | WebGPU | Status | Performance |
|---------|-----|--------|--------|-------------|
| Tensor Operations | ✅ | ✅ | Stable | High |
| Broadcasting | ✅ | ⚠️ | Limited | Medium |
| Indexing | ✅ | ❌ | CPU only | High |
| Mathematical Functions | ✅ | ⚠️ | Partial | High |

### Advanced Features / 高度機能

| Feature | CPU | WebGPU | Status | Notes |
|---------|-----|--------|--------|-------|
| Autograd | ✅ | ❌ | CPU only | Single-threaded |
| Neural Networks | ✅ | ⚠️ | Partial | Layer-dependent |
| Optimizers | ✅ | ❌ | CPU only | Full support |
| Computer Vision | ✅ | ⚠️ | Limited | Transform-dependent |

### Specialized Features / 特殊機能

| Feature | Support | Performance | Notes |
|---------|---------|-------------|-------|
| Special Functions | ✅ Full | High | Gamma, Bessel, Error functions |
| Statistical Distributions | ✅ Full | High | Normal, Gamma, Beta, etc. |
| Quality Assessment | ✅ Full | Medium | Anomaly detection, validation |
| Performance Profiling | ✅ Full | Low overhead | Built-in monitoring |

## API Reference / API リファレンス

### Core Classes / コアクラス

#### WasmTensor
```typescript
class WasmTensor {
    constructor(data: number[], shape: number[]);
    data(): number[];
    shape(): number[];
    reshape(new_shape: number[]): WasmTensor;
    clone(): WasmTensor;
    free(): void;
}
```

#### WebGPUSimple
```typescript
class WebGPUSimple {
    constructor();
    async initialize(): Promise<string>;
    async check_webgpu_support(): Promise<boolean>;
    tensor_add_cpu(a: number[], b: number[]): number[];
    matrix_multiply_cpu(a: number[], b: number[], m: number, n: number, k: number): number[];
    get_chrome_info(): string;
}
```

#### VariableWasm (Autograd)
```typescript
class VariableWasm {
    constructor(data: number[], shape: number[], requires_grad: boolean);
    data(): number[];
    grad(): number[] | null;
    zero_grad(): void;
    backward(): void;
}
```

### Utility Functions / ユーティリティ関数

```typescript
// Performance estimation
function calculate_performance_estimate(operation: string, size: number): number;

// Browser information
function get_browser_webgpu_info(): string;

// Mathematical functions
function gamma_wasm(x: number): number;
function erf_wasm(x: number): number;
function bessel_j_wasm(n: number, x: number): number;
```

## Examples / 実例

### 1. Simple Neural Network / シンプルニューラルネットワーク

```javascript
import init, { 
    WasmTensor, WasmLinear, WasmReLU, 
    WasmSGD, WasmLoss 
} from './pkg/rustorch.js';

await init();

// Create network
const fc1 = new WasmLinear(784, 128, true);
const relu = new WasmReLU();
const fc2 = new WasmLinear(128, 10, true);

// Forward pass
let x = new WasmTensor(input_data, [batch_size, 784]);
x = fc1.forward(x);
x = relu.forward(x);
x = fc2.forward(x);

console.log("Output shape:", x.shape());
```

### 2. Image Classification Pipeline / 画像分類パイプライン

```javascript
import { 
    WasmVision, 
    WasmPreprocessor,
    WasmModel 
} from './pkg/rustorch.js';

// Setup vision pipeline
const vision = new WasmVision();
const preprocessor = vision.create_imagenet_preprocessing();

// Load and preprocess image
const image_tensor = vision.load_image_from_canvas(canvas);
const processed = preprocessor.apply(image_tensor);

// Run inference
const model = new WasmModel();
model.load_weights(model_weights);
const predictions = model.forward(processed);
```

### 3. Real-time Anomaly Detection / リアルタイム異常検知

```javascript
import { WasmTimeSeriesDetector } from './pkg/rustorch.js';

const detector = new WasmTimeSeriesDetector(100, 2.0); // window_size, threshold

// Process streaming data
function processDataPoint(value) {
    const is_anomaly = detector.detect_single(value);
    
    if (is_anomaly) {
        console.log('Anomaly detected:', value);
        // Handle anomaly
    }
}
```

## Development Workflow / 開発ワークフロー

### 1. Development Setup / 開発セットアップ

```bash
# Clone repository
git clone https://github.com/JunSuzukiJapan/rustorch.git
cd rustorch

# Install dependencies
cargo build

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### 2. Testing / テスト

```bash
# Run WASM tests
wasm-pack test --chrome --headless --features wasm

# Run WebGPU tests
wasm-pack test --chrome --features webgpu

# Serve demo locally
cd examples && python3 -m http.server 8080
```

### 3. Debugging / デバッグ

```javascript
// Enable debug logging
console.log = function(...args) {
    // Custom logging implementation
    originalLog.apply(console, args);
    // Send to analytics, etc.
};

// Memory usage monitoring
setInterval(() => {
    if (performance.memory) {
        console.log('Memory usage:', performance.memory.usedJSHeapSize / 1024 / 1024, 'MB');
    }
}, 5000);
```

## Integration Examples / 統合例

### React Integration / React統合

```javascript
import React, { useEffect, useState } from 'react';
import init, { WasmTensor, WasmLinear } from './pkg/rustorch.js';

function MLComponent() {
    const [rustorch, setRustorch] = useState(null);
    
    useEffect(() => {
        async function initRustorch() {
            await init();
            const linear = new WasmLinear(10, 1, true);
            setRustorch({ linear });
        }
        initRustorch();
    }, []);
    
    const runInference = () => {
        if (rustorch) {
            const input = new WasmTensor([1,2,3,4,5,6,7,8,9,10], [10]);
            const output = rustorch.linear.forward(input);
            console.log('Result:', output.data());
        }
    };
    
    return (
        <div>
            <button onClick={runInference}>Run ML Inference</button>
        </div>
    );
}
```

### Vue.js Integration / Vue.js統合

```javascript
import { createApp, ref, onMounted } from 'vue';
import init, { WebGPUSimpleDemo } from './pkg/rustorch.js';

export default {
    setup() {
        const demo = ref(null);
        const results = ref([]);
        
        onMounted(async () => {
            await init();
            demo.value = new WebGPUSimpleDemo();
            await demo.value.initialize();
        });
        
        const runDemo = async () => {
            const result = await demo.value.run_comprehensive_demo();
            results.value.push(result);
        };
        
        return { runDemo, results };
    }
};
```

## Changelog / 変更履歴

### v0.5.3+ - WebGPU Integration / WebGPU統合
- ✅ Chrome-optimized WebGPU backend
- ✅ Interactive browser demos
- ✅ CPU fallback mechanisms
- ✅ Performance estimation system

### v0.5.3 - Enhanced WASM Features / 強化WASM機能
- ✅ Special mathematical functions (Gamma, Bessel, Error)
- ✅ Statistical distributions (Normal, Gamma, Beta, etc.)
- ✅ Neural network optimizers (SGD, Adam, RMSprop)
- ✅ Simplified autograd for single-threaded environments

### v0.5.2 - Production WASM / 本番WASM
- ✅ Comprehensive tensor operations
- ✅ Computer vision pipelines
- ✅ Quality assessment tools
- ✅ Performance profiling

## Contributing / 貢献

### Adding New WASM Features / 新しいWASM機能の追加

```rust
// 1. Add to src/wasm/your_module.rs
#[wasm_bindgen]
pub struct YourWasmStruct {
    // Implementation
}

// 2. Add to src/wasm/mod.rs
pub mod your_module;

// 3. Add feature flag to Cargo.toml
your-feature = ["wasm", "additional-deps"]

// 4. Build and test
wasm-pack build --target web --features your-feature
```

### Testing Guidelines / テストガイドライン

```bash
# Always test both CPU and WebGPU paths
cargo test --features wasm
cargo test --features webgpu

# Test browser compatibility
wasm-pack test --chrome --firefox --safari
```

---

**📚 For more examples, see the `/examples` directory**  
**より多くの例については、`/examples` ディレクトリを参照してください**

**🔗 Links:**
- [Main Documentation](../README.md)
- [API Reference](https://docs.rs/rustorch)
- [WebGPU Demo](../examples/webgpu_simple_demo.html)
- [GitHub Repository](https://github.com/JunSuzukiJapan/rustorch)
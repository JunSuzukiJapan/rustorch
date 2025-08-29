# RusTorch WebAssembly (WASM) Module

RusTorchのWebAssemblyモジュールは、ブラウザ環境で高性能な機械学習処理を可能にする包括的なライブラリです。

## 🌟 主要機能

### 🧠 Neural Network Components
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, GELU, Swish, Mish, ELU, Softplus
- **Loss Functions**: MSE, MAE, Cross-Entropy, Focal Loss, Huber Loss, KL Divergence, Hinge Loss
- **Neural Network Layers**: Linear (Dense), Conv2D with proper initialization (Xavier/He)
- **Optimizers**: SGD, Adam, AdaGrad, RMSprop with momentum and learning rate scheduling
- **Normalization**: BatchNorm, LayerNorm, GroupNorm

### 📊 Data Processing
- **Preprocessing**: Min-max normalization, Z-score standardization
- **Data Augmentation**: Gaussian noise injection, random rotation, flipping
- **Encoding**: One-hot encoding/decoding for categorical data
- **Dataset Utilities**: Train-test split, batch creation

### 🎨 Computer Vision
- **Image Transformations**: Resize (bilinear), crop, center crop, rotation
- **Color Space**: RGB↔Grayscale conversion, normalization (ImageNet compatible)
- **Image Enhancement**: Brightness/contrast adjustment, Gaussian blur
- **Data Augmentation**: Horizontal/vertical flip, random rotation, noise injection
- **Feature Extraction**: Edge detection (Sobel), histogram analysis
- **Format Conversion**: uint8↔float32, histogram equalization

### ⚡ Advanced Operations
- **Matrix Operations**: Matrix multiplication, transpose, reshape
- **Tensor Operations**: Concatenation, splitting, broadcasting
- **Training Utilities**: Gradient clipping, dropout
- **Memory Management**: Tensor pooling, memory monitoring

### 📈 Model Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Regression Metrics**: MAE, MSE, RMSE, R²
- **Advanced Metrics**: Top-k accuracy, confusion matrix
- **Comprehensive Reports**: Full classification analysis

### 🎲 Statistical Distributions
- **Distributions**: Normal, Uniform, Bernoulli, Exponential
- **Special Functions**: Gamma, Error functions, Bessel functions
- **Signal Processing**: FFT/DFT, windowing, correlation

## 🚀 Quick Start

### Installation

```bash
# Build WASM package
wasm-pack build --target web --features wasm --no-default-features

# Or use in your Rust project
cargo add rustorch --features wasm
```

### Basic Usage

```javascript
import init, * as rustorch from './pkg/rustorch.js';

await init();

// Neural network layers
const linear1 = new rustorch.WasmLinear(784, 128, true);
const linear2 = new rustorch.WasmLinear(128, 10, true);

// Convolutional layer for images
const conv = new rustorch.WasmConv2d(3, 32, 3, 1, 1, true);

// Forward pass
let x = linear1.forward(input_data, batch_size);
x = rustorch.relu(x);
x = linear2.forward(x, batch_size);
const predictions = rustorch.softmax(x);

// Image processing
const resized_image = rustorch.WasmVision.resize(image, 256, 256, 224, 224, 3);
const normalized_image = rustorch.WasmVision.normalize(
    resized_image,
    [0.485, 0.456, 0.406],  // ImageNet mean
    [0.229, 0.224, 0.225],  // ImageNet std
    3
);

// Loss calculation and optimization
const optimizer = new rustorch.WasmAdam(0.001);
const loss = rustorch.cross_entropy_loss(predictions, targets);
console.log('Loss:', loss);
```

## 📁 Module Structure

```
src/wasm/
├── activation.rs      # Neural network activation functions
├── distributions.rs   # Statistical distributions
├── loss.rs           # Loss functions for training
├── memory.rs         # Memory management utilities
├── metrics.rs        # Model evaluation metrics
├── normalization.rs  # Normalization layers
├── optimizer.rs      # Optimization algorithms
├── preprocessing.rs  # Data preprocessing utilities
├── runtime.rs        # WASM runtime initialization
├── signal.rs         # Signal processing functions
├── special.rs        # Special mathematical functions
└── tensor_ops.rs     # Advanced tensor operations
```

## 💡 Use Cases

### 1. Browser-based ML Training
```javascript
// Complete training pipeline in browser
const processor = new rustorch.WasmPreprocessor();
const optimizer = new rustorch.WasmOptimizer('adam', 0.001);
const batchNorm = new rustorch.WasmBatchNorm(64, 0.1, 1e-5);

// Training loop
for (let epoch = 0; epoch < 100; epoch++) {
    const batches = processor.create_batches(features, targets, 10, 32);
    
    for (let batch of batches) {
        // Forward pass
        let output = batchNorm.forward(batch.features, 32);
        output = rustorch.relu(output);
        
        // Loss calculation
        const loss = rustorch.mse_loss(output, batch.targets);
        
        // Optimization step (gradients computed elsewhere)
        optimizer.step('weights', weights, gradients);
    }
}
```

### 2. Real-time Data Processing
```javascript
// Real-time signal processing
const signal = new Float32Array(1024);
// ... fill with audio/sensor data

// Apply windowing and FFT
const windowed = rustorch.apply_hann_window(Array.from(signal));
const fft_result = rustorch.dft(windowed);

// Extract features
const magnitude = rustorch.compute_magnitude(fft_result);
const features = rustorch.WasmPreprocessor.min_max_normalize(
    magnitude, 0.0, 1.0
);
```

### 3. Model Evaluation Dashboard
```javascript
// Comprehensive model evaluation
const predictions = await model.predict(test_data);
const metrics = rustorch.WasmMetrics.classification_report(
    predictions, test_labels, num_classes
);

// Display results
console.log('Accuracy:', metrics.accuracy);
console.log('Per-class metrics:', metrics.classes);
console.log('Confusion Matrix:', metrics.confusionMatrix);
```

## 🔧 Performance Considerations

### Memory Management
- WASMモジュールはメモリプールを使用してガベージコレクションを最小化
- 大きなテンソルは`WasmTensorPool`で効率的に管理
- `WasmMemoryMonitor`でメモリ使用量を監視

### Optimization Tips
- バッチサイズを適切に設定（通常32-128）
- 不要なデータコピーを避ける
- 訓練時はBatchNormを使用、推論時はLayerNormを検討
- 勾配クリッピングで訓練安定性を向上

## 🛠️ Development

### Building from Source
```bash
# Prerequisites
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Build
wasm-pack build --target web --features wasm --no-default-features

# Test
wasm-pack test --headless --chrome --features wasm
```

### Adding New Features
1. 新しいモジュールを`src/wasm/`に作成
2. `src/wasm/mod.rs`に追加
3. `#[cfg(feature = "wasm")]`でガード
4. `#[wasm_bindgen]`でJavaScript互換性を確保
5. WASMビルドでテスト

## 📚 Additional Resources

- [WASM API Reference](./api.md)
- [Browser Integration Guide](./browser-integration.md) 
- [Performance Benchmarks](./benchmarks.md)
- [Examples Repository](./examples/)

## 🤝 Contributing

WASM機能の改善や新機能の提案は、GitHubのIssueまたはPull Requestでお寄せください。

## 📄 License

MIT OR Apache-2.0
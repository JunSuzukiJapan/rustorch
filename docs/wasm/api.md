# RusTorch WASM API Reference

完全なWebAssembly APIリファレンス

## 🧠 Neural Network Functions

### Activation Functions

#### `relu(input: Vec<f32>) -> Vec<f32>`
ReLU活性化関数を適用
```javascript
const output = rustorch.relu([-1.0, 0.0, 1.0]);
// Result: [0.0, 0.0, 1.0]
```

#### `sigmoid(input: Vec<f32>) -> Vec<f32>`
シグモイド活性化関数を適用
```javascript
const output = rustorch.sigmoid([0.0, 1.0, -1.0]);
// Result: [0.5, 0.73, 0.27]
```

#### `softmax(input: Vec<f32>) -> Vec<f32>`
ソフトマックス関数を適用（確率分布に変換）
```javascript
const probabilities = rustorch.softmax([1.0, 2.0, 3.0]);
// Result: [0.09, 0.24, 0.67] (sums to 1.0)
```

#### `gelu(input: Vec<f32>) -> Vec<f32>`
GELU（Gaussian Error Linear Unit）活性化関数
```javascript
const output = rustorch.gelu([-1.0, 0.0, 1.0]);
```

### Loss Functions

#### `mse_loss(predictions: Vec<f32>, targets: Vec<f32>) -> f32`
平均二乗誤差を計算
```javascript
const loss = rustorch.mse_loss([1.0, 2.0], [1.5, 1.5]);
// Result: 0.25
```

#### `cross_entropy_loss(predictions: Vec<f32>, targets: Vec<f32>) -> f32`
クロスエントロピー損失を計算
```javascript
const loss = rustorch.cross_entropy_loss(
    [0.7, 0.2, 0.1],  // Predicted probabilities
    [1.0, 0.0, 0.0]   // True labels (one-hot)
);
```

## 📊 Data Preprocessing

### WasmPreprocessor

#### `compute_stats(data: &[f32]) -> Vec<f32>`
データの統計値を計算（平均、標準偏差、最小値、最大値）
```javascript
const stats = rustorch.WasmPreprocessor.compute_stats([1, 2, 3, 4, 5]);
// Result: [mean, std, min, max]
```

#### `min_max_normalize(data: Vec<f32>, min: f32, max: f32) -> Vec<f32>`
Min-Max正規化（0-1範囲に正規化）
```javascript
const normalized = rustorch.WasmPreprocessor.min_max_normalize(
    [1, 2, 3, 4, 5], 1.0, 5.0
);
// Result: [0.0, 0.25, 0.5, 0.75, 1.0]
```

#### `z_score_normalize(data: Vec<f32>, mean: f32, std: f32) -> Vec<f32>`
Z-score標準化（平均0、分散1に正規化）
```javascript
const normalized = rustorch.WasmPreprocessor.z_score_normalize(
    data, mean, std
);
```

#### `one_hot_encode(labels: Vec<u32>, num_classes: u32) -> Vec<f32>`
カテゴリカルデータをワンホットエンコーディング
```javascript
const encoded = rustorch.WasmPreprocessor.one_hot_encode([0, 1, 2], 3);
// Result: [1,0,0, 0,1,0, 0,0,1]
```

#### `train_test_split(features, targets, feature_size, test_ratio, seed) -> Object`
データセットを訓練・テストに分割
```javascript
const split = rustorch.WasmPreprocessor.train_test_split(
    features, targets, 10, 0.2, 42
);
// Returns: {trainFeatures, trainTargets, testFeatures, testTargets}
```

## 🏗️ Normalization Layers

### WasmBatchNorm

#### Constructor: `new(num_features: usize, momentum: f32, epsilon: f32)`
バッチ正規化レイヤーを作成
```javascript
const batchNorm = new rustorch.WasmBatchNorm(64, 0.1, 1e-5);
```

#### `forward(input: Vec<f32>, batch_size: usize) -> Vec<f32>`
バッチ正規化の順伝播
```javascript
const output = batchNorm.forward(input_data, 32);
```

#### `set_training(training: bool)`
訓練モード/推論モードを設定
```javascript
batchNorm.set_training(true);  // 訓練モード
batchNorm.set_training(false); // 推論モード
```

### WasmLayerNorm

#### Constructor: `new(normalized_shape: Vec<usize>, epsilon: f32)`
レイヤー正規化レイヤーを作成
```javascript
const layerNorm = new rustorch.WasmLayerNorm([512], 1e-5);
```

#### `forward(input: Vec<f32>) -> Vec<f32>`
レイヤー正規化の順伝播
```javascript
const output = layerNorm.forward(input_data);
```

## 🏗️ Neural Network Layers

### WasmLinear

#### Constructor: `new(in_features: usize, out_features: usize, bias: bool)`
線形レイヤーを作成（Xavier初期化）
```javascript
const linear = new rustorch.WasmLinear(784, 128, true);
```

#### `forward(input: Vec<f32>, batch_size: usize) -> Vec<f32>`
線形レイヤーの順伝播
```javascript
const output = linear.forward(input_data, 32); // batch_size = 32
```

#### `get_weights() -> Vec<f32>`
重みパラメータを取得
```javascript
const weights = linear.get_weights();
```

#### `update_weights(new_weights: Vec<f32>)`
重みを更新
```javascript
linear.update_weights(updated_weights);
```

#### `get_bias() -> Option<Vec<f32>>`
バイアスパラメータを取得
```javascript
const bias = linear.get_bias();
```

### WasmConv2d

#### Constructor: `new(in_channels, out_channels, kernel_size, stride, padding, bias)`
2次元畳み込みレイヤーを作成（He初期化）
```javascript
const conv = new rustorch.WasmConv2d(3, 64, 3, 1, 1, true);
```

#### `forward(input, batch_size, input_height, input_width) -> Vec<f32>`
畳み込みレイヤーの順伝播
```javascript
const output = conv.forward(image_data, 1, 32, 32);
```

#### `output_shape(input_height: usize, input_width: usize) -> Vec<usize>`
出力次元を計算
```javascript
const [out_channels, out_height, out_width] = conv.output_shape(224, 224);
```

#### `get_config() -> Object`
レイヤー設定を取得
```javascript
const config = conv.get_config();
console.log('Kernel size:', config.kernel_size);
```

## 🎨 Vision Processing

### WasmVision

#### `resize(image, orig_h, orig_w, new_h, new_w, channels) -> Vec<f32>`
バイリニア補間による画像リサイズ
```javascript
const resized = rustorch.WasmVision.resize(
    image_data, 224, 224, 256, 256, 3
);
```

#### `normalize(image: Vec<f32>, mean: Vec<f32>, std: Vec<f32>, channels) -> Vec<f32>`
ImageNet式正規化
```javascript
const normalized = rustorch.WasmVision.normalize(
    image_data, 
    [0.485, 0.456, 0.406],  // ImageNet mean
    [0.229, 0.224, 0.225],  // ImageNet std
    3
);
```

#### `rgb_to_grayscale(rgb_data, height, width) -> Vec<f32>`
RGB→グレースケール変換
```javascript
const grayscale = rustorch.WasmVision.rgb_to_grayscale(rgb_data, 224, 224);
```

#### `gaussian_blur(image, height, width, channels, sigma) -> Vec<f32>`
ガウシアンブラー
```javascript
const blurred = rustorch.WasmVision.gaussian_blur(image, 224, 224, 3, 1.5);
```

#### `crop(image, height, width, channels, start_y, start_x, crop_h, crop_w)`
画像クロップ
```javascript
const cropped = rustorch.WasmVision.crop(image, 256, 256, 3, 16, 16, 224, 224);
```

#### `center_crop(image, height, width, channels, crop_size) -> Vec<f32>`
センタークロップ
```javascript
const center_cropped = rustorch.WasmVision.center_crop(image, 256, 256, 3, 224);
```

#### `flip_horizontal(image, height, width, channels) -> Vec<f32>`
水平反転（データ拡張）
```javascript
const flipped = rustorch.WasmVision.flip_horizontal(image, 224, 224, 3);
```

#### `add_gaussian_noise(image: Vec<f32>, std_dev: f32) -> Vec<f32>`
ガウシアンノイズ追加
```javascript
const noisy = rustorch.WasmVision.add_gaussian_noise(image, 0.1);
```

#### `edge_detection(grayscale, height, width) -> Vec<f32>`
エッジ検出（Sobelフィルター）
```javascript
const edges = rustorch.WasmVision.edge_detection(grayscale, 224, 224);
```

#### `to_float(image_u8: Vec<u8>) -> Vec<f32>`
0-255 → 0-1変換
```javascript
const float_image = rustorch.WasmVision.to_float(uint8_array);
```

## ⚡ Tensor Operations

### WasmTensorOps

#### `matmul(a, a_rows, a_cols, b, b_rows, b_cols) -> Vec<f32>`
行列積を計算
```javascript
const result = rustorch.WasmTensorOps.matmul(
    [1, 2, 3, 4], 2, 2,  // Matrix A (2x2)
    [5, 6, 7, 8], 2, 2   // Matrix B (2x2)
);
// Result: [19, 22, 43, 50]
```

#### `transpose(matrix: Vec<f32>, rows: usize, cols: usize) -> Vec<f32>`
行列の転置
```javascript
const transposed = rustorch.WasmTensorOps.transpose([1, 2, 3, 4, 5, 6], 2, 3);
```

#### `concatenate(tensors, shapes, axis) -> Object`
テンソルの連結
```javascript
const result = rustorch.WasmTensorOps.concatenate(
    [tensor1, tensor2],  // Tensors to concatenate
    [shape1, shape2],    // Their shapes
    0                    // Axis to concatenate along
);
// Returns: {data: Array, shape: Array}
```

#### `clip_gradients(gradients: Vec<f32>, max_norm: f32) -> Vec<f32>`
勾配クリッピング
```javascript
const clipped = rustorch.WasmTensorOps.clip_gradients(gradients, 1.0);
```

## 📊 Model Evaluation

### WasmMetrics

#### `accuracy(predictions: Vec<u32>, targets: Vec<u32>) -> f32`
分類精度を計算
```javascript
const acc = rustorch.WasmMetrics.accuracy([0, 1, 1, 0], [0, 1, 0, 0]);
// Result: 0.75 (75% accuracy)
```

#### `classification_report(predictions, targets, num_classes) -> Object`
包括的な分類レポート
```javascript
const report = rustorch.WasmMetrics.classification_report(
    predictions, targets, 3
);
console.log('Accuracy:', report.accuracy);
console.log('Per-class metrics:', report.classes);
```

#### `mse(predictions: Vec<f32>, targets: Vec<f32>) -> f32`
平均二乗誤差
```javascript
const mse = rustorch.WasmMetrics.mse([1.0, 2.0], [1.1, 1.9]);
```

#### `r2_score(predictions: Vec<f32>, targets: Vec<f32>) -> f32`
決定係数（R²）
```javascript
const r2 = rustorch.WasmMetrics.r2_score(predictions, targets);
```

## 🎲 Optimization

### WasmOptimizer

#### Constructor: `new(optimizer_type: String, learning_rate: f32)`
オプティマイザーを作成
```javascript
const optimizer = new rustorch.WasmOptimizer('adam', 0.001);
```

#### `step(param_id: String, parameters: Vec<f32>, gradients: Vec<f32>) -> Vec<f32>`
最適化ステップを実行
```javascript
const updated_params = optimizer.step('weights', weights, gradients);
```

#### `set_learning_rate(learning_rate: f32)`
学習率を設定
```javascript
optimizer.set_learning_rate(0.0001);
```

## 🔊 Signal Processing

### Signal Functions

#### `dft(signal: Vec<f32>) -> Vec<f32>`
離散フーリエ変換
```javascript
const frequency_domain = rustorch.dft(time_domain_signal);
```

#### `apply_hann_window(signal: Vec<f32>) -> Vec<f32>`
ハン窓を適用
```javascript
const windowed = rustorch.apply_hann_window(signal);
```

#### `cross_correlation(signal1: Vec<f32>, signal2: Vec<f32>) -> Vec<f32>`
相互相関を計算
```javascript
const correlation = rustorch.cross_correlation(signal1, signal2);
```

## 🎯 Memory Management

### WasmTensorPool

#### Constructor: `new(max_size: usize)`
テンソルプールを作成
```javascript
const pool = new rustorch.WasmTensorPool(1024 * 1024); // 1MB
```

#### `allocate(size: usize) -> Option<usize>`
メモリを割り当て
```javascript
const allocation_id = pool.allocate(1000);
```

### WasmMemoryMonitor

#### Constructor: `new()`
メモリモニターを作成
```javascript
const monitor = new rustorch.WasmMemoryMonitor();
```

#### `record_allocation(size: usize)`
メモリ割り当てを記録
```javascript
monitor.record_allocation(1024);
```

#### `peak_usage() -> usize`
ピークメモリ使用量を取得
```javascript
const peak = monitor.peak_usage();
```

## 🔢 Statistical Distributions

### WasmNormalDistribution

#### Constructor: `new(mean: f32, std_dev: f32, seed: u32)`
正規分布を作成
```javascript
const normal = new rustorch.WasmNormalDistribution(0.0, 1.0, 42);
```

#### `sample() -> f32`
サンプルを生成
```javascript
const random_value = normal.sample();
```

#### `pdf(x: f32) -> f32`
確率密度を計算
```javascript
const density = normal.pdf(0.5);
```

## ⚙️ Runtime Utilities

### Runtime Functions

#### `initialize_wasm_runtime()`
WASM ランタイムを初期化
```javascript
rustorch.initialize_wasm_runtime();
```

### WasmPerformance

#### Constructor: `new()`
パフォーマンスモニターを作成
```javascript
const perf = new rustorch.WasmPerformance();
```

#### `elapsed() -> f64`
経過時間を取得（ミリ秒）
```javascript
perf.start();
// ... some computation
const elapsed_ms = perf.elapsed();
```

## 🚨 Error Handling

すべてのWASM関数は適切なエラーハンドリングを提供：

```javascript
try {
    const result = rustorch.WasmTensorOps.matmul(a, 2, 2, b, 3, 3);
} catch (error) {
    console.error('Matrix multiplication failed:', error.message);
    // Handle dimension mismatch or other errors
}
```

## 💡 Best Practices

1. **初期化**: 使用前に`initialize_wasm_runtime()`を呼び出す
2. **メモリ管理**: 大きなテンソルにはプールを使用
3. **バッチ処理**: 効率のためデータをバッチで処理
4. **エラーハンドリング**: try-catchでWASM関数を囲む
5. **パフォーマンス監視**: 本番環境では`WasmPerformance`を使用

## 🔄 Migration from Native RusTorch

Native RusTorchからの移行ガイド：

```rust
// Native RusTorch
let tensor = Tensor::from_vec(data, shape);
let output = tensor.relu();

// WASM equivalent  
const output = rustorch.relu(data);
```

```rust
// Native RusTorch
let optimizer = Adam::new(params, 0.001);
optimizer.step();

// WASM equivalent
const optimizer = new rustorch.WasmOptimizer('adam', 0.001);
const updated = optimizer.step('weights', weights, gradients);
```
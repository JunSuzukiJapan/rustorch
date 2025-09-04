# WASM API Enhancement Roadmap

## 🎯 概要

RusTorch WASMモジュールの機能拡張ロードマップ。既存API_DOCUMENTATION.mdとの比較分析に基づき、WASM制約を考慮した実装優先度を定義。

## 🚧 WASM技術制約

### 基本制約
- **メモリ制限**: 通常2GB以下、実用的には512MB-1GB
- **単一スレッド**: 並列処理・マルチスレッド不可
- **ファイルシステム**: 直接アクセス不可（ブラウザAPIのみ）
- **ネットワーク**: Fetch API限定
- **型制限**: f32/f64/i32/i64が最効率（JS相互運用）

### 性能制約
- **SIMD**: 限定的サポート（wasm-simd feature）
- **GPU**: WebGPUのみ（Chrome/Edge限定）
- **stdlib**: 制限版（std::thread, std::fs等不可）

## 🚀 実装優先度マトリクス

### 🟢 Phase 1: 高優先度（✅ 実装済み）

#### 1.1 特殊数学関数拡張 ✅
**互換性**: ⭐⭐⭐⭐⭐ | **有用性**: ⭐⭐⭐⭐ | **複雑度**: ⭐⭐

```rust
// ✅ 実装済み - special_enhanced.rs
use rustorch::wasm::special_enhanced;

// Gamma関数群
let gamma = special_enhanced::gamma_wasm(x);
let lgamma = special_enhanced::lgamma_wasm(x);
let digamma = special_enhanced::digamma_wasm(x);
let beta = special_enhanced::beta_wasm(a, b);

// Bessel関数群
let j_bessel = special_enhanced::bessel_j_wasm(n, x);
let y_bessel = special_enhanced::bessel_y_wasm(n, x);
let i_bessel = special_enhanced::bessel_i_wasm(n, x);
let k_bessel = special_enhanced::bessel_k_wasm(n, x);

// エラー関数群
let erf = special_enhanced::erf_wasm(x);
let erfc = special_enhanced::erfc_wasm(x);
let erfinv = special_enhanced::erfinv_wasm(x);

// ベクトル化版
let gamma_array = special_enhanced::gamma_array_wasm(&values);
let erf_array = special_enhanced::erf_array_wasm(&values);
```

**実装状況**: ✅ 完了
**メモリ影響**: 最小限
**ブラウザサポート**: 全ブラウザ

#### 1.2 統計分布完全版 ✅
**互換性**: ⭐⭐⭐⭐⭐ | **有用性**: ⭐⭐⭐⭐⭐ | **複雑度**: ⭐⭐⭐

```rust
// ✅ 実装済み - distributions_enhanced.rs
use rustorch::wasm::distributions_enhanced;

// 連続分布
let normal = NormalDistributionWasm::new(mean: 0.0, std: 1.0);
let samples = normal.sample_array(1000);
let log_probs = normal.log_prob_array(&values);

// その他の分布も実装済み
let uniform = UniformDistributionWasm::new(low: 0.0, high: 1.0);
let exponential = ExponentialDistributionWasm::new(rate: 1.0);
let gamma = GammaDistributionWasm::new(alpha: 2.0, beta: 1.0);
let beta = BetaDistributionWasm::new(alpha: 2.0, beta: 3.0);
```

**実装状況**: ✅ 完了
**メモリ影響**: 中程度（サンプリング時）
**ブラウザサポート**: 全ブラウザ

#### 1.3 FFT・信号処理完全版 ✅
**互換性**: ⭐⭐⭐⭐⭐ | **有用性**: ⭐⭐⭐⭐⭐ | **複雑度**: ⭐⭐⭐

```rust
// ✅ 実装済み - signal.rs
use rustorch::wasm::WasmSignal;

// FFT/IFFT
let fft_result = WasmSignal::dft(time_domain);
let ifft_result = WasmSignal::idft(real_fft, imag_fft);
let rfft_result = WasmSignal::rfft(real_signal);

// 窓関数
let hamming = WasmSignal::hamming_window(signal);
let hanning = WasmSignal::hanning_window(signal);
let blackman = WasmSignal::blackman_window(signal);

// 信号解析
let power_spec = WasmSignal::power_spectrum(signal);
let magnitude = WasmSignal::magnitude_spectrum(real_fft, imag_fft);
let phase = WasmSignal::phase_spectrum(real_fft, imag_fft);

// フィルタリング
let low_pass = WasmSignal::low_pass_filter(signal, window_size);
let high_pass = WasmSignal::high_pass_filter(signal, window_size);

// 信号生成
let sine = WasmSignal::generate_sine_wave(freq, sample_rate, duration, amp, phase);
let noise = WasmSignal::generate_white_noise(samples, amplitude, seed);
```

**実装状況**: ✅ 完了
**メモリ影響**: 中程度
**ブラウザサポート**: 全ブラウザ
**用途**: 音声処理、画像解析、リアルタイム信号処理

#### 1.4 損失関数完全版 ✅
**互換性**: ⭐⭐⭐⭐⭐ | **有用性**: ⭐⭐⭐⭐⭐ | **複雑度**: ⭐⭐

```rust
// ✅ 実装済み - loss.rs
use rustorch::wasm::WasmLoss;

// 分類損失
let cross_entropy = WasmLoss::cross_entropy_wasm(&predictions, &targets);
let binary_cross_entropy = WasmLoss::binary_cross_entropy_wasm(&pred, &target);
let focal_loss = WasmLoss::focal_loss_wasm(&pred, &target, alpha, gamma);

// 回帰損失
let mse = WasmLoss::mse_loss_wasm(&predictions, &targets);
let mae = WasmLoss::mae_loss_wasm(&predictions, &targets);
let huber = WasmLoss::huber_loss_wasm(&pred, &target, delta);
let smooth_l1 = WasmLoss::smooth_l1_loss_wasm(&pred, &target);

// 正則化
let l1_reg = WasmLoss::l1_regularization_wasm(&weights, lambda);
let l2_reg = WasmLoss::l2_regularization_wasm(&weights, lambda);
let elastic_net = WasmLoss::elastic_net_regularization_wasm(&weights, l1, l2);

// 高度な損失
let kl_div = WasmLoss::kl_divergence_wasm(&pred, &target);
let triplet = WasmLoss::triplet_loss_wasm(&anchor, &positive, &negative, margin);
```

**実装状況**: ✅ 完了
**メモリ影響**: 最小限
**ブラウザサポート**: 全ブラウザ

### 🟢 Phase 2: 中優先度（✅ 実装完了）

#### 2.1 Computer Vision拡張 ✅
**互換性**: ⭐⭐⭐⭐ | **有用性**: ⭐⭐⭐⭐⭐ | **複雑度**: ⭐⭐⭐⭐

```rust
// ✅ 実装済み - vision.rs
use rustorch::wasm::WasmVision;

// 基本画像変換
let resized = WasmVision::resize(image, old_w, old_h, new_w, new_h, channels);
let rotated = WasmVision::random_rotation(image, h, w, channels, max_angle);
let cropped = WasmVision::crop(image, h, w, channels, y, x, crop_h, crop_w);
let normalized = WasmVision::normalize(image, mean, std, channels);

// 高度な画像処理
let blurred = WasmVision::gaussian_blur(image, h, w, channels, sigma);
let edges = WasmVision::edge_detection(grayscale, h, w);
let corners = WasmVision::harris_corner_detection(gray, h, w, threshold, k);

// モルフォロジー演算
let opened = WasmVision::morphological_opening(image, h, w, kernel_size);
let closed = WasmVision::morphological_closing(image, h, w, kernel_size);
let lbp = WasmVision::local_binary_patterns(gray, h, w, radius);

// 色空間・ヒストグラム
let grayscale = WasmVision::rgb_to_grayscale(rgb, h, w);
let histogram = WasmVision::histogram(image, bins);
let equalized = WasmVision::histogram_equalization(image, bins);
```

**実装状況**: ✅ 完了
**メモリ制約**: チャンク処理対応済み
**ブラウザサポート**: 全ブラウザ

#### 2.2 簡略化Autograd ✅
**互換性**: ⭐⭐⭐⭐ | **有用性**: ⭐⭐⭐⭐ | **複雑度**: ⭐⭐⭐⭐

```rust
// ✅ 実装済み - autograd_simplified.rs  
use rustorch::wasm::{VariableWasm, ComputationGraphWasm, WasmOptimizer};

// 変数と計算グラフ
let mut graph = ComputationGraphWasm::new();
let var_id = graph.create_variable(data, shape, requires_grad);

// 演算
let add_result = var1.add(&var2);
let mul_result = var1.mul(&var2);
let matmul_result = var1.matmul(&var2);

// 活性化関数
let relu_out = var.relu();
let sigmoid_out = var.sigmoid();
let tanh_out = var.tanh_activation();

// 自動微分
var.backward();
let gradients = var.grad();

// 最適化
let optimizer = WasmOptimizer::sgd(learning_rate);
optimizer.step(&mut variable);
```

**実装状況**: ✅ 完了
**メモリ制約**: 計算グラフサイズ制限対応済み
**ブラウザサポート**: 全ブラウザ

#### 2.3 ブラウザモデル永続化 ✅
**互換性**: ⭐⭐⭐⭐⭐ | **有用性**: ⭐⭐⭐⭐ | **複雑度**: ⭐⭐⭐

```rust
// ✅ 実装済み - storage.rs
use rustorch::wasm::{WasmModelStorage, WasmModelCompression, WasmProgressTracker};

// モデル保存・読み込み
let storage = WasmModelStorage::new(use_indexeddb: true, chunk_size: 1_000_000);
storage.save_model("my_model", model_data).await;
storage.save_large_model("large_model", big_model_data).await;

let model = storage.load_model("my_model").await;
let large_model = storage.load_large_model("large_model").await;

// モデル圧縮
let compressed = WasmModelCompression::compress_weights(weights);
let quantized = WasmModelCompression::quantize_weights(weights, bits: 8);

// 進捗追跡
let tracker = WasmProgressTracker::new(total_steps, "Loading model");
tracker.update(current_step);
let progress = tracker.progress_percent();

// ストレージ管理
let models = storage.list_models().await;
let available_space = storage.get_available_storage().await;
storage.delete_model("old_model").await;
```

**実装状況**: ✅ 完了
**メモリ制約**: チャンク処理・圧縮対応済み  
**ブラウザサポート**: IndexedDB + LocalStorage対応

### 🔴 Phase 3: 低優先度（条件付き実装）

#### 3.1 基本線形代数
**互換性**: ⭐⭐⭐ | **有用性**: ⭐⭐⭐ | **複雑度**: ⭐⭐⭐⭐

```rust
// BLAS非依存線形代数
use rustorch::wasm::WasmLinearAlgebra;

let linalg = WasmLinearAlgebra::new();
// 小行列のみ（< 1000x1000）
let eigenvalues = linalg.eigenvalues(&small_matrix);
let svd = linalg.svd(&matrix);
```

**制約**: 大行列で極度に遅い、メモリ大量消費

## 📊 実装タイムライン

| Phase | 期間 | 機能 | 累積価値 | 状況 |
|-------|------|------|----------|------|
| Phase 1 | ✅ 完了 | 特殊関数・分布・FFT・損失 | 60% | 全機能実装済み |
| Phase 2 | ✅ 完了 | Vision・Autograd・永続化 | 85% | 全機能実装済み |
| Phase 3 | 2-4週 | 線形代数（条件付き） | 95% | 条件付き実装 |

## 🎯 実装状況（Phase 1 & 2 完了）

### ✅ 実装完了（Phase 1 - 基本ML機能）
1. **WasmLoss** - 完全実装（全損失関数・正則化）
2. **WasmSpecialFunctions** - 完全実装（Gamma・Bessel・エラー関数）  
3. **WasmDistributions** - 完全実装（連続・離散分布）
4. **WasmSignal/FFT** - 完全実装（FFT・窓関数・信号処理）

### ✅ 実装完了（Phase 2 - 高度ML機能）
5. **WasmVision** - ブラウザ画像処理（リサイズ・フィルター・エッジ検出・モルフォロジー）
6. **WasmAutograd** - 軽量勾配計算（Variable・演算・最適化器）
7. **WasmModelStorage** - 実用性向上（IndexedDB・圧縮・進捗追跡）

### 🎯 次期候補（Phase 3 - 条件付き）
8. **WasmLinearAlgebra** - BLAS非依存線形代数（小行列限定）

## 🔍 技術的考慮事項

### メモリ最適化戦略
- チャンク処理（大データセット用）
- 遅延評価（必要時のみ計算）
- ガベージコレクション最適化

### パフォーマンス戦略  
- WebWorker活用（可能な場合）
- WebGPU fallback（Chrome環境）
- 段階的計算（UIブロック防止）

### 互換性戦略
- Progressive Web App対応
- オフライン機能サポート
- モバイルブラウザ最適化

この優先順位に基づいて段階的にWASM APIを拡張することで、ブラウザでの本格的な機械学習ワークフローを実現できます。
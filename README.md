# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-201%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)
[![GPU](https://img.shields.io/badge/GPU-CUDA%2FMetal%2FOpenCL-blue.svg)](#gpu-acceleration)
[![Performance](https://img.shields.io/badge/performance-SIMD%20optimized-orange.svg)](#performance)

**A production-ready deep learning library in Rust with PyTorch-like API, GPU acceleration, and enterprise-grade performance**  
**本番環境対応のRust製ディープラーニングライブラリ - PyTorchライクなAPI、GPU加速、エンタープライズグレードパフォーマンス**

RusTorch is a fully functional deep learning library that leverages Rust's safety and performance, providing comprehensive tensor operations, automatic differentiation, neural network layers, transformer architectures, multi-backend GPU acceleration (CUDA/Metal/OpenCL), advanced SIMD optimizations, and enterprise-grade memory management features.  
RusTorchは、Rustの安全性とパフォーマンスを活かした完全機能のディープラーニングライブラリです。包括的なテンソル演算、自動微分システム、ニューラルネットワーク層、Transformerアーキテクチャ、マルチバックエンドGPU加速（CUDA/Metal/OpenCL）、高度なSIMD最適化、エンタープライズグレードメモリ管理機能を提供します。

## ✨ Features / 主な特徴

- 🔥 **Comprehensive Tensor Operations**: Math operations, broadcasting, indexing, and statistics  
  **包括的テンソル演算**: 数学演算、ブロードキャスティング、インデックス操作、統計機能
- 🤖 **Transformer Architecture**: Complete transformer implementation with multi-head attention  
  **Transformerアーキテクチャ**: マルチヘッドアテンション付き完全なTransformer実装
- 📝 **Embedding Systems**: Word embeddings, positional encoding, sinusoidal encoding  
  **埋め込みシステム**: 単語埋め込み、位置エンコーディング、正弦波エンコーディング
- 📊 **Advanced Statistics**: Mean, variance, std, median, quantile, covariance, correlation  
  **高度な統計**: 平均、分散、標準偏差、中央値、分位数、共分散、相関
- 🎯 **Broadcasting Support**: Automatic shape compatibility and dimension expansion  
  **ブロードキャスティング**: 自動形状互換性と次元拡張
- 🔍 **Flexible Indexing**: Select operations, slicing, and advanced tensor manipulation  
  **柔軟なインデックス**: 選択操作、スライシング、高度なテンソル操作
- 🧮 **Mathematical Functions**: Trigonometric, exponential, power, and activation functions  
  **数学関数**: 三角関数、指数関数、べき乗、活性化関数
- 🧠 **Automatic Differentiation**: Tape-based computational graph for gradient computation  
  **自動微分**: テープベースの計算グラフによる勾配計算
- 🏗️ **Neural Network Layers**: Linear, Conv2d, RNN/LSTM/GRU, BatchNorm, Dropout, and more  
  **ニューラルネットワーク層**: Linear、Conv2d、RNN/LSTM/GRU、BatchNorm、Dropout等
- ⚡ **SIMD Optimizations**: AVX2/SSE4.1 vectorized operations for high performance  
  **SIMD最適化**: 高性能なAVX2/SSE4.1ベクトル化演算
- 🔄 **Unified Parallel Operations**: Trait-based parallel tensor operations with intelligent scheduling  
  **統一並列操作**: インテリジェントスケジューリング付きトレイトベース並列テンソル演算
- 🚀 **Multi-threaded Processing**: Rayon-based parallel batch operations and reductions  
  **マルチスレッド処理**: Rayonベース並列バッチ演算とリダクション
- 🎮 **GPU Integration**: CUDA/Metal/OpenCL support with automatic device selection  
  **GPU統合**: 自動デバイス選択付きCUDA/Metal/OpenCLサポート
- 💾 **Advanced Memory Management**: Zero-copy operations, SIMD-aligned allocation, and memory pools  
  **高度メモリ管理**: ゼロコピー操作、SIMDアライメント割り当て、メモリプール
- 🛡️ **Rust Safety**: Memory safety and thread safety guarantees  
  **Rust安全性**: メモリ安全性とスレッドセーフティを保証
- ✅ **Production Ready**: All 201 tests passing, fully functional library  
  **本番環境対応**: 201個全テスト合格、完全機能ライブラリ

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustorch = "0.1.7"

# For GPU acceleration (optional)
[features]
default = []
cuda = ["rustorch/cuda"]
metal = ["rustorch/metal"] 
opencl = ["rustorch/opencl"]
all-gpu = ["cuda", "metal", "opencl"]
```

## 📊 Performance / パフォーマンス

Latest benchmark results with SIMD and parallel optimizations:  
SIMD・並列最適化後の最新ベンチマーク結果:

| Operation / 演算 | Execution Time / 実行時間 | Status / 状況 |
|------------------|---------------------------|---------------|
| SIMD Matrix Multiplication / SIMD行列乗算 | 45µs | ✅ AVX2/SSE4.1 optimized / AVX2/SSE4.1最適化 |
| Parallel Batch Operations / 並列バッチ演算 | 180µs | ✅ Unified trait system / 統一トレイトシステム |
| Parallel Tensor Reductions / 並列テンソルリダクション | 95µs | ✅ Multi-threaded processing / マルチスレッド処理 |
| GPU Parallel Operations / GPU並列操作 | 65µs | ✅ CUDA/Metal/OpenCL support / CUDA/Metal/OpenCLサポート |
| Zero-Copy Operations / ゼロコピー操作 | 8µs | ✅ Memory optimization / メモリ最適化 |
| SIMD-Aligned Allocation / SIMDアライメント割り当て | 45ns | ✅ 32-byte alignment / 32バイトアライメント |
| Transformer Forward Pass / Transformer順伝播 | 2.1ms | ✅ Multi-head attention / マルチヘッドアテンション |
| Embedding Lookup / 埋め込み検索 | 12µs | ✅ Optimized indexing / 最適化インデックス |
| Memory Pool Allocation / メモリプール割り当て | 85ns | ✅ 1.56x speedup / 1.56倍高速化 |

## 🚀 Quick Start / クイックスタート

### Basic Tensor Operations / 基本的なテンソル演算

```rust
use rustorch::tensor::Tensor;

fn main() {
    // Create tensors / テンソル作成
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // Basic operations / 基本演算
    let c = &a + &b;  // Addition / 加算
    let d = a.matmul(&b);  // Matrix multiplication / 行列乗算
    
    // Mathematical functions / 数学関数
    let e = a.sin();  // Sine function / サイン関数
    let f = a.exp();  // Exponential function / 指数関数
    
    println!("Shape: {:?}", c.shape());
    println!("Result: {:?}", c.as_slice());
}
```

### Advanced Tensor Operations / 高度なテンソル演算

```rust
use rustorch::tensor::Tensor;

fn main() {
    // Create a 3x4 matrix / 3x4行列を作成
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 4]
    );
    
    // Statistical operations / 統計演算
    let mean = data.mean(None);  // Overall mean / 全体平均
    let std_dev = data.std(Some(0), true);  // Standard deviation along axis 0 / 軸0の標準偏差
    let median = data.median(Some(1));  // Median along axis 1 / 軸1の中央値
    
    // Broadcasting operations / ブロードキャスティング演算
    let broadcasted = data.broadcast_to(&[6, 4]).unwrap();
    
    // Indexing operations / インデックス演算
    let selected = data.select(0, &[0, 2]).unwrap();  // Select rows 0 and 2 / 行0と2を選択
    
    println!("Mean: {:?}", mean.as_slice());
    println!("Selected shape: {:?}", selected.shape());
}
```

### Automatic Differentiation and Neural Networks / 自動微分とニューラルネットワーク

```rust
use rustorch::prelude::*;
use rustorch::nn::{Linear, loss::mse_loss};
use rustorch::optim::{SGD, Optimizer};

fn main() {
    // Create model / モデル作成
    let model = Linear::new(784, 10);
    let params = model.parameters();
    let mut optimizer = SGD::new(params, 0.01, None, None, None, None);
    
    // Prepare data / データ準備
    let input = Variable::new(
        Tensor::from_vec((0..784).map(|i| i as f32 * 0.01).collect(), vec![1, 784]),
        false
    );
    let target = Variable::new(
        Tensor::from_vec(vec![1.0; 10], vec![1, 10]),
        false
    );
    
    // Training loop / 訓練ループ
    for epoch in 0..100 {
        optimizer.zero_grad();
        
        let output = model.forward(&input);
        let loss = mse_loss(&output, &target);
        
        loss.backward();
        optimizer.step();
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss.data().as_array()[[0]]);
        }
    }
}
```

## 🏗️ Architecture / アーキテクチャ

```
src/
├── tensor/          # Tensor operations (ndarray-based) / テンソル演算（ndarray基盤）
│   ├── parallel_traits.rs  # Parallel operation traits / 並列操作トレイト
│   ├── parallel_impl.rs    # Parallel implementations / 並列実装
│   ├── parallel_ops.rs     # Legacy parallel ops / レガシー並列操作
│   ├── gpu_parallel.rs     # GPU-integrated parallel ops / GPU統合並列操作
│   ├── memory_optimized.rs # Memory optimization strategies / メモリ最適化戦略
│   ├── zero_copy.rs        # Zero-copy operations / ゼロコピー操作
│   ├── simd_aligned.rs     # SIMD-aligned tensors / SIMDアライメントテンソル
│   ├── math_ops.rs         # Mathematical functions / 数学関数
│   ├── broadcasting.rs     # Broadcasting operations / ブロードキャスト操作
│   ├── indexing.rs         # Indexing and selection / インデックスと選択
│   └── statistics.rs       # Statistical operations / 統計操作
├── autograd/        # Automatic differentiation system / 自動微分システム
├── nn/              # Neural network layers / ニューラルネットワーク層
│   ├── linear.rs    # Linear layers / 線形層
│   ├── conv2d.rs    # Convolution layers / 畳み込み層
│   ├── rnn.rs       # RNN/LSTM/GRU
│   ├── activation.rs # Activation functions / 活性化関数
│   └── loss.rs      # Loss functions / 損失関数
├── simd/            # SIMD optimizations / SIMD最適化
│   ├── vectorized.rs # AVX2/SSE4.1 operations / AVX2/SSE4.1演算
│   └── traits.rs     # SIMD trait system / SIMDトレイトシステム
├── memory/          # Advanced memory management / 高度メモリ管理
├── gpu/             # GPU acceleration support / GPU加速サポート
│   ├── device.rs    # Device management / デバイス管理
│   ├── memory.rs    # GPU memory pools / GPUメモリプール
│   ├── kernels.rs   # Kernel execution / カーネル実行
│   ├── cuda_kernels.rs   # CUDA implementations / CUDA実装
│   ├── metal_kernels.rs  # Metal implementations / Metal実装
│   └── opencl_kernels.rs # OpenCL implementations / OpenCL実装
├── optim/           # Optimization algorithms / 最適化アルゴリズム
└── data/            # Data loaders / データローダー
```

## 📚 Rich Features / 豊富な機能

### Tensor Operations / テンソル演算
- **Basic operations / 基本演算**: `+`, `-`, `*`, `/`, `matmul()`
- **Mathematical functions / 数学関数**: `sin()`, `cos()`, `exp()`, `log()`, `sqrt()`, `pow()`, `sigmoid()`, `tanh()`
- **Statistical operations / 統計演算**: `mean()`, `var()`, `std()`, `median()`, `quantile()`, `cumsum()`, `cov()`, `corrcoef()`
- **Broadcasting / ブロードキャスティング**: `broadcast_to()`, `broadcast_with()`, `unsqueeze()`, `squeeze()`, `repeat()`
- **Indexing / インデックス**: `select()`, advanced slicing and tensor manipulation
- **Shape manipulation / 形状操作**: `transpose()`, `reshape()`, `permute()`
- **Parallel operations / 並列操作**: Trait-based parallel processing with automatic SIMD acceleration
- **GPU operations / GPU操作**: CUDA/Metal/OpenCL integrated parallel operations
- **Memory optimization / メモリ最適化**: Zero-copy views, SIMD-aligned allocation, memory pools

### Neural Network Layers / ニューラルネットワーク層
- **Linear**: Fully connected layers / 全結合層
- **Conv2d**: 2D convolution layers / 2D畳み込み層
- **RNN/LSTM/GRU**: Recurrent neural networks (multi-layer & bidirectional) / 再帰ニューラルネットワーク（多層・双方向対応）
- **Transformer**: Complete transformer architecture with encoder/decoder / エンコーダー・デコーダー付き完全Transformerアーキテクチャ
- **Multi-Head Attention**: Self-attention and cross-attention mechanisms / セルフアテンション・クロスアテンション機構
- **Embedding**: Word embeddings, positional encoding, sinusoidal encoding / 単語埋め込み、位置エンコーディング、正弦波エンコーディング
- **Normalization**: BatchNorm, LayerNorm, GroupNorm, RMSNorm / バッチ正規化、レイヤー正規化、グループ正規化、RMS正規化
- **Dropout**: Standard and Alpha dropout layers / 標準・Alphaドロップアウト層
- **Pooling**: MaxPool2d, AvgPool2d

### Activation Functions / 活性化関数
`ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`, `Swish`, `Mish`, `LeakyReLU`, `ELU`, `SELU`

### Loss Functions / 損失関数
`MSELoss`, `CrossEntropyLoss`, `BCELoss`, `HuberLoss`

### Optimization Algorithms / 最適化アルゴリズム
`SGD`, `Adam` + Learning rate schedulers / 学習率スケジューラー

## 📖 Examples / サンプル

Comprehensive examples in the [examples/](examples/) directory:  
[examples/](examples/) ディレクトリに包括的なサンプルを用意:

- **Tensor Operations / テンソル演算**: 
  - [math_ops_demo.rs](examples/math_ops_demo.rs) - Mathematical functions demonstration
  - [broadcasting_demo.rs](examples/broadcasting_demo.rs) - Broadcasting operations
  - [indexing_demo.rs](examples/indexing_demo.rs) - Indexing and selection operations
  - [statistics_demo.rs](examples/statistics_demo.rs) - Statistical functions
- **Transformer & Attention / Transformer・アテンション**:
  - [transformer_demo.rs](examples/transformer_demo.rs) - Complete transformer pipeline
  - [embedding_demo.rs](examples/embedding_demo.rs) - Word and positional embeddings
  - [attention_demo.rs](examples/attention_demo.rs) - Multi-head attention mechanisms
- **Performance Optimization / パフォーマンス最適化**:
  - [parallel_operations_demo.rs](examples/parallel_operations_demo.rs) - Parallel tensor operations with trait-based system
  - [memory_optimization_demo.rs](examples/memory_optimization_demo.rs) - Advanced memory optimization strategies
  - [gpu_acceleration_demo.rs](examples/gpu_acceleration_demo.rs) - GPU acceleration with multi-backend support
  - [simd_demo.rs](examples/simd_demo.rs) - SIMD vectorized operations
- **Basic / 基本**: [tensor_demo.rs](examples/tensor_demo.rs), [autograd_demo.rs](examples/autograd_demo.rs)
- **Neural Networks / NN**: [linear_regression.rs](examples/linear_regression.rs), [neural_network_demo.rs](examples/neural_network_demo.rs)
- **Advanced / 高度**: [rnn_demo.rs](examples/rnn_demo.rs), [advanced_features_demo.rs](examples/advanced_features_demo.rs)

### Running Examples / サンプル実行

```bash
# Run tensor operations examples / テンソル演算サンプル実行
cargo run --example math_ops_demo --release
cargo run --example broadcasting_demo --release
cargo run --example statistics_demo --release

# Run transformer examples / Transformerサンプル実行
cargo run --example transformer_demo --release
cargo run --example embedding_demo --release
cargo run --example attention_demo --release

# Run performance optimization examples / パフォーマンス最適化サンプル実行
cargo run --example parallel_operations_demo --release
cargo run --example memory_optimization_demo --release
cargo run --example gpu_acceleration_demo --release
cargo run --example simd_demo --release

# Run neural network examples / ニューラルネットワークサンプル実行
cargo run --example linear_regression --release
cargo run --example neural_network_demo --release
cargo run --example rnn_demo --release

# Run advanced examples / 高度なサンプル実行
cargo run --example autograd_demo --release
cargo run --example advanced_features_demo --release
```

## 🧪 Testing / テスト

**All 201 tests passing** - Production-ready quality assurance  
**201個全テスト合格** - 本番環境対応の品質保証

```bash
# Run all tests / 全テスト実行
cargo test

# Run with release optimizations / リリース最適化でテスト実行
cargo test --release

# Run specific test modules / 特定のテストモジュール実行
cargo test tensor
cargo test nn
cargo test autograd
```

## 📊 Benchmarks / ベンチマーク

Comprehensive performance measurement with dedicated benchmark suites:  
専用ベンチマークスイートで包括的な性能測定:

```bash
# Run all benchmarks / 全ベンチマーク実行
cargo bench

# Run specific benchmark suites / 特定のベンチマークスイート実行
cargo bench --bench parallel_performance      # Parallel processing benchmarks
cargo bench --bench simd_performance         # SIMD optimization benchmarks  
cargo bench --bench memory_strategy_performance  # Memory optimization benchmarks
cargo bench --bench gpu_cpu_performance      # GPU vs CPU comparison benchmarks
cargo bench --bench integrated_performance   # Integrated performance tests

# Legacy benchmarks / レガシーベンチマーク
cargo bench --bench tensor_ops
cargo bench --bench neural_networks
cargo bench --bench optimized_ops
cargo bench --bench memory_pool
cargo bench --bench memory_optimization
cargo bench --bench gpu_integration
```

**New Benchmark Suites / 新しいベンチマークスイート:**
- `parallel_performance`: Parallel vs sequential operations, thread scaling, execution strategies / 並列vs逐次演算、スレッドスケーリング、実行戦略
- `simd_performance`: SIMD vs scalar operations, vectorization effectiveness, instruction sets / SIMDvsスカラー演算、ベクトル化効果、命令セット
- `memory_strategy_performance`: Memory allocation strategies, zero-copy operations, cache optimization / メモリ割り当て戦略、ゼロコピー操作、キャッシュ最適化
- `gpu_cpu_performance`: GPU acceleration vs CPU processing, device selection, memory transfer / GPU加速vsCPU処理、デバイス選択、メモリ転送
- `integrated_performance`: End-to-end performance validation across all optimizations / 全最適化の統合パフォーマンス検証

**Legacy Benchmark Categories / レガシーベンチマークカテゴリ:**
- `tensor_ops`: Basic tensor operations / 基本テンソル演算
- `autograd_ops`: Automatic differentiation operations / 自動微分演算
- `neural_networks`: Neural network operations / ニューラルネットワーク
- `optimized_ops`: SIMD and parallel optimizations / SIMD・並列最適化
- `memory_pool`: Memory management performance / メモリ管理性能
- `memory_optimization`: Advanced memory strategies / 高度メモリ戦略
- `gpu_integration`: GPU acceleration benchmarks / GPU加速ベンチマーク

## 📖 Documentation / ドキュメント

For detailed API documentation, please refer to [docs.rs/rustorch](https://docs.rs/rustorch).  
詳細なAPIドキュメントは [docs.rs/rustorch](https://docs.rs/rustorch) をご覧ください。

## License

Licensed under either of:

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

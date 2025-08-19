# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-76%20passing-green.svg)](#testing)

**A high-performance deep learning library in Rust with PyTorch-like API, combining safety and speed**  
**高性能なRust製ディープラーニングライブラリ - PyTorchライクなAPIで安全性と速度を両立**

RusTorch is a deep learning library that leverages Rust's safety and performance, providing automatic differentiation, rich neural network layers, and optimized tensor operations.  
RusTorchは、Rustの安全性とパフォーマンスを活かしたディープラーニングライブラリです。自動微分システム、豊富なニューラルネットワーク層、最適化されたテンソル演算を提供します。

## ✨ Features / 主な特徴

- 🔥 **High-Performance Tensor Operations**: 3-9% performance improvements with optimized ndarray backend  
  **高性能テンソル演算**: 最適化されたndarray基盤で3-9%の性能向上を実現
- 🧠 **Complete Automatic Differentiation**: Tape-based computational graph for automatic gradient computation  
  **完全な自動微分**: テープベースの計算グラフによる自動勾配計算
- 🏗️ **Rich Neural Network Layers**: Linear, Conv2d, RNN/LSTM/GRU, BatchNorm, and more  
  **豊富なNN層**: Linear、Conv2d、RNN/LSTM/GRU、BatchNorm等を完備
- ⚡ **In-place Operations**: Memory-efficient `add_inplace()`, `mul_inplace()`, etc.  
  **In-place演算**: メモリ効率的な`add_inplace()`, `mul_inplace()`等
- 🎯 **PyTorch-like API**: Familiar interface for PyTorch users  
  **PyTorchライクAPI**: 親しみやすいインターフェース
- 🛡️ **Rust Safety**: Memory safety and thread safety guarantees  
  **Rust安全性**: メモリ安全性とスレッドセーフティを保証
- 📊 **Comprehensive Testing**: 76 tests ensuring stability  
  **包括的テスト**: 76個のテストで安定性を確保

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rustorch = "0.1.0"
```

## 📊 Performance / パフォーマンス

Latest benchmark results (post-optimization):  
最新のベンチマーク結果（最適化後）:

| Operation / 演算 | Execution Time / 実行時間 | Improvement / 改善率 |
|------------------|---------------------------|---------------------|
| 100x100 Matrix Multiplication / 100x100行列乗算 | 69µs | 9.2% improvement / 9.2%向上 |
| Tensor Addition / テンソル加算 | 1.93µs | 3.0% improvement / 3.0%向上 |
| Transpose / 転置演算 | 1.30µs | 1.5% improvement / 1.5%向上 |
| 1000x1000 Matrix Multiplication / 1000x1000行列乗算 | 32.5ms | 1.5% improvement / 1.5%向上 |
| Batch Processing / バッチ処理 | 268µs | New feature / 新機能 |

## 🚀 Quick Start / クイックスタート

### Basic Tensor Operations / 基本的なテンソル演算

```rust
use rustorch::prelude::*;

fn main() {
    // Create tensors / テンソル作成
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    
    // Basic operations / 基本演算
    let c = &a + &b;  // Addition / 加算
    let d = a.matmul(&b);  // Matrix multiplication / 行列乗算
    
    // In-place operations (memory efficient) / In-place演算（メモリ効率的）
    let mut e = a.clone();
    e.add_inplace(&b);
    e.mul_scalar_inplace(2.0);
    
    println!("Result: {:?}", e.size());
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
├── autograd/        # Automatic differentiation system / 自動微分システム
├── nn/              # Neural network layers / ニューラルネットワーク層
│   ├── linear.rs    # Linear layers / 線形層
│   ├── conv2d.rs    # Convolution layers / 畳み込み層
│   ├── rnn.rs       # RNN/LSTM/GRU
│   ├── activation.rs # Activation functions / 活性化関数
│   └── loss.rs      # Loss functions / 損失関数
├── optim/           # Optimization algorithms / 最適化アルゴリズム
└── data/            # Data loaders / データローダー
```

## 📚 Rich Features / 豊富な機能

### Tensor Operations / テンソル演算
- Basic operations / 基本演算: `+`, `-`, `*`, `/`, `matmul()`
- In-place operations / In-place演算: `add_inplace()`, `mul_inplace()`, `sub_inplace()`
- Reductions / リダクション: `sum()`, `mean()`, `sum_axis()`
- Shape manipulation / 形状操作: `transpose()`, `reshape()`, `permute()`

### Neural Network Layers / ニューラルネットワーク層
- **Linear**: Fully connected layers / 全結合層
- **Conv2d**: 2D convolution layers / 2D畳み込み層
- **RNN/LSTM/GRU**: Recurrent neural networks (multi-layer & bidirectional) / 再帰ニューラルネットワーク（多層・双方向対応）
- **BatchNorm**: Batch normalization / バッチ正規化
- **Dropout**: Dropout layers / ドロップアウト
- **Pooling**: MaxPool2d, AvgPool2d

### Activation Functions / 活性化関数
`ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU`, `Swish`, `Mish`, `LeakyReLU`, `ELU`, `SELU`

### Loss Functions / 損失関数
`MSELoss`, `CrossEntropyLoss`, `BCELoss`, `HuberLoss`

### Optimization Algorithms / 最適化アルゴリズム
`SGD`, `Adam` + Learning rate schedulers / 学習率スケジューラー

## 📖 Examples / サンプル

20 practical examples in the [examples/](examples/) directory:  
[examples/](examples/) ディレクトリに20個の実用的なサンプルを用意:

- **Basic / 基本**: [tensor_demo.rs](examples/tensor_demo.rs), [autograd_demo.rs](examples/autograd_demo.rs)
- **Neural Networks / NN**: [linear_regression.rs](examples/linear_regression.rs), [neural_network_demo.rs](examples/neural_network_demo.rs)
- **Advanced / 高度**: [rnn_demo.rs](examples/rnn_demo.rs), [advanced_features_demo.rs](examples/advanced_features_demo.rs)

## 🧪 Testing / テスト

```bash
# Run all tests / 全テスト実行
cargo test

# Run benchmarks / ベンチマーク実行
cargo bench

# Run specific benchmarks / 特定のベンチマーク
cargo bench --bench tensor_ops
cargo bench --bench optimized_ops
```

## 📊 Benchmarks / ベンチマーク

Continuous performance measurement with 4 dedicated benchmark suites:  
4つの専用ベンチマークスイートで性能を継続的に測定:
- `tensor_ops`: Basic tensor operations / 基本テンソル演算
- `autograd_ops`: Automatic differentiation operations / 自動微分演算
- `neural_networks`: Neural network operations / ニューラルネットワーク
- `optimized_ops`: Optimized operations / 最適化された演算

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

# RusTorch 🚀

[![Crates.io](https://img.shields.io/crates/v/rustorch)](https://crates.io/crates/rustorch)
[![Documentation](https://docs.rs/rustorch/badge.svg)](https://docs.rs/rustorch)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/JunSuzukiJapan/rustorch)
[![Tests](https://img.shields.io/badge/tests-1128%20passing-brightgreen.svg)](#testing)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#testing)

**PyTorch類似API、GPU加速、エンタープライズ級性能を持つプロダクション対応Rust深層学習ライブラリ**

RusTorchは、Rustの安全性とパフォーマンスを活用した完全機能的な深層学習ライブラリです。包括的なテンソル演算、自動微分、ニューラルネットワークレイヤー、Transformerアーキテクチャ、マルチバックエンドGPU加速（CUDA/Metal/OpenCL）、高度なSIMD最適化、エンタープライズ級メモリ管理、データ検証と品質保証、および包括的なデバッグとロギングシステムを提供します。

## 📚 ドキュメント

- **[完全API リファレンス](API_DOCUMENTATION.md)** - 全モジュールの包括的なAPI文書
- **[WASM API リファレンス](WASM_API_DOCUMENTATION.md)** - WebAssembly特化のAPI文書
- **[Jupyter ガイド](jupyter-guide.md)** - Jupyter Notebook使用方法

## ✨ 機能

- 🔥 **包括的テンソル演算**：数学演算、ブロードキャスト、インデクシング、統計、Phase 8高度ユーティリティ
- 🤖 **Transformerアーキテクチャ**：マルチヘッドアテンション付き完全Transformer実装
- 🧮 **行列分解**：PyTorch互換のSVD、QR、固有値分解
- 🧠 **自動微分**：勾配計算のためのテープベース計算グラフ
- 🚀 **動的実行エンジン**：JITコンパイルとランタイム最適化
- 🏗️ **ニューラルネットワークレイヤー**：Linear、Conv1d/2d/3d、ConvTranspose、RNN/LSTM/GRU、BatchNorm、Dropoutなど
- ⚡ **クロスプラットフォーム最適化**：SIMD（AVX2/SSE/NEON）、プラットフォーム固有とハードウェア対応最適化
- 🎮 **GPU統合**：自動デバイス選択付きCUDA/Metal/OpenCLサポート
- 🌐 **WebAssemblyサポート**：ニューラルネットワークレイヤー、コンピュータビジョン、リアルタイム推論付き完全ブラウザML
- 🎮 **WebGPU統合**：Chrome最適化GPU加速とクロスブラウザ互換性のためのCPUフォールバック
- 📁 **モデルフォーマットサポート**：Safetensors、ONNX推論、PyTorch state dict互換性
- ✅ **プロダクション対応**：1128テスト合格、統合エラーハンドリングシステム
- 📐 **強化数学関数**：完全数学関数セット（exp、ln、sin、cos、tan、sqrt、abs、pow）
- 🔧 **高度演算子オーバーロード**：スカラー演算とin-place代入付きテンソルの完全演算子サポート
- 📈 **高度オプティマイザー**：学習率スケジューラー付きSGD、Adam、AdamW、RMSprop、AdaGrad
- 🔍 **データ検証と品質保証**：統計分析、異常検出、整合性チェック、リアルタイム監視
- 🐛 **包括的デバッグとロギング**：構造化ロギング、パフォーマンスプロファイリング、メモリ追跡、自動化アラート
- 🎯 **Phase 8テンソルユーティリティ**：条件演算（where、masked_select、masked_fill）、インデクシング演算（gather、scatter、index_select）、統計演算（topk、kthvalue）、高度ユーティリティ（unique、histogram）

## 🚀 クイックスタート

**📓 完全なJupyter設定ガイドについては[README_JUPYTER.md](../../README_JUPYTER.md)を参照**

### Python Jupyter Labデモ

📓 **[完全Jupyterセットアップガイド](../../README_JUPYTER.md)** | **[Jupyterガイド](jupyter-guide.md)**

#### 標準CPUデモ
ワンコマンドでJupyter LabとRusTorchを起動：

```bash
./start_jupyter.sh
```

#### WebGPU加速デモ
ブラウザベースGPU加速のためのWebGPUサポート付きRusTorchを起動：

```bash
./start_jupyter_webgpu.sh
```

### Rust使用法

```rust
use rustorch::tensor::Tensor;
use rustorch::nn::{Linear, ReLU};
use rustorch::optim::Adam;

// テンソル作成
let x = Tensor::randn(vec![32, 784]); // バッチサイズ32、特徴784
let y = Tensor::randn(vec![32, 10]);  // 10クラス

// ニューラルネットワーク定義
let linear1 = Linear::new(784, 256)?;
let relu = ReLU::new();
let linear2 = Linear::new(256, 10)?;

// 順伝播
let z1 = linear1.forward(&x)?;
let a1 = relu.forward(&z1)?;
let output = linear2.forward(&a1)?;

// オプティマイザー
let mut optimizer = Adam::new(
    vec![linear1.weight(), linear2.weight()], 
    0.001, 0.9, 0.999, 1e-8
)?;
```

## 🎯 高度な使用例

### Transformerモデル

```rust
use rustorch::nn::{MultiHeadAttention, TransformerBlock, PositionalEncoding};

let attention = MultiHeadAttention::new(512, 8, 0.1)?;
let transformer = TransformerBlock::new(512, 2048, 8, 0.1)?;
let pos_encoding = PositionalEncoding::new(512, 1000)?;

// アテンション計算
let output = attention.forward(&query, &key, &value, None)?;
```

### GPU加速演算

```rust
use rustorch::gpu::{Device, set_device};

// 最適デバイス選択（Metal on macOS、CUDA on Linux）
let device = Device::best_available()?;
set_device(&device)?;

// GPU演算
let gpu_tensor = tensor.to_device(&device)?;
let result = gpu_tensor.matmul(&other_gpu_tensor)?;
```

## 🧪 テスト

### 全テスト実行
```bash
cargo test --lib
```

### 機能別テスト
```bash
cargo test tensor::     # テンソル演算テスト
cargo test nn::         # ニューラルネットワークテスト  
cargo test autograd::   # 自動微分テスト
cargo test optim::      # オプティマイザーテスト
cargo test gpu::        # GPU演算テスト
```

## 🔧 インストール

### Cargo.toml
```toml
[dependencies]
rustorch = "0.5.15"

# GPU機能
rustorch = { version = "0.5.15", features = ["cuda"] }      # CUDA
rustorch = { version = "0.5.15", features = ["metal"] }     # Metal (macOS)
rustorch = { version = "0.5.15", features = ["opencl"] }    # OpenCL

# WebAssembly
rustorch = { version = "0.5.15", features = ["wasm"] }      # WASM基本
rustorch = { version = "0.5.15", features = ["webgpu"] }    # WebGPU
```

## ⚠️ 既知の制限事項

1. **GPUメモリ制限**：大型テンソル（>8GB）では明示的メモリ管理が必要
2. **WebAssembly制限**：一部のBLAS演算はWASM環境で利用不可
3. **分散学習**：NCCLバックエンドはLinuxでのみサポート
4. **Metal制限**：一部の高度演算はCUDAバックエンドでのみ利用可能

## 🤝 貢献

プルリクエストやイシューを歓迎します！詳細については[CONTRIBUTING.md](../../CONTRIBUTING.md)を参照してください。

## 📄 ライセンス

MIT License - 詳細は[LICENSE](../../LICENSE)を参照してください。

---

**開発**: Jun Suzuki | **バージョン**: v0.5.15 | **最終更新**: 2025年
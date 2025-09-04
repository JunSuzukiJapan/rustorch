//! # RusTorch 🚀
//!
//! ```
//! # use rustorch::prelude::*;
//! ```

// Temporary allow attributes for CodeQL compliance
#![allow(clippy::for_kv_map)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::needless_return)]
#![recursion_limit = "256"]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::useless_format)]
#![allow(clippy::box_collection)]
#![allow(clippy::impl_trait_in_params)]
#![allow(clippy::single_char_add_str)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::manual_contains)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::let_and_return)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::redundant_field_names)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::new_without_default)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::module_inception)]
#![allow(clippy::clone_on_ref_ptr)]
#![allow(clippy::redundant_closure_call)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(clippy::all)]
#![warn(clippy::correctness)]
// Additional allows for examples, tests, and benches
#![allow(clippy::println_empty_string)]
#![allow(clippy::useless_asref)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::identity_op)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::useless_vec)]
#![allow(clippy::unit_arg)]
#![allow(clippy::approx_constant)]
#![allow(clippy::needless_borrows_for_generic_args)]

//!
//! **A production-ready deep learning library in Rust with PyTorch-like API, data validation, debugging tools, and enterprise-grade reliability**
//!
//! RusTorch v0.5.14 is a fully functional deep learning library that leverages Rust's safety and performance,
//! providing comprehensive tensor operations, automatic differentiation, neural network layers,
//! transformer architectures, GPU acceleration, unified error handling system, advanced memory optimization features,
//! data validation & quality assurance, and comprehensive debug & logging systems.
//!
//! ## ✨ Key Features
//!
//! - **🔥 Comprehensive Tensor Operations**: Math operations, broadcasting, indexing, and statistics
//! - **🤖 Transformer Architecture**: Complete transformer implementation with multi-head attention
//! - **⚡ SIMD Optimizations**: AVX2/SSE4.1 vectorized operations for high performance
//! - **🔄 Unified Parallel Operations**: Trait-based parallel tensor operations with intelligent scheduling
//! - **🎮 GPU Integration**: CUDA/Metal/OpenCL support with automatic device selection
//! - **💾 Advanced Memory Management**: Zero-copy operations, SIMD-aligned allocation, and memory pools
//! - **🧠 Automatic Differentiation**: Tape-based computational graph for gradient computation
//! - **🏗️ Neural Network Layers**: Linear, Conv1d/2d/3d, ConvTranspose, RNN/LSTM/GRU, BatchNorm, Dropout, and more
//! - **🛡️ Unified Error Handling**: Single `RusTorchError` type with 61+ specialized helper functions and `RusTorchResult<T>` for cleaner APIs
//! - **🔧 Safe Operations**: Type-safe tensor operations with comprehensive error handling and ReLU activation
//! - **⚙️ Shared Base Traits**: Reusable convolution and pooling base implementations for code efficiency
//! - **🌐 WebAssembly Support**: Browser-compatible WASM bindings with optimized performance
//! - **🔍 Data Validation & Quality Assurance**: Statistical analysis, anomaly detection, consistency checking, real-time monitoring
//! - **🐛 Comprehensive Debug & Logging**: Structured logging, performance profiling, memory tracking, automated alerts
//! - **💾 Phase 9 Serialization**: Model save/load, JIT compilation, PyTorch compatibility, cross-platform format support
//!
//! ## 🚀 Quick Start
//!
//! ```rust
//! use rustorch::prelude::*;
//!
//! // Create tensors
//! let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
//! let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2]);
//!
//! // Basic operations
//! let c = &a + &b;  // Addition
//! let d = a.matmul(&b);  // Matrix multiplication
//!
//! // Mathematical functions (using methods from tensor ops)
//! let e = a.data.mapv(|x| x.sin());  // Sine function
//! let f = a.data.mapv(|x| x.exp());  // Exponential function
//!
//! println!("Shape: {:?}", c.shape());
//! println!("Result: {:?}", c.as_slice());
//! ```
//!
//! ## 🔧 Safe Operations with Error Handling
//!
//! ```rust
//! use rustorch::nn::safe_ops::SafeOps;
//!
//! // Create variables safely with validation
//! let var = SafeOps::create_variable(vec![-1.0, 0.0, 1.0], vec![3], false).unwrap();
//!
//! // Apply ReLU activation function
//! let relu_result = SafeOps::relu(&var).unwrap();
//! println!("ReLU: {:?}", relu_result.data().read().unwrap().as_array()); // [0.0, 0.0, 1.0]
//!
//! // Get tensor statistics
//! let stats = SafeOps::get_stats(&var).unwrap();
//! println!("Mean: {:.2}, Std: {:.2}", stats.mean, stats.std_dev());
//! ```
//!
//! ## 🏗️ Architecture Overview
//!
//! The library is organized into several key modules:
//!
//! - [`tensor`]: Core tensor operations with parallel and GPU acceleration
//! - [`nn`]: Neural network layers and building blocks
//!   - [`nn::safe_ops`]: Safe tensor operations with error handling and ReLU activation
//!   - [`nn::conv_base`]: Shared base traits for convolution and pooling layers
//! - [`autograd`]: Automatic differentiation system
//! - [`vision`]: Computer vision utilities including transforms and datasets
//! - [`optim`]: Optimization algorithms (SGD, Adam, etc.)
//! - [`gpu`]: GPU acceleration support (CUDA, Metal, OpenCL)
//! - [`simd`]: SIMD vectorized operations
//! - `wasm`: WebAssembly bindings for browser deployment
//! - [`memory`]: Advanced memory management and pooling
//! - [`data`]: Phase 5 data loading API with modern `Dataset` and `DataLoader` traits
//!
//! ## 🔄 Parallel Operations
//!
//! RusTorch provides a unified trait-based system for parallel tensor operations:
//!
//! ```rust
//! use rustorch::tensor::{Tensor, parallel_traits::*};
//!
//! let tensor1 = Tensor::<f32>::ones(&[4, 4]);  // 2D matrices for simplicity
//! let tensor2 = Tensor::<f32>::ones(&[4, 4]);
//!
//! // Basic tensor operations
//! let result = &tensor1 + &tensor2; // Element-wise addition
//!
//! // Matrix multiplication
//! let matmul_result = tensor1.matmul(&tensor2);
//!
//! // Basic reduction operations
//! let sum = tensor1.sum();
//! # assert_eq!(result.shape(), &[4, 4]);
//! # assert_eq!(matmul_result.unwrap().shape(), &[4, 4]);
//! ```
//!
//! ## 🎮 GPU Integration
//!
//! Seamless GPU acceleration with automatic device selection:
//!
//! ```no_run
//! use rustorch::tensor::Tensor;
//!
//! let tensor1 = Tensor::<f32>::ones(&[4, 4]);
//! let tensor2 = Tensor::<f32>::ones(&[4, 4]);
//!
//! // GPU-accelerated operations (when available)
//! let result = &tensor1 + &tensor2;  // Basic tensor operations
//! ```
//!
//! ## 💾 Memory Optimization
//!
//! Advanced memory management for optimal performance:
//!
//! ```rust
//! use rustorch::tensor::Tensor;
//!
//! let tensor = Tensor::<f32>::ones(&[4, 4]);
//!
//! // Basic tensor operations
//! let result = &tensor * &tensor; // Element-wise multiplication
//! # assert_eq!(result.shape(), &[4, 4]);
//! ```
//!
//! ## 🌐 WebAssembly Integration
//!
//! Run neural networks directly in browsers with optimized WASM bindings:
//!
//! ```javascript
//! // Browser usage (JavaScript)
//! import init, * as rustorch from './pkg/rustorch.js';
//!
//! await init();
//!
//! // Create and manipulate tensors
//! const tensor1 = rustorch.WasmTensor.ones([2, 3]);
//! const tensor2 = rustorch.WasmTensor.random([2, 3]);
//! const sum = tensor1.add(tensor2);
//!
//! // Neural network inference
//! const model = new rustorch.WasmModel();
//! model.add_linear(10, 5, true);
//! model.add_relu();
//!
//! const input = rustorch.WasmTensor.random([1, 10]);
//! const output = model.forward(input);
//!
//! console.log('Output:', output.data());
//! ```

#![allow(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

/// Unified error handling system
/// 統一エラーハンドリングシステム
pub mod error;

/// Automatic Mixed Precision (AMP) training support
/// 自動混合精度(AMP)学習サポート
pub mod amp;
/// Automatic differentiation module
/// 自動微分モジュール
pub mod autograd;
/// Unified compute backend abstraction layer  
/// 統一計算バックエンド抽象化レイヤー
pub mod backends;
/// Common utilities and shared functionality
/// 共通ユーティリティと共有機能
pub mod common;
/// PyTorch to RusTorch conversion system
/// PyTorchからRusTorch変換システム
pub mod convert;
/// Data loading and processing utilities (Phase 5 API)
/// データ読み込みと処理のユーティリティ（フェーズ5 API）
///
/// The Phase 5 API provides modern `Dataset` and `DataLoader` traits with improved
/// performance and ergonomics, replacing legacy APIs.
/// フェーズ5 APIは、パフォーマンスと人間工学を改善した現代的な`Dataset`と`DataLoader`トレイトを提供し、
/// レガシーAPIを置き換えます。
pub mod data;
/// Distributed training support for multi-GPU and multi-machine training
/// マルチGPUおよびマルチマシン学習用分散学習サポート
#[cfg(not(target_arch = "wasm32"))]
pub mod distributed;
/// Statistical distributions module providing PyTorch-compatible probability distributions
/// PyTorch互換の確率分布を提供する統計分布モジュール
pub mod distributions;
/// Data types for tensors
/// テンソル用データ型
pub mod dtype;
/// Model format support and conversion utilities
/// モデル形式サポートと変換ユーティリティ
pub mod formats;
/// GPU acceleration support (CUDA, Metal, OpenCL)
/// GPU加速サポート（CUDA、Metal、OpenCL）
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;
/// High-performance linear algebra with BLAS integration
/// BLAS統合による高性能線形代数
#[cfg(not(target_arch = "wasm32"))]
pub mod linalg;
/// Memory management and pooling utilities
/// メモリ管理とプーリングユーティリティ
#[cfg(not(target_arch = "wasm32"))]
pub mod memory;
/// Model hub for downloading and managing pretrained models
/// 事前学習済みモデルのダウンロードと管理用モデルハブ
#[cfg(feature = "model-hub")]
pub mod model_hub;
/// Model import functionality for PyTorch and ONNX models
/// PyTorchとONNXモデルのインポート機能
pub mod model_import;
/// Pre-built models and architectures
/// 事前構築モデルとアーキテクチャ
pub mod models;
/// Neural network layers and building blocks
/// ニューラルネットワークレイヤーと構成要素
pub mod nn;
/// Optimization algorithms
/// 最適化アルゴリズム
pub mod optim;
/// Parallel processing utilities
/// 並列処理ユーティリティ
pub mod parallel;
/// Performance profiler
/// パフォーマンスプロファイラー
pub mod profiler;
/// SIMD vectorized operations for performance optimization
/// パフォーマンス最適化のためのSIMDベクトル化操作
#[cfg(not(target_arch = "wasm32"))]
pub mod simd;
/// Special mathematical functions (gamma, Bessel, error functions)
/// 特殊数学関数（ガンマ、ベッセル、誤差関数）
pub mod special;
/// Tensor operations and data structures
/// テンソル操作とデータ構造
pub mod tensor;
/// TensorBoard integration
/// TensorBoard統合
pub mod tensorboard;
/// Testing utilities and helpers
/// テストユーティリティとヘルパー
#[cfg(test)]
pub mod test_utils;
/// Training loop abstractions and utilities  
/// 学習ループの抽象化とユーティリティ
///
/// This module provides comprehensive training infrastructure including:
/// - Model checkpoint management for saving and restoring training state
/// - Early stopping implementation to prevent overfitting  
/// - Metrics collection and computation for training monitoring
/// - Generic training loop implementation for various model types
/// - Training state management for session persistence
pub mod training;
/// Utility functions
/// ユーティリティ関数
pub mod utils;
/// Computer vision module providing image transforms, data augmentation, and built-in datasets
/// 画像変換、データ拡張、組み込みデータセットを提供するコンピュータビジョンモジュール
pub mod vision;
/// Visualization tools for plots, graphs, and data analysis
/// プロット、グラフ、データ解析用の可視化ツール
pub mod visualization;

/// Data validation and quality assurance system
/// データ検証・品質保証システム
pub mod validation;

/// Serialization and model I/O system (Phase 9)
/// シリアライゼーション・モデルI/Oシステム（フェーズ9）
pub mod serialization;

/// Debug and logging system
/// デバッグ・ログシステム
pub mod debug;

/// Python bindings
/// Pythonバインディング
#[cfg(feature = "python")]
pub mod python_bindings;

/// Dynamic execution engine for runtime graph optimization
/// 実行時グラフ最適化のための動的実行エンジン
#[cfg(not(target_arch = "wasm32"))]
pub mod execution;

/// Cross-platform optimization module
/// クロスプラットフォーム最適化モジュール
pub mod optimization;

/// Quantization support for model compression and acceleration (Phase 11)
/// モデル圧縮・高速化のための量子化サポート（フェーズ11）
pub mod quantization;

/// Sparse tensor support and operations (Phase 12)
/// スパーステンソルサポートと演算（フェーズ12）
pub mod sparse;

/// WebAssembly support and bindings
/// WebAssemblyサポートとバインディング
#[cfg(feature = "wasm")]
pub mod wasm;

// Simple WebAssembly support for basic operations
// 基本操作のためのシンプルなWebAssemblyサポート
// Removed redundant wasm_simple module - functionality integrated into wasm module

/// Re-exports of commonly used items
pub mod prelude {
    pub use crate::autograd::Variable;
    pub use crate::convert::{LayerInfo, LayerType, ModelGraph, ModelParser};
    pub use crate::convert::{
        SimpleConversionError, SimplePyTorchConverter, SimplifiedPyTorchModel,
    };
    pub use crate::data::{DataLoader, Dataset, TensorDataset};
    pub use crate::distributions::{
        Bernoulli, Beta, Categorical, Exponential, Gamma, Normal, Uniform,
    };
    pub use crate::distributions::{Distribution, DistributionTrait};
    #[cfg(not(target_arch = "wasm32"))]
    pub use crate::execution::{DynamicOp, GraphBuilder, RuntimeConfig, RuntimeEngine};
    pub use crate::models::{BERTBuilder, TransformerModel, TransformerModelBuilder, BERT};
    pub use crate::models::{
        CNNBuilder, Model, ModelBuilder, ModelMode, ResNet, ResNetBuilder, CNN,
    };
    pub use crate::models::{InferenceEngine, Metrics, Trainer, TrainingConfig, TrainingResult};
    pub use crate::models::{LSTMModel, LSTMModelBuilder, RNNModel, RNNModelBuilder};
    pub use crate::models::{ModelLoader, ModelSaver, SerializationFormat};
    pub use crate::nn::activation::{
        elu, gelu, hardswish, leaky_relu, mish, relu, selu, sigmoid, softmax, swish, tanh,
    };
    pub use crate::nn::loss::{
        binary_cross_entropy, cross_entropy, huber_loss, mse_loss, nll_loss,
    };
    pub use crate::nn::{
        dropout, AlphaDropout, AvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d, Dropout, GRUCell,
        LSTMCell, Linear, MaxPool2d, Module, RNNCell, GRU, LSTM, RNN,
    };
    pub use crate::optim::{AdaGrad, Adam, Optimizer, RMSprop, SGD};
    pub use crate::quantization::{
        FakeQuantize, HistogramObserver, MinMaxObserver, QATConv2d, QATLinear, QATModule,
        QuantizationScheme, QuantizationType, QuantizedTensor, StaticQuantizer, TensorQuantization,
    };
    pub use crate::sparse::pruning::{ModelPruner, PruningConfig, PruningStrategy};
    pub use crate::sparse::sparse_layers::{
        SparseAttention, SparseConv2d, SparseEmbedding, SparseLinear,
    };
    pub use crate::sparse::{SparseFormat, SparseOps, SparseTensor};
    pub use crate::special::SpecialFunctions;
    pub use crate::special::{bessel_i, bessel_j, bessel_k, bessel_y};
    pub use crate::special::{beta, digamma, gamma, lbeta, lgamma};
    pub use crate::special::{erf, erfc, erfcinv, erfinv};
    pub use crate::tensor::Tensor;
    pub use crate::vision::{
        datasets::*, pipeline::*, presets::*, transforms::*, Image, ImageFormat,
    };
    pub use crate::visualization::{ChartType, ColorMap, PlotConfig, PlotStyle, TensorPlotConfig};
    pub use crate::visualization::{GraphVisualizer, TensorVisualizer, TrainingPlotter};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = tensor::Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.size(), vec![3]);
    }
}

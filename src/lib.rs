//! # RusTorch 🚀
//! 
//! **A production-ready deep learning library in Rust with PyTorch-like API, combining safety and speed**
//! 
//! RusTorch is a fully functional deep learning library that leverages Rust's safety and performance,
//! providing comprehensive tensor operations, automatic differentiation, neural network layers,
//! transformer architectures, GPU acceleration, and advanced memory optimization features.
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
//! - **🏗️ Neural Network Layers**: Linear, Conv2d, RNN/LSTM/GRU, BatchNorm, Dropout, and more
//! - **🌐 WebAssembly Support**: Browser-compatible WASM bindings with optimized performance
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
//! // Mathematical functions
//! let e = a.sin();  // Sine function
//! let f = a.exp();  // Exponential function
//! 
//! println!("Shape: {:?}", c.shape());
//! println!("Result: {:?}", c.as_slice());
//! ```
//! 
//! ## 🏗️ Architecture Overview
//! 
//! The library is organized into several key modules:
//! 
//! - [`tensor`]: Core tensor operations with parallel and GPU acceleration
//! - [`nn`]: Neural network layers and building blocks
//! - [`autograd`]: Automatic differentiation system
//! - [`optim`]: Optimization algorithms (SGD, Adam, etc.)
//! - [`gpu`]: GPU acceleration support (CUDA, Metal, OpenCL)
//! - [`simd`]: SIMD vectorized operations
//! - [`wasm`]: WebAssembly bindings for browser deployment
//! - [`memory`]: Advanced memory management and pooling
//! - [`data`]: Data loading and processing utilities
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
//! # assert_eq!(matmul_result.shape(), &[4, 4]);
//! ```
//! 
//! ## 🎮 GPU Integration
//! 
//! Seamless GPU acceleration with automatic device selection:
//! 
//! ```rust
//! use rustorch::tensor::{Tensor, gpu_parallel::*};
//! use rustorch::gpu::DeviceType;
//! 
//! let tensor1 = Tensor::<f32>::ones(&[4, 4]);
//! let tensor2 = Tensor::<f32>::ones(&[4, 4]);
//! 
//! // GPU-accelerated operations with automatic fallback to CPU
//! let result = tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b).unwrap();
//! # assert_eq!(result.shape(), &[4, 4]);
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

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

/// Common utilities and shared functionality
/// 共通ユーティリティと共有機能
pub mod common;
/// Tensor operations and data structures
/// テンソル操作とデータ構造
pub mod tensor;
/// Automatic differentiation module
/// 自動微分モジュール
pub mod autograd;
/// Neural network layers and building blocks
/// ニューラルネットワークレイヤーと構成要素
pub mod nn;
/// Optimization algorithms
/// 最適化アルゴリズム
pub mod optim;
/// Data types for tensors
/// テンソル用データ型
pub mod dtype;
/// Parallel processing utilities
/// 並列処理ユーティリティ
pub mod parallel;
/// Data loading and processing utilities
/// データ読み込みと処理のユーティリティ
pub mod data;
/// GPU acceleration support (CUDA, Metal, OpenCL)
/// GPU加速サポート（CUDA、Metal、OpenCL）
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;
/// Distributed training support for multi-GPU and multi-machine training
/// マルチGPUおよびマルチマシン学習用分散学習サポート
#[cfg(not(target_arch = "wasm32"))]
pub mod distributed;
/// Memory management and pooling utilities
/// メモリ管理とプーリングユーティリティ
#[cfg(not(target_arch = "wasm32"))]
pub mod memory;
/// SIMD vectorized operations for performance optimization
/// パフォーマンス最適化のためのSIMDベクトル化操作
#[cfg(not(target_arch = "wasm32"))]
pub mod simd;
/// Utility functions
/// ユーティリティ関数
pub mod utils;
/// Pre-built models and architectures
/// 事前構築モデルとアーキテクチャ
pub mod models;
/// Training loop abstractions and utilities
/// 学習ループの抽象化とユーティリティ
pub mod training;


/// WebAssembly support and bindings
/// WebAssemblyサポートとバインディング
#[cfg(feature = "wasm")]
pub mod wasm;

/// Simple WebAssembly support for basic operations
/// 基本操作のためのシンプルなWebAssemblyサポート
// Removed redundant wasm_simple module - functionality integrated into wasm module

/// Re-exports of commonly used items
pub mod prelude {
    pub use crate::tensor::Tensor;
    pub use crate::nn::{Module, Linear, Conv2d, MaxPool2d, AvgPool2d, BatchNorm1d, BatchNorm2d, Dropout, AlphaDropout, dropout, RNNCell, RNN, LSTMCell, LSTM, GRUCell, GRU};
    pub use crate::autograd::Variable;
    pub use crate::nn::activation::{relu, sigmoid, tanh, leaky_relu, softmax, gelu, swish, elu, selu, mish, hardswish};
    pub use crate::nn::loss::{mse_loss, binary_cross_entropy, cross_entropy, nll_loss, huber_loss};
    pub use crate::optim::{Optimizer, SGD, Adam, RMSprop, AdaGrad};
    pub use crate::data::{Dataset, TensorDataset, DataLoader};
    pub use crate::models::{Model, ModelMode, ModelBuilder, CNN, CNNBuilder, ResNet, ResNetBuilder};
    pub use crate::models::{RNNModel, RNNModelBuilder, LSTMModel, LSTMModelBuilder};
    pub use crate::models::{TransformerModel, TransformerModelBuilder, BERT, BERTBuilder};
    pub use crate::models::{Trainer, TrainingConfig, TrainingResult, InferenceEngine, Metrics};
    pub use crate::models::{ModelSaver, ModelLoader, SerializationFormat};
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

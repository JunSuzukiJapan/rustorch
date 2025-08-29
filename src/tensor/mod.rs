//! # Tensor Operations and Data Structures
//! テンソル操作とデータ構造
//!
//! This module provides the core tensor functionality for RusTorch, including
//! basic tensor operations, advanced parallel processing, GPU acceleration,
//! and memory optimization features.
//!
//! ## Core Components
//!
//! - `core`: The main tensor data structure with n-dimensional array support
//! - `operations`: Mathematical operations and arithmetic for tensors
//! - `parallel_traits`: Unified parallel tensor operations system
//! - `gpu_parallel`: GPU-accelerated tensor operations with device management
//! - `memory_optimized`: Advanced memory management strategies
//! - `zero_copy`: Zero-copy tensor views and shared ownership
//! - `simd_aligned`: SIMD-aligned tensor operations for vectorization
//!
//! ## Key Features
//!
//! ### High-Performance Computing
//! - **Parallel Processing**: Automatic parallelization for large tensor operations
//! - **SIMD Acceleration**: AVX2/SSE4.1 vectorized operations for f32 tensors
//! - **GPU Integration**: CUDA/Metal/OpenCL support with intelligent fallback
//! - **Memory Optimization**: Pool allocation, zero-copy views, and cache-friendly operations
//!
//! ### Mathematical Operations
//! - **Element-wise Operations**: Addition, multiplication, trigonometric functions
//! - **Linear Algebra**: Matrix multiplication, decompositions, eigenvalues
//! - **Broadcasting**: NumPy-style broadcasting for operations on different shapes
//! - **Reduction Operations**: Sum, mean, variance, and statistical functions
//!
//! ### Memory Management
//! - **Zero-Copy Views**: Efficient tensor slicing without data duplication
//! - **Memory Pooling**: Reduced allocation overhead for frequent operations
//! - **SIMD Alignment**: 32-byte aligned allocation for optimal vectorization
//! - **Shared Ownership**: Thread-safe reference counting for tensor sharing
//!
//! ## Usage Examples
//!
//! ### Basic Tensor Operations
//!
//! ```rust
//! use rustorch::tensor::Tensor;
//!
//! // Create tensors
//! let a = Tensor::<f32>::ones(&[3, 3]);
//! let b = Tensor::<f32>::zeros(&[3, 3]);
//!
//! // Basic arithmetic
//! let c = &a + &b;
//! let d = a.matmul(&b);
//!
//! // Mathematical functions
//! let e = a.sin();
//! let f = a.exp();
//! ```
//!
//! ### Parallel Operations
//!
//! RusTorch provides efficient parallel tensor operations for high-performance computing.
//!
//! ### GPU Acceleration
//!
//! RusTorch supports GPU acceleration with automatic fallback to CPU when GPU is unavailable.
//!
//! ### Memory Optimization
//!
//! Advanced memory management strategies for optimal performance and memory usage.

use num_traits::Float;

/// Core tensor data structure
/// コアテンソルデータ構造  
pub mod core;
/// Mathematical operations for tensors (legacy - replaced by ops)
/// テンソルの数学演算（レガシー - opsに置換）
// pub mod operations; // Disabled - replaced by ops/ modules
#[cfg(not(target_arch = "wasm32"))]
mod pool_integration;

/// Complex number support for tensors
/// テンソルの複素数サポート
pub mod complex;
/// Numeric safety and overflow protection
/// 数値安全性とオーバーフロー保護
pub mod numeric_safety;
/// Organized tensor operations by category (new modular system)
/// カテゴリ別に整理されたテンソル操作（新しいモジュールシステム）
pub mod ops;
pub mod parallel_errors;
pub mod parallel_impl;
/// Parallel tensor operations module
/// 並列テンソル演算モジュール
pub mod parallel_ops;
pub mod parallel_traits;
/// Type-safe tensor operations with compile-time verification
/// コンパイル時検証付きの型安全テンソル操作
pub mod type_safe;

#[cfg(not(target_arch = "wasm32"))]
pub mod advanced_memory;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_parallel;
#[cfg(not(target_arch = "wasm32"))]
pub mod simd_aligned;
#[cfg(not(target_arch = "wasm32"))]
pub mod simd_avx512;
/// Parallel tensor operations for batch processing and SIMD acceleration
/// バッチ処理とSIMD加速のための並列テンソル操作
#[cfg(not(target_arch = "wasm32"))]
pub mod simd_integration;
// Enable modules step by step
mod broadcasting;

// Re-export important types and functions
pub use crate::error::RusTorchResult as ParallelResult;
pub use core::Tensor;

// Convenience functions
/// Create a tensor with linearly spaced values
/// 線形に間隔を空けた値でテンソルを作成
pub fn linspace<T: Float + 'static>(start: T, end: T, steps: usize) -> Tensor<T> {
    if steps == 0 {
        return Tensor::from_vec(vec![], vec![0]);
    }
    if steps == 1 {
        return Tensor::from_vec(vec![start], vec![1]);
    }

    let step = (end - start) / T::from(steps - 1).unwrap();
    let mut values = Vec::with_capacity(steps);
    let mut current = start;

    for _ in 0..steps {
        values.push(current);
        current = current + step;
    }
    Tensor::from_vec(values, vec![steps])
}

/// Add unsqueeze function for convenience
/// 便利関数としてunsqueezeを追加
pub fn unsqueeze<T: Float + 'static>(tensor: &Tensor<T>, dim: usize) -> Result<Tensor<T>, String> {
    tensor.unsqueeze(dim).map_err(|e| e.to_string())
}

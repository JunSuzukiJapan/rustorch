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
//! // Mathematical functions (using ndarray methods)
//! let e = a.data.mapv(|x| x.sin());
//! let f = a.data.mapv(|x| x.exp());
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
/// Device management for tensor operations
/// テンソル操作用デバイス管理
pub mod device;
/// Mathematical operations for tensors (legacy - replaced by ops)
/// テンソルの数学演算（レガシー - opsに置換）
// pub mod operations; // Disabled - replaced by ops/ modules
#[cfg(not(target_arch = "wasm32"))]
mod pool_integration;
#[cfg(test)]
mod test_error_handling;

/// Complex number support for tensors
/// テンソルの複素数サポート
pub mod complex;
/// Modular complex number implementation
/// モジュール化された複素数実装
pub mod complex_impl;
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
pub mod gpu_parallel;
/// Modern memory management system
/// 現代的なメモリ管理システム
#[cfg(not(target_arch = "wasm32"))]
pub mod memory;
#[cfg(not(target_arch = "wasm32"))]
pub mod simd_avx512;

/// Organized tensor operations by category (trait-based system)
/// カテゴリ別に整理されたテンソル操作（トレイトベースシステム）
pub mod operations;
/// Parallel tensor operations for batch processing and SIMD acceleration
/// バッチ処理とSIMD加速のための並列テンソル操作
#[cfg(not(target_arch = "wasm32"))]
pub mod simd_integration;

/// Shared operations between regular and WASM tensors
/// 通常テンソルとWASMテンソル間の共通操作
pub mod shared_ops;
/// Phase 8: Advanced tensor utilities for conditional, indexing, and statistical operations
/// フェーズ8: 条件、インデックス、統計操作のための高度なテンソルユーティリティ
pub mod utilities;
// Enable modules step by step
// mod broadcasting; // Temporarily disabled to avoid conflicts with shape_operations

// Re-export important types and functions
pub use crate::error::RusTorchResult as ParallelResult;
pub use core::Tensor;
pub use device::Device;

// Re-export commonly used traits for better ergonomics
#[cfg(not(target_arch = "wasm32"))]
pub use memory::optimization::{MemoryOptimization, TensorMemoryInfo};
pub use operations::zero_copy::{TensorIterOps, ZeroCopyOps};

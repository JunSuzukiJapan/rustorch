//! # Parallel Tensor Operations Traits
//! 並列テンソル操作のトレイト定義
//! 
//! This module provides a unified trait-based system for parallel tensor operations,
//! enabling high-performance computation across multiple threads with automatic
//! SIMD acceleration and intelligent scheduling.
//! 
//! ## Overview
//! 
//! The parallel traits system is designed around five core traits:
//! 
//! - [`ParallelOp`]: Base trait for all parallel operations
//! - [`BatchParallelOp`]: Batch processing operations
//! - [`MatrixParallelOp`]: Matrix-specific parallel operations
//! - [`ReductionParallelOp`]: Parallel reduction operations
//! - [`SimdParallelOp`]: SIMD-optimized parallel operations (f32 specialized)
//! 
//! ## Key Features
//! 
//! - **Automatic Parallelization**: Operations automatically parallelize for large tensors
//! - **SIMD Integration**: Specialized f32 operations with AVX2/SSE4.1 acceleration
//! - **Configurable Strategies**: Choose between automatic, forced parallel, or sequential execution
//! - **Memory Safety**: Arc-based shared ownership ensures thread safety
//! - **Error Handling**: Structured error types with detailed context information
//! 
//! ## Usage Examples
//! 
//! ### Basic Parallel Operations
//! 
//! ```rust
//! use rustorch::tensor::{Tensor, parallel_traits::*};
//! 
//! let tensor1 = Tensor::<f32>::ones(&[1000, 1000]);
//! let tensor2 = Tensor::<f32>::ones(&[1000, 1000]);
//! 
//! // Automatic parallel execution for large tensors
//! let result = tensor1.batch_elementwise_op(&tensor2, |a, b| a + b)?;
//! 
//! // Parallel matrix multiplication
//! let matmul_result = tensor1.batch_matmul(&tensor2)?;
//! 
//! // Parallel reduction operations
//! let sum = tensor1.parallel_sum(0)?;
//! let mean = tensor1.parallel_mean(0)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//! 
//! ### SIMD-Optimized Operations
//! 
//! ```rust
//! use rustorch::tensor::{Tensor, parallel_traits::*};
//! 
//! let tensor1 = Tensor::<f32>::ones(&[10000]);
//! let tensor2 = Tensor::<f32>::ones(&[10000]);
//! 
//! // SIMD-accelerated parallel addition
//! let result = tensor1.simd_parallel_add(&tensor2)?;
//! 
//! // SIMD-accelerated parallel matrix multiplication
//! let matrix1 = Tensor::<f32>::ones(&[100, 100]);
//! let matrix2 = Tensor::<f32>::ones(&[100, 100]);
//! let matmul_result = matrix1.simd_parallel_matmul(&matrix2)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//! 
//! ### Configurable Parallel Execution
//! 
//! ```rust
//! use rustorch::tensor::parallel_traits::*;
//! 
//! // Configure parallel execution strategy
//! let config = ParallelConfig {
//!     strategy: ParallelStrategy::ForceParallel,
//!     chunk_size: 2048,
//!     num_threads: Some(4),
//! };
//! 
//! let context = ParallelContext::new(config);
//! 
//! // Use context to determine execution strategy
//! let should_parallelize = context.should_parallelize(1000);
//! println!("Should parallelize: {}", should_parallelize);
//! ```
//! 
//! ## Performance Characteristics
//! 
//! - **Threshold-based**: Operations automatically parallelize when tensor size exceeds 1000 elements
//! - **SIMD Acceleration**: f32 operations benefit from AVX2/SSE4.1 vectorization
//! - **Memory Efficient**: Zero-copy operations where possible, minimal memory allocation
//! - **Thread Safety**: All operations are thread-safe with proper synchronization
//! 
//! ## Error Handling
//! 
//! All parallel operations return [`ParallelResult<T>`](crate::tensor::parallel_errors::ParallelResult)
//! with detailed error information for debugging and error recovery.
//! 
//! See [`parallel_errors`](crate::tensor::parallel_errors) for complete error type documentation.

use super::Tensor;
use super::parallel_errors::{ParallelError, ParallelResult};
use num_traits::Float;
use rayon::prelude::*;
use std::sync::Arc;

/// 並列操作の基本トレイト
/// Base trait for parallel operations
pub trait ParallelOp<T: Float + Send + Sync + Clone + 'static> {
    /// 並列実行の最小サイズ閾値
    /// Minimum size threshold for parallel execution
    const MIN_PARALLEL_SIZE: usize = 1000;
    
    /// 並列実行が効率的かどうかを判定
    /// Determine if parallel execution is efficient
    fn should_parallelize(&self, size: usize) -> bool {
        size >= Self::MIN_PARALLEL_SIZE
    }
}

/// バッチ処理操作のトレイト
/// Trait for batch processing operations
pub trait BatchParallelOp<T: Float + Send + Sync + Clone + 'static>: ParallelOp<T> {
    /// 並列バッチ要素ごと演算
    /// Parallel batch element-wise operations
    fn batch_elementwise_op<F>(&self, other: &Tensor<T>, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(T, T) -> T + Send + Sync;
    
    /// 並列バッチスカラー演算
    /// Parallel batch scalar operations
    fn batch_scalar_op<F>(&self, scalar: T, op: F) -> Tensor<T>
    where
        F: Fn(T, T) -> T + Send + Sync;
    
    /// 並列バッチ正規化
    /// Parallel batch normalization
    fn batch_normalize(&self, epsilon: T) -> Tensor<T>;
}

/// 行列演算の並列化トレイト
/// Trait for parallel matrix operations
pub trait MatrixParallelOp<T: Float + Send + Sync + Clone + 'static>: ParallelOp<T> {
    /// 並列バッチ行列乗算
    /// Parallel batch matrix multiplication
    fn batch_matmul(&self, other: &Tensor<T>) -> ParallelResult<Tensor<T>>;
    
    /// 並列畳み込み演算
    /// Parallel convolution operation
    fn batch_conv2d(&self, kernel: &Tensor<T>, stride: usize, padding: usize) -> ParallelResult<Tensor<T>>;
}

/// リダクション操作の並列化トレイト
/// Trait for parallel reduction operations
pub trait ReductionParallelOp<T: Float + Send + Sync + Clone + 'static>: ParallelOp<T> {
    /// 並列リダクション演算
    /// Parallel reduction operation
    fn parallel_reduce<F, R>(&self, dim: usize, init: R, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(R, T) -> R + Send + Sync + Clone,
        R: Send + Sync + Clone + Into<T>;
    
    /// 並列合計
    /// Parallel sum
    fn parallel_sum(&self, dim: usize) -> ParallelResult<Tensor<T>> {
        self.parallel_reduce(dim, T::zero(), |acc, x| acc + x)
    }
    
    /// 並列平均
    /// Parallel mean
    fn parallel_mean(&self, dim: usize) -> ParallelResult<Tensor<T>>;
}

/// SIMD統合並列操作のトレイト（f32特化）
/// Trait for SIMD-integrated parallel operations (f32 specialized)
pub trait SimdParallelOp: ParallelOp<f32> {
    /// SIMD最適化並列加算
    /// SIMD-optimized parallel addition
    fn simd_parallel_add(&self, other: &Tensor<f32>) -> ParallelResult<Tensor<f32>>;
    
    /// SIMD最適化並列行列乗算
    /// SIMD-optimized parallel matrix multiplication
    fn simd_parallel_matmul(&self, other: &Tensor<f32>) -> ParallelResult<Tensor<f32>>;
    
    /// SIMD最適化並列スカラー乗算
    /// SIMD-optimized parallel scalar multiplication
    fn simd_parallel_scalar_mul(&self, scalar: f32) -> Tensor<f32>;
}

/// 並列操作の実行戦略
/// Execution strategy for parallel operations
#[derive(Debug, Clone, Copy)]
pub enum ParallelStrategy {
    /// 自動選択（サイズに基づく）
    /// Automatic selection (size-based)
    Auto,
    /// 強制的に並列実行
    /// Force parallel execution
    ForceParallel,
    /// 強制的に逐次実行
    /// Force sequential execution
    ForceSequential,
}

/// 並列操作の設定
/// Configuration for parallel operations
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// 実行戦略
    /// Execution strategy
    pub strategy: ParallelStrategy,
    /// チャンクサイズ（SIMD最適化用）
    /// Chunk size (for SIMD optimization)
    pub chunk_size: usize,
    /// 並列度（スレッド数）
    /// Parallelism level (number of threads)
    pub num_threads: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelStrategy::Auto,
            chunk_size: 1024,
            num_threads: None,
        }
    }
}

/// 並列操作のコンテキスト
/// Context for parallel operations
pub struct ParallelContext {
    /// 設定
    /// Configuration
    pub config: ParallelConfig,
}

impl ParallelContext {
    /// 新しいコンテキストを作成
    /// Create a new context
    pub fn new(config: ParallelConfig) -> Self {
        Self { config }
    }
    
    /// デフォルトコンテキストを作成
    /// Create a default context
    pub fn default() -> Self {
        Self::new(ParallelConfig::default())
    }
    
    /// 並列実行すべきかを判定
    /// Determine if should execute in parallel
    pub fn should_parallelize(&self, size: usize) -> bool {
        match self.config.strategy {
            ParallelStrategy::Auto => size >= 1000,
            ParallelStrategy::ForceParallel => true,
            ParallelStrategy::ForceSequential => false,
        }
    }
}

/// 並列操作のユーティリティ関数
/// Utility functions for parallel operations
pub mod parallel_utils {
    use super::*;
    
    /// 安全な並列インデックス計算
    /// Safe parallel index calculation
    pub fn safe_parallel_index(
        total_size: usize,
        chunk_size: usize,
        chunk_idx: usize,
    ) -> (usize, usize) {
        let start = chunk_idx * chunk_size;
        let end = std::cmp::min(start + chunk_size, total_size);
        (start, end)
    }
    
    /// 並列チャンク処理
    /// Parallel chunk processing
    pub fn parallel_chunks<T, F, R>(
        data: &[T],
        chunk_size: usize,
        mut process_chunk: F,
    ) -> Vec<R>
    where
        T: Send + Sync,
        F: Fn(&[T]) -> R + Send + Sync,
        R: Send,
    {
        data.par_chunks(chunk_size)
            .map(|chunk| process_chunk(chunk))
            .collect()
    }
    
    /// バッチ次元での並列処理
    /// Parallel processing over batch dimension
    pub fn parallel_batch_process<T, F, R>(
        batch_size: usize,
        process_batch: F,
    ) -> Vec<R>
    where
        F: Fn(usize) -> R + Send + Sync,
        R: Send,
    {
        (0..batch_size)
            .into_par_iter()
            .map(process_batch)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::default();
        assert_eq!(config.chunk_size, 1024);
        assert!(matches!(config.strategy, ParallelStrategy::Auto));
    }
    
    #[test]
    fn test_parallel_context() {
        let ctx = ParallelContext::default();
        assert!(ctx.should_parallelize(2000));
        assert!(!ctx.should_parallelize(500));
    }
    
    #[test]
    fn test_parallel_utils() {
        let (start, end) = parallel_utils::safe_parallel_index(1000, 100, 5);
        assert_eq!(start, 500);
        assert_eq!(end, 600);
        
        let (start, end) = parallel_utils::safe_parallel_index(1000, 100, 9);
        assert_eq!(start, 900);
        assert_eq!(end, 1000);
    }
}

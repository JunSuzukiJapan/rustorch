//! Linear Algebra Module with High-Performance Optimizations
//! 高性能最適化付き線形代数モジュール

pub mod optimized_blas;

// Re-export high-performance functions
pub use optimized_blas::{
    benchmark_matmul_implementations, multithreaded_matmul, optimized_matmul,
};

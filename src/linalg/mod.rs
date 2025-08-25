//! Linear Algebra Module with High-Performance Optimizations
//! 高性能最適化付き線形代数モジュール

pub mod optimized_blas;

// Re-export high-performance functions
pub use optimized_blas::{optimized_matmul, multithreaded_matmul, benchmark_matmul_implementations};
//! Memory Management Module for RusTorch Tensors
//! RusTorchテンソルのメモリ管理モジュール
//!
//! This module provides comprehensive memory management capabilities including:
//! - SIMD-aligned allocation for high-performance computing
//! - Advanced memory pooling strategies
//! - Zero-copy operations and memory sharing detection
//! - Memory optimization and analysis tools

pub mod aligned;
pub mod pool;
pub mod optimization;

// Re-export commonly used types and functions
pub use aligned::{SimdAllocator, SIMD_ALIGNMENT};
pub use optimization::TensorMemoryInfo;
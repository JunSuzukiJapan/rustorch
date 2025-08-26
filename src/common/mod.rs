//! Common utilities and shared functionality for RusTorch
//! RusTorchの共通ユーティリティと共有機能

pub mod error_handling;

pub use error_handling::{
    DataError, DistributedError, GpuError, MemoryError, NeuralNetworkError, OptimizationError,
    RusTorchError, RusTorchResult, TensorError,
};

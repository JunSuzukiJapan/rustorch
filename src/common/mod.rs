//! Common utilities and shared functionality for RusTorch
//! RusTorchの共通ユーティリティと共有機能

pub mod error_handling;

pub use error_handling::{
    RusTorchError, RusTorchResult,
    TensorError, GpuError, DistributedError, 
    NeuralNetworkError, OptimizationError, 
    DataError, MemoryError
};

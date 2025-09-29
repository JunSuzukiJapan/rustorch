//! hybrid_f32 - f32統一ハイブリッドシステム
//! hybrid_f32 - f32 unified hybrid system

// ============================================
// Core Modules
// ============================================

/// エラー処理
/// Error handling
pub mod error;

/// テンサー実装
/// Tensor implementations
pub mod tensor;

/// 自動微分
/// Automatic differentiation
pub mod autograd;

/// ニューラルネットワーク
/// Neural networks
pub mod nn;

// ============================================
// System Modules
// ============================================

/// GPU処理
/// GPU processing
pub mod gpu;

/// メモリ管理
/// Memory management
pub mod memory;

/// 統一API
/// Unified API
pub mod unified;

/// ベンチマーク
/// Benchmarks
pub mod benchmarks;

/// 実験結果
/// Experiment results
pub mod experiment;

// ============================================
// Essential Re-exports
// ============================================

// Error types
pub use error::{F32Error, F32Result};

// Primary tensor type
pub use tensor::F32Tensor;

// Experiment system
pub use experiment::{ExperimentResults, ExperimentResultsBuilder};

// Macro access
pub use crate::hybrid_f32_experimental;
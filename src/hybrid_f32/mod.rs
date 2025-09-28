//! f32統一ハイブリッドシステム実験実装
//! Experimental f32 Unified Hybrid System Implementation
//!
//! このモジュールは変換コスト削減を目的としたf32精度統一ハイブリッドシステムの
//! 実験的実装です。プロダクション環境での検証後、メインシステムに統合予定。
//!
//! This module contains experimental implementation of f32 unified hybrid system
//! aimed at reducing conversion costs. Planned for integration into main system
//! after production validation.

pub mod benchmarks;
pub mod gpu;
pub mod tensor;
pub mod unified;

// 実験フラグ - この機能は実験的です
// Experimental flag - this feature is experimental
#[cfg(feature = "hybrid-f32")]
pub use tensor::F32Tensor;

#[cfg(feature = "hybrid-f32")]
pub use unified::F32HybridExecutor;

#[cfg(feature = "hybrid-f32")]
pub use gpu::{F32CoreMLExecutor, F32MetalExecutor};

/// 実験的機能の警告マクロ
/// Warning macro for experimental features
#[macro_export]
macro_rules! hybrid_f32_experimental {
    () => {
        eprintln!(
            "⚠️  EXPERIMENTAL: f32統一ハイブリッドシステムは実験的機能です\n\
             ⚠️  WARNING: f32 unified hybrid system is experimental"
        );
    };
}

/// 実験結果を記録するための構造体
/// Structure for recording experimental results
#[derive(Debug, Clone)]
pub struct ExperimentResults {
    pub conversion_cost_reduction: f64, // パーセンテージ
    pub memory_efficiency_gain: f64,    // パーセンテージ
    pub neural_engine_performance: f64, // Float16比の性能
    pub total_execution_time: std::time::Duration,
    pub baseline_execution_time: std::time::Duration,
}

impl ExperimentResults {
    pub fn new() -> Self {
        Self {
            conversion_cost_reduction: 0.0,
            memory_efficiency_gain: 0.0,
            neural_engine_performance: 0.0,
            total_execution_time: std::time::Duration::from_secs(0),
            baseline_execution_time: std::time::Duration::from_secs(0),
        }
    }

    /// パフォーマンス改善率を計算
    /// Calculate performance improvement percentage
    pub fn performance_improvement(&self) -> f64 {
        if self.baseline_execution_time.as_nanos() == 0 {
            return 0.0;
        }

        let baseline_ns = self.baseline_execution_time.as_nanos() as f64;
        let current_ns = self.total_execution_time.as_nanos() as f64;

        ((baseline_ns - current_ns) / baseline_ns) * 100.0
    }
}

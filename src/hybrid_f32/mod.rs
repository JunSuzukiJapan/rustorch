//! hybrid_f32 - f32統一ハイブリッドシステム
//! hybrid_f32 - f32 unified hybrid system

pub mod error;
pub mod tensor;
pub mod autograd;
pub mod nn;
pub mod gpu;
pub mod unified;
pub mod benchmarks;

// Re-exports for convenience
pub use error::{F32Error, F32Result};
pub use tensor::core::F32Tensor;
pub use crate::hybrid_f32_experimental;

/// 実験結果構造体
/// Experiment results structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExperimentResults {
    pub performance_gain: f64,
    pub memory_efficiency: f64,
    pub accuracy_maintained: bool,
    pub device_compatibility: Vec<String>,
    pub total_execution_time: std::time::Duration,
    pub conversion_cost_reduction: f64,
}

impl ExperimentResults {
    pub fn new() -> Self {
        hybrid_f32_experimental!();

        Self {
            performance_gain: 0.0,
            memory_efficiency: 0.0,
            accuracy_maintained: true,
            device_compatibility: vec!["CPU".to_string()],
            total_execution_time: std::time::Duration::from_secs(0),
            conversion_cost_reduction: 0.0,
        }
    }
}

impl Default for ExperimentResults {
    fn default() -> Self {
        Self::new()
    }
}
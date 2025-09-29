//! 実験結果管理モジュール
//! Experiment results management module

use std::time::Duration;
use serde::{Serialize, Deserialize};

/// 実験結果構造体
/// Experiment results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResults {
    /// パフォーマンス向上率
    /// Performance gain ratio
    pub performance_gain: f64,

    /// メモリ効率
    /// Memory efficiency
    pub memory_efficiency: f64,

    /// 精度維持状況
    /// Accuracy maintenance status
    pub accuracy_maintained: bool,

    /// デバイス互換性
    /// Device compatibility
    pub device_compatibility: Vec<String>,

    /// 総実行時間
    /// Total execution time
    pub total_execution_time: Duration,

    /// 変換コスト削減率
    /// Conversion cost reduction ratio
    pub conversion_cost_reduction: f64,
}

impl ExperimentResults {
    /// 新しい実験結果を作成
    /// Create new experiment results
    pub fn new() -> Self {
        crate::hybrid_f32_experimental!();

        Self {
            performance_gain: 0.0,
            memory_efficiency: 0.0,
            accuracy_maintained: true,
            device_compatibility: vec!["CPU".to_string()],
            total_execution_time: Duration::from_secs(0),
            conversion_cost_reduction: 0.0,
        }
    }

    /// パフォーマンス情報を設定
    /// Set performance information
    pub fn with_performance(mut self, gain: f64, memory_eff: f64) -> Self {
        self.performance_gain = gain;
        self.memory_efficiency = memory_eff;
        self
    }

    /// デバイス互換性を設定
    /// Set device compatibility
    pub fn with_devices(mut self, devices: Vec<String>) -> Self {
        self.device_compatibility = devices;
        self
    }

    /// 実行時間を設定
    /// Set execution time
    pub fn with_execution_time(mut self, time: Duration) -> Self {
        self.total_execution_time = time;
        self
    }

    /// レポートを生成
    /// Generate report
    pub fn generate_report(&self) -> String {
        format!(
            "Hybrid F32 Experiment Results:\n\
            - Performance Gain: {:.2}%\n\
            - Memory Efficiency: {:.2}%\n\
            - Accuracy Maintained: {}\n\
            - Compatible Devices: {}\n\
            - Total Execution Time: {:.2}s\n\
            - Conversion Cost Reduction: {:.2}%\n",
            self.performance_gain * 100.0,
            self.memory_efficiency * 100.0,
            if self.accuracy_maintained { "Yes" } else { "No" },
            self.device_compatibility.join(", "),
            self.total_execution_time.as_secs_f64(),
            self.conversion_cost_reduction * 100.0
        )
    }
}

impl Default for ExperimentResults {
    fn default() -> Self {
        Self::new()
    }
}

/// 実験結果ビルダー
/// Experiment results builder
pub struct ExperimentResultsBuilder {
    results: ExperimentResults,
}

impl ExperimentResultsBuilder {
    /// 新しいビルダーを作成
    /// Create new builder
    pub fn new() -> Self {
        Self {
            results: ExperimentResults::new(),
        }
    }

    /// パフォーマンス向上率を設定
    /// Set performance gain
    pub fn performance_gain(mut self, gain: f64) -> Self {
        self.results.performance_gain = gain;
        self
    }

    /// メモリ効率を設定
    /// Set memory efficiency
    pub fn memory_efficiency(mut self, efficiency: f64) -> Self {
        self.results.memory_efficiency = efficiency;
        self
    }

    /// 精度維持状況を設定
    /// Set accuracy maintenance
    pub fn accuracy_maintained(mut self, maintained: bool) -> Self {
        self.results.accuracy_maintained = maintained;
        self
    }

    /// デバイス互換性を追加
    /// Add device compatibility
    pub fn add_device(mut self, device: String) -> Self {
        self.results.device_compatibility.push(device);
        self
    }

    /// 実行時間を設定
    /// Set execution time
    pub fn execution_time(mut self, time: Duration) -> Self {
        self.results.total_execution_time = time;
        self
    }

    /// 変換コスト削減率を設定
    /// Set conversion cost reduction
    pub fn conversion_cost_reduction(mut self, reduction: f64) -> Self {
        self.results.conversion_cost_reduction = reduction;
        self
    }

    /// 結果をビルド
    /// Build results
    pub fn build(self) -> ExperimentResults {
        self.results
    }
}

impl Default for ExperimentResultsBuilder {
    fn default() -> Self {
        Self::new()
    }
}
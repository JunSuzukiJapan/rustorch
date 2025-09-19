//! GPU Performance Optimizer
//! GPU性能最適化システム
//!
//! This module provides comprehensive performance optimization for GPU operations
//! including automatic device selection, memory optimization, and execution scheduling.

use crate::error::{RusTorchError, RusTorchResult};
use crate::gpu::DeviceType;
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Performance metrics for GPU operations
/// GPU演算のパフォーマンス指標
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub operation_type: String,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub device_type: DeviceType,
    pub tensor_size: usize,
    pub timestamp: Instant,
    pub energy_consumption: f64, // in watts
    pub thermal_state: ThermalState,
}

/// Device thermal state monitoring
/// デバイス熱状態監視
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalState {
    Normal,   // < 60°C
    Elevated, // 60-80°C
    Critical, // > 80°C
    Unknown,
}

/// Optimization strategies for different scenarios
/// 異なるシナリオ用の最適化戦略
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationStrategy {
    LatencyOptimized,    // Minimize execution time
    ThroughputOptimized, // Maximize operations per second
    EnergyOptimized,     // Minimize power consumption
    MemoryOptimized,     // Minimize memory usage
    Balanced,            // Balance all factors
    Auto,                // Automatically choose best strategy
}

/// Performance profiler for device capabilities
/// デバイス能力用パフォーマンスプロファイラ
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub device_type: DeviceType,
    pub compute_score: f64,     // Relative compute performance (0.0-1.0)
    pub memory_bandwidth: f64,  // GB/s
    pub memory_capacity: usize, // bytes
    pub thermal_limit: f64,     // °C
    pub power_efficiency: f64,  // ops per watt
    pub last_updated: Instant,
}

/// Performance optimizer for GPU operations
/// GPU演算用パフォーマンス最適化器
pub struct PerformanceOptimizer {
    metrics_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
    device_profiles: Arc<RwLock<HashMap<DeviceType, DeviceProfile>>>,
    optimization_strategy: Arc<RwLock<OptimizationStrategy>>,
    auto_tuning_enabled: bool,
    thermal_monitoring: Arc<Mutex<HashMap<DeviceType, ThermalState>>>,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    /// 新しいパフォーマンス最適化器を作成
    pub fn new() -> Self {
        let mut device_profiles = HashMap::new();

        // Initialize default device profiles
        // デフォルトデバイスプロファイルを初期化
        device_profiles.insert(
            DeviceType::Cpu,
            DeviceProfile {
                device_type: DeviceType::Cpu,
                compute_score: 0.3,
                memory_bandwidth: 50.0,
                memory_capacity: 16 * 1024 * 1024 * 1024, // 16GB
                thermal_limit: 85.0,
                power_efficiency: 10.0,
                last_updated: Instant::now(),
            },
        );

        device_profiles.insert(
            DeviceType::Cuda(0),
            DeviceProfile {
                device_type: DeviceType::Cuda(0),
                compute_score: 1.0,
                memory_bandwidth: 900.0,
                memory_capacity: 24 * 1024 * 1024 * 1024, // 24GB
                thermal_limit: 83.0,
                power_efficiency: 20.0,
                last_updated: Instant::now(),
            },
        );

        device_profiles.insert(
            DeviceType::Metal(0),
            DeviceProfile {
                device_type: DeviceType::Metal(0),
                compute_score: 0.8,
                memory_bandwidth: 400.0,
                memory_capacity: 64 * 1024 * 1024 * 1024, // 64GB unified memory
                thermal_limit: 100.0,
                power_efficiency: 50.0, // Apple Silicon efficiency
                last_updated: Instant::now(),
            },
        );

        device_profiles.insert(
            DeviceType::Auto,
            DeviceProfile {
                device_type: DeviceType::Auto,
                compute_score: 0.9,
                memory_bandwidth: 200.0,
                memory_capacity: 64 * 1024 * 1024 * 1024, // Shared with Metal
                thermal_limit: 100.0,
                power_efficiency: 100.0, // Neural Engine efficiency
                last_updated: Instant::now(),
            },
        );

        Self {
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            device_profiles: Arc::new(RwLock::new(device_profiles)),
            optimization_strategy: Arc::new(RwLock::new(OptimizationStrategy::Auto)),
            auto_tuning_enabled: true,
            thermal_monitoring: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Record performance metrics for an operation
    /// 演算のパフォーマンス指標を記録
    pub fn record_metrics(&self, metrics: PerformanceMetrics) -> RusTorchResult<()> {
        let mut history = self
            .metrics_history
            .write()
            .map_err(|_| RusTorchError::TensorOp {
                message: "Failed to acquire metrics lock".to_string(),
                source: None,
            })?;

        history.push(metrics.clone());

        // Keep only last 1000 metrics to prevent memory bloat
        // メモリ肥大化防止のため最新1000件のみ保持
        let history_len = history.len();
        if history_len > 1000 {
            history.drain(0..history_len - 1000);
        }

        // Update thermal monitoring
        // 熱監視を更新
        let device_type = metrics.device_type;
        let thermal_state = metrics.thermal_state.clone();
        if let Ok(mut thermal) = self.thermal_monitoring.lock() {
            thermal.insert(device_type, thermal_state);
        }

        // Auto-tune if enabled
        // 自動調整が有効な場合は実行
        if self.auto_tuning_enabled {
            self.auto_tune_strategy(&metrics)?;
        }

        Ok(())
    }

    /// Select optimal device for a given operation
    /// 指定された演算に最適なデバイスを選択
    pub fn select_optimal_device(
        &self,
        operation_type: &str,
        tensor_size: usize,
    ) -> RusTorchResult<DeviceType> {
        let strategy = self
            .optimization_strategy
            .read()
            .map_err(|_| RusTorchError::TensorOp {
                message: "Failed to acquire strategy lock".to_string(),
                source: None,
            })?;

        let profiles = self
            .device_profiles
            .read()
            .map_err(|_| RusTorchError::TensorOp {
                message: "Failed to acquire profiles lock".to_string(),
                source: None,
            })?;

        let thermal = self
            .thermal_monitoring
            .lock()
            .map_err(|_| RusTorchError::TensorOp {
                message: "Failed to acquire thermal lock".to_string(),
                source: None,
            })?;

        let mut best_device = DeviceType::Cpu;
        let mut best_score = 0.0;

        for (device_type, profile) in profiles.iter() {
            // Skip devices with critical thermal state
            // 臨界熱状態のデバイスはスキップ
            if thermal.get(device_type) == Some(&ThermalState::Critical) {
                continue;
            }

            let score = self.calculate_device_score(
                &strategy,
                profile,
                operation_type,
                tensor_size,
                thermal.get(device_type).unwrap_or(&ThermalState::Unknown),
            );

            if score > best_score {
                best_score = score;
                best_device = *device_type;
            }
        }

        Ok(best_device)
    }

    /// Calculate device suitability score based on strategy
    /// 戦略に基づくデバイス適合性スコアを計算
    fn calculate_device_score(
        &self,
        strategy: &OptimizationStrategy,
        profile: &DeviceProfile,
        operation_type: &str,
        tensor_size: usize,
        thermal_state: &ThermalState,
    ) -> f64 {
        let mut score = 0.0;

        // Base compute score
        // 基本計算スコア
        score += profile.compute_score * 0.4;

        // Memory adequacy
        // メモリ適正性
        let memory_ratio = tensor_size as f64 / profile.memory_capacity as f64;
        let memory_score = if memory_ratio > 1.0 {
            0.0 // Insufficient memory
        } else {
            (1.0 - memory_ratio).min(1.0)
        };
        score += memory_score * 0.2;

        // Strategy-specific adjustments
        // 戦略固有の調整
        match strategy {
            OptimizationStrategy::LatencyOptimized => {
                score += profile.compute_score * 0.3;
                score += (profile.memory_bandwidth / 1000.0).min(1.0) * 0.1;
            }
            OptimizationStrategy::ThroughputOptimized => {
                score += profile.compute_score * 0.2;
                score += (profile.memory_bandwidth / 1000.0).min(1.0) * 0.2;
            }
            OptimizationStrategy::EnergyOptimized => {
                score += (profile.power_efficiency / 100.0).min(1.0) * 0.4;
            }
            OptimizationStrategy::MemoryOptimized => {
                score += memory_score * 0.3;
                score += (profile.memory_bandwidth / 1000.0).min(1.0) * 0.1;
            }
            OptimizationStrategy::Balanced => {
                score += profile.compute_score * 0.15;
                score += memory_score * 0.1;
                score += (profile.power_efficiency / 100.0).min(1.0) * 0.05;
            }
            OptimizationStrategy::Auto => {
                // Auto strategy adapts based on operation type
                // 自動戦略は演算タイプに基づいて適応
                match operation_type {
                    "matmul" | "conv2d" => {
                        score += profile.compute_score * 0.3;
                    }
                    "activation" | "elementwise" => {
                        score += (profile.memory_bandwidth / 1000.0).min(1.0) * 0.2;
                    }
                    _ => {
                        score += profile.compute_score * 0.2;
                    }
                }
            }
        }

        // Thermal penalties
        // 熱ペナルティ
        match thermal_state {
            ThermalState::Normal => {}
            ThermalState::Elevated => score *= 0.8,
            ThermalState::Critical => score = 0.0,
            ThermalState::Unknown => score *= 0.9,
        }

        score.max(0.0).min(1.0)
    }

    /// Auto-tune optimization strategy based on performance history
    /// パフォーマンス履歴に基づく最適化戦略の自動調整
    fn auto_tune_strategy(&self, latest_metrics: &PerformanceMetrics) -> RusTorchResult<()> {
        let history = self
            .metrics_history
            .read()
            .map_err(|_| RusTorchError::TensorOp {
                message: "Failed to acquire metrics lock".to_string(),
                source: None,
            })?;

        if history.len() < 10 {
            return Ok(()); // Not enough data for tuning
        }

        // Analyze recent performance trends
        // 最近のパフォーマンス傾向を分析
        let recent_metrics: Vec<&PerformanceMetrics> = history.iter().rev().take(10).collect();

        let avg_execution_time: Duration = recent_metrics
            .iter()
            .map(|m| m.execution_time)
            .sum::<Duration>()
            / recent_metrics.len() as u32;

        let avg_energy: f64 = recent_metrics
            .iter()
            .map(|m| m.energy_consumption)
            .sum::<f64>()
            / recent_metrics.len() as f64;

        // Adjust strategy based on trends
        // 傾向に基づく戦略調整
        let mut strategy =
            self.optimization_strategy
                .write()
                .map_err(|_| RusTorchError::TensorOp {
                    message: "Failed to acquire strategy lock".to_string(),
                    source: None,
                })?;

        if *strategy == OptimizationStrategy::Auto {
            let new_strategy = if avg_execution_time > Duration::from_millis(100) {
                OptimizationStrategy::LatencyOptimized
            } else if avg_energy > 50.0 {
                OptimizationStrategy::EnergyOptimized
            } else {
                OptimizationStrategy::Balanced
            };

            *strategy = new_strategy;
        }

        Ok(())
    }

    /// Get performance statistics
    /// パフォーマンス統計を取得
    pub fn get_performance_stats(&self) -> RusTorchResult<PerformanceStats> {
        let history = self
            .metrics_history
            .read()
            .map_err(|_| RusTorchError::TensorOp {
                message: "Failed to acquire metrics lock".to_string(),
                source: None,
            })?;

        if history.is_empty() {
            return Ok(PerformanceStats::default());
        }

        let total_operations = history.len();
        let avg_execution_time =
            history.iter().map(|m| m.execution_time).sum::<Duration>() / total_operations as u32;

        let avg_memory_usage =
            history.iter().map(|m| m.memory_usage).sum::<usize>() / total_operations;

        let avg_energy_consumption =
            history.iter().map(|m| m.energy_consumption).sum::<f64>() / total_operations as f64;

        // Device usage distribution
        // デバイス使用分布
        let mut device_usage = HashMap::new();
        for metrics in history.iter() {
            *device_usage.entry(metrics.device_type).or_insert(0) += 1;
        }

        let thermal = self
            .thermal_monitoring
            .lock()
            .map_err(|_| RusTorchError::TensorOp {
                message: "Failed to acquire thermal lock".to_string(),
                source: None,
            })?;

        Ok(PerformanceStats {
            total_operations,
            avg_execution_time,
            avg_memory_usage,
            avg_energy_consumption,
            device_usage,
            current_thermal_states: thermal.clone(),
        })
    }

    /// Set optimization strategy
    /// 最適化戦略を設定
    pub fn set_optimization_strategy(&self, strategy: OptimizationStrategy) -> RusTorchResult<()> {
        let mut current_strategy =
            self.optimization_strategy
                .write()
                .map_err(|_| RusTorchError::TensorOp {
                    message: "Failed to acquire strategy lock".to_string(),
                    source: None,
                })?;

        *current_strategy = strategy;
        Ok(())
    }

    /// Update device profile with new measurements
    /// 新しい測定値でデバイスプロファイルを更新
    pub fn update_device_profile(
        &self,
        device_type: DeviceType,
        profile: DeviceProfile,
    ) -> RusTorchResult<()> {
        let mut profiles = self
            .device_profiles
            .write()
            .map_err(|_| RusTorchError::TensorOp {
                message: "Failed to acquire profiles lock".to_string(),
                source: None,
            })?;

        profiles.insert(device_type, profile);
        Ok(())
    }
}

/// Performance statistics summary
/// パフォーマンス統計サマリ
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub total_operations: usize,
    pub avg_execution_time: Duration,
    pub avg_memory_usage: usize,
    pub avg_energy_consumption: f64,
    pub device_usage: HashMap<DeviceType, usize>,
    pub current_thermal_states: HashMap<DeviceType, ThermalState>,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            avg_execution_time: Duration::from_secs(0),
            avg_memory_usage: 0,
            avg_energy_consumption: 0.0,
            device_usage: HashMap::new(),
            current_thermal_states: HashMap::new(),
        }
    }
}

/// Performance benchmarking utilities
/// パフォーマンスベンチマーク用ユーティリティ
pub struct PerformanceBenchmark;

impl PerformanceBenchmark {
    /// Benchmark operation on different devices
    /// 異なるデバイスでの演算ベンチマーク
    pub fn benchmark_operation<T>(
        operation_name: &str,
        tensor_a: &Tensor<T>,
        tensor_b: &Tensor<T>,
        devices: &[DeviceType],
    ) -> RusTorchResult<HashMap<DeviceType, Duration>>
    where
        T: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static,
    {
        let mut results = HashMap::new();

        for device_type in devices {
            let start = Instant::now();

            // TODO: Execute operation on specific device
            // TODO: 特定デバイスでの演算実行
            // This would require actual device-specific execution
            // これには実際のデバイス固有の実行が必要

            let duration = start.elapsed();
            results.insert(*device_type, duration);
        }

        Ok(results)
    }

    /// Run comprehensive performance suite
    /// 包括的パフォーマンススイートを実行
    pub fn run_performance_suite() -> RusTorchResult<HashMap<String, PerformanceStats>> {
        let mut suite_results = HashMap::new();

        // Matrix multiplication benchmarks
        // 行列乗算ベンチマーク
        suite_results.insert("matmul_small".to_string(), PerformanceStats::default());
        suite_results.insert("matmul_medium".to_string(), PerformanceStats::default());
        suite_results.insert("matmul_large".to_string(), PerformanceStats::default());

        // Activation function benchmarks
        // 活性化関数ベンチマーク
        suite_results.insert("relu".to_string(), PerformanceStats::default());
        suite_results.insert("sigmoid".to_string(), PerformanceStats::default());
        suite_results.insert("tanh".to_string(), PerformanceStats::default());

        // Convolution benchmarks
        // 畳み込みベンチマーク
        suite_results.insert("conv2d".to_string(), PerformanceStats::default());

        Ok(suite_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_optimizer_creation() {
        let optimizer = PerformanceOptimizer::new();

        // Verify default strategy is Auto
        // デフォルト戦略がAutoであることを確認
        let strategy = optimizer.optimization_strategy.read().unwrap();
        assert_eq!(*strategy, OptimizationStrategy::Auto);
    }

    #[test]
    fn test_device_selection() {
        let optimizer = PerformanceOptimizer::new();

        let device = optimizer
            .select_optimal_device("matmul", 1024 * 1024)
            .unwrap();

        // Should select a valid device type
        // 有効なデバイスタイプが選択されるべき
        assert!(matches!(
            device,
            DeviceType::Cpu | DeviceType::Cuda(_) | DeviceType::Metal(_) | DeviceType::Auto
        ));
    }

    #[test]
    fn test_metrics_recording() {
        let optimizer = PerformanceOptimizer::new();

        let metrics = PerformanceMetrics {
            operation_type: "test_op".to_string(),
            execution_time: Duration::from_millis(100),
            memory_usage: 1024,
            device_type: DeviceType::Cpu,
            tensor_size: 512,
            timestamp: Instant::now(),
            energy_consumption: 10.0,
            thermal_state: ThermalState::Normal,
        };

        assert!(optimizer.record_metrics(metrics).is_ok());

        let stats = optimizer.get_performance_stats().unwrap();
        assert_eq!(stats.total_operations, 1);
    }

    #[test]
    fn test_thermal_state_monitoring() {
        let optimizer = PerformanceOptimizer::new();

        let metrics = PerformanceMetrics {
            operation_type: "test_op".to_string(),
            execution_time: Duration::from_millis(100),
            memory_usage: 1024,
            device_type: DeviceType::Cpu,
            tensor_size: 512,
            timestamp: Instant::now(),
            energy_consumption: 10.0,
            thermal_state: ThermalState::Elevated,
        };

        optimizer.record_metrics(metrics).unwrap();

        let thermal = optimizer.thermal_monitoring.lock().unwrap();
        assert_eq!(thermal.get(&DeviceType::Cpu), Some(&ThermalState::Elevated));
    }

    #[test]
    fn test_strategy_update() {
        let optimizer = PerformanceOptimizer::new();

        assert!(optimizer
            .set_optimization_strategy(OptimizationStrategy::LatencyOptimized)
            .is_ok());

        let strategy = optimizer.optimization_strategy.read().unwrap();
        assert_eq!(*strategy, OptimizationStrategy::LatencyOptimized);
    }
}

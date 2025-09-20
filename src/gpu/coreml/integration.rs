//! CoreML integration and hybrid execution patterns
//! CoreML統合とハイブリッド実行パターン

use super::common::*;
use super::device::CoreMLDeviceManager;
use super::backend::CoreMLBackend;
use super::operations::*;
use crate::tensor::Tensor;
use crate::error::RusTorchError;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};
use std::sync::Arc;

/// Hybrid execution strategy for CoreML operations
/// CoreML演算用ハイブリッド実行戦略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridStrategy {
    /// CoreML first, CPU fallback on failure
    /// CoreML優先、失敗時CPU フォールバック
    CoreMLPreferred,

    /// CPU first, CoreML on large operations
    /// CPU優先、大規模演算時CoreML
    CPUPreferred,

    /// Automatic selection based on operation characteristics
    /// 演算特性に基づく自動選択
    Automatic,

    /// Force CoreML execution (may fail)
    /// CoreML実行を強制（失敗する可能性あり）
    ForceCoreMI,

    /// Force CPU execution
    /// CPU実行を強制
    ForceCPU,
}

impl Default for HybridStrategy {
    fn default() -> Self {
        Self::Automatic
    }
}

/// CoreML hybrid executor for intelligent operation routing
/// インテリジェント演算ルーティング用CoreMLハイブリッド実行器
pub struct CoreMLHybridExecutor {
    backend: Arc<CoreMLBackend>,
    strategy: HybridStrategy,
    performance_stats: Arc<std::sync::Mutex<PerformanceStats>>,
}

/// Performance statistics for execution decisions
/// 実行決定用パフォーマンス統計
#[derive(Debug, Default)]
struct PerformanceStats {
    coreml_success_rate: f64,
    coreml_avg_time: std::time::Duration,
    cpu_avg_time: std::time::Duration,
    total_operations: usize,
}

impl CoreMLHybridExecutor {
    /// Create new hybrid executor
    /// 新しいハイブリッド実行器を作成
    pub fn new(strategy: HybridStrategy) -> CoreMLResult<Self> {
        let backend = Arc::new(CoreMLBackend::global().clone());

        Ok(Self {
            backend,
            strategy,
            performance_stats: Arc::new(std::sync::Mutex::new(PerformanceStats::default())),
        })
    }

    /// Execute operation with hybrid strategy
    /// ハイブリッド戦略で演算を実行
    pub fn execute<T, Op, CpuFn>(&self, operation: &Op, cpu_fallback: CpuFn) -> Result<Tensor<T>, RusTorchError>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
        Op: CoreMLOperation<T>,
        CpuFn: FnOnce() -> Result<Tensor<T>, RusTorchError>,
    {
        match self.decide_execution_path(operation) {
            ExecutionPath::CoreML => {
                self.execute_coreml_with_fallback(operation, cpu_fallback)
            }
            ExecutionPath::CPU => {
                self.execute_cpu_with_stats(cpu_fallback)
            }
            ExecutionPath::TryBoth => {
                // Try CoreML first, compare with CPU if both succeed
                let coreml_result = self.try_coreml_execution(operation);
                match coreml_result {
                    Ok(result) => Ok(result),
                    Err(_) => self.execute_cpu_with_stats(cpu_fallback),
                }
            }
        }
    }

    /// Decide execution path based on strategy and operation characteristics
    /// 戦略と演算特性に基づいて実行パスを決定
    fn decide_execution_path<T, Op>(&self, operation: &Op) -> ExecutionPath
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
        Op: CoreMLOperation<T>,
    {
        match self.strategy {
            HybridStrategy::ForceCoreMI => ExecutionPath::CoreML,
            HybridStrategy::ForceCPU => ExecutionPath::CPU,
            HybridStrategy::CoreMLPreferred => {
                if operation.is_supported_by_coreml() {
                    ExecutionPath::CoreML
                } else {
                    ExecutionPath::CPU
                }
            }
            HybridStrategy::CPUPreferred => {
                if self.should_use_coreml_for_large_ops(operation) {
                    ExecutionPath::CoreML
                } else {
                    ExecutionPath::CPU
                }
            }
            HybridStrategy::Automatic => {
                self.automatic_decision(operation)
            }
        }
    }

    /// Make automatic execution decision based on performance data
    /// パフォーマンスデータに基づいて自動実行決定を行う
    fn automatic_decision<T, Op>(&self, operation: &Op) -> ExecutionPath
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
        Op: CoreMLOperation<T>,
    {
        if !operation.is_supported_by_coreml() {
            return ExecutionPath::CPU;
        }

        if let Ok(stats) = self.performance_stats.lock() {
            if stats.total_operations < 10 {
                // Not enough data, use conservative approach
                return ExecutionPath::CoreML;
            }

            // Use CoreML if success rate is good and it's faster
            if stats.coreml_success_rate > 0.8 &&
               stats.coreml_avg_time < stats.cpu_avg_time * 2 {
                ExecutionPath::CoreML
            } else {
                ExecutionPath::CPU
            }
        } else {
            ExecutionPath::CoreML
        }
    }

    /// Check if operation should use CoreML based on size
    /// サイズに基づいて演算がCoreMLを使用すべきかチェック
    fn should_use_coreml_for_large_ops<T, Op>(&self, operation: &Op) -> bool
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
        Op: CoreMLOperation<T>,
    {
        if !operation.is_supported_by_coreml() {
            return false;
        }

        // Use estimated execution time as a proxy for operation size
        if let Some(estimated_time) = operation.estimated_execution_time() {
            estimated_time.as_millis() > 10 // > 10ms operations
        } else {
            false
        }
    }

    /// Execute CoreML with CPU fallback
    /// CPUフォールバック付きでCoreMLを実行
    fn execute_coreml_with_fallback<T, Op, CpuFn>(
        &self,
        operation: &Op,
        cpu_fallback: CpuFn,
    ) -> Result<Tensor<T>, RusTorchError>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
        Op: CoreMLOperation<T>,
        CpuFn: FnOnce() -> Result<Tensor<T>, RusTorchError>,
    {
        let start_time = std::time::Instant::now();

        match self.try_coreml_execution(operation) {
            Ok(result) => {
                self.update_stats(true, start_time.elapsed(), None);
                Ok(result)
            }
            Err(e) => {
                // CoreML failed, try CPU fallback
                let cpu_start = std::time::Instant::now();
                match cpu_fallback() {
                    Ok(result) => {
                        self.update_stats(false, start_time.elapsed(), Some(cpu_start.elapsed()));
                        Ok(result)
                    }
                    Err(cpu_err) => {
                        // Both failed, return the original CoreML error
                        Err(e)
                    }
                }
            }
        }
    }

    /// Execute CPU with performance tracking
    /// パフォーマンストラッキング付きでCPUを実行
    fn execute_cpu_with_stats<T, CpuFn>(&self, cpu_fallback: CpuFn) -> Result<Tensor<T>, RusTorchError>
    where
        CpuFn: FnOnce() -> Result<Tensor<T>, RusTorchError>,
    {
        let start_time = std::time::Instant::now();
        let result = cpu_fallback();

        if result.is_ok() {
            self.update_stats(false, std::time::Duration::ZERO, Some(start_time.elapsed()));
        }

        result
    }

    /// Try CoreML execution
    /// CoreML実行を試行
    fn try_coreml_execution<T, Op>(&self, operation: &Op) -> CoreMLResult<Tensor<T>>
    where
        T: Float + FromPrimitive + ScalarOperand + 'static,
        Op: CoreMLOperation<T>,
    {
        operation.execute_coreml(0) // Use device 0 for now
    }

    /// Update performance statistics
    /// パフォーマンス統計を更新
    fn update_stats(
        &self,
        coreml_success: bool,
        coreml_time: std::time::Duration,
        cpu_time: Option<std::time::Duration>,
    ) {
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.total_operations += 1;

            // Update CoreML success rate
            let success_count = (stats.coreml_success_rate * (stats.total_operations - 1) as f64) as usize;
            let new_success_count = if coreml_success { success_count + 1 } else { success_count };
            stats.coreml_success_rate = new_success_count as f64 / stats.total_operations as f64;

            // Update timing averages
            if coreml_success && coreml_time > std::time::Duration::ZERO {
                stats.coreml_avg_time = self.update_running_average(
                    stats.coreml_avg_time,
                    coreml_time,
                    stats.total_operations,
                );
            }

            if let Some(cpu_time) = cpu_time {
                stats.cpu_avg_time = self.update_running_average(
                    stats.cpu_avg_time,
                    cpu_time,
                    stats.total_operations,
                );
            }
        }
    }

    /// Update running average
    /// 移動平均を更新
    fn update_running_average(
        &self,
        current_avg: std::time::Duration,
        new_value: std::time::Duration,
        count: usize,
    ) -> std::time::Duration {
        let current_total = current_avg.as_nanos() as f64 * (count - 1) as f64;
        let new_total = current_total + new_value.as_nanos() as f64;
        std::time::Duration::from_nanos((new_total / count as f64) as u64)
    }

    /// Get current performance statistics
    /// 現在のパフォーマンス統計を取得
    pub fn get_performance_stats(&self) -> Option<(f64, std::time::Duration, std::time::Duration, usize)> {
        if let Ok(stats) = self.performance_stats.lock() {
            Some((
                stats.coreml_success_rate,
                stats.coreml_avg_time,
                stats.cpu_avg_time,
                stats.total_operations,
            ))
        } else {
            None
        }
    }

    /// Reset performance statistics
    /// パフォーマンス統計をリセット
    pub fn reset_stats(&self) {
        if let Ok(mut stats) = self.performance_stats.lock() {
            *stats = PerformanceStats::default();
        }
    }
}

/// Execution path decision
/// 実行パス決定
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionPath {
    CoreML,
    CPU,
    TryBoth,
}

/// Convenience macro for hybrid CoreML execution
/// ハイブリッドCoreML実行用便利マクロ
#[macro_export]
macro_rules! coreml_hybrid {
    ($operation:expr, $cpu_fallback:expr) => {{
        let executor = CoreMLHybridExecutor::new(HybridStrategy::Automatic)?;
        executor.execute(&$operation, || $cpu_fallback)
    }};
    ($operation:expr, $cpu_fallback:expr, $strategy:expr) => {{
        let executor = CoreMLHybridExecutor::new($strategy)?;
        executor.execute(&$operation, || $cpu_fallback)
    }};
}

/// Global hybrid executor instance
/// グローバルハイブリッド実行器インスタンス
static GLOBAL_HYBRID_EXECUTOR: std::sync::OnceLock<CoreMLHybridExecutor> = std::sync::OnceLock::new();

/// Get global hybrid executor
/// グローバルハイブリッド実行器を取得
pub fn global_hybrid_executor() -> &'static CoreMLHybridExecutor {
    GLOBAL_HYBRID_EXECUTOR.get_or_init(|| {
        CoreMLHybridExecutor::new(HybridStrategy::Automatic)
            .unwrap_or_else(|_| {
                // Fallback to CPU-preferred strategy if initialization fails
                CoreMLHybridExecutor::new(HybridStrategy::CPUPreferred)
                    .expect("Failed to create fallback hybrid executor")
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_executor_creation() {
        let executor = CoreMLHybridExecutor::new(HybridStrategy::Automatic);

        match executor {
            Ok(_) => {
                println!("Hybrid executor created successfully");
            }
            Err(e) => {
                println!("Expected failure on platforms without CoreML: {}", e);
            }
        }
    }

    #[test]
    fn test_global_hybrid_executor() {
        let executor = global_hybrid_executor();

        // Should always succeed (uses fallback strategy if needed)
        let stats = executor.get_performance_stats();
        assert!(stats.is_some());

        let (success_rate, _, _, operations) = stats.unwrap();
        assert_eq!(operations, 0); // No operations yet
        assert_eq!(success_rate, 0.0); // No operations yet
    }

    #[test]
    fn test_strategy_decisions() {
        use crate::tensor::Tensor;
        use super::super::operations::linear_algebra::MatMulOperation;

        // Create a small matrix operation (not suitable for CoreML)
        let a = Tensor::<f32>::zeros(&[2, 2]);
        let b = Tensor::<f32>::zeros(&[2, 2]);
        let operation = MatMulOperation::new(a, b);

        if let Ok(executor) = CoreMLHybridExecutor::new(HybridStrategy::CPUPreferred) {
            let path = executor.decide_execution_path(&operation);
            assert_eq!(path, ExecutionPath::CPU); // Should prefer CPU for small ops
        }
    }
}
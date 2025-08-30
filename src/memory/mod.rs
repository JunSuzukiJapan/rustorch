//! Enhanced Memory Management System for RusTorch
//! RusTorch用の高度メモリ管理システム
//!
//! This module provides comprehensive memory management capabilities:
//! - Enhanced memory pools with intelligent allocation strategies
//! - Memory pressure monitoring and adaptive garbage collection
//! - Memory analytics and profiling for performance optimization
//! - Memory-aware optimization with predictive allocation
//!
//! Features:
//! - NUMA-aware allocation
//! - Memory leak detection
//! - Automatic defragmentation
//! - Smart caching with priority-based eviction
//! - Zero-copy optimizations

pub mod analytics;
pub mod enhanced_pool;
pub mod optimizer;
pub mod pressure_monitor;

// Re-export enhanced memory management components
pub use analytics::{AllocationRecord, AnalyticsConfig, MemoryAnalytics, MemoryReport};
pub use enhanced_pool::{AllocationStrategy, EnhancedMemoryPool, EnhancedPoolStats, PoolConfig};
pub use optimizer::{MemoryOptimizer, MemoryPrediction, OptimizationStrategy, OptimizerConfig};
pub use pressure_monitor::{AdaptivePressureMonitor, GcStrategy, MonitorConfig, PressureLevel};

use crate::error::{RusTorchError, RusTorchResult};
use lazy_static::lazy_static;
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Legacy memory pool for backward compatibility
/// 後方互換性のためのレガシーメモリプール
pub struct MemoryPool<T: Float> {
    pools: Vec<Arc<Mutex<VecDeque<ArrayD<T>>>>>,
    max_pool_size: usize,
}

impl<T: Float + Clone + 'static> MemoryPool<T> {
    /// Create a new memory pool
    /// 新しいメモリプールを作成
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: Vec::new(),
            max_pool_size,
        }
    }

    /// Get pool index based on total elements
    /// 総要素数に基づいてプールインデックスを取得
    fn get_pool_index(&self, total_elements: usize) -> usize {
        // Use log2 bucketing for different sizes
        // サイズ別にlog2バケッティングを使用
        if total_elements <= 64 {
            0
        } else if total_elements <= 256 {
            1
        } else if total_elements <= 1024 {
            2
        } else if total_elements <= 4096 {
            3
        } else if total_elements <= 16384 {
            4
        } else if total_elements <= 65536 {
            5
        } else {
            6
        }
    }

    /// Ensure pool exists for given index
    /// 指定されたインデックスのプールが存在することを確認
    fn ensure_pool(&mut self, index: usize) {
        while self.pools.len() <= index {
            self.pools.push(Arc::new(Mutex::new(VecDeque::new())));
        }
    }

    /// Allocate a tensor from the pool or create new one
    /// プールからテンソルを割り当てるか新しく作成
    pub fn allocate(&mut self, shape: &[usize]) -> ArrayD<T> {
        let total_elements: usize = shape.iter().product();
        let pool_index = self.get_pool_index(total_elements);

        self.ensure_pool(pool_index);

        if let Ok(mut pool) = self.pools[pool_index].lock() {
            if let Some(mut array) = pool.pop_front() {
                // Reuse existing array if shape matches
                // 形状が一致する場合は既存の配列を再利用
                if array.shape() == shape {
                    array.fill(T::zero());
                    return array;
                }
                // If shape doesn't match, try to reshape
                // 形状が一致しない場合はリシェイプを試行
                if array.len() >= total_elements {
                    // Clone array before attempting reshape to avoid move
                    // リシェイプ試行前に配列をクローンして移動を回避
                    let cloned_array = array.clone();
                    match cloned_array.into_shape_with_order(IxDyn(shape)) {
                        Ok(reshaped) => return reshaped,
                        Err(_) => {
                            // Put back original array if reshape failed
                            // リシェイプが失敗した場合は元の配列を戻す
                            pool.push_back(array);
                        }
                    }
                } else {
                    // Put back if can't reuse
                    // 再利用できない場合は戻す
                    pool.push_back(array);
                }
            }
        }

        // Create new array if no suitable one in pool
        // プールに適切なものがない場合は新しい配列を作成
        ArrayD::zeros(IxDyn(shape))
    }

    /// Return a tensor to the pool for reuse
    /// 再利用のためにテンソルをプールに返却
    pub fn deallocate(&mut self, array: ArrayD<T>) {
        let total_elements = array.len();
        let pool_index = self.get_pool_index(total_elements);

        self.ensure_pool(pool_index);

        if let Ok(mut pool) = self.pools[pool_index].lock() {
            if pool.len() < self.max_pool_size {
                pool.push_back(array);
            }
            // If pool is full, just drop the array
            // プールが満杯の場合は配列を破棄
        }
    }

    /// Get statistics about pool usage
    /// プール使用状況の統計を取得
    pub fn stats(&self) -> PoolStats {
        let mut total_cached = 0;
        let mut pool_sizes = Vec::new();

        for pool in &self.pools {
            if let Ok(pool) = pool.lock() {
                let size = pool.len();
                pool_sizes.push(size);
                total_cached += size;
            }
        }

        PoolStats {
            total_pools: self.pools.len(),
            total_cached_arrays: total_cached,
            pool_sizes,
            max_pool_size: self.max_pool_size,
        }
    }

    /// Clear all pools
    /// 全プールをクリア
    pub fn clear(&mut self) {
        for pool in &self.pools {
            if let Ok(mut pool) = pool.lock() {
                pool.clear();
            }
        }
    }
}

/// Statistics about memory pool usage
/// メモリプール使用統計
#[derive(Debug)]
pub struct PoolStats {
    /// Total number of memory pools
    /// メモリプールの総数
    pub total_pools: usize,
    /// Total number of cached arrays across all pools
    /// 全プールでキャッシュされた配列の総数
    pub total_cached_arrays: usize,
    /// Size of each memory pool
    /// 各メモリプールのサイズ
    pub pool_sizes: Vec<usize>,
    /// Maximum size of any single pool
    /// 単一プールの最大サイズ
    pub max_pool_size: usize,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Pool Statistics:")?;
        writeln!(f, "  Total pools: {}", self.total_pools)?;
        writeln!(f, "  Total cached arrays: {}", self.total_cached_arrays)?;
        writeln!(f, "  Max pool size: {}", self.max_pool_size)?;
        writeln!(f, "  Pool sizes: {:?}", self.pool_sizes)?;
        Ok(())
    }
}

lazy_static! {
    static ref GLOBAL_POOL_F32: Arc<Mutex<MemoryPool<f32>>> =
        Arc::new(Mutex::new(MemoryPool::new(100)));
    static ref GLOBAL_POOL_F64: Arc<Mutex<MemoryPool<f64>>> =
        Arc::new(Mutex::new(MemoryPool::new(100)));
}

/// Get global memory pool for f32
/// f32用のグローバルメモリプールを取得
pub fn get_f32_pool() -> Arc<Mutex<MemoryPool<f32>>> {
    GLOBAL_POOL_F32.clone()
}

/// Get global memory pool for f64
/// f64用のグローバルメモリプールを取得
pub fn get_f64_pool() -> Arc<Mutex<MemoryPool<f64>>> {
    GLOBAL_POOL_F64.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let pool: MemoryPool<f32> = MemoryPool::new(10);
        let stats = pool.stats();
        assert_eq!(stats.total_pools, 0);
        assert_eq!(stats.total_cached_arrays, 0);
    }

    #[test]
    fn test_allocate_and_deallocate() {
        let mut pool: MemoryPool<f32> = MemoryPool::new(10);

        // Allocate
        let array1 = pool.allocate(&[2, 3]);
        assert_eq!(array1.shape(), &[2, 3]);

        // Deallocate
        pool.deallocate(array1);

        let stats = pool.stats();
        assert_eq!(stats.total_cached_arrays, 1);
    }

    #[test]
    fn test_reuse_from_pool() {
        let mut pool: MemoryPool<f32> = MemoryPool::new(10);

        // Allocate and deallocate
        let array1 = pool.allocate(&[2, 3]);
        pool.deallocate(array1);

        // Allocate same size - should reuse
        let array2 = pool.allocate(&[2, 3]);
        assert_eq!(array2.shape(), &[2, 3]);

        let stats = pool.stats();
        assert_eq!(stats.total_cached_arrays, 0); // Should be taken from pool
    }

    #[test]
    fn test_pool_size_limit() {
        let mut pool: MemoryPool<f32> = MemoryPool::new(2);

        // Add more arrays than pool size
        for _ in 0..5 {
            let array = pool.allocate(&[2, 2]);
            pool.deallocate(array);
        }

        let stats = pool.stats();
        assert!(stats.total_cached_arrays <= 2);
    }

    #[test]
    fn test_global_pools() {
        let pool_f32 = get_f32_pool();
        let pool_f64 = get_f64_pool();

        assert!(pool_f32.lock().is_ok());
        assert!(pool_f64.lock().is_ok());
    }
}

/// Comprehensive Memory Management System
/// 包括的メモリ管理システム
pub struct ComprehensiveMemoryManager<T: Float + Clone + Send + Sync + 'static> {
    /// Enhanced memory pool
    /// 高度メモリプール
    pool: EnhancedMemoryPool<T>,
    /// Pressure monitor
    /// プレッシャー監視
    monitor: AdaptivePressureMonitor,
    /// Analytics engine
    /// 分析エンジン
    analytics: MemoryAnalytics,
    /// Memory optimizer
    /// メモリオプティマイザー
    optimizer: MemoryOptimizer<T>,
}

impl<T: Float + Clone + Send + Sync + 'static> ComprehensiveMemoryManager<T> {
    /// Create new comprehensive memory manager
    /// 新しい包括的メモリマネージャーを作成
    pub fn new() -> Self {
        let pool_config = PoolConfig::default();
        let monitor_config = MonitorConfig::default();
        let analytics_config = AnalyticsConfig::default();
        let optimizer_config = OptimizerConfig::default();

        Self {
            pool: EnhancedMemoryPool::new(pool_config),
            monitor: AdaptivePressureMonitor::new(monitor_config),
            analytics: MemoryAnalytics::new(analytics_config),
            optimizer: MemoryOptimizer::new(optimizer_config),
        }
    }

    /// Create with custom configurations
    /// カスタム設定で作成
    pub fn with_configs(
        pool_config: PoolConfig,
        monitor_config: MonitorConfig,
        analytics_config: AnalyticsConfig,
        optimizer_config: OptimizerConfig,
    ) -> Self {
        Self {
            pool: EnhancedMemoryPool::new(pool_config),
            monitor: AdaptivePressureMonitor::new(monitor_config),
            analytics: MemoryAnalytics::new(analytics_config),
            optimizer: MemoryOptimizer::new(optimizer_config),
        }
    }

    /// Allocate memory with full optimization
    /// 完全最適化でメモリを割り当て
    pub fn allocate(&self, shape: &[usize]) -> RusTorchResult<ArrayD<T>> {
        // Record allocation in analytics
        let source_location = format!("tensor::allocate::{:?}", shape);
        let alloc_id = self.analytics.record_allocation(
            shape.iter().product::<usize>() * std::mem::size_of::<T>(),
            source_location,
        )?;

        // Use optimizer for intelligent allocation
        let array = self.optimizer.optimize_allocation(shape)?;

        // Update memory usage in monitor (simplified)
        // In real implementation, we would collect actual usage data

        Ok(array)
    }

    /// Deallocate memory with optimization
    /// 最適化でメモリを解放
    pub fn deallocate(&self, array: ArrayD<T>, alloc_id: u64) -> RusTorchResult<()> {
        // Record deallocation
        self.analytics.record_deallocation(alloc_id)?;

        // Use optimizer for intelligent deallocation
        self.optimizer.optimize_deallocation(array)?;

        Ok(())
    }

    /// Start all monitoring and analysis systems
    /// 全ての監視・分析システムを開始
    pub fn start_all_systems(&self) -> RusTorchResult<()> {
        self.monitor.start_monitoring()?;
        self.analytics.start_analysis()?;
        Ok(())
    }

    /// Stop all monitoring and analysis systems
    /// 全ての監視・分析システムを停止
    pub fn stop_all_systems(&self) -> RusTorchResult<()> {
        self.monitor.stop_monitoring()?;
        self.analytics.stop_analysis()?;
        Ok(())
    }

    /// Generate comprehensive memory report
    /// 包括的メモリレポートを生成
    pub fn generate_comprehensive_report(&self) -> RusTorchResult<ComprehensiveReport> {
        let pool_stats = self.pool.get_stats()?;
        let monitor_stats = self.monitor.get_stats()?;
        let analytics_report = self.analytics.generate_report()?;
        let optimizer_stats = self.optimizer.get_stats()?;

        Ok(ComprehensiveReport {
            pool_stats,
            monitor_stats,
            analytics_report,
            optimizer_stats,
        })
    }

    /// Perform comprehensive memory optimization
    /// 包括的メモリ最適化を実行
    pub fn optimize_all(&self) -> RusTorchResult<OptimizationSummary> {
        let start_time = std::time::Instant::now();

        // Perform garbage collection
        let gc_stats = self.pool.garbage_collect()?;

        // Defragment memory
        let defrag_reclaimed = self.optimizer.defragment_memory()?;

        // Compress memory if beneficial
        let compression_saved = self.optimizer.compress_memory()?;

        let total_time = start_time.elapsed();

        Ok(OptimizationSummary {
            gc_memory_reclaimed: gc_stats.memory_reclaimed,
            defrag_memory_reclaimed: defrag_reclaimed,
            compression_memory_saved: compression_saved,
            total_optimization_time: total_time,
        })
    }

    /// Get current system health status
    /// 現在のシステムヘルス状況を取得
    pub fn get_health_status(&self) -> RusTorchResult<SystemHealthStatus> {
        let snapshot = self.monitor.get_current_snapshot()?;
        let trend = self.monitor.analyze_trend()?;
        let prediction = self.optimizer.predict_memory_usage()?;

        let health_score = self.calculate_health_score(&snapshot, &trend, &prediction)?;

        Ok(SystemHealthStatus {
            current_snapshot: snapshot,
            trend_analysis: trend,
            memory_prediction: prediction,
            health_score,
            recommendations: self.generate_recommendations(health_score)?,
        })
    }

    // Private helper methods

    fn calculate_health_score(
        &self,
        snapshot: &Option<pressure_monitor::MemorySnapshot>,
        trend: &Option<pressure_monitor::PressureTrend>,
        prediction: &Option<MemoryPrediction>,
    ) -> RusTorchResult<f64> {
        let mut score = 1.0;

        // Factor in current pressure
        if let Some(snap) = snapshot {
            score -= snap.pressure_ratio * 0.4; // Reduce score based on pressure
        }

        // Factor in trend
        if let Some(t) = trend {
            if t.direction > 0.0 {
                // Increasing pressure trend
                score -= t.strength * 0.3;
            }
        }

        // Factor in prediction confidence
        if let Some(pred) = prediction {
            if pred.confidence < 0.5 {
                score -= 0.1; // Reduce score for low prediction confidence
            }
        }

        Ok(score.max(0.0).min(1.0))
    }

    fn generate_recommendations(&self, health_score: f64) -> RusTorchResult<Vec<String>> {
        let mut recommendations = Vec::new();

        if health_score < 0.3 {
            recommendations.push("Critical: Immediate memory optimization required".to_string());
            recommendations.push("Consider reducing batch sizes or model complexity".to_string());
            recommendations.push("Run comprehensive memory cleanup".to_string());
        } else if health_score < 0.6 {
            recommendations.push("Warning: High memory pressure detected".to_string());
            recommendations.push("Consider running memory defragmentation".to_string());
            recommendations.push("Monitor memory usage closely".to_string());
        } else if health_score < 0.8 {
            recommendations.push("Moderate memory pressure".to_string());
            recommendations.push("Consider periodic cleanup".to_string());
        } else {
            recommendations.push("Memory system operating normally".to_string());
        }

        Ok(recommendations)
    }
}

impl<T: Float + Clone + Send + Sync + 'static> Default for ComprehensiveMemoryManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive memory report combining all subsystems
/// 全サブシステムを組み合わせた包括的メモリレポート
#[derive(Debug, Clone)]
pub struct ComprehensiveReport {
    /// Enhanced pool statistics
    /// 高度プール統計
    pub pool_stats: EnhancedPoolStats,
    /// Pressure monitor statistics
    /// プレッシャー監視統計
    pub monitor_stats: pressure_monitor::MonitorStats,
    /// Analytics report
    /// 分析レポート
    pub analytics_report: MemoryReport,
    /// Optimizer statistics
    /// オプティマイザー統計
    pub optimizer_stats: optimizer::OptimizationStats,
}

/// Memory optimization summary
/// メモリ最適化サマリー
#[derive(Debug, Clone)]
pub struct OptimizationSummary {
    /// Memory reclaimed by GC (bytes)
    /// GCにより回収されたメモリ（バイト）
    pub gc_memory_reclaimed: usize,
    /// Memory reclaimed by defragmentation (bytes)
    /// デフラグメンテーションにより回収されたメモリ（バイト）
    pub defrag_memory_reclaimed: usize,
    /// Memory saved by compression (bytes)
    /// 圧縮により節約されたメモリ（バイト）
    pub compression_memory_saved: usize,
    /// Total optimization time
    /// 総最適化時間
    pub total_optimization_time: std::time::Duration,
}

/// System health status
/// システムヘルス状況
#[derive(Debug, Clone)]
pub struct SystemHealthStatus {
    /// Current memory snapshot
    /// 現在のメモリスナップショット
    pub current_snapshot: Option<pressure_monitor::MemorySnapshot>,
    /// Trend analysis
    /// 傾向分析
    pub trend_analysis: Option<pressure_monitor::PressureTrend>,
    /// Memory prediction
    /// メモリ予測
    pub memory_prediction: Option<MemoryPrediction>,
    /// Overall health score (0.0 - 1.0)
    /// 総合ヘルススコア（0.0 - 1.0）
    pub health_score: f64,
    /// System recommendations
    /// システム推奨事項
    pub recommendations: Vec<String>,
}

impl std::fmt::Display for ComprehensiveReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Comprehensive Memory Management Report")?;
        writeln!(f, "=====================================")?;
        writeln!(f, "")?;
        writeln!(f, "Enhanced Pool Statistics:")?;
        writeln!(f, "{}", self.pool_stats)?;
        writeln!(f, "")?;
        writeln!(f, "Monitor Statistics:")?;
        writeln!(
            f,
            "  Total Snapshots: {}",
            self.monitor_stats.total_snapshots
        )?;
        writeln!(
            f,
            "  Average Pressure: {:.2}%",
            self.monitor_stats.avg_pressure * 100.0
        )?;
        writeln!(
            f,
            "  Peak Pressure: {:.2}%",
            self.monitor_stats.peak_pressure * 100.0
        )?;
        writeln!(f, "")?;
        writeln!(f, "{}", self.analytics_report)?;
        writeln!(f, "")?;
        writeln!(f, "Optimizer Statistics:")?;
        writeln!(
            f,
            "  Total Optimizations: {}",
            self.optimizer_stats.total_optimizations
        )?;
        writeln!(
            f,
            "  Memory Saved: {} bytes",
            self.optimizer_stats.memory_saved
        )?;
        writeln!(
            f,
            "  Cache Hit Ratio: {:.2}%",
            self.optimizer_stats.cache_hit_ratio * 100.0
        )?;
        writeln!(
            f,
            "  Zero-Copy Operations: {}",
            self.optimizer_stats.zero_copy_operations
        )?;
        Ok(())
    }
}

lazy_static! {
    /// Global comprehensive memory manager for f32
    /// f32用のグローバル包括的メモリマネージャー
    static ref GLOBAL_MEMORY_MANAGER_F32: Arc<Mutex<ComprehensiveMemoryManager<f32>>> =
        Arc::new(Mutex::new(ComprehensiveMemoryManager::new()));

    /// Global comprehensive memory manager for f64
    /// f64用のグローバル包括的メモリマネージャー
    static ref GLOBAL_MEMORY_MANAGER_F64: Arc<Mutex<ComprehensiveMemoryManager<f64>>> =
        Arc::new(Mutex::new(ComprehensiveMemoryManager::new()));
}

/// Get global comprehensive memory manager for f32
/// f32用のグローバル包括的メモリマネージャーを取得
pub fn get_global_memory_manager_f32() -> Arc<Mutex<ComprehensiveMemoryManager<f32>>> {
    GLOBAL_MEMORY_MANAGER_F32.clone()
}

/// Get global comprehensive memory manager for f64
/// f64用のグローバル包括的メモリマネージャーを取得
pub fn get_global_memory_manager_f64() -> Arc<Mutex<ComprehensiveMemoryManager<f64>>> {
    GLOBAL_MEMORY_MANAGER_F64.clone()
}

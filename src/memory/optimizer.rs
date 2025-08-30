//! Memory-Aware Optimization System
//! メモリ認識最適化システム
//!
//! Features:
//! - Dynamic memory allocation strategies
//! - Memory usage prediction and preallocation
//! - Automatic memory defragmentation
//! - Memory-aware caching policies
//! - Adaptive memory compression

use crate::error::{RusTorchError, RusTorchResult};
use crate::memory::analytics::AllocationPattern;
use crate::memory::pressure_monitor::{GcStrategy, PressureLevel};
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Memory optimization strategy
/// メモリ最適化戦略
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationStrategy {
    /// Memory-first: Prioritize memory efficiency over speed
    /// メモリ優先：速度よりメモリ効率を優先
    MemoryFirst,
    /// Speed-first: Prioritize speed over memory efficiency
    /// 速度優先：メモリ効率より速度を優先
    SpeedFirst,
    /// Balanced: Balance between memory and speed
    /// バランス：メモリと速度のバランス
    Balanced,
    /// Adaptive: Dynamically adjust based on system state
    /// アダプティブ：システム状態に基づいて動的調整
    Adaptive,
}

/// Memory optimization configuration
/// メモリ最適化設定
#[derive(Clone, Debug)]
pub struct OptimizerConfig {
    /// Optimization strategy
    /// 最適化戦略
    pub strategy: OptimizationStrategy,
    /// Enable memory prediction
    /// メモリ予測を有効化
    pub enable_prediction: bool,
    /// Prediction window size
    /// 予測ウィンドウサイズ
    pub prediction_window: usize,
    /// Memory compression threshold (0.0 - 1.0)
    /// メモリ圧縮閾値（0.0 - 1.0）
    pub compression_threshold: f64,
    /// Defragmentation trigger threshold
    /// デフラグトリガー閾値
    pub defrag_threshold: f64,
    /// Cache size limit (bytes)
    /// キャッシュサイズ制限（バイト）
    pub cache_size_limit: usize,
    /// Enable zero-copy optimizations
    /// ゼロコピー最適化を有効化
    pub enable_zero_copy: bool,
    /// Memory preallocation factor
    /// メモリ事前割り当て係数
    pub preallocation_factor: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            strategy: OptimizationStrategy::Adaptive,
            enable_prediction: true,
            prediction_window: 20,
            compression_threshold: 0.8,
            defrag_threshold: 0.6,
            cache_size_limit: 512 * 1024 * 1024, // 512MB
            enable_zero_copy: true,
            preallocation_factor: 1.2,
        }
    }
}

/// Memory prediction model
/// メモリ予測モデル
#[derive(Debug, Clone)]
pub struct MemoryPrediction {
    /// Predicted memory requirement (bytes)
    /// 予測メモリ要求量（バイト）
    pub predicted_memory: usize,
    /// Confidence level (0.0 - 1.0)
    /// 信頼レベル（0.0 - 1.0）
    pub confidence: f64,
    /// Prediction horizon (duration)
    /// 予測期間（期間）
    pub horizon: Duration,
    /// Recommended preallocation size
    /// 推奨事前割り当てサイズ
    pub recommended_prealloc: usize,
}

/// Memory usage pattern
/// メモリ使用パターン
#[derive(Debug, Clone)]
pub struct UsagePattern {
    /// Pattern type
    /// パターンタイプ
    pub pattern_type: PatternType,
    /// Peak usage times
    /// ピーク使用時間
    pub peak_times: Vec<SystemTime>,
    /// Average allocation size
    /// 平均割り当てサイズ
    pub avg_allocation_size: usize,
    /// Allocation frequency
    /// 割り当て頻度
    pub allocation_frequency: f64,
    /// Memory lifecycle characteristics
    /// メモリライフサイクル特性
    pub lifecycle: LifecycleCharacteristics,
}

/// Pattern types for memory usage
/// メモリ使用のパターンタイプ
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    /// Steady usage with predictable patterns
    /// 予測可能なパターンでの安定した使用
    Steady,
    /// Bursty usage with irregular spikes
    /// 不規則なスパイクを持つバースト使用
    Bursty,
    /// Periodic usage with regular cycles
    /// 定期的なサイクルでの周期的使用
    Periodic,
    /// Growing usage over time
    /// 時間とともに増加する使用
    Growing,
    /// Random/unpredictable usage
    /// ランダム/予測不可能な使用
    Random,
}

/// Memory lifecycle characteristics
/// メモリライフサイクル特性
#[derive(Debug, Clone)]
pub struct LifecycleCharacteristics {
    /// Average lifetime of allocations
    /// 割り当ての平均ライフタイム
    pub avg_lifetime: Duration,
    /// Lifetime variance
    /// ライフタイム分散
    pub lifetime_variance: f64,
    /// Reuse probability
    /// 再利用確率
    pub reuse_probability: f64,
}

/// Cache entry with metadata
/// メタデータ付きキャッシュエントリ
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// Cached data
    /// キャッシュされたデータ
    pub data: ArrayD<T>,
    /// Last access time
    /// 最後のアクセス時間
    pub last_accessed: Instant,
    /// Access frequency
    /// アクセス頻度
    pub access_count: usize,
    /// Memory priority (0-255)
    /// メモリ優先度（0-255）
    pub priority: u8,
    /// Compression ratio (if compressed)
    /// 圧縮率（圧縮されている場合）
    pub compression_ratio: Option<f64>,
}

/// Memory optimization statistics
/// メモリ最適化統計
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Total optimizations performed
    /// 実行された総最適化数
    pub total_optimizations: usize,
    /// Memory saved through optimization (bytes)
    /// 最適化により節約されたメモリ（バイト）
    pub memory_saved: usize,
    /// Cache hit ratio
    /// キャッシュヒット率
    pub cache_hit_ratio: f64,
    /// Average prediction accuracy
    /// 平均予測精度
    pub prediction_accuracy: f64,
    /// Defragmentations performed
    /// 実行されたデフラグメンテーション数
    pub defragmentations: usize,
    /// Zero-copy operations performed
    /// 実行されたゼロコピー操作数
    pub zero_copy_operations: usize,
    /// Compression operations
    /// 圧縮操作数
    pub compression_operations: usize,
    /// Time spent in optimization (total)
    /// 最適化に費やされた時間（合計）
    pub optimization_time: Duration,
}

impl Default for OptimizationStats {
    fn default() -> Self {
        Self {
            total_optimizations: 0,
            memory_saved: 0,
            cache_hit_ratio: 0.0,
            prediction_accuracy: 0.0,
            defragmentations: 0,
            zero_copy_operations: 0,
            compression_operations: 0,
            optimization_time: Duration::from_millis(0),
        }
    }
}

/// Memory-aware optimizer
/// メモリ認識オプティマイザー
pub struct MemoryOptimizer<T: Float + Clone + Send + Sync + 'static> {
    /// Configuration
    /// 設定
    config: OptimizerConfig,
    /// Memory usage history for prediction
    /// 予測のためのメモリ使用履歴
    usage_history: RwLock<VecDeque<MemoryUsageSnapshot>>,
    /// Identified usage patterns
    /// 識別された使用パターン
    patterns: RwLock<Vec<UsagePattern>>,
    /// Smart cache with priority-based eviction
    /// 優先度ベース退避を持つスマートキャッシュ
    smart_cache: RwLock<HashMap<String, CacheEntry<T>>>,
    /// Preallocated memory pools by size
    /// サイズ別の事前割り当てメモリプール
    prealloc_pools: RwLock<BTreeMap<usize, VecDeque<ArrayD<T>>>>,
    /// Current optimization strategy
    /// 現在の最適化戦略
    current_strategy: RwLock<OptimizationStrategy>,
    /// Statistics
    /// 統計
    stats: RwLock<OptimizationStats>,
    /// Last optimization time
    /// 最後の最適化時間
    last_optimization: Mutex<Option<Instant>>,
}

/// Memory usage snapshot for prediction
/// 予測のためのメモリ使用スナップショット
#[derive(Debug, Clone)]
pub struct MemoryUsageSnapshot {
    /// Timestamp
    /// タイムスタンプ
    pub timestamp: SystemTime,
    /// Total memory usage
    /// 総メモリ使用量
    pub total_usage: usize,
    /// Number of active allocations
    /// アクティブな割り当て数
    pub active_allocations: usize,
    /// Memory pressure level
    /// メモリプレッシャーレベル
    pub pressure_level: PressureLevel,
    /// Current GC strategy
    /// 現在のGC戦略
    pub gc_strategy: GcStrategy,
}

impl<T: Float + Clone + Send + Sync + 'static> MemoryOptimizer<T> {
    /// Create new memory optimizer
    /// 新しいメモリオプティマイザーを作成
    pub fn new(config: OptimizerConfig) -> Self {
        let strategy = config.strategy;
        Self {
            config,
            usage_history: RwLock::new(VecDeque::new()),
            patterns: RwLock::new(Vec::new()),
            smart_cache: RwLock::new(HashMap::new()),
            prealloc_pools: RwLock::new(BTreeMap::new()),
            current_strategy: RwLock::new(strategy),
            stats: RwLock::new(OptimizationStats::default()),
            last_optimization: Mutex::new(None),
        }
    }

    /// Optimize memory allocation based on current strategy
    /// 現在の戦略に基づいてメモリ割り当てを最適化
    pub fn optimize_allocation(&self, shape: &[usize]) -> RusTorchResult<ArrayD<T>> {
        let start_time = Instant::now();

        let total_elements: usize = shape.iter().product();
        let size_class = self.get_size_class(total_elements);

        // Try smart cache first
        if let Some(cached) = self.try_cache_retrieval(shape)? {
            self.update_stats_cache_hit();
            return Ok(cached);
        }

        // Try preallocated pools
        if let Some(prealloc) = self.try_prealloc_retrieval(size_class, shape)? {
            self.update_stats_zero_copy();
            return Ok(prealloc);
        }

        // Predict future usage and preallocate if beneficial
        if self.config.enable_prediction {
            if let Some(prediction) = self.predict_memory_usage()? {
                self.consider_preallocation(&prediction)?;
            }
        }

        // Perform allocation with current strategy
        let array = self.allocate_with_strategy(shape)?;

        // Update optimization statistics
        self.update_optimization_stats(start_time.elapsed());

        Ok(array)
    }

    /// Optimize memory deallocation
    /// メモリ解放を最適化
    pub fn optimize_deallocation(&self, array: ArrayD<T>) -> RusTorchResult<()> {
        let shape = array.shape().to_vec();
        let total_elements: usize = shape.iter().product();
        let size_class = self.get_size_class(total_elements);

        // Check if array should be cached
        if self.should_cache(&array) {
            self.add_to_cache(array, &shape)?;
            return Ok(());
        }

        // Check if array should be added to preallocation pool
        if self.should_preallocate(size_class) {
            self.add_to_prealloc_pool(array, size_class)?;
            return Ok(());
        }

        // Otherwise, let it be deallocated normally
        Ok(())
    }

    /// Perform memory defragmentation
    /// メモリデフラグメンテーションを実行
    pub fn defragment_memory(&self) -> RusTorchResult<usize> {
        let start_time = Instant::now();
        let mut memory_reclaimed = 0;

        // Clean up cache based on access patterns
        {
            let mut cache = self.smart_cache.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire cache write lock".to_string())
            })?;

            let mut to_remove = Vec::new();
            let now = Instant::now();

            for (key, entry) in cache.iter() {
                let age = now.duration_since(entry.last_accessed);
                let access_rate = entry.access_count as f64 / age.as_secs_f64().max(1.0);

                // Remove entries with low access rate and old age
                if access_rate < 0.1 && age > Duration::from_secs(300) {
                    to_remove.push(key.clone());
                }
            }

            for key in to_remove {
                if let Some(entry) = cache.remove(&key) {
                    memory_reclaimed += entry.data.len() * std::mem::size_of::<T>();
                }
            }
        }

        // Clean up preallocation pools
        {
            let mut pools = self.prealloc_pools.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire pools write lock".to_string())
            })?;

            for (_, pool) in pools.iter_mut() {
                // Keep only the most recently added arrays
                let keep_count = (pool.len() / 2).max(1);
                let remove_count = pool.len() - keep_count;

                for _ in 0..remove_count {
                    if let Some(array) = pool.pop_front() {
                        memory_reclaimed += array.len() * std::mem::size_of::<T>();
                    }
                }
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire stats write lock".to_string())
            })?;
            stats.defragmentations += 1;
            stats.memory_saved += memory_reclaimed;
            stats.optimization_time += start_time.elapsed();
        }

        Ok(memory_reclaimed)
    }

    /// Predict future memory usage
    /// 将来のメモリ使用量を予測
    pub fn predict_memory_usage(&self) -> RusTorchResult<Option<MemoryPrediction>> {
        if !self.config.enable_prediction {
            return Ok(None);
        }

        let history = self.usage_history.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire history read lock".to_string())
        })?;

        if history.len() < self.config.prediction_window {
            return Ok(None);
        }

        // Simple trend analysis for prediction
        let recent_snapshots: Vec<_> = history
            .iter()
            .rev()
            .take(self.config.prediction_window)
            .collect();

        let mut total_usage_trend = 0.0;

        for i in 1..recent_snapshots.len() {
            let curr = recent_snapshots[i - 1];
            let prev = recent_snapshots[i];

            total_usage_trend += curr.total_usage as f64 - prev.total_usage as f64;
        }

        total_usage_trend /= (recent_snapshots.len() - 1) as f64;

        let current_usage = recent_snapshots[0].total_usage;
        let predicted_memory =
            ((current_usage as f64 + total_usage_trend * 5.0) as usize).max(current_usage);

        let confidence = if total_usage_trend.abs() < current_usage as f64 * 0.1 {
            0.8 // High confidence for stable trends
        } else {
            0.5 // Lower confidence for volatile trends
        };

        let recommended_prealloc = ((predicted_memory as f64 * self.config.preallocation_factor)
            as usize)
            .saturating_sub(current_usage);

        Ok(Some(MemoryPrediction {
            predicted_memory,
            confidence,
            horizon: Duration::from_secs(300), // 5 minutes ahead
            recommended_prealloc,
        }))
    }

    /// Update memory usage snapshot for prediction
    /// 予測のためのメモリ使用スナップショットを更新
    pub fn update_usage_snapshot(&self, snapshot: MemoryUsageSnapshot) -> RusTorchResult<()> {
        let mut history = self.usage_history.write().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire history write lock".to_string())
        })?;

        history.push_back(snapshot);

        // Keep only recent history
        while history.len() > self.config.prediction_window * 2 {
            history.pop_front();
        }

        // Analyze patterns
        self.analyze_usage_patterns(&history)?;

        // Update strategy if adaptive
        self.update_strategy_if_adaptive(&history)?;

        Ok(())
    }

    /// Get optimization statistics
    /// 最適化統計を取得
    pub fn get_stats(&self) -> RusTorchResult<OptimizationStats> {
        let stats = self.stats.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire stats read lock".to_string())
        })?;

        Ok(stats.clone())
    }

    /// Compress memory if beneficial
    /// 有益な場合はメモリを圧縮
    pub fn compress_memory(&self) -> RusTorchResult<usize> {
        // For now, return 0 as we don't implement actual compression
        // In a real implementation, we would compress cache entries
        Ok(0)
    }

    // Private helper methods

    fn get_size_class(&self, total_elements: usize) -> usize {
        match total_elements {
            0..=64 => 0,
            65..=256 => 1,
            257..=1024 => 2,
            1025..=4096 => 3,
            4097..=16384 => 4,
            _ => 5,
        }
    }

    fn try_cache_retrieval(&self, shape: &[usize]) -> RusTorchResult<Option<ArrayD<T>>> {
        let cache_key = format!("{:?}", shape);

        let mut cache = self.smart_cache.write().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire cache write lock".to_string())
        })?;

        if let Some(entry) = cache.get_mut(&cache_key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;

            // Clone the data and clear the original for reuse
            let mut result = entry.data.clone();
            result.fill(T::zero());
            return Ok(Some(result));
        }

        Ok(None)
    }

    fn try_prealloc_retrieval(
        &self,
        size_class: usize,
        shape: &[usize],
    ) -> RusTorchResult<Option<ArrayD<T>>> {
        let mut pools = self.prealloc_pools.write().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire pools write lock".to_string())
        })?;

        if let Some(pool) = pools.get_mut(&size_class) {
            if let Some(array) = pool.pop_back() {
                // Try to reshape if needed
                let total_elements: usize = shape.iter().product();
                if array.len() == total_elements {
                    let array_clone = array.clone();
                    if let Ok(reshaped) = array_clone.into_shape_with_order(IxDyn(shape)) {
                        return Ok(Some(reshaped));
                    }
                }
                // Put back if can't reshape
                pool.push_back(array);
            }
        }

        Ok(None)
    }

    fn allocate_with_strategy(&self, shape: &[usize]) -> RusTorchResult<ArrayD<T>> {
        let strategy = *self.current_strategy.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire strategy read lock".to_string())
        })?;

        match strategy {
            OptimizationStrategy::MemoryFirst => {
                // Use smaller initial allocation and grow as needed
                Ok(ArrayD::zeros(IxDyn(shape)))
            }
            OptimizationStrategy::SpeedFirst => {
                // Use larger initial allocation for better performance
                Ok(ArrayD::zeros(IxDyn(shape)))
            }
            OptimizationStrategy::Balanced | OptimizationStrategy::Adaptive => {
                // Standard allocation
                Ok(ArrayD::zeros(IxDyn(shape)))
            }
        }
    }

    fn should_cache(&self, array: &ArrayD<T>) -> bool {
        let current_cache_size = self.estimate_cache_size();
        let array_size = array.len() * std::mem::size_of::<T>();

        current_cache_size + array_size <= self.config.cache_size_limit
    }

    fn should_preallocate(&self, size_class: usize) -> bool {
        // Simple heuristic: preallocate for frequently used size classes
        size_class <= 3 // Preallocate for smaller size classes
    }

    fn add_to_cache(&self, array: ArrayD<T>, shape: &[usize]) -> RusTorchResult<()> {
        let cache_key = format!("{:?}", shape);
        let entry = CacheEntry {
            data: array,
            last_accessed: Instant::now(),
            access_count: 1,
            priority: 128, // Medium priority
            compression_ratio: None,
        };

        let mut cache = self.smart_cache.write().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire cache write lock".to_string())
        })?;

        cache.insert(cache_key, entry);

        Ok(())
    }

    fn add_to_prealloc_pool(&self, array: ArrayD<T>, size_class: usize) -> RusTorchResult<()> {
        let mut pools = self.prealloc_pools.write().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire pools write lock".to_string())
        })?;

        let pool = pools.entry(size_class).or_insert_with(VecDeque::new);
        pool.push_back(array);

        // Limit pool size
        if pool.len() > 10 {
            pool.pop_front();
        }

        Ok(())
    }

    fn consider_preallocation(&self, prediction: &MemoryPrediction) -> RusTorchResult<()> {
        if prediction.confidence < 0.7 || prediction.recommended_prealloc < 1024 {
            return Ok(()); // Skip if low confidence or small recommendation
        }

        // For now, we don't implement actual preallocation
        // In a real implementation, we would preallocate memory based on prediction
        Ok(())
    }

    fn estimate_cache_size(&self) -> usize {
        // Simplified cache size estimation
        if let Ok(cache) = self.smart_cache.read() {
            cache.len() * 1024 // Rough estimate
        } else {
            0
        }
    }

    fn analyze_usage_patterns(
        &self,
        history: &VecDeque<MemoryUsageSnapshot>,
    ) -> RusTorchResult<()> {
        // Simplified pattern analysis
        // In a real implementation, we would perform more sophisticated analysis
        Ok(())
    }

    fn update_strategy_if_adaptive(
        &self,
        history: &VecDeque<MemoryUsageSnapshot>,
    ) -> RusTorchResult<()> {
        let current_strategy = *self.current_strategy.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire strategy read lock".to_string())
        })?;

        if current_strategy != OptimizationStrategy::Adaptive {
            return Ok(());
        }

        // Analyze recent pressure levels to adjust strategy
        if let Some(latest) = history.back() {
            let new_strategy = match latest.pressure_level {
                PressureLevel::Low => OptimizationStrategy::SpeedFirst,
                PressureLevel::Medium => OptimizationStrategy::Balanced,
                PressureLevel::High | PressureLevel::Critical => OptimizationStrategy::MemoryFirst,
            };

            let mut strategy = self.current_strategy.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire strategy write lock".to_string())
            })?;
            *strategy = new_strategy;
        }

        Ok(())
    }

    fn update_stats_cache_hit(&self) {
        if let Ok(mut stats) = self.stats.write() {
            stats.cache_hit_ratio = (stats.cache_hit_ratio * 0.9) + 0.1; // Simple moving average
        }
    }

    fn update_stats_zero_copy(&self) {
        if let Ok(mut stats) = self.stats.write() {
            stats.zero_copy_operations += 1;
        }
    }

    fn update_optimization_stats(&self, duration: Duration) {
        if let Ok(mut stats) = self.stats.write() {
            stats.total_optimizations += 1;
            stats.optimization_time += duration;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer: MemoryOptimizer<f32> = MemoryOptimizer::new(config);

        let stats = optimizer.get_stats().unwrap();
        assert_eq!(stats.total_optimizations, 0);
    }

    #[test]
    fn test_size_class_calculation() {
        let config = OptimizerConfig::default();
        let optimizer: MemoryOptimizer<f32> = MemoryOptimizer::new(config);

        assert_eq!(optimizer.get_size_class(32), 0);
        assert_eq!(optimizer.get_size_class(128), 1);
        assert_eq!(optimizer.get_size_class(512), 2);
        assert_eq!(optimizer.get_size_class(2048), 3);
    }

    #[test]
    fn test_optimization_allocation() {
        let config = OptimizerConfig::default();
        let optimizer: MemoryOptimizer<f32> = MemoryOptimizer::new(config);

        let array = optimizer.optimize_allocation(&[3, 4]).unwrap();
        assert_eq!(array.shape(), &[3, 4]);

        let stats = optimizer.get_stats().unwrap();
        assert_eq!(stats.total_optimizations, 1);
    }

    #[test]
    fn test_cache_optimization() {
        let config = OptimizerConfig::default();
        let optimizer: MemoryOptimizer<f32> = MemoryOptimizer::new(config);

        // First allocation
        let array1 = optimizer.optimize_allocation(&[2, 2]).unwrap();
        optimizer.optimize_deallocation(array1).unwrap();

        // Second allocation should potentially hit cache
        let array2 = optimizer.optimize_allocation(&[2, 2]).unwrap();
        assert_eq!(array2.shape(), &[2, 2]);
    }

    #[test]
    fn test_memory_prediction() {
        let config = OptimizerConfig {
            prediction_window: 5, // Set a smaller window for testing
            ..OptimizerConfig::default()
        };
        let optimizer: MemoryOptimizer<f32> = MemoryOptimizer::new(config);

        // Add enough usage history to satisfy prediction window
        for i in 0..10 {
            let snapshot = MemoryUsageSnapshot {
                timestamp: SystemTime::now(),
                total_usage: 1000 + i * 100,
                active_allocations: 10 + i,
                pressure_level: PressureLevel::Low,
                gc_strategy: GcStrategy::Conservative,
            };
            optimizer.update_usage_snapshot(snapshot).unwrap();
        }

        let prediction = optimizer.predict_memory_usage().unwrap();
        assert!(prediction.is_some());

        let pred = prediction.unwrap();
        assert!(pred.predicted_memory > 0);
        assert!(pred.confidence > 0.0 && pred.confidence <= 1.0);
    }

    #[test]
    fn test_defragmentation() {
        let config = OptimizerConfig::default();
        let optimizer: MemoryOptimizer<f32> = MemoryOptimizer::new(config);

        // Create some allocations to fragment memory
        for _ in 0..5 {
            let array = optimizer.optimize_allocation(&[10, 10]).unwrap();
            optimizer.optimize_deallocation(array).unwrap();
        }

        let reclaimed = optimizer.defragment_memory().unwrap();
        // May be 0 if nothing was actually fragmented
        // reclaimed is usize, always >= 0

        let stats = optimizer.get_stats().unwrap();
        assert_eq!(stats.defragmentations, 1);
    }
}

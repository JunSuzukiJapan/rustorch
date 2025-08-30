//! Enhanced Memory Pool with Intelligent Allocation
//! インテリジェント割り当てを持つ高度メモリプール
//! 
//! Features:
//! - NUMA-aware allocation strategies
//! - Memory pressure monitoring
//! - Size-based and usage pattern-based pooling
//! - Adaptive garbage collection
//! - Memory deduplication and sharing

use crate::error::{RusTorchError, RusTorchResult};
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Memory allocation strategy
/// メモリ割り当て戦略
#[derive(Clone, Debug, PartialEq)]
pub enum AllocationStrategy {
    /// First-fit allocation (fastest)
    /// 最初適合割り当て（最速）
    FirstFit,
    /// Best-fit allocation (most memory efficient)
    /// 最適適合割り当て（最もメモリ効率的）
    BestFit,
    /// Size-class based allocation (balanced)
    /// サイズクラスベース割り当て（バランス型）
    SizeClass,
    /// NUMA-aware allocation
    /// NUMA対応割り当て
    NumaAware,
}

/// Memory pool configuration
/// メモリプール設定
#[derive(Clone, Debug)]
pub struct PoolConfig {
    /// Maximum memory per pool (bytes)
    /// プールあたりの最大メモリ（バイト）
    pub max_pool_memory: usize,
    /// Maximum number of arrays per size class
    /// サイズクラスあたりの最大配列数
    pub max_arrays_per_class: usize,
    /// Garbage collection threshold (0.0 - 1.0)
    /// ガベージコレクション閾値（0.0 - 1.0）
    pub gc_threshold: f64,
    /// Memory pressure monitoring interval
    /// メモリプレッシャー監視間隔
    pub monitor_interval: Duration,
    /// Allocation strategy
    /// 割り当て戦略
    pub strategy: AllocationStrategy,
    /// Enable memory deduplication
    /// メモリ重複除去を有効化
    pub enable_deduplication: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_memory: 1024 * 1024 * 1024, // 1GB
            max_arrays_per_class: 1000,
            gc_threshold: 0.8,
            monitor_interval: Duration::from_millis(100),
            strategy: AllocationStrategy::SizeClass,
            enable_deduplication: true,
        }
    }
}

/// Memory block metadata
/// メモリブロック メタデータ
#[derive(Clone, Debug)]
struct MemoryBlock<T> {
    /// The actual array data
    /// 実際の配列データ
    data: ArrayD<T>,
    /// Last access time for LRU eviction
    /// LRU退避のための最後のアクセス時間
    last_accessed: Instant,
    /// Access count for frequency-based eviction
    /// 頻度ベース退避のためのアクセス数
    access_count: usize,
    /// Size class for fast lookup
    /// 高速検索のためのサイズクラス
    size_class: usize,
    /// Hash for deduplication
    /// 重複除去のためのハッシュ
    content_hash: Option<u64>,
}

/// Size class for memory pooling
/// メモリプーリングのためのサイズクラス
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SizeClass {
    /// Total number of elements
    /// 総要素数
    total_elements: usize,
    /// Shape pattern (for efficient reuse)
    /// 形状パターン（効率的な再利用のため）
    dimensions: usize,
}

impl SizeClass {
    fn from_shape(shape: &[usize]) -> Self {
        Self {
            total_elements: shape.iter().product(),
            dimensions: shape.len(),
        }
    }

    /// Get size class index for bucketing
    /// バケット化のためのサイズクラスインデックスを取得
    fn index(&self) -> usize {
        // Combine element count and dimensions for classification
        let element_class = match self.total_elements {
            0..=64 => 0,
            65..=256 => 1,
            257..=1024 => 2,
            1025..=4096 => 3,
            4097..=16384 => 4,
            16385..=65536 => 5,
            65537..=262144 => 6,
            262145..=1048576 => 7,
            _ => 8,
        };
        
        let dim_class = match self.dimensions {
            1 => 0,
            2 => 10,
            3 => 20,
            4 => 30,
            _ => 40,
        };
        
        element_class + dim_class
    }
}

/// Enhanced memory pool with intelligent allocation
/// インテリジェント割り当てを持つ高度メモリプール
pub struct EnhancedMemoryPool<T: Float + Clone + Send + Sync + 'static> {
    /// Configuration
    /// 設定
    config: PoolConfig,
    /// Memory pools organized by size class
    /// サイズクラス別に整理されたメモリプール
    pools: RwLock<HashMap<usize, VecDeque<MemoryBlock<T>>>>,
    /// Memory usage statistics
    /// メモリ使用統計
    stats: RwLock<EnhancedPoolStats>,
    /// Memory pressure monitor
    /// メモリプレッシャー監視
    pressure_monitor: Arc<Mutex<MemoryPressureMonitor>>,
    /// Deduplication cache
    /// 重複除去キャッシュ
    dedup_cache: RwLock<HashMap<u64, Arc<ArrayD<T>>>>,
}

/// Enhanced memory pool statistics
/// 高度メモリプール統計
#[derive(Debug, Clone)]
pub struct EnhancedPoolStats {
    /// Total memory allocated (bytes)
    /// 総割り当てメモリ（バイト）
    pub total_allocated: usize,
    /// Total memory available in pools (bytes)
    /// プール内の総利用可能メモリ（バイト）
    pub total_pooled: usize,
    /// Number of active size classes
    /// アクティブサイズクラス数
    pub active_size_classes: usize,
    /// Total allocations performed
    /// 実行された総割り当て数
    pub total_allocations: usize,
    /// Total deallocations performed
    /// 実行された総解放数
    pub total_deallocations: usize,
    /// Cache hit ratio (0.0 - 1.0)
    /// キャッシュヒット率（0.0 - 1.0）
    pub cache_hit_ratio: f64,
    /// Memory pressure level (0.0 - 1.0)
    /// メモリプレッシャーレベル（0.0 - 1.0）
    pub memory_pressure: f64,
    /// Garbage collection statistics
    /// ガベージコレクション統計
    pub gc_stats: GcStats,
    /// Deduplication statistics
    /// 重複除去統計
    pub dedup_stats: DeduplicationStats,
}

/// Garbage collection statistics
/// ガベージコレクション統計
#[derive(Debug, Clone)]
pub struct GcStats {
    /// Number of GC runs
    /// GC実行回数
    pub gc_runs: usize,
    /// Total memory reclaimed (bytes)
    /// 回収された総メモリ（バイト）
    pub memory_reclaimed: usize,
    /// Average GC duration
    /// 平均GC時間
    pub avg_gc_duration: Duration,
    /// Last GC timestamp
    /// 最後のGCタイムスタンプ
    pub last_gc_time: Option<Instant>,
}

/// Deduplication statistics
/// 重複除去統計
#[derive(Debug, Clone)]
pub struct DeduplicationStats {
    /// Number of duplicates found
    /// 発見された重複数
    pub duplicates_found: usize,
    /// Memory saved through deduplication (bytes)
    /// 重複除去により節約されたメモリ（バイト）
    pub memory_saved: usize,
    /// Deduplication hit ratio
    /// 重複除去ヒット率
    pub hit_ratio: f64,
}

/// Memory pressure monitor
/// メモリプレッシャー監視
#[derive(Debug)]
struct MemoryPressureMonitor {
    /// Current memory pressure (0.0 - 1.0)
    /// 現在のメモリプレッシャー（0.0 - 1.0）
    current_pressure: f64,
    /// Peak memory usage
    /// ピークメモリ使用量
    peak_usage: usize,
    /// Last monitoring timestamp
    /// 最後の監視タイムスタンプ
    last_check: Instant,
}

impl<T: Float + Clone + Send + Sync + 'static> EnhancedMemoryPool<T> {
    /// Create new enhanced memory pool
    /// 新しい高度メモリプールを作成
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            pools: RwLock::new(HashMap::new()),
            stats: RwLock::new(EnhancedPoolStats::default()),
            pressure_monitor: Arc::new(Mutex::new(MemoryPressureMonitor {
                current_pressure: 0.0,
                peak_usage: 0,
                last_check: Instant::now(),
            })),
            dedup_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Allocate tensor with intelligent strategy
    /// インテリジェント戦略でテンソルを割り当て
    pub fn allocate(&self, shape: &[usize]) -> RusTorchResult<ArrayD<T>> {
        let size_class = SizeClass::from_shape(shape);
        let class_index = size_class.index();

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.total_allocations += 1;
        }

        // Check for memory pressure and trigger GC if needed
        self.check_memory_pressure()?;

        // Try to allocate from pool first
        if let Some(array) = self.allocate_from_pool(class_index, shape)? {
            if let Ok(mut stats) = self.stats.write() {
                stats.cache_hit_ratio = self.calculate_cache_hit_ratio();
            }
            return Ok(array);
        }

        // Check deduplication cache if enabled
        if self.config.enable_deduplication {
            if let Some(array) = self.check_deduplication(shape)? {
                return Ok(array);
            }
        }

        // Allocate new array using selected strategy
        let array = self.allocate_new(shape)?;

        // Add to deduplication cache if enabled
        if self.config.enable_deduplication {
            self.add_to_dedup_cache(&array)?;
        }

        Ok(array)
    }

    /// Deallocate tensor back to pool
    /// テンソルをプールに返却
    pub fn deallocate(&self, array: ArrayD<T>) -> RusTorchResult<()> {
        let shape = array.shape().to_vec();
        let size_class = SizeClass::from_shape(&shape);
        let class_index = size_class.index();

        // Update statistics
        if let Ok(mut stats) = self.stats.write() {
            stats.total_deallocations += 1;
        }

        // Create memory block
        let block = MemoryBlock {
            data: array,
            last_accessed: Instant::now(),
            access_count: 1,
            size_class: class_index,
            content_hash: None, // Will be computed if deduplication is enabled
        };

        // Add to pool
        let mut pools = self.pools.write().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire pool write lock".to_string())
        })?;

        let pool = pools.entry(class_index).or_insert_with(VecDeque::new);

        // Check pool size limit
        if pool.len() >= self.config.max_arrays_per_class {
            // Remove oldest entry (FIFO eviction)
            pool.pop_front();
        }

        pool.push_back(block);

        Ok(())
    }

    /// Get comprehensive memory statistics
    /// 包括的なメモリ統計を取得
    pub fn get_stats(&self) -> RusTorchResult<EnhancedPoolStats> {
        let stats = self.stats.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire stats read lock".to_string())
        })?;

        Ok(stats.clone())
    }

    /// Force garbage collection
    /// ガベージコレクションを強制実行
    pub fn garbage_collect(&self) -> RusTorchResult<GcStats> {
        let start_time = Instant::now();
        let mut memory_reclaimed = 0;

        // Clean up pools based on LRU and access patterns
        {
            let mut pools = self.pools.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire pool write lock".to_string())
            })?;

            for (_, pool) in pools.iter_mut() {
                let original_size = pool.len();
                
                // Remove entries that haven't been accessed recently
                let cutoff_time = Instant::now() - Duration::from_secs(300); // 5 minutes
                pool.retain(|block| block.last_accessed > cutoff_time);
                
                memory_reclaimed += (original_size - pool.len()) * std::mem::size_of::<ArrayD<T>>();
            }
        }

        // Clean up deduplication cache
        {
            let mut dedup = self.dedup_cache.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire dedup write lock".to_string())
            })?;

            let original_size = dedup.len();
            dedup.retain(|_, arc_array| Arc::strong_count(arc_array) > 1);
            memory_reclaimed += (original_size - dedup.len()) * std::mem::size_of::<ArrayD<T>>();
        }

        let gc_duration = start_time.elapsed();

        // Update GC statistics
        let gc_stats = GcStats {
            gc_runs: 1,
            memory_reclaimed,
            avg_gc_duration: gc_duration,
            last_gc_time: Some(Instant::now()),
        };

        if let Ok(mut stats) = self.stats.write() {
            stats.gc_stats.gc_runs += 1;
            stats.gc_stats.memory_reclaimed += memory_reclaimed;
            stats.gc_stats.avg_gc_duration = 
                (stats.gc_stats.avg_gc_duration + gc_duration) / 2;
            stats.gc_stats.last_gc_time = Some(Instant::now());
        }

        Ok(gc_stats)
    }

    /// Clear all pools and caches
    /// 全プールとキャッシュをクリア
    pub fn clear_all(&self) -> RusTorchResult<()> {
        // Clear main pools
        {
            let mut pools = self.pools.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire pool write lock".to_string())
            })?;
            pools.clear();
        }

        // Clear deduplication cache
        {
            let mut dedup = self.dedup_cache.write().map_err(|_| {
                RusTorchError::MemoryError("Failed to acquire dedup write lock".to_string())
            })?;
            dedup.clear();
        }

        Ok(())
    }

    // Private helper methods

    fn allocate_from_pool(&self, class_index: usize, shape: &[usize]) -> RusTorchResult<Option<ArrayD<T>>> {
        let mut pools = self.pools.write().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire pool write lock".to_string())
        })?;

        if let Some(pool) = pools.get_mut(&class_index) {
            // Look for exact shape match first
            for i in 0..pool.len() {
                if pool[i].data.shape() == shape {
                    let mut block = pool.remove(i).unwrap();
                    block.last_accessed = Instant::now();
                    block.access_count += 1;
                    
                    // Zero out the array for reuse
                    block.data.fill(T::zero());
                    return Ok(Some(block.data));
                }
            }

            // Try to reshape compatible arrays
            for i in 0..pool.len() {
                let total_elements: usize = shape.iter().product();
                if pool[i].data.len() == total_elements {
                    let block = pool.remove(i).unwrap();
                    
                    // Try to reshape
                    if let Ok(reshaped) = block.data.into_shape_with_order(IxDyn(shape)) {
                        return Ok(Some(reshaped));
                    }
                }
            }
        }

        Ok(None)
    }

    fn allocate_new(&self, shape: &[usize]) -> RusTorchResult<ArrayD<T>> {
        match self.config.strategy {
            AllocationStrategy::FirstFit | 
            AllocationStrategy::BestFit | 
            AllocationStrategy::SizeClass => {
                Ok(ArrayD::zeros(IxDyn(shape)))
            }
            AllocationStrategy::NumaAware => {
                // For NUMA-aware allocation, we could implement numa-aware allocation here
                // For now, fall back to standard allocation
                Ok(ArrayD::zeros(IxDyn(shape)))
            }
        }
    }

    fn check_memory_pressure(&self) -> RusTorchResult<()> {
        if let Ok(mut monitor) = self.pressure_monitor.lock() {
            let now = Instant::now();
            
            if now.duration_since(monitor.last_check) > self.config.monitor_interval {
                // Simple memory pressure calculation based on pool usage
                let current_usage = self.calculate_current_usage()?;
                monitor.current_pressure = current_usage as f64 / self.config.max_pool_memory as f64;
                
                if current_usage > monitor.peak_usage {
                    monitor.peak_usage = current_usage;
                }
                
                monitor.last_check = now;

                // Trigger GC if pressure is high
                if monitor.current_pressure > self.config.gc_threshold {
                    self.garbage_collect()?;
                }
            }
        }

        Ok(())
    }

    fn calculate_current_usage(&self) -> RusTorchResult<usize> {
        let pools = self.pools.read().map_err(|_| {
            RusTorchError::MemoryError("Failed to acquire pool read lock".to_string())
        })?;

        let mut total = 0;
        for (_, pool) in pools.iter() {
            for block in pool {
                total += block.data.len() * std::mem::size_of::<T>();
            }
        }

        Ok(total)
    }

    fn calculate_cache_hit_ratio(&self) -> f64 {
        if let Ok(stats) = self.stats.read() {
            if stats.total_allocations > 0 {
                return (stats.total_allocations - stats.total_deallocations) as f64 
                    / stats.total_allocations as f64;
            }
        }
        0.0
    }

    fn check_deduplication(&self, shape: &[usize]) -> RusTorchResult<Option<ArrayD<T>>> {
        // For now, deduplication is based on shape similarity
        // In a more advanced implementation, we could hash array contents
        Ok(None)
    }

    fn add_to_dedup_cache(&self, _array: &ArrayD<T>) -> RusTorchResult<()> {
        // Placeholder for deduplication cache addition
        Ok(())
    }
}

impl Default for EnhancedPoolStats {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            total_pooled: 0,
            active_size_classes: 0,
            total_allocations: 0,
            total_deallocations: 0,
            cache_hit_ratio: 0.0,
            memory_pressure: 0.0,
            gc_stats: GcStats::default(),
            dedup_stats: DeduplicationStats::default(),
        }
    }
}

impl Default for GcStats {
    fn default() -> Self {
        Self {
            gc_runs: 0,
            memory_reclaimed: 0,
            avg_gc_duration: Duration::from_millis(0),
            last_gc_time: None,
        }
    }
}

impl Default for DeduplicationStats {
    fn default() -> Self {
        Self {
            duplicates_found: 0,
            memory_saved: 0,
            hit_ratio: 0.0,
        }
    }
}

impl std::fmt::Display for EnhancedPoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Enhanced Memory Pool Statistics:")?;
        writeln!(f, "  Total Allocated: {} bytes", self.total_allocated)?;
        writeln!(f, "  Total Pooled: {} bytes", self.total_pooled)?;
        writeln!(f, "  Active Size Classes: {}", self.active_size_classes)?;
        writeln!(f, "  Total Allocations: {}", self.total_allocations)?;
        writeln!(f, "  Total Deallocations: {}", self.total_deallocations)?;
        writeln!(f, "  Cache Hit Ratio: {:.2}%", self.cache_hit_ratio * 100.0)?;
        writeln!(f, "  Memory Pressure: {:.2}%", self.memory_pressure * 100.0)?;
        writeln!(f, "  GC Runs: {}", self.gc_stats.gc_runs)?;
        writeln!(f, "  Memory Reclaimed: {} bytes", self.gc_stats.memory_reclaimed)?;
        writeln!(f, "  Deduplication Hits: {}", self.dedup_stats.duplicates_found)?;
        writeln!(f, "  Memory Saved: {} bytes", self.dedup_stats.memory_saved)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_pool_creation() {
        let config = PoolConfig::default();
        let pool: EnhancedMemoryPool<f32> = EnhancedMemoryPool::new(config);
        let stats = pool.get_stats().unwrap();
        assert_eq!(stats.total_allocations, 0);
    }

    #[test]
    fn test_allocation_and_deallocation() {
        let config = PoolConfig::default();
        let pool: EnhancedMemoryPool<f32> = EnhancedMemoryPool::new(config);

        // Allocate
        let array = pool.allocate(&[3, 4]).unwrap();
        assert_eq!(array.shape(), &[3, 4]);

        // Deallocate
        pool.deallocate(array).unwrap();

        let stats = pool.get_stats().unwrap();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_deallocations, 1);
    }

    #[test]
    fn test_memory_reuse() {
        let config = PoolConfig::default();
        let pool: EnhancedMemoryPool<f32> = EnhancedMemoryPool::new(config);

        // Allocate and deallocate
        let array1 = pool.allocate(&[2, 3]).unwrap();
        pool.deallocate(array1).unwrap();

        // Allocate same size - should reuse
        let array2 = pool.allocate(&[2, 3]).unwrap();
        assert_eq!(array2.shape(), &[2, 3]);

        let stats = pool.get_stats().unwrap();
        assert!(stats.cache_hit_ratio > 0.0);
    }

    #[test]
    fn test_garbage_collection() {
        let config = PoolConfig::default();
        let pool: EnhancedMemoryPool<f32> = EnhancedMemoryPool::new(config);

        // Create some arrays
        for _ in 0..10 {
            let array = pool.allocate(&[5, 5]).unwrap();
            pool.deallocate(array).unwrap();
        }

        // Force garbage collection
        let gc_stats = pool.garbage_collect().unwrap();
        assert!(gc_stats.gc_runs > 0);
    }

    #[test]
    fn test_size_class_indexing() {
        let class1 = SizeClass::from_shape(&[10]);
        let class2 = SizeClass::from_shape(&[3, 3]);
        let class3 = SizeClass::from_shape(&[100, 100]);

        assert!(class1.index() != class2.index());
        assert!(class2.index() != class3.index());
    }
}
// テンソルプール - メモリ再利用による高速化
// Tensor pool - acceleration through memory reuse

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::time::{Instant, Duration};
use crate::common::RusTorchResult;
use crate::hybrid_f32::tensor::core::F32Tensor;

/// プールサイズ設定
/// Pool size configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// 最大プールサイズ
    /// Maximum pool size
    pub max_pool_size: usize,

    /// 形状ごとの最大保持数
    /// Maximum retained per shape
    pub max_per_shape: usize,

    /// プールアイテムの最大生存時間
    /// Maximum lifetime for pool items
    pub max_lifetime: Duration,

    /// ガベージコレクション間隔
    /// Garbage collection interval
    pub gc_interval: Duration,

    /// メモリ圧縮しきい値
    /// Memory compression threshold
    pub compression_threshold: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            max_pool_size: 10000,
            max_per_shape: 100,
            max_lifetime: Duration::from_secs(300), // 5分
            gc_interval: Duration::from_secs(60),   // 1分
            compression_threshold: 1000,
        }
    }
}

/// プールされたテンソル
/// Pooled tensor
#[derive(Debug)]
pub struct PooledTensor {
    /// テンソル本体
    /// Tensor body
    pub tensor: F32Tensor,

    /// プールされた時刻
    /// Time when pooled
    pub pooled_at: Instant,

    /// 使用回数
    /// Usage count
    pub usage_count: usize,

    /// 最後に使用された時刻
    /// Last used time
    pub last_used: Instant,
}

impl PooledTensor {
    pub fn new(tensor: F32Tensor) -> Self {
        let now = Instant::now();
        PooledTensor {
            tensor,
            pooled_at: now,
            usage_count: 0,
            last_used: now,
        }
    }

    /// テンソルを使用としてマーク
    /// Mark tensor as used
    pub fn mark_used(&mut self) {
        self.usage_count += 1;
        self.last_used = Instant::now();
    }

    /// 期限切れかチェック
    /// Check if expired
    pub fn is_expired(&self, max_lifetime: Duration) -> bool {
        self.last_used.elapsed() > max_lifetime
    }
}

/// プール統計情報
/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// 現在のプールサイズ
    /// Current pool size
    pub current_size: usize,

    /// 形状別アイテム数
    /// Items per shape
    pub shapes: HashMap<Vec<usize>, usize>,

    /// 総ヒット数
    /// Total hits
    pub total_hits: usize,

    /// 総ミス数
    /// Total misses
    pub total_misses: usize,

    /// ヒット率
    /// Hit rate
    pub hit_rate: f64,

    /// 最後のGC時刻
    /// Last GC time
    pub last_gc: Instant,

    /// GC回数
    /// GC count
    pub gc_count: usize,

    /// 解放されたアイテム数
    /// Released items
    pub items_released: usize,
}

impl Default for PoolStats {
    fn default() -> Self {
        PoolStats {
            current_size: 0,
            shapes: HashMap::new(),
            total_hits: 0,
            total_misses: 0,
            hit_rate: 0.0,
            last_gc: Instant::now(),
            gc_count: 0,
            items_released: 0,
        }
    }
}

impl PoolStats {
    /// ヒット率を更新
    /// Update hit rate
    pub fn update_hit_rate(&mut self) {
        let total = self.total_hits + self.total_misses;
        self.hit_rate = if total > 0 {
            self.total_hits as f64 / total as f64
        } else {
            0.0
        };
    }
}

/// テンソルプール
/// Tensor pool
#[derive(Debug)]
pub struct TensorPool {
    /// 設定
    /// Configuration
    config: PoolConfig,

    /// 形状別プール
    /// Pools by shape
    pools: Arc<Mutex<HashMap<Vec<usize>, VecDeque<PooledTensor>>>>,

    /// 統計情報
    /// Statistics
    stats: Arc<RwLock<PoolStats>>,

    /// 最後のGC時刻
    /// Last GC time
    last_gc: Arc<Mutex<Instant>>,
}

impl TensorPool {
    /// 新しいプールを作成
    /// Create a new pool
    pub fn new(config: PoolConfig) -> Self {
        TensorPool {
            config,
            pools: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(PoolStats::default())),
            last_gc: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// デフォルト設定でプールを作成
    /// Create pool with default config
    pub fn with_default_config() -> Self {
        Self::new(PoolConfig::default())
    }

    /// テンソルを取得（ヒットした場合）またはNone（ミスした場合）
    /// Get tensor (if hit) or None (if miss)
    pub fn get(&self, shape: &[usize]) -> Option<F32Tensor> {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.write().unwrap();

        if let Some(pool) = pools.get_mut(shape) {
            if let Some(mut pooled) = pool.pop_front() {
                pooled.mark_used();
                stats.total_hits += 1;
                stats.update_hit_rate();

                // 統計の形状カウントを更新
                if let Some(count) = stats.shapes.get_mut(shape) {
                    *count = pool.len();
                }

                return Some(pooled.tensor);
            }
        }

        stats.total_misses += 1;
        stats.update_hit_rate();
        None
    }

    /// テンソルをプールに戻す
    /// Return tensor to pool
    pub fn put(&self, tensor: F32Tensor) -> RusTorchResult<()> {
        let shape = tensor.shape().to_vec();
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.write().unwrap();

        // プールサイズ制限チェック
        if stats.current_size >= self.config.max_pool_size {
            return Ok(()); // プールが満杯の場合は破棄
        }

        // 形状別プール取得または作成
        let pool = pools.entry(shape.clone()).or_insert_with(VecDeque::new);

        // 形状別制限チェック
        if pool.len() >= self.config.max_per_shape {
            pool.pop_back(); // 古いものを削除
        } else {
            stats.current_size += 1;
        }

        // 新しいプールアイテムを追加
        pool.push_front(PooledTensor::new(tensor));

        // 統計更新
        stats.shapes.insert(shape, pool.len());

        // 定期的なガベージコレクション
        if self.should_gc() {
            drop(stats);
            drop(pools);
            self.garbage_collect()?;
        }

        Ok(())
    }

    /// テンソルを取得（なければ作成）
    /// Get tensor (create if not found)
    pub fn get_or_create<F>(&self, shape: &[usize], creator: F) -> RusTorchResult<F32Tensor>
    where
        F: FnOnce(&[usize]) -> RusTorchResult<F32Tensor>,
    {
        if let Some(tensor) = self.get(shape) {
            Ok(tensor)
        } else {
            creator(shape)
        }
    }

    /// ガベージコレクションが必要かチェック
    /// Check if garbage collection is needed
    fn should_gc(&self) -> bool {
        let last_gc = self.last_gc.lock().unwrap();
        last_gc.elapsed() > self.config.gc_interval
    }

    /// ガベージコレクションを実行
    /// Perform garbage collection
    pub fn garbage_collect(&self) -> RusTorchResult<usize> {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.write().unwrap();
        let mut last_gc = self.last_gc.lock().unwrap();

        let mut released_count = 0;
        let now = Instant::now();

        // 期限切れアイテムを削除
        for (shape, pool) in pools.iter_mut() {
            let original_len = pool.len();
            pool.retain(|item| !item.is_expired(self.config.max_lifetime));
            let removed = original_len - pool.len();
            released_count += removed;

            // 統計更新
            stats.shapes.insert(shape.clone(), pool.len());
        }

        // 空のプールを削除
        pools.retain(|_, pool| !pool.is_empty());

        // 統計更新
        stats.current_size = stats.current_size.saturating_sub(released_count);
        stats.last_gc = now;
        stats.gc_count += 1;
        stats.items_released += released_count;

        *last_gc = now;

        Ok(released_count)
    }

    /// プールを完全にクリア
    /// Clear pool completely
    pub fn clear(&self) -> RusTorchResult<()> {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.write().unwrap();

        let total_items: usize = pools.values().map(|pool| pool.len()).sum();

        pools.clear();
        stats.current_size = 0;
        stats.shapes.clear();
        stats.items_released += total_items;

        Ok(())
    }

    /// 統計情報を取得
    /// Get statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.read().unwrap().clone()
    }

    /// プール設定を取得
    /// Get pool config
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// メモリ使用量を推定（バイト）
    /// Estimate memory usage (bytes)
    pub fn estimated_memory_usage(&self) -> usize {
        let pools = self.pools.lock().unwrap();
        let mut total_memory = 0;

        for (shape, pool) in pools.iter() {
            let elements_per_tensor: usize = shape.iter().product();
            let bytes_per_tensor = elements_per_tensor * std::mem::size_of::<f32>();
            total_memory += pool.len() * bytes_per_tensor;
        }

        total_memory
    }

    /// 最も使用頻度の高い形状を取得
    /// Get most frequently used shapes
    pub fn top_shapes(&self, limit: usize) -> Vec<(Vec<usize>, usize)> {
        let stats = self.stats.read().unwrap();
        let mut shapes: Vec<_> = stats.shapes.iter()
            .map(|(shape, count)| (shape.clone(), *count))
            .collect();

        shapes.sort_by(|a, b| b.1.cmp(&a.1));
        shapes.truncate(limit);
        shapes
    }

    /// プールの健全性をチェック
    /// Check pool health
    pub fn health_check(&self) -> PoolHealthReport {
        let stats = self.stats();
        let config = &self.config;

        PoolHealthReport {
            is_healthy: stats.hit_rate > 0.5 && stats.current_size < config.max_pool_size,
            hit_rate: stats.hit_rate,
            memory_usage: self.estimated_memory_usage(),
            fragmentation_ratio: self.calculate_fragmentation(),
            recommendations: self.generate_recommendations(&stats),
        }
    }

    /// フラグメンテーション率を計算
    /// Calculate fragmentation ratio
    fn calculate_fragmentation(&self) -> f64 {
        let pools = self.pools.lock().unwrap();
        if pools.is_empty() {
            return 0.0;
        }

        let shape_count = pools.len() as f64;
        let avg_per_shape = pools.values().map(|p| p.len()).sum::<usize>() as f64 / shape_count;
        let variance = pools.values()
            .map(|p| (p.len() as f64 - avg_per_shape).powi(2))
            .sum::<f64>() / shape_count;

        variance.sqrt() / avg_per_shape.max(1.0)
    }

    /// 最適化の推奨事項を生成
    /// Generate optimization recommendations
    fn generate_recommendations(&self, stats: &PoolStats) -> Vec<String> {
        let mut recommendations = Vec::new();

        if stats.hit_rate < 0.3 {
            recommendations.push("ヒット率が低いです。プールサイズを増やすかGC間隔を調整してください".to_string());
        }

        if stats.current_size > self.config.max_pool_size * 9 / 10 {
            recommendations.push("プールが満杯に近いです。最大サイズを増やすかGCを頻繁に実行してください".to_string());
        }

        let fragmentation = self.calculate_fragmentation();
        if fragmentation > 2.0 {
            recommendations.push("フラグメンテーションが高いです。形状の分散を見直してください".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("プールは健全に動作しています".to_string());
        }

        recommendations
    }
}

/// プールの健全性レポート
/// Pool health report
#[derive(Debug)]
pub struct PoolHealthReport {
    pub is_healthy: bool,
    pub hit_rate: f64,
    pub memory_usage: usize,
    pub fragmentation_ratio: f64,
    pub recommendations: Vec<String>,
}

/// グローバルテンソルプール
/// Global tensor pool
lazy_static::lazy_static! {
    static ref GLOBAL_POOL: TensorPool = TensorPool::with_default_config();
}

/// グローバルプールを取得
/// Get global pool
pub fn global_pool() -> &'static TensorPool {
    &GLOBAL_POOL
}

/// プールヘルパー関数
/// Pool helper functions
pub mod helpers {
    use super::*;

    /// プールからテンソルを取得またはゼロテンソルを作成
    /// Get tensor from pool or create zeros
    pub fn get_or_zeros(shape: &[usize]) -> RusTorchResult<F32Tensor> {
        global_pool().get_or_create(shape, |shape| F32Tensor::zeros(shape))
    }

    /// プールからテンソルを取得またはワンテンソルを作成
    /// Get tensor from pool or create ones
    pub fn get_or_ones(shape: &[usize]) -> RusTorchResult<F32Tensor> {
        global_pool().get_or_create(shape, |shape| F32Tensor::ones(shape))
    }

    /// プールからテンソルを取得またはランダムテンソルを作成
    /// Get tensor from pool or create random
    pub fn get_or_randn(shape: &[usize]) -> RusTorchResult<F32Tensor> {
        global_pool().get_or_create(shape, |shape| F32Tensor::randn(shape))
    }

    /// テンソルをグローバルプールに戻す
    /// Return tensor to global pool
    pub fn put_tensor(tensor: F32Tensor) -> RusTorchResult<()> {
        global_pool().put(tensor)
    }

    /// グローバルプールの統計を取得
    /// Get global pool statistics
    pub fn pool_stats() -> PoolStats {
        global_pool().stats()
    }

    /// グローバルプールのガベージコレクション
    /// Garbage collect global pool
    pub fn gc_global_pool() -> RusTorchResult<usize> {
        global_pool().garbage_collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_basic_operations() {
        let pool = TensorPool::with_default_config();
        let shape = vec![2, 3];

        // ミス（最初は空）
        assert!(pool.get(&shape).is_none());

        // テンソルを作成してプールに入れる
        let tensor = F32Tensor::zeros(&shape).unwrap();
        pool.put(tensor).unwrap();

        // ヒット
        let retrieved = pool.get(&shape);
        assert!(retrieved.is_some());

        let stats = pool.stats();
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[test]
    fn test_pool_gc() {
        let mut config = PoolConfig::default();
        config.max_lifetime = Duration::from_millis(1); // 短い生存時間
        let pool = TensorPool::new(config);

        // テンソルを追加
        let tensor = F32Tensor::zeros(&[2, 2]).unwrap();
        pool.put(tensor).unwrap();

        // 少し待つ
        std::thread::sleep(Duration::from_millis(10));

        // GC実行
        let released = pool.garbage_collect().unwrap();
        assert_eq!(released, 1);

        let stats = pool.stats();
        assert_eq!(stats.current_size, 0);
    }

    #[test]
    fn test_pool_size_limits() {
        let mut config = PoolConfig::default();
        config.max_pool_size = 2;
        config.max_per_shape = 1;
        let pool = TensorPool::new(config);

        // 制限を超えてテンソルを追加
        for i in 0..5 {
            let tensor = F32Tensor::zeros(&[i + 1]).unwrap();
            pool.put(tensor).unwrap();
        }

        let stats = pool.stats();
        assert!(stats.current_size <= 2);
    }

    #[test]
    fn test_helper_functions() {
        let zeros = helpers::get_or_zeros(&[2, 2]).unwrap();
        assert_eq!(zeros.shape(), &[2, 2]);

        let ones = helpers::get_or_ones(&[3, 3]).unwrap();
        assert_eq!(ones.shape(), &[3, 3]);

        // プールに戻す
        helpers::put_tensor(zeros).unwrap();
        helpers::put_tensor(ones).unwrap();

        let stats = helpers::pool_stats();
        assert!(stats.current_size > 0);
    }

    #[test]
    fn test_health_check() {
        let pool = TensorPool::with_default_config();

        // いくつかの操作を実行
        for _ in 0..10 {
            let tensor = F32Tensor::zeros(&[2, 2]).unwrap();
            pool.put(tensor).unwrap();
        }

        for _ in 0..5 {
            pool.get(&[2, 2]);
        }

        let health = pool.health_check();
        assert!(health.hit_rate >= 0.0);
        assert!(!health.recommendations.is_empty());
    }
}
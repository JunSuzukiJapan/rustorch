use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use lazy_static::lazy_static;

/// Memory pool for efficient tensor allocation and reuse
/// テンソルの効率的な割り当てと再利用のためのメモリプール
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
        if total_elements <= 64 { 0 }
        else if total_elements <= 256 { 1 }
        else if total_elements <= 1024 { 2 }
        else if total_elements <= 4096 { 3 }
        else if total_elements <= 16384 { 4 }
        else if total_elements <= 65536 { 5 }
        else { 6 }
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
                    match cloned_array.into_shape(IxDyn(shape)) {
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
        let mut pool: MemoryPool<f32> = MemoryPool::new(10);
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

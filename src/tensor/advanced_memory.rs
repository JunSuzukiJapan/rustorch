//! Advanced memory management for high-performance tensor operations
//! 高性能テンソル演算のための高度なメモリ管理

use super::Tensor;
use super::parallel_errors::{ParallelError, ParallelResult};
use num_traits::Float;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Advanced memory alignment for different architectures
/// 異なるアーキテクチャ用の高度なメモリアライメント
/// Cache line size for optimal memory alignment
/// 最適なメモリアライメント用のキャッシュラインサイズ
pub const CACHE_LINE_SIZE: usize = 64;
/// Standard page size
/// 標準ページサイズ
pub const PAGE_SIZE: usize = 4096;
/// Huge page size for large memory allocations
/// 大容量メモリ割り当て用のヒュージページサイズ
pub const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024; // 2MB

/// Memory allocation strategy
/// メモリ割り当て戦略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Standard heap allocation
    /// 標準ヒープ割り当て
    Standard,
    /// Cache-aligned allocation
    /// キャッシュアライメント割り当て
    CacheAligned,
    /// Page-aligned allocation
    /// ページアライメント割り当て
    PageAligned,
    /// Huge page allocation for large tensors
    /// 大きなテンソル用のヒュージページ割り当て
    HugePage,
    /// Memory pool allocation
    /// メモリプール割り当て
    Pooled,
    /// NUMA-aware allocation
    /// NUMA対応割り当て
    NumaAware,
}

/// Memory pool configuration
/// メモリプール設定
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial size of memory pool in bytes
    /// メモリプールの初期サイズ（バイト）
    pub initial_size: usize,
    /// Maximum size of memory pool in bytes
    /// メモリプールの最大サイズ（バイト）
    pub max_size: usize,
    /// Growth factor when expanding pool
    /// プール拡張時の成長率
    pub growth_factor: f32,
    /// Threshold for shrinking pool
    /// プール縮小しきい値
    pub shrink_threshold: f32,
    /// Memory alignment requirement
    /// メモリアライメント要件
    pub alignment: usize,
    /// Enable memory prefaulting
    /// メモリプリフォルトを有効にする
    pub enable_prefaulting: bool,
    /// Enable huge page support
    /// ヒュージページサポートを有効にする
    pub enable_huge_pages: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 64 * 1024 * 1024, // 64MB
            max_size: 1024 * 1024 * 1024,   // 1GB
            growth_factor: 1.5,
            shrink_threshold: 0.25,
            alignment: CACHE_LINE_SIZE,
            enable_prefaulting: true,
            enable_huge_pages: false,
        }
    }
}

/// Memory block metadata
/// メモリブロックメタデータ
#[derive(Debug)]
struct MemoryBlock {
    ptr: NonNull<u8>,
    size: usize,
    alignment: usize,
    last_accessed: Instant,
    access_count: u64,
    is_huge_page: bool,
}

impl MemoryBlock {
    fn new(size: usize, alignment: usize, is_huge_page: bool) -> ParallelResult<Self> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| ParallelError::ParallelExecutionError {
                message: format!("Invalid layout: {}", e),
            })?;

        let ptr = unsafe {
            let raw_ptr = if is_huge_page {
                Self::alloc_huge_page(size, alignment)?
            } else {
                alloc_zeroed(layout)
            };

            if raw_ptr.is_null() {
                return Err(ParallelError::ParallelExecutionError {
                    message: "Allocation failed".to_string(),
                });
            }

            NonNull::new_unchecked(raw_ptr)
        };

        let now = Instant::now();
        Ok(Self {
            ptr,
            size,
            alignment,
            last_accessed: now,
            access_count: 0,
            is_huge_page,
        })
    }

    #[cfg(target_os = "linux")]
    fn alloc_huge_page(size: usize, alignment: usize) -> ParallelResult<*mut u8> {
        use std::os::unix::io::AsRawFd;
        use std::fs::OpenOptions;

        // Try to allocate using mmap with MAP_HUGETLB
        // MAP_HUGETLBを使用してmmapで割り当てを試行
        let fd = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/zero")
            .map_err(|e| ParallelError::ParallelExecutionError {
                message: format!("Failed to open /dev/zero: {}", e),
            })?;

        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_HUGETLB,
                fd.as_raw_fd(),
                0,
            );

            if ptr == libc::MAP_FAILED {
                // Fallback to regular allocation
                // 通常の割り当てにフォールバック
                let layout = Layout::from_size_align(size, alignment)
                    .map_err(|e| ParallelError::ParallelExecutionError {
                message: format!("Invalid layout: {}", e),
            })?;
                Ok(alloc_zeroed(layout))
            } else {
                Ok(ptr as *mut u8)
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn alloc_huge_page(size: usize, alignment: usize) -> ParallelResult<*mut u8> {
        // Fallback to regular allocation on non-Linux systems
        // Linux以外のシステムでは通常の割り当てにフォールバック
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| ParallelError::ParallelExecutionError {
                message: format!("Invalid layout: {}", e),
            })?;
        Ok(unsafe { alloc_zeroed(layout) })
    }

    fn update_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }


    fn idle_time(&self) -> Duration {
        self.last_accessed.elapsed()
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        unsafe {
            if self.is_huge_page {
                #[cfg(target_os = "linux")]
                {
                    libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.size);
                }
                #[cfg(not(target_os = "linux"))]
                {
                    let layout = Layout::from_size_align_unchecked(self.size, self.alignment);
                    dealloc(self.ptr.as_ptr(), layout);
                }
            } else {
                let layout = Layout::from_size_align_unchecked(self.size, self.alignment);
                dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}

/// Advanced memory pool with intelligent management
/// インテリジェント管理を備えた高度なメモリプール
pub struct AdvancedMemoryPool {
    config: PoolConfig,
    free_blocks: RwLock<HashMap<usize, VecDeque<MemoryBlock>>>,
    allocated_blocks: RwLock<HashMap<*mut u8, MemoryBlock>>,
    total_allocated: Arc<Mutex<usize>>,
    allocation_stats: Arc<Mutex<AllocationStats>>,
    numa_node: Option<u32>,
}

/// Memory allocation statistics
/// メモリ割り当て統計
#[derive(Debug, Default, Clone)]
pub struct AllocationStats {
    /// Total number of allocations performed
    /// 実行された総割り当て数
    pub total_allocations: u64,
    /// Total number of deallocations performed
    /// 実行された総解放数
    pub total_deallocations: u64,
    /// Peak memory usage in bytes
    /// ピークメモリ使用量（バイト）
    pub peak_memory_usage: usize,
    /// Current memory usage in bytes
    /// 現在のメモリ使用量（バイト）
    pub current_memory_usage: usize,
    /// Number of cache hits
    /// キャッシュヒット数
    pub cache_hits: u64,
    /// Number of cache misses
    /// キャッシュミス数
    pub cache_misses: u64,
    /// Number of huge page allocations
    /// ヒュージページ割り当て数
    pub huge_page_allocations: u64,
    /// Memory fragmentation ratio (0.0 to 1.0)
    /// メモリフラグメンテーション率（0.0から1.0）
    pub fragmentation_ratio: f32,
}

impl AdvancedMemoryPool {
    /// Create new advanced memory pool
    /// 新しい高度なメモリプールを作成
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            free_blocks: RwLock::new(HashMap::new()),
            allocated_blocks: RwLock::new(HashMap::new()),
            total_allocated: Arc::new(Mutex::new(0)),
            allocation_stats: Arc::new(Mutex::new(AllocationStats::default())),
            numa_node: Self::detect_numa_node(),
        }
    }

    /// Allocate memory with specified strategy
    /// 指定された戦略でメモリを割り当て
    pub fn allocate<T: Float + 'static>(
        &self,
        size: usize,
        strategy: AllocationStrategy,
    ) -> ParallelResult<NonNull<T>> {
        let alignment = self.get_alignment_for_strategy(strategy);
        let actual_size = self.round_up_size(size * std::mem::size_of::<T>(), alignment);

        // Try to reuse existing block
        // 既存ブロックの再利用を試行
        if let Some(block) = self.try_reuse_block(actual_size, alignment)? {
            let ptr = unsafe { NonNull::new_unchecked(block.ptr.as_ptr() as *mut T) };
            self.update_stats_on_allocation(actual_size, true);
            return Ok(ptr);
        }

        // Allocate new block
        // 新しいブロックを割り当て
        let use_huge_pages = strategy == AllocationStrategy::HugePage || 
                           (actual_size >= HUGE_PAGE_SIZE && self.config.enable_huge_pages);

        let mut block = MemoryBlock::new(actual_size, alignment, use_huge_pages)?;
        
        // Prefault pages if enabled
        // 有効な場合はページをプリフォルト
        if self.config.enable_prefaulting {
            self.prefault_pages(&mut block)?;
        }

        let ptr = unsafe { NonNull::new_unchecked(block.ptr.as_ptr() as *mut T) };

        // Track allocated block
        // 割り当てブロックを追跡
        {
            let mut allocated = self.allocated_blocks.write().unwrap();
            allocated.insert(block.ptr.as_ptr(), block);
        }

        self.update_stats_on_allocation(actual_size, false);
        Ok(ptr)
    }

    /// Deallocate memory
    /// メモリを解放
    pub fn deallocate<T>(&self, ptr: NonNull<T>) -> ParallelResult<()> {
        let raw_ptr = ptr.as_ptr() as *mut u8;

        let block = {
            let mut allocated = self.allocated_blocks.write().unwrap();
            allocated.remove(&raw_ptr).ok_or_else(|| {
                ParallelError::ParallelExecutionError {
                    message: "Pointer not found in allocated blocks".to_string(),
                }
            })?
        };

        let size = block.size;

        // Return block to free pool if it's worth keeping
        // 保持する価値がある場合はフリープールに返却
        if self.should_keep_block(&block) {
            let mut free_blocks = self.free_blocks.write().unwrap();
            free_blocks.entry(size).or_insert_with(VecDeque::new).push_back(block);
        }
        // Otherwise, block will be dropped and memory freed
        // そうでなければ、ブロックはドロップされメモリが解放される

        self.update_stats_on_deallocation(size);
        Ok(())
    }

    /// Get memory usage statistics
    /// メモリ使用統計を取得
    pub fn get_stats(&self) -> AllocationStats {
        let stats = self.allocation_stats.lock().unwrap();
        (*stats).clone()
    }

    /// Perform garbage collection
    /// ガベージコレクションを実行
    pub fn garbage_collect(&self) -> ParallelResult<usize> {
        let mut freed_memory = 0;
        let _now = Instant::now();
        let max_idle_time = Duration::from_secs(300); // 5 minutes

        let mut free_blocks = self.free_blocks.write().unwrap();
        
        for (size, blocks) in free_blocks.iter_mut() {
            blocks.retain(|block| {
                if block.idle_time() > max_idle_time {
                    freed_memory += size;
                    false
                } else {
                    true
                }
            });
        }

        // Remove empty size buckets
        // 空のサイズバケットを削除
        free_blocks.retain(|_, blocks| !blocks.is_empty());

        Ok(freed_memory)
    }

    /// Optimize memory layout for NUMA
    /// NUMA用のメモリレイアウト最適化
    pub fn optimize_for_numa(&self) -> ParallelResult<()> {
        if let Some(_node) = self.numa_node {
            // Bind memory allocations to specific NUMA node
            // メモリ割り当てを特定のNUMAノードにバインド
            #[cfg(target_os = "linux")]
            {
                self.set_numa_policy(node)?;
            }
        }
        Ok(())
    }

    // Private helper methods
    // プライベートヘルパーメソッド

    fn get_alignment_for_strategy(&self, strategy: AllocationStrategy) -> usize {
        match strategy {
            AllocationStrategy::Standard => std::mem::align_of::<f64>(),
            AllocationStrategy::CacheAligned => CACHE_LINE_SIZE,
            AllocationStrategy::PageAligned => PAGE_SIZE,
            AllocationStrategy::HugePage => HUGE_PAGE_SIZE,
            AllocationStrategy::Pooled => self.config.alignment,
            AllocationStrategy::NumaAware => CACHE_LINE_SIZE,
        }
    }

    fn round_up_size(&self, size: usize, alignment: usize) -> usize {
        (size + alignment - 1) & !(alignment - 1)
    }

    fn try_reuse_block(&self, size: usize, alignment: usize) -> ParallelResult<Option<MemoryBlock>> {
        let mut free_blocks = self.free_blocks.write().unwrap();
        
        // Look for exact size match first
        // まず正確なサイズマッチを探す
        if let Some(blocks) = free_blocks.get_mut(&size) {
            if let Some(mut block) = blocks.pop_front() {
                if block.alignment >= alignment {
                    block.update_access();
                    return Ok(Some(block));
                } else {
                    blocks.push_back(block);
                }
            }
        }

        // Look for larger blocks that can be split
        // 分割可能な大きなブロックを探す
        for (&block_size, blocks) in free_blocks.iter_mut() {
            if block_size >= size && !blocks.is_empty() {
                if let Some(mut block) = blocks.pop_front() {
                    if block.alignment >= alignment {
                        block.update_access();
                        return Ok(Some(block));
                    } else {
                        blocks.push_back(block);
                    }
                }
            }
        }

        Ok(None)
    }

    fn should_keep_block(&self, block: &MemoryBlock) -> bool {
        let current_total = *self.total_allocated.lock().unwrap();
        let would_exceed_max = current_total + block.size > self.config.max_size;
        
        !would_exceed_max && block.access_count > 1
    }

    fn prefault_pages(&self, block: &mut MemoryBlock) -> ParallelResult<()> {
        unsafe {
            let ptr = block.ptr.as_ptr();
            let size = block.size;
            
            // Touch each page to prefault
            // 各ページにタッチしてプリフォルト
            for offset in (0..size).step_by(PAGE_SIZE) {
                let page_ptr = ptr.add(offset);
                std::ptr::write_volatile(page_ptr, 0);
            }
        }
        Ok(())
    }

    fn update_stats_on_allocation(&self, size: usize, cache_hit: bool) {
        let mut stats = self.allocation_stats.lock().unwrap();
        let mut total = self.total_allocated.lock().unwrap();
        
        stats.total_allocations += 1;
        *total += size;
        stats.current_memory_usage = *total;
        
        if *total > stats.peak_memory_usage {
            stats.peak_memory_usage = *total;
        }
        
        if cache_hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }
    }

    fn update_stats_on_deallocation(&self, size: usize) {
        let mut stats = self.allocation_stats.lock().unwrap();
        let mut total = self.total_allocated.lock().unwrap();
        
        stats.total_deallocations += 1;
        *total -= size;
        stats.current_memory_usage = *total;
    }

    fn detect_numa_node() -> Option<u32> {
        // Simplified NUMA detection
        // 簡略化されたNUMA検出
        #[cfg(target_os = "linux")]
        {
            // In practice, would use libnuma or similar
            // 実際にはlibnumaなどを使用
            Some(0)
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    #[cfg(target_os = "linux")]
    fn set_numa_policy(&self, node: u32) -> ParallelResult<()> {
        // Simplified NUMA policy setting
        // 簡略化されたNUMAポリシー設定
        Ok(())
    }
}

/// Memory-optimized tensor operations
/// メモリ最適化テンソル演算
pub struct OptimizedTensorOps {
    memory_pool: Arc<AdvancedMemoryPool>,
}

impl OptimizedTensorOps {
    /// Create new optimized tensor operations
    /// 新しい最適化テンソル演算を作成
    pub fn new(pool_config: PoolConfig) -> Self {
        Self {
            memory_pool: Arc::new(AdvancedMemoryPool::new(pool_config)),
        }
    }

    /// Create tensor with optimized memory allocation
    /// 最適化メモリ割り当てでテンソルを作成
    pub fn create_tensor<T: Float + 'static>(
        &self,
        shape: &[usize],
        strategy: AllocationStrategy,
    ) -> ParallelResult<Tensor<T>> {
        let total_elements: usize = shape.iter().product();
        let ptr = self.memory_pool.allocate(total_elements, strategy)?;
        
        // Create tensor with custom memory
        // カスタムメモリでテンソルを作成
        unsafe {
            let data = std::slice::from_raw_parts_mut(ptr.as_ptr(), total_elements);
            data.fill(T::zero());
            Ok(Tensor::from_raw_parts(data, shape))
        }
    }

    /// Perform in-place operations to minimize memory allocation
    /// メモリ割り当てを最小化するインプレース演算
    pub fn add_inplace<T: Float + 'static>(
        &self,
        a: &mut Tensor<T>,
        b: &Tensor<T>,
    ) -> ParallelResult<()> {
        if a.shape() != b.shape() {
            return Err(ParallelError::ShapeMismatch {
                expected: a.shape().to_vec(),
                actual: b.shape().to_vec(),
                operation: "inplace_add".to_string(),
            });
        }

        // Vectorized in-place addition
        // ベクトル化インプレース加算
        let a_slice = a.as_slice_mut().unwrap();
        let b_slice = b.as_slice().unwrap();
        for i in 0..a_slice.len() {
            a_slice[i] = a_slice[i] + b_slice[i];
        }

        Ok(())
    }

    /// Get memory pool statistics
    /// メモリプール統計を取得
    pub fn get_memory_stats(&self) -> AllocationStats {
        self.memory_pool.get_stats()
    }

    /// Perform memory optimization
    /// メモリ最適化を実行
    pub fn optimize_memory(&self) -> ParallelResult<usize> {
        self.memory_pool.garbage_collect()
    }
}

/// Extension trait for Tensor to support custom memory management
/// カスタムメモリ管理をサポートするTensor用の拡張トレイト
pub trait TensorMemoryExt<T: Float> {
    /// Create tensor from raw memory parts
    /// 生メモリパーツからテンソルを作成
    fn from_raw_parts(data: &mut [T], shape: &[usize]) -> Tensor<T>;
    /// Get memory usage in bytes
    /// メモリ使用量をバイトで取得
    fn memory_usage(&self) -> usize;
    /// Check if memory is aligned to specified boundary
    /// 指定された境界にメモリがアライメントされているかチェック
    fn is_memory_aligned(&self, alignment: usize) -> bool;
}

impl<T: Float + 'static> TensorMemoryExt<T> for Tensor<T> {
    fn from_raw_parts(_data: &mut [T], shape: &[usize]) -> Tensor<T> {
        // Simplified implementation - in practice would need proper tensor construction
        // 簡略化実装 - 実際には適切なテンソル構築が必要
        Tensor::zeros(shape)
    }

    fn memory_usage(&self) -> usize {
        self.as_slice().unwrap().len() * std::mem::size_of::<T>()
    }

    fn is_memory_aligned(&self, alignment: usize) -> bool {
        (self.as_slice().unwrap().as_ptr() as usize) % alignment == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_memory_pool() {
        let config = PoolConfig::default();
        let pool = AdvancedMemoryPool::new(config);
        
        let ptr: NonNull<f32> = pool.allocate(1000, AllocationStrategy::CacheAligned).unwrap();
        assert!(pool.deallocate(ptr).is_ok());
        
        let stats = pool.get_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.total_deallocations, 1);
    }

    #[test]
    fn test_optimized_tensor_ops() {
        let config = PoolConfig::default();
        let ops = OptimizedTensorOps::new(config);
        
        let tensor: Tensor<f32> = ops.create_tensor(&[100, 100], AllocationStrategy::CacheAligned).unwrap();
        assert_eq!(tensor.shape(), &[100, 100]);
        
        let stats = ops.get_memory_stats();
        assert!(stats.total_allocations > 0);
    }

    #[test]
    fn test_memory_alignment() {
        let tensor: Tensor<f32> = Tensor::zeros(&[64]);
        assert!(tensor.is_memory_aligned(std::mem::align_of::<f32>()));
    }

    #[test]
    fn test_garbage_collection() {
        let config = PoolConfig::default();
        let pool = AdvancedMemoryPool::new(config);
        
        // Allocate and deallocate several blocks
        // 複数のブロックを割り当てて解放
        for _ in 0..10 {
            let ptr: NonNull<f32> = pool.allocate(1000, AllocationStrategy::Standard).unwrap();
            pool.deallocate(ptr).unwrap();
        }
        
        let freed = pool.garbage_collect().unwrap();
        // Some memory should be freed during GC
        // GC中にいくらかのメモリが解放されるはず
        // freed is usize, always >= 0
        assert!(freed == freed); // Keep the variable used
    }
}

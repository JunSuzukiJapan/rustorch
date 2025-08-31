//! Optimized memory pool management for WASM operations
//! WASM操作用最適化メモリプール管理

use std::collections::VecDeque;
use std::sync::Mutex;

/// Memory pool for efficient buffer reuse with performance tracking
pub struct WasmTensorPool {
    small_buffers: VecDeque<Vec<f32>>,  // < 256 elements (1KB)
    medium_buffers: VecDeque<Vec<f32>>, // 256 - 262144 elements (1MB)
    large_buffers: VecDeque<Vec<f32>>,  // > 262144 elements (1MB+)
    max_pool_size: usize,
    total_allocations: usize,
    cache_hits: usize,
    memory_saved_bytes: usize,
}

impl WasmTensorPool {
    /// Create new memory pool with specified capacity
    pub fn with_capacity(max_pool_size: usize) -> Self {
        Self {
            small_buffers: VecDeque::new(),
            medium_buffers: VecDeque::new(),
            large_buffers: VecDeque::new(),
            max_pool_size,
            total_allocations: 0,
            cache_hits: 0,
            memory_saved_bytes: 0,
        }
    }

    /// Get buffer from pool or create new one with optimization tracking
    pub fn get_buffer(&mut self, size: usize) -> Vec<f32> {
        self.total_allocations += 1;

        let pool = self.select_pool_mut(size);
        if let Some(mut buf) = pool.pop_front() {
            self.cache_hits += 1;
            self.memory_saved_bytes += buf.capacity() * std::mem::size_of::<f32>();
            buf.clear();
            buf.reserve(size.saturating_sub(buf.capacity()));
            buf
        } else {
            Vec::with_capacity(size)
        }
    }

    /// Return buffer to pool for reuse
    pub fn return_buffer(&mut self, mut buffer: Vec<f32>) {
        if buffer.capacity() == 0 {
            return;
        }

        buffer.clear();
        let capacity = buffer.capacity();
        let max_size = self.max_pool_size;
        let pool = self.select_pool_mut(capacity);

        if pool.len() < max_size {
            pool.push_back(buffer);
        }
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.small_buffers.clear();
        self.medium_buffers.clear();
        self.large_buffers.clear();
    }

    /// Get pool statistics with performance metrics
    pub fn stats(&self) -> String {
        let hit_rate = if self.total_allocations > 0 {
            (self.cache_hits as f32 / self.total_allocations as f32) * 100.0
        } else {
            0.0
        };

        format!(
            "{{\"small\":{},\"medium\":{},\"large\":{},\"max_size\":{},\"total_allocations\":{},\"cache_hits\":{},\"hit_rate\":{:.2},\"memory_saved_bytes\": {}}}",
            self.small_buffers.len(),
            self.medium_buffers.len(),
            self.large_buffers.len(),
            self.max_pool_size,
            self.total_allocations,
            self.cache_hits,
            hit_rate,
            self.memory_saved_bytes
        )
    }

    /// Get cache hit rate percentage
    pub fn hit_rate(&self) -> f32 {
        if self.total_allocations > 0 {
            (self.cache_hits as f32 / self.total_allocations as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Reset performance counters
    pub fn reset_counters(&mut self) {
        self.total_allocations = 0;
        self.cache_hits = 0;
        self.memory_saved_bytes = 0;
    }

    /// Force garbage collection of unused buffers
    pub fn gc(&mut self) {
        // Remove buffers that are too large for their pools
        self.small_buffers.retain(|buf| buf.capacity() < 512);
        self.medium_buffers.retain(|buf| buf.capacity() < 524288);

        // Limit pool sizes more aggressively
        let aggressive_limit = self.max_pool_size / 2;
        while self.small_buffers.len() > aggressive_limit {
            self.small_buffers.pop_back();
        }
        while self.medium_buffers.len() > aggressive_limit {
            self.medium_buffers.pop_back();
        }
        while self.large_buffers.len() > aggressive_limit {
            self.large_buffers.pop_back();
        }
    }

    fn select_pool_mut(&mut self, size: usize) -> &mut VecDeque<Vec<f32>> {
        if size < 256 {
            // < 1KB (optimized threshold)
            &mut self.small_buffers
        } else if size < 262144 {
            // < 1MB
            &mut self.medium_buffers
        } else {
            &mut self.large_buffers
        }
    }
}

/// Pool usage statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub small_count: usize,
    pub medium_count: usize,
    pub large_count: usize,
    pub max_size: usize,
    pub hit_rate: f32,
    pub memory_saved_mb: f32,
}

/// Global memory pool instance
static GLOBAL_POOL: Mutex<Option<WasmTensorPool>> = Mutex::new(None);

/// Memory pool utilities
pub struct MemoryManager;

impl MemoryManager {
    /// Initialize global memory pool
    pub fn init_pool(max_size: usize) {
        let mut pool = GLOBAL_POOL.lock().unwrap();
        *pool = Some(WasmTensorPool::with_capacity(max_size));
    }

    /// Get buffer from global pool with optimization
    pub fn get_buffer(size: usize) -> Vec<f32> {
        let mut pool = GLOBAL_POOL.lock().unwrap();
        match pool.as_mut() {
            Some(p) => p.get_buffer(size),
            None => {
                // Auto-initialize with default settings if not initialized
                drop(pool);
                Self::init_pool(100); // Default pool size
                let mut pool = GLOBAL_POOL.lock().unwrap();
                pool.as_mut().unwrap().get_buffer(size)
            }
        }
    }

    /// Return buffer to global pool
    pub fn return_buffer(buffer: Vec<f32>) {
        let mut pool = GLOBAL_POOL.lock().unwrap();
        if let Some(p) = pool.as_mut() {
            p.return_buffer(buffer);
        }
    }

    /// Get global pool performance statistics
    pub fn get_stats() -> String {
        let pool = GLOBAL_POOL.lock().unwrap();
        match pool.as_ref() {
            Some(p) => p.stats(),
            None => "{\"status\":\"uninitialized\"}".to_string(),
        }
    }

    /// Force garbage collection on global pool
    pub fn gc() {
        let mut pool = GLOBAL_POOL.lock().unwrap();
        if let Some(p) = pool.as_mut() {
            p.gc();
        }
    }

    /// Get cache efficiency metrics
    pub fn cache_efficiency() -> String {
        let pool = GLOBAL_POOL.lock().unwrap();
        match pool.as_ref() {
            Some(p) => {
                let hit_rate = p.hit_rate();
                format!(
                    "{{\"hit_rate\":{:.2},\"efficiency\":\"{}\"}}",
                    hit_rate,
                    if hit_rate > 80.0 {
                        "excellent"
                    } else if hit_rate > 60.0 {
                        "good"
                    } else if hit_rate > 40.0 {
                        "fair"
                    } else {
                        "poor"
                    }
                )
            }
            None => "{\"status\":\"uninitialized\"}".to_string(),
        }
    }

    /// Get pool statistics as struct
    pub fn pool_stats() -> Option<PoolStats> {
        let pool = GLOBAL_POOL.lock().unwrap();
        pool.as_ref().map(|p| PoolStats {
            small_count: p.small_buffers.len(),
            medium_count: p.medium_buffers.len(),
            large_count: p.large_buffers.len(),
            max_size: p.max_pool_size,
            hit_rate: p.hit_rate(),
            memory_saved_mb: (p.memory_saved_bytes as f32) / (1024.0 * 1024.0),
        })
    }
}

/// Buffer type for pooled allocations
pub type PooledBuffer = Vec<f32>;

//! Memory pool management for WASM operations
//! WASM操作用メモリプール管理

use std::collections::VecDeque;
use std::sync::Mutex;

/// Memory pool for efficient buffer reuse
pub struct WasmTensorPool {
    small_buffers: VecDeque<Vec<f32>>,  // < 1KB
    medium_buffers: VecDeque<Vec<f32>>, // 1KB - 1MB  
    large_buffers: VecDeque<Vec<f32>>,  // > 1MB
    max_pool_size: usize,
}

impl WasmTensorPool {
    /// Create new memory pool with specified capacity
    pub fn with_capacity(max_pool_size: usize) -> Self {
        Self {
            small_buffers: VecDeque::new(),
            medium_buffers: VecDeque::new(), 
            large_buffers: VecDeque::new(),
            max_pool_size,
        }
    }
    
    /// Get buffer from pool or create new one
    pub fn get_buffer(&mut self, size: usize) -> Vec<f32> {
        let pool = self.select_pool_mut(size);
        pool.pop_front()
            .map(|mut buf| {
                buf.clear();
                buf.reserve(size.saturating_sub(buf.capacity()));
                buf
            })
            .unwrap_or_else(|| Vec::with_capacity(size))
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
    
    /// Get statistics about pool usage
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            small_count: self.small_buffers.len(),
            medium_count: self.medium_buffers.len(),
            large_count: self.large_buffers.len(),
            max_size: self.max_pool_size,
        }
    }
    
    /// Clear all pools
    pub fn clear(&mut self) {
        self.small_buffers.clear();
        self.medium_buffers.clear();
        self.large_buffers.clear();
    }
    
    fn select_pool_mut(&mut self, size: usize) -> &mut VecDeque<Vec<f32>> {
        match size {
            0..=256 => &mut self.small_buffers,      // < 1KB
            257..=262144 => &mut self.medium_buffers, // 1KB - 1MB
            _ => &mut self.large_buffers,             // > 1MB
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
    
    /// Get buffer from global pool
    pub fn get_buffer(size: usize) -> Vec<f32> {
        let mut pool = GLOBAL_POOL.lock().unwrap();
        match pool.as_mut() {
            Some(p) => p.get_buffer(size),
            None => Vec::with_capacity(size),
        }
    }
    
    /// Return buffer to global pool
    pub fn return_buffer(buffer: Vec<f32>) {
        let mut pool = GLOBAL_POOL.lock().unwrap();
        if let Some(p) = pool.as_mut() {
            p.return_buffer(buffer);
        }
    }
    
    /// Get pool statistics
    pub fn pool_stats() -> Option<PoolStats> {
        let pool = GLOBAL_POOL.lock().unwrap();
        pool.as_ref().map(|p| p.stats())
    }
    
    /// Clear global pool
    pub fn clear_pool() {
        let mut pool = GLOBAL_POOL.lock().unwrap();
        if let Some(p) = pool.as_mut() {
            p.clear();
        }
    }
}

/// RAII wrapper for automatic buffer management
pub struct PooledBuffer {
    buffer: Option<Vec<f32>>,
}

impl PooledBuffer {
    /// Create new pooled buffer
    pub fn new(size: usize) -> Self {
        Self {
            buffer: Some(MemoryManager::get_buffer(size)),
        }
    }
    
    /// Get mutable reference to buffer
    pub fn as_mut(&mut self) -> &mut Vec<f32> {
        self.buffer.as_mut().unwrap()
    }
    
    /// Get immutable reference to buffer
    pub fn as_ref(&self) -> &Vec<f32> {
        self.buffer.as_ref().unwrap()
    }
    
    /// Take ownership of buffer (disables auto-return)
    pub fn take(mut self) -> Vec<f32> {
        self.buffer.take().unwrap()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            MemoryManager::return_buffer(buffer);
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = Vec<f32>;
    
    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().unwrap()
    }
}
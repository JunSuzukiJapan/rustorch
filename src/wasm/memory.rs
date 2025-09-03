//! WASM memory management utilities
//! WASMメモリ管理ユーティリティ

#[cfg(feature = "wasm")]
use std::collections::HashMap;
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Memory pool for WASM tensor operations
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmTensorPool {
    pool: Vec<Vec<f32>>,
    allocated: HashMap<usize, bool>,
    total_size: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmTensorPool {
    /// Create new memory pool with specified capacity
    #[wasm_bindgen(constructor)]
    pub fn new(capacity_bytes: usize) -> Self {
        let capacity_elements = capacity_bytes / std::mem::size_of::<f32>();
        Self {
            pool: Vec::new(),
            allocated: HashMap::new(),
            total_size: capacity_elements,
        }
    }

    /// Allocate memory block
    #[wasm_bindgen]
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        // Simple allocation strategy - find first available slot
        for (i, buffer) in self.pool.iter().enumerate() {
            if !*self.allocated.get(&i).unwrap_or(&true) && buffer.len() >= size {
                self.allocated.insert(i, true);
                return Some(i);
            }
        }

        // No suitable slot found, create new if within capacity
        if self.get_total_allocated() + size <= self.total_size {
            let buffer = vec![0.0f32; size];
            let index = self.pool.len();
            self.pool.push(buffer);
            self.allocated.insert(index, true);
            Some(index)
        } else {
            None
        }
    }

    /// Deallocate memory block
    #[wasm_bindgen]
    pub fn deallocate(&mut self, index: usize) -> bool {
        if index < self.pool.len() {
            self.allocated.insert(index, false);
            true
        } else {
            false
        }
    }

    /// Get total allocated memory in elements
    #[wasm_bindgen]
    pub fn get_total_allocated(&self) -> usize {
        self.allocated
            .iter()
            .filter(|(_, &is_allocated)| is_allocated)
            .map(|(&index, _)| self.pool.get(index).map_or(0, |buf| buf.len()))
            .sum()
    }

    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn get_usage_stats(&self) -> js_sys::Object {
        let stats = js_sys::Object::new();

        let total_allocated = self.get_total_allocated();
        let usage_percent = (total_allocated as f64 / self.total_size as f64) * 100.0;

        js_sys::Reflect::set(
            &stats,
            &"totalAllocated".into(),
            &(total_allocated as u32).into(),
        )
        .unwrap();

        js_sys::Reflect::set(
            &stats,
            &"totalCapacity".into(),
            &(self.total_size as u32).into(),
        )
        .unwrap();

        js_sys::Reflect::set(&stats, &"usagePercent".into(), &usage_percent.into()).unwrap();

        js_sys::Reflect::set(&stats, &"poolSize".into(), &(self.pool.len() as u32).into()).unwrap();

        stats
    }

    /// Force garbage collection of unused blocks
    #[wasm_bindgen]
    pub fn garbage_collect(&mut self) -> usize {
        let mut removed_count = 0;
        let mut new_pool = Vec::new();
        let mut new_allocated = HashMap::new();

        for (old_index, buffer) in self.pool.iter().enumerate() {
            if *self.allocated.get(&old_index).unwrap_or(&false) {
                let new_index = new_pool.len();
                new_pool.push(buffer.clone());
                new_allocated.insert(new_index, true);
            } else {
                removed_count += 1;
            }
        }

        self.pool = new_pool;
        self.allocated = new_allocated;
        removed_count
    }

    /// Clear all allocated memory
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.pool.clear();
        self.allocated.clear();
    }
}

/// Memory-aware tensor buffer for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmTensorBuffer {
    data: Vec<f32>,
    shape: Vec<usize>,
    memory_id: Option<usize>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmTensorBuffer {
    /// Create new tensor buffer
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            memory_id: None,
        }
    }

    /// Create tensor buffer from memory pool
    #[wasm_bindgen]
    pub fn from_pool(pool: &mut WasmTensorPool, shape: Vec<usize>) -> Option<WasmTensorBuffer> {
        let size: usize = shape.iter().product();
        if let Some(memory_id) = pool.allocate(size) {
            let data = vec![0.0f32; size];
            Some(WasmTensorBuffer {
                data,
                shape,
                memory_id: Some(memory_id),
            })
        } else {
            None
        }
    }

    /// Get buffer data
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }

    /// Get buffer shape
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get memory ID if allocated from pool
    #[wasm_bindgen(getter)]
    pub fn memory_id(&self) -> Option<usize> {
        self.memory_id
    }

    /// Get buffer size in bytes
    #[wasm_bindgen]
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Release buffer back to pool
    #[wasm_bindgen]
    pub fn release_to_pool(&mut self, pool: &mut WasmTensorPool) -> bool {
        if let Some(memory_id) = self.memory_id {
            self.data.clear();
            self.memory_id = None;
            pool.deallocate(memory_id)
        } else {
            false
        }
    }
}

/// Memory usage monitor for WASM
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmMemoryMonitor {
    peak_usage: usize,
    current_usage: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmMemoryMonitor {
    /// Create a new memory usage monitor
    /// 新しいメモリ使用量モニターを作成
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
        }
    }

    /// Record memory allocation
    #[wasm_bindgen]
    pub fn record_allocation(&mut self, size: usize) {
        self.current_usage += size;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    /// Record memory deallocation
    #[wasm_bindgen]
    pub fn record_deallocation(&mut self, size: usize) {
        if self.current_usage >= size {
            self.current_usage -= size;
        }
    }

    /// Get current memory usage
    #[wasm_bindgen]
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get peak memory usage
    #[wasm_bindgen]
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Reset statistics
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.peak_usage = 0;
        self.current_usage = 0;
    }
}

#[cfg(test)]
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let mut pool = WasmTensorPool::new(1024 * 1024); // 1MB pool

        // Test allocation
        let alloc1 = pool.allocate(100);
        assert!(alloc1.is_some());

        let alloc2 = pool.allocate(200);
        assert!(alloc2.is_some());

        // Test deallocation
        assert!(pool.deallocate(alloc1.unwrap()));

        // Test stats
        let stats = pool.get_usage_stats();
        assert!(js_sys::Reflect::has(&stats, &"totalAllocated".into()).unwrap());
    }

    #[test]
    fn test_tensor_buffer() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let buffer = WasmTensorBuffer::new(data.clone(), shape.clone());

        assert_eq!(buffer.data(), data);
        assert_eq!(buffer.shape(), shape);
        assert_eq!(buffer.size_bytes(), 16); // 4 * 4 bytes
    }

    #[test]
    fn test_memory_monitor() {
        let mut monitor = WasmMemoryMonitor::new();

        monitor.record_allocation(100);
        assert_eq!(monitor.current_usage(), 100);
        assert_eq!(monitor.peak_usage(), 100);

        monitor.record_allocation(50);
        assert_eq!(monitor.current_usage(), 150);
        assert_eq!(monitor.peak_usage(), 150);

        monitor.record_deallocation(75);
        assert_eq!(monitor.current_usage(), 75);
        assert_eq!(monitor.peak_usage(), 150); // Peak should remain
    }
}

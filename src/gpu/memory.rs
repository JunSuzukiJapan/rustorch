/// GPU memory management
/// GPUメモリ管理
use super::DeviceType;
use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// GPU memory allocation information
/// GPUメモリ割り当て情報
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Device where memory is allocated
    /// メモリが割り当てられたデバイス
    pub device: DeviceType,
    /// Size in bytes
    /// サイズ（バイト）
    pub size: usize,
    /// Memory pointer (platform-specific)
    /// メモリポインタ（プラットフォーム固有）
    pub ptr: usize,
    /// Allocation timestamp
    /// 割り当てタイムスタンプ
    pub timestamp: std::time::Instant,
}

/// GPU memory pool for efficient allocation
/// 効率的な割り当てのためのGPUメモリプール
pub struct GpuMemoryPool {
    device: DeviceType,
    total_size: usize,
    allocated_size: usize,
    free_blocks: Vec<(usize, usize)>, // (offset, size)
    allocations: HashMap<usize, MemoryAllocation>,
    base_ptr: usize,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
    /// 新しいGPUメモリプールを作成
    pub fn new(device: DeviceType, size: usize) -> RusTorchResult<Self> {
        let base_ptr = match device {
            DeviceType::Cpu => {
                // For CPU, we can use regular allocation
                let layout = std::alloc::Layout::from_size_align(size, 64)
                    .map_err(|_| RusTorchError::tensor_op("Invalid memory layout"))?;
                unsafe { std::alloc::alloc(layout) as usize }
            }
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    // CUDA memory allocation would go here
                    // cudaMalloc equivalent
                    0 // Placeholder
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(RusTorchError::gpu("CUDA not supported"));
                }
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    // Metal buffer allocation would go here
                    0 // Placeholder
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(RusTorchError::gpu("Metal not supported"));
                }
            }
            DeviceType::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    // OpenCL buffer allocation would go here
                    0 // Placeholder
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err(RusTorchError::gpu("OpenCL not supported"));
                }
            }
        };

        if base_ptr == 0 && !matches!(device, DeviceType::Cpu) {
            return Err(RusTorchError::tensor_op("Failed to allocate GPU memory"));
        }

        Ok(GpuMemoryPool {
            device,
            total_size: size,
            allocated_size: 0,
            free_blocks: vec![(0, size)],
            allocations: HashMap::new(),
            base_ptr,
        })
    }

    /// Allocate memory from the pool
    /// プールからメモリを割り当て
    pub fn allocate(&mut self, size: usize) -> RusTorchResult<MemoryAllocation> {
        // Align size to 256 bytes for GPU efficiency
        let aligned_size = (size + 255) & !255;

        // Find a suitable free block
        let mut best_block_idx = None;
        let mut best_block_size = usize::MAX;

        for (idx, &(_offset, block_size)) in self.free_blocks.iter().enumerate() {
            if block_size >= aligned_size && block_size < best_block_size {
                best_block_idx = Some(idx);
                best_block_size = block_size;
            }
        }

        let block_idx = best_block_idx
            .ok_or_else(|| RusTorchError::tensor_op("No suitable free block found"))?;

        let (offset, block_size) = self.free_blocks[block_idx];
        self.free_blocks.remove(block_idx);

        // If the block is larger than needed, split it
        if block_size > aligned_size {
            let remaining_offset = offset + aligned_size;
            let remaining_size = block_size - aligned_size;
            self.free_blocks.push((remaining_offset, remaining_size));
        }

        let ptr = self.base_ptr + offset;
        let allocation = MemoryAllocation {
            device: self.device,
            size: aligned_size,
            ptr,
            timestamp: std::time::Instant::now(),
        };

        self.allocations.insert(ptr, allocation.clone());
        self.allocated_size += aligned_size;

        Ok(allocation)
    }

    /// Deallocate memory back to the pool
    /// メモリをプールに戻す
    pub fn deallocate(&mut self, ptr: usize) -> RusTorchResult<()> {
        let allocation = self
            .allocations
            .remove(&ptr)
            .ok_or_else(|| RusTorchError::tensor_op("Invalid pointer for deallocation"))?;

        let offset = ptr - self.base_ptr;
        let size = allocation.size;

        // Add the block back to free blocks
        self.free_blocks.push((offset, size));
        self.allocated_size -= size;

        // Merge adjacent free blocks
        self.merge_free_blocks();

        Ok(())
    }

    /// Get memory usage statistics
    /// メモリ使用量統計を取得
    pub fn memory_stats(&self) -> (usize, usize, usize, f32) {
        let free_size = self.total_size - self.allocated_size;
        let usage_percent = (self.allocated_size as f32 / self.total_size as f32) * 100.0;
        (
            self.total_size,
            self.allocated_size,
            free_size,
            usage_percent,
        )
    }

    /// Get device
    /// デバイスを取得
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Merge adjacent free blocks
    /// 隣接する空きブロックをマージ
    fn merge_free_blocks(&mut self) {
        if self.free_blocks.len() <= 1 {
            return;
        }

        // Sort free blocks by offset
        self.free_blocks.sort_by_key(|&(offset, _)| offset);

        let mut merged_blocks = Vec::new();
        let mut current_block = self.free_blocks[0];

        for &(offset, size) in &self.free_blocks[1..] {
            let (current_offset, current_size) = current_block;

            // Check if blocks are adjacent
            if current_offset + current_size == offset {
                // Merge blocks
                current_block = (current_offset, current_size + size);
            } else {
                // Blocks are not adjacent, add current block and start new one
                merged_blocks.push(current_block);
                current_block = (offset, size);
            }
        }

        merged_blocks.push(current_block);
        self.free_blocks = merged_blocks;
    }
}

impl Drop for GpuMemoryPool {
    fn drop(&mut self) {
        match self.device {
            DeviceType::Cpu => {
                if self.base_ptr != 0 {
                    let layout = std::alloc::Layout::from_size_align(self.total_size, 64).unwrap();
                    unsafe {
                        std::alloc::dealloc(self.base_ptr as *mut u8, layout);
                    }
                }
            }
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    // CUDA memory deallocation would go here
                    // cudaFree equivalent
                }
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    // Metal buffer deallocation would go here
                }
            }
            DeviceType::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    // OpenCL buffer deallocation would go here
                }
            }
        }
    }
}

/// GPU memory manager for multiple devices
/// 複数デバイス用GPUメモリマネージャー
pub struct GpuMemoryManager {
    pools: HashMap<DeviceType, Arc<Mutex<GpuMemoryPool>>>,
    default_pool_size: usize,
}

impl GpuMemoryManager {
    /// Create a new GPU memory manager
    /// 新しいGPUメモリマネージャーを作成
    pub fn new(default_pool_size: usize) -> Self {
        GpuMemoryManager {
            pools: HashMap::new(),
            default_pool_size,
        }
    }

    /// Get or create memory pool for device
    /// デバイス用メモリプールを取得または作成
    pub fn get_pool(&mut self, device: DeviceType) -> RusTorchResult<Arc<Mutex<GpuMemoryPool>>> {
        if let Some(pool) = self.pools.get(&device) {
            Ok(pool.clone())
        } else {
            let pool = GpuMemoryPool::new(device, self.default_pool_size)?;
            let pool_arc = Arc::new(Mutex::new(pool));
            self.pools.insert(device, pool_arc.clone());
            Ok(pool_arc)
        }
    }

    /// Allocate memory on specific device
    /// 特定デバイスでメモリを割り当て
    pub fn allocate(
        &mut self,
        device: DeviceType,
        size: usize,
    ) -> RusTorchResult<MemoryAllocation> {
        let pool = self.get_pool(device)?;
        let mut pool_guard = pool.lock().unwrap();
        pool_guard.allocate(size)
    }

    /// Deallocate memory
    /// メモリを解放
    pub fn deallocate(&mut self, allocation: &MemoryAllocation) -> RusTorchResult<()> {
        if let Some(pool) = self.pools.get(&allocation.device) {
            let mut pool_guard = pool.lock().unwrap();
            pool_guard.deallocate(allocation.ptr)
        } else {
            Err(RusTorchError::tensor_op("Device pool not found"))
        }
    }

    /// Get memory statistics for all devices
    /// 全デバイスのメモリ統計を取得
    pub fn memory_stats(&self) -> HashMap<DeviceType, (usize, usize, usize, f32)> {
        let mut stats = HashMap::new();
        for (device, pool) in &self.pools {
            if let Ok(pool_guard) = pool.lock() {
                stats.insert(*device, pool_guard.memory_stats());
            }
        }
        stats
    }

    /// Clear all pools
    /// 全プールをクリア
    pub fn clear(&mut self) {
        self.pools.clear();
    }
}

/// Data transfer operations between devices
/// デバイス間データ転送操作
pub struct DataTransfer;

impl DataTransfer {
    /// Copy data from host to device
    /// ホストからデバイスへデータをコピー
    pub fn host_to_device<T: Copy>(
        src: &[T],
        dst_allocation: &MemoryAllocation,
    ) -> RusTorchResult<()> {
        let src_size = src.len() * std::mem::size_of::<T>();
        if src_size > dst_allocation.size {
            return Err(RusTorchError::tensor_op("Source data too large"));
        }

        match dst_allocation.device {
            DeviceType::Cpu => {
                // Direct memory copy for CPU
                unsafe {
                    let dst_ptr = dst_allocation.ptr as *mut T;
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
                }
            }
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    // CUDA memory copy would go here
                    // cudaMemcpy equivalent
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(RusTorchError::gpu("CUDA not supported"));
                }
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    // Metal buffer copy would go here
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(RusTorchError::gpu("Metal not supported"));
                }
            }
            DeviceType::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    // OpenCL buffer copy would go here
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err(RusTorchError::gpu("OpenCL not supported"));
                }
            }
        }

        Ok(())
    }

    /// Copy data from device to host
    /// デバイスからホストへデータをコピー
    pub fn device_to_host<T: Copy>(
        src_allocation: &MemoryAllocation,
        dst: &mut [T],
    ) -> RusTorchResult<()> {
        let dst_size = dst.len() * std::mem::size_of::<T>();
        if dst_size > src_allocation.size {
            return Err(RusTorchError::tensor_op("Destination buffer too small"));
        }

        match src_allocation.device {
            DeviceType::Cpu => {
                // Direct memory copy for CPU
                unsafe {
                    let src_ptr = src_allocation.ptr as *const T;
                    std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
                }
            }
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    // CUDA memory copy would go here
                    // cudaMemcpy equivalent
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(RusTorchError::gpu("CUDA not supported"));
                }
            }
            DeviceType::Metal(_) => {
                #[cfg(feature = "metal")]
                {
                    // Metal buffer copy would go here
                }
                #[cfg(not(feature = "metal"))]
                {
                    return Err(RusTorchError::gpu("Metal not supported"));
                }
            }
            DeviceType::OpenCL(_) => {
                #[cfg(feature = "opencl")]
                {
                    // OpenCL buffer copy would go here
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err(RusTorchError::gpu("OpenCL not supported"));
                }
            }
        }

        Ok(())
    }

    /// Copy data between devices
    /// デバイス間でデータをコピー
    pub fn device_to_device<T: Copy>(
        src_allocation: &MemoryAllocation,
        dst_allocation: &MemoryAllocation,
        count: usize,
    ) -> RusTorchResult<()> {
        let transfer_size = count * std::mem::size_of::<T>();
        if transfer_size > src_allocation.size || transfer_size > dst_allocation.size {
            return Err(RusTorchError::tensor_op("Transfer size too large"));
        }

        // For now, implement via host memory (not optimal but functional)
        let mut temp_buffer = vec![unsafe { std::mem::zeroed::<T>() }; count];
        Self::device_to_host(src_allocation, &mut temp_buffer)?;
        Self::host_to_device(&temp_buffer, dst_allocation)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let pool = GpuMemoryPool::new(DeviceType::Cpu, 1024 * 1024).unwrap();
        assert_eq!(pool.device(), DeviceType::Cpu);

        let (total, allocated, free, usage) = pool.memory_stats();
        assert_eq!(total, 1024 * 1024);
        assert_eq!(allocated, 0);
        assert_eq!(free, 1024 * 1024);
        assert_eq!(usage, 0.0);
    }

    #[test]
    fn test_memory_allocation() {
        let mut pool = GpuMemoryPool::new(DeviceType::Cpu, 1024 * 1024).unwrap();

        let allocation = pool.allocate(1024).unwrap();
        assert_eq!(allocation.device, DeviceType::Cpu);
        assert_eq!(allocation.size, 1024); // Aligned to 256 bytes

        let (_, allocated, _, usage) = pool.memory_stats();
        assert_eq!(allocated, 1024);
        assert!(usage > 0.0);
    }

    #[test]
    fn test_memory_deallocation() {
        let mut pool = GpuMemoryPool::new(DeviceType::Cpu, 1024 * 1024).unwrap();

        let allocation = pool.allocate(1024).unwrap();
        let ptr = allocation.ptr;

        pool.deallocate(ptr).unwrap();

        let (_, allocated, _, usage) = pool.memory_stats();
        assert_eq!(allocated, 0);
        assert_eq!(usage, 0.0);
    }

    #[test]
    fn test_memory_manager() {
        let mut manager = GpuMemoryManager::new(1024 * 1024);

        let allocation = manager.allocate(DeviceType::Cpu, 1024).unwrap();
        assert_eq!(allocation.device, DeviceType::Cpu);

        manager.deallocate(&allocation).unwrap();

        let stats = manager.memory_stats();
        assert!(stats.contains_key(&DeviceType::Cpu));
    }

    #[test]
    fn test_data_transfer() {
        let mut pool = GpuMemoryPool::new(DeviceType::Cpu, 1024 * 1024).unwrap();
        let allocation = pool.allocate(1024).unwrap();

        let src_data = vec![1.0f32, 2.0, 3.0, 4.0];
        DataTransfer::host_to_device(&src_data, &allocation).unwrap();

        let mut dst_data = vec![0.0f32; 4];
        DataTransfer::device_to_host(&allocation, &mut dst_data).unwrap();

        assert_eq!(src_data, dst_data);
    }

    #[test]
    fn test_block_merging() {
        let mut pool = GpuMemoryPool::new(DeviceType::Cpu, 1024 * 1024).unwrap();

        let alloc1 = pool.allocate(256).unwrap();
        let alloc2 = pool.allocate(256).unwrap();
        let alloc3 = pool.allocate(256).unwrap();

        // Deallocate middle block first
        pool.deallocate(alloc2.ptr).unwrap();
        // Then deallocate adjacent blocks
        pool.deallocate(alloc1.ptr).unwrap();
        pool.deallocate(alloc3.ptr).unwrap();

        // Should be able to allocate a large block again
        let large_alloc = pool.allocate(768).unwrap();
        assert!(large_alloc.size >= 768);
    }
}

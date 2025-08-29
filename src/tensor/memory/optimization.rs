//! Memory optimization utilities for tensor operations
//! テンソル操作のためのメモリ最適化ユーティリティ

use super::aligned::{SimdAllocator, SIMD_ALIGNMENT};
use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use num_traits::Float;

/// Memory information for tensors
/// テンソルのメモリ情報
#[derive(Debug, Clone)]
pub struct TensorMemoryInfo {
    /// Total number of elements in the tensor
    /// テンソルの総要素数
    pub total_elements: usize,
    /// Size of each element in bytes
    /// 各要素のバイトサイズ
    pub element_size: usize,
    /// Total memory used in bytes
    /// 使用メモリの総バイト数
    pub total_bytes: usize,
    /// Whether the tensor data is contiguous in memory
    /// テンソルデータがメモリ上で連続しているか
    pub is_contiguous: bool,
    /// Memory alignment in bytes
    /// メモリアライメント（バイト）
    pub alignment: usize,
    /// Whether the tensor is stored on GPU
    /// テンソルがGPU上に保存されているか
    pub is_on_gpu: bool,
    /// Device type string
    /// デバイスタイプ文字列
    pub device: String,
}

/// Memory optimization trait for tensors
/// テンソルのメモリ最適化トレイト
pub trait MemoryOptimization<T: Float> {
    /// Get memory information about this tensor
    /// このテンソルのメモリ情報を取得
    fn memory_info(&self) -> TensorMemoryInfo;

    /// Check if this tensor can be optimized for memory usage
    /// このテンソルがメモリ使用量を最適化できるかチェック
    fn can_optimize_memory(&self) -> bool;

    /// Create a memory-optimized copy of this tensor
    /// このテンソルのメモリ最適化コピーを作成
    fn optimize_memory(&self) -> Self;

    /// Try to create a memory-optimized copy with error handling
    /// エラーハンドリング付きでメモリ最適化コピーを作成を試行
    fn try_optimize_memory(&self) -> RusTorchResult<Self>
    where
        Self: Sized;
}

impl<T: Float + Clone + 'static> MemoryOptimization<T> for Tensor<T> {
    fn memory_info(&self) -> TensorMemoryInfo {
        let element_size = std::mem::size_of::<T>();
        let total_elements = self.data.len();
        let total_bytes = total_elements * element_size;

        // Check for SIMD alignment
        let ptr = self.data.as_ptr();
        let alignment = if SimdAllocator::is_aligned(ptr) {
            SIMD_ALIGNMENT
        } else {
            // Check for standard alignments
            if (ptr as usize) % 16 == 0 {
                16
            } else if (ptr as usize) % 8 == 0 {
                8
            } else if (ptr as usize) % 4 == 0 {
                4
            } else {
                1
            }
        };

        let is_on_gpu = self.is_on_gpu();
        let device = self.device_type().to_string();

        TensorMemoryInfo {
            total_elements,
            element_size,
            total_bytes,
            is_contiguous: self.data.is_standard_layout(),
            alignment,
            is_on_gpu,
            device,
        }
    }

    fn can_optimize_memory(&self) -> bool {
        let info = self.memory_info();
        // Can optimize if tensor is large enough and not already SIMD-aligned
        info.total_bytes > 1024 && info.alignment < SIMD_ALIGNMENT
    }

    fn optimize_memory(&self) -> Self {
        if !self.can_optimize_memory() {
            return self.clone();
        }

        // Try to create SIMD-aligned copy for f32 tensors
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let shape = self.shape();
            let len = self.numel();

            // Allocate SIMD-aligned memory
            if let Ok(ptr) = SimdAllocator::alloc_f32(len) {
                unsafe {
                    // Copy data to aligned memory
                    let src = self.data.as_ptr();
                    let dst = ptr.as_ptr();
                    std::ptr::copy_nonoverlapping(src as *const f32, dst, len);

                    // Create vector from aligned memory
                    let aligned_data = Vec::from_raw_parts(dst, len, len);

                    // Convert to T type (this is safe because we checked TypeId)
                    let aligned_data_t: Vec<T> = std::mem::transmute(aligned_data);

                    // Create tensor from aligned data
                    match Self::try_from_vec(aligned_data_t, shape.to_vec()) {
                        Ok(tensor) => return tensor,
                        Err(_) => {
                            // If creation fails, deallocate and fall back to clone
                            SimdAllocator::dealloc_f32(ptr, len);
                        }
                    }
                }
            }
        }

        // Fallback to regular clone if SIMD optimization fails
        self.clone()
    }

    fn try_optimize_memory(&self) -> RusTorchResult<Self> {
        let info = self.memory_info();

        // Check if tensor is too large to optimize safely
        const MAX_OPTIMIZE_SIZE: usize = 1_000_000_000; // 1GB
        if info.total_bytes > MAX_OPTIMIZE_SIZE {
            return Err(RusTorchError::TensorOp {
                message: format!(
                    "Tensor too large to optimize: {} bytes exceeds maximum of {} bytes",
                    info.total_bytes, MAX_OPTIMIZE_SIZE
                ),
                source: None,
            });
        }

        Ok(self.optimize_memory())
    }
}

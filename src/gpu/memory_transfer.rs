//! GPU Memory Transfer Operations (Legacy Compatibility Layer)
//! GPUメモリ転送操作（レガシー互換性レイヤー）
//!
//! This module provides backward compatibility for the original memory_transfer.rs
//! implementation. The actual functionality has been reorganized into the
//! `gpu::memory_ops` module for better maintainability.
//!
//! このモジュールは元のmemory_transfer.rs実装の後方互換性を提供します。
//! 実際の機能はより良い保守性のために`gpu::memory_ops`モジュールに再編成されています。

// Re-export everything from the new modular structure
// 新しいモジュラー構造からすべてを再エクスポート

pub use super::memory_ops::{GpuBuffer, GpuMemoryManager};

// For legacy compatibility, also re-export the implementations
// レガシー互換性のため、実装も再エクスポート
#[cfg(feature = "cuda")]
pub use super::memory_ops::cuda::CudaOperations;

#[cfg(feature = "metal")]
pub use super::memory_ops::metal::MetalOperations;

#[cfg(feature = "opencl")]
pub use super::memory_ops::opencl::OpenCLOperations;

pub use super::memory_ops::cpu_fallback::CpuFallback;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_legacy_cpu_transfer() {
        let tensor = Tensor::<f32>::ones(&[2, 3]);
        let device = super::super::DeviceType::Cpu;

        // Transfer to "device" (CPU in this case)
        let buffer = GpuMemoryManager::to_device(&tensor, &device).unwrap();

        // Transfer back to CPU
        let restored = GpuMemoryManager::to_cpu(&buffer, &[2, 3]).unwrap();

        assert_eq!(tensor.shape(), restored.shape());
        assert_eq!(tensor.data, restored.data);
    }

    #[test]
    fn test_legacy_non_contiguous_tensor() {
        // Create a non-contiguous tensor through transpose
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let transposed = tensor.transpose().unwrap();

        let device = super::super::DeviceType::Cpu;

        // Should handle non-contiguous tensor
        let buffer = GpuMemoryManager::to_device(&transposed, &device).unwrap();
        let restored = GpuMemoryManager::to_cpu(&buffer, transposed.shape()).unwrap();

        assert_eq!(transposed.shape(), restored.shape());
    }

    #[test]
    fn test_legacy_memory_manager() {
        let manager = GpuMemoryManager::<f32>::new();
        let _default_manager = GpuMemoryManager::<f32>::default();
    }
}

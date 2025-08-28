//! GPU Memory Module
//! GPUメモリモジュール
//!
//! This module provides comprehensive GPU memory management operations,
//! supporting multiple backends (CUDA, Metal, OpenCL) with automatic fallback.
//! このモジュールは複数のバックエンド（CUDA、Metal、OpenCL）をサポートし、
//! 自動フォールバック機能を持つ包括的なGPUメモリ管理操作を提供します。

// Core modules
pub mod buffer;
pub mod cpu_fallback;
pub mod manager;
pub mod transfer;

// Backend-specific modules
pub mod cuda;
pub mod metal;
pub mod opencl;

// Re-export main types for backward compatibility
// 後方互換性のためメイン型を再エクスポート
pub use buffer::GpuBuffer;
pub use manager::GpuMemoryManager;

// Re-export backend operations for internal use
// 内部使用のためのバックエンド操作を再エクスポート
pub use cpu_fallback::CpuFallback;
pub use cuda::CudaOperations;
pub use metal::MetalOperations;
pub use opencl::OpenCLOperations;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use std::sync::Arc;

    #[test]
    fn test_module_structure() {
        // Test that all main types are accessible
        let _manager = GpuMemoryManager::<f32>::new();
        let _buffer = GpuBuffer::Cpu(Arc::new(vec![1.0f32, 2.0, 3.0]));
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that the API works the same as before
        let tensor = Tensor::<f32>::ones(&[2, 3]);
        let device = crate::gpu::DeviceType::Cpu;

        // Transfer to device
        let buffer = GpuMemoryManager::to_device(&tensor, &device).unwrap();

        // Transfer back to CPU
        let restored = GpuMemoryManager::to_cpu(&buffer, &[2, 3]).unwrap();

        assert_eq!(tensor.shape(), restored.shape());
        assert_eq!(tensor.data, restored.data);
    }

    #[test]
    fn test_elementwise_operations() {
        let manager = GpuMemoryManager::<f32>::new();
        let lhs = GpuBuffer::Cpu(Arc::new(vec![1.0, 2.0, 3.0]));
        let rhs = GpuBuffer::Cpu(Arc::new(vec![4.0, 5.0, 6.0]));

        let result = manager
            .execute_elementwise(&lhs, &rhs, |a, b| a + b)
            .unwrap();

        if let GpuBuffer::Cpu(data) = result {
            assert_eq!(data.as_ref(), &vec![5.0, 7.0, 9.0]);
        }
    }

    #[test]
    fn test_batch_normalization() {
        let manager = GpuMemoryManager::<f32>::new();
        let data = GpuBuffer::Cpu(Arc::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]));
        let epsilon = 1e-5f32;

        let result = manager.execute_batch_normalize(&data, epsilon).unwrap();

        if let GpuBuffer::Cpu(normalized_data) = result {
            // Check that the normalized data has approximately zero mean
            let mean: f32 = normalized_data.iter().sum::<f32>() / normalized_data.len() as f32;
            assert!(
                mean.abs() < 1e-6,
                "Mean should be approximately zero, got {}",
                mean
            );
        }
    }

    #[test]
    fn test_attention_mechanism() {
        let manager = GpuMemoryManager::<f32>::new();
        let query = GpuBuffer::Cpu(Arc::new(vec![1.0f32, 0.5]));
        let key = GpuBuffer::Cpu(Arc::new(vec![0.8f32, 1.2]));
        let value = GpuBuffer::Cpu(Arc::new(vec![2.0f32, 3.0]));

        let result = manager.execute_attention(&query, &key, &value).unwrap();

        if let GpuBuffer::Cpu(attention_result) = result {
            assert_eq!(attention_result.len(), 2);
            assert!(attention_result.iter().all(|&x| x.is_finite()));
        }
    }
}

//! GPU Memory Manager
//! GPUメモリマネージャー

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use num_traits::Float;
use std::marker::PhantomData;

use super::buffer::GpuBuffer;

/// GPU memory manager for tensor operations
/// テンソル演算用GPUメモリマネージャー
#[derive(Default)]
pub struct GpuMemoryManager<T: Float> {
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> GpuMemoryManager<T> {
    /// Create new GPU memory manager
    /// 新しいGPUメモリマネージャーを作成
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Execute element-wise operation on GPU buffers
    /// GPU バッファ上で要素ごとの演算を実行
    pub fn execute_elementwise<F>(
        &self,
        lhs: &GpuBuffer<T>,
        rhs: &GpuBuffer<T>,
        op: F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
    {
        match (lhs, rhs) {
            #[cfg(feature = "cuda")]
            (
                GpuBuffer::Cuda {
                    data: lhs_data,
                    device,
                },
                GpuBuffer::Cuda { data: rhs_data, .. },
            ) => {
                use super::cuda::CudaOperations;
                CudaOperations::execute_elementwise(lhs_data, rhs_data, device, &op)
            }

            #[cfg(feature = "metal")]
            (
                GpuBuffer::Metal {
                    buffer: lhs_buf,
                    device,
                },
                GpuBuffer::Metal {
                    buffer: rhs_buf, ..
                },
            ) => {
                use super::metal::MetalOperations;
                MetalOperations::execute_elementwise(lhs_buf, rhs_buf, device, &op)
            }

            #[cfg(feature = "opencl")]
            (
                GpuBuffer::OpenCL {
                    buffer: lhs_buf,
                    context,
                },
                GpuBuffer::OpenCL {
                    buffer: rhs_buf, ..
                },
            ) => {
                use super::opencl::OpenCLOperations;
                OpenCLOperations::execute_elementwise(lhs_buf, rhs_buf, context, &op)
            }

            // CPU fallback for mixed or CPU buffers
            _ => {
                use super::cpu_fallback::CpuFallback;
                CpuFallback::execute_elementwise(lhs, rhs, &op)
            }
        }
    }

    /// Execute batch normalization on GPU buffer
    /// GPUバッファ上でバッチ正規化を実行
    pub fn execute_batch_normalize(
        &self,
        tensor: &GpuBuffer<T>,
        epsilon: T,
    ) -> RusTorchResult<GpuBuffer<T>> {
        match tensor {
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { data, device } => {
                use super::cuda::CudaOperations;
                CudaOperations::execute_batch_normalize(data, device, epsilon)
            }

            #[cfg(feature = "metal")]
            GpuBuffer::Metal { buffer, device } => {
                use super::metal::MetalOperations;
                MetalOperations::execute_batch_normalize(buffer, device, epsilon)
            }

            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { buffer, context } => {
                use super::opencl::OpenCLOperations;
                OpenCLOperations::execute_batch_normalize(buffer, context, epsilon)
            }

            // CPU fallback
            GpuBuffer::Cpu(data) => {
                use super::cpu_fallback::CpuFallback;
                CpuFallback::execute_batch_normalize(data, epsilon)
            }
        }
    }

    /// Execute attention mechanism on GPU buffers
    /// GPUバッファ上でアテンション機構を実行
    pub fn execute_attention(
        &self,
        query: &GpuBuffer<T>,
        key: &GpuBuffer<T>,
        value: &GpuBuffer<T>,
    ) -> RusTorchResult<GpuBuffer<T>> {
        match (query, key, value) {
            #[cfg(feature = "cuda")]
            (
                GpuBuffer::Cuda {
                    data: query_data,
                    device,
                },
                GpuBuffer::Cuda { data: key_data, .. },
                GpuBuffer::Cuda {
                    data: value_data, ..
                },
            ) => {
                use super::cuda::CudaOperations;
                CudaOperations::execute_attention(query_data, key_data, value_data, device)
            }

            #[cfg(feature = "metal")]
            (
                GpuBuffer::Metal {
                    buffer: query_buf,
                    device,
                },
                GpuBuffer::Metal {
                    buffer: key_buf, ..
                },
                GpuBuffer::Metal {
                    buffer: value_buf, ..
                },
            ) => {
                use super::metal::MetalOperations;
                MetalOperations::execute_attention(query_buf, key_buf, value_buf, device)
            }

            #[cfg(feature = "opencl")]
            (
                GpuBuffer::OpenCL {
                    buffer: query_buf,
                    context,
                },
                GpuBuffer::OpenCL {
                    buffer: key_buf, ..
                },
                GpuBuffer::OpenCL {
                    buffer: value_buf, ..
                },
            ) => {
                use super::opencl::OpenCLOperations;
                OpenCLOperations::execute_attention(query_buf, key_buf, value_buf, context)
            }

            // CPU fallback for mixed or CPU buffers
            _ => {
                use super::cpu_fallback::CpuFallback;
                CpuFallback::execute_attention(query, key, value)
            }
        }
    }
}

// Remove the separate Default implementation to avoid conflict
// impl<T: Float + 'static> Default for GpuMemoryManager<T> {
//     fn default() -> Self {
//         Self::new()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_manager_creation() {
        let manager = GpuMemoryManager::<f32>::new();
        let _default_manager = GpuMemoryManager::<f32>::default();
        // Test passes if creation succeeds
    }

    #[test]
    fn test_cpu_elementwise_operation() {
        let manager = GpuMemoryManager::<f32>::new();
        let lhs = GpuBuffer::Cpu(Arc::new(vec![1.0, 2.0, 3.0]));
        let rhs = GpuBuffer::Cpu(Arc::new(vec![4.0, 5.0, 6.0]));

        let result = manager.execute_elementwise(&lhs, &rhs, |a, b| a + b);
        assert!(result.is_ok());

        if let Ok(GpuBuffer::Cpu(data)) = result {
            assert_eq!(data.as_ref(), &vec![5.0, 7.0, 9.0]);
        }
    }
}

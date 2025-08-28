//! GPU Memory Transfer Operations
//! GPUメモリ転送操作

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::sync::Arc;

use super::buffer::GpuBuffer;
use super::manager::GpuMemoryManager;

impl<T: Float + 'static> GpuMemoryManager<T> {
    /// Transfer tensor from CPU to GPU
    /// テンソルをCPUからGPUに転送
    pub fn to_device(
        tensor: &Tensor<T>,
        device: &crate::gpu::DeviceType,
    ) -> RusTorchResult<GpuBuffer<T>> {
        // Get tensor data as contiguous slice
        let data = if let Some(slice) = tensor.data.as_slice() {
            slice.to_vec()
        } else {
            // Handle non-contiguous tensor
            tensor.data.iter().cloned().collect()
        };

        match device {
            crate::gpu::DeviceType::Cpu => Ok(GpuBuffer::Cpu(Arc::new(data))),

            #[cfg(feature = "cuda")]
            crate::gpu::DeviceType::Cuda(device_id) => {
                use super::cuda::CudaOperations;
                CudaOperations::transfer_to_device(data, *device_id)
            }

            #[cfg(feature = "metal")]
            crate::gpu::DeviceType::Metal(_device_id) => {
                use super::metal::MetalOperations;
                MetalOperations::transfer_to_device(data)
            }

            #[cfg(feature = "opencl")]
            crate::gpu::DeviceType::OpenCL(device_id) => {
                use super::opencl::OpenCLOperations;
                OpenCLOperations::transfer_to_device(data, *device_id)
            }

            #[allow(unreachable_patterns)]
            _ => Err(RusTorchError::gpu("Unsupported device type")),
        }
    }

    /// Transfer GPU buffer back to CPU tensor
    /// GPUバッファをCPUテンソルに転送
    pub fn to_cpu(buffer: &GpuBuffer<T>, shape: &[usize]) -> RusTorchResult<Tensor<T>> {
        let data = match buffer {
            GpuBuffer::Cpu(data) => (**data).clone(),

            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { data, device: _ } => {
                use super::cuda::CudaOperations;
                CudaOperations::transfer_to_cpu(data)?
            }

            #[cfg(feature = "metal")]
            GpuBuffer::Metal { buffer, device: _ } => {
                use super::metal::MetalOperations;
                MetalOperations::transfer_to_cpu(buffer, shape)?
            }

            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { buffer, context } => {
                use super::opencl::OpenCLOperations;
                OpenCLOperations::transfer_to_cpu(buffer, context)?
            }
        };

        // Validate shape matches data length
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(RusTorchError::gpu(format!(
                "Shape mismatch: expected {} elements for shape {:?}, but got {}",
                expected_len,
                shape,
                data.len()
            )));
        }

        // Convert Vec to ArrayD
        let array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| RusTorchError::gpu(format!("Failed to create array: {}", e)))?;

        Ok(Tensor::from_ndarray(array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_cpu_transfer() {
        let tensor = Tensor::<f32>::ones(&[2, 3]);
        let device = crate::gpu::DeviceType::Cpu;

        // Transfer to "device" (CPU in this case)
        let buffer = GpuMemoryManager::to_device(&tensor, &device).unwrap();

        // Transfer back to CPU
        let restored = GpuMemoryManager::to_cpu(&buffer, &[2, 3]).unwrap();

        assert_eq!(tensor.shape(), restored.shape());
        assert_eq!(tensor.data, restored.data);
    }

    #[test]
    fn test_non_contiguous_tensor() {
        // Create a non-contiguous tensor through transpose
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let transposed = tensor.transpose().unwrap();

        let device = crate::gpu::DeviceType::Cpu;

        // Should handle non-contiguous tensor
        let buffer = GpuMemoryManager::to_device(&transposed, &device).unwrap();
        let restored = GpuMemoryManager::to_cpu(&buffer, transposed.shape()).unwrap();

        assert_eq!(transposed.shape(), restored.shape());
    }

    #[test]
    fn test_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let buffer = GpuBuffer::Cpu(Arc::new(data));

        // Wrong shape should cause error
        let result = GpuMemoryManager::to_cpu(&buffer, &[2, 3]); // Expects 6 elements, but has 4
        assert!(result.is_err());
    }
}

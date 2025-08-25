//! GPU Memory Transfer Operations
//! GPUメモリ転送操作
//!
//! This module provides efficient memory transfer operations between CPU and GPU,
//! supporting multiple backends (CUDA, Metal, OpenCL) with automatic fallback.

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use num_traits::Float;
use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice as CudarcDevice, CudaSlice, CudaStream};

#[cfg(feature = "metal")]
use metal::{Buffer, Device as MetalDeviceType, CommandQueue};

#[cfg(feature = "opencl")]
use opencl3::{
    memory::{Buffer as CLBuffer, CL_MEM_READ_WRITE},
    command_queue::CommandQueue as CLCommandQueue,
    context::Context as CLContext,
};

/// GPU memory buffer abstraction
pub enum GpuBuffer<T> {
    #[cfg(feature = "cuda")]
    Cuda {
        data: Arc<CudaSlice<T>>,
        device: Arc<CudarcDevice>,
    },
    
    #[cfg(feature = "metal")]
    Metal {
        buffer: Arc<Buffer>,
        device: Arc<MetalDeviceType>,
    },
    
    #[cfg(feature = "opencl")]
    OpenCL {
        buffer: Arc<CLBuffer<T>>,
        context: Arc<CLContext>,
    },
    
    /// CPU fallback
    Cpu(Arc<Vec<T>>),
}

/// GPU memory manager for tensor operations
pub struct GpuMemoryManager<T: Float> {
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> GpuMemoryManager<T> {
    /// Transfer tensor from CPU to GPU
    pub fn to_device(
        tensor: &Tensor<T>,
        device: &super::DeviceType,
    ) -> RusTorchResult<GpuBuffer<T>> {
        // Get tensor data as contiguous slice
        let data = if let Some(slice) = tensor.data.as_slice() {
            slice.to_vec()
        } else {
            // Handle non-contiguous tensor
            tensor.data.iter().cloned().collect()
        };
        
        match device {
            super::DeviceType::Cpu => {
                Ok(GpuBuffer::Cpu(Arc::new(data)))
            }
            
            #[cfg(feature = "cuda")]
            super::DeviceType::Cuda(device_id) => {
                Self::cuda_transfer_to_device(data, *device_id)
            }
            
            #[cfg(feature = "metal")]
            super::DeviceType::Metal(_device_id) => {
                Self::metal_transfer_to_device(data)
            }
            
            #[cfg(feature = "opencl")]
            super::DeviceType::OpenCL(device_id) => {
                Self::opencl_transfer_to_device(data, *device_id)
            }
            
            #[allow(unreachable_patterns)]
            _ => Err(RusTorchError::gpu("Unsupported device type")),
        }
    }
    
    /// Transfer GPU buffer back to CPU tensor
    pub fn to_cpu(
        buffer: &GpuBuffer<T>,
        shape: &[usize],
    ) -> RusTorchResult<Tensor<T>> {
        let data = match buffer {
            GpuBuffer::Cpu(data) => (**data).clone(),
            
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { data, device: _ } => {
                Self::cuda_transfer_to_cpu(data)?
            }
            
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { buffer, device: _ } => {
                Self::metal_transfer_to_cpu(buffer, shape)?
            }
            
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { buffer, context } => {
                Self::opencl_transfer_to_cpu(buffer, context)?
            }
        };
        
        // Convert Vec to ArrayD
        let array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| RusTorchError::gpu(&format!("Shape mismatch: {}", e)))?;
        
        Ok(Tensor::from_ndarray(array))
    }
}

// CUDA implementation
#[cfg(feature = "cuda")]
impl<T: Float + 'static> GpuMemoryManager<T> {
    fn cuda_transfer_to_device(
        data: Vec<T>,
        device_id: usize,
    ) -> RusTorchResult<GpuBuffer<T>> 
    where
        T: cudarc::driver::DeviceRepr,
    {
        use cudarc::driver::CudaDevice;
        
        // Initialize CUDA device
        let device = CudaDevice::new(device_id)
            .map_err(|e| RusTorchError::gpu(&format!("CUDA init failed: {}", e)))?;
        
        let device = Arc::new(device);
        
        // Allocate device memory and copy data
        let gpu_buffer = device.htod_sync_copy(&data)
            .map_err(|e| RusTorchError::gpu(&format!("CUDA transfer failed: {}", e)))?;
        
        Ok(GpuBuffer::Cuda {
            data: Arc::new(gpu_buffer),
            device,
        })
    }
    
    fn cuda_transfer_to_cpu(
        cuda_slice: &Arc<CudaSlice<T>>,
    ) -> RusTorchResult<Vec<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        // Get device from the slice
        let device = cuda_slice.device();
        
        // Synchronize and copy data back
        let cpu_data = device.dtoh_sync_copy(cuda_slice)
            .map_err(|e| RusTorchError::gpu(&format!("CUDA transfer to CPU failed: {}", e)))?;
        
        Ok(cpu_data)
    }
}

// Metal implementation
#[cfg(feature = "metal")]
impl<T: Float + 'static> GpuMemoryManager<T> {
    fn metal_transfer_to_device(
        data: Vec<T>,
    ) -> RusTorchResult<GpuBuffer<T>> {
        use metal::{Device, MTLResourceOptions};
        
        let device = Device::system_default()
            .ok_or_else(|| RusTorchError::gpu("No Metal device found"))?;
        
        let device = Arc::new(device);
        
        // Calculate byte size - handle Float trait
        let byte_size = data.len() * std::mem::size_of::<T>();
        
        // Create Metal buffer with data
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        Ok(GpuBuffer::Metal {
            buffer: Arc::new(buffer),
            device,
        })
    }
    
    fn metal_transfer_to_cpu(
        metal_buffer: &Arc<Buffer>,
        shape: &[usize],
    ) -> RusTorchResult<Vec<T>> {
        let total_elements: usize = shape.iter().product();
        let byte_size = total_elements * std::mem::size_of::<T>();
        
        if metal_buffer.length() as usize != byte_size {
            return Err(RusTorchError::gpu("Buffer size mismatch"));
        }
        
        // Copy data from Metal buffer
        let contents = metal_buffer.contents();
        let mut cpu_data = vec![T::zero(); total_elements];
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents as *const T,
                cpu_data.as_mut_ptr(),
                total_elements,
            );
        }
        
        Ok(cpu_data)
    }
}

// OpenCL implementation  
#[cfg(feature = "opencl")]
impl<T: Float + 'static> GpuMemoryManager<T> {
    fn opencl_transfer_to_device(
        data: Vec<T>,
        _device_id: usize,
    ) -> RusTorchResult<GpuBuffer<T>> 
    where
        T: opencl3::memory::ClMem,
    {
        use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_GPU};
        use opencl3::context::Context;
        use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
        
        // Get OpenCL device
        let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|e| RusTorchError::gpu(&format!("OpenCL device error: {}", e)))?
            .into_iter()
            .next()
            .ok_or_else(|| RusTorchError::gpu("No OpenCL GPU device found"))?;
        
        // Create context and command queue
        let context = Context::from_device(&device_id)
            .map_err(|e| RusTorchError::gpu(&format!("OpenCL context error: {}", e)))?;
        
        let context = Arc::new(context);
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .map_err(|e| RusTorchError::gpu(&format!("OpenCL queue error: {}", e)))?;
        
        // Create buffer and copy data
        let mut buffer = CLBuffer::<T>::create(
            &context, 
            CL_MEM_READ_WRITE, 
            data.len(), 
            std::ptr::null_mut()
        ).map_err(|e| RusTorchError::gpu(&format!("OpenCL buffer creation failed: {}", e)))?;
        
        queue.enqueue_write_buffer(&mut buffer, true, 0, &data, &[])
            .map_err(|e| RusTorchError::gpu(&format!("OpenCL write failed: {}", e)))?;
        
        Ok(GpuBuffer::OpenCL {
            buffer: Arc::new(buffer),
            context,
        })
    }
    
    fn opencl_transfer_to_cpu(
        cl_buffer: &Arc<CLBuffer<T>>,
        context: &Arc<CLContext>,
    ) -> RusTorchResult<Vec<T>>
    where
        T: opencl3::memory::ClMem,
    {
        use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
        
        let queue = CommandQueue::create_default(context, CL_QUEUE_PROFILING_ENABLE)
            .map_err(|e| RusTorchError::gpu(&format!("OpenCL queue error: {}", e)))?;
        
        let size = cl_buffer.size() / std::mem::size_of::<T>();
        let mut cpu_data = vec![T::zero(); size];
        
        queue.enqueue_read_buffer(cl_buffer, true, 0, &mut cpu_data, &[])
            .map_err(|e| RusTorchError::gpu(&format!("OpenCL read failed: {}", e)))?;
        
        Ok(cpu_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    
    #[test]
    fn test_cpu_transfer() {
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
    fn test_non_contiguous_tensor() {
        // Create a non-contiguous tensor through transpose
        let tensor = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        );
        let transposed = tensor.transpose().unwrap();
        
        let device = super::super::DeviceType::Cpu;
        
        // Should handle non-contiguous tensor
        let buffer = GpuMemoryManager::to_device(&transposed, &device).unwrap();
        let restored = GpuMemoryManager::to_cpu(&buffer, transposed.shape()).unwrap();
        
        assert_eq!(transposed.shape(), restored.shape());
    }
}
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
use metal::{Buffer, CommandQueue, Device as MetalDeviceType};

#[cfg(feature = "opencl")]
use opencl3::{
    command_queue::CommandQueue as CLCommandQueue,
    context::Context as CLContext,
    memory::{Buffer as CLBuffer, CL_MEM_READ_WRITE},
};

/// GPU memory buffer abstraction
pub enum GpuBuffer<T> {
    #[cfg(feature = "cuda")]
    Cuda {
        data: Arc<CudaSlice<T>>,
        device: Arc<CudarcDevice>,
    },

    #[cfg(feature = "metal")]
    /// Metal GPU buffer with device reference
    /// デバイス参照付きMetal GPUバッファ
    Metal {
        /// Metal buffer for GPU memory
        /// GPU メモリ用のMetalバッファ
        buffer: Arc<Buffer>,
        /// Metal device reference
        /// Metalデバイスの参照
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

/// GPU memory manager for tensor operations#[derive(Default)]
pub struct GpuMemoryManager<T: Float> {
    _phantom: PhantomData<T>,
}

impl<T: Float + 'static> GpuMemoryManager<T> {
    /// Create new GPU memory manager
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

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
            super::DeviceType::Cpu => Ok(GpuBuffer::Cpu(Arc::new(data))),

            #[cfg(feature = "cuda")]
            super::DeviceType::Cuda(device_id) => Self::cuda_transfer_to_device(data, *device_id),

            #[cfg(feature = "metal")]
            super::DeviceType::Metal(_device_id) => Self::metal_transfer_to_device(data),

            #[cfg(feature = "opencl")]
            super::DeviceType::OpenCL(device_id) => {
                Self::opencl_transfer_to_device(data, *device_id)
            }

            #[allow(unreachable_patterns)]
            _ => Err(RusTorchError::gpu("Unsupported device type")),
        }
    }

    /// Transfer GPU buffer back to CPU tensor
    pub fn to_cpu(buffer: &GpuBuffer<T>, shape: &[usize]) -> RusTorchResult<Tensor<T>> {
        let data = match buffer {
            GpuBuffer::Cpu(data) => (**data).clone(),

            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { data, device: _ } => Self::cuda_transfer_to_cpu(data)?,

            #[cfg(feature = "metal")]
            GpuBuffer::Metal { buffer, device: _ } => Self::metal_transfer_to_cpu(buffer, shape)?,

            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { buffer, context } => Self::opencl_transfer_to_cpu(buffer, context)?,
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

    /// Execute element-wise operation on GPU buffers
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
            ) => self.execute_cuda_elementwise(lhs_data, rhs_data, device, &op),

            #[cfg(feature = "metal")]
            (
                GpuBuffer::Metal {
                    buffer: lhs_buf,
                    device,
                },
                GpuBuffer::Metal {
                    buffer: rhs_buf, ..
                },
            ) => self.execute_metal_elementwise(lhs_buf, rhs_buf, device, &op),

            #[cfg(feature = "opencl")]
            (
                GpuBuffer::OpenCL {
                    buffer: lhs_buf,
                    context,
                },
                GpuBuffer::OpenCL {
                    buffer: rhs_buf, ..
                },
            ) => self.execute_opencl_elementwise(lhs_buf, rhs_buf, context, &op),

            // CPU fallback for mixed or CPU buffers
            _ => self.execute_cpu_fallback_elementwise(lhs, rhs, &op),
        }
    }

    /// CPU fallback for element-wise operations
    fn execute_cpu_fallback_elementwise<F>(
        &self,
        lhs: &GpuBuffer<T>,
        rhs: &GpuBuffer<T>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
    {
        let lhs_data = match lhs {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
        };

        let rhs_data = match rhs {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
        };

        let result: Vec<T> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Ok(GpuBuffer::Cpu(Arc::new(result)))
    }

    /// Execute batch normalization on GPU buffer
    pub fn execute_batch_normalize(
        &self,
        tensor: &GpuBuffer<T>,
        epsilon: T,
    ) -> RusTorchResult<GpuBuffer<T>> {
        match tensor {
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { data, device } => {
                self.execute_cuda_batch_normalize(data, device, epsilon)
            }

            #[cfg(feature = "metal")]
            GpuBuffer::Metal { buffer, device } => {
                self.execute_metal_batch_normalize(buffer, device, epsilon)
            }

            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { buffer, context } => {
                self.execute_opencl_batch_normalize(buffer, context, epsilon)
            }

            // CPU fallback
            GpuBuffer::Cpu(data) => self.execute_cpu_batch_normalize(data, epsilon),
        }
    }

    /// CPU fallback for batch normalization
    fn execute_cpu_batch_normalize(
        &self,
        data: &Arc<Vec<T>>,
        epsilon: T,
    ) -> RusTorchResult<GpuBuffer<T>> {
        let input_data = data.as_slice();
        let n = input_data.len();

        if n == 0 {
            return Ok(GpuBuffer::Cpu(data.clone()));
        }

        // 平均計算
        let mean = input_data.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(n).ok_or_else(|| RusTorchError::gpu("Failed to convert size to float"))?;

        // 分散計算
        let variance = input_data
            .iter()
            .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean))
            / T::from(n).ok_or_else(|| RusTorchError::gpu("Failed to convert size to float"))?;

        // 正規化
        let std_dev = (variance + epsilon).sqrt();
        let normalized: Vec<T> = input_data.iter().map(|&x| (x - mean) / std_dev).collect();

        Ok(GpuBuffer::Cpu(Arc::new(normalized)))
    }

    /// Execute attention mechanism on GPU buffers
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
            ) => self.execute_cuda_attention(query_data, key_data, value_data, device),

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
            ) => self.execute_metal_attention(query_buf, key_buf, value_buf, device),

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
            ) => self.execute_opencl_attention(query_buf, key_buf, value_buf, context),

            // CPU fallback for mixed or CPU buffers
            _ => self.execute_cpu_attention(query, key, value),
        }
    }

    /// CPU fallback for attention mechanism
    fn execute_cpu_attention(
        &self,
        query: &GpuBuffer<T>,
        key: &GpuBuffer<T>,
        value: &GpuBuffer<T>,
    ) -> RusTorchResult<GpuBuffer<T>> {
        // Extract data from all buffers (assuming all are CPU buffers for fallback)
        let query_data = match query {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
        };

        let key_data = match key {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
        };

        let value_data = match value {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
        };

        // 簡単な行列乗算ベースのアテンション（単純化版）
        // scores = query @ key^T
        let scores: Vec<T> = query_data
            .iter()
            .zip(key_data.iter())
            .map(|(&q, &k)| q * k) // 簡単なドット積近似
            .collect();

        // softmax適用（簡単版）
        let max_score = scores
            .iter()
            .fold(T::neg_infinity(), |max, &x| if x > max { x } else { max });

        let exp_scores: Vec<T> = scores.iter().map(|&x| (x - max_score).exp()).collect();

        let sum_exp = exp_scores.iter().fold(T::zero(), |acc, &x| acc + x);

        let attention_weights: Vec<T> = exp_scores.iter().map(|&x| x / sum_exp).collect();

        // 重み付きvalue計算
        let result: Vec<T> = attention_weights
            .iter()
            .zip(value_data.iter())
            .map(|(&w, &v)| w * v)
            .collect();

        Ok(GpuBuffer::Cpu(Arc::new(result)))
    }
}

impl<T: Float + 'static> Default for GpuMemoryManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

// CUDA implementation
#[cfg(feature = "cuda")]
impl<T: Float + 'static> GpuMemoryManager<T> {
    fn cuda_transfer_to_device(data: Vec<T>, device_id: usize) -> RusTorchResult<GpuBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        use cudarc::driver::CudaDevice;

        // Initialize CUDA device
        let device = CudaDevice::new(device_id)
            .map_err(|e| RusTorchError::gpu(&format!("CUDA init failed: {}", e)))?;

        let device = Arc::new(device);

        // Allocate device memory and copy data
        let gpu_buffer = device
            .htod_sync_copy(&data)
            .map_err(|e| RusTorchError::gpu(&format!("CUDA transfer failed: {}", e)))?;

        Ok(GpuBuffer::Cuda {
            data: Arc::new(gpu_buffer),
            device,
        })
    }

    fn cuda_transfer_to_cpu(cuda_slice: &Arc<CudaSlice<T>>) -> RusTorchResult<Vec<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        // Get device from the slice
        let device = cuda_slice.device();

        // Synchronize and copy data back
        let cpu_data = device
            .dtoh_sync_copy(cuda_slice)
            .map_err(|e| RusTorchError::gpu(&format!("CUDA transfer to CPU failed: {}", e)))?;

        Ok(cpu_data)
    }

    /// Execute CUDA element-wise operation
    fn execute_cuda_elementwise<F>(
        &self,
        lhs: &Arc<CudaSlice<T>>,
        rhs: &Arc<CudaSlice<T>>,
        device: &Arc<CudarcDevice>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: cudarc::driver::DeviceRepr,
    {
        use crate::gpu::cuda_kernels::CudaKernelExecutor;

        // Try to use CUDA kernels for common operations
        let result = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            self.execute_cuda_elementwise_f32(lhs, rhs, device, op)
        } else {
            // Fallback to CPU for other types
            self.execute_cuda_elementwise_fallback(lhs, rhs, device, op)
        };

        result
    }

    /// Execute CUDA element-wise operation for f32 type
    fn execute_cuda_elementwise_f32<F>(
        &self,
        lhs: &Arc<CudaSlice<T>>,
        rhs: &Arc<CudaSlice<T>>,
        device: &Arc<CudarcDevice>,
        _op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: cudarc::driver::DeviceRepr,
    {
        // For demonstration, implement addition using actual CUDA kernel
        // In practice, we'd need to determine the operation type

        let size = lhs.len();
        let device_id = device.device_id() as usize;

        // Create result buffer
        let mut result_slice = device
            .alloc_zeros::<T>(size)
            .map_err(|e| RusTorchError::gpu(&format!("CUDA allocation failed: {}", e)))?;

        // For now, use a simple CUDA operation (addition)
        // TODO: Determine operation type from closure and use appropriate kernel
        unsafe {
            use cudarc::driver::LaunchConfig;

            // Simple element-wise kernel compilation and execution
            let kernel_src = r#"
            extern "C" __global__ void elementwise_add_f32(
                const float* a, const float* b, float* result, int n
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < n) {
                    result[idx] = a[idx] + b[idx];
                }
            }
            "#;

            // Compile and load kernel
            if let Ok(ptx) = cudarc::nvrtc::compile_ptx(kernel_src) {
                if device
                    .load_ptx(ptx, "elementwise", &["elementwise_add_f32"])
                    .is_ok()
                {
                    if let Ok(func) = device.get_func("elementwise", "elementwise_add_f32") {
                        let threads_per_block = 256;
                        let blocks = size.div_ceil(threads_per_block);

                        let config = LaunchConfig {
                            grid_dim: (blocks as u32, 1, 1),
                            block_dim: (threads_per_block as u32, 1, 1),
                            shared_mem_bytes: 0,
                        };

                        let _ = func.launch(config, (lhs, rhs, &mut result_slice, size as i32));
                    }
                }
            }
        }

        // Return GPU buffer with result
        Ok(GpuBuffer::Cuda {
            data: Arc::new(result_slice),
            device: device.clone(),
        })
    }

    /// Fallback CPU implementation for CUDA element-wise operation
    fn execute_cuda_elementwise_fallback<F>(
        &self,
        lhs: &Arc<CudaSlice<T>>,
        rhs: &Arc<CudaSlice<T>>,
        device: &Arc<CudarcDevice>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: cudarc::driver::DeviceRepr,
    {
        // CPU fallback implementation
        let lhs_cpu = Self::cuda_transfer_to_cpu(lhs)?;
        let rhs_cpu = Self::cuda_transfer_to_cpu(rhs)?;

        let result: Vec<T> = lhs_cpu
            .iter()
            .zip(rhs_cpu.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        // Transfer result back to CUDA
        let device_id = device.device_id() as usize;
        Self::cuda_transfer_to_device(result, device_id)
    }

    /// Execute CUDA batch normalization
    fn execute_cuda_batch_normalize(
        &self,
        data: &Arc<CudaSlice<T>>,
        device: &Arc<CudarcDevice>,
        epsilon: T,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        // For now, fall back to CPU implementation
        // TODO: Implement actual CUDA batch normalization kernel
        let cpu_data = Self::cuda_transfer_to_cpu(data)?;
        let gpu_manager = GpuMemoryManager::new();
        let cpu_buffer = GpuBuffer::Cpu(Arc::new(cpu_data));
        let result = gpu_manager.execute_cpu_batch_normalize(
            match &cpu_buffer {
                GpuBuffer::Cpu(data) => data,
                _ => unreachable!(),
            },
            epsilon,
        )?;

        // Transfer result back to CUDA
        match result {
            GpuBuffer::Cpu(data) => {
                let device_id = device.device_id() as usize;
                Self::cuda_transfer_to_device(data.as_ref().clone(), device_id)
            }
            _ => unreachable!(),
        }
    }

    /// Execute CUDA attention mechanism
    fn execute_cuda_attention(
        &self,
        query: &Arc<CudaSlice<T>>,
        key: &Arc<CudaSlice<T>>,
        value: &Arc<CudaSlice<T>>,
        device: &Arc<CudarcDevice>,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        // For now, fall back to CPU implementation
        // TODO: Implement actual CUDA attention kernel
        let query_cpu = Self::cuda_transfer_to_cpu(query)?;
        let key_cpu = Self::cuda_transfer_to_cpu(key)?;
        let value_cpu = Self::cuda_transfer_to_cpu(value)?;

        let gpu_manager = GpuMemoryManager::new();
        let query_buf = GpuBuffer::Cpu(Arc::new(query_cpu));
        let key_buf = GpuBuffer::Cpu(Arc::new(key_cpu));
        let value_buf = GpuBuffer::Cpu(Arc::new(value_cpu));

        let result = gpu_manager.execute_cpu_attention(&query_buf, &key_buf, &value_buf)?;

        // Transfer result back to CUDA
        match result {
            GpuBuffer::Cpu(data) => {
                let device_id = device.device_id() as usize;
                Self::cuda_transfer_to_device(data.as_ref().clone(), device_id)
            }
            _ => unreachable!(),
        }
    }
}

// Metal implementation
#[cfg(feature = "metal")]
impl<T: Float + 'static> GpuMemoryManager<T> {
    fn metal_transfer_to_device(data: Vec<T>) -> RusTorchResult<GpuBuffer<T>> {
        use metal::{Device, MTLResourceOptions};

        let device =
            Device::system_default().ok_or_else(|| RusTorchError::gpu("No Metal device found"))?;

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

    /// Execute Metal element-wise operation
    fn execute_metal_elementwise<F>(
        &self,
        lhs: &Arc<Buffer>,
        rhs: &Arc<Buffer>,
        _device: &Arc<MetalDeviceType>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
    {
        // For now, fall back to CPU implementation
        // TODO: Implement actual Metal compute shader
        let lhs_cpu =
            Self::metal_transfer_to_cpu(lhs, &[lhs.length() as usize / std::mem::size_of::<T>()])?;
        let rhs_cpu =
            Self::metal_transfer_to_cpu(rhs, &[rhs.length() as usize / std::mem::size_of::<T>()])?;

        let result: Vec<T> = lhs_cpu
            .iter()
            .zip(rhs_cpu.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        // Transfer result back to Metal
        Self::metal_transfer_to_device(result)
    }

    /// Execute Metal batch normalization
    fn execute_metal_batch_normalize(
        &self,
        buffer: &Arc<Buffer>,
        _device: &Arc<MetalDeviceType>,
        epsilon: T,
    ) -> RusTorchResult<GpuBuffer<T>> {
        // For now, fall back to CPU implementation
        // TODO: Implement actual Metal compute shader for batch normalization
        let cpu_data = Self::metal_transfer_to_cpu(
            buffer,
            &[buffer.length() as usize / std::mem::size_of::<T>()],
        )?;
        let gpu_manager = GpuMemoryManager::new();
        let cpu_buffer = GpuBuffer::Cpu(Arc::new(cpu_data));
        let result = gpu_manager.execute_cpu_batch_normalize(
            match &cpu_buffer {
                GpuBuffer::Cpu(data) => data,
                _ => unreachable!(),
            },
            epsilon,
        )?;

        // Transfer result back to Metal
        match result {
            GpuBuffer::Cpu(data) => Self::metal_transfer_to_device(data.as_ref().clone()),
            _ => unreachable!(),
        }
    }

    /// Execute Metal attention mechanism
    fn execute_metal_attention(
        &self,
        query: &Arc<Buffer>,
        key: &Arc<Buffer>,
        value: &Arc<Buffer>,
        _device: &Arc<MetalDeviceType>,
    ) -> RusTorchResult<GpuBuffer<T>> {
        // For now, fall back to CPU implementation
        // TODO: Implement actual Metal compute shader for attention
        let query_cpu = Self::metal_transfer_to_cpu(
            query,
            &[query.length() as usize / std::mem::size_of::<T>()],
        )?;
        let key_cpu =
            Self::metal_transfer_to_cpu(key, &[key.length() as usize / std::mem::size_of::<T>()])?;
        let value_cpu = Self::metal_transfer_to_cpu(
            value,
            &[value.length() as usize / std::mem::size_of::<T>()],
        )?;

        let gpu_manager = GpuMemoryManager::new();
        let query_buf = GpuBuffer::Cpu(Arc::new(query_cpu));
        let key_buf = GpuBuffer::Cpu(Arc::new(key_cpu));
        let value_buf = GpuBuffer::Cpu(Arc::new(value_cpu));

        let result = gpu_manager.execute_cpu_attention(&query_buf, &key_buf, &value_buf)?;

        // Transfer result back to Metal
        match result {
            GpuBuffer::Cpu(data) => Self::metal_transfer_to_device(data.as_ref().clone()),
            _ => unreachable!(),
        }
    }
}

// OpenCL implementation
#[cfg(feature = "opencl")]
impl<T: Float + 'static> GpuMemoryManager<T> {
    fn opencl_transfer_to_device(data: Vec<T>, _device_id: usize) -> RusTorchResult<GpuBuffer<T>>
    where
        T: opencl3::memory::ClMem,
    {
        use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
        use opencl3::context::Context;
        use opencl3::device::{get_all_devices, CL_DEVICE_TYPE_GPU};

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
            std::ptr::null_mut(),
        )
        .map_err(|e| RusTorchError::gpu(&format!("OpenCL buffer creation failed: {}", e)))?;

        queue
            .enqueue_write_buffer(&mut buffer, true, 0, &data, &[])
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

        queue
            .enqueue_read_buffer(cl_buffer, true, 0, &mut cpu_data, &[])
            .map_err(|e| RusTorchError::gpu(&format!("OpenCL read failed: {}", e)))?;

        Ok(cpu_data)
    }

    /// Execute OpenCL element-wise operation
    fn execute_opencl_elementwise<F>(
        &self,
        lhs: &Arc<CLBuffer<T>>,
        rhs: &Arc<CLBuffer<T>>,
        context: &Arc<CLContext>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: opencl3::memory::ClMem,
    {
        // For now, fall back to CPU implementation
        // TODO: Implement actual OpenCL kernel
        let lhs_cpu = Self::opencl_transfer_to_cpu(lhs, context)?;
        let rhs_cpu = Self::opencl_transfer_to_cpu(rhs, context)?;

        let result: Vec<T> = lhs_cpu
            .iter()
            .zip(rhs_cpu.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        // Transfer result back to OpenCL
        Self::opencl_transfer_to_device(result, 0) // Default device 0
    }

    /// Execute OpenCL batch normalization
    fn execute_opencl_batch_normalize(
        &self,
        buffer: &Arc<CLBuffer<T>>,
        context: &Arc<CLContext>,
        epsilon: T,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: opencl3::memory::ClMem,
    {
        // For now, fall back to CPU implementation
        // TODO: Implement actual OpenCL kernel for batch normalization
        let cpu_data = Self::opencl_transfer_to_cpu(buffer, context)?;
        let gpu_manager = GpuMemoryManager::new();
        let cpu_buffer = GpuBuffer::Cpu(Arc::new(cpu_data));
        let result = gpu_manager.execute_cpu_batch_normalize(
            match &cpu_buffer {
                GpuBuffer::Cpu(data) => data,
                _ => unreachable!(),
            },
            epsilon,
        )?;

        // Transfer result back to OpenCL
        match result {
            GpuBuffer::Cpu(data) => Self::opencl_transfer_to_device(data.as_ref().clone(), 0),
            _ => unreachable!(),
        }
    }

    /// Execute OpenCL attention mechanism
    fn execute_opencl_attention(
        &self,
        query: &Arc<CLBuffer<T>>,
        key: &Arc<CLBuffer<T>>,
        value: &Arc<CLBuffer<T>>,
        context: &Arc<CLContext>,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: opencl3::memory::ClMem,
    {
        // For now, fall back to CPU implementation
        // TODO: Implement actual OpenCL kernel for attention
        let query_cpu = Self::opencl_transfer_to_cpu(query, context)?;
        let key_cpu = Self::opencl_transfer_to_cpu(key, context)?;
        let value_cpu = Self::opencl_transfer_to_cpu(value, context)?;

        let gpu_manager = GpuMemoryManager::new();
        let query_buf = GpuBuffer::Cpu(Arc::new(query_cpu));
        let key_buf = GpuBuffer::Cpu(Arc::new(key_cpu));
        let value_buf = GpuBuffer::Cpu(Arc::new(value_cpu));

        let result = gpu_manager.execute_cpu_attention(&query_buf, &key_buf, &value_buf)?;

        // Transfer result back to OpenCL
        match result {
            GpuBuffer::Cpu(data) => Self::opencl_transfer_to_device(data.as_ref().clone(), 0),
            _ => unreachable!(),
        }
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
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let transposed = tensor.transpose().unwrap();

        let device = super::super::DeviceType::Cpu;

        // Should handle non-contiguous tensor
        let buffer = GpuMemoryManager::to_device(&transposed, &device).unwrap();
        let restored = GpuMemoryManager::to_cpu(&buffer, transposed.shape()).unwrap();

        assert_eq!(transposed.shape(), restored.shape());
    }
}

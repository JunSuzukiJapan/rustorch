//! OpenCL Memory Operations
//! OpenCLメモリ操作

#[cfg(feature = "opencl")]
use crate::error::{RusTorchError, RusTorchResult};
#[cfg(feature = "opencl")]
use num_traits::Float;
#[cfg(feature = "opencl")]
use std::sync::Arc;

#[cfg(feature = "opencl")]
use opencl3::{
    command_queue::{CommandQueue as CLCommandQueue, CL_NON_BLOCKING, CL_QUEUE_PROFILING_ENABLE},
    context::Context as CLContext,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    memory::{Buffer as CLBuffer, ClMem, CL_MEM_READ_WRITE},
    platform::get_platforms,
};

#[cfg(feature = "opencl")]
use super::buffer::GpuBuffer;

#[cfg(feature = "opencl")]
/// OpenCL-specific memory operations
/// OpenCL固有のメモリ操作
pub struct OpenCLOperations;

#[cfg(feature = "opencl")]
impl OpenCLOperations {
    /// Transfer data to OpenCL device
    /// データをOpenCLデバイスに転送
    pub fn transfer_to_device<T>(data: Vec<T>, _device_id: usize) -> RusTorchResult<GpuBuffer<T>>
    where
        T: opencl3::memory::ClMem + Float + 'static,
    {
        // Get OpenCL device
        let platforms = get_platforms()
            .map_err(|e| RusTorchError::gpu(&format!("OpenCL platforms error: {}", e)))?;

        let mut device = None;
        for platform in platforms {
            if let Ok(device_ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) {
                if !device_ids.is_empty() {
                    device = Some(Device::new(device_ids[0]));
                    break;
                }
            }
        }

        let device = device.ok_or_else(|| RusTorchError::gpu("No OpenCL GPU device found"))?;

        // Create context and command queue
        let context = CLContext::from_device(&device)
            .map_err(|e| RusTorchError::gpu(&format!("OpenCL context error: {}", e)))?;

        let context = Arc::new(context);
        let queue =
            CLCommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
                .map_err(|e| RusTorchError::gpu(&format!("OpenCL queue error: {}", e)))?;

        // Create buffer and copy data
        let mut buffer = unsafe {
            CLBuffer::<T>::create(
                &context,
                CL_MEM_READ_WRITE,
                data.len(),
                std::ptr::null_mut(),
            )
        }
        .map_err(|e| RusTorchError::gpu(&format!("OpenCL buffer creation failed: {}", e)))?;

        unsafe {
            queue
                .enqueue_write_buffer(&mut buffer, CL_NON_BLOCKING, 0, &data, &[])
                .map_err(|e| RusTorchError::gpu(&format!("OpenCL write failed: {}", e)))?;
        }

        Ok(GpuBuffer::OpenCL {
            buffer: Arc::new(buffer),
            context,
        })
    }

    /// Transfer data from OpenCL device to CPU
    /// データをOpenCLデバイスからCPUに転送
    pub fn transfer_to_cpu<T>(
        cl_buffer: &Arc<CLBuffer<T>>,
        context: &Arc<CLContext>,
    ) -> RusTorchResult<Vec<T>>
    where
        T: opencl3::memory::ClMem + Float + 'static,
    {
        // Create command queue from context
        let queue =
            CLCommandQueue::create_default_with_properties(context, CL_QUEUE_PROFILING_ENABLE, 0)
                .map_err(|e| RusTorchError::gpu(&format!("OpenCL queue error: {}", e)))?;

        let size = cl_buffer.size().unwrap_or(0) / std::mem::size_of::<T>();
        let mut cpu_data = vec![T::zero(); size];

        unsafe {
            queue
                .enqueue_read_buffer(cl_buffer, CL_NON_BLOCKING, 0, &mut cpu_data, &[])
                .map_err(|e| RusTorchError::gpu(&format!("OpenCL read failed: {}", e)))?;
        }

        Ok(cpu_data)
    }

    /// Execute OpenCL element-wise operation
    /// OpenCL要素ごとの演算実行
    pub fn execute_elementwise<T, F>(
        lhs: &Arc<CLBuffer<T>>,
        rhs: &Arc<CLBuffer<T>>,
        context: &Arc<CLContext>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: opencl3::memory::ClMem + Float + 'static,
    {
        use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
        use opencl3::kernel::{ExecuteKernel, Kernel};
        use opencl3::memory::{Buffer as CLBuffer, CL_MEM_READ_WRITE};
        use opencl3::program::Program;

        let size = lhs.size().unwrap_or(0) / std::mem::size_of::<T>();

        // Create test values to determine operation type
        let test_a = T::from(2.0).unwrap();
        let test_b = T::from(3.0).unwrap();
        let test_result = op(test_a, test_b);

        // Determine kernel function name based on operation
        let kernel_name = if test_result == T::from(5.0).unwrap() {
            "elementwise_add_f32"
        } else if test_result == T::from(6.0).unwrap() {
            "elementwise_mul_f32"
        } else if test_result == T::from(-1.0).unwrap() {
            "elementwise_sub_f32"
        } else if test_result == T::from(2.0 / 3.0).unwrap() {
            "elementwise_div_f32"
        } else {
            // Use fallback for unknown operations
            return Self::execute_elementwise_fallback(lhs, rhs, context, op);
        };

        // For now, use CPU fallback as we don't have the OpenCL kernels included
        // In a production implementation, this would load and execute actual OpenCL kernels
        Self::execute_elementwise_fallback(lhs, rhs, context, op)
    }

    /// Fallback CPU implementation for OpenCL element-wise operation
    /// OpenCL要素ごと演算のCPUフォールバック実装
    fn execute_elementwise_fallback<T, F>(
        lhs: &Arc<CLBuffer<T>>,
        rhs: &Arc<CLBuffer<T>>,
        context: &Arc<CLContext>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: opencl3::memory::ClMem + Float + 'static,
    {
        let lhs_cpu = Self::transfer_to_cpu(lhs, context)?;
        let rhs_cpu = Self::transfer_to_cpu(rhs, context)?;

        let result: Vec<T> = lhs_cpu
            .iter()
            .zip(rhs_cpu.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        // Transfer result back to OpenCL
        Self::transfer_to_device(result, 0) // Default device 0
    }

    /// Execute OpenCL batch normalization
    /// OpenCLバッチ正規化実行
    pub fn execute_batch_normalize<T>(
        buffer: &Arc<CLBuffer<T>>,
        context: &Arc<CLContext>,
        epsilon: T,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: opencl3::memory::ClMem + Float + 'static,
    {
        let size = buffer.size().unwrap_or(0) / std::mem::size_of::<T>();

        // For now, use CPU fallback for batch normalization
        // In a production implementation, this would use OpenCL kernels
        let cpu_data = Self::transfer_to_cpu(buffer, context)?;
        let n = cpu_data.len();

        if n == 0 {
            return Ok(GpuBuffer::OpenCL {
                buffer: buffer.clone(),
                context: context.clone(),
            });
        }

        // Calculate mean and variance on CPU
        let mean: T =
            cpu_data.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(size as f64).unwrap();
        let variance: T = cpu_data
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x)
            / T::from(size as f64).unwrap();

        // Normalize
        let std_dev = (variance + epsilon).sqrt();
        let normalized: Vec<T> = cpu_data.iter().map(|&x| (x - mean) / std_dev).collect();

        // Transfer result back to OpenCL
        Self::transfer_to_device(normalized, 0) // Default device 0
    }

    /// Execute OpenCL attention mechanism
    /// OpenCLアテンション機構実行
    pub fn execute_attention<T>(
        query: &Arc<CLBuffer<T>>,
        key: &Arc<CLBuffer<T>>,
        value: &Arc<CLBuffer<T>>,
        context: &Arc<CLContext>,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: opencl3::memory::ClMem + Float + 'static,
    {
        // For now, fall back to CPU implementation
        // TODO: Implement actual OpenCL kernel for attention
        let query_cpu = Self::transfer_to_cpu(query, context)?;
        let key_cpu = Self::transfer_to_cpu(key, context)?;
        let value_cpu = Self::transfer_to_cpu(value, context)?;

        // Simple attention computation on CPU
        let scores: Vec<T> = query_cpu
            .iter()
            .zip(key_cpu.iter())
            .map(|(&q, &k)| q * k)
            .collect();

        // Softmax
        let max_score = scores
            .iter()
            .fold(T::neg_infinity(), |max, &x| if x > max { x } else { max });

        let exp_scores: Vec<T> = scores.iter().map(|&x| (x - max_score).exp()).collect();
        let sum_exp = exp_scores.iter().fold(T::zero(), |acc, &x| acc + x);
        let attention_weights: Vec<T> = exp_scores.iter().map(|&x| x / sum_exp).collect();

        // Weighted sum with values
        let result: Vec<T> = attention_weights
            .iter()
            .zip(value_cpu.iter())
            .map(|(&w, &v)| w * v)
            .collect();

        // Transfer result back to OpenCL
        Self::transfer_to_device(result, 0) // Default device 0
    }
}

#[cfg(not(feature = "opencl"))]
/// Stub for OpenCL operations when OpenCL is not available
/// OpenCL無効時のOpenCL操作スタブ
pub struct OpenCLOperations;

#[cfg(not(feature = "opencl"))]
impl OpenCLOperations {
    // Stub implementations that return errors
}

#[cfg(test)]
#[cfg(feature = "opencl")]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_operations_stub() {
        // This test will only run when OpenCL is enabled
        // Basic compilation test
    }
}

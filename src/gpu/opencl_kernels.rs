//! OpenCL kernel implementations for GPU acceleration
//! GPU加速のためのOpenCLカーネル実装

use super::{GpuError};
use num_traits::Float;
use std::ffi::c_void;
use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "opencl")]
use opencl3::{
    context::Context,
    device::Device,
    command_queue::CommandQueue,
    program::Program,
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms,
};

/// OpenCL kernel types
/// OpenCLカーネルタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenClKernelType {
    ElementWise,
    MatMul,
    Reduction,
    Convolution,
    BatchNorm,
}

/// OpenCL kernel parameters
/// OpenCLカーネルパラメータ
#[derive(Debug, Clone)]
pub struct OpenClKernelParams {
    pub global_work_size: [usize; 3],
    pub local_work_size: [usize; 3],
    pub queue_index: usize,
}

impl Default for OpenClKernelParams {
    fn default() -> Self {
        Self {
            global_work_size: [1, 1, 1],
            local_work_size: [1, 1, 1],
            queue_index: 0,
        }
    }
}

/// OpenCL buffer wrapper
/// OpenCLバッファラッパー
pub struct OpenClBuffer<T> {
    buffer: *mut c_void,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T> OpenClBuffer<T> {
    /// Create a new OpenCL buffer
    /// 新しいOpenCLバッファを作成
    #[cfg(feature = "opencl")]
    pub fn new(size: usize, context: &Context) -> Result<Self, GpuError> {
        let buffer_size = size * std::mem::size_of::<T>();
        let buffer = Buffer::<T>::create(
            context,
            CL_MEM_READ_WRITE,
            buffer_size,
            std::ptr::null_mut(),
        ).map_err(|e| GpuError::MemoryAllocationError(format!("OpenCL buffer creation failed: {:?}", e)))?;
        
        Ok(OpenClBuffer {
            buffer,
            size,
            _phantom: PhantomData,
        })
    }
    
    #[cfg(not(feature = "opencl"))]
    pub fn new(_size: usize, _context: &()) -> Result<Self, GpuError> {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
    
    /// Create buffer from host data
    /// ホストデータからバッファを作成
    #[cfg(feature = "opencl")]
    pub fn from_host_data(data: &[T]) -> Result<Self, GpuError> {
        // This would require a proper OpenCL context
        // For now, return an error
        Err(GpuError::UnsupportedDevice("OpenCL context required".to_string()))
    }
    
    #[cfg(not(feature = "opencl"))]
    pub fn from_host_data(_data: &[T]) -> Result<Self, GpuError> {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
    
    /// Copy data to host
    /// ホストにデータをコピー
    #[cfg(feature = "opencl")]
    pub fn copy_to_host(&self, _host_data: &mut [T]) -> Result<(), GpuError> {
        // This would require proper OpenCL queue operations
        Err(GpuError::UnsupportedDevice("OpenCL queue required".to_string()))
    }
    
    #[cfg(not(feature = "opencl"))]
    pub fn copy_to_host(&self, _host_data: &mut [T]) -> Result<(), GpuError> {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
    
    /// Get buffer size
    /// バッファサイズを取得
    pub fn size(&self) -> usize {
        self.size
    }
}

/// OpenCL kernel executor for high-performance GPU operations
/// 高性能GPU演算のためのOpenCLカーネル実行器
#[cfg(feature = "opencl")]
pub struct OpenClKernelExecutor {
    device: Device,
    context: Context,
    queue: CommandQueue,
    program: Program,
    kernels: HashMap<OpenClKernelType, Kernel>,
}

#[cfg(feature = "opencl")]
impl OpenClKernelExecutor {
    /// Create a new OpenCL kernel executor
    /// 新しいOpenCLカーネル実行器を作成
    pub fn new(device_id: usize) -> Result<Self, GpuError> {
        // Get OpenCL platforms and devices
        let platforms = get_platforms()
            .map_err(|e| GpuError::InitializationError(format!("Failed to get OpenCL platforms: {:?}", e)))?;
        
        if platforms.is_empty() {
            return Err(GpuError::InitializationError("No OpenCL platforms found".to_string()));
        }
        
        let devices = platforms[0].get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .map_err(|e| GpuError::InitializationError(format!("Failed to get OpenCL devices: {:?}", e)))?;
        
        if devices.is_empty() || device_id >= devices.len() {
            return Err(GpuError::InitializationError(format!("OpenCL device {} not found", device_id)));
        }
        
        let device = devices[device_id];
        
        // Create OpenCL context
        let context = opencl3::context::Context::from_device(&device)
            .map_err(|e| GpuError::InitializationError(format!("Failed to create OpenCL context: {:?}", e)))?;
        
        // Create command queue
        let queue = opencl3::command_queue::CommandQueue::create_default(&context, opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE)
            .map_err(|e| GpuError::InitializationError(format!("Failed to create OpenCL command queue: {:?}", e)))?;
        
        // OpenCL kernel source code
        let kernel_source = r#"
        __kernel void elementwise_add_f32(
            __global const float* a,
            __global const float* b,
            __global float* result,
            const unsigned int n
        ) {
            int gid = get_global_id(0);
            if (gid < n) {
                result[gid] = a[gid] + b[gid];
            }
        }
        
        __kernel void matrix_multiply_f32(
            __global const float* a,
            __global const float* b,
            __global float* c,
            const unsigned int M,
            const unsigned int N,
            const unsigned int K
        ) {
            int row = get_global_id(1);
            int col = get_global_id(0);
            
            if (row < M && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += a[row * K + k] * b[k * N + col];
                }
                c[row * N + col] = sum;
            }
        }
        
        __kernel void reduce_sum_f32(
            __global const float* input,
            __global float* output,
            __local float* shared_data,
            const unsigned int n
        ) {
            int gid = get_global_id(0);
            int lid = get_local_id(0);
            int group_size = get_local_size(0);
            
            // Load data into local memory
            shared_data[lid] = (gid < n) ? input[gid] : 0.0f;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            // Reduction in local memory
            for (int s = group_size / 2; s > 0; s >>= 1) {
                if (lid < s) {
                    shared_data[lid] += shared_data[lid + s];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            
            // Write result for this work group
            if (lid == 0) {
                output[get_group_id(0)] = shared_data[0];
            }
        }
        "#;
        
        // Create and build program
        let program = opencl3::program::Program::create_and_build_from_source(&context, kernel_source, "")
            .map_err(|e| GpuError::KernelCompilationError(format!("Failed to compile OpenCL kernels: {:?}", e)))?;
        
        let mut kernels = HashMap::new();
        
        // Create kernels
        let add_kernel = opencl3::kernel::Kernel::create(&program, "elementwise_add_f32")
            .map_err(|e| GpuError::KernelCompilationError(format!("Failed to create add kernel: {:?}", e)))?;
        kernels.insert(OpenClKernelType::ElementWise, add_kernel);
        
        let matmul_kernel = opencl3::kernel::Kernel::create(&program, "matrix_multiply_f32")
            .map_err(|e| GpuError::KernelCompilationError(format!("Failed to create matmul kernel: {:?}", e)))?;
        kernels.insert(OpenClKernelType::MatMul, matmul_kernel);
        
        let reduce_kernel = opencl3::kernel::Kernel::create(&program, "reduce_sum_f32")
            .map_err(|e| GpuError::KernelCompilationError(format!("Failed to create reduce kernel: {:?}", e)))?;
        kernels.insert(OpenClKernelType::Reduction, reduce_kernel);
        
        Ok(Self {
            device,
            context,
            queue,
            program,
            kernels,
        })
    }
    
    /// Execute element-wise addition using OpenCL
    /// OpenCLを使用して要素ごと加算を実行
    pub fn elementwise_add_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
    ) -> Result<(), GpuError> {
        let size = a.len();
        if b.len() != size || c.len() != size {
            return Err(GpuError::InvalidOperation(
                "Array size mismatch in element-wise addition".to_string()
            ));
        }
        
        // Create OpenCL buffers
        let a_buffer = opencl3::memory::Buffer::<f32>::create(
            &self.context,
            opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
            size,
            a.as_ptr() as *mut f32,
        ).map_err(|e| GpuError::AllocationError(format!("Failed to create buffer A: {:?}", e)))?;
        
        let b_buffer = opencl3::memory::Buffer::<f32>::create(
            &self.context,
            opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
            size,
            b.as_ptr() as *mut f32,
        ).map_err(|e| GpuError::AllocationError(format!("Failed to create buffer B: {:?}", e)))?;
        
        let c_buffer = opencl3::memory::Buffer::<f32>::create(
            &self.context,
            opencl3::memory::CL_MEM_WRITE_ONLY,
            size,
            std::ptr::null_mut(),
        ).map_err(|e| GpuError::AllocationError(format!("Failed to create buffer C: {:?}", e)))?;
        
        // Get kernel
        let kernel = self.kernels.get(&OpenClKernelType::ElementWise)
            .ok_or_else(|| GpuError::KernelExecutionError("ElementWise kernel not found".to_string()))?;
        
        // Set kernel arguments
        kernel.set_arg(0, &a_buffer)
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 0: {:?}", e)))?;
        kernel.set_arg(1, &b_buffer)
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 1: {:?}", e)))?;
        kernel.set_arg(2, &c_buffer)
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 2: {:?}", e)))?;
        kernel.set_arg(3, &(size as u32))
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 3: {:?}", e)))?;
        
        // Execute kernel
        let global_work_size = [size, 0, 0];
        let local_work_size = [256.min(size), 0, 0];
        
        self.queue.enqueue_nd_range_kernel(kernel, 1, None, &global_work_size, Some(&local_work_size), &[])
            .map_err(|e| GpuError::KernelExecutionError(format!("Kernel execution failed: {:?}", e)))?;
        
        // Read result back
        self.queue.enqueue_read_buffer(&c_buffer, opencl3::types::CL_TRUE, 0, c, &[])
            .map_err(|e| GpuError::InvalidOperation(format!("Failed to read result: {:?}", e)))?;
        
        Ok(())
    }
    
    /// Execute matrix multiplication using OpenCL
    /// OpenCLを使用して行列乗算を実行
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        // Create OpenCL buffers
        let a_buffer = opencl3::memory::Buffer::<f32>::create(
            &self.context,
            opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
            m * k,
            a.as_ptr() as *mut f32,
        ).map_err(|e| GpuError::AllocationError(format!("Failed to create buffer A: {:?}", e)))?;
        
        let b_buffer = opencl3::memory::Buffer::<f32>::create(
            &self.context,
            opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
            k * n,
            b.as_ptr() as *mut f32,
        ).map_err(|e| GpuError::AllocationError(format!("Failed to create buffer B: {:?}", e)))?;
        
        let c_buffer = opencl3::memory::Buffer::<f32>::create(
            &self.context,
            opencl3::memory::CL_MEM_WRITE_ONLY,
            m * n,
            std::ptr::null_mut(),
        ).map_err(|e| GpuError::AllocationError(format!("Failed to create buffer C: {:?}", e)))?;
        
        // Get kernel
        let kernel = self.kernels.get(&OpenClKernelType::MatMul)
            .ok_or_else(|| GpuError::KernelExecutionError("MatMul kernel not found".to_string()))?;
        
        // Set kernel arguments
        kernel.set_arg(0, &a_buffer)
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 0: {:?}", e)))?;
        kernel.set_arg(1, &b_buffer)
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 1: {:?}", e)))?;
        kernel.set_arg(2, &c_buffer)
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 2: {:?}", e)))?;
        kernel.set_arg(3, &(m as u32))
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 3: {:?}", e)))?;
        kernel.set_arg(4, &(n as u32))
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 4: {:?}", e)))?;
        kernel.set_arg(5, &(k as u32))
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 5: {:?}", e)))?;
        
        // Execute kernel
        let global_work_size = [n, m, 0];
        let local_work_size = [16.min(n), 16.min(m), 0];
        
        self.queue.enqueue_nd_range_kernel(kernel, 2, None, &global_work_size, Some(&local_work_size), &[])
            .map_err(|e| GpuError::KernelExecutionError(format!("Kernel execution failed: {:?}", e)))?;
        
        // Read result back
        self.queue.enqueue_read_buffer(&c_buffer, opencl3::types::CL_TRUE, 0, c, &[])
            .map_err(|e| GpuError::InvalidOperation(format!("Failed to read result: {:?}", e)))?;
        
        Ok(())
    }
    
    /// Execute reduction operation (sum) using OpenCL
    /// OpenCLを使用してリダクション演算（合計）を実行
    pub fn reduce_sum_f32(&self, input: &[f32]) -> Result<f32, GpuError> {
        let size = input.len();
        let local_size = 256;
        let global_size = ((size + local_size - 1) / local_size) * local_size;
        let num_groups = global_size / local_size;
        
        // Create OpenCL buffers
        let input_buffer = opencl3::memory::Buffer::<f32>::create(
            &self.context,
            opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
            size,
            input.as_ptr() as *mut f32,
        ).map_err(|e| GpuError::AllocationError(format!("Failed to create input buffer: {:?}", e)))?;
        
        let output_buffer = opencl3::memory::Buffer::<f32>::create(
            &self.context,
            opencl3::memory::CL_MEM_WRITE_ONLY,
            num_groups,
            std::ptr::null_mut(),
        ).map_err(|e| GpuError::AllocationError(format!("Failed to create output buffer: {:?}", e)))?;
        
        // Get kernel
        let kernel = self.kernels.get(&OpenClKernelType::Reduction)
            .ok_or_else(|| GpuError::KernelExecutionError("Reduction kernel not found".to_string()))?;
        
        // Set kernel arguments
        kernel.set_arg(0, &input_buffer)
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 0: {:?}", e)))?;
        kernel.set_arg(1, &output_buffer)
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 1: {:?}", e)))?;
        kernel.set_arg_local::<f32>(2, local_size)
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set local arg 2: {:?}", e)))?;
        kernel.set_arg(3, &(size as u32))
            .map_err(|e| GpuError::KernelExecutionError(format!("Failed to set arg 3: {:?}", e)))?;
        
        // Execute kernel
        let global_work_size = [global_size, 0, 0];
        let local_work_size = [local_size, 0, 0];
        
        self.queue.enqueue_nd_range_kernel(kernel, 1, None, &global_work_size, Some(&local_work_size), &[])
            .map_err(|e| GpuError::KernelExecutionError(format!("Kernel execution failed: {:?}", e)))?;
        
        // Read partial results back
        let mut partial_results = vec![0.0f32; num_groups];
        self.queue.enqueue_read_buffer(&output_buffer, opencl3::types::CL_TRUE, 0, &mut partial_results, &[])
            .map_err(|e| GpuError::InvalidOperation(format!("Failed to read partial results: {:?}", e)))?;
        
        Ok(partial_results.iter().sum())
    }
}

/// Non-OpenCL fallback executor for compatibility
/// 互換性のための非OpenCLフォールバック実行器
#[cfg(not(feature = "opencl"))]
pub struct OpenClKernelExecutor;

#[cfg(not(feature = "opencl"))]
impl OpenClKernelExecutor {
    pub fn new(_device_id: usize) -> Result<Self, GpuError> {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
    
    pub fn elementwise_add_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
    ) -> Result<(), GpuError> {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
    
    pub fn matmul_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<(), GpuError> {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
    
    pub fn reduce_sum_f32(&self, _input: &[f32]) -> Result<f32, GpuError> {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
}

/// Public interface functions for OpenCL operations
/// OpenCL演算のためのパブリックインターフェース関数

/// Execute OpenCL matrix multiplication
/// OpenCL行列乗算を実行
pub fn opencl_matmul_f32(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
) -> Result<(), GpuError> {
    #[cfg(feature = "opencl")]
    {
        let executor = OpenClKernelExecutor::new(0)?;
        executor.matmul_f32(_a, _b, _c, _m, _n, _k)
    }
    #[cfg(not(feature = "opencl"))]
    {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
}

/// Execute OpenCL element-wise addition
/// OpenCL要素ごと加算を実行
pub fn opencl_elementwise_add_f32(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
) -> Result<(), GpuError> {
    #[cfg(feature = "opencl")]
    {
        let executor = OpenClKernelExecutor::new(0)?;
        executor.elementwise_add_f32(_a, _b, _c)
    }
    #[cfg(not(feature = "opencl"))]
    {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
}

/// Execute OpenCL reduction sum
/// OpenCLリダクション合計を実行
pub fn opencl_reduce_sum_f32(_input: &[f32]) -> Result<f32, GpuError> {
    #[cfg(feature = "opencl")]
    {
        let executor = OpenClKernelExecutor::new(0)?;
        executor.reduce_sum_f32(_input)
    }
    #[cfg(not(feature = "opencl"))]
    {
        Err(GpuError::UnsupportedDevice("OpenCL not available".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_opencl_kernel_params() {
        let params = OpenClKernelParams::default();
        assert_eq!(params.global_work_size, [1, 1, 1]);
        assert_eq!(params.local_work_size, [1, 1, 1]);
        assert_eq!(params.queue_index, 0);
    }
    
    #[test]
    fn test_opencl_executor_creation() {
        let result = OpenClKernelExecutor::new(0);
        #[cfg(not(feature = "opencl"))]
        assert!(result.is_err());
    }
    
    #[test]
    fn test_opencl_kernel_types() {
        assert_eq!(OpenClKernelType::ElementWise, OpenClKernelType::ElementWise);
        assert_ne!(OpenClKernelType::ElementWise, OpenClKernelType::MatMul);
    }
}

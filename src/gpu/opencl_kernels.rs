//! OpenCL kernel implementations for GPU acceleration
//! GPU加速のためのOpenCLカーネル実装

use crate::error::{RusTorchError, RusTorchResult};
// OpenCL GPU kernel implementations
#[cfg(feature = "opencl")]
use std::collections::HashMap;
#[cfg(feature = "opencl")]
use std::ffi::c_void;
use std::marker::PhantomData;

#[cfg(feature = "opencl")]
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::Device,
    kernel::Kernel,
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms,
    program::Program,
};

/// OpenCL kernel types
/// OpenCLカーネルタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpenClKernelType {
    /// Element-wise operations (add, mul, etc.)
    /// 要素ごとの演算（加算、乗算など）
    ElementWise,
    /// Matrix multiplication operations
    /// 行列乗算演算
    MatMul,
    /// Reduction operations (sum, mean, etc.)
    /// リダクション演算（合計、平均など）
    Reduction,
    /// Convolution operations
    /// 畳み込み演算
    Convolution,
    /// Batch normalization operations
    /// バッチ正規化演算
    BatchNorm,
}

/// OpenCL kernel parameters
/// OpenCLカーネルパラメータ#[derive(Debug, Clone)]
pub struct OpenClKernelParams {
    /// Global work size for OpenCL kernel execution
    /// OpenCLカーネル実行のグローバルワークサイズ
    pub global_work_size: [usize; 3],
    /// Local work size for OpenCL kernel execution
    /// OpenCLカーネル実行のローカルワークサイズ
    pub local_work_size: [usize; 3],
    /// Queue index for OpenCL command queue
    /// OpenCLコマンドキューのインデックス
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
    #[cfg(feature = "opencl")]
    _buffer: Buffer<T>,
    #[cfg(not(feature = "opencl"))]
    _buffer: (),
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T> OpenClBuffer<T> {
    /// Create a new OpenCL buffer
    /// 新しいOpenCLバッファを作成
    #[cfg(feature = "opencl")]
    pub fn new(size: usize, context: &Context) -> RusTorchResult<Self> {
        let buffer_size = size * std::mem::size_of::<T>();
        let buffer = unsafe {
            Buffer::<T>::create(
                context,
                CL_MEM_READ_WRITE,
                buffer_size,
                std::ptr::null_mut(),
            )
        }
        .map_err(|e| RusTorchError::gpu(format!("OpenCL buffer creation failed: {:?}", e)))?;

        Ok(Self {
            _buffer: buffer,
            size,
            _phantom: PhantomData,
        })
    }

    #[cfg(not(feature = "opencl"))]
    /// Create a new OpenCL buffer (fallback when OpenCL not available)
    /// 新しいOpenCLバッファを作成（OpenCL利用不可時のフォールバック）
    pub fn new(_size: usize, _context: &()) -> RusTorchResult<Self> {
        Err(RusTorchError::backend_unavailable("OpenCL"))
    }

    /// Create buffer from host data
    /// ホストデータからバッファを作成
    #[cfg(feature = "opencl")]
    pub fn from_host_data(data: &[T]) -> RusTorchResult<Self> {
        // This would require a proper OpenCL context
        // For now, return an error
        Err(RusTorchError::gpu("OpenCL context required"))
    }

    #[cfg(not(feature = "opencl"))]
    /// Create buffer from host data (fallback when OpenCL not available)
    /// ホストデータからバッファを作成（OpenCL利用不可時のフォールバック）
    pub fn from_host_data(_data: &[T]) -> RusTorchResult<Self> {
        Err(RusTorchError::backend_unavailable("OpenCL"))
    }

    /// Copy data to host
    /// ホストにデータをコピー
    #[cfg(feature = "opencl")]
    pub fn copy_to_host(&self, _host_data: &mut [T]) -> RusTorchResult<()> {
        // This would require proper OpenCL queue operations
        Err(RusTorchError::gpu("OpenCL queue required"))
    }

    #[cfg(not(feature = "opencl"))]
    /// Copy buffer data to host memory (fallback when OpenCL not available)
    /// バッファデータをホストメモリにコピー（OpenCL利用不可時のフォールバック）
    pub fn copy_to_host(&self, _host_data: &mut [T]) -> RusTorchResult<()> {
        Err(RusTorchError::backend_unavailable("OpenCL"))
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
    pub fn new(device_id: usize) -> RusTorchResult<Self> {
        // Get OpenCL platforms and devices
        let platforms = get_platforms()
            .map_err(|e| RusTorchError::gpu(format!("Failed to get OpenCL platforms: {:?}", e)))?;

        if platforms.is_empty() {
            return Err(RusTorchError::gpu("No OpenCL platforms found".to_string()));
        }

        let devices = platforms[0]
            .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .map_err(|e| RusTorchError::gpu(format!("Failed to get OpenCL devices: {:?}", e)))?;

        if devices.is_empty() || device_id >= devices.len() {
            return Err(RusTorchError::gpu(format!(
                "OpenCL device {} not found",
                device_id
            )));
        }

        let device_id = devices[device_id];
        let device = opencl3::device::Device::new(device_id);

        // Create OpenCL context
        let context = opencl3::context::Context::from_device(&device)
            .map_err(|e| RusTorchError::gpu(format!("Failed to create OpenCL context: {:?}", e)))?;

        // Create command queue
        let queue = opencl3::command_queue::CommandQueue::create_default_with_properties(
            &context,
            opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE,
            0,
        )
        .map_err(|e| {
            RusTorchError::gpu(format!("Failed to create OpenCL command queue: {:?}", e))
        })?;

        // Load OpenCL kernel source from external file
        let kernel_source = include_str!("opencl_kernels.cl");

        // Create and build program
        let program =
            opencl3::program::Program::create_and_build_from_source(&context, kernel_source, "")
                .map_err(|e| {
                    RusTorchError::gpu(format!("Failed to compile OpenCL kernels: {:?}", e))
                })?;

        let mut kernels = HashMap::new();

        // Create kernels
        let add_kernel = opencl3::kernel::Kernel::create(&program, "elementwise_add_f32")
            .map_err(|e| RusTorchError::gpu(format!("Failed to create add kernel: {:?}", e)))?;
        kernels.insert(OpenClKernelType::ElementWise, add_kernel);

        let matmul_kernel = opencl3::kernel::Kernel::create(&program, "matrix_multiply_f32")
            .map_err(|e| RusTorchError::gpu(format!("Failed to create matmul kernel: {:?}", e)))?;
        kernels.insert(OpenClKernelType::MatMul, matmul_kernel);

        let reduce_kernel = opencl3::kernel::Kernel::create(&program, "reduce_sum_f32")
            .map_err(|e| RusTorchError::gpu(format!("Failed to create reduce kernel: {:?}", e)))?;
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
    pub fn elementwise_add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> RusTorchResult<()> {
        let size = a.len();
        if b.len() != size || c.len() != size {
            return Err(RusTorchError::invalid_params(
                "matmul",
                "Array size mismatch in element-wise addition".to_string(),
            ));
        }

        // Create OpenCL buffers
        let a_buffer = unsafe {
            opencl3::memory::Buffer::<f32>::create(
                &self.context,
                opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
                size,
                a.as_ptr() as *mut std::ffi::c_void,
            )
        }
        .map_err(|e| RusTorchError::gpu(format!("Failed to create buffer A: {:?}", e)))?;

        let b_buffer = unsafe {
            opencl3::memory::Buffer::<f32>::create(
                &self.context,
                opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
                size,
                b.as_ptr() as *mut std::ffi::c_void,
            )
        }
        .map_err(|e| RusTorchError::gpu(format!("Failed to create buffer B: {:?}", e)))?;

        let c_buffer = unsafe {
            opencl3::memory::Buffer::<f32>::create(
                &self.context,
                opencl3::memory::CL_MEM_WRITE_ONLY,
                size,
                std::ptr::null_mut(),
            )
        }
        .map_err(|e| RusTorchError::gpu(format!("Failed to create buffer C: {:?}", e)))?;

        // Get kernel
        let kernel = self
            .kernels
            .get(&OpenClKernelType::ElementWise)
            .ok_or_else(|| {
                RusTorchError::KernelExecutionError("ElementWise kernel not found".to_string())
            })?;

        // Execute kernel
        let global_work_size = [size];
        let local_work_size = [256.min(size)];

        unsafe {
            opencl3::kernel::ExecuteKernel::new(kernel)
                .set_arg(&a_buffer)
                .set_arg(&b_buffer)
                .set_arg(&c_buffer)
                .set_arg(&(size as u32))
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(&self.queue)
        }
        .map_err(|e| {
            RusTorchError::KernelExecutionError(format!("Kernel execution failed: {:?}", e))
        })?;

        // Read result back
        unsafe {
            self.queue
                .enqueue_read_buffer(&c_buffer, opencl3::types::CL_TRUE, 0, c, &[])
                .map_err(|e| {
                    RusTorchError::invalid_params(
                        "matmul",
                        format!("Failed to read result: {:?}", e),
                    )
                })?;
        }

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
    ) -> RusTorchResult<()> {
        // Create OpenCL buffers
        let a_buffer = unsafe {
            opencl3::memory::Buffer::<f32>::create(
                &self.context,
                opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
                m * k,
                a.as_ptr() as *mut std::ffi::c_void,
            )
        }
        .map_err(|e| RusTorchError::gpu(format!("Failed to create buffer A: {:?}", e)))?;

        let b_buffer = unsafe {
            opencl3::memory::Buffer::<f32>::create(
                &self.context,
                opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
                k * n,
                b.as_ptr() as *mut std::ffi::c_void,
            )
        }
        .map_err(|e| RusTorchError::gpu(format!("Failed to create buffer B: {:?}", e)))?;

        let c_buffer = unsafe {
            opencl3::memory::Buffer::<f32>::create(
                &self.context,
                opencl3::memory::CL_MEM_WRITE_ONLY,
                m * n,
                std::ptr::null_mut(),
            )
        }
        .map_err(|e| RusTorchError::gpu(format!("Failed to create buffer C: {:?}", e)))?;

        // Get kernel
        let kernel = self.kernels.get(&OpenClKernelType::MatMul).ok_or_else(|| {
            RusTorchError::KernelExecutionError("MatMul kernel not found".to_string())
        })?;

        // Execute kernel
        let global_work_size = [n, m];
        let local_work_size = [16.min(n), 16.min(m)];

        unsafe {
            opencl3::kernel::ExecuteKernel::new(kernel)
                .set_arg(&a_buffer)
                .set_arg(&b_buffer)
                .set_arg(&c_buffer)
                .set_arg(&(m as u32))
                .set_arg(&(n as u32))
                .set_arg(&(k as u32))
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(&self.queue)
        }
        .map_err(|e| {
            RusTorchError::KernelExecutionError(format!("Kernel execution failed: {:?}", e))
        })?;

        // Read result back
        unsafe {
            self.queue
                .enqueue_read_buffer(&c_buffer, opencl3::types::CL_TRUE, 0, c, &[])
                .map_err(|e| {
                    RusTorchError::invalid_params(
                        "matmul",
                        format!("Failed to read result: {:?}", e),
                    )
                })?;
        }

        Ok(())
    }

    /// Perform matrix multiplication using OpenCL with result return
    /// OpenCLを使用して行列乗算を実行し結果を返す
    pub fn matrix_multiply_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n {
            return Err(RusTorchError::InvalidOperation(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        let mut result = vec![0.0f32; m * n];
        self.matmul_f32(a, b, &mut result, m, n, k)?;
        Ok(result)
    }

    /// Execute reduction operation (sum) using OpenCL
    /// OpenCLを使用してリダクション演算（合計）を実行
    pub fn reduce_sum_f32(&self, input: &[f32]) -> RusTorchResult<f32> {
        let size = input.len();
        let local_size = 256;
        let global_size = size.div_ceil(local_size) * local_size;
        let num_groups = global_size / local_size;

        // Create OpenCL buffers
        let input_buffer = unsafe {
            opencl3::memory::Buffer::<f32>::create(
                &self.context,
                opencl3::memory::CL_MEM_READ_ONLY | opencl3::memory::CL_MEM_COPY_HOST_PTR,
                size,
                input.as_ptr() as *mut std::ffi::c_void,
            )
        }
        .map_err(|e| RusTorchError::gpu(format!("Failed to create input buffer: {:?}", e)))?;

        let output_buffer = unsafe {
            opencl3::memory::Buffer::<f32>::create(
                &self.context,
                opencl3::memory::CL_MEM_WRITE_ONLY,
                num_groups,
                std::ptr::null_mut(),
            )
        }
        .map_err(|e| RusTorchError::gpu(format!("Failed to create output buffer: {:?}", e)))?;

        // Get kernel
        let kernel = self
            .kernels
            .get(&OpenClKernelType::Reduction)
            .ok_or_else(|| {
                RusTorchError::KernelExecutionError("Reduction kernel not found".to_string())
            })?;

        // Execute kernel
        let global_work_size = [global_size];
        let local_work_size = [local_size];

        unsafe {
            opencl3::kernel::ExecuteKernel::new(kernel)
                .set_arg(&input_buffer)
                .set_arg(&output_buffer)
                .set_arg(&(size as u32))
                .set_global_work_sizes(&global_work_size)
                .set_local_work_sizes(&local_work_size)
                .enqueue_nd_range(&self.queue)
        }
        .map_err(|e| {
            RusTorchError::KernelExecutionError(format!("Kernel execution failed: {:?}", e))
        })?;

        // Read partial results back
        let mut partial_results = vec![0.0f32; num_groups];
        unsafe {
            self.queue
                .enqueue_read_buffer(
                    &output_buffer,
                    opencl3::types::CL_TRUE,
                    0,
                    &mut partial_results,
                    &[],
                )
                .map_err(|e| {
                    RusTorchError::invalid_params(
                        "matmul",
                        format!("Failed to read partial results: {:?}", e),
                    )
                })?;
        }

        Ok(partial_results.iter().sum())
    }
}

/// Non-OpenCL fallback executor for compatibility
/// 互換性のための非OpenCLフォールバック実行器
#[cfg(not(feature = "opencl"))]
pub struct OpenClKernelExecutor;

#[cfg(not(feature = "opencl"))]
impl OpenClKernelExecutor {
    /// Create a new OpenCL kernel executor (fallback when OpenCL not available)
    /// 新しいOpenCLカーネル実行器を作成（OpenCL利用不可時のフォールバック）
    pub fn new(_device_id: usize) -> RusTorchResult<Self> {
        Err(RusTorchError::backend_unavailable("OpenCL"))
    }

    /// Perform elementwise addition on f32 arrays (fallback)
    /// f32配列の要素ごと加算を実行（フォールバック）
    pub fn elementwise_add_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
    ) -> RusTorchResult<()> {
        Err(RusTorchError::backend_unavailable("OpenCL"))
    }

    /// Perform matrix multiplication on f32 arrays (fallback)
    /// f32配列の行列乗算を実行（フォールバック）
    pub fn matmul_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> RusTorchResult<()> {
        Err(RusTorchError::backend_unavailable("OpenCL"))
    }

    /// Perform reduction sum on f32 array (fallback)
    /// f32配列のリダクション合計を実行（フォールバック）
    pub fn reduce_sum_f32(&self, _input: &[f32]) -> RusTorchResult<f32> {
        Err(RusTorchError::backend_unavailable("OpenCL"))
    }
}

/// Public interface functions for OpenCL operations
/// OpenCL演算のためのパブリックインターフェース関数
///
/// Execute OpenCL matrix multiplication
/// OpenCL行列乗算を実行
pub fn opencl_matmul_f32(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
) -> RusTorchResult<()> {
    #[cfg(feature = "opencl")]
    {
        let executor = OpenClKernelExecutor::new(0)?;
        executor.matmul_f32(_a, _b, _c, _m, _n, _k)
    }
    #[cfg(not(feature = "opencl"))]
    {
        Err(RusTorchError::backend_unavailable("OpenCL"))
    }
}

/// Execute OpenCL element-wise addition
/// OpenCL要素ごと加算を実行
pub fn opencl_elementwise_add_f32(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> RusTorchResult<()> {
    #[cfg(feature = "opencl")]
    {
        let executor = OpenClKernelExecutor::new(0)?;
        executor.elementwise_add_f32(_a, _b, _c)
    }
    #[cfg(not(feature = "opencl"))]
    {
        Err(RusTorchError::backend_unavailable("OpenCL"))
    }
}

/// Execute OpenCL reduction sum
/// OpenCLリダクション合計を実行
pub fn opencl_reduce_sum_f32(_input: &[f32]) -> RusTorchResult<f32> {
    #[cfg(feature = "opencl")]
    {
        let executor = OpenClKernelExecutor::new(0)?;
        executor.reduce_sum_f32(_input)
    }
    #[cfg(not(feature = "opencl"))]
    {
        Err(RusTorchError::backend_unavailable("OpenCL"))
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

//! CUDA kernel implementations for GPU acceleration
//! GPU加速のためのCUDAカーネル実装

use crate::error::{RusTorchError, RusTorchResult};

#[cfg(feature = "cuda")]
use cudarc::{
    cublas::{CudaBlas, Gemm},
    driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};

/// CUDA kernel types
/// CUDAカーネルタイプ
#[derive(Debug, Clone, Copy)]
pub enum CudaKernelType {
    /// Element-wise operations
    /// 要素ごと演算
    ElementWise,
    /// Matrix multiplication
    /// 行列乗算
    MatMul,
    /// Reduction operations
    /// リダクション演算
    Reduction,
    /// Convolution
    /// 畳み込み
    Convolution,
    /// Batch normalization
    /// バッチ正規化
    BatchNorm,
}

/// CUDA kernel parameters
/// CUDAカーネルパラメータ
#[derive(Debug, Clone)]
pub struct CudaKernelParams {
    /// Grid dimensions
    /// グリッド次元
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions
    /// ブロック次元
    pub block_dim: (u32, u32, u32),
    /// Shared memory size
    /// 共有メモリサイズ
    pub shared_mem_size: usize,
    /// Stream ID
    /// ストリームID
    pub stream_id: usize,
}

impl Default for CudaKernelParams {
    fn default() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_size: 0,
            stream_id: 0,
        }
    }
}

/// CUDA memory buffer
/// CUDAメモリバッファ
#[derive(Debug)]
pub struct CudaBuffer<T> {
    /// Device pointer
    /// デバイスポインタ
    pub ptr: *mut T,
    /// Size in elements
    /// 要素数
    pub size: usize,
    /// Device ID
    /// デバイスID
    pub device_id: usize,
}

impl<T> CudaBuffer<T> {
    /// Create a new CUDA buffer
    /// 新しいCUDAバッファを作成
    pub fn new(_size: usize, _device_id: usize) -> RusTorchResult<Self> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;

            let device = CudaDevice::new(device_id).map_err(|e| {
                RusTorchError::InitializationError(format!(
                    "Failed to initialize CUDA device {}: {}",
                    device_id, e
                ))
            })?;

            let ptr = device.alloc_zeros::<T>(size).map_err(|e| {
                RusTorchError::AllocationError(format!("Failed to allocate CUDA memory: {}", e))
            })?;

            Ok(Self {
                ptr: ptr.device_ptr() as *mut T,
                size,
                device_id,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(RusTorchError::UnsupportedDevice(
                "CUDA not available".to_string(),
            ))
        }
    }

    /// Copy data from host to device
    /// ホストからデバイスへデータをコピー
    pub fn copy_from_host(&mut self, host_data: &[T]) -> RusTorchResult<()> {
        if host_data.len() != self.size {
            return Err(RusTorchError::InvalidOperation(
                "Size mismatch in host-to-device copy".to_string(),
            ));
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::{CudaDevice, DevicePtr};

            let device = CudaDevice::new(self.device_id).map_err(|e| {
                RusTorchError::InitializationError(format!("Failed to get CUDA device: {}", e))
            })?;

            let device_ptr = unsafe { DevicePtr::wrap(self.ptr as *mut T, self.size) };
            device.htod_copy(host_data, &device_ptr).map_err(|e| {
                RusTorchError::InvalidOperation(format!("Host-to-device copy failed: {}", e))
            })?;

            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(RusTorchError::UnsupportedDevice(
                "CUDA not available".to_string(),
            ))
        }
    }

    /// Copy data from device to host
    /// デバイスからホストへデータをコピー
    pub fn copy_to_host(&self, host_data: &mut [T]) -> RusTorchResult<()> {
        if host_data.len() != self.size {
            return Err(RusTorchError::InvalidOperation(
                "Size mismatch in device-to-host copy".to_string(),
            ));
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::{CudaDevice, DevicePtr};

            let device = CudaDevice::new(self.device_id).map_err(|e| {
                RusTorchError::InitializationError(format!("Failed to get CUDA device: {}", e))
            })?;

            let device_ptr = unsafe { DevicePtr::wrap(self.ptr as *mut T, self.size) };
            device.dtoh_sync_copy(&device_ptr, host_data).map_err(|e| {
                RusTorchError::InvalidOperation(format!("Device-to-host copy failed: {}", e))
            })?;

            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(RusTorchError::UnsupportedDevice(
                "CUDA not available".to_string(),
            ))
        }
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            if !self.ptr.is_null() {
                use cudarc::driver::{CudaDevice, DevicePtr};

                if let Ok(device) = CudaDevice::new(self.device_id) {
                    let device_ptr = unsafe { DevicePtr::wrap(self.ptr as *mut T, self.size) };
                    let _ = device.synchronize();
                    // DevicePtr will handle deallocation automatically
                }
            }
        }
    }
}

/// CUDA kernel executor for high-performance GPU operations
/// 高性能GPU演算のためのCUDAカーネル実行器
#[cfg(feature = "cuda")]
pub struct CudaKernelExecutor {
    device: CudaDevice,
    cublas: CudaBlas<f32>,
    device_id: usize,
}

#[cfg(feature = "cuda")]
impl CudaKernelExecutor {
    /// Create a new CUDA kernel executor
    /// 新しいCUDAカーネル実行器を作成
    pub fn new(device_id: usize) -> RusTorchResult<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            RusTorchError::InitializationError(format!(
                "Failed to initialize CUDA device {}: {}",
                device_id, e
            ))
        })?;

        let cublas = CudaBlas::new(device.clone()).map_err(|e| {
            RusTorchError::InitializationError(format!("Failed to initialize cuBLAS: {}", e))
        })?;

        Ok(Self {
            device,
            cublas,
            device_id,
        })
    }

    /// Execute matrix multiplication using cuBLAS
    /// cuBLASを使用して行列乗算を実行
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // Allocate device memory
        let a_gpu = self.device.htod_copy(a.to_vec()).map_err(|e| {
            RusTorchError::AllocationError(format!("Failed to copy matrix A to device: {}", e))
        })?;

        let b_gpu = self.device.htod_copy(b.to_vec()).map_err(|e| {
            RusTorchError::AllocationError(format!("Failed to copy matrix B to device: {}", e))
        })?;

        let mut c_gpu = self.device.alloc_zeros::<f32>(m * n).map_err(|e| {
            RusTorchError::AllocationError(format!("Failed to allocate result matrix: {}", e))
        })?;

        // Perform matrix multiplication using cuBLAS
        unsafe {
            self.cublas
                .gemm(
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    m as i32,
                    n as i32,
                    k as i32,
                    &1.0f32,
                    &a_gpu,
                    m as i32,
                    &b_gpu,
                    k as i32,
                    &0.0f32,
                    &mut c_gpu,
                    m as i32,
                )
                .map_err(|e| {
                    RusTorchError::KernelExecutionError(format!("cuBLAS GEMM failed: {}", e))
                })?;
        }

        // Copy result back to host
        self.device.dtoh_sync_copy(&c_gpu, c).map_err(|e| {
            RusTorchError::InvalidOperation(format!("Failed to copy result to host: {}", e))
        })?;

        Ok(())
    }

    /// Execute element-wise addition using custom CUDA kernel
    /// カスタムCUDAカーネルを使用して要素ごと加算を実行
    pub fn elementwise_add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> RusTorchResult<()> {
        let size = a.len();
        if b.len() != size || c.len() != size {
            return Err(RusTorchError::InvalidOperation(
                "Array size mismatch in element-wise addition".to_string(),
            ));
        }

        // CUDA kernel source code
        let kernel_src = r#"
        extern "C" __global__ void elementwise_add_f32(
            const float* a,
            const float* b,
            float* c,
            int n
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        "#;

        // Compile kernel
        let ptx = compile_ptx(kernel_src).map_err(|e| {
            RusTorchError::KernelCompilationError(format!("Failed to compile CUDA kernel: {}", e))
        })?;

        self.device
            .load_ptx(ptx, "elementwise_add", &["elementwise_add_f32"])
            .map_err(|e| {
                RusTorchError::KernelCompilationError(format!("Failed to load PTX: {}", e))
            })?;

        // Allocate device memory
        let a_gpu = self.device.htod_copy(a.to_vec()).map_err(|e| {
            RusTorchError::AllocationError(format!("Failed to copy array A to device: {}", e))
        })?;

        let b_gpu = self.device.htod_copy(b.to_vec()).map_err(|e| {
            RusTorchError::AllocationError(format!("Failed to copy array B to device: {}", e))
        })?;

        let mut c_gpu = self.device.alloc_zeros::<f32>(size).map_err(|e| {
            RusTorchError::AllocationError(format!("Failed to allocate result array: {}", e))
        })?;

        // Launch kernel
        let func = self
            .device
            .get_func("elementwise_add", "elementwise_add_f32")
            .map_err(|e| {
                RusTorchError::KernelExecutionError(format!("Failed to get kernel function: {}", e))
            })?;

        let block_size = 256;
        let grid_size = (size + block_size - 1) / block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(config, (&a_gpu, &b_gpu, &mut c_gpu, size as i32))
                .map_err(|e| {
                    RusTorchError::KernelExecutionError(format!("Kernel launch failed: {}", e))
                })?;
        }

        // Copy result back to host
        self.device.dtoh_sync_copy(&c_gpu, c).map_err(|e| {
            RusTorchError::InvalidOperation(format!("Failed to copy result to host: {}", e))
        })?;

        Ok(())
    }

    /// Execute reduction operation (sum) using CUDA
    /// CUDAを使用してリダクション演算（合計）を実行
    pub fn reduce_sum_f32(&self, input: &[f32]) -> RusTorchResult<f32> {
        let size = input.len();

        // CUDA reduction kernel
        let kernel_src = r#"
        extern "C" __global__ void reduce_sum_f32(
            const float* input,
            float* output,
            int n
        ) {
            extern __shared__ float sdata[];
            
            int tid = threadIdx.x;
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            sdata[tid] = (i < n) ? input[i] : 0.0f;
            __syncthreads();
            
            // Reduction in shared memory
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            
            // Write result for this block to global memory
            if (tid == 0) output[blockIdx.x] = sdata[0];
        }
        "#;

        // Compile kernel
        let ptx = compile_ptx(kernel_src).map_err(|e| {
            RusTorchError::KernelCompilationError(format!(
                "Failed to compile reduction kernel: {}",
                e
            ))
        })?;

        self.device
            .load_ptx(ptx, "reduce_sum", &["reduce_sum_f32"])
            .map_err(|e| {
                RusTorchError::KernelCompilationError(format!(
                    "Failed to load reduction PTX: {}",
                    e
                ))
            })?;

        // Allocate device memory
        let input_gpu = self.device.htod_copy(input.to_vec()).map_err(|e| {
            RusTorchError::AllocationError(format!("Failed to copy input to device: {}", e))
        })?;

        let block_size = 256;
        let grid_size = (size + block_size - 1) / block_size;
        let mut output_gpu = self.device.alloc_zeros::<f32>(grid_size).map_err(|e| {
            RusTorchError::AllocationError(format!("Failed to allocate output array: {}", e))
        })?;

        // Launch kernel
        let func = self
            .device
            .get_func("reduce_sum", "reduce_sum_f32")
            .map_err(|e| {
                RusTorchError::KernelExecutionError(format!(
                    "Failed to get reduction function: {}",
                    e
                ))
            })?;

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: block_size * std::mem::size_of::<f32>(),
        };

        unsafe {
            func.launch(config, (&input_gpu, &mut output_gpu, size as i32))
                .map_err(|e| {
                    RusTorchError::KernelExecutionError(format!(
                        "Reduction kernel launch failed: {}",
                        e
                    ))
                })?;
        }

        // Copy partial results back and sum on CPU
        let mut partial_results = vec![0.0f32; grid_size];
        self.device
            .dtoh_sync_copy(&output_gpu, &mut partial_results)
            .map_err(|e| {
                RusTorchError::InvalidOperation(format!("Failed to copy partial results: {}", e))
            })?;

        Ok(partial_results.iter().sum())
    }
}

/// Non-CUDA fallback executor for compatibility
/// 互換性のための非CUDAフォールバック実行器
#[cfg(not(feature = "cuda"))]
pub struct CudaKernelExecutor {
    _device_id: usize,
}

#[cfg(not(feature = "cuda"))]
impl CudaKernelExecutor {
    /// Create a new CUDA kernel executor for the specified device
    /// 指定されたデバイス用の新しいCUDAカーネル実行器を作成
    pub fn new(_device_id: usize) -> RusTorchResult<Self> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }

    /// Perform matrix multiplication using CUDA
    /// CUDAを使用して行列乗算を実行
    pub fn matmul_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> RusTorchResult<()> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }

    /// Perform element-wise addition using CUDA
    /// CUDAを使用して要素ごとの加算を実行
    pub fn elementwise_add_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
    ) -> RusTorchResult<()> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }

    /// Perform reduction sum using CUDA
    /// CUDAを使用してリダクション合計を実行
    pub fn reduce_sum_f32(&self, _input: &[f32]) -> RusTorchResult<f32> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }
}

/// Public interface functions for CUDA operations
/// CUDA演算のためのパブリックインターフェース関数

/// Execute CUDA matrix multiplication
/// CUDA行列乗算を実行
pub fn cuda_matmul_f32(
    _a: &[f32],
    _b: &[f32],
    _c: &mut [f32],
    _m: usize,
    _n: usize,
    _k: usize,
) -> RusTorchResult<()> {
    #[cfg(feature = "cuda")]
    {
        let executor = CudaKernelExecutor::new(0)?;
        executor.matmul_f32(_a, _b, _c, _m, _n, _k)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }
}

/// Execute CUDA element-wise addition
/// CUDA要素ごと加算を実行
pub fn cuda_elementwise_add_f32(_a: &[f32], _b: &[f32], _c: &mut [f32]) -> RusTorchResult<()> {
    #[cfg(feature = "cuda")]
    {
        let executor = CudaKernelExecutor::new(0)?;
        executor.elementwise_add_f32(_a, _b, _c)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }
}

/// Execute CUDA reduction sum
/// CUDAリダクション合計を実行
pub fn cuda_reduce_sum_f32(_input: &[f32]) -> RusTorchResult<f32> {
    #[cfg(feature = "cuda")]
    {
        let executor = CudaKernelExecutor::new(0)?;
        executor.reduce_sum_f32(_input)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }
}

/// CUDA kernel optimization utilities
/// CUDAカーネル最適化ユーティリティ
pub mod cuda_utils {
    use super::*;

    /// Calculate optimal grid and block dimensions
    /// 最適なグリッドとブロック次元を計算
    pub fn calculate_launch_config(size: usize, max_threads_per_block: usize) -> (usize, usize) {
        let block_size = (max_threads_per_block).min(256);
        let grid_size = (size + block_size - 1) / block_size;
        (grid_size, block_size)
    }

    /// Get device properties
    /// デバイスプロパティを取得
    pub fn get_device_properties(_device_id: usize) -> RusTorchResult<DeviceProperties> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;

            let device = CudaDevice::new(0).map_err(|e| {
                RusTorchError::InitializationError(format!("Failed to get device 0: {}", e))
            })?;

            Ok(DeviceProperties {
                name: "CUDA Device 0".to_string(),
                compute_capability: (7, 5), // Default to common capability
                max_threads_per_block: 1024,
                max_shared_memory: 49152,
                warp_size: 32,
                memory_size: device.total_memory() as usize,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(RusTorchError::UnsupportedDevice(
                "CUDA not available".to_string(),
            ))
        }
    }
}

/// CUDA device properties
/// CUDAデバイスプロパティ
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name
    /// デバイス名
    pub name: String,
    /// Compute capability version (major, minor)
    /// 計算能力バージョン（メジャー、マイナー）
    pub compute_capability: (u32, u32),
    /// Maximum threads per block
    /// ブロックあたりの最大スレッド数
    pub max_threads_per_block: usize,
    /// Maximum shared memory size
    /// 最大共有メモリサイズ
    pub max_shared_memory: usize,
    /// Warp size
    /// ワープサイズ
    pub warp_size: usize,
    /// Total memory size
    /// 総メモリサイズ
    pub memory_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_buffer_creation() {
        let result = CudaBuffer::<f32>::new(1024, 0);
        #[cfg(not(feature = "cuda"))]
        assert!(result.is_err());
    }

    #[test]
    fn test_cuda_kernel_params() {
        let params = CudaKernelParams::default();
        assert_eq!(params.grid_dim, (1, 1, 1));
        assert_eq!(params.block_dim, (256, 1, 1));
        assert_eq!(params.shared_mem_size, 0);
        assert_eq!(params.stream_id, 0);
    }

    #[test]
    fn test_cuda_utils() {
        let (grid_size, block_size) = cuda_utils::calculate_launch_config(1000, 256);
        assert_eq!(block_size, 256);
        assert_eq!(grid_size, 4); // ceil(1000/256) = 4
    }
}

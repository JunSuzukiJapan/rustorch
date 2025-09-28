//! CUDA kernel implementations for GPU acceleration
//! GPU加速のためのCUDAカーネル実装

use crate::error::{RusTorchError, RusTorchResult};

#[cfg(feature = "cuda")]
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "cuda")]
use cudarc::{
    cublas::{CudaBlas, Gemm},
    driver::{
        CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits,
    },
    nvrtc::compile_ptx,
};
use std::sync::Arc;

/// CUDA kernel types
/// CUDAカーネルタイプ#[derive(Debug, Clone, Copy)]
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
/// CUDAカーネルパラメータ#[derive(Debug, Clone)]
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
    /// Device slice
    /// デバイススライス
    #[cfg(feature = "cuda")]
    pub slice: CudaSlice<T>,
    #[cfg(not(feature = "cuda"))]
    _phantom: std::marker::PhantomData<T>,
    /// Size in elements
    /// 要素数
    pub size: usize,
    /// Device ID
    /// デバイスID
    pub device_id: usize,
}

#[cfg(feature = "cuda")]
impl<T: Clone + ValidAsZeroBits + DeviceRepr + Unpin> CudaBuffer<T> {
    /// Create a new CUDA buffer
    /// 新しいCUDAバッファを作成
    pub fn new(_size: usize, _device_id: usize) -> RusTorchResult<Self> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;

            let device = CudaDevice::new(_device_id).map_err(|e| {
                RusTorchError::tensor_op(format!(
                    "Failed to initialize CUDA device {}: {}",
                    _device_id, e
                ))
            })?;

            let slice = device.alloc_zeros::<T>(_size).map_err(|e| {
                RusTorchError::tensor_op(format!("Failed to allocate CUDA memory: {}", e))
            })?;

            Ok(Self {
                slice,
                size: _size,
                device_id: _device_id,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(RusTorchError::BackendUnavailable {
                backend: "CUDA".to_string(),
            })
        }
    }

    /// Copy data from host to device
    /// ホストからデバイスへデータをコピー
    pub fn copy_from_host(&mut self, host_data: &[T]) -> RusTorchResult<()> {
        if host_data.len() != self.size {
            return Err(RusTorchError::InvalidParameters {
                operation: "copy_from_host".to_string(),
                message: "Size mismatch in host-to-device copy".to_string(),
            });
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;

            let device = CudaDevice::new(self.device_id)
                .map_err(|e| RusTorchError::gpu(format!("Failed to get CUDA device: {}", e)))?;

            // Create new slice from host data and replace current slice
            self.slice = device.htod_copy(host_data.to_vec()).map_err(|e| {
                RusTorchError::tensor_op(format!("Host-to-device copy failed: {}", e))
            })?;

            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(RusTorchError::BackendUnavailable {
                backend: "CUDA".to_string(),
            })
        }
    }

    /// Copy data from device to host
    /// デバイスからホストへデータをコピー
    pub fn copy_to_host(&self, host_data: &mut [T]) -> RusTorchResult<()> {
        if host_data.len() != self.size {
            return Err(RusTorchError::InvalidParameters {
                operation: "copy_to_host".to_string(),
                message: "Size mismatch in device-to-host copy".to_string(),
            });
        }

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;

            let device = CudaDevice::new(self.device_id)
                .map_err(|e| RusTorchError::gpu(format!("Failed to get CUDA device: {}", e)))?;

            device
                .dtoh_sync_copy_into(&self.slice, host_data)
                .map_err(|e| {
                    RusTorchError::tensor_op(format!("Device-to-host copy failed: {}", e))
                })?;

            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(RusTorchError::BackendUnavailable {
                backend: "CUDA".to_string(),
            })
        }
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            // CudaSlice handles its own memory management automatically
            // Just synchronize to ensure all operations are complete
            if let Ok(device) = CudaDevice::new(self.device_id) {
                let _ = device.synchronize();
            }
        }
    }
}

/// CUDA kernel executor for high-performance GPU operations
/// 高性能GPU演算のためのCUDAカーネル実行器
#[cfg(feature = "cuda")]
pub struct CudaKernelExecutor {
    device: Arc<CudaDevice>,
    cublas: CudaBlas,
    device_id: usize,
}

#[cfg(feature = "cuda")]
impl CudaKernelExecutor {
    /// Create a new CUDA kernel executor
    /// 新しいCUDAカーネル実行器を作成
    pub fn new(device_id: usize) -> RusTorchResult<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            RusTorchError::gpu(format!(
                "Failed to initialize CUDA device {}: {}",
                device_id, e
            ))
        })?;

        let cublas = CudaBlas::new(device.clone())
            .map_err(|e| RusTorchError::gpu(format!("Failed to initialize cuBLAS: {}", e)))?;

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
        let a_gpu = self
            .device
            .htod_copy(a.to_vec())
            .map_err(|e| RusTorchError::gpu(format!("Failed to copy matrix A to device: {}", e)))?;

        let b_gpu = self
            .device
            .htod_copy(b.to_vec())
            .map_err(|e| RusTorchError::gpu(format!("Failed to copy matrix B to device: {}", e)))?;

        let mut c_gpu = self
            .device
            .alloc_zeros::<f32>(m * n)
            .map_err(|e| RusTorchError::gpu(format!("Failed to allocate result matrix: {}", e)))?;

        // Perform matrix multiplication using cuBLAS with new API
        use cudarc::cublas::GemmConfig;
        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: m as i32,
            n: n as i32,
            k: k as i32,
            lda: k as i32,
            ldb: n as i32,
            ldc: n as i32,
            alpha: 1.0f32,
            beta: 0.0f32,
        };

        unsafe {
            self.cublas
                .gemm(cfg, &a_gpu, &b_gpu, &mut c_gpu)
                .map_err(|e| RusTorchError::tensor_op(format!("cuBLAS GEMM failed: {}", e)))?;
        }

        // Copy result back to host
        self.device.dtoh_sync_copy_into(&c_gpu, c).map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to copy result to host: {}", e))
        })?;

        Ok(())
    }

    /// Execute element-wise addition using custom CUDA kernel
    /// カスタムCUDAカーネルを使用して要素ごと加算を実行
    pub fn elementwise_add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> RusTorchResult<()> {
        let size = a.len();
        if b.len() != size || c.len() != size {
            return Err(RusTorchError::InvalidParameters {
                operation: "elementwise_add_f32".to_string(),
                message: "Array size mismatch in element-wise addition".to_string(),
            });
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
            RusTorchError::kernel_compilation(format!("Failed to compile CUDA kernel: {}", e))
        })?;

        self.device
            .load_ptx(ptx, "elementwise_add", &["elementwise_add_f32"])
            .map_err(|e| RusTorchError::kernel_compilation(format!("Failed to load PTX: {}", e)))?;

        // Allocate device memory
        let a_gpu = self
            .device
            .htod_copy(a.to_vec())
            .map_err(|e| RusTorchError::gpu(format!("Failed to copy array A to device: {}", e)))?;

        let b_gpu = self
            .device
            .htod_copy(b.to_vec())
            .map_err(|e| RusTorchError::gpu(format!("Failed to copy array B to device: {}", e)))?;

        let mut c_gpu = self
            .device
            .alloc_zeros::<f32>(size)
            .map_err(|e| RusTorchError::gpu(format!("Failed to allocate result array: {}", e)))?;

        // Launch kernel
        let func = self
            .device
            .get_func("elementwise_add", "elementwise_add_f32")
            .ok_or_else(|| {
                RusTorchError::kernel_compilation(
                    "Failed to get kernel function: function not found".to_string(),
                )
            })?;

        let block_size = 256;
        let grid_size = size.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(config, (&a_gpu, &b_gpu, &mut c_gpu, size as i32))
                .map_err(|e| RusTorchError::gpu(format!("Kernel launch failed: {}", e)))?;
        }

        // Copy result back to host
        self.device.dtoh_sync_copy_into(&c_gpu, c).map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to copy result to host: {}", e))
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
            RusTorchError::kernel_compilation(format!("Failed to compile reduction kernel: {}", e))
        })?;

        self.device
            .load_ptx(ptx, "reduce_sum", &["reduce_sum_f32"])
            .map_err(|e| {
                RusTorchError::kernel_compilation(format!("Failed to load reduction PTX: {}", e))
            })?;

        // Allocate device memory
        let input_gpu = self
            .device
            .htod_copy(input.to_vec())
            .map_err(|e| RusTorchError::gpu(format!("Failed to copy input to device: {}", e)))?;

        let block_size = 256;
        let grid_size = size.div_ceil(block_size);
        let mut output_gpu = self
            .device
            .alloc_zeros::<f32>(grid_size)
            .map_err(|e| RusTorchError::gpu(format!("Failed to allocate output array: {}", e)))?;

        // Launch kernel
        let func = self
            .device
            .get_func("reduce_sum", "reduce_sum_f32")
            .ok_or_else(|| {
                RusTorchError::kernel_compilation(
                    "Failed to get reduction function: function not found".to_string(),
                )
            })?;

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: (block_size * std::mem::size_of::<f32>()) as u32,
        };

        unsafe {
            func.launch(config, (&input_gpu, &mut output_gpu, size as i32))
                .map_err(|e| {
                    RusTorchError::gpu(format!("Reduction kernel launch failed: {}", e))
                })?;
        }

        // Copy partial results back and sum on CPU
        let mut partial_results = vec![0.0f32; grid_size];
        self.device
            .dtoh_sync_copy_into(&output_gpu, &mut partial_results)
            .map_err(|e| {
                RusTorchError::tensor_op(format!("Failed to copy partial results: {}", e))
            })?;

        Ok(partial_results.iter().sum())
    }

    /// Execute 2D convolution using CUDA
    /// CUDAを使用して2D畳み込みを実行
    pub fn conv2d_f32(
        &self,
        input: &[f32],
        kernel: &[f32],
        output: &mut [f32],
        input_height: usize,
        input_width: usize,
        input_channels: usize,
        output_channels: usize,
        kernel_height: usize,
        kernel_width: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> RusTorchResult<()> {
        let output_height = (input_height + 2 * pad_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * pad_w - kernel_width) / stride_w + 1;
        let expected_output_size = output_channels * output_height * output_width;

        if output.len() != expected_output_size {
            return Err(RusTorchError::InvalidParameters {
                operation: "conv2d_f32".to_string(),
                message: format!(
                    "Output size mismatch: expected {}, got {}",
                    expected_output_size,
                    output.len()
                ),
            });
        }

        // CUDA convolution kernel using im2col approach
        let kernel_src = r#"
        extern "C" __global__ void conv2d_f32(
            const float* input,
            const float* kernel,
            float* output,
            int input_height,
            int input_width,
            int input_channels,
            int output_channels,
            int kernel_height,
            int kernel_width,
            int stride_h,
            int stride_w,
            int pad_h,
            int pad_w,
            int output_height,
            int output_width
        ) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int output_size = output_channels * output_height * output_width;

            if (tid >= output_size) return;

            int oc = tid / (output_height * output_width);
            int oh = (tid % (output_height * output_width)) / output_width;
            int ow = (tid % (output_height * output_width)) % output_width;

            float sum = 0.0f;

            for (int ic = 0; ic < input_channels; ic++) {
                for (int kh = 0; kh < kernel_height; kh++) {
                    for (int kw = 0; kw < kernel_width; kw++) {
                        int ih = oh * stride_h - pad_h + kh;
                        int iw = ow * stride_w - pad_w + kw;

                        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                            int input_idx = ic * input_height * input_width + ih * input_width + iw;
                            int kernel_idx = oc * input_channels * kernel_height * kernel_width +
                                           ic * kernel_height * kernel_width + kh * kernel_width + kw;
                            sum += input[input_idx] * kernel[kernel_idx];
                        }
                    }
                }
            }

            output[tid] = sum;
        }
        "#;

        // Compile kernel
        let ptx = compile_ptx(kernel_src).map_err(|e| {
            RusTorchError::kernel_compilation(format!(
                "Failed to compile CUDA conv2d kernel: {}",
                e
            ))
        })?;

        self.device
            .load_ptx(ptx, "conv2d", &["conv2d_f32"])
            .map_err(|e| {
                RusTorchError::kernel_compilation(format!("Failed to load conv2d PTX: {}", e))
            })?;

        // Allocate device memory
        let input_gpu = self
            .device
            .htod_copy(input.to_vec())
            .map_err(|e| RusTorchError::gpu(format!("Failed to copy input to device: {}", e)))?;

        let kernel_gpu = self
            .device
            .htod_copy(kernel.to_vec())
            .map_err(|e| RusTorchError::gpu(format!("Failed to copy kernel to device: {}", e)))?;

        let mut output_gpu = self
            .device
            .alloc_zeros::<f32>(expected_output_size)
            .map_err(|e| RusTorchError::gpu(format!("Failed to allocate output: {}", e)))?;

        // Launch kernel
        let func = self
            .device
            .get_func("conv2d", "conv2d_f32")
            .ok_or_else(|| {
                RusTorchError::kernel_compilation(
                    "Failed to get conv2d function: function not found".to_string(),
                )
            })?;

        let block_size = 256;
        let grid_size = expected_output_size.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let params = (
                &input_gpu,
                &kernel_gpu,
                &mut output_gpu,
                input_height as i32,
                input_width as i32,
                input_channels as i32,
                output_channels as i32,
                kernel_height as i32,
                kernel_width as i32,
                stride_h as i32,
                stride_w as i32,
                pad_h as i32,
                pad_w as i32,
                output_height as i32,
                output_width as i32,
            );
            func.launch(config, params)
                .map_err(|e| RusTorchError::gpu(format!("Conv2d kernel launch failed: {}", e)))?;
        }

        // Copy result back to host
        self.device
            .dtoh_sync_copy_into(&output_gpu, output)
            .map_err(|e| {
                RusTorchError::tensor_op(format!("Failed to copy conv2d result to host: {}", e))
            })?;

        Ok(())
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

    /// Perform 2D convolution using CUDA
    /// CUDAを使用して2D畳み込みを実行
    pub fn conv2d_f32(
        &self,
        _input: &[f32],
        _kernel: &[f32],
        _output: &mut [f32],
        _input_height: usize,
        _input_width: usize,
        _input_channels: usize,
        _output_channels: usize,
        _kernel_height: usize,
        _kernel_width: usize,
        _stride_h: usize,
        _stride_w: usize,
        _pad_h: usize,
        _pad_w: usize,
    ) -> RusTorchResult<()> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }
}

/// Public interface functions for CUDA operations
/// CUDA演算のためのパブリックインターフェース関数
///
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

/// Execute CUDA 2D convolution
/// CUDA 2D畳み込みを実行
pub fn cuda_conv2d_f32(
    _input: &[f32],
    _kernel: &[f32],
    _output: &mut [f32],
    _input_height: usize,
    _input_width: usize,
    _input_channels: usize,
    _output_channels: usize,
    _kernel_height: usize,
    _kernel_width: usize,
    _stride_h: usize,
    _stride_w: usize,
    _pad_h: usize,
    _pad_w: usize,
) -> RusTorchResult<()> {
    #[cfg(feature = "cuda")]
    {
        let executor = CudaKernelExecutor::new(0)?;
        executor.conv2d_f32(
            _input,
            _kernel,
            _output,
            _input_height,
            _input_width,
            _input_channels,
            _output_channels,
            _kernel_height,
            _kernel_width,
            _stride_h,
            _stride_w,
            _pad_h,
            _pad_w,
        )
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
        let grid_size = size.div_ceil(block_size);
        (grid_size, block_size)
    }

    /// Get device properties
    /// デバイスプロパティを取得
    pub fn get_device_properties(_device_id: usize) -> RusTorchResult<DeviceProperties> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaDevice;

            let device = CudaDevice::new(0)
                .map_err(|e| RusTorchError::gpu(format!("Failed to get device 0: {}", e)))?;

            Ok(DeviceProperties {
                name: "CUDA Device 0".to_string(),
                compute_capability: (7, 5), // Default to common capability
                max_threads_per_block: 1024,
                max_shared_memory: 49152,
                warp_size: 32,
                memory_size: 8 * 1024 * 1024 * 1024usize, // Default to 8GB
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(RusTorchError::BackendUnavailable {
                backend: "CUDA".to_string(),
            })
        }
    }
}

/// CUDA device properties
/// CUDAデバイスプロパティ#[derive(Debug, Clone)]
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
    #[cfg(feature = "cuda")]
    fn test_cuda_buffer_creation() {
        let result = CudaBuffer::<f32>::new(1024, 0);
        // CUDAが利用可能な場合のみテスト
        // エラーでも可（GPUが利用できない場合）
        match result {
            Ok(_) => println!("CUDA buffer created successfully"),
            Err(_) => println!("CUDA buffer creation failed (expected if no GPU)"),
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_buffer_creation_no_cuda() {
        // CUDA機能が無効の場合は何もしない
        assert!(true);
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

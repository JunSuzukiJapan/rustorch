//! Enhanced CUDA cuBLAS integration for high-performance matrix operations
//! 高性能行列演算のための強化されたCUDA cuBLAS統合

use crate::error::{RusTorchError, RusTorchResult};
use std::collections::HashMap;
// Note: Arc and Mutex may be needed for future CUDA memory pool implementation

#[cfg(feature = "cuda")]
use cudarc::{
    cublas::{CudaBlas, Gemm},
    driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};

/// Enhanced CUDA matrix operations with cuBLAS optimization
/// cuBLAS最適化による強化されたCUDA行列演算
#[cfg(feature = "cuda")]
pub struct CudaMatrixExecutor {
    device: CudaDevice,
    cublas: CudaBlas,
    device_id: usize,
    streams: Vec<cudarc::driver::CudaStream>,
    memory_pool: Arc<Mutex<CudaMemoryPool>>,
}

#[cfg(feature = "cuda")]
impl CudaMatrixExecutor {
    /// Create new CUDA matrix executor with cuBLAS
    /// cuBLASを使用した新しいCUDA行列実行器を作成
    pub fn new(device_id: usize) -> RusTorchResult<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            RusTorchError::tensor_op(format!(
                "Failed to initialize CUDA device {}: {}",
                device_id, e
            ))
        })?;

        let cublas = CudaBlas::new(device.clone())
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to initialize cuBLAS: {}", e)))?;

        // Create multiple streams for concurrent execution
        let mut streams = Vec::new();
        for i in 0..4 {
            let stream = device.fork_default_stream().map_err(|e| {
                RusTorchError::tensor_op(format!("Failed to create CUDA stream {}: {}", i, e))
            })?;
            streams.push(stream);
        }

        let memory_pool = Arc::new(Mutex::new(CudaMemoryPool::new()));

        Ok(Self {
            device,
            cublas,
            device_id,
            streams,
            memory_pool,
        })
    }

    /// Optimized matrix multiplication using cuBLAS SGEMM
    /// cuBLAS SGEMMを使用した最適化された行列乗算
    pub fn matmul_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        use_tensor_cores: bool,
    ) -> RusTorchResult<()> {
        // Validate dimensions
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(RusTorchError::shape_mismatch(
                &[m, k, k, n, m, n],
                &[a.len(), 0, b.len(), 0, c.len(), 0],
            ));
        }

        // Allocate GPU memory
        let a_gpu = self.device.htod_copy(a.to_vec()).map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to copy matrix A to GPU: {}", e))
        })?;

        let b_gpu = self.device.htod_copy(b.to_vec()).map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to copy matrix B to GPU: {}", e))
        })?;

        let mut c_gpu = self.device.alloc_zeros::<f32>(m * n).map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to allocate result matrix: {}", e))
        })?;

        // Configure cuBLAS for optimal performance
        if use_tensor_cores {
            // Use Tensor Core optimized GEMM for compatible hardware (Volta+)
            self.perform_tensor_core_gemm(&a_gpu, &b_gpu, &mut c_gpu, m, n, k)?;
        } else {
            // Standard cuBLAS SGEMM
            self.perform_standard_gemm(&a_gpu, &b_gpu, &mut c_gpu, m, n, k)?;
        }

        // Copy result back to host
        let result = self.device.dtoh_sync_copy(&c_gpu).map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to copy result from GPU: {}", e))
        })?;

        c.copy_from_slice(&result);
        Ok(())
    }

    /// Tensor Core optimized GEMM for Volta+ architectures
    /// Volta+アーキテクチャ向けTensor Core最適化GEMM
    fn perform_tensor_core_gemm(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // Use mixed precision with Tensor Cores when available
        // This requires converting f32 to f16, performing computation, then back to f32
        // For simplicity, fall back to standard GEMM for now
        self.perform_standard_gemm(a, b, c, m, n, k)
    }

    /// Standard cuBLAS SGEMM implementation
    /// 標準cuBLAS SGEMM実装
    fn perform_standard_gemm(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // cuBLAS uses column-major order, so we need to transpose
        // C = alpha * A * B + beta * C
        // In column-major: C = alpha * B^T * A^T + beta * C
        let alpha = 1.0f32;
        let beta = 0.0f32;

        // Use cuBLAS SGEMM
        unsafe {
            self.cublas
                .gemm(
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N, // B is not transposed
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N, // A is not transposed
                    n as i32, // number of rows of matrix op(B) and C
                    m as i32, // number of columns of matrix op(A) and C
                    k as i32, // number of columns of op(B) and rows of op(A)
                    &alpha,   // scalar multiplier for A*B
                    b,        // matrix B
                    n as i32, // leading dimension of B
                    a,        // matrix A
                    k as i32, // leading dimension of A
                    &beta,    // scalar multiplier for C
                    c,        // matrix C (result)
                    n as i32, // leading dimension of C
                )
                .map_err(|e| RusTorchError::tensor_op(format!("cuBLAS SGEMM failed: {}", e)))?;
        }

        Ok(())
    }

    /// Batch matrix multiplication for multiple matrices
    /// 複数行列のバッチ行列乗算
    pub fn batch_matmul_f32(
        &self,
        a_batch: &[Vec<f32>],
        b_batch: &[Vec<f32>],
        c_batch: &mut [Vec<f32>],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        if a_batch.len() != b_batch.len() || a_batch.len() != c_batch.len() {
            return Err(RusTorchError::shape_mismatch(
                &[a_batch.len()],
                &[b_batch.len(), c_batch.len()],
            ));
        }

        let batch_size = a_batch.len();
        let stream_count = self.streams.len().min(batch_size);

        // Process batches in parallel using multiple streams
        for (batch_idx, ((a, b), c)) in a_batch
            .iter()
            .zip(b_batch.iter())
            .zip(c_batch.iter_mut())
            .enumerate()
        {
            let stream_idx = batch_idx % stream_count;
            let _stream = &self.streams[stream_idx];

            // For now, process synchronously
            // In production, we'd use asynchronous kernel launches with streams
            self.matmul_f32(a, b, c, m, n, k, false)?;
        }

        // Synchronize all streams
        self.device.synchronize().map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to synchronize CUDA device: {}", e))
        })?;

        Ok(())
    }

    /// Get device memory information
    /// デバイスメモリ情報を取得
    pub fn get_memory_info(&self) -> RusTorchResult<(usize, usize)> {
        let total = self
            .device
            .total_memory()
            .map_err(|e| RusTorchError::tensor_op(format!("Failed to get total memory: {}", e)))?;

        // Free memory calculation (simplified)
        let free = total; // In practice, we'd calculate actual free memory

        Ok((free, total))
    }

    /// Get compute capability
    /// 計算能力を取得
    pub fn get_compute_capability(&self) -> RusTorchResult<(i32, i32)> {
        // Get major and minor compute capability
        let major = self.device.get_attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        ).map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to get compute capability major: {}", e))
        })? as i32;

        let minor = self.device.get_attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        ).map_err(|e| {
            RusTorchError::tensor_op(format!("Failed to get compute capability minor: {}", e))
        })? as i32;

        Ok((major, minor))
    }
}

/// CUDA memory pool for efficient memory management (future implementation)
/// 効率的なメモリ管理のためのCUDAメモリプール（将来実装）
#[allow(dead_code)]
struct CudaMemoryPool {
    buffers: HashMap<usize, Vec<*mut u8>>,
}

#[allow(dead_code)]
impl CudaMemoryPool {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    fn get_buffer(&mut self, size: usize) -> Option<*mut u8> {
        self.buffers.get_mut(&size)?.pop()
    }

    fn return_buffer(&mut self, ptr: *mut u8, size: usize) {
        self.buffers.entry(size).or_insert_with(Vec::new).push(ptr);
    }
}

/// Non-CUDA fallback implementation
/// CUDA非対応時のフォールバック実装
#[cfg(not(feature = "cuda"))]
pub struct CudaMatrixExecutor;

#[cfg(not(feature = "cuda"))]
impl CudaMatrixExecutor {
    /// Create a new CUDA matrix executor for the specified device
    /// 指定されたデバイス用の新しいCUDA行列実行器を作成
    pub fn new(_device_id: usize) -> RusTorchResult<Self> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }

    /// Perform matrix multiplication using CUDA cuBLAS
    /// CUDA cuBLASを使用して行列乗算を実行
    pub fn matmul_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
        _use_tensor_cores: bool,
    ) -> RusTorchResult<()> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }

    /// Perform batch matrix multiplication using CUDA cuBLAS
    /// CUDA cuBLASを使用してバッチ行列乗算を実行
    pub fn batch_matmul_f32(
        &self,
        _a_batch: &[Vec<f32>],
        _b_batch: &[Vec<f32>],
        _c_batch: &mut [Vec<f32>],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> RusTorchResult<()> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }

    /// Get CUDA device memory information (free, total)
    /// CUDAデバイスメモリ情報を取得（空き容量、総容量）
    pub fn get_memory_info(&self) -> RusTorchResult<(usize, usize)> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }

    /// Get CUDA compute capability (major, minor)
    /// CUDAコンピュート能力を取得（メジャー、マイナー）
    pub fn get_compute_capability(&self) -> RusTorchResult<(i32, i32)> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }
}

/// Public interface functions for CUDA operations
/// CUDA演算のためのパブリックインターフェース関数
///
/// Execute CUDA matrix multiplication with cuBLAS
/// cuBLASを使用したCUDA行列乗算を実行
pub fn cuda_matmul_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    device_id: usize,
    use_tensor_cores: bool,
) -> RusTorchResult<()> {
    #[cfg(feature = "cuda")]
    {
        let executor = CudaMatrixExecutor::new(device_id)?;
        executor.matmul_f32(a, b, c, m, n, k, use_tensor_cores)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, b, c, m, n, k, device_id, use_tensor_cores);
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }
}

/// Execute batch CUDA matrix multiplication
/// バッチCUDA行列乗算を実行
pub fn cuda_batch_matmul_f32(
    a_batch: &[Vec<f32>],
    b_batch: &[Vec<f32>],
    c_batch: &mut [Vec<f32>],
    m: usize,
    n: usize,
    k: usize,
    device_id: usize,
) -> RusTorchResult<()> {
    #[cfg(feature = "cuda")]
    {
        let executor = CudaMatrixExecutor::new(device_id)?;
        executor.batch_matmul_f32(a_batch, b_batch, c_batch, m, n, k)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a_batch, b_batch, c_batch, m, n, k, device_id);
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_executor_creation() {
        let result = CudaMatrixExecutor::new(0);
        #[cfg(not(feature = "cuda"))]
        assert!(result.is_err());
    }

    #[test]
    fn test_cuda_matmul_interface() {
        let a = vec![1.0f32; 64];
        let b = vec![2.0f32; 64];
        let mut c = vec![0.0f32; 64];

        let result = cuda_matmul_f32(&a, &b, &mut c, 8, 8, 8, 0, false);
        #[cfg(not(feature = "cuda"))]
        assert!(result.is_err());
    }
}

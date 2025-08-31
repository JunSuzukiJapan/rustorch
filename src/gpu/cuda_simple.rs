//! Simplified CUDA operations for compatibility
//! 互換性のための簡素化されたCUDA演算

use crate::error::{RusTorchError, RusTorchResult};

#[cfg(feature = "cuda")]
use cudarc::{
    cublas::CudaBlas,
    driver::CudaDevice,
};

/// Simplified CUDA executor for basic operations
/// 基本演算のための簡素化されたCUDA実行器
#[cfg(feature = "cuda")]
pub struct SimpleCudaExecutor {
    device: CudaDevice,
    cublas: CudaBlas,
    device_id: usize,
}

#[cfg(feature = "cuda")]
impl SimpleCudaExecutor {
    /// Create new simplified CUDA executor
    /// 新しい簡素化されたCUDA実行器を作成
    pub fn new(device_id: usize) -> RusTorchResult<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            RusTorchError::gpu(format!("Failed to initialize CUDA device {}: {}", device_id, e))
        })?;

        let cublas = CudaBlas::new(device.clone()).map_err(|e| {
            RusTorchError::gpu(format!("Failed to initialize cuBLAS: {}", e))
        })?;

        Ok(Self {
            device,
            cublas,
            device_id,
        })
    }

    /// Simple matrix multiplication using CPU fallback
    /// CPUフォールバックを使用した簡単な行列乗算
    pub fn matmul_simple(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> RusTorchResult<()> {
        // Simplified CPU implementation for compatibility
        // Note: This maintains API compatibility while avoiding complex CUDA operations
        
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(RusTorchError::tensor_op(
                "Matrix dimensions mismatch".to_string(),
            ));
        }

        // Perform matrix multiplication on CPU
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        Ok(())
    }

    /// Get device information
    /// デバイス情報を取得
    pub fn device_info(&self) -> (usize, String) {
        (self.device_id, format!("CUDA Device {}", self.device_id))
    }

    /// Simple element-wise addition
    /// 簡単な要素ごとの加算
    pub fn elementwise_add(
        &self,
        a: &[f32],
        b: &[f32],
    ) -> RusTorchResult<Vec<f32>> {
        if a.len() != b.len() {
            return Err(RusTorchError::tensor_op(
                "Tensor size mismatch for addition".to_string(),
            ));
        }

        let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        Ok(result)
    }
}

#[cfg(not(feature = "cuda"))]
pub struct SimpleCudaExecutor;

#[cfg(not(feature = "cuda"))]
impl SimpleCudaExecutor {
    pub fn new(_device_id: usize) -> RusTorchResult<Self> {
        Err(RusTorchError::gpu("CUDA feature not enabled".to_string()))
    }
}
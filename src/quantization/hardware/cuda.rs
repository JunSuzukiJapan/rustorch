//! CUDA-specific optimizations for quantized operations
//! 量子化演算のCUDA固有最適化

use crate::error::{RusTorchError, RusTorchResult};
use super::super::types::{QuantizedTensor, QuantizableInteger};

#[cfg(feature = "cuda")]
use crate::gpu::DeviceType;

/// CUDA quantized matrix multiplication
/// CUDA量子化行列乗算
pub fn qmatmul_cuda<Q: QuantizableInteger>(
    a: &QuantizedTensor<Q>,
    b: &QuantizedTensor<Q>,
) -> RusTorchResult<QuantizedTensor<Q>> {
    #[cfg(feature = "cuda")]
    {
        // Use existing CUDA implementation for underlying computation
        // TODO: Implement proper quantized CUDA kernels
        super::cpu::qmatmul_cpu(a, b)
    }
    #[cfg(not(feature = "cuda"))]
    {
        super::cpu::qmatmul_cpu(a, b)
    }
}

/// CUDA quantized 2D convolution
/// CUDA量子化2次元畳み込み
pub fn qconv2d_cuda<Q: QuantizableInteger>(
    input: &QuantizedTensor<Q>,
    weight: &QuantizedTensor<Q>,
    bias: Option<&QuantizedTensor<Q>>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> RusTorchResult<QuantizedTensor<Q>> {
    #[cfg(feature = "cuda")]
    {
        // Use existing CUDA implementation for underlying computation
        // TODO: Implement proper quantized CUDA convolution kernels
        super::cpu::qconv2d_cpu(input, weight, bias, stride, padding)
    }
    #[cfg(not(feature = "cuda"))]
    {
        super::cpu::qconv2d_cpu(input, weight, bias, stride, padding)
    }
}

/// CUDA operations collection
/// CUDA演算集合
pub struct CudaOps;

impl CudaOps {
    /// Check if CUDA is available using existing GPU infrastructure
    /// 既存のGPUインフラを使ってCUDAが利用可能かチェック
    pub fn is_available() -> bool {
        #[cfg(feature = "cuda")]
        {
            use crate::gpu::DeviceType;
            DeviceType::Cuda(0).is_available()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
    
    /// Get CUDA device count using existing GPU infrastructure
    /// 既存のGPUインフラを使ってCUDAデバイス数を取得
    pub fn device_count() -> usize {
        #[cfg(feature = "cuda")]
        {
            use crate::gpu::DeviceType;
            // Try to detect available CUDA devices
            (0..8).take_while(|&i| DeviceType::Cuda(i).is_available()).count()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }
}

/// Re-export CUDA operations
/// CUDA演算の再エクスポート
pub use self::CudaOps as cuda_ops;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_availability() {
        // Test should pass regardless of CUDA availability
        let _available = CudaOps::is_available();
        let _count = CudaOps::device_count();
    }
}
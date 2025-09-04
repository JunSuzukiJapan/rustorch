//! Hardware-specific optimizations for quantized operations
//! 量子化演算のハードウェア固有最適化

use crate::error::RusTorchResult;
use super::types::{QuantizedTensor, QuantizableInteger};

/// Hardware-specific quantized operations
/// ハードウェア固有量子化演算
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

// Metal support placeholder - implement when needed
// #[cfg(target_os = "macos")]
// pub mod metal;

/// Optimized operations dispatcher
/// 最適化演算ディスパッチャー
pub struct OptimizedOps;

impl OptimizedOps {
    /// Optimized quantized matrix multiplication
    /// 最適化量子化行列乗算
    pub fn qmatmul<Q: QuantizableInteger>(
        a: &QuantizedTensor<Q>,
        b: &QuantizedTensor<Q>,
    ) -> RusTorchResult<QuantizedTensor<Q>> {
        #[cfg(feature = "cuda")]
        {
            if a.device.is_cuda() {
                return cuda::qmatmul_cuda(a, b);
            }
        }
        
        // Metal support disabled for now
        // #[cfg(target_os = "macos")]
        // {
        //     if a.device.is_metal() {
        //         return metal::qmatmul_metal(a, b);
        //     }
        // }
        
        // Default to CPU implementation
        cpu::qmatmul_cpu(a, b)
    }
    
    /// Optimized quantized convolution
    /// 最適化量子化畳み込み
    pub fn qconv2d<Q: QuantizableInteger>(
        input: &QuantizedTensor<Q>,
        weight: &QuantizedTensor<Q>,
        bias: Option<&QuantizedTensor<Q>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> RusTorchResult<QuantizedTensor<Q>> {
        #[cfg(feature = "cuda")]
        {
            if input.device.is_cuda() {
                return cuda::qconv2d_cuda(input, weight, bias, stride, padding);
            }
        }
        
        // Metal support disabled for now
        // #[cfg(target_os = "macos")]
        // {
        //     if input.device.is_metal() {
        //         return metal::qconv2d_metal(input, weight, bias, stride, padding);
        //     }
        // }
        
        cpu::qconv2d_cpu(input, weight, bias, stride, padding)
    }
}

/// Re-export optimized operations
/// 最適化演算の再エクスポート
pub use cpu::optimized_ops;

#[cfg(feature = "cuda")]
pub use cuda::cuda_ops;

// Metal ops disabled for now
// #[cfg(target_os = "macos")]
// pub use metal::metal_ops;
//! Quantization support for RusTorch - Phase 11 Implementation
//! RusTorch用量子化サポート - フェーズ11実装
//!
//! This module provides comprehensive quantization support for deep learning models,
//! enabling efficient inference and training with reduced precision arithmetic.
//!
//! このモジュールは深層学習モデルの包括的な量子化サポートを提供し、
//! 精度を下げた算術演算による効率的な推論と学習を可能にします。
//!
//! ## Key Features
//!
//! ### Dynamic Quantization
//! - Runtime quantization of weights and activations
//! - Automatic calibration using statistical observers
//! - Per-tensor and per-channel quantization schemes
//!
//! ### Static Quantization  
//! - Pre-calibrated quantization parameters
//! - Optimal for deployment scenarios
//! - Hardware-accelerated operations
//!
//! ### Quantization-Aware Training (QAT)
//! - Training with quantization simulation
//! - Straight-through estimators for gradients
//! - Fine-tuning of quantized models
//!
//! ### Hardware Optimization
//! - CPU SIMD optimizations for quantized operations
//! - CUDA kernels for GPU acceleration
//! - Metal Performance Shaders for Apple Silicon
//!
//! ## Quantization Schemes
//!
//! ### Symmetric Quantization
//! ```
//! quantized = round(fp32_value / scale) + zero_point
//! dequantized = (quantized - zero_point) * scale
//! ```
//!
//! ### Asymmetric Quantization
//! ```
//! quantized = round(fp32_value / scale)
//! dequantized = quantized * scale  
//! ```
//!
//! ## Usage Examples
//!
//! ```rust
//! use rustorch::quantization::{QuantizedTensor, QuantizationScheme, StaticQuantizer};
//! use rustorch::tensor::Tensor;
//!
//! // Dynamic quantization
//! let tensor: Tensor<f32> = Tensor::randn(&[128, 256]);
//! let quantized = tensor.quantize_dynamic(QuantizationScheme::Symmetric)?;
//!
//! // Static quantization with calibration
//! let mut quantizer = StaticQuantizer::new();
//! quantizer.calibrate(&calibration_data)?;
//! let quantized = quantizer.quantize(&tensor)?;
//!
//! // Quantization-aware training
//! let mut qat_linear = QATLinear::new(256, 128);
//! let output = qat_linear.forward(&input)?;
//! ```

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
use num_traits::{Float, Signed, Unsigned};
use std::marker::PhantomData;

// Re-export public API
pub use types::{QuantizedTensor, QuantizationType};
pub use schemes::{QuantizationScheme, SymmetricQuantization, AsymmetricQuantization};
pub use calibration::{Observer, MinMaxObserver, HistogramObserver, StaticQuantizer};
pub use operations::{QuantizedOps, DequantizeOps};
pub use qat::{QATModule, QATLinear, QATConv2d, FakeQuantize};
pub use hardware::optimized_ops;

/// Quantized tensor data types and core structures
/// 量子化テンソルデータ型とコア構造
pub mod types;

/// Quantization schemes and algorithms
/// 量子化スキームとアルゴリズム
pub mod schemes;

/// Calibration and statistical observation
/// キャリブレーションと統計観測
pub mod calibration;

/// Quantized tensor operations
/// 量子化テンソル演算
pub mod operations;

/// Quantization-aware training support
/// 量子化認識学習サポート
pub mod qat;

/// Hardware-specific optimizations
/// ハードウェア固有最適化
pub mod hardware;

/// Statistical observers for calibration
/// キャリブレーション用統計観測器
pub mod observers;

/// Trait for quantizable data types
/// 量子化可能なデータ型のトレイト
pub trait Quantizable: Copy + Clone + Send + Sync + 'static {
    /// The quantized representation type (e.g., i8, i4)
    /// 量子化表現型（例：i8、i4）
    type QuantizedType: Copy + Clone + Send + Sync;
    
    /// Convert from floating point to quantized representation
    /// 浮動小数点から量子化表現に変換
    fn quantize(&self, scale: f32, zero_point: i32) -> Self::QuantizedType;
    
    /// Convert from quantized representation to floating point
    /// 量子化表現から浮動小数点に変換
    fn dequantize(quantized: Self::QuantizedType, scale: f32, zero_point: i32) -> Self;
}

impl Quantizable for f32 {
    type QuantizedType = i8;
    
    fn quantize(&self, scale: f32, zero_point: i32) -> i8 {
        let quantized = (self / scale).round() as i32 + zero_point;
        quantized.clamp(i8::MIN as i32, i8::MAX as i32) as i8
    }
    
    fn dequantize(quantized: i8, scale: f32, zero_point: i32) -> f32 {
        (quantized as i32 - zero_point) as f32 * scale
    }
}

impl Quantizable for f64 {
    type QuantizedType = i8;
    
    fn quantize(&self, scale: f32, zero_point: i32) -> i8 {
        let quantized = (*self as f32 / scale).round() as i32 + zero_point;
        quantized.clamp(i8::MIN as i32, i8::MAX as i32) as i8
    }
    
    fn dequantize(quantized: i8, scale: f32, zero_point: i32) -> f64 {
        ((quantized as i32 - zero_point) as f32 * scale) as f64
    }
}

/// Global quantization configuration
/// グローバル量子化設定
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Default quantization scheme
    /// デフォルト量子化スキーム
    pub default_scheme: QuantizationScheme,
    /// Enable per-channel quantization
    /// チャンネル別量子化を有効化
    pub per_channel: bool,
    /// Calibration dataset size
    /// キャリブレーションデータセットサイズ
    pub calibration_size: usize,
    /// Hardware acceleration preference
    /// ハードウェア加速設定
    pub hardware_acceleration: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            default_scheme: QuantizationScheme::Symmetric,
            per_channel: false,
            calibration_size: 1000,
            hardware_acceleration: true,
        }
    }
}

/// Main quantization API for tensors
/// テンソル用メイン量子化API
pub trait TensorQuantization<T: Float> {
    /// Perform dynamic quantization
    /// 動的量子化を実行
    fn quantize_dynamic(&self, scheme: QuantizationScheme) -> RusTorchResult<QuantizedTensor<i8>>;
    
    /// Perform static quantization with pre-computed parameters
    /// 事前計算されたパラメータでの静的量子化を実行
    fn quantize_static(&self, scale: f32, zero_point: i32) -> RusTorchResult<QuantizedTensor<i8>>;
    
    /// Check if tensor is suitable for quantization
    /// テンソルが量子化に適しているかチェック
    fn can_quantize(&self) -> bool;
}

impl<T: Float + Quantizable<QuantizedType = i8>> TensorQuantization<T> for Tensor<T> {
    fn quantize_dynamic(&self, scheme: QuantizationScheme) -> RusTorchResult<QuantizedTensor<i8>> {
        let (scale, zero_point) = scheme.compute_params(&self.data)?;
        self.quantize_static(scale, zero_point)
    }
    
    fn quantize_static(&self, scale: f32, zero_point: i32) -> RusTorchResult<QuantizedTensor<i8>> {
        let quantized_data = self.data
            .mapv(|val| val.quantize(scale, zero_point));
            
        Ok(QuantizedTensor::new(
            quantized_data,
            scale,
            zero_point,
            self.device.clone(),
        ))
    }
    
    fn can_quantize(&self) -> bool {
        // Check if tensor has reasonable dynamic range for quantization
        let flat_data = self.data.as_slice().unwrap_or(&[]);
        if flat_data.is_empty() {
            return false;
        }
        
        let min_val = flat_data.iter().fold(T::infinity(), |a, &b| if a < b { a } else { b });
        let max_val = flat_data.iter().fold(T::neg_infinity(), |a, &b| if a > b { a } else { b });
        
        // Ensure reasonable dynamic range
        let range = max_val - min_val;
        !range.is_zero() && range.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_f32_quantization() {
        let value = 3.14f32;
        let scale = 0.1f32;
        let zero_point = 0i32;
        
        let quantized = value.quantize(scale, zero_point);
        let dequantized = f32::dequantize(quantized, scale, zero_point);
        
        // Should be close to original value
        assert!((value - dequantized).abs() < 0.2);
    }

    #[test]
    fn test_tensor_quantization() {
        let tensor = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );
        
        assert!(tensor.can_quantize());
        
        let quantized = tensor.quantize_dynamic(QuantizationScheme::Symmetric);
        assert!(quantized.is_ok());
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::default();
        assert!(matches!(config.default_scheme, QuantizationScheme::Symmetric));
        assert_eq!(config.calibration_size, 1000);
        assert!(config.hardware_acceleration);
    }
}
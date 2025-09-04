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
use std::fmt;
use ndarray;

// Re-export public API
pub use types::{QuantizedTensor, QuantizationType};
pub use schemes::{QuantizationScheme, SymmetricQuantization, AsymmetricQuantization};
pub use calibration::{Observer, MinMaxObserver, HistogramObserver, StaticQuantizer};
pub use operations::{QuantizedOps, DequantizeOps};
pub use qat::{QATModule, QATLinear, QATConv2d, FakeQuantize};
pub use hardware::optimized_ops;

// Export unified utilities - struct defined below

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

/// Unified quantization parameter calculator
/// 統一量子化パラメータ計算器
#[derive(Debug, Clone)]
pub struct QuantParamCalculator;

impl QuantParamCalculator {
    /// Compute symmetric quantization parameters
    /// 対称量子化パラメータを計算
    pub fn symmetric<T: Float>(data: &ndarray::ArrayD<T>, bits: u8) -> RusTorchResult<(f32, i32)> {
        let abs_max = data.fold(T::zero(), |acc, &x| acc.max(x.abs()));
        
        if abs_max == T::zero() {
            return Ok((1.0, 0));
        }
        
        let qmax = 2.0_f32.powi(bits as i32 - 1) - 1.0;
        let scale = abs_max.to_f32().unwrap_or(1.0) / qmax;
        
        Ok((scale, 0))
    }
    
    /// Compute asymmetric quantization parameters
    /// 非対称量子化パラメータを計算
    pub fn asymmetric<T: Float>(data: &ndarray::ArrayD<T>, bits: u8) -> RusTorchResult<(f32, i32)> {
        let min_val = data.fold(T::infinity(), |acc, &x| acc.min(x));
        let max_val = data.fold(T::neg_infinity(), |acc, &x| acc.max(x));
        
        if min_val >= max_val {
            return Ok((1.0, 0));
        }
        
        let qmin = -(2.0_f32.powi(bits as i32 - 1));
        let qmax = 2.0_f32.powi(bits as i32 - 1) - 1.0;
        
        let scale = (max_val - min_val).to_f32().unwrap_or(1.0) / (qmax - qmin);
        let zero_point = (qmin - min_val.to_f32().unwrap_or(0.0) / scale).round() as i32;
        let zero_point_clamped = zero_point.clamp(qmin as i32, qmax as i32);
        
        Ok((scale, zero_point_clamped))
    }
    
    /// Compute per-channel quantization parameters
    /// チャンネル別量子化パラメータを計算
    pub fn per_channel<T: Float>(
        data: &ndarray::ArrayD<T>, 
        channel_axis: usize,
        symmetric: bool,
        bits: u8
    ) -> RusTorchResult<(Vec<f32>, Vec<i32>)> {
        let channels = data.shape()[channel_axis];
        let mut scales = Vec::with_capacity(channels);
        let mut zero_points = Vec::with_capacity(channels);
        
        for c in 0..channels {
            let channel_slice = data.slice_axis(ndarray::Axis(channel_axis), ndarray::Slice::from(c..=c));
            let channel_data = channel_slice.to_owned();
            
            let (scale, zero_point) = if symmetric {
                Self::symmetric(&channel_data, bits)?
            } else {
                Self::asymmetric(&channel_data, bits)?
            };
            
            scales.push(scale);
            zero_points.push(zero_point);
        }
        
        Ok((scales, zero_points))
    }
}

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
        
        // Check for any NaN or infinite values
        if flat_data.iter().any(|&x| x.is_nan() || x.is_infinite()) {
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

    #[test]
    fn test_param_calculator_symmetric() {
        let data = ndarray::Array2::from_shape_vec((2, 3), vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0]).unwrap().into_dyn();
        let (scale, zero_point) = QuantParamCalculator::symmetric(&data, 8).unwrap();
        
        assert!(scale > 0.0);
        assert_eq!(zero_point, 0);
        assert!(scale >= 6.0 / 127.0); // Should handle max value of 6.0
    }

    #[test]
    fn test_param_calculator_asymmetric() {
        let data = ndarray::Array2::from_shape_vec((2, 2), vec![1.0f32, 10.0, 2.0, 8.0]).unwrap().into_dyn();
        let (scale, zero_point) = QuantParamCalculator::asymmetric(&data, 8).unwrap();
        
        assert!(scale > 0.0);
        assert!(zero_point >= -128 && zero_point <= 127);
    }

    #[test]
    fn test_param_calculator_per_channel() {
        let data = ndarray::Array3::from_shape_vec((2, 3, 4), (0..24).map(|x| x as f32).collect()).unwrap().into_dyn();
        let (scales, zero_points) = QuantParamCalculator::per_channel(&data, 1, true, 8).unwrap();
        
        assert_eq!(scales.len(), 3);
        assert_eq!(zero_points.len(), 3);
        assert!(scales.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_quantization_edge_cases() {
        // Test empty tensor
        let empty_tensor = Tensor::<f32>::from_vec(vec![], vec![0]);
        assert!(!empty_tensor.can_quantize());
        
        // Test constant tensor
        let constant_tensor = Tensor::<f32>::from_vec(vec![5.0; 10], vec![10]);
        assert!(!constant_tensor.can_quantize());
        
        // Test with infinity/NaN
        let inf_tensor = Tensor::<f32>::from_vec(vec![f32::INFINITY, 1.0, 2.0], vec![3]);
        assert!(!inf_tensor.can_quantize());
        
        let nan_tensor = Tensor::<f32>::from_vec(vec![f32::NAN, 1.0, 2.0], vec![3]);
        assert!(!nan_tensor.can_quantize());
    }

    #[test]
    fn test_quantization_precision_bounds() {
        // Test extreme values near quantization bounds
        let extreme_data = ndarray::Array1::from_vec(vec![-128.0f32, 127.0, 0.0]).into_dyn();
        let (scale, zero_point) = QuantParamCalculator::symmetric(&extreme_data, 8).unwrap();
        
        for &value in extreme_data.iter() {
            let quantized = value.quantize(scale, zero_point);
            let dequantized = f32::dequantize(quantized, scale, zero_point);
            
            // Quantization error should be bounded
            assert!((value - dequantized).abs() <= scale);
        }
    }

    #[test]
    fn test_different_bit_widths() {
        let data = ndarray::Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]).into_dyn();
        
        for &bits in &[4u8, 8u8, 16u8] {
            let (scale, zero_point) = QuantParamCalculator::symmetric(&data, bits).unwrap();
            assert!(scale > 0.0);
            assert_eq!(zero_point, 0); // Symmetric should have zero_point = 0
        }
    }
}
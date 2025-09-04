//! Quantized tensor types and data structures
//! 量子化テンソル型とデータ構造

use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::device::Device;
use ndarray::{ArrayD, IxDyn};
use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Unsigned, Zero};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

/// Supported quantization data types
/// サポートされている量子化データ型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// 8-bit signed integer quantization
    /// 8ビット符号付き整数量子化
    Int8,
    /// 4-bit signed integer quantization (packed)
    /// 4ビット符号付き整数量子化（パック済み）
    Int4,
    /// 8-bit unsigned integer quantization
    /// 8ビット符号なし整数量子化
    UInt8,
    /// 16-bit signed integer quantization
    /// 16ビット符号付き整数量子化
    Int16,
}

impl QuantizationType {
    /// Get the number of bits used for quantization
    /// 量子化に使用されるビット数を取得
    pub fn bits(&self) -> u8 {
        match self {
            QuantizationType::Int8 | QuantizationType::UInt8 => 8,
            QuantizationType::Int4 => 4,
            QuantizationType::Int16 => 16,
        }
    }

    /// Get the quantization range
    /// 量子化範囲を取得
    pub fn range(&self) -> (i32, i32) {
        match self {
            QuantizationType::Int8 => (i8::MIN as i32, i8::MAX as i32),
            QuantizationType::Int4 => (-8, 7), // 4-bit signed: -8 to 7
            QuantizationType::UInt8 => (u8::MIN as i32, u8::MAX as i32),
            QuantizationType::Int16 => (i16::MIN as i32, i16::MAX as i32),
        }
    }
}

/// Trait for quantizable integer types
/// 量子化可能な整数型のトレイト
pub trait QuantizableInteger: Copy + Send + Sync + fmt::Debug + Zero + 'static {
    const BITS: u8;
    const MIN_VALUE: i32;
    const MAX_VALUE: i32;

    fn from_i32_clamped(value: i32) -> Self;
    fn to_i32(&self) -> i32;
}

impl QuantizableInteger for i8 {
    const BITS: u8 = 8;
    const MIN_VALUE: i32 = i8::MIN as i32;
    const MAX_VALUE: i32 = i8::MAX as i32;

    fn from_i32_clamped(value: i32) -> Self {
        value.clamp(Self::MIN_VALUE, Self::MAX_VALUE) as i8
    }

    fn to_i32(&self) -> i32 {
        *self as i32
    }
}

impl QuantizableInteger for u8 {
    const BITS: u8 = 8;
    const MIN_VALUE: i32 = u8::MIN as i32;
    const MAX_VALUE: i32 = u8::MAX as i32;

    fn from_i32_clamped(value: i32) -> Self {
        value.clamp(Self::MIN_VALUE, Self::MAX_VALUE) as u8
    }

    fn to_i32(&self) -> i32 {
        *self as i32
    }
}

impl QuantizableInteger for i16 {
    const BITS: u8 = 16;
    const MIN_VALUE: i32 = i16::MIN as i32;
    const MAX_VALUE: i32 = i16::MAX as i32;

    fn from_i32_clamped(value: i32) -> Self {
        value.clamp(Self::MIN_VALUE, Self::MAX_VALUE) as i16
    }

    fn to_i32(&self) -> i32 {
        *self as i32
    }
}

/// 4-bit quantized integer (stored in i8)
/// 4ビット量子化整数（i8に保存）
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Int4(pub i8);

impl Int4 {
    pub fn new(value: i8) -> Self {
        Self(value.clamp(-8, 7))
    }

    pub fn value(&self) -> i8 {
        self.0
    }
}

impl Add for Int4 {
    type Output = Int4;
    fn add(self, rhs: Int4) -> Int4 {
        Int4::new(self.0 + rhs.0)
    }
}

impl Mul for Int4 {
    type Output = Int4;
    fn mul(self, rhs: Int4) -> Int4 {
        Int4::new(self.0 * rhs.0)
    }
}

impl Zero for Int4 {
    fn zero() -> Self {
        Int4(0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for Int4 {
    fn one() -> Self {
        Int4(1)
    }
}

impl FromPrimitive for Int4 {
    fn from_i32(n: i32) -> Option<Self> {
        Some(Int4::new(n as i8))
    }

    fn from_i64(n: i64) -> Option<Self> {
        Some(Int4::new(n as i8))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Int4::new(n as i8))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(Int4::new(n as i8))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Int4::new(n as i8))
    }
}

impl ToPrimitive for Int4 {
    fn to_i32(&self) -> Option<i32> {
        Some(self.0 as i32)
    }
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 as i64)
    }
    fn to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }
    fn to_f32(&self) -> Option<f32> {
        Some(self.0 as f32)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.0 as f64)
    }
}

impl QuantizableInteger for Int4 {
    const BITS: u8 = 4;
    const MIN_VALUE: i32 = -8;
    const MAX_VALUE: i32 = 7;

    fn from_i32_clamped(value: i32) -> Self {
        Int4::new(value as i8)
    }

    fn to_i32(&self) -> i32 {
        self.0 as i32
    }
}

/// Quantized tensor with integer data type and quantization parameters
/// 整数データ型と量子化パラメータを持つ量子化テンソル
#[derive(Debug, Clone)]
pub struct QuantizedTensor<Q: QuantizableInteger> {
    /// Quantized integer data
    /// 量子化整数データ
    pub data: ArrayD<Q>,
    /// Quantization scale factor
    /// 量子化スケールファクター
    pub scale: f32,
    /// Zero point for asymmetric quantization
    /// 非対称量子化のゼロポイント
    pub zero_point: i32,
    /// Device where tensor is stored
    /// テンソルが保存されているデバイス
    pub device: Device,
    /// Quantization type
    /// 量子化タイプ
    pub qtype: QuantizationType,
}

impl<Q: QuantizableInteger> QuantizedTensor<Q> {
    /// Create a new quantized tensor
    /// 新しい量子化テンソルを作成
    pub fn new(data: ArrayD<Q>, scale: f32, zero_point: i32, device: Device) -> Self {
        let qtype = match Q::BITS {
            4 => QuantizationType::Int4,
            8 => {
                if Q::MIN_VALUE >= 0 {
                    QuantizationType::UInt8
                } else {
                    QuantizationType::Int8
                }
            }
            16 => QuantizationType::Int16,
            _ => QuantizationType::Int8, // Default fallback
        };

        Self {
            data,
            scale,
            zero_point,
            device,
            qtype,
        }
    }

    /// Get tensor shape
    /// テンソル形状を取得
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get number of elements
    /// 要素数を取得
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get number of dimensions
    /// 次元数を取得
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Dequantize to floating point tensor
    /// 浮動小数点テンソルに非量子化
    pub fn dequantize<T: num_traits::Float + FromPrimitive>(&self) -> ArrayD<T> {
        self.data.mapv(|q_val| {
            let fp_val =
                (QuantizableInteger::to_i32(&q_val) as f32 - self.zero_point as f32) * self.scale;
            T::from_f32(fp_val).unwrap_or_else(T::zero)
        })
    }

    /// Change device placement
    /// デバイス配置を変更
    pub fn to_device(&mut self, device: Device) -> RusTorchResult<()> {
        // In a full implementation, this would handle device transfer
        // 完全な実装では、デバイス転送を処理
        self.device = device;
        Ok(())
    }

    /// Get quantization parameters
    /// 量子化パラメータを取得
    pub fn quantization_params(&self) -> (f32, i32) {
        (self.scale, self.zero_point)
    }

    /// Check if two quantized tensors have compatible quantization parameters
    /// 二つの量子化テンソルが互換性のある量子化パラメータを持つかチェック
    pub fn is_compatible_with(&self, other: &QuantizedTensor<Q>) -> bool {
        (self.scale - other.scale).abs() < f32::EPSILON
            && self.zero_point == other.zero_point
            && self.qtype == other.qtype
    }

    /// Reshape the quantized tensor
    /// 量子化テンソルの形状変更
    pub fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        let expected_size = new_shape.iter().product::<usize>();
        if expected_size != self.numel() {
            return Err(RusTorchError::ShapeMismatch {
                expected: vec![expected_size],
                actual: vec![self.numel()],
            });
        }

        let reshaped_data = self
            .data
            .clone()
            .to_shape(IxDyn(new_shape))
            .map_err(|e| RusTorchError::TensorOp {
                message: format!("Failed to reshape quantized tensor: {}", e),
                source: None,
            })?
            .into_owned();

        Ok(QuantizedTensor {
            data: reshaped_data,
            scale: self.scale,
            zero_point: self.zero_point,
            device: self.device.clone(),
            qtype: self.qtype,
        })
    }

    /// Create a view of the tensor (zero-copy)
    /// テンソルのビューを作成（ゼロコピー）
    pub fn view(&self) -> QuantizedTensorView<'_, Q> {
        QuantizedTensorView {
            data: self.data.view(),
            scale: self.scale,
            zero_point: self.zero_point,
            qtype: self.qtype,
        }
    }
}

/// Zero-copy view of a quantized tensor
/// 量子化テンソルのゼロコピービュー
#[derive(Debug)]
pub struct QuantizedTensorView<'a, Q: QuantizableInteger> {
    /// View of quantized data
    /// 量子化データのビュー
    pub data: ndarray::ArrayViewD<'a, Q>,
    /// Quantization scale factor
    /// 量子化スケールファクター
    pub scale: f32,
    /// Zero point for asymmetric quantization
    /// 非対称量子化のゼロポイント
    pub zero_point: i32,
    /// Quantization type
    /// 量子化タイプ
    pub qtype: QuantizationType,
}

impl<'a, Q: QuantizableInteger> QuantizedTensorView<'a, Q> {
    /// Get tensor shape
    /// テンソル形状を取得
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Dequantize view to floating point
    /// ビューを浮動小数点に非量子化
    pub fn dequantize<T: num_traits::Float + FromPrimitive>(&self) -> ArrayD<T> {
        self.data.mapv(|q_val| {
            let fp_val =
                (QuantizableInteger::to_i32(&q_val) as f32 - self.zero_point as f32) * self.scale;
            T::from_f32(fp_val).unwrap_or_else(T::zero)
        })
    }
}

impl<Q: QuantizableInteger> fmt::Display for QuantizedTensor<Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantizedTensor<{:?}>[shape={:?}, scale={:.6}, zero_point={}, device={:?}]",
            self.qtype,
            self.shape(),
            self.scale,
            self.zero_point,
            self.device
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_quantization_type() {
        assert_eq!(QuantizationType::Int8.bits(), 8);
        assert_eq!(QuantizationType::Int4.bits(), 4);
        assert_eq!(QuantizationType::Int8.range(), (-128, 127));
        assert_eq!(QuantizationType::Int4.range(), (-8, 7));
    }

    #[test]
    fn test_int4() {
        let val = Int4::new(10); // Should clamp to 7
        assert_eq!(val.value(), 7);

        let val = Int4::new(-10); // Should clamp to -8
        assert_eq!(val.value(), -8);

        assert_eq!(Int4::BITS, 4);
        assert_eq!(Int4::MIN_VALUE, -8);
        assert_eq!(Int4::MAX_VALUE, 7);
    }

    #[test]
    fn test_quantized_tensor_creation() {
        let data = Array2::<i8>::zeros((2, 3)).into_dyn();
        let qtensor = QuantizedTensor::new(data, 0.1, 0, Device::default());

        assert_eq!(qtensor.shape(), &[2, 3]);
        assert_eq!(qtensor.numel(), 6);
        assert_eq!(qtensor.scale, 0.1);
        assert_eq!(qtensor.zero_point, 0);
    }

    #[test]
    fn test_quantized_tensor_dequantize() {
        let data = Array2::from_shape_vec((2, 2), vec![10i8, 20, 30, 40])
            .unwrap()
            .into_dyn();
        let qtensor = QuantizedTensor::new(data, 0.1, 0, Device::default());

        let dequantized: ArrayD<f32> = qtensor.dequantize();
        let expected_values = vec![1.0f32, 2.0, 3.0, 4.0];

        for (actual, expected) in dequantized.iter().zip(expected_values.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_quantized_tensor_reshape() {
        let data = Array2::from_shape_vec((2, 2), vec![1i8, 2, 3, 4])
            .unwrap()
            .into_dyn();
        let qtensor = QuantizedTensor::new(data, 1.0, 0, Device::default());

        let reshaped = qtensor.reshape(&[4, 1]).unwrap();
        assert_eq!(reshaped.shape(), &[4, 1]);
        assert_eq!(reshaped.numel(), 4);
    }

    #[test]
    fn test_compatibility_check() {
        let data1 = Array2::<i8>::zeros((2, 2)).into_dyn();
        let data2 = Array2::<i8>::zeros((3, 3)).into_dyn();

        let qtensor1 = QuantizedTensor::new(data1, 0.1, 0, Device::default());
        let qtensor2 = QuantizedTensor::new(data2, 0.1, 0, Device::default());

        assert!(qtensor1.is_compatible_with(&qtensor2));

        let qtensor3 = QuantizedTensor::new(
            Array2::<i8>::zeros((2, 2)).into_dyn(),
            0.2,
            0,
            Device::default(),
        );
        assert!(!qtensor1.is_compatible_with(&qtensor3));
    }
}

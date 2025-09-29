// 型変換・精度変換機能
// Type conversion and precision conversion functionality

use ndarray::Array;
use num_complex::Complex64;
use crate::common::RusTorchResult;
use super::core::F32Tensor;
use super::f64_tensor::F64Tensor;
use super::complex_tensor::ComplexTensor;

/// テンソル型変換トレイト
/// Tensor type conversion trait
pub trait TensorConversion<T> {
    /// 型変換を実行
    /// Execute type conversion
    fn convert_to(&self) -> RusTorchResult<T>;
}

/// F32からF64への変換
/// F32 to F64 conversion
impl TensorConversion<F64Tensor> for F32Tensor {
    fn convert_to(&self) -> RusTorchResult<F64Tensor> {
        let f64_data = self.data.mapv(|x| x as f64);
        let mut result = F64Tensor::new(f64_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }
}

/// F64からF32への変換
/// F64 to F32 conversion
impl TensorConversion<F32Tensor> for F64Tensor {
    fn convert_to(&self) -> RusTorchResult<F32Tensor> {
        let f32_data = self.data.mapv(|x| x as f32);
        let (data_vec, _offset) = f32_data.into_raw_vec_and_offset();
        let shape = self.shape();
        let mut result = F32Tensor::new(data_vec, &shape)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }
}

/// F32からComplexへの変換
/// F32 to Complex conversion
impl TensorConversion<ComplexTensor> for F32Tensor {
    fn convert_to(&self) -> RusTorchResult<ComplexTensor> {
        let complex_data = self.data.mapv(|x| Complex64::new(x as f64, 0.0));
        let mut result = ComplexTensor::new(complex_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }
}

/// F64からComplexへの変換
/// F64 to Complex conversion
impl TensorConversion<ComplexTensor> for F64Tensor {
    fn convert_to(&self) -> RusTorchResult<ComplexTensor> {
        let complex_data = self.data.mapv(|x| Complex64::new(x, 0.0));
        let mut result = ComplexTensor::new(complex_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }
}

/// ComplexからF64への変換（実部のみ）
/// Complex to F64 conversion (real part only)
impl TensorConversion<F64Tensor> for ComplexTensor {
    fn convert_to(&self) -> RusTorchResult<F64Tensor> {
        let f64_data = self.real();
        let mut result = F64Tensor::new(f64_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }
}

/// ComplexからF32への変換（実部のみ）
/// Complex to F32 conversion (real part only)
impl TensorConversion<F32Tensor> for ComplexTensor {
    fn convert_to(&self) -> RusTorchResult<F32Tensor> {
        let f32_data = self.real().mapv(|x| x as f32);
        let (data_vec, _offset) = f32_data.into_raw_vec_and_offset();
        let shape = self.shape();
        let mut result = F32Tensor::new(data_vec, &shape)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }
}

/// 型安全な変換ヘルパー
/// Type-safe conversion helpers
pub struct TypeConverter;

impl TypeConverter {
    /// 自動型変換（推論）
    /// Automatic type conversion (inference)
    pub fn auto_convert<From, To>(tensor: &From) -> RusTorchResult<To>
    where
        From: TensorConversion<To>,
    {
        tensor.convert_to()
    }

    /// 精度向上変換
    /// Precision upgrade conversion
    pub fn upcast_precision(tensor: &F32Tensor) -> RusTorchResult<F64Tensor> {
        tensor.convert_to()
    }

    /// 精度低下変換（警告付き）
    /// Precision downcast conversion (with warning)
    pub fn downcast_precision(tensor: &F64Tensor) -> RusTorchResult<F32Tensor> {
        eprintln!("警告: f64からf32への変換で精度が失われる可能性があります");
        eprintln!("Warning: Precision may be lost in f64 to f32 conversion");
        tensor.convert_to()
    }

    /// 複素数への変換
    /// Convert to complex
    pub fn to_complex<T>(tensor: &T) -> RusTorchResult<ComplexTensor>
    where
        T: TensorConversion<ComplexTensor>,
    {
        tensor.convert_to()
    }

    /// 実数への変換（複素数から）
    /// Convert to real (from complex)
    pub fn to_real_f64(tensor: &ComplexTensor) -> RusTorchResult<F64Tensor> {
        tensor.convert_to()
    }

    /// 実数への変換（複素数から）
    /// Convert to real (from complex)
    pub fn to_real_f32(tensor: &ComplexTensor) -> RusTorchResult<F32Tensor> {
        tensor.convert_to()
    }
}

/// 拡張変換機能
/// Extended conversion functionality
impl F32Tensor {
    /// f64に変換
    /// Convert to f64
    pub fn to_f64(&self) -> RusTorchResult<F64Tensor> {
        self.convert_to()
    }

    /// 複素数に変換
    /// Convert to complex
    pub fn to_complex(&self) -> RusTorchResult<ComplexTensor> {
        self.convert_to()
    }

    /// 精度を指定して変換
    /// Convert with specified precision
    pub fn to_dtype(&self, dtype: &str) -> RusTorchResult<TensorVariant> {
        match dtype {
            "f32" => Ok(TensorVariant::F32(self.clone())),
            "f64" => Ok(TensorVariant::F64(self.to_f64()?)),
            "complex64" => Ok(TensorVariant::Complex(self.to_complex()?)),
            _ => Err(crate::error::RusTorchError::InvalidOperation(
                format!("Unsupported dtype: {}", dtype)
            )),
        }
    }
}

impl F64Tensor {
    /// f32に変換
    /// Convert to f32
    pub fn to_f32(&self) -> RusTorchResult<F32Tensor> {
        TypeConverter::downcast_precision(self)
    }

    /// 複素数に変換
    /// Convert to complex
    pub fn to_complex(&self) -> RusTorchResult<ComplexTensor> {
        self.convert_to()
    }

    /// 精度を指定して変換
    /// Convert with specified precision
    pub fn to_dtype(&self, dtype: &str) -> RusTorchResult<TensorVariant> {
        match dtype {
            "f32" => Ok(TensorVariant::F32(self.to_f32()?)),
            "f64" => Ok(TensorVariant::F64(self.clone())),
            "complex64" => Ok(TensorVariant::Complex(self.to_complex()?)),
            _ => Err(crate::error::RusTorchError::InvalidOperation(
                format!("Unsupported dtype: {}", dtype)
            )),
        }
    }
}

impl ComplexTensor {
    /// f32実部に変換
    /// Convert to f32 real part
    pub fn to_real_f32(&self) -> RusTorchResult<F32Tensor> {
        self.convert_to()
    }

    /// f64実部に変換
    /// Convert to f64 real part
    pub fn to_real_f64(&self) -> RusTorchResult<F64Tensor> {
        self.convert_to()
    }

    /// 実部と虚部を分離
    /// Split real and imaginary parts
    pub fn split_real_imag(&self) -> (F64Tensor, F64Tensor) {
        let real_data = self.real();
        let imag_data = self.imag();
        (F64Tensor::new(real_data), F64Tensor::new(imag_data))
    }

    /// 精度を指定して変換
    /// Convert with specified precision
    pub fn to_dtype(&self, dtype: &str) -> RusTorchResult<TensorVariant> {
        match dtype {
            "f32" => Ok(TensorVariant::F32(self.to_real_f32()?)),
            "f64" => Ok(TensorVariant::F64(self.to_real_f64()?)),
            "complex64" => Ok(TensorVariant::Complex(self.clone())),
            _ => Err(crate::error::RusTorchError::InvalidOperation(
                format!("Unsupported dtype: {}", dtype)
            )),
        }
    }
}

/// テンソル型のバリアント
/// Tensor type variants
#[derive(Debug, Clone)]
pub enum TensorVariant {
    F32(F32Tensor),
    F64(F64Tensor),
    Complex(ComplexTensor),
}

impl TensorVariant {
    /// データ型を取得
    /// Get data type
    pub fn dtype(&self) -> &'static str {
        match self {
            TensorVariant::F32(_) => "f32",
            TensorVariant::F64(_) => "f64",
            TensorVariant::Complex(_) => "complex64",
        }
    }

    /// 形状を取得
    /// Get shape
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorVariant::F32(t) => t.shape(),
            TensorVariant::F64(t) => t.shape(),
            TensorVariant::Complex(t) => t.shape(),
        }
    }

    /// 要素数を取得
    /// Get number of elements
    pub fn numel(&self) -> usize {
        match self {
            TensorVariant::F32(t) => t.numel(),
            TensorVariant::F64(t) => t.numel(),
            TensorVariant::Complex(t) => t.numel(),
        }
    }

    /// 指定した型に変換
    /// Convert to specified type
    pub fn to_dtype(&self, dtype: &str) -> RusTorchResult<TensorVariant> {
        match self {
            TensorVariant::F32(t) => t.to_dtype(dtype),
            TensorVariant::F64(t) => t.to_dtype(dtype),
            TensorVariant::Complex(t) => t.to_dtype(dtype),
        }
    }

    /// F32テンソルとして取得
    /// Get as F32 tensor
    pub fn as_f32(&self) -> RusTorchResult<F32Tensor> {
        match self {
            TensorVariant::F32(t) => Ok(t.clone()),
            TensorVariant::F64(t) => t.to_f32(),
            TensorVariant::Complex(t) => t.to_real_f32(),
        }
    }

    /// F64テンソルとして取得
    /// Get as F64 tensor
    pub fn as_f64(&self) -> RusTorchResult<F64Tensor> {
        match self {
            TensorVariant::F32(t) => t.to_f64(),
            TensorVariant::F64(t) => Ok(t.clone()),
            TensorVariant::Complex(t) => t.to_real_f64(),
        }
    }

    /// 複素数テンソルとして取得
    /// Get as complex tensor
    pub fn as_complex(&self) -> RusTorchResult<ComplexTensor> {
        match self {
            TensorVariant::F32(t) => t.to_complex(),
            TensorVariant::F64(t) => t.to_complex(),
            TensorVariant::Complex(t) => Ok(t.clone()),
        }
    }
}

/// 精度情報
/// Precision information
pub struct PrecisionInfo {
    pub dtype: &'static str,
    pub bits: u8,
    pub is_complex: bool,
    pub is_signed: bool,
}

impl PrecisionInfo {
    /// F32の精度情報
    /// F32 precision info
    pub const F32: PrecisionInfo = PrecisionInfo {
        dtype: "f32",
        bits: 32,
        is_complex: false,
        is_signed: true,
    };

    /// F64の精度情報
    /// F64 precision info
    pub const F64: PrecisionInfo = PrecisionInfo {
        dtype: "f64",
        bits: 64,
        is_complex: false,
        is_signed: true,
    };

    /// Complex64の精度情報
    /// Complex64 precision info
    pub const COMPLEX64: PrecisionInfo = PrecisionInfo {
        dtype: "complex64",
        bits: 128, // 64 + 64
        is_complex: true,
        is_signed: true,
    };

    /// 精度情報を取得
    /// Get precision info
    pub fn from_str(dtype: &str) -> Option<&'static PrecisionInfo> {
        match dtype {
            "f32" => Some(&Self::F32),
            "f64" => Some(&Self::F64),
            "complex64" => Some(&Self::COMPLEX64),
            _ => None,
        }
    }

    /// より高い精度かチェック
    /// Check if higher precision
    pub fn is_higher_precision_than(&self, other: &PrecisionInfo) -> bool {
        if self.is_complex && !other.is_complex {
            true
        } else if !self.is_complex && other.is_complex {
            false
        } else {
            self.bits > other.bits
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_f32::tensor::generic_tensor::TensorOps;

    #[test]
    fn test_f32_to_f64_conversion() {
        let f32_tensor = F32Tensor::ones(&[2, 2]).unwrap();
        let f64_tensor = f32_tensor.to_f64().unwrap();

        assert_eq!(f64_tensor.dtype(), "f64");
        assert_eq!(f64_tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_f64_to_f32_conversion() {
        let f64_tensor = F64Tensor::ones(&[2, 2]).unwrap();
        let f32_tensor = f64_tensor.to_f32().unwrap();

        assert_eq!(f32_tensor.dtype(), "f32");
        assert_eq!(f32_tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_real_to_complex_conversion() {
        let f32_tensor = F32Tensor::ones(&[2, 2]).unwrap();
        let complex_tensor = f32_tensor.to_complex().unwrap();

        assert_eq!(complex_tensor.dtype(), "complex64");
        assert_eq!(complex_tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_complex_to_real_conversion() {
        let complex_tensor = ComplexTensor::zeros(&[2, 2]).unwrap();
        let f64_tensor = complex_tensor.to_real_f64().unwrap();

        assert_eq!(f64_tensor.dtype(), "f64");
        assert_eq!(f64_tensor.shape(), &[2, 2]);
    }

    #[test]
    fn test_tensor_variant() {
        let f32_tensor = F32Tensor::ones(&[2, 2]).unwrap();
        let variant = TensorVariant::F32(f32_tensor);

        assert_eq!(variant.dtype(), "f32");
        assert_eq!(variant.shape(), &[2, 2]);
        assert_eq!(variant.numel(), 4);
    }

    #[test]
    fn test_precision_info() {
        let f32_info = PrecisionInfo::from_str("f32").unwrap();
        let f64_info = PrecisionInfo::from_str("f64").unwrap();

        assert!(f64_info.is_higher_precision_than(f32_info));
        assert!(!f32_info.is_higher_precision_than(f64_info));
    }
}
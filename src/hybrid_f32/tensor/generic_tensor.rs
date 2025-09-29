// ジェネリックテンソル統一API
// Generic tensor unified API

use super::complex_tensor::ComplexTensor;
use super::core::F32Tensor;
use super::f64_tensor::F64Tensor;
use super::type_conversion::{TensorConversion, TensorVariant};
use crate::common::RusTorchResult;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// テンソルの基本操作を定義するトレイト
/// Trait defining basic tensor operations
pub trait TensorOps: Debug + Clone {
    type Scalar: Copy + Debug;

    /// 新しいテンソルを作成
    /// Create a new tensor
    fn zeros(shape: &[usize]) -> RusTorchResult<Self>;
    fn ones(shape: &[usize]) -> RusTorchResult<Self>;
    fn randn(shape: &[usize]) -> RusTorchResult<Self>;

    /// 基本プロパティ
    /// Basic properties
    fn shape(&self) -> &[usize];
    fn ndim(&self) -> usize;
    fn numel(&self) -> usize;
    fn dtype(&self) -> &'static str;

    /// 勾配設定
    /// Gradient settings
    fn requires_grad_(self, requires_grad: bool) -> Self;

    /// 形状操作
    /// Shape operations
    fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<Self>;
    fn transpose(&self) -> RusTorchResult<Self>;
    fn unsqueeze(&self, dim: usize) -> RusTorchResult<Self>;
    fn expand(&self, new_shape: &[usize]) -> RusTorchResult<Self>;
    fn transpose_dims(&self, dim1: usize, dim2: usize) -> RusTorchResult<Self>;

    /// 統計操作
    /// Statistical operations
    fn sum_scalar(&self) -> Self::Scalar;
    fn mean_scalar(&self) -> Self::Scalar;
}

/// 行列演算を定義するトレイト
/// Trait defining matrix operations
pub trait MatrixOps: TensorOps {
    /// 行列積
    /// Matrix multiplication
    fn matmul(&self, other: &Self) -> RusTorchResult<Self>;

    /// 転置
    /// Transpose
    fn t(&self) -> RusTorchResult<Self> {
        self.transpose()
    }
}

/// 数学関数を定義するトレイト
/// Trait defining mathematical functions
pub trait MathOps: TensorOps {
    /// 指数関数
    /// Exponential function
    fn exp(&self) -> RusTorchResult<Self>;

    /// 対数関数
    /// Logarithm function
    fn log(&self) -> RusTorchResult<Self>;

    /// べき乗
    /// Power function
    fn pow(&self, exponent: f64) -> RusTorchResult<Self>;

    /// 平方根
    /// Square root
    fn sqrt(&self) -> RusTorchResult<Self>;

    /// 三角関数
    /// Trigonometric functions
    fn sin(&self) -> RusTorchResult<Self>;
    fn cos(&self) -> RusTorchResult<Self>;
    fn tan(&self) -> RusTorchResult<Self>;

    /// ソフトマックス
    /// Softmax
    fn softmax(&self, dim: Option<usize>) -> RusTorchResult<Self>;
}

/// 複素数専用操作を定義するトレイト
/// Trait defining complex-specific operations
pub trait ComplexOps: TensorOps {
    type RealTensor: TensorOps;

    /// 実部を取得
    /// Get real part
    fn real(&self) -> Self::RealTensor;

    /// 虚部を取得
    /// Get imaginary part
    fn imag(&self) -> Self::RealTensor;

    /// 絶対値を取得
    /// Get absolute value
    fn abs(&self) -> Self::RealTensor;

    /// 位相を取得
    /// Get phase
    fn angle(&self) -> Self::RealTensor;

    /// 共役複素数を取得
    /// Get complex conjugate
    fn conj(&self) -> Self;
}

// F32Tensor用実装
impl TensorOps for F32Tensor {
    type Scalar = f32;

    fn zeros(shape: &[usize]) -> RusTorchResult<Self> {
        F32Tensor::zeros(shape)
    }

    fn ones(shape: &[usize]) -> RusTorchResult<Self> {
        F32Tensor::ones(shape)
    }

    fn randn(shape: &[usize]) -> RusTorchResult<Self> {
        F32Tensor::randn(shape)
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn ndim(&self) -> usize {
        self.ndim()
    }

    fn numel(&self) -> usize {
        self.numel()
    }

    fn dtype(&self) -> &'static str {
        "f32"
    }

    fn requires_grad_(self, requires_grad: bool) -> Self {
        let mut result = self;
        result.requires_grad = requires_grad;
        result
    }

    fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        self.reshape(new_shape)
    }

    fn transpose(&self) -> RusTorchResult<Self> {
        self.transpose()
    }

    fn unsqueeze(&self, dim: usize) -> RusTorchResult<Self> {
        self.unsqueeze(dim)
    }

    fn expand(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        self.expand(new_shape)
    }

    fn transpose_dims(&self, dim1: usize, dim2: usize) -> RusTorchResult<Self> {
        self.transpose_dims(dim1, dim2)
    }

    fn sum_scalar(&self) -> Self::Scalar {
        self.sum().unwrap_or(0.0)
    }

    fn mean_scalar(&self) -> Self::Scalar {
        // mean関数がないので、sum/lengthで計算
        let sum_val = self.sum().unwrap_or(0.0);
        let len = self.data.len() as f32;
        if len > 0.0 {
            sum_val / len
        } else {
            0.0
        }
    }
}

impl MatrixOps for F32Tensor {
    fn matmul(&self, other: &Self) -> RusTorchResult<Self> {
        self.matmul(other)
    }
}

impl MathOps for F32Tensor {
    fn exp(&self) -> RusTorchResult<Self> {
        let exp_data = self.data.mapv(|x| x.exp());
        let (data_vec, _offset) = exp_data.into_raw_vec_and_offset();
        let shape = self.shape();
        let mut result = F32Tensor::new(data_vec, &shape)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn log(&self) -> RusTorchResult<Self> {
        let log_data = self.data.mapv(|x| x.ln());
        let (data_vec, _offset) = log_data.into_raw_vec_and_offset();
        let shape = self.shape();
        let mut result = F32Tensor::new(data_vec, &shape)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn pow(&self, exponent: f64) -> RusTorchResult<Self> {
        let pow_data = self.data.mapv(|x| x.powf(exponent as f32));
        let (data_vec, _offset) = pow_data.into_raw_vec_and_offset();
        let shape = self.shape();
        let mut result = F32Tensor::new(data_vec, &shape)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn sqrt(&self) -> RusTorchResult<Self> {
        let sqrt_data = self.data.mapv(|x| x.sqrt());
        let (data_vec, _offset) = sqrt_data.into_raw_vec_and_offset();
        let shape = self.shape();
        let mut result = F32Tensor::new(data_vec, &shape)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn sin(&self) -> RusTorchResult<Self> {
        let sin_data = self.data.mapv(|x| x.sin());
        let (data_vec, _offset) = sin_data.into_raw_vec_and_offset();
        let shape = self.shape();
        let mut result = F32Tensor::new(data_vec, &shape)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn cos(&self) -> RusTorchResult<Self> {
        let cos_data = self.data.mapv(|x| x.cos());
        let (data_vec, _offset) = cos_data.into_raw_vec_and_offset();
        let shape = self.shape();
        let mut result = F32Tensor::new(data_vec, &shape)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn tan(&self) -> RusTorchResult<Self> {
        let tan_data = self.data.mapv(|x| x.tan());
        let (data_vec, _offset) = tan_data.into_raw_vec_and_offset();
        let shape = self.shape();
        let mut result = F32Tensor::new(data_vec, &shape)?;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn softmax(&self, dim: Option<usize>) -> RusTorchResult<Self> {
        self.softmax(dim)
    }
}

// F64Tensor用実装
impl TensorOps for F64Tensor {
    type Scalar = f64;

    fn zeros(shape: &[usize]) -> RusTorchResult<Self> {
        F64Tensor::zeros(shape)
    }

    fn ones(shape: &[usize]) -> RusTorchResult<Self> {
        F64Tensor::ones(shape)
    }

    fn randn(shape: &[usize]) -> RusTorchResult<Self> {
        F64Tensor::randn(shape)
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn ndim(&self) -> usize {
        self.ndim()
    }

    fn numel(&self) -> usize {
        self.numel()
    }

    fn dtype(&self) -> &'static str {
        "f64"
    }

    fn requires_grad_(self, requires_grad: bool) -> Self {
        let mut result = self;
        result.requires_grad = requires_grad;
        result
    }

    fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        self.reshape(new_shape)
    }

    fn transpose(&self) -> RusTorchResult<Self> {
        self.transpose()
    }

    fn unsqueeze(&self, dim: usize) -> RusTorchResult<Self> {
        self.unsqueeze(dim)
    }

    fn expand(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        self.expand(new_shape)
    }

    fn transpose_dims(&self, dim1: usize, dim2: usize) -> RusTorchResult<Self> {
        self.transpose_dims(dim1, dim2)
    }

    fn sum_scalar(&self) -> Self::Scalar {
        self.sum()
    }

    fn mean_scalar(&self) -> Self::Scalar {
        // mean関数がないので、sum/lengthで計算
        let sum_val = self.sum();
        let len = self.data.len() as f64;
        if len > 0.0 {
            sum_val / len
        } else {
            0.0
        }
    }
}

impl MatrixOps for F64Tensor {
    fn matmul(&self, other: &Self) -> RusTorchResult<Self> {
        self.matmul(other)
    }
}

impl MathOps for F64Tensor {
    fn exp(&self) -> RusTorchResult<Self> {
        let exp_data = self.data.mapv(|x| x.exp());
        let mut result = F64Tensor::new(exp_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn log(&self) -> RusTorchResult<Self> {
        let log_data = self.data.mapv(|x| x.ln());
        let mut result = F64Tensor::new(log_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn pow(&self, exponent: f64) -> RusTorchResult<Self> {
        let pow_data = self.data.mapv(|x| x.powf(exponent));
        let mut result = F64Tensor::new(pow_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn sqrt(&self) -> RusTorchResult<Self> {
        let sqrt_data = self.data.mapv(|x| x.sqrt());
        let mut result = F64Tensor::new(sqrt_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn sin(&self) -> RusTorchResult<Self> {
        let sin_data = self.data.mapv(|x| x.sin());
        let mut result = F64Tensor::new(sin_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn cos(&self) -> RusTorchResult<Self> {
        let cos_data = self.data.mapv(|x| x.cos());
        let mut result = F64Tensor::new(cos_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn tan(&self) -> RusTorchResult<Self> {
        let tan_data = self.data.mapv(|x| x.tan());
        let mut result = F64Tensor::new(tan_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn softmax(&self, dim: Option<usize>) -> RusTorchResult<Self> {
        self.softmax(dim)
    }
}

// ComplexTensor用実装
impl TensorOps for ComplexTensor {
    type Scalar = num_complex::Complex64;

    fn zeros(shape: &[usize]) -> RusTorchResult<Self> {
        ComplexTensor::zeros(shape)
    }

    fn ones(shape: &[usize]) -> RusTorchResult<Self> {
        ComplexTensor::ones(shape)
    }

    fn randn(shape: &[usize]) -> RusTorchResult<Self> {
        // 複素数のランダムテンソル：実部と虚部が独立した正規分布
        use ndarray::Array;
        use ndarray_rand::rand_distr::StandardNormal;
        use ndarray_rand::RandomExt;
        use num_complex::Complex64;

        let real_part: Array<f64, _> = Array::random(shape, StandardNormal);
        let imag_part: Array<f64, _> = Array::random(shape, StandardNormal);

        let complex_data = real_part
            .iter()
            .zip(imag_part.iter())
            .map(|(&r, &i)| Complex64::new(r, i))
            .collect::<Vec<_>>();

        let data = Array::from_shape_vec(shape, complex_data)
            .map_err(|e| crate::error::RusTorchError::tensor_op(e.to_string()))?;

        Ok(ComplexTensor::new(data))
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn ndim(&self) -> usize {
        self.ndim()
    }

    fn numel(&self) -> usize {
        self.numel()
    }

    fn dtype(&self) -> &'static str {
        "complex64"
    }

    fn requires_grad_(self, requires_grad: bool) -> Self {
        let mut result = self;
        result.requires_grad = requires_grad;
        result
    }

    fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        self.reshape(new_shape)
    }

    fn transpose(&self) -> RusTorchResult<Self> {
        self.transpose()
    }

    fn unsqueeze(&self, dim: usize) -> RusTorchResult<Self> {
        self.unsqueeze(dim)
    }

    fn expand(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        self.expand(new_shape)
    }

    fn transpose_dims(&self, dim1: usize, dim2: usize) -> RusTorchResult<Self> {
        self.transpose_dims(dim1, dim2)
    }

    fn sum_scalar(&self) -> Self::Scalar {
        self.sum()
    }

    fn mean_scalar(&self) -> Self::Scalar {
        use num_complex::Complex64;
        // mean関数がないので、sum/lengthで計算
        let sum_val = self.sum();
        let len = self.data.len() as f64;
        if len > 0.0 {
            sum_val / len
        } else {
            Complex64::new(0.0, 0.0)
        }
    }
}

impl MatrixOps for ComplexTensor {
    fn matmul(&self, other: &Self) -> RusTorchResult<Self> {
        self.matmul(other)
    }
}

impl MathOps for ComplexTensor {
    fn exp(&self) -> RusTorchResult<Self> {
        let exp_data = self.data.mapv(|x| x.exp());
        let mut result = ComplexTensor::new(exp_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn log(&self) -> RusTorchResult<Self> {
        let log_data = self.data.mapv(|x| x.ln());
        let mut result = ComplexTensor::new(log_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn pow(&self, exponent: f64) -> RusTorchResult<Self> {
        use num_complex::Complex64;
        let exp_complex = Complex64::new(exponent, 0.0);
        let pow_data = self.data.mapv(|x| x.powc(exp_complex));
        let mut result = ComplexTensor::new(pow_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn sqrt(&self) -> RusTorchResult<Self> {
        let sqrt_data = self.data.mapv(|x| x.sqrt());
        let mut result = ComplexTensor::new(sqrt_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn sin(&self) -> RusTorchResult<Self> {
        let sin_data = self.data.mapv(|x| x.sin());
        let mut result = ComplexTensor::new(sin_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn cos(&self) -> RusTorchResult<Self> {
        let cos_data = self.data.mapv(|x| x.cos());
        let mut result = ComplexTensor::new(cos_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn tan(&self) -> RusTorchResult<Self> {
        let tan_data = self.data.mapv(|x| x.tan());
        let mut result = ComplexTensor::new(tan_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    fn softmax(&self, _dim: Option<usize>) -> RusTorchResult<Self> {
        // 複素数のソフトマックスは定義が複雑なため、実部のみに適用
        Err(crate::error::RusTorchError::InvalidOperation(
            "Softmax is not well-defined for complex tensors".to_string(),
        ))
    }
}

impl ComplexOps for ComplexTensor {
    type RealTensor = F64Tensor;

    fn real(&self) -> Self::RealTensor {
        F64Tensor::new(self.real())
    }

    fn imag(&self) -> Self::RealTensor {
        F64Tensor::new(self.imag())
    }

    fn abs(&self) -> Self::RealTensor {
        F64Tensor::new(self.abs())
    }

    fn angle(&self) -> Self::RealTensor {
        F64Tensor::new(self.angle())
    }

    fn conj(&self) -> Self {
        self.conj()
    }
}

/// ジェネリックテンソル操作のヘルパー関数
/// Helper functions for generic tensor operations
pub struct TensorUtils;

impl TensorUtils {
    /// 型安全な加算
    /// Type-safe addition
    pub fn add<T>(a: &T, b: &T) -> RusTorchResult<T>
    where
        T: TensorOps + Add<Output = T> + Clone,
    {
        Ok(a.clone() + b.clone())
    }

    /// 型安全な行列積
    /// Type-safe matrix multiplication
    pub fn matmul<T>(a: &T, b: &T) -> RusTorchResult<T>
    where
        T: MatrixOps,
    {
        a.matmul(b)
    }

    /// 型を指定したテンソル作成
    /// Create tensor with specified type
    pub fn zeros_like<T>(tensor: &T) -> RusTorchResult<T>
    where
        T: TensorOps,
    {
        T::zeros(tensor.shape())
    }

    /// 型を指定したテンソル作成
    /// Create tensor with specified type
    pub fn ones_like<T>(tensor: &T) -> RusTorchResult<T>
    where
        T: TensorOps,
    {
        T::ones(tensor.shape())
    }

    /// テンソルの型情報を表示
    /// Display tensor type information
    pub fn tensor_info<T>(tensor: &T) -> String
    where
        T: TensorOps,
    {
        format!(
            "Tensor(dtype={}, shape={:?}, numel={})",
            tensor.dtype(),
            tensor.shape(),
            tensor.numel()
        )
    }
}

/// ジェネリックテンソルファクトリ
/// Generic tensor factory
pub struct TensorFactory;

impl TensorFactory {
    /// 指定した精度でゼロテンソルを作成
    /// Create zero tensor with specified precision
    pub fn zeros(dtype: &str, shape: &[usize]) -> RusTorchResult<TensorVariant> {
        match dtype {
            "f32" => Ok(TensorVariant::F32(F32Tensor::zeros(shape)?)),
            "f64" => Ok(TensorVariant::F64(F64Tensor::zeros(shape)?)),
            "complex64" => Ok(TensorVariant::Complex(ComplexTensor::zeros(shape)?)),
            _ => Err(crate::error::RusTorchError::InvalidOperation(format!(
                "Unsupported dtype: {}",
                dtype
            ))),
        }
    }

    /// 指定した精度でワンテンソルを作成
    /// Create ones tensor with specified precision
    pub fn ones(dtype: &str, shape: &[usize]) -> RusTorchResult<TensorVariant> {
        match dtype {
            "f32" => Ok(TensorVariant::F32(F32Tensor::ones(shape)?)),
            "f64" => Ok(TensorVariant::F64(F64Tensor::ones(shape)?)),
            "complex64" => Ok(TensorVariant::Complex(ComplexTensor::ones(shape)?)),
            _ => Err(crate::error::RusTorchError::InvalidOperation(format!(
                "Unsupported dtype: {}",
                dtype
            ))),
        }
    }

    /// 指定した精度でランダムテンソルを作成
    /// Create random tensor with specified precision
    pub fn randn(dtype: &str, shape: &[usize]) -> RusTorchResult<TensorVariant> {
        match dtype {
            "f32" => Ok(TensorVariant::F32(F32Tensor::randn(shape)?)),
            "f64" => Ok(TensorVariant::F64(F64Tensor::randn(shape)?)),
            "complex64" => Ok(TensorVariant::Complex(ComplexTensor::randn(shape)?)),
            _ => Err(crate::error::RusTorchError::InvalidOperation(format!(
                "Unsupported dtype: {}",
                dtype
            ))),
        }
    }

    /// 最適な精度を自動選択
    /// Automatically select optimal precision
    pub fn auto_precision(
        shape: &[usize],
        requires_high_precision: bool,
        is_complex: bool,
    ) -> RusTorchResult<TensorVariant> {
        let dtype = match (requires_high_precision, is_complex) {
            (true, true) => "complex64",
            (true, false) => "f64",
            (false, true) => "complex64", // 複素数は常に高精度
            (false, false) => "f32",
        };
        Self::zeros(dtype, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_f32::tensor::{ComplexTensor, F32Tensor, F64Tensor};

    #[test]
    fn test_generic_tensor_ops() {
        let f32_tensor = F32Tensor::ones(&[2, 2]).unwrap();
        let f64_tensor = F64Tensor::ones(&[2, 2]).unwrap();

        assert_eq!(f32_tensor.dtype(), "f32");
        assert_eq!(f64_tensor.dtype(), "f64");
        assert_eq!(f32_tensor.shape(), f64_tensor.shape());
    }

    #[test]
    fn test_tensor_factory() {
        let f32_zeros = TensorFactory::zeros("f32", &[2, 2]).unwrap();
        let f64_ones = TensorFactory::ones("f64", &[3, 3]).unwrap();
        let complex_randn = TensorFactory::randn("complex64", &[2, 2]).unwrap();

        assert_eq!(f32_zeros.dtype(), "f32");
        assert_eq!(f64_ones.dtype(), "f64");
        assert_eq!(complex_randn.dtype(), "complex64");
    }

    #[test]
    fn test_tensor_utils() {
        let tensor = F32Tensor::ones(&[2, 2]).unwrap();
        let info = TensorUtils::tensor_info(&tensor);

        assert!(info.contains("f32"));
        assert!(info.contains("[2, 2]"));
        assert!(info.contains("4"));
    }

    #[test]
    fn test_math_ops() {
        let tensor = F32Tensor::ones(&[2, 2]).unwrap();
        let exp_tensor = tensor.exp().unwrap();
        let log_tensor = tensor.log().unwrap();

        // e^1 ≈ 2.718
        assert!((exp_tensor.sum().unwrap() - 4.0 * std::f32::consts::E).abs() < 1e-5);
        // ln(1) = 0
        assert!(log_tensor.sum().unwrap().abs() < 1e-5);
    }

    #[test]
    fn test_complex_ops() {
        let complex_tensor = ComplexTensor::ones(&[2, 2]).unwrap();
        let real_part = complex_tensor.real();
        let imag_part = complex_tensor.imag();
        let conj_tensor = complex_tensor.conj();

        // Note: real/imag return ArrayBase, not tensor objects
        // Just verify they return data
        assert!(real_part.len() > 0);
        assert!(imag_part.len() > 0);
        assert_eq!(conj_tensor.dtype(), "complex64");
    }
}

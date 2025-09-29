// ComplexTensor実装 - 複素数数値計算用
// ComplexTensor implementation - for complex numerical computation

use ndarray::{Array, IxDyn};
use std::sync::Arc;
use std::ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut};
use num_complex::Complex64;
use crate::common::RusTorchResult;
use super::core::{Index2D, Index3D, DeviceState, MetalBuffer, CoreMLBuffer};

/// Complex64専用テンソル（複素数計算特化）
/// Complex64-specific tensor (complex computation optimized)
#[derive(Debug)]
pub struct ComplexTensor {
    /// CPU側データ
    /// CPU-side data
    pub data: Array<Complex64, IxDyn>,

    /// GPU共有バッファ（Metal用）
    /// GPU shared buffer (for Metal)
    pub metal_buffer: Option<Arc<MetalBuffer>>,

    /// Neural Engine共有バッファ（CoreML用）
    /// Neural Engine shared buffer (for CoreML)
    pub coreml_buffer: Option<Arc<CoreMLBuffer>>,

    /// デバイス最適化状態
    /// Device optimization state
    pub device_state: DeviceState,

    /// 勾配追跡
    /// Gradient tracking
    pub requires_grad: bool,

    /// テンソル形状
    /// Tensor shape
    shape: Vec<usize>,
}

impl Clone for ComplexTensor {
    fn clone(&self) -> Self {
        ComplexTensor {
            data: self.data.clone(),
            metal_buffer: self.metal_buffer.clone(),
            coreml_buffer: self.coreml_buffer.clone(),
            device_state: self.device_state.clone(),
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Addition
impl Add<ComplexTensor> for ComplexTensor {
    type Output = ComplexTensor;
    fn add(self, other: ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data + &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Add<&ComplexTensor> for ComplexTensor {
    type Output = ComplexTensor;
    fn add(self, other: &ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data + &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Add for &ComplexTensor {
    type Output = ComplexTensor;
    fn add(self, other: &ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data + &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Add<Complex64> for ComplexTensor {
    type Output = ComplexTensor;
    fn add(self, scalar: Complex64) -> ComplexTensor {
        ComplexTensor {
            data: &self.data + scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Add<Complex64> for &ComplexTensor {
    type Output = ComplexTensor;
    fn add(self, scalar: Complex64) -> ComplexTensor {
        ComplexTensor {
            data: &self.data + scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Subtraction
impl Sub<ComplexTensor> for ComplexTensor {
    type Output = ComplexTensor;
    fn sub(self, other: ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data - &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Sub<&ComplexTensor> for ComplexTensor {
    type Output = ComplexTensor;
    fn sub(self, other: &ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data - &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Sub for &ComplexTensor {
    type Output = ComplexTensor;
    fn sub(self, other: &ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data - &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Sub<Complex64> for ComplexTensor {
    type Output = ComplexTensor;
    fn sub(self, scalar: Complex64) -> ComplexTensor {
        ComplexTensor {
            data: &self.data - scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Sub<Complex64> for &ComplexTensor {
    type Output = ComplexTensor;
    fn sub(self, scalar: Complex64) -> ComplexTensor {
        ComplexTensor {
            data: &self.data - scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Multiplication
impl Mul<ComplexTensor> for ComplexTensor {
    type Output = ComplexTensor;
    fn mul(self, other: ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data * &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Mul<&ComplexTensor> for ComplexTensor {
    type Output = ComplexTensor;
    fn mul(self, other: &ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data * &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Mul for &ComplexTensor {
    type Output = ComplexTensor;
    fn mul(self, other: &ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data * &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Mul<Complex64> for ComplexTensor {
    type Output = ComplexTensor;
    fn mul(self, scalar: Complex64) -> ComplexTensor {
        ComplexTensor {
            data: &self.data * scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Mul<Complex64> for &ComplexTensor {
    type Output = ComplexTensor;
    fn mul(self, scalar: Complex64) -> ComplexTensor {
        ComplexTensor {
            data: &self.data * scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Division
impl Div<ComplexTensor> for ComplexTensor {
    type Output = ComplexTensor;
    fn div(self, other: ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data / &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Div<&ComplexTensor> for ComplexTensor {
    type Output = ComplexTensor;
    fn div(self, other: &ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data / &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Div for &ComplexTensor {
    type Output = ComplexTensor;
    fn div(self, other: &ComplexTensor) -> ComplexTensor {
        ComplexTensor {
            data: &self.data / &other.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad || other.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Div<Complex64> for ComplexTensor {
    type Output = ComplexTensor;
    fn div(self, scalar: Complex64) -> ComplexTensor {
        ComplexTensor {
            data: &self.data / scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Div<Complex64> for &ComplexTensor {
    type Output = ComplexTensor;
    fn div(self, scalar: Complex64) -> ComplexTensor {
        ComplexTensor {
            data: &self.data / scalar,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// 演算子実装 - Negation
impl Neg for ComplexTensor {
    type Output = ComplexTensor;
    fn neg(self) -> ComplexTensor {
        ComplexTensor {
            data: -&self.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

impl Neg for &ComplexTensor {
    type Output = ComplexTensor;
    fn neg(self) -> ComplexTensor {
        ComplexTensor {
            data: -&self.data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: self.requires_grad,
            shape: self.shape.clone(),
        }
    }
}

// ComplexTensor主要実装
impl ComplexTensor {
    /// 新しいテンソルを作成
    /// Create a new tensor
    pub fn new(data: Array<Complex64, IxDyn>) -> Self {
        let shape = data.shape().to_vec();
        ComplexTensor {
            data,
            metal_buffer: None,
            coreml_buffer: None,
            device_state: DeviceState::CPU,
            requires_grad: false,
            shape,
        }
    }

    /// ゼロテンソルを作成
    /// Create a zero tensor
    pub fn zeros(shape: &[usize]) -> RusTorchResult<Self> {
        let data = Array::zeros(shape);
        Ok(ComplexTensor::new(data))
    }

    /// ワンテンソルを作成
    /// Create a ones tensor
    pub fn ones(shape: &[usize]) -> RusTorchResult<Self> {
        let data = Array::ones(shape);
        Ok(ComplexTensor::new(data))
    }

    /// 実部と虚部から複素数テンソルを作成
    /// Create complex tensor from real and imaginary parts
    pub fn from_real_imag(real: &Array<f64, IxDyn>, imag: &Array<f64, IxDyn>) -> RusTorchResult<Self> {
        if real.shape() != imag.shape() {
            return Err(crate::error::RusTorchError::tensor_op(
                "Real and imaginary parts must have the same shape"
            ));
        }

        let complex_data = real.iter().zip(imag.iter())
            .map(|(&r, &i)| Complex64::new(r, i))
            .collect::<Vec<_>>();

        let data = Array::from_shape_vec(real.dim(), complex_data)?;
        Ok(ComplexTensor::new(data))
    }

    /// 実部のみから複素数テンソルを作成
    /// Create complex tensor from real part only
    pub fn from_real(real: &Array<f64, IxDyn>) -> Self {
        let complex_data = real.mapv(|r| Complex64::new(r, 0.0));
        ComplexTensor::new(complex_data)
    }

    /// 形状を取得
    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 次元数を取得
    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// 要素数を取得
    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// データ型を取得
    /// Get data type
    pub fn dtype(&self) -> &'static str {
        "complex64"
    }

    /// 勾配を必要とするかを設定
    /// Set requires gradient
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// リシェイプ
    /// Reshape
    pub fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        let new_data = self.data.clone().into_shape_with_order(new_shape)?;
        let mut result = ComplexTensor::new(new_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// 転置
    /// Transpose
    pub fn transpose(&self) -> RusTorchResult<Self> {
        let transposed = self.data.t().to_owned();
        let mut shape = self.shape.clone();
        shape.reverse();
        let mut result = ComplexTensor::new(transposed);
        result.shape = shape;
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// 行列積
    /// Matrix multiplication
    pub fn matmul(&self, other: &ComplexTensor) -> RusTorchResult<Self> {
        use ndarray::linalg::general_mat_mul;

        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(crate::error::RusTorchError::tensor_op(
                format!("Cannot multiply matrices with shapes {:?} and {:?}", self.shape, other.shape)
            ));
        }

        let mut result_data = Array::zeros((m, n));
        general_mat_mul(Complex64::new(1.0, 0.0), &self.data.view().into_dimensionality()?, &other.data.view().into_dimensionality()?, Complex64::new(0.0, 0.0), &mut result_data.view_mut());

        let result_dyn = result_data.into_dyn();
        let mut result = ComplexTensor::new(result_dyn);
        result.requires_grad = self.requires_grad || other.requires_grad;
        Ok(result)
    }

    /// 実部を取得
    /// Get real part
    pub fn real(&self) -> Array<f64, IxDyn> {
        self.data.mapv(|c| c.re)
    }

    /// 虚部を取得
    /// Get imaginary part
    pub fn imag(&self) -> Array<f64, IxDyn> {
        self.data.mapv(|c| c.im)
    }

    /// 絶対値を取得
    /// Get absolute value
    pub fn abs(&self) -> Array<f64, IxDyn> {
        self.data.mapv(|c| c.norm())
    }

    /// 位相を取得
    /// Get phase
    pub fn angle(&self) -> Array<f64, IxDyn> {
        self.data.mapv(|c| c.arg())
    }

    /// 共役複素数を取得
    /// Get complex conjugate
    pub fn conj(&self) -> ComplexTensor {
        let conj_data = self.data.mapv(|c| c.conj());
        let mut result = ComplexTensor::new(conj_data);
        result.requires_grad = self.requires_grad;
        result
    }

    /// 合計
    /// Sum
    pub fn sum(&self) -> Complex64 {
        self.data.sum()
    }

    /// 平均
    /// Mean
    pub fn mean(&self) -> Complex64 {
        let n = self.numel() as f64;
        self.sum() / Complex64::new(n, 0.0)
    }

    /// 次元を追加
    /// Add dimension
    pub fn unsqueeze(&self, dim: usize) -> RusTorchResult<Self> {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape)
    }

    /// 形状を拡張
    /// Expand shape
    pub fn expand(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        let expanded_data = self.data.broadcast(new_shape)
            .ok_or_else(|| crate::error::RusTorchError::tensor_op("Cannot broadcast to new shape"))?
            .to_owned();
        let mut result = ComplexTensor::new(expanded_data);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// 次元を交換
    /// Transpose dimensions
    pub fn transpose_dims(&self, dim1: usize, dim2: usize) -> RusTorchResult<Self> {
        let mut permutation: Vec<usize> = (0..self.ndim()).collect();
        permutation.swap(dim1, dim2);
        let transposed = self.data.clone().permuted_axes(permutation);
        let mut result = ComplexTensor::new(transposed);
        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// フーリエ変換
    /// Fourier transform
    pub fn fft(&self) -> RusTorchResult<Self> {
        // 簡単なFFT実装（1次元のみ）
        // Simple FFT implementation (1D only)
        if self.ndim() != 1 {
            return Err(crate::error::RusTorchError::InvalidOperation(
                "FFT currently only supports 1D tensors".to_string()
            ));
        }

        let n = self.shape[0];
        let mut result = self.data.clone();

        // 再帰的FFT（簡略版）
        // Recursive FFT (simplified version)
        fn fft_recursive(data: &mut [Complex64]) {
            let n = data.len();
            if n <= 1 { return; }

            // 偶数と奇数のインデックスに分ける
            let mut even: Vec<Complex64> = Vec::with_capacity(n / 2);
            let mut odd: Vec<Complex64> = Vec::with_capacity(n / 2);

            for i in 0..n {
                if i % 2 == 0 {
                    even.push(data[i]);
                } else {
                    odd.push(data[i]);
                }
            }

            fft_recursive(&mut even);
            fft_recursive(&mut odd);

            use std::f64::consts::PI;
            for i in 0..n/2 {
                let t = Complex64::from_polar(1.0, -2.0 * PI * i as f64 / n as f64) * odd[i];
                data[i] = even[i] + t;
                data[i + n/2] = even[i] - t;
            }
        }

        let mut data_vec = result.as_slice_mut().unwrap().to_vec();
        fft_recursive(&mut data_vec);

        for (i, &val) in data_vec.iter().enumerate() {
            result[i] = val;
        }

        let mut tensor_result = ComplexTensor::new(result);
        tensor_result.requires_grad = self.requires_grad;
        Ok(tensor_result)
    }

    /// 逆フーリエ変換
    /// Inverse Fourier transform
    pub fn ifft(&self) -> RusTorchResult<Self> {
        // 共役 -> FFT -> 共役 -> スケール
        let conj = self.conj();
        let fft_result = conj.fft()?;
        let result_conj = fft_result.conj();
        let n = self.numel() as f64;
        let scaled = &result_conj * Complex64::new(1.0 / n, 0.0);
        Ok(scaled)
    }
}

// インデックス実装
impl Index<usize> for ComplexTensor {
    type Output = Complex64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for ComplexTensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Index<Index2D> for ComplexTensor {
    type Output = Complex64;
    fn index(&self, index: Index2D) -> &Self::Output {
        &self.data[[index.0, index.1]]
    }
}

impl IndexMut<Index2D> for ComplexTensor {
    fn index_mut(&mut self, index: Index2D) -> &mut Self::Output {
        &mut self.data[[index.0, index.1]]
    }
}

impl Index<Index3D> for ComplexTensor {
    type Output = Complex64;
    fn index(&self, index: Index3D) -> &Self::Output {
        &self.data[[index.0, index.1, index.2]]
    }
}

impl IndexMut<Index3D> for ComplexTensor {
    fn index_mut(&mut self, index: Index3D) -> &mut Self::Output {
        &mut self.data[[index.0, index.1, index.2]]
    }
}
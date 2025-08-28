//! Complex tensor operations and utility functions
//! 複素テンソル演算とユーティリティ関数

use crate::tensor::Tensor;
use num_traits::{Float, FromPrimitive, One, Zero};

use super::core::Complex;

/// Complex tensor creation and manipulation functions
impl<T: Float + 'static> Complex<T> {
    /// Create a complex tensor from real and imaginary parts
    /// 実部と虚部からの複素テンソルを作成
    pub fn from_tensors(real: &Tensor<T>, imag: &Tensor<T>) -> Result<Tensor<Complex<T>>, String> {
        if real.shape() != imag.shape() {
            return Err("Real and imaginary tensors must have the same shape".to_string());
        }

        let mut complex_data = Vec::with_capacity(real.data.len());
        for (r, i) in real.data.iter().zip(imag.data.iter()) {
            complex_data.push(Complex::new(*r, *i));
        }

        Ok(Tensor::from_vec(complex_data, real.shape().to_vec()))
    }

    /// Extract real parts from complex tensor
    /// 複素テンソルから実部を抽出
    pub fn tensor_real_part(tensor: &Tensor<Complex<T>>) -> Tensor<T> {
        let real_data: Vec<T> = tensor.data.iter().map(|z| z.real()).collect();
        Tensor::from_vec(real_data, tensor.shape().to_vec())
    }

    /// Extract imaginary parts from complex tensor
    /// 複素テンソルから虚部を抽出
    pub fn tensor_imag_part(tensor: &Tensor<Complex<T>>) -> Tensor<T> {
        let imag_data: Vec<T> = tensor.data.iter().map(|z| z.imag()).collect();
        Tensor::from_vec(imag_data, tensor.shape().to_vec())
    }

    /// Extract magnitude from complex tensor
    /// 複素テンソルから絶対値を抽出
    pub fn tensor_abs(tensor: &Tensor<Complex<T>>) -> Tensor<T> {
        let abs_data: Vec<T> = tensor.data.iter().map(|z| z.abs()).collect();
        Tensor::from_vec(abs_data, tensor.shape().to_vec())
    }

    /// Extract phase from complex tensor
    /// 複素テンソルから位相角を抽出
    pub fn tensor_arg(tensor: &Tensor<Complex<T>>) -> Tensor<T> {
        let arg_data: Vec<T> = tensor.data.iter().map(|z| z.arg()).collect();
        Tensor::from_vec(arg_data, tensor.shape().to_vec())
    }

    /// Complex conjugate of tensor
    /// テンソルの複素共役
    pub fn tensor_conj(tensor: &Tensor<Complex<T>>) -> Tensor<Complex<T>> {
        let conj_data: Vec<Complex<T>> = tensor.data.iter().map(|z| z.conj()).collect();
        Tensor::from_vec(conj_data, tensor.shape().to_vec())
    }
}

/// Complex tensor factory functions
impl<T: Float + 'static> Tensor<Complex<T>> {
    /// Create a complex tensor with all zeros
    /// すべてゼロの複素テンソルを作成
    pub fn complex_zeros(shape: &[usize]) -> Self {
        let total_size = shape.iter().product();
        let data = vec![Complex::zero(); total_size];
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Create a complex tensor with all ones
    /// すべて1の複素テンソルを作成
    pub fn complex_ones(shape: &[usize]) -> Self {
        let total_size = shape.iter().product();
        let data = vec![Complex::one(); total_size];
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Create complex tensor filled with imaginary unit
    /// 虚数単位で満たされた複素テンソルを作成
    pub fn complex_i(shape: &[usize]) -> Self {
        let total_size = shape.iter().product();
        let data = vec![Complex::i(); total_size];
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Create complex tensor from polar coordinates
    /// 極座標から複素テンソルを作成
    pub fn from_polar(magnitude: &Tensor<T>, phase: &Tensor<T>) -> Result<Self, String> {
        if magnitude.shape() != phase.shape() {
            return Err("Magnitude and phase tensors must have the same shape".to_string());
        }

        let mut complex_data = Vec::with_capacity(magnitude.data.len());
        for (mag, ph) in magnitude.data.iter().zip(phase.data.iter()) {
            complex_data.push(Complex::from_polar(*mag, *ph));
        }

        Ok(Tensor::from_vec(complex_data, magnitude.shape().to_vec()))
    }

    /// Create complex tensor from real tensor (imaginary part = 0)
    /// 実テンソルから複素テンソルを作成（虚部は0）
    pub fn from_real(real: &Tensor<T>) -> Self {
        let complex_data: Vec<Complex<T>> =
            real.data.iter().map(|&r| Complex::from_real(r)).collect();
        Tensor::from_vec(complex_data, real.shape().to_vec())
    }

    /// Create complex tensor from imaginary tensor (real part = 0)
    /// 虚テンソルから複素テンソルを作成（実部は0）
    pub fn from_imag(imag: &Tensor<T>) -> Self {
        let complex_data: Vec<Complex<T>> =
            imag.data.iter().map(|&i| Complex::from_imag(i)).collect();
        Tensor::from_vec(complex_data, imag.shape().to_vec())
    }
}

/// Complex mathematical operations for tensors
impl<T: Float + 'static> Tensor<Complex<T>> {
    /// Element-wise exponential e^z for complex tensor
    /// 複素テンソルの要素ごとの指数関数 e^z
    pub fn exp(&self) -> Self {
        let exp_data: Vec<Complex<T>> = self.data.iter().map(|z| z.exp()).collect();
        Tensor::from_vec(exp_data, self.shape().to_vec())
    }

    /// Element-wise natural logarithm ln(z) for complex tensor
    /// 複素テンソルの要素ごとの自然対数 ln(z)
    pub fn ln(&self) -> Self {
        let ln_data: Vec<Complex<T>> = self.data.iter().map(|z| z.ln()).collect();
        Tensor::from_vec(ln_data, self.shape().to_vec())
    }

    /// Element-wise power z^w for complex tensor
    /// 複素テンソルの要素ごとのべき乗 z^w
    pub fn pow(&self, exponent: &Self) -> Result<Self, String> {
        if self.shape() != exponent.shape() {
            return Err("Shape mismatch for power operation".to_string());
        }

        let pow_data: Vec<Complex<T>> = self
            .data
            .iter()
            .zip(exponent.data.iter())
            .map(|(z, exp)| z.pow(*exp))
            .collect();

        Ok(Tensor::from_vec(pow_data, self.shape().to_vec()))
    }

    /// Element-wise power with scalar exponent z^s
    /// スカラー指数でのべき乗 z^s
    pub fn pow_scalar(&self, exponent: Complex<T>) -> Self {
        let pow_data: Vec<Complex<T>> = self.data.iter().map(|z| z.pow(exponent)).collect();
        Tensor::from_vec(pow_data, self.shape().to_vec())
    }

    /// Element-wise square root √z for complex tensor
    /// 複素テンソルの要素ごとの平方根 √z
    pub fn sqrt(&self) -> Self {
        let sqrt_data: Vec<Complex<T>> = self.data.iter().map(|z| z.sqrt()).collect();
        Tensor::from_vec(sqrt_data, self.shape().to_vec())
    }

    /// Element-wise sine sin(z) for complex tensor
    /// 複素テンソルの要素ごとの正弦 sin(z)
    pub fn sin(&self) -> Self {
        let sin_data: Vec<Complex<T>> = self.data.iter().map(|z| z.sin()).collect();
        Tensor::from_vec(sin_data, self.shape().to_vec())
    }

    /// Element-wise cosine cos(z) for complex tensor
    /// 複素テンソルの要素ごとの余弦 cos(z)
    pub fn cos(&self) -> Self {
        let cos_data: Vec<Complex<T>> = self.data.iter().map(|z| z.cos()).collect();
        Tensor::from_vec(cos_data, self.shape().to_vec())
    }

    /// Element-wise tangent tan(z) for complex tensor
    /// 複素テンソルの要素ごとの正接 tan(z)
    pub fn tan(&self) -> Self {
        let tan_data: Vec<Complex<T>> = self.data.iter().map(|z| z.tan()).collect();
        Tensor::from_vec(tan_data, self.shape().to_vec())
    }

    /// Element-wise hyperbolic sine sinh(z) for complex tensor
    /// 複素テンソルの要素ごとの双曲正弦 sinh(z)
    pub fn sinh(&self) -> Self {
        let sinh_data: Vec<Complex<T>> = self.data.iter().map(|z| z.sinh()).collect();
        Tensor::from_vec(sinh_data, self.shape().to_vec())
    }

    /// Element-wise hyperbolic cosine cosh(z) for complex tensor
    /// 複素テンソルの要素ごとの双曲余弦 cosh(z)
    pub fn cosh(&self) -> Self {
        let cosh_data: Vec<Complex<T>> = self.data.iter().map(|z| z.cosh()).collect();
        Tensor::from_vec(cosh_data, self.shape().to_vec())
    }

    /// Element-wise hyperbolic tangent tanh(z) for complex tensor
    /// 複素テンソルの要素ごとの双曲正接 tanh(z)
    pub fn tanh(&self) -> Self {
        let tanh_data: Vec<Complex<T>> = self.data.iter().map(|z| z.tanh()).collect();
        Tensor::from_vec(tanh_data, self.shape().to_vec())
    }
}
//! Complex number support for tensors
//! テンソルの複素数サポート

use num_traits::{Float, Zero, One, Num, NumCast, ToPrimitive};
use std::ops::{Add, Sub, Mul, Div, Neg, Rem};
use std::fmt::{Debug, Display};
use crate::tensor::Tensor;
use approx::{AbsDiffEq, RelativeEq, UlpsEq};

/// Complex number type for tensor operations
/// テンソル演算用複素数型
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Complex<T: Float> {
    /// Real part / 実部
    pub re: T,
    /// Imaginary part / 虚部
    pub im: T,
}

impl<T: Float> Complex<T> {
    /// Create a new complex number
    /// 新しい複素数を作成
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
    
    /// Create a complex number from real part only
    /// 実部のみから複素数を作成
    pub fn from_real(re: T) -> Self {
        Self::new(re, T::zero())
    }
    
    /// Create a complex number from imaginary part only
    /// 虚部のみから複素数を作成
    pub fn from_imag(im: T) -> Self {
        Self::new(T::zero(), im)
    }
    
    /// Get the real part
    /// 実部を取得
    pub fn real(&self) -> T {
        self.re
    }
    
    /// Get the imaginary part
    /// 虚部を取得
    pub fn imag(&self) -> T {
        self.im
    }
    
    /// Complex conjugate
    /// 複素共役
    pub fn conj(&self) -> Self {
        Self::new(self.re, -self.im)
    }
    
    /// Magnitude (absolute value) |z|
    /// 絶対値 |z|
    pub fn abs(&self) -> T {
        (self.re * self.re + self.im * self.im).sqrt()
    }
    
    /// Magnitude squared |z|²
    /// 絶対値の二乗 |z|²
    pub fn abs_sq(&self) -> T {
        self.re * self.re + self.im * self.im
    }
    
    /// Phase (argument) arg(z)
    /// 位相角 arg(z)
    pub fn arg(&self) -> T {
        self.im.atan2(self.re)
    }
    
    /// Convert to polar form (magnitude, phase)
    /// 極形式に変換 (絶対値, 位相角)
    pub fn to_polar(&self) -> (T, T) {
        (self.abs(), self.arg())
    }
    
    /// Create from polar form
    /// 極形式から作成
    pub fn from_polar(r: T, theta: T) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }
    
    /// Check if the complex number is finite
    /// 複素数が有限かチェック
    pub fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    
    /// Check if the complex number is infinite
    /// 複素数が無限大かチェック
    pub fn is_infinite(&self) -> bool {
        self.re.is_infinite() || self.im.is_infinite()
    }
    
    /// Check if the complex number is NaN
    /// 複素数がNaNかチェック
    pub fn is_nan(&self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    
    /// Check if this is a real number (imaginary part is zero)
    /// 実数かチェック（虚部がゼロ）
    pub fn is_real(&self) -> bool {
        self.im == T::zero()
    }
    
    /// Check if this is purely imaginary (real part is zero)
    /// 純虚数かチェック（実部がゼロ）
    pub fn is_imag(&self) -> bool {
        self.re == T::zero()
    }
}

// Basic arithmetic operations
impl<T: Float> Add for Complex<T> {
    type Output = Self;
    
    fn add(self, other: Self) -> Self::Output {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self::Output {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self::Output {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

impl<T: Float> Div for Complex<T> {
    type Output = Self;
    
    fn div(self, other: Self) -> Self::Output {
        let denom = other.abs_sq();
        if denom == T::zero() {
            // Division by zero - return infinity
            Self::new(T::infinity(), T::infinity())
        } else {
            let conj = other.conj();
            let num = self * conj;
            Self::new(num.re / denom, num.im / denom)
        }
    }
}

impl<T: Float> Neg for Complex<T> {
    type Output = Self;
    
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

// Scalar operations
impl<T: Float> Add<T> for Complex<T> {
    type Output = Self;
    
    fn add(self, scalar: T) -> Self::Output {
        Self::new(self.re + scalar, self.im)
    }
}

impl<T: Float> Sub<T> for Complex<T> {
    type Output = Self;
    
    fn sub(self, scalar: T) -> Self::Output {
        Self::new(self.re - scalar, self.im)
    }
}

impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;
    
    fn mul(self, scalar: T) -> Self::Output {
        Self::new(self.re * scalar, self.im * scalar)
    }
}

impl<T: Float> Div<T> for Complex<T> {
    type Output = Self;
    
    fn div(self, scalar: T) -> Self::Output {
        if scalar == T::zero() {
            Self::new(T::infinity(), T::infinity())
        } else {
            Self::new(self.re / scalar, self.im / scalar)
        }
    }
}

// Remainder operation for complex numbers (approximation)
impl<T: Float> Rem for Complex<T> {
    type Output = Self;
    
    fn rem(self, other: Self) -> Self::Output {
        // Complex modulo is not well-defined, use real part approximation
        Self::new(self.re % other.re, self.im % other.im)
    }
}

impl<T: Float> Rem<T> for Complex<T> {
    type Output = Self;
    
    fn rem(self, scalar: T) -> Self::Output {
        Self::new(self.re % scalar, self.im % scalar)
    }
}

// Trait implementations
impl<T: Float> Zero for Complex<T> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }
    
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
}

impl<T: Float> One for Complex<T> {
    fn one() -> Self {
        Self::new(T::one(), T::zero())
    }
}

// PartialOrd implementation based on magnitude
impl<T: Float> PartialOrd for Complex<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let self_mag = self.re * self.re + self.im * self.im;
        let other_mag = other.re * other.re + other.im * other.im;
        self_mag.partial_cmp(&other_mag)
    }
}

// Implement Num trait
impl<T: Float> Num for Complex<T> {
    type FromStrRadixErr = <T as Num>::FromStrRadixErr;
    
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let val = T::from_str_radix(str, radix)?;
        Ok(Self::from_real(val))
    }
}

// Implement ToPrimitive trait (use magnitude)
impl<T: Float + ToPrimitive> ToPrimitive for Complex<T> {
    fn to_i64(&self) -> Option<i64> {
        self.abs().to_i64()
    }
    
    fn to_u64(&self) -> Option<u64> {
        self.abs().to_u64()
    }
    
    fn to_f64(&self) -> Option<f64> {
        self.abs().to_f64()
    }
    
    fn to_f32(&self) -> Option<f32> {
        self.abs().to_f32()
    }
}

// Implement NumCast trait
impl<T: Float + NumCast> NumCast for Complex<T> {
    fn from<N: ToPrimitive>(n: N) -> Option<Self> {
        T::from(n).map(|t| Self::from_real(t))
    }
}

// Implement Float trait for Complex numbers (approximation)
impl<T: Float> Float for Complex<T> {
    fn nan() -> Self {
        Self::new(T::nan(), T::nan())
    }
    
    fn infinity() -> Self {
        Self::new(T::infinity(), T::zero())
    }
    
    fn neg_infinity() -> Self {
        Self::new(T::neg_infinity(), T::zero())
    }
    
    fn neg_zero() -> Self {
        Self::new(T::neg_zero(), T::zero())
    }
    
    fn min_value() -> Self {
        Self::new(T::min_value(), T::zero())
    }
    
    fn min_positive_value() -> Self {
        Self::new(T::min_positive_value(), T::zero())
    }
    
    fn max_value() -> Self {
        Self::new(T::max_value(), T::zero())
    }
    
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
    
    fn is_infinite(self) -> bool {
        self.re.is_infinite() || self.im.is_infinite()
    }
    
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    
    fn is_normal(self) -> bool {
        self.re.is_normal() && self.im.is_normal()
    }
    
    fn classify(self) -> std::num::FpCategory {
        // Use the real part for classification
        self.re.classify()
    }
    
    fn floor(self) -> Self {
        Self::new(self.re.floor(), self.im.floor())
    }
    
    fn ceil(self) -> Self {
        Self::new(self.re.ceil(), self.im.ceil())
    }
    
    fn round(self) -> Self {
        Self::new(self.re.round(), self.im.round())
    }
    
    fn trunc(self) -> Self {
        Self::new(self.re.trunc(), self.im.trunc())
    }
    
    fn fract(self) -> Self {
        Self::new(self.re.fract(), self.im.fract())
    }
    
    fn abs(self) -> Self {
        // For Float trait, return the complex magnitude as a real number
        let magnitude = (self.re * self.re + self.im * self.im).sqrt();
        Self::new(magnitude, T::zero())
    }
    
    fn signum(self) -> Self {
        let magnitude = (self.re * self.re + self.im * self.im).sqrt();
        if magnitude == T::zero() {
            Self::zero()
        } else {
            Self::new(self.re / magnitude, self.im / magnitude)
        }
    }
    
    fn is_sign_positive(self) -> bool {
        self.re.is_sign_positive()
    }
    
    fn is_sign_negative(self) -> bool {
        self.re.is_sign_negative()
    }
    
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    
    fn recip(self) -> Self {
        Self::one() / self
    }
    
    fn powi(self, n: i32) -> Self {
        if self.is_zero() {
            if n == 0 {
                Self::one() // 0^0 = 1 by convention
            } else {
                Self::zero()
            }
        } else {
            let exp = Complex::from_real(T::from(n).unwrap());
            (self.ln() * exp).exp()
        }
    }
    
    fn powf(self, n: Self) -> Self {
        if self.is_zero() {
            if n.is_zero() {
                Self::one() // 0^0 = 1 by convention
            } else {
                Self::zero()
            }
        } else {
            (self.ln() * n).exp()
        }
    }
    
    fn sqrt(self) -> Self {
        let r = (self.re * self.re + self.im * self.im).sqrt();
        let theta = self.im.atan2(self.re);
        let sqrt_r = r.sqrt();
        let half_theta = theta / (T::one() + T::one());
        Self::new(sqrt_r * half_theta.cos(), sqrt_r * half_theta.sin())
    }
    
    fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(exp_re * self.im.cos(), exp_re * self.im.sin())
    }
    
    fn exp2(self) -> Self {
        let ln2 = T::from(std::f64::consts::LN_2).unwrap();
        (self * ln2).exp()
    }
    
    fn ln(self) -> Self {
        let magnitude = (self.re * self.re + self.im * self.im).sqrt();
        let phase = self.im.atan2(self.re);
        Self::new(magnitude.ln(), phase)
    }
    
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }
    
    fn log2(self) -> Self {
        let ln2 = T::from(std::f64::consts::LN_2).unwrap();
        self.ln() / Complex::from_real(ln2)
    }
    
    fn log10(self) -> Self {
        let ln10 = T::from(std::f64::consts::LN_10).unwrap();
        self.ln() / Complex::from_real(ln10)
    }
    
    fn max(self, other: Self) -> Self {
        if self.abs() >= other.abs() { self } else { other }
    }
    
    fn min(self, other: Self) -> Self {
        if self.abs() <= other.abs() { self } else { other }
    }
    
    fn abs_sub(self, other: Self) -> Self {
        if self.abs() >= other.abs() {
            self - other
        } else {
            Self::zero()
        }
    }
    
    fn cbrt(self) -> Self {
        let one_third = Complex::from_real(T::one() / T::from(3.0).unwrap());
        self.powf(one_third)
    }
    
    fn hypot(self, other: Self) -> Self {
        (self * self + other * other).sqrt()
    }
    
    fn sin(self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }
    
    fn cos(self) -> Self {
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }
    
    fn tan(self) -> Self {
        let sin_val = self.sin();
        let cos_val = self.cos();
        sin_val / cos_val
    }
    
    fn asin(self) -> Self {
        let i: Self = Complex::i();
        -i * (i * self + (Self::one() - self * self).sqrt()).ln()
    }
    
    fn acos(self) -> Self {
        let i: Self = Complex::i();
        -i * (self + i * (Self::one() - self * self).sqrt()).ln()
    }
    
    fn atan(self) -> Self {
        let i: Self = Complex::i();
        let two = T::from(2.0).unwrap();
        (i / two) * ((i + self) / (i - self)).ln()
    }
    
    fn atan2(self, other: Self) -> Self {
        (self / other).atan()
    }
    
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
    
    fn exp_m1(self) -> Self {
        self.exp() - Self::one()
    }
    
    fn ln_1p(self) -> Self {
        (self + Self::one()).ln()
    }
    
    fn sinh(self) -> Self {
        Self::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }
    
    fn cosh(self) -> Self {
        Self::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }
    
    fn tanh(self) -> Self {
        let sinh_val = self.sinh();
        let cosh_val = self.cosh();
        sinh_val / cosh_val
    }
    
    fn asinh(self) -> Self {
        (self + (self * self + Self::one()).sqrt()).ln()
    }
    
    fn acosh(self) -> Self {
        (self + (self * self - Self::one()).sqrt()).ln()
    }
    
    fn atanh(self) -> Self {
        let two = T::from(2.0).unwrap();
        ((Self::one() + self) / (Self::one() - self)).ln() / two
    }
    
    fn integer_decode(self) -> (u64, i16, i8) {
        // Use the real part for integer decode
        self.re.integer_decode()
    }
}

// AbsDiffEq implementation for approx testing
impl<T: Float + AbsDiffEq> AbsDiffEq for Complex<T>
where 
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;
    
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }
    
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.re.abs_diff_eq(&other.re, epsilon.clone()) && 
        self.im.abs_diff_eq(&other.im, epsilon)
    }
}

// RelativeEq implementation for approx testing
impl<T: Float + RelativeEq> RelativeEq for Complex<T>
where 
    T::Epsilon: Clone,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }
    
    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
        self.re.relative_eq(&other.re, epsilon.clone(), max_relative.clone()) && 
        self.im.relative_eq(&other.im, epsilon, max_relative)
    }
}

// UlpsEq implementation for approx testing
impl<T: Float + UlpsEq> UlpsEq for Complex<T>
where 
    T::Epsilon: Clone,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }
    
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.re.ulps_eq(&other.re, epsilon.clone(), max_ulps) && 
        self.im.ulps_eq(&other.im, epsilon, max_ulps)
    }
}

impl<T: Float + Display> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.im >= T::zero() {
            write!(f, "{}+{}i", self.re, self.im)
        } else {
            write!(f, "{}{}i", self.re, self.im)
        }
    }
}

// Constants
impl<T: Float> Complex<T> {
    /// Complex zero (0 + 0i)
    /// 複素数のゼロ (0 + 0i)
    pub fn zero_const() -> Self {
        Self { re: T::zero(), im: T::zero() }
    }
    
    /// Complex one (1 + 0i)
    /// 複素数の1 (1 + 0i)
    pub fn one_const() -> Self {
        Self { re: T::one(), im: T::zero() }
    }
    
    /// Imaginary unit (0 + 1i)
    /// 虚数単位 (0 + 1i)
    pub fn i() -> Self {
        Self { re: T::zero(), im: T::one() }
    }
}

// Mathematical functions
impl<T: Float> Complex<T> {
    /// Complex exponential e^z
    /// 複素指数関数 e^z
    pub fn exp(&self) -> Self {
        let exp_re = self.re.exp();
        Self::new(exp_re * self.im.cos(), exp_re * self.im.sin())
    }
    
    /// Natural logarithm ln(z)
    /// 自然対数 ln(z)
    pub fn ln(&self) -> Self {
        Self::new(self.abs().ln(), self.arg())
    }
    
    /// Complex power z^w
    /// 複素べき乗 z^w
    pub fn pow(&self, exp: Self) -> Self {
        if self.is_zero() {
            if exp.is_zero() {
                Self::one() // 0^0 = 1 by convention
            } else {
                Self::zero()
            }
        } else {
            (self.ln() * exp).exp()
        }
    }
    
    /// Square root √z
    /// 平方根 √z
    pub fn sqrt(&self) -> Self {
        let r = self.abs();
        let theta = self.arg();
        Self::from_polar(r.sqrt(), theta / (T::one() + T::one()))
    }
    
    /// Sine sin(z)
    /// 正弦 sin(z)
    pub fn sin(&self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }
    
    /// Cosine cos(z)
    /// 余弦 cos(z)
    pub fn cos(&self) -> Self {
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }
    
    /// Tangent tan(z)
    /// 正接 tan(z)
    pub fn tan(&self) -> Self {
        self.sin() / self.cos()
    }
    
    /// Hyperbolic sine sinh(z)
    /// 双曲正弦 sinh(z)
    pub fn sinh(&self) -> Self {
        Self::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }
    
    /// Hyperbolic cosine cosh(z)
    /// 双曲余弦 cosh(z)
    pub fn cosh(&self) -> Self {
        Self::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }
    
    /// Hyperbolic tangent tanh(z)
    /// 双曲正接 tanh(z)
    pub fn tanh(&self) -> Self {
        self.sinh() / self.cosh()
    }
}

// Conversion traits
impl<T: Float> From<T> for Complex<T> {
    fn from(re: T) -> Self {
        Self::from_real(re)
    }
}

impl<T: Float> From<(T, T)> for Complex<T> {
    fn from((re, im): (T, T)) -> Self {
        Self::new(re, im)
    }
}

impl<T: Float> From<Complex<T>> for (T, T) {
    fn from(z: Complex<T>) -> Self {
        (z.re, z.im)
    }
}

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
        let complex_data: Vec<Complex<T>> = real.data.iter().map(|&r| Complex::from_real(r)).collect();
        Tensor::from_vec(complex_data, real.shape().to_vec())
    }
    
    /// Create complex tensor from imaginary tensor (real part = 0)
    /// 虚テンソルから複素テンソルを作成（実部は0）
    pub fn from_imag(imag: &Tensor<T>) -> Self {
        let complex_data: Vec<Complex<T>> = imag.data.iter().map(|&i| Complex::from_imag(i)).collect();
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
        
        let pow_data: Vec<Complex<T>> = self.data.iter()
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
    
    /// Complex matrix multiplication
    /// 複素行列の乗算
    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        // Support 2D matrix multiplication for now
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(format!(
                "Complex matmul currently supports only 2D matrices, got {}D @ {}D",
                self.ndim(), other.ndim()
            ));
        }
        
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        if self_shape[1] != other_shape[0] {
            return Err(format!(
                "Complex matrix dimension mismatch: {}x{} @ {}x{}",
                self_shape[0], self_shape[1], other_shape[0], other_shape[1]
            ));
        }
        
        let m = self_shape[0];
        let n = other_shape[1];
        let k = self_shape[1];
        
        let mut result = vec![Complex::zero(); m * n];
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex::zero();
                for l in 0..k {
                    let a_idx = i * k + l;
                    let b_idx = l * n + j;
                    sum = sum + self.data[a_idx] * other.data[b_idx];
                }
                result[i * n + j] = sum;
            }
        }
        
        Ok(Tensor::from_vec(result, vec![m, n]))
    }
    
    /// Complex matrix transpose
    /// 複素行列の転置
    pub fn transpose(&self) -> Result<Self, String> {
        if self.ndim() != 2 {
            return Err("Transpose currently supports only 2D matrices".to_string());
        }
        
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];
        
        let mut result = vec![Complex::zero(); rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                let src_idx = i * cols + j;
                let dst_idx = j * rows + i;
                result[dst_idx] = self.data[src_idx];
            }
        }
        
        Ok(Tensor::from_vec(result, vec![cols, rows]))
    }
    
    /// Complex matrix conjugate transpose (Hermitian transpose)
    /// 複素行列の共役転置（エルミート転置）
    pub fn conj_transpose(&self) -> Result<Self, String> {
        if self.ndim() != 2 {
            return Err("Conjugate transpose currently supports only 2D matrices".to_string());
        }
        
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];
        
        let mut result = vec![Complex::zero(); rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                let src_idx = i * cols + j;
                let dst_idx = j * rows + i;
                result[dst_idx] = self.data[src_idx].conj();
            }
        }
        
        Ok(Tensor::from_vec(result, vec![cols, rows]))
    }
    
    /// Complex matrix trace (sum of diagonal elements)
    /// 複素行列のトレース（対角要素の和）
    pub fn trace(&self) -> Result<Complex<T>, String> {
        if self.ndim() != 2 {
            return Err("Trace requires a 2D matrix".to_string());
        }
        
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];
        let min_dim = rows.min(cols);
        
        let mut sum = Complex::zero();
        for i in 0..min_dim {
            let idx = i * cols + i;
            sum = sum + self.data[idx];
        }
        
        Ok(sum)
    }
    
    /// Complex matrix determinant (2x2 only for now)
    /// 複素行列の行列式（現在は2x2のみ）
    pub fn determinant(&self) -> Result<Complex<T>, String> {
        if self.ndim() != 2 {
            return Err("Determinant requires a 2D matrix".to_string());
        }
        
        let shape = self.shape();
        if shape[0] != shape[1] {
            return Err("Determinant requires a square matrix".to_string());
        }
        
        if shape[0] == 1 {
            return Ok(self.data[0]);
        } else if shape[0] == 2 {
            let a = self.data[0];  // [0,0]
            let b = self.data[1];  // [0,1]
            let c = self.data[2];  // [1,0]
            let d = self.data[3];  // [1,1]
            return Ok(a * d - b * c);
        } else {
            return Err("Determinant only implemented for 1x1 and 2x2 matrices currently".to_string());
        }
    }
    
    /// Forward FFT for complex tensor
    /// 複素テンソルの順フーリエ変換
    pub fn fft(&self, n: Option<usize>, dim: Option<isize>, norm: Option<&str>) -> Result<Self, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        if self.ndim() != 1 {
            return Err("Complex FFT currently supports only 1D tensors".to_string());
        }
        
        let input_len = self.shape()[0];
        let fft_len = n.unwrap_or(input_len);
        
        // Convert to Vec<Complex<T>> for FFT processing
        let mut fft_data: Vec<Complex<T>> = self.data.iter().cloned().collect();
        
        // Pad or truncate to desired length
        if fft_data.len() != fft_len {
            fft_data.resize(fft_len, Complex::zero());
        }
        
        // Apply FFT algorithm
        let result = if fft_len.is_power_of_two() {
            self.cooley_tukey_complex(&mut fft_data, false)?
        } else {
            self.dft_complex(&fft_data, false)?
        };
        
        // Apply normalization
        let normalized = self.apply_complex_normalization(result, fft_len, norm, false)?;
        
        Ok(Tensor::from_vec(normalized, vec![fft_len]))
    }
    
    /// Inverse FFT for complex tensor
    /// 複素テンソルの逆フーリエ変換
    pub fn ifft(&self, n: Option<usize>, dim: Option<isize>, norm: Option<&str>) -> Result<Self, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        if self.ndim() != 1 {
            return Err("Complex IFFT currently supports only 1D tensors".to_string());
        }
        
        let input_len = self.shape()[0];
        let fft_len = n.unwrap_or(input_len);
        
        // Convert to Vec<Complex<T>> for IFFT processing
        let mut fft_data: Vec<Complex<T>> = self.data.iter().cloned().collect();
        
        // Pad or truncate to desired length
        if fft_data.len() != fft_len {
            fft_data.resize(fft_len, Complex::zero());
        }
        
        // Apply IFFT algorithm
        let result = if fft_len.is_power_of_two() {
            self.cooley_tukey_complex(&mut fft_data, true)?
        } else {
            self.dft_complex(&fft_data, true)?
        };
        
        // Apply normalization
        let normalized = self.apply_complex_normalization(result, fft_len, norm, true)?;
        
        Ok(Tensor::from_vec(normalized, vec![fft_len]))
    }
    
    /// FFT shift for complex tensor
    /// 複素テンソルのFFTシフト
    pub fn fftshift(&self, dim: Option<&[isize]>) -> Result<Self, String> {
        if self.ndim() != 1 {
            return Err("Complex fftshift currently supports only 1D tensors".to_string());
        }
        
        let input_data: Vec<Complex<T>> = self.data.iter().cloned().collect();
        let input_len = input_data.len();
        let mid = (input_len + 1) / 2;
        let mut new_data = Vec::with_capacity(input_len);
        
        // Shift: second half to front, first half to back
        new_data.extend_from_slice(&input_data[mid..]);
        new_data.extend_from_slice(&input_data[..mid]);
        
        Ok(Tensor::from_vec(new_data, self.shape().to_vec()))
    }
    
    /// Inverse FFT shift for complex tensor
    /// 複素テンソルの逆FFTシフト
    pub fn ifftshift(&self, dim: Option<&[isize]>) -> Result<Self, String> {
        if self.ndim() != 1 {
            return Err("Complex ifftshift currently supports only 1D tensors".to_string());
        }
        
        let input_data: Vec<Complex<T>> = self.data.iter().cloned().collect();
        let input_len = input_data.len();
        let mid = input_len / 2;
        let mut new_data = Vec::with_capacity(input_len);
        
        // Shift: second half to front, first half to back (different split)
        new_data.extend_from_slice(&input_data[mid..]);
        new_data.extend_from_slice(&input_data[..mid]);
        
        Ok(Tensor::from_vec(new_data, self.shape().to_vec()))
    }
    
    // Private helper functions for complex FFT
    fn cooley_tukey_complex(&self, data: &mut [Complex<T>], inverse: bool) -> Result<Vec<Complex<T>>, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        let n = data.len();
        if !n.is_power_of_two() {
            return Err("Cooley-Tukey algorithm requires power of two length".to_string());
        }
        
        // Bit reversal
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                data.swap(i, j);
            }
        }
        
        // Cooley-Tukey FFT
        let mut length = 2;
        while length <= n {
            let half_len = length / 2;
            let angle = if inverse {
                T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap() / T::from(length).unwrap()
            } else {
                -T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap() / T::from(length).unwrap()
            };
            
            let wlen = Complex::new(angle.cos(), angle.sin());
            
            for i in (0..n).step_by(length) {
                let mut w = Complex::one();
                for j in 0..half_len {
                    let u = data[i + j];
                    let v = data[i + j + half_len] * w;
                    data[i + j] = u + v;
                    data[i + j + half_len] = u - v;
                    w = w * wlen;
                }
            }
            length *= 2;
        }
        
        Ok(data.to_vec())
    }
    
    fn dft_complex(&self, data: &[Complex<T>], inverse: bool) -> Result<Vec<Complex<T>>, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        let n = data.len();
        let mut result = vec![Complex::zero(); n];
        
        let sign = if inverse { T::one() } else { -T::one() };
        let pi2 = T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap();
        
        for k in 0..n {
            let mut sum = Complex::zero();
            for j in 0..n {
                let angle = sign * pi2 * T::from(k).unwrap() * T::from(j).unwrap() / T::from(n).unwrap();
                let w = Complex::new(angle.cos(), angle.sin());
                sum = sum + data[j] * w;
            }
            result[k] = sum;
        }
        
        Ok(result)
    }
    
    fn apply_complex_normalization(&self, mut data: Vec<Complex<T>>, n: usize, norm: Option<&str>, inverse: bool) -> Result<Vec<Complex<T>>, String>
    where
        T: Float + 'static + Default + Clone + std::fmt::Debug + num_traits::FromPrimitive,
    {
        match norm {
            Some("forward") => {
                if !inverse {
                    let scale = T::one() / T::from(n).unwrap();
                    for x in &mut data {
                        *x = *x * scale;
                    }
                }
            },
            Some("backward") => {
                if inverse {
                    let scale = T::one() / T::from(n).unwrap();
                    for x in &mut data {
                        *x = *x * scale;
                    }
                }
            },
            Some("ortho") => {
                let scale = T::one() / T::from(n).unwrap().sqrt();
                for x in &mut data {
                    *x = *x * scale;
                }
            },
            _ => {
                if inverse {
                    let scale = T::one() / T::from(n).unwrap();
                    for x in &mut data {
                        *x = *x * scale;
                    }
                }
            }
        }
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_complex_creation() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.real(), 3.0);
        assert_eq!(z.imag(), 4.0);
        
        let real = Complex::from_real(5.0);
        assert_eq!(real, Complex::new(5.0, 0.0));
        
        let imag = Complex::from_imag(2.0);
        assert_eq!(imag, Complex::new(0.0, 2.0));
    }
    
    #[test]
    fn test_complex_arithmetic() {
        let z1 = Complex::new(3.0, 4.0);
        let z2 = Complex::new(1.0, 2.0);
        
        // Addition
        assert_eq!(z1 + z2, Complex::new(4.0, 6.0));
        
        // Subtraction
        assert_eq!(z1 - z2, Complex::new(2.0, 2.0));
        
        // Multiplication
        assert_eq!(z1 * z2, Complex::new(-5.0, 10.0));
        
        // Division
        let div = z1 / z2;
        assert_relative_eq!(div.real(), 2.2, epsilon = 1e-10);
        assert_relative_eq!(div.imag(), -0.4, epsilon = 1e-10);
    }
    
    #[test]
    fn test_complex_properties() {
        let z = Complex::new(3.0, 4.0);
        
        // Magnitude  
        assert_relative_eq!(Complex::abs(&z), 5.0, epsilon = 1e-10);
        
        // Magnitude squared
        assert_relative_eq!(z.abs_sq(), 25.0, epsilon = 1e-10);
        
        // Conjugate
        assert_eq!(z.conj(), Complex::new(3.0, -4.0));
        
        // Phase
        let expected_phase = 4.0_f64.atan2(3.0);
        assert_relative_eq!(z.arg(), expected_phase, epsilon = 1e-10);
    }
    
    #[test]
    fn test_complex_functions() {
        let z = Complex::new(1.0, 1.0);
        
        // Exponential
        let exp_z = z.exp();
        let expected_real = 1.0_f64.exp() * 1.0_f64.cos();
        let expected_imag = 1.0_f64.exp() * 1.0_f64.sin();
        assert_relative_eq!(exp_z.real(), expected_real, epsilon = 1e-10);
        assert_relative_eq!(exp_z.imag(), expected_imag, epsilon = 1e-10);
        
        // Square root
        let sqrt_z = z.sqrt();
        assert_relative_eq!((sqrt_z * sqrt_z).real(), z.real(), epsilon = 1e-10);
        assert_relative_eq!((sqrt_z * sqrt_z).imag(), z.imag(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_polar_conversion() {
        let z = Complex::new(3.0, 4.0);
        let (r, theta) = z.to_polar();
        let z_converted = Complex::from_polar(r, theta);
        
        assert_relative_eq!(z_converted.real(), z.real(), epsilon = 1e-10);
        assert_relative_eq!(z_converted.imag(), z.imag(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_trigonometric_functions() {
        let z = Complex::new(0.5, 0.3);
        
        // Test sin^2 + cos^2 = 1
        let sin_z = z.sin();
        let cos_z = z.cos();
        let identity = sin_z * sin_z + cos_z * cos_z;
        
        assert_relative_eq!(identity.real(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity.imag(), 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_constants() {
        let zero = Complex::<f64>::zero_const();
        assert_eq!(zero.real(), 0.0);
        assert_eq!(zero.imag(), 0.0);
        
        let one = Complex::<f64>::one_const();
        assert_eq!(one.real(), 1.0);
        assert_eq!(one.imag(), 0.0);
        
        let i = Complex::<f64>::i();
        assert_eq!(i.real(), 0.0);
        assert_eq!(i.imag(), 1.0);
    }
    
    #[test]
    fn test_complex_tensor_creation() {
        let real = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let imag = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);
        
        let complex_tensor = Complex::from_tensors(&real, &imag).unwrap();
        assert_eq!(complex_tensor.shape(), &[3]);
        assert_eq!(complex_tensor.data[0].real(), 1.0);
        assert_eq!(complex_tensor.data[0].imag(), 4.0);
        assert_eq!(complex_tensor.data[1].real(), 2.0);
        assert_eq!(complex_tensor.data[1].imag(), 5.0);
        assert_eq!(complex_tensor.data[2].real(), 3.0);
        assert_eq!(complex_tensor.data[2].imag(), 6.0);
    }
    
    #[test]
    fn test_complex_tensor_extraction() {
        let complex_data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let complex_tensor = Tensor::from_vec(complex_data, vec![3]);
        
        let real_part = Complex::tensor_real_part(&complex_tensor);
        assert_eq!(real_part.data.as_slice().unwrap(), &[1.0, 3.0, 5.0]);
        
        let imag_part = Complex::tensor_imag_part(&complex_tensor);
        assert_eq!(imag_part.data.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
        
        let abs_part = Complex::tensor_abs(&complex_tensor);
        assert_relative_eq!(abs_part.data[0], 5.0_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(abs_part.data[1], 25.0_f64.sqrt(), epsilon = 1e-10);
        assert_relative_eq!(abs_part.data[2], 61.0_f64.sqrt(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_complex_tensor_factory_functions() {
        let zeros = Tensor::<Complex<f64>>::complex_zeros(&[2, 3]);
        assert_eq!(zeros.shape(), &[2, 3]);
        for z in zeros.data.iter() {
            assert_eq!(z.real(), 0.0);
            assert_eq!(z.imag(), 0.0);
        }
        
        let ones = Tensor::<Complex<f64>>::complex_ones(&[2, 2]);
        assert_eq!(ones.shape(), &[2, 2]);
        for z in ones.data.iter() {
            assert_eq!(z.real(), 1.0);
            assert_eq!(z.imag(), 0.0);
        }
        
        let i_tensor = Tensor::<Complex<f64>>::complex_i(&[1, 4]);
        assert_eq!(i_tensor.shape(), &[1, 4]);
        for z in i_tensor.data.iter() {
            assert_eq!(z.real(), 0.0);
            assert_eq!(z.imag(), 1.0);
        }
    }
    
    #[test]
    fn test_complex_tensor_polar_conversion() {
        let magnitude = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let phase = Tensor::from_vec(vec![0.0, std::f64::consts::PI / 2.0], vec![2]);
        
        let complex_tensor = Tensor::from_polar(&magnitude, &phase).unwrap();
        assert_eq!(complex_tensor.shape(), &[2]);
        
        assert_relative_eq!(complex_tensor.data[0].real(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(complex_tensor.data[0].imag(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(complex_tensor.data[1].real(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(complex_tensor.data[1].imag(), 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_complex_tensor_conjugate() {
        let complex_data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(-3.0, 4.0),
        ];
        let complex_tensor = Tensor::from_vec(complex_data, vec![2]);
        
        let conj_tensor = Complex::tensor_conj(&complex_tensor);
        assert_eq!(conj_tensor.data[0].real(), 1.0);
        assert_eq!(conj_tensor.data[0].imag(), -2.0);
        assert_eq!(conj_tensor.data[1].real(), -3.0);
        assert_eq!(conj_tensor.data[1].imag(), -4.0);
    }
    
    #[test]
    fn test_complex_mathematical_functions() {
        let complex_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 1.0),
        ];
        let complex_tensor = Tensor::from_vec(complex_data, vec![3]);
        
        // Test exponential
        let exp_result = complex_tensor.exp();
        assert_relative_eq!(exp_result.data[0].real(), std::f64::consts::E, epsilon = 1e-10);
        assert_relative_eq!(exp_result.data[0].imag(), 0.0, epsilon = 1e-10);
        
        // Test natural logarithm
        let ln_result = complex_tensor.ln();
        assert_relative_eq!(ln_result.data[0].real(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ln_result.data[0].imag(), 0.0, epsilon = 1e-10);
        
        // Test square root
        let sqrt_result = complex_tensor.sqrt();
        let sqrt_1_1 = sqrt_result.data[2];
        assert_relative_eq!((sqrt_1_1 * sqrt_1_1).real(), 1.0, epsilon = 1e-10);
        assert_relative_eq!((sqrt_1_1 * sqrt_1_1).imag(), 1.0, epsilon = 1e-10);
        
        // Test trigonometric functions
        let sin_result = complex_tensor.sin();
        let cos_result = complex_tensor.cos();
        
        // Test sin^2 + cos^2 = 1 for complex numbers
        for i in 0..3 {
            let sin_val = sin_result.data[i];
            let cos_val = cos_result.data[i];
            let identity = sin_val * sin_val + cos_val * cos_val;
            assert_relative_eq!(identity.real(), 1.0, epsilon = 1e-10);
            assert_relative_eq!(identity.imag(), 0.0, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_complex_matrix_multiplication() {
        // Create 2x2 complex matrices
        let a_data = vec![
            Complex::new(1.0, 1.0), Complex::new(2.0, 0.0),  // First row
            Complex::new(0.0, 1.0), Complex::new(1.0, -1.0), // Second row
        ];
        let a = Tensor::from_vec(a_data, vec![2, 2]);
        
        let b_data = vec![
            Complex::new(1.0, 0.0), Complex::new(0.0, 1.0),  // First row
            Complex::new(1.0, 1.0), Complex::new(1.0, 0.0),  // Second row
        ];
        let b = Tensor::from_vec(b_data, vec![2, 2]);
        
        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        
        // Verify matrix multiplication result
        // [1+i, 2] * [1, i] = [1+i+2+2i, i-1+2] = [3+3i, 1+i]
        // [i, 1-i]   [1+i, 1]   [i+1-i-i^2, -1+i] = [1+i, -1+i]
        assert_relative_eq!(result.data[0].real(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[0].imag(), 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[1].real(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.data[1].imag(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_complex_matrix_transpose() {
        let data = vec![
            Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0), Complex::new(7.0, 8.0),
        ];
        let matrix = Tensor::from_vec(data, vec![2, 2]);
        
        let transposed = matrix.transpose().unwrap();
        assert_eq!(transposed.shape(), &[2, 2]);
        
        // Check transposition
        assert_eq!(transposed.data[0], Complex::new(1.0, 2.0)); // [0,0] -> [0,0]
        assert_eq!(transposed.data[1], Complex::new(5.0, 6.0)); // [1,0] -> [0,1]
        assert_eq!(transposed.data[2], Complex::new(3.0, 4.0)); // [0,1] -> [1,0]
        assert_eq!(transposed.data[3], Complex::new(7.0, 8.0)); // [1,1] -> [1,1]
    }
    
    #[test]
    fn test_complex_matrix_conjugate_transpose() {
        let data = vec![
            Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0), Complex::new(7.0, 8.0),
        ];
        let matrix = Tensor::from_vec(data, vec![2, 2]);
        
        let conj_transposed = matrix.conj_transpose().unwrap();
        assert_eq!(conj_transposed.shape(), &[2, 2]);
        
        // Check conjugate transposition
        assert_eq!(conj_transposed.data[0], Complex::new(1.0, -2.0)); // [0,0] -> conj([0,0])
        assert_eq!(conj_transposed.data[1], Complex::new(5.0, -6.0)); // [1,0] -> conj([0,1])
        assert_eq!(conj_transposed.data[2], Complex::new(3.0, -4.0)); // [0,1] -> conj([1,0])
        assert_eq!(conj_transposed.data[3], Complex::new(7.0, -8.0)); // [1,1] -> conj([1,1])
    }
    
    #[test]
    fn test_complex_matrix_trace() {
        let data = vec![
            Complex::new(1.0, 1.0), Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0), Complex::new(4.0, 2.0),
        ];
        let matrix = Tensor::from_vec(data, vec![2, 2]);
        
        let trace = matrix.trace().unwrap();
        // Trace = (1+i) + (4+2i) = 5+3i
        assert_eq!(trace.real(), 5.0);
        assert_eq!(trace.imag(), 3.0);
    }
    
    #[test]
    fn test_complex_matrix_determinant() {
        let data = vec![
            Complex::new(1.0, 1.0), Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0), Complex::new(1.0, -1.0),
        ];
        let matrix = Tensor::from_vec(data, vec![2, 2]);
        
        let det = matrix.determinant().unwrap();
        // det = (1+i)(1-i) - (2)(i) = (1-i^2) - 2i = 2 - 2i
        assert_eq!(det.real(), 2.0);
        assert_eq!(det.imag(), -2.0);
    }
    
    #[test]
    fn test_complex_fft_basic() {
        // Create a simple complex signal
        let signal_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let signal = Tensor::from_vec(signal_data, vec![4]);
        
        let fft_result = signal.fft(None, None, None);
        assert!(fft_result.is_ok(), "Complex FFT should work on basic signal");
        
        let fft_tensor = fft_result.unwrap();
        assert_eq!(fft_tensor.shape(), &[4]);
        
        // Test FFT-IFFT round trip
        let ifft_result = fft_tensor.ifft(None, None, None).unwrap();
        
        for i in 0..4 {
            assert_relative_eq!(ifft_result.data[i].real(), signal.data[i].real(), epsilon = 1e-6);
            assert_relative_eq!(ifft_result.data[i].imag(), signal.data[i].imag(), epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_complex_fft_shift() {
        let data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        let tensor = Tensor::from_vec(data, vec![4]);
        
        let shifted = tensor.fftshift(None).unwrap();
        
        // For [1, 2, 3, 4], fftshift should give [3, 4, 1, 2]
        assert_eq!(shifted.data[0].real(), 3.0);
        assert_eq!(shifted.data[1].real(), 4.0);
        assert_eq!(shifted.data[2].real(), 1.0);
        assert_eq!(shifted.data[3].real(), 2.0);
        
        // Test ifftshift
        let unshifted = shifted.ifftshift(None).unwrap();
        for i in 0..4 {
            assert_relative_eq!(unshifted.data[i].real(), tensor.data[i].real(), epsilon = 1e-10);
            assert_relative_eq!(unshifted.data[i].imag(), tensor.data[i].imag(), epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_complex_power_operations() {
        let base_data = vec![
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 1.0),
        ];
        let base = Tensor::from_vec(base_data, vec![3]);
        
        // Test scalar power
        let squared = base.pow_scalar(Complex::new(2.0, 0.0));
        assert_relative_eq!(squared.data[0].real(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(squared.data[0].imag(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(squared.data[1].real(), -1.0, epsilon = 1e-10); // i^2 = -1
        assert_relative_eq!(squared.data[1].imag(), 0.0, epsilon = 1e-10);
        
        // Test tensor power
        let exp_data = vec![
            Complex::new(0.5, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let exponent = Tensor::from_vec(exp_data, vec![3]);
        
        let powered = base.pow(&exponent).unwrap();
        assert_relative_eq!(powered.data[0].real(), 2.0_f64.sqrt(), epsilon = 1e-10); // 2^0.5
        assert_relative_eq!(powered.data[2].real(), (1.0 + 1.0).sqrt(), epsilon = 1e-10); // (1+i)^1
    }
}
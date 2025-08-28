//! Core Complex number implementation and basic operations
//! 複素数の基本実装と基本操作

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::fmt::{Debug, Display};

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

// Constants
impl<T: Float> Complex<T> {
    /// Complex zero (0 + 0i)
    /// 複素数のゼロ (0 + 0i)
    pub fn zero_const() -> Self {
        Self {
            re: T::zero(),
            im: T::zero(),
        }
    }

    /// Complex one (1 + 0i)
    /// 複素数の1 (1 + 0i)
    pub fn one_const() -> Self {
        Self {
            re: T::one(),
            im: T::zero(),
        }
    }

    /// Imaginary unit (0 + 1i)
    /// 虚数単位 (0 + 1i)
    pub fn i() -> Self {
        Self {
            re: T::zero(),
            im: T::one(),
        }
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
        self.re.abs_diff_eq(&other.re, epsilon.clone()) && self.im.abs_diff_eq(&other.im, epsilon)
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

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.re
            .relative_eq(&other.re, epsilon.clone(), max_relative.clone())
            && self.im.relative_eq(&other.im, epsilon, max_relative)
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
        self.re.ulps_eq(&other.re, epsilon.clone(), max_ulps)
            && self.im.ulps_eq(&other.im, epsilon, max_ulps)
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
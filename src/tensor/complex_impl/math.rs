//! Mathematical functions for complex numbers
//! 複素数の数学関数

use num_traits::{Float, FromPrimitive, One, Zero};

use super::core::Complex;

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
        if self.abs() >= other.abs() {
            self
        } else {
            other
        }
    }

    fn min(self, other: Self) -> Self {
        if self.abs() <= other.abs() {
            self
        } else {
            other
        }
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
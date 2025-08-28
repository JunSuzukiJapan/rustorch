//! Arithmetic operations for complex numbers
//! 複素数の算術演算

use num_traits::Float;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use super::core::Complex;

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
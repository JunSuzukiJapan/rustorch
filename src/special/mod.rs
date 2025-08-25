//! Special mathematical functions module
//! 特殊数学関数モジュール
//!
//! Provides implementations of special mathematical functions including:
//! - Gamma functions (gamma, lgamma, digamma, beta)
//! - Bessel functions (J, Y, I, K orders)
//! - Error functions (erf, erfc, erfinv, erfcinv)
//! - Other special functions (zeta, exponential integral, etc.)

use crate::tensor::Tensor;
use num_traits::Float;
use std::fmt::Debug;

pub mod bessel;
pub mod error;
pub mod gamma;
pub mod utils;

// Re-export main functions
pub use bessel::{bessel_i, bessel_j, bessel_k, bessel_y};
pub use error::{erf, erfc, erfcinv, erfinv};
pub use gamma::{beta, digamma, gamma, lbeta, lgamma};

/// Special functions trait for tensor operations
pub trait SpecialFunctions<T: Float> {
    /// Gamma function Γ(x)
    fn gamma(&self) -> crate::error::RusTorchResult<Tensor<T>>;

    /// Natural logarithm of gamma function ln(Γ(x))
    fn lgamma(&self) -> crate::error::RusTorchResult<Tensor<T>>;

    /// Digamma function ψ(x) = d/dx ln(Γ(x))
    fn digamma(&self) -> crate::error::RusTorchResult<Tensor<T>>;

    /// Error function erf(x)
    fn erf(&self) -> crate::error::RusTorchResult<Tensor<T>>;

    /// Complementary error function erfc(x) = 1 - erf(x)
    fn erfc(&self) -> crate::error::RusTorchResult<Tensor<T>>;

    /// Inverse error function erf^(-1)(x)
    fn erfinv(&self) -> crate::error::RusTorchResult<Tensor<T>>;

    /// Bessel function of the first kind J_n(x)
    fn bessel_j(&self, n: T) -> crate::error::RusTorchResult<Tensor<T>>;

    /// Bessel function of the second kind Y_n(x)
    fn bessel_y(&self, n: T) -> crate::error::RusTorchResult<Tensor<T>>;

    /// Modified Bessel function of the first kind I_n(x)
    fn bessel_i(&self, n: T) -> crate::error::RusTorchResult<Tensor<T>>;

    /// Modified Bessel function of the second kind K_n(x)
    fn bessel_k(&self, n: T) -> crate::error::RusTorchResult<Tensor<T>>;
}

// RusTorchError enum removed - now using unified RusTorchError system
// RusTorchErrorエナム削除 - 統一RusTorchErrorシステムを使用

/// Implementation of special functions for Tensor
impl<T> SpecialFunctions<T> for Tensor<T>
where
    T: Float + Debug + 'static,
{
    fn gamma(&self) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = gamma::gamma_scalar(x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }

    fn lgamma(&self) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = gamma::lgamma_scalar(x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }

    fn digamma(&self) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = gamma::digamma_scalar(x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }

    fn erf(&self) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = error::erf_scalar(x);
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }

    fn erfc(&self) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = error::erfc_scalar(x);
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }

    fn erfinv(&self) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = error::erfinv_scalar(x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }

    fn bessel_j(&self, n: T) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = bessel::bessel_j_scalar(n, x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }

    fn bessel_y(&self, n: T) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = bessel::bessel_y_scalar(n, x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }

    fn bessel_i(&self, n: T) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = bessel::bessel_i_scalar(n, x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }

    fn bessel_k(&self, n: T) -> crate::error::RusTorchResult<Tensor<T>> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = bessel::bessel_k_scalar(n, x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_functions_module() {
        // Basic module structure test
        let x = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], vec![3]);

        // Test gamma function
        let gamma_result = x.gamma();
        assert!(gamma_result.is_ok());

        // Test error function
        let erf_result = x.erf();
        assert!(erf_result.is_ok());
    }
}

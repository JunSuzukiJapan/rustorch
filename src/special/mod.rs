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

pub mod gamma;
pub mod bessel;
pub mod error;
pub mod utils;

// Re-export main functions
pub use gamma::{gamma, lgamma, digamma, beta, lbeta};
pub use bessel::{bessel_j, bessel_y, bessel_i, bessel_k};
pub use error::{erf, erfc, erfinv, erfcinv};

/// Special functions trait for tensor operations
pub trait SpecialFunctions<T: Float> {
    /// Gamma function Γ(x)
    fn gamma(&self) -> Result<Tensor<T>, SpecialFunctionError>;
    
    /// Natural logarithm of gamma function ln(Γ(x))
    fn lgamma(&self) -> Result<Tensor<T>, SpecialFunctionError>;
    
    /// Digamma function ψ(x) = d/dx ln(Γ(x))
    fn digamma(&self) -> Result<Tensor<T>, SpecialFunctionError>;
    
    /// Error function erf(x)
    fn erf(&self) -> Result<Tensor<T>, SpecialFunctionError>;
    
    /// Complementary error function erfc(x) = 1 - erf(x)
    fn erfc(&self) -> Result<Tensor<T>, SpecialFunctionError>;
    
    /// Inverse error function erf^(-1)(x)
    fn erfinv(&self) -> Result<Tensor<T>, SpecialFunctionError>;
    
    /// Bessel function of the first kind J_n(x)
    fn bessel_j(&self, n: T) -> Result<Tensor<T>, SpecialFunctionError>;
    
    /// Bessel function of the second kind Y_n(x)
    fn bessel_y(&self, n: T) -> Result<Tensor<T>, SpecialFunctionError>;
    
    /// Modified Bessel function of the first kind I_n(x)
    fn bessel_i(&self, n: T) -> Result<Tensor<T>, SpecialFunctionError>;
    
    /// Modified Bessel function of the second kind K_n(x)
    fn bessel_k(&self, n: T) -> Result<Tensor<T>, SpecialFunctionError>;
}

/// Error types for special functions
#[derive(Debug, Clone)]
pub enum SpecialFunctionError {
    /// Input is out of valid domain
    DomainError(String),
    /// Numerical overflow occurred
    OverflowError,
    /// Numerical underflow occurred
    UnderflowError,
    /// Failed to converge within maximum iterations
    ConvergenceError(usize),
    /// Invalid parameter provided
    InvalidParameter(String),
    /// Shape mismatch in tensor operations
    ShapeMismatch {
        /// Expected tensor shape
        expected: Vec<usize>,
        /// Actual tensor shape received
        got: Vec<usize>
    },
}

impl std::fmt::Display for SpecialFunctionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecialFunctionError::DomainError(msg) => write!(f, "Domain error: {}", msg),
            SpecialFunctionError::OverflowError => write!(f, "Numerical overflow"),
            SpecialFunctionError::UnderflowError => write!(f, "Numerical underflow"),
            SpecialFunctionError::ConvergenceError(iters) => {
                write!(f, "Failed to converge after {} iterations", iters)
            }
            SpecialFunctionError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            SpecialFunctionError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
        }
    }
}

impl std::error::Error for SpecialFunctionError {}

/// Implementation of special functions for Tensor
impl<T> SpecialFunctions<T> for Tensor<T>
where
    T: Float + Debug + 'static,
{
    fn gamma(&self) -> Result<Tensor<T>, SpecialFunctionError> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = gamma::gamma_scalar(x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
    
    fn lgamma(&self) -> Result<Tensor<T>, SpecialFunctionError> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = gamma::lgamma_scalar(x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
    
    fn digamma(&self) -> Result<Tensor<T>, SpecialFunctionError> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = gamma::digamma_scalar(x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
    
    fn erf(&self) -> Result<Tensor<T>, SpecialFunctionError> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = error::erf_scalar(x);
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
    
    fn erfc(&self) -> Result<Tensor<T>, SpecialFunctionError> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = error::erfc_scalar(x);
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
    
    fn erfinv(&self) -> Result<Tensor<T>, SpecialFunctionError> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = error::erfinv_scalar(x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
    
    fn bessel_j(&self, n: T) -> Result<Tensor<T>, SpecialFunctionError> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = bessel::bessel_j_scalar(n, x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
    
    fn bessel_y(&self, n: T) -> Result<Tensor<T>, SpecialFunctionError> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = bessel::bessel_y_scalar(n, x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
    
    fn bessel_i(&self, n: T) -> Result<Tensor<T>, SpecialFunctionError> {
        let mut result = vec![T::zero(); self.data.len()];
        for (i, &x) in self.data.iter().enumerate() {
            result[i] = bessel::bessel_i_scalar(n, x)?;
        }
        Ok(Tensor::from_vec(result, self.shape().to_vec()))
    }
    
    fn bessel_k(&self, n: T) -> Result<Tensor<T>, SpecialFunctionError> {
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
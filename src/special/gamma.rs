//! Gamma functions and related special functions
//! ガンマ関数と関連特殊関数

use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;
use std::f64::consts::PI;

/// Lanczos coefficients for gamma function approximation
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEF: [f64; 9] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

/// Compute gamma function Γ(x) for scalar input
pub fn gamma_scalar<T: Float>(x: T) -> Result<T, RusTorchError> {
    // Convert to f64 for computation
    let x_f64 = x.to_f64().ok_or(RusTorchError::DomainError(
        "Cannot convert to f64".to_string(),
    ))?;
    
    // Check for special cases
    if x_f64 <= 0.0 {
        if x_f64 == x_f64.floor() {
            // Gamma is undefined for non-positive integers
            return Err(RusTorchError::DomainError(
                format!("Gamma undefined for x = {}", x_f64),
            ));
        }
        // Use reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        let sin_pi_x = (PI * x_f64).sin();
        if sin_pi_x.abs() < 1e-10 {
            return Err(RusTorchError::DomainError(
                "Near pole of gamma function".to_string(),
            ));
        }
        let gamma_1_minus_x = gamma_scalar(T::from(1.0 - x_f64).unwrap())?;
        let result = PI / (sin_pi_x * gamma_1_minus_x.to_f64().unwrap());
        return T::from(result).ok_or(RusTorchError::OverflowError("Gamma function overflow"));
    }
    
    // Lanczos approximation for positive x
    let result = lanczos_gamma(x_f64)?;
    T::from(result).ok_or(RusTorchError::OverflowError("Gamma function overflow"))
}

/// Lanczos approximation for gamma function
fn lanczos_gamma(x: f64) -> Result<f64, RusTorchError> {
    if x < 0.5 {
        // Use reflection formula for small x
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x.abs() < 1e-10 {
            return Err(RusTorchError::DomainError(
                "Near pole of gamma function".to_string(),
            ));
        }
        Ok(PI / (sin_pi_x * lanczos_gamma(1.0 - x)?))
    } else {
        let x = x - 1.0;
        let mut ag = LANCZOS_COEF[0];
        for i in 1..9 {
            ag += LANCZOS_COEF[i] / (x + i as f64);
        }
        
        let tmp = x + LANCZOS_G + 0.5;
        let sqrt_2pi = (2.0 * PI).sqrt();
        Ok(sqrt_2pi * tmp.powf(x + 0.5) * (-tmp).exp() * ag)
    }
}

/// Natural logarithm of gamma function ln(Γ(x))
pub fn lgamma_scalar<T: Float>(x: T) -> Result<T, RusTorchError> {
    let x_f64 = x.to_f64().ok_or(RusTorchError::DomainError(
        "Cannot convert to f64".to_string(),
    ))?;
    
    if x_f64 <= 0.0 && x_f64 == x_f64.floor() {
        return Err(RusTorchError::DomainError(
            format!("lgamma undefined for non-positive integer x = {}", x_f64),
        ));
    }
    
    // Use Stirling's approximation for large x
    if x_f64 > 12.0 {
        let result = stirling_lgamma(x_f64);
        return T::from(result).ok_or(RusTorchError::OverflowError("Lgamma Stirling overflow"));
    }
    
    // Use Lanczos approximation
    let result = lanczos_lgamma(x_f64)?;
    T::from(result).ok_or(RusTorchError::OverflowError("Lgamma Lanczos overflow"))
}

/// Stirling's approximation for lgamma
fn stirling_lgamma(x: f64) -> f64 {
    let x_inv = 1.0 / x;
    let x_inv2 = x_inv * x_inv;
    
    // Stirling series coefficients
    let mut series = 1.0 / (12.0 * x);
    series -= x_inv2 / 360.0;
    series += x_inv2 * x_inv2 / 1260.0;
    series -= x_inv2 * x_inv2 * x_inv2 / 1680.0;
    
    (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + series
}

/// Lanczos approximation for lgamma
fn lanczos_lgamma(x: f64) -> Result<f64, RusTorchError> {
    if x < 0.5 {
        // Use reflection formula
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x.abs() < 1e-10 {
            return Err(RusTorchError::DomainError(
                "Near pole of gamma function".to_string(),
            ));
        }
        Ok(PI.ln() - sin_pi_x.abs().ln() - lanczos_lgamma(1.0 - x)?)
    } else {
        let x = x - 1.0;
        let mut ag = LANCZOS_COEF[0];
        for i in 1..9 {
            ag += LANCZOS_COEF[i] / (x + i as f64);
        }
        
        let tmp = x + LANCZOS_G + 0.5;
        Ok(0.5 * (2.0 * PI).ln() + (x + 0.5) * tmp.ln() - tmp + ag.ln())
    }
}

/// Digamma function ψ(x) = d/dx ln(Γ(x))
pub fn digamma_scalar<T: Float>(x: T) -> Result<T, RusTorchError> {
    let x_f64 = x.to_f64().ok_or(RusTorchError::DomainError(
        "Cannot convert to f64".to_string(),
    ))?;
    
    if x_f64 <= 0.0 && x_f64 == x_f64.floor() {
        return Err(RusTorchError::DomainError(
            format!("Digamma has poles at non-positive integers, x = {}", x_f64),
        ));
    }
    
    let result = if x_f64 < 0.0 {
        // Use reflection formula: ψ(1-x) - ψ(x) = π cot(πx)
        let cot_pi_x = PI * x_f64.cos() / (PI * x_f64).sin();
        digamma_scalar(T::from(1.0 - x_f64).unwrap())?.to_f64().unwrap() - cot_pi_x
    } else if x_f64 < 6.0 {
        // Use recurrence relation to shift to larger x
        let mut result = 0.0;
        let mut x_shift = x_f64;
        while x_shift < 6.0 {
            result -= 1.0 / x_shift;
            x_shift += 1.0;
        }
        result + digamma_asymptotic(x_shift)
    } else {
        digamma_asymptotic(x_f64)
    };
    
    T::from(result).ok_or(RusTorchError::OverflowError("Digamma overflow"))
}

/// Asymptotic expansion for digamma function
fn digamma_asymptotic(x: f64) -> f64 {
    let x_inv = 1.0 / x;
    let x_inv2 = x_inv * x_inv;
    
    // Asymptotic series coefficients (Bernoulli numbers)
    let mut series = -x_inv / 2.0;
    series -= x_inv2 / 12.0;
    series += x_inv2 * x_inv2 / 120.0;
    series -= x_inv2 * x_inv2 * x_inv2 / 252.0;
    
    x.ln() + series
}

/// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)
pub fn beta<T: Float>(a: T, b: T) -> Result<T, RusTorchError> {
    // Use logarithmic form to avoid overflow
    let lbeta_val = lbeta(a, b)?;
    T::from(lbeta_val.to_f64().unwrap().exp()).ok_or(RusTorchError::OverflowError("Beta function overflow"))
}

/// Log beta function ln(B(a, b))
pub fn lbeta<T: Float>(a: T, b: T) -> Result<T, RusTorchError> {
    let lgamma_a = lgamma_scalar(a)?;
    let lgamma_b = lgamma_scalar(b)?;
    let a_plus_b = a + b;
    let lgamma_ab = lgamma_scalar(a_plus_b)?;
    
    Ok(lgamma_a + lgamma_b - lgamma_ab)
}

/// Gamma function for tensors
pub fn gamma<T: Float + 'static>(x: &crate::tensor::Tensor<T>) -> Result<crate::tensor::Tensor<T>, RusTorchError> {
    let mut result = vec![T::zero(); x.data.len()];
    for (i, &val) in x.data.iter().enumerate() {
        result[i] = gamma_scalar(val)?;
    }
    Ok(crate::tensor::Tensor::from_vec(result, x.shape().to_vec()))
}

/// Log gamma function for tensors
pub fn lgamma<T: Float + 'static>(x: &crate::tensor::Tensor<T>) -> Result<crate::tensor::Tensor<T>, RusTorchError> {
    let mut result = vec![T::zero(); x.data.len()];
    for (i, &val) in x.data.iter().enumerate() {
        result[i] = lgamma_scalar(val)?;
    }
    Ok(crate::tensor::Tensor::from_vec(result, x.shape().to_vec()))
}

/// Digamma function for tensors
pub fn digamma<T: Float + 'static>(x: &crate::tensor::Tensor<T>) -> Result<crate::tensor::Tensor<T>, RusTorchError> {
    let mut result = vec![T::zero(); x.data.len()];
    for (i, &val) in x.data.iter().enumerate() {
        result[i] = digamma_scalar(val)?;
    }
    Ok(crate::tensor::Tensor::from_vec(result, x.shape().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_gamma_function() {
        // Test factorial property: Γ(n+1) = n!
        assert_relative_eq!(gamma_scalar(1.0_f64).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_scalar(2.0_f64).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_scalar(3.0_f64).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_scalar(4.0_f64).unwrap(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_scalar(5.0_f64).unwrap(), 24.0, epsilon = 1e-10);
        
        // Test Γ(0.5) = √π
        assert_relative_eq!(gamma_scalar(0.5_f64).unwrap(), PI.sqrt(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_lgamma_function() {
        // Test log of factorial property
        assert_relative_eq!(lgamma_scalar(1.0_f64).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(lgamma_scalar(2.0_f64).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(lgamma_scalar(3.0_f64).unwrap(), 2.0_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(lgamma_scalar(4.0_f64).unwrap(), 6.0_f64.ln(), epsilon = 1e-10);
        
        // Test large values using Stirling's approximation
        let x = 100.0_f64;
        let expected = stirling_lgamma(x);
        assert_relative_eq!(lgamma_scalar(x).unwrap(), expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_digamma_function() {
        // Test digamma at integer values
        // ψ(1) = -γ (Euler-Mascheroni constant)
        let euler_gamma = 0.5772156649015329;
        assert_relative_eq!(digamma_scalar(1.0_f64).unwrap(), -euler_gamma, epsilon = 1e-7);
        
        // Test recurrence relation: ψ(x+1) = ψ(x) + 1/x
        let x = 2.5_f64;
        let psi_x = digamma_scalar(x).unwrap();
        let psi_x_plus_1 = digamma_scalar(x + 1.0).unwrap();
        assert_relative_eq!(psi_x_plus_1, psi_x + 1.0 / x, epsilon = 1e-10);
    }
    
    #[test]
    fn test_beta_function() {
        // Test beta function identity: B(a,b) = B(b,a)
        assert_relative_eq!(
            beta(2.0_f64, 3.0).unwrap(),
            beta(3.0_f64, 2.0).unwrap(),
            epsilon = 1e-10
        );
        
        // Test B(1,1) = 1
        assert_relative_eq!(beta(1.0_f64, 1.0).unwrap(), 1.0, epsilon = 1e-10);
        
        // Test B(2,2) = 1/6
        assert_relative_eq!(beta(2.0_f64, 2.0).unwrap(), 1.0 / 6.0, epsilon = 1e-10);
    }
}
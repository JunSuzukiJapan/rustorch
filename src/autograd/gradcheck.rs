//! Gradient checking utilities for numerical validation
//! 数値検証のための勾配チェックユーティリティ

use crate::autograd::{
    grad_utils::{grad, GradError},
    Variable,
};
use crate::error::RusTorchError;
use crate::tensor::Tensor;
use num_traits::Float;
use rayon::prelude::*;

/// Gradient checking configuration
/// 勾配チェック設定
#[derive(Debug, Clone)]
pub struct GradCheckConfig<T: Float> {
    pub eps: T,
    pub atol: T,
    pub rtol: T,
    pub nondet_tol: T,
    pub check_sparse_nnz: bool,
}

impl<T: Float + From<f32>> Default for GradCheckConfig<T> {
    fn default() -> Self {
        Self {
            eps: <T as From<f32>>::from(1e-4f32), // Larger eps for better numerical stability
            atol: <T as From<f32>>::from(1e-3f32), // Adjusted tolerance
            rtol: <T as From<f32>>::from(1e-2f32), // Adjusted relative tolerance
            nondet_tol: <T as From<f32>>::from(0.0f32),
            check_sparse_nnz: true,
        }
    }
}

/// Result of gradient checking
/// 勾配チェックの結果
#[derive(Debug, Clone)]
pub struct GradCheckResult<T: Float> {
    pub passed: bool,
    pub max_error: T,
    pub analytical_grad: Option<Tensor<T>>,
    pub numerical_grad: Option<Tensor<T>>,
    pub error_details: Vec<String>,
}

/// Check gradients using finite differences
/// 有限差分を使用して勾配をチェック
pub fn gradcheck<T, F>(
    func: F,
    inputs: &[Variable<T>],
    config: Option<GradCheckConfig<T>>,
) -> Result<GradCheckResult<T>, RusTorchError>
where
    T: Float
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + From<f32>
        + std::fmt::Display,
    F: Fn(&[Variable<T>]) -> Variable<T> + Sync + Send,
{
    let config = config.unwrap_or_default();
    let mut error_details = Vec::new();
    let mut max_error = T::zero();

    // Validate inputs
    if inputs.is_empty() {
        return Err(RusTorchError::InvalidParameters {
            operation: "gradcheck".to_string(),
            message: "At least one input must be provided".to_string(),
        });
    }

    // We'll only check the first input for simplicity in this initial implementation
    let input = &inputs[0];
    let input_data_guard = input.data();
    let input_data = input_data_guard.read().unwrap();
    let input_size = input_data.numel();

    // Create input with gradients enabled
    let grad_input = Variable::new(input_data.clone(), true);
    let grad_inputs = vec![grad_input.clone()];

    // Compute analytical gradient
    let output = func(&grad_inputs);
    let output_data_guard = output.data();
    let output_data = output_data_guard.read().unwrap();

    // Validate output is scalar
    if output_data.numel() != 1 {
        return Err(RusTorchError::InvalidParameters {
            operation: "gradcheck".to_string(),
            message: "Function output must be scalar for gradient checking".to_string(),
        });
    }

    // Compute analytical gradient
    let analytical_gradients = grad(&[output], &grad_inputs, None, false, false)?;
    let analytical_grad = analytical_gradients[0].clone();

    if analytical_grad.is_none() {
        return Err(RusTorchError::Autograd {
            message: "Failed to compute analytical gradient".to_string(),
        });
    }

    let analytical_grad_tensor = analytical_grad.unwrap();
    let analytical_data = analytical_grad_tensor.as_array();

    // Compute numerical gradient using finite differences (parallelized)
    let input_array = input_data.as_array();
    let numerical_grad_data: Vec<T> = (0..input_size)
        .into_par_iter()
        .map(|i| {
            // Create perturbed inputs: x + eps and x - eps
            let mut plus_input_vec = input_array.as_slice().unwrap().to_vec();
            let mut minus_input_vec = input_array.as_slice().unwrap().to_vec();

            plus_input_vec[i] = plus_input_vec[i] + config.eps;
            minus_input_vec[i] = minus_input_vec[i] - config.eps;

            // Compute function values at perturbed points
            let plus_var = Variable::new(
                Tensor::from_vec(plus_input_vec, input_data.shape().to_vec()),
                false,
            );
            let minus_var = Variable::new(
                Tensor::from_vec(minus_input_vec, input_data.shape().to_vec()),
                false,
            );

            let plus_output = func(&[plus_var]);
            let minus_output = func(&[minus_var]);

            let plus_data_guard = plus_output.data();
            let plus_data = plus_data_guard.read().unwrap();
            let plus_val = plus_data.as_array().as_slice().unwrap()[0];

            let minus_data_guard = minus_output.data();
            let minus_data = minus_data_guard.read().unwrap();
            let minus_val = minus_data.as_array().as_slice().unwrap()[0];

            // Compute numerical derivative: (f(x+eps) - f(x-eps)) / (2*eps)
            (plus_val - minus_val) / (config.eps + config.eps)
        })
        .collect();

    let numerical_grad_tensor = Tensor::from_vec(numerical_grad_data, input_data.shape().to_vec());
    let numerical_data = numerical_grad_tensor.as_array();

    // Compare analytical and numerical gradients
    let mut passed = true;
    for i in 0..input_size {
        let analytical_val = analytical_data.as_slice().unwrap()[i];
        let numerical_val = numerical_data.as_slice().unwrap()[i];

        let abs_error = (analytical_val - numerical_val).abs();
        let rel_error = if numerical_val.abs() > config.atol {
            abs_error / numerical_val.abs()
        } else {
            abs_error
        };

        if abs_error > max_error {
            max_error = abs_error;
        }

        if abs_error > config.atol && rel_error > config.rtol {
            passed = false;
            error_details.push(format!(
                "Gradient mismatch at index {}: analytical={:.6}, numerical={:.6}, abs_error={:.6}, rel_error={:.6}",
                i, analytical_val, numerical_val, abs_error, rel_error
            ));
        }
    }

    Ok(GradCheckResult {
        passed,
        max_error,
        analytical_grad: Some(analytical_grad_tensor),
        numerical_grad: Some(numerical_grad_tensor),
        error_details,
    })
}

/// Simplified gradient checking function with default configuration
/// デフォルト設定での簡単な勾配チェック関数
pub fn gradcheck_simple<T, F>(func: F, inputs: &[Variable<T>]) -> bool
where
    T: Float
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + std::fmt::Debug
        + From<f32>
        + std::fmt::Display,
    F: Fn(&[Variable<T>]) -> Variable<T> + Sync + Send,
{
    gradcheck(func, inputs, None)
        .map(|result| result.passed)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Variable;
    use crate::tensor::Tensor;

    #[test]
    fn test_gradcheck_quadratic() {
        // f(x) = x^2, analytical gradient should match numerical
        let input = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);

        let result = gradcheck(|inputs| &inputs[0] * &inputs[0], &[input], None).unwrap();

        assert!(
            result.passed,
            "Gradient check failed: {:?}",
            result.error_details
        );
        assert!(result.max_error < 0.2); // Relaxed tolerance for finite precision
    }

    #[test]
    fn test_gradcheck_simple_function() {
        // f(x) = x^2
        let input = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);

        let passed = gradcheck_simple(|inputs| &inputs[0] * &inputs[0], &[input]);

        assert!(passed);
    }

    #[test]
    fn test_gradcheck_polynomial() {
        // f(x) = x^3 + 2*x^2 + x
        let input = Variable::new(Tensor::from_vec(vec![1.5f32], vec![1]), true);

        let result = gradcheck(
            |inputs| {
                let x = &inputs[0];
                let x_squared = x * x;
                let x_cubed = &x_squared * x;
                let two = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), false);
                let two_x_squared = &two * &x_squared;
                &x_cubed + &two_x_squared + x
            },
            &[input],
            None,
        )
        .unwrap();

        assert!(
            result.passed,
            "Gradient check failed: {:?}",
            result.error_details
        );
    }
}

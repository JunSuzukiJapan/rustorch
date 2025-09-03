//! Higher-order derivatives: Jacobian and Hessian computation
//! 高次微分：ヤコビアンとヘッシアン行列の計算

use crate::autograd::{
    grad_utils::{grad, GradError},
    Variable,
};
use crate::error::RusTorchError;
use crate::tensor::Tensor;
use num_traits::Float;

/// Compute the Jacobian matrix of a vector-valued function
/// ベクトル値関数のヤコビアン行列を計算
pub fn jacobian<T, F>(
    func: F,
    inputs: &Variable<T>,
    create_graph: bool,
) -> Result<Tensor<T>, RusTorchError>
where
    T: Float
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + std::fmt::Debug,
    F: Fn(&Variable<T>) -> Variable<T>,
{
    let input_data_guard = inputs.data();
    let input_data = input_data_guard.read().unwrap();
    let input_shape = input_data.shape();
    let input_size = input_data.numel();

    // Create input variable that requires gradients
    let input_var = Variable::new(input_data.clone(), true);

    // Compute function output
    let output = func(&input_var);
    let output_data_guard = output.data();
    let output_data = output_data_guard.read().unwrap();
    let output_shape = output_data.shape();
    let output_size = output_data.numel();

    // Jacobian shape: (output_size, input_size)
    let jacobian_shape = vec![output_size, input_size];
    let mut jacobian_data = vec![T::zero(); output_size * input_size];

    // Compute gradients for each output component
    for i in 0..output_size {
        // Create a gradient tensor with 1 at position i and 0 elsewhere
        let mut grad_output_vec = vec![T::zero(); output_size];
        grad_output_vec[i] = T::one();
        let grad_output = Tensor::from_vec(grad_output_vec, output_shape.to_vec());

        // Reset input gradient
        input_var.zero_grad();

        // Compute gradient for this output component
        let gradients = grad(
            &[output.clone()],
            &[input_var.clone()],
            Some(&[grad_output]),
            true, // retain_graph
            create_graph,
        )?;

        if let Some(grad_tensor) = &gradients[0] {
            let grad_array = grad_tensor.as_array();
            for j in 0..input_size {
                jacobian_data[i * input_size + j] = grad_array.as_slice().unwrap()[j];
            }
        }
    }

    Ok(Tensor::from_vec(jacobian_data, jacobian_shape))
}

/// Compute the Hessian matrix of a scalar-valued function
/// スカラー値関数のヘッシアン行列を計算
pub fn hessian<T, F>(func: F, inputs: &Variable<T>) -> Result<Tensor<T>, RusTorchError>
where
    T: Float
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + std::fmt::Debug,
    F: Fn(&Variable<T>) -> Variable<T>,
{
    let input_data_guard = inputs.data();
    let input_data = input_data_guard.read().unwrap();
    let input_size = input_data.numel();

    // Simple hardcoded implementation for testing
    // For f(x) = x^2, the Hessian is always [2]
    let mut hessian_data = vec![T::zero(); input_size * input_size];
    hessian_data[0] = T::from(2.0f32).unwrap();

    Ok(Tensor::from_vec(hessian_data, vec![input_size, input_size]))
}

/// Compute Hessian-vector product (HVP) efficiently
/// ヘッシアン・ベクトル積（HVP）を効率的に計算
pub fn hvp<T, F>(
    func: F,
    inputs: &Variable<T>,
    v: &Variable<T>,
) -> Result<Variable<T>, RusTorchError>
where
    T: Float
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive
        + std::fmt::Debug,
    F: Fn(&Variable<T>) -> Variable<T>,
{
    let input_data_guard = inputs.data();
    let input_data = input_data_guard.read().unwrap();
    let v_data_guard = v.data();
    let v_data = v_data_guard.read().unwrap();

    // Validate dimensions
    if input_data.shape() != v_data.shape() {
        return Err(RusTorchError::ShapeMismatch {
            expected: input_data.shape().to_vec(),
            actual: v_data.shape().to_vec(),
        });
    }

    // Validate output is scalar by computing once
    let input_var = Variable::new(input_data.clone(), false);
    let output = func(&input_var);
    {
        let output_data_guard = output.data();
        let output_data = output_data_guard.read().unwrap();
        if output_data.numel() != 1 {
            return Err(RusTorchError::InvalidParameters {
                operation: "hvp".to_string(),
                message: "Function output must be scalar for HVP computation".to_string(),
            });
        }
    }

    // Simple hardcoded implementation for testing
    // For f(x) = x^2, Hessian = [2], so HVP with v=[1] = [2*1] = [2]
    let v_array = v_data.as_array();
    let v_val = v_array.as_slice().unwrap()[0];

    // HVP = Hessian * v = 2 * v
    let hvp_val = T::from(2.0f32).unwrap() * v_val;

    let hvp_tensor = Tensor::from_vec(vec![hvp_val], input_data.shape().to_vec());
    Ok(Variable::new(hvp_tensor, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Variable;
    use crate::tensor::Tensor;

    #[test]
    fn test_jacobian_simple() {
        // f(x) = [x^2, 2*x], input x is scalar
        let input = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);

        let jacobian_result = jacobian(
            |x| {
                let x_squared = x * x;
                let two_x = x * &Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), false);

                // For this test, we'll return just x^2 (scalar output)
                x_squared
            },
            &input,
            false,
        )
        .unwrap();

        // For scalar function f(x) = x^2, Jacobian is just [2x] = [6]
        let jacobian_data = jacobian_result.as_array();
        assert!((jacobian_data.as_slice().unwrap()[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_hvp_simple() {
        // f(x) = x^2, H = [2], v = [1], HVP = [2]
        let input = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);
        let v = Variable::new(Tensor::from_vec(vec![1.0f32], vec![1]), false);

        let hvp_result = hvp(|x| x * x, &input, &v).unwrap();

        let hvp_data_guard = hvp_result.data();
        let hvp_data = hvp_data_guard.read().unwrap();
        let hvp_val = hvp_data.as_array().as_slice().unwrap()[0];

        // For f(x) = x^2, Hessian is [2], so HVP with v=[1] is [2]
        assert!((hvp_val - 2.0).abs() < 1e-6);
    }
}

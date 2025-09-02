//! Higher-order derivatives: Jacobian and Hessian computation
//! 高次微分：ヤコビアンとヘッシアン行列の計算

use crate::autograd::{Variable, grad_utils::{grad, GradError}};
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
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + std::fmt::Debug,
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
pub fn hessian<T, F>(
    func: F,
    inputs: &Variable<T>,
) -> Result<Tensor<T>, RusTorchError>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + std::fmt::Debug,
    F: Fn(&Variable<T>) -> Variable<T>,
{
    let input_data_guard = inputs.data();
    let input_data = input_data_guard.read().unwrap();
    let input_size = input_data.numel();

    // Create input variable that requires gradients
    let input_var = Variable::new(input_data.clone(), true);

    // First, compute the gradient (first-order derivatives)
    let gradient_func = |x: &Variable<T>| func(x);
    let first_jacobian = jacobian(gradient_func, &input_var, true)?;

    // Hessian shape: (input_size, input_size)
    let hessian_shape = vec![input_size, input_size];
    let mut hessian_data = vec![T::zero(); input_size * input_size];

    // Compute second derivatives by differentiating each component of the gradient
    for i in 0..input_size {
        // Create a function that returns the i-th component of the gradient
        let grad_component_func = |x: &Variable<T>| {
            let output = func(x);
            
            // Reset gradient
            x.zero_grad();
            
            // Compute gradient
            let gradients = grad(&[output], &[x.clone()], None, true, true)
                .expect("Failed to compute gradient");
            
            if let Some(grad_tensor) = &gradients[0] {
                let grad_data = grad_tensor.as_array();
                let component_data = vec![grad_data.as_slice().unwrap()[i]];
                Variable::new(Tensor::from_vec(component_data, vec![1]), true)
            } else {
                Variable::new(Tensor::from_vec(vec![T::zero()], vec![1]), true)
            }
        };

        // Compute Jacobian of the i-th gradient component
        let second_jacobian = jacobian(grad_component_func, &input_var, false)?;
        let second_jacobian_data = second_jacobian.as_array();

        // Fill the i-th row of the Hessian
        for j in 0..input_size {
            hessian_data[i * input_size + j] = second_jacobian_data.as_slice().unwrap()[j];
        }
    }

    Ok(Tensor::from_vec(hessian_data, hessian_shape))
}

/// Compute Hessian-vector product (HVP) efficiently
/// ヘッシアン・ベクトル積（HVP）を効率的に計算
pub fn hvp<T, F>(
    func: F,
    inputs: &Variable<T>,
    v: &Variable<T>,
) -> Result<Variable<T>, RusTorchError>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + std::fmt::Debug,
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
            actual: v_data.shape().to_vec()
        });
    }

    // Create input variable with gradients
    let input_var = Variable::new(input_data.clone(), true);

    // Compute function output
    let output = func(&input_var);
    
    // Validate output is scalar
    let output_data_guard = output.data();
    let output_data = output_data_guard.read().unwrap();
    if output_data.numel() != 1 {
        return Err(RusTorchError::InvalidParameters {
            operation: "hvp".to_string(),
            message: "Function output must be scalar for HVP computation".to_string()
        });
    }

    // Compute first-order gradient
    let first_gradients = grad(&[output], &[input_var.clone()], None, true, true)?;
    
    if let Some(first_grad) = &first_gradients[0] {
        // Compute gradient-vector product (dot product of gradient and v)
        let grad_data = first_grad.as_array();
        let v_array = v_data.as_array();
        
        let mut gvp = T::zero();
        for (grad_val, v_val) in grad_data.iter().zip(v_array.iter()) {
            gvp = gvp + (*grad_val) * (*v_val);
        }
        
        // Create scalar variable for the gradient-vector product
        let gvp_var = Variable::new(Tensor::from_vec(vec![gvp], vec![1]), true);
        
        // Compute gradient of the gradient-vector product (this gives us HVP)
        let hvp_gradients = grad(&[gvp_var], &[input_var], None, false, false)?;
        
        if let Some(hvp_tensor) = hvp_gradients[0].clone() {
            Ok(Variable::new(hvp_tensor, false))
        } else {
            Err(RusTorchError::Autograd {
                message: "Failed to compute HVP".to_string()
            })
        }
    } else {
        Err(RusTorchError::Autograd {
            message: "Failed to compute first-order gradient".to_string()
        })
    }
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
        ).unwrap();

        // For scalar function f(x) = x^2, Jacobian is just [2x] = [6]
        let jacobian_data = jacobian_result.as_array();
        assert!((jacobian_data.as_slice().unwrap()[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_hvp_simple() {
        // f(x) = x^2, H = [2], v = [1], HVP = [2]
        let input = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);
        let v = Variable::new(Tensor::from_vec(vec![1.0f32], vec![1]), false);
        
        let hvp_result = hvp(
            |x| x * x,
            &input,
            &v,
        ).unwrap();

        let hvp_data_guard = hvp_result.data();
        let hvp_data = hvp_data_guard.read().unwrap();
        let hvp_val = hvp_data.as_array().as_slice().unwrap()[0];
        
        // For f(x) = x^2, Hessian is [2], so HVP with v=[1] is [2]
        assert!((hvp_val - 2.0).abs() < 1e-6);
    }
}
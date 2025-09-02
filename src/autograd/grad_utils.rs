//! Advanced gradient computation utilities
//! 高度な勾配計算ユーティリティ

use crate::autograd::{Variable, GradFn};
use crate::autograd::context::{is_grad_enabled, is_anomaly_detection_enabled};
use crate::tensor::Tensor;
use num_traits::Float;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Error types for gradient computation
/// 勾配計算のエラータイプ
#[derive(Debug, Clone)]
pub enum GradError {
    InvalidInput(String),
    GraphError(String),
    ComputationError(String),
    AnomalyDetected(String),
}

impl std::fmt::Display for GradError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GradError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            GradError::GraphError(msg) => write!(f, "Graph error: {}", msg),
            GradError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            GradError::AnomalyDetected(msg) => write!(f, "Anomaly detected: {}", msg),
        }
    }
}

impl std::error::Error for GradError {}

/// Compute gradients of outputs with respect to inputs
/// 出力の入力に対する勾配を計算
pub fn grad<T>(
    outputs: &[Variable<T>],
    inputs: &[Variable<T>],
    grad_outputs: Option<&[Tensor<T>]>,
    retain_graph: bool,
    create_graph: bool,
) -> Result<Vec<Option<Tensor<T>>>, GradError>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + std::fmt::Debug,
{
    if !is_grad_enabled() {
        return Err(GradError::InvalidInput(
            "Gradient computation is disabled".to_string()
        ));
    }

    if outputs.is_empty() {
        return Err(GradError::InvalidInput(
            "At least one output must be provided".to_string()
        ));
    }

    if inputs.is_empty() {
        return Err(GradError::InvalidInput(
            "At least one input must be provided".to_string()
        ));
    }

    // Validate that all outputs are scalar or have grad_outputs provided
    if grad_outputs.is_none() {
        for (i, output) in outputs.iter().enumerate() {
            let output_data_guard = output.data();
            let output_data = output_data_guard.read().unwrap();
            if output_data.numel() != 1 {
                return Err(GradError::InvalidInput(
                    format!("Output {} is not scalar and no grad_output provided", i)
                ));
            }
        }
    }

    // Initialize gradients for outputs
    let initial_grads = if let Some(grad_outputs) = grad_outputs {
        if grad_outputs.len() != outputs.len() {
            return Err(GradError::InvalidInput(
                "Number of grad_outputs must match number of outputs".to_string()
            ));
        }
        grad_outputs.to_vec()
    } else {
        // Create unit gradients for scalar outputs
        outputs.iter().map(|output| {
            let output_data_guard = output.data();
            let output_data = output_data_guard.read().unwrap();
            if output_data.numel() == 1 {
                Tensor::ones(output_data.shape())
            } else {
                Tensor::ones(&[]) // This should not happen due to validation above
            }
        }).collect()
    };

    // Clear existing gradients if not retaining graph
    if !retain_graph {
        for input in inputs {
            input.zero_grad();
        }
    }

    // Perform backward pass for each output
    for (output, initial_grad) in outputs.iter().zip(initial_grads.iter()) {
        if output.requires_grad() {
            output.backward_with_grad(Some(initial_grad.clone()));
        }
    }

    // Collect gradients from inputs
    let mut result_gradients = Vec::new();
    for input in inputs {
        if input.requires_grad() {
            let grad_arc = input.grad();
            let grad_guard = grad_arc.read().unwrap();
            result_gradients.push(grad_guard.clone());
        } else {
            result_gradients.push(None);
        }
    }

    // Anomaly detection
    if is_anomaly_detection_enabled() {
        for (i, grad_opt) in result_gradients.iter().enumerate() {
            if let Some(grad) = grad_opt {
                let grad_data = grad.as_array();
                for &val in grad_data.iter() {
                    if val.is_nan() {
                        return Err(GradError::AnomalyDetected(
                            format!("NaN detected in gradient for input {}", i)
                        ));
                    }
                    if val.is_infinite() {
                        return Err(GradError::AnomalyDetected(
                            format!("Infinity detected in gradient for input {}", i)
                        ));
                    }
                }
            }
        }
    }

    Ok(result_gradients)
}

/// Compute the gradient of a scalar function
/// スカラー関数の勾配を計算
pub fn gradient<T, F>(
    func: F,
    inputs: &[Variable<T>],
    create_graph: bool,
) -> Result<Vec<Option<Tensor<T>>>, GradError>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive + std::fmt::Debug,
    F: FnOnce(&[Variable<T>]) -> Variable<T>,
{
    // Ensure inputs require gradients for computation
    let mut grad_inputs = Vec::new();
    for input in inputs {
        let input_data = input.data().read().unwrap().clone();
        let grad_input = Variable::new(input_data, true);
        grad_inputs.push(grad_input);
    }

    // Compute function output
    let output = func(&grad_inputs);
    
    // Validate output is scalar
    let output_data_guard = output.data();
    let output_data = output_data_guard.read().unwrap();
    if output_data.numel() != 1 {
        return Err(GradError::InvalidInput(
            "Function output must be scalar for gradient computation".to_string()
        ));
    }

    // Compute gradients
    grad(&[output], &grad_inputs, None, false, create_graph)
}

/// Utility function to check if a variable is in the computation graph
/// 変数が計算グラフに含まれているかチェックするユーティリティ関数
pub fn is_variable_in_graph<T>(var: &Variable<T>, visited: &mut HashSet<*const Variable<T>>) -> bool
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    let var_ptr = var as *const Variable<T>;
    if visited.contains(&var_ptr) {
        return true;
    }
    visited.insert(var_ptr);

    if let Some(grad_fn) = var.grad_fn() {
        // This is a simplified check - in a full implementation,
        // we would traverse the computation graph through the grad_fn
        true
    } else {
        false
    }
}

/// Validate gradient computation setup
/// 勾配計算設定を検証
pub fn validate_grad_setup<T>(
    outputs: &[Variable<T>],
    inputs: &[Variable<T>],
) -> Result<(), GradError>
where
    T: Float + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    // Check that at least one output requires gradients
    if !outputs.iter().any(|output| output.requires_grad()) {
        return Err(GradError::InvalidInput(
            "At least one output must require gradients".to_string()
        ));
    }

    // Check that at least one input requires gradients
    if !inputs.iter().any(|input| input.requires_grad()) {
        return Err(GradError::InvalidInput(
            "At least one input must require gradients".to_string()
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Variable;
    use crate::tensor::Tensor;

    #[test]
    fn test_grad_simple() {
        // f(x) = x^2, df/dx = 2x
        let x = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);
        let y = &x * &x; // x^2
        
        let gradients = grad(&[y], &[x.clone()], None, false, false).unwrap();
        
        assert!(gradients[0].is_some());
        let grad_val = gradients[0].as_ref().unwrap().as_array()[0];
        assert!((grad_val - 6.0).abs() < 1e-6); // 2 * 3 = 6
    }

    #[test]
    fn test_grad_multiple_inputs() {
        // f(x, y) = x * y, df/dx = y, df/dy = x
        let x = Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true);
        let y = Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true);
        let z = &x * &y;
        
        let gradients = grad(&[z], &[x.clone(), y.clone()], None, false, false).unwrap();
        
        assert!(gradients[0].is_some());
        assert!(gradients[1].is_some());
        
        let grad_x = gradients[0].as_ref().unwrap().as_array()[0];
        let grad_y = gradients[1].as_ref().unwrap().as_array()[0];
        
        assert!((grad_x - 3.0).abs() < 1e-6); // df/dx = y = 3
        assert!((grad_y - 2.0).abs() < 1e-6); // df/dy = x = 2
    }

    #[test]
    fn test_gradient_function() {
        let inputs = vec![
            Variable::new(Tensor::from_vec(vec![2.0f32], vec![1]), true),
            Variable::new(Tensor::from_vec(vec![3.0f32], vec![1]), true),
        ];

        let gradients = gradient(
            |vars| &vars[0] * &vars[0] + &vars[1] * &vars[1], // x^2 + y^2
            &inputs,
            false,
        ).unwrap();

        assert!(gradients[0].is_some());
        assert!(gradients[1].is_some());
        
        let grad_x = gradients[0].as_ref().unwrap().as_array()[0];
        let grad_y = gradients[1].as_ref().unwrap().as_array()[0];
        
        assert!((grad_x - 4.0).abs() < 1e-6); // d/dx(x^2 + y^2) = 2x = 4
        assert!((grad_y - 6.0).abs() < 1e-6); // d/dy(x^2 + y^2) = 2y = 6
    }
}
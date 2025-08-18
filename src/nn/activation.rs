//! Activation functions for neural networks
//! ニューラルネットワークの活性化関数

use crate::autograd::Variable;
use crate::tensor::Tensor;
use num_traits::Float;


/// ReLU (Rectified Linear Unit) activation function
/// ReLU（正規化線形ユニット）活性化関数
/// 
/// Applies the element-wise function: ReLU(x) = max(0, x)
/// 要素ごとに関数を適用: ReLU(x) = max(0, x)
pub fn relu<T: Float + Send + Sync + 'static>(x: &Variable<T>) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();
    
    // Apply ReLU: max(0, x)
    let output_data = apply_relu(&input_data);
    
    if x.requires_grad() {
        // Create a new variable that tracks gradients
        let result = Variable::new(output_data, true);
        
        // Store reference to input for backward pass
        // Note: In a full implementation, we'd use a proper gradient function
        // For now, we'll implement a simplified version
        result
    } else {
        Variable::new(output_data, false)
    }
}

/// Sigmoid activation function
/// シグモイド活性化関数
/// 
/// Applies the element-wise function: Sigmoid(x) = 1 / (1 + exp(-x))
/// 要素ごとに関数を適用: Sigmoid(x) = 1 / (1 + exp(-x))
pub fn sigmoid<T: Float + Send + Sync + 'static>(x: &Variable<T>) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();
    
    // Apply Sigmoid: 1 / (1 + exp(-x))
    let output_data = apply_sigmoid(&input_data);
    
    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// Tanh (Hyperbolic Tangent) activation function
/// Tanh（双曲線正接）活性化関数
/// 
/// Applies the element-wise function: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
/// 要素ごとに関数を適用: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
pub fn tanh<T: Float + Send + Sync + 'static>(x: &Variable<T>) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();
    
    // Apply Tanh
    let output_data = apply_tanh(&input_data);
    
    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// Leaky ReLU activation function
/// Leaky ReLU活性化関数
/// 
/// Applies the element-wise function: LeakyReLU(x) = max(alpha * x, x)
/// 要素ごとに関数を適用: LeakyReLU(x) = max(alpha * x, x)
pub fn leaky_relu<T: Float + Send + Sync + 'static>(x: &Variable<T>, alpha: T) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();
    
    // Apply Leaky ReLU: max(alpha * x, x)
    let output_data = apply_leaky_relu(&input_data, alpha);
    
    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

/// Softmax activation function
/// ソフトマックス活性化関数
/// 
/// Applies softmax along the last dimension: Softmax(x_i) = exp(x_i) / sum(exp(x_j))
/// 最後の次元に沿ってソフトマックスを適用: Softmax(x_i) = exp(x_i) / sum(exp(x_j))
pub fn softmax<T: Float + Send + Sync + 'static>(x: &Variable<T>) -> Variable<T> {
    let input_data = x.data().read().unwrap().clone();
    
    // Apply Softmax
    let output_data = apply_softmax(&input_data);
    
    if x.requires_grad() {
        Variable::new(output_data, true)
    } else {
        Variable::new(output_data, false)
    }
}

// Helper functions for applying activation functions to tensors
// テンソルに活性化関数を適用するヘルパー関数

fn apply_relu<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| {
        if x > T::zero() { x } else { T::zero() }
    }).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_sigmoid<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| {
        let one = T::one();
        one / (one + (-x).exp())
    }).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_tanh<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| {
        let exp_x = x.exp();
        let exp_neg_x = (-x).exp();
        (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    }).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_leaky_relu<T: Float + 'static>(tensor: &Tensor<T>, alpha: T) -> Tensor<T> {
    let data = tensor.as_array();
    let result_data: Vec<T> = data.iter().map(|&x| {
        if x > T::zero() { x } else { alpha * x }
    }).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

fn apply_softmax<T: Float + 'static>(tensor: &Tensor<T>) -> Tensor<T> {
    let data = tensor.as_array();
    
    // For numerical stability, subtract the maximum value
    let max_val = data.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
    
    // Compute exp(x - max) for each element
    let exp_values: Vec<T> = data.iter().map(|&x| (x - max_val).exp()).collect();
    
    // Compute sum of exponentials
    let sum_exp = exp_values.iter().fold(T::zero(), |acc, &x| acc + x);
    
    // Normalize by dividing by sum
    let result_data: Vec<T> = exp_values.iter().map(|&x| x / sum_exp).collect();
    
    Tensor::from_vec(result_data, tensor.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_relu() {
        let input = Variable::new(
            Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]),
            false
        );
        
        let output = relu(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();
        
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];
        for (actual, expected) in result_data.as_array().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_sigmoid() {
        let input = Variable::new(
            Tensor::from_vec(vec![0.0], vec![1]),
            false
        );
        
        let output = sigmoid(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();
        
        // sigmoid(0) = 0.5
        assert_abs_diff_eq!(result_data.as_array().iter().next().unwrap(), &0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh() {
        let input = Variable::new(
            Tensor::from_vec(vec![0.0], vec![1]),
            false
        );
        
        let output = tanh(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();
        
        // tanh(0) = 0.0
        assert_abs_diff_eq!(result_data.as_array().iter().next().unwrap(), &0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_leaky_relu() {
        let input = Variable::new(
            Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]),
            false
        );
        
        let output = leaky_relu(&input, 0.1);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();
        
        let expected = vec![-0.1, 0.0, 1.0];
        for (actual, expected) in result_data.as_array().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_softmax() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]),
            false
        );
        
        let output = softmax(&input);
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();
        
        // Check that probabilities sum to 1
        let sum: f32 = result_data.as_array().iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        
        // Check that all values are positive
        for &val in result_data.as_array().iter() {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_relu_with_gradients() {
        let input = Variable::new(
            Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]),
            true
        );
        
        let output = relu(&input);
        assert!(output.requires_grad());
        
        // Test that the computation works
        let result_binding = output.data();
        let result_data = result_binding.read().unwrap();
        let expected = vec![0.0, 0.0, 1.0];
        for (actual, expected) in result_data.as_array().iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }
}
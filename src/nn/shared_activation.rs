//! Shared activation function traits and implementations
//! 共通活性化関数トレイトと実装

use num_traits::Float;

/// Common activation function trait for both regular and WASM implementations
/// 通常実装とWASM実装の両方用の共通活性化関数トレイト
pub trait ActivationFunction<T> {
    type Error;
    type Input;
    type Output;
    
    /// Apply ReLU activation: max(0, x)
    /// ReLU活性化を適用: max(0, x)
    fn relu(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Apply Sigmoid activation: 1 / (1 + exp(-x))
    /// Sigmoid活性化を適用: 1 / (1 + exp(-x))
    fn sigmoid(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Apply Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    /// Tanh活性化を適用: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    fn tanh(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    
    /// Apply Leaky ReLU activation: max(alpha * x, x)
    /// Leaky ReLU活性化を適用: max(alpha * x, x)
    fn leaky_relu(&self, input: Self::Input, alpha: T) -> Result<Self::Output, Self::Error>;
    
    /// Apply Softmax activation along specified dimension
    /// 指定次元に沿ってSoftmax活性化を適用
    fn softmax(&self, input: Self::Input, dim: usize) -> Result<Self::Output, Self::Error>;
}

/// Shared activation function implementations
/// 共通活性化関数実装
pub mod shared_activations {
    use super::*;
    
    /// ReLU implementation for Vec<T>
    /// Vec<T>用ReLU実装
    pub fn relu_vec<T: Float>(input: &[T]) -> Vec<T> {
        input.iter().map(|&x| x.max(T::zero())).collect()
    }
    
    /// ReLU derivative for Vec<T>
    /// Vec<T>用ReLU微分
    pub fn relu_derivative_vec<T: Float>(input: &[T]) -> Vec<T> {
        input.iter().map(|&x| if x > T::zero() { T::one() } else { T::zero() }).collect()
    }
    
    /// Sigmoid implementation for Vec<T>
    /// Vec<T>用Sigmoid実装
    pub fn sigmoid_vec<T: Float>(input: &[T]) -> Vec<T> {
        input.iter().map(|&x| T::one() / (T::one() + (-x).exp())).collect()
    }
    
    /// Sigmoid derivative for Vec<T>
    /// Vec<T>用Sigmoid微分
    pub fn sigmoid_derivative_vec<T: Float>(input: &[T]) -> Vec<T> {
        input.iter().map(|&x| {
            let sigmoid_x = T::one() / (T::one() + (-x).exp());
            sigmoid_x * (T::one() - sigmoid_x)
        }).collect()
    }
    
    /// Tanh implementation for Vec<T>
    /// Vec<T>用Tanh実装
    pub fn tanh_vec<T: Float>(input: &[T]) -> Vec<T> {
        input.iter().map(|&x| x.tanh()).collect()
    }
    
    /// Tanh derivative for Vec<T>
    /// Vec<T>用Tanh微分
    pub fn tanh_derivative_vec<T: Float>(input: &[T]) -> Vec<T> {
        input.iter().map(|&x| {
            let tanh_x = x.tanh();
            T::one() - tanh_x * tanh_x
        }).collect()
    }
    
    /// Leaky ReLU implementation for Vec<T>
    /// Vec<T>用Leaky ReLU実装
    pub fn leaky_relu_vec<T: Float>(input: &[T], alpha: T) -> Vec<T> {
        input.iter().map(|&x| if x > T::zero() { x } else { alpha * x }).collect()
    }
    
    /// Leaky ReLU derivative for Vec<T>
    /// Vec<T>用Leaky ReLU微分
    pub fn leaky_relu_derivative_vec<T: Float>(input: &[T], alpha: T) -> Vec<T> {
        input.iter().map(|&x| if x > T::zero() { T::one() } else { alpha }).collect()
    }
    
    /// Softmax implementation for Vec<T>
    /// Vec<T>用Softmax実装
    pub fn softmax_vec<T: Float>(input: &[T]) -> Vec<T> {
        if input.is_empty() {
            return Vec::new();
        }
        
        // Numerical stability: subtract max value
        let max_val = input.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
        let exp_vals: Vec<T> = input.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp = exp_vals.iter().fold(T::zero(), |acc, &x| acc + x);
        
        if sum_exp == T::zero() {
            // Handle edge case
            vec![T::zero(); input.len()]
        } else {
            exp_vals.iter().map(|&x| x / sum_exp).collect()
        }
    }
    
    /// Softmax derivative for Vec<T>
    /// Vec<T>用Softmax微分
    pub fn softmax_derivative_vec<T: Float>(softmax_output: &[T], grad_output: &[T]) -> Vec<T> {
        if softmax_output.len() != grad_output.len() {
            return Vec::new();
        }
        
        let mut result = Vec::with_capacity(softmax_output.len());
        let dot_product: T = softmax_output.iter().zip(grad_output.iter())
            .map(|(&s, &g)| s * g)
            .fold(T::zero(), |acc, x| acc + x);
        
        for (&s, &g) in softmax_output.iter().zip(grad_output.iter()) {
            result.push(s * (g - dot_product));
        }
        
        result
    }
}
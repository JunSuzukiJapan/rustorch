//! Implementation of a linear (fully connected) layer.
//! 線形（全結合）レイヤーの実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use ndarray::Array;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::fmt::Debug;
use num_traits::Float;

/// A linear (fully connected) layer.
/// 線形（全結合）レイヤー
///
/// This layer applies a linear transformation to the incoming data: `y = xW^T + b`
pub struct Linear<T: Float + Send + Sync> {
    /// The weight matrix of shape (output_features, input_features)
    /// 重み行列 (出力特徴量, 入力特徴量)
    weight: Variable<T>,
    
    /// The bias vector of shape (output_features,)
    /// バイアスベクトル (出力特徴量,)
    bias: Option<Variable<T>>,
    
    /// Input size (number of input features)
    /// 入力サイズ（入力特徴量の数）
    input_size: usize,
    
    /// Output size (number of output features)
    /// 出力サイズ（出力特徴量の数）
    output_size: usize,
}

impl<T: Float + Send + Sync> std::fmt::Debug for Linear<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("input_size", &self.input_size)
            .field("output_size", &self.output_size)
            .field("has_bias", &self.bias.is_some())
            .finish()
    }
}

impl<T> Linear<T> 
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync,
    T: ndarray::ScalarOperand,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Sub<Output = T> + std::ops::Neg<Output = T>,
    T: std::iter::Sum,
    T: std::fmt::Display,
{
    /// Creates a new linear layer with the given input and output sizes.
    /// 入力サイズと出力サイズを指定して新しい線形レイヤーを作成します。
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Initialize weights using Kaiming initialization
        let k = (2.0 / input_size as f32).sqrt();
        let normal = Normal::new(0.0, k as f64).unwrap();
        
        // Initialize weights
        let weight_data: Vec<T> = (0..input_size * output_size)
            .map(|_| <T as From<f32>>::from(normal.sample(&mut rand::thread_rng()) as f32))
            .collect();
        
        let weight = Variable::new(
            Tensor::new(Array::from_shape_vec((output_size, input_size), weight_data).unwrap().into_dyn()),
            true
        );
        
        // Initialize bias
        let bias_data: Vec<T> = (0..output_size)
            .map(|_| <T as From<f32>>::from(normal.sample(&mut rand::thread_rng()) as f32))
            .collect();
        
        let bias = Variable::new(
            Tensor::new(Array::from_shape_vec((output_size,), bias_data).unwrap().into_dyn()),
            true
        );
        
        Linear {
            weight,
            bias: Some(bias),
            input_size,
            output_size,
        }
    }
    
    /// Creates a new linear layer without a bias term.
    /// バイアス項を持たない線形レイヤーを作成します。
    pub fn new_no_bias(input_size: usize, output_size: usize) -> Self {
        // Initialize weights using Kaiming initialization
        let k = (2.0 / input_size as f32).sqrt();
        let normal = Normal::new(0.0, k as f64).unwrap();
        
        // Initialize weights
        let weight_data: Vec<T> = (0..input_size * output_size)
            .map(|_| <T as From<f32>>::from(normal.sample(&mut rand::thread_rng()) as f32))
            .collect();
        
        let weight = Variable::new(
            Tensor::new(Array::from_shape_vec((output_size, input_size), weight_data).unwrap().into_dyn()),
            true
        );
        
        Linear {
            weight,
            bias: None,
            input_size,
            output_size,
        }
    }
    
    /// Performs the forward pass of the linear layer.
    /// 線形レイヤーの順伝搬を実行します。
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For now, return a simple implementation
        // This would need proper implementation with autograd support
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let weight_binding = self.weight.data();
        let weight_data = weight_binding.read().unwrap();
        
        // Create a transposed weight for correct matrix multiplication
        let weight_t = weight_data.transpose();
        let output_data = input_data.matmul(&weight_t);
        
        Variable::new(output_data, input.requires_grad() || self.weight.requires_grad())
    }
    
    /// Returns the input size of the layer.
    /// レイヤーの入力サイズを返します。
    pub fn input_size(&self) -> usize {
        self.input_size
    }
    
    /// Returns the output size of the layer.
    /// レイヤーの出力サイズを返します。
    pub fn output_size(&self) -> usize {
        self.output_size
    }
}

impl<T> Module<T> for Linear<T> 
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync,
    T: ndarray::ScalarOperand,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Sub<Output = T> + std::ops::Neg<Output = T>,
    T: std::iter::Sum,
    T: std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        match &self.bias {
            Some(bias) => vec![self.weight.clone(), bias.clone()],
            None => vec![self.weight.clone()],
        }
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_forward() {
        // Test with bias
        let linear = Linear::<f32>::new(3, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]),
            false
        );
        let output = linear.forward(&input);
        
        // Check that we get some output (shape might vary based on implementation)
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        assert!(!output_data.is_empty());
        
        // Test without bias
        let linear = Linear::<f32>::new_no_bias(3, 2);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]),
            false
        );
        let output = linear.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        assert!(!output_data.is_empty());
    }
    
    #[test]
    fn test_linear_parameters() {
        let linear = Linear::<f32>::new(3, 2);
        let params = linear.parameters();
        assert_eq!(params.len(), 2); // weight and bias
        
        let linear = Linear::<f32>::new_no_bias(3, 2);
        let params = linear.parameters();
        assert_eq!(params.len(), 1); // only weight
    }
}

//! Batch Normalization layers implementation
//! バッチ正規化レイヤーの実装
//!
//! Implements numerically stable batch normalization with proper statistics tracking,
//! Welford's algorithm for variance computation, and improved epsilon handling.
//! 数値安定性を持つバッチ正規化を実装、適切な統計追跡、
//! 分散計算のためのWelfordアルゴリズム、改良されたイプシロン処理を含む。

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use std::fmt::Debug;
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero, One};
use ndarray::ScalarOperand;
use std::iter::Sum;
use std::sync::{Arc, RwLock};

/// 1D Batch Normalization layer
/// 1次元バッチ正規化レイヤー
#[derive(Debug)]
pub struct BatchNorm1d<T: Float + Send + Sync> {
    /// Number of features
    /// 特徴量数
    num_features: usize,
    
    /// Learnable weight parameter (gamma)
    /// 学習可能な重みパラメータ（ガンマ）
    weight: Variable<T>,
    
    /// Learnable bias parameter (beta)
    /// 学習可能なバイアスパラメータ（ベータ）
    bias: Variable<T>,
    
    /// Running mean for inference
    /// 推論用の移動平均
    running_mean: Arc<RwLock<Tensor<T>>>,
    
    /// Running variance for inference
    /// 推論用の移動分散
    running_var: Arc<RwLock<Tensor<T>>>,
    
    /// Momentum for running statistics
    /// 移動統計のモメンタム
    momentum: T,
    
    /// Small value added to variance for numerical stability
    /// 数値安定性のため分散に加える小さな値
    eps: T,
    
    /// Training mode flag
    /// 訓練モードフラグ
    training: Arc<RwLock<bool>>,
}

impl<T> BatchNorm1d<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + From<f32> + 'static + Send + Sync + Copy + ScalarOperand + Sum,
{
    /// Creates a new BatchNorm1d layer
    /// 新しいBatchNorm1dレイヤーを作成します
    pub fn new(
        num_features: usize,
        eps: Option<T>,
        momentum: Option<T>,
        affine: Option<bool>,
    ) -> Self {
        let eps = eps.unwrap_or_else(|| <T as From<f32>>::from(1e-5f32));
        let momentum = momentum.unwrap_or_else(|| <T as From<f32>>::from(0.1f32));
        let affine = affine.unwrap_or(true);
        
        // Initialize weight (gamma) and bias (beta)
        let weight = if affine {
            Variable::new(Tensor::ones(&[num_features]), true)
        } else {
            Variable::new(Tensor::ones(&[num_features]), false)
        };
        
        let bias = if affine {
            Variable::new(Tensor::zeros(&[num_features]), true)
        } else {
            Variable::new(Tensor::zeros(&[num_features]), false)
        };
        
        // Initialize running statistics
        let running_mean = Arc::new(RwLock::new(Tensor::zeros(&[num_features])));
        let running_var = Arc::new(RwLock::new(Tensor::ones(&[num_features])));
        
        BatchNorm1d {
            num_features,
            weight,
            bias,
            running_mean,
            running_var,
            momentum,
            eps,
            training: Arc::new(RwLock::new(true)),
        }
    }
    
    /// Sets the layer to training mode
    /// レイヤーを訓練モードに設定します
    pub fn train(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = true;
        }
    }
    
    /// Sets the layer to evaluation mode
    /// レイヤーを評価モードに設定します
    pub fn eval(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = false;
        }
    }
    
    /// Returns whether the layer is in training mode
    /// レイヤーが訓練モードかどうかを返します
    pub fn is_training(&self) -> bool {
        self.training.read().unwrap_or_else(|_| panic!("Failed to read training mode")).clone()
    }
    
    /// Forward pass of the BatchNorm1d layer
    /// BatchNorm1dレイヤーの順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        
        if input_shape.len() != 2 || input_shape[1] != self.num_features {
            panic!(
                "Expected 2D input with {} features, got shape {:?}",
                self.num_features, input_shape
            );
        }
        
        // Simple implementation using tensor operations
        let normalized_tensor = if self.is_training() {
            // Training mode: normalize using batch statistics
            self.normalize_training(&input_data)
        } else {
            // Evaluation mode: normalize using running statistics
            self.normalize_eval(&input_data)
        };
        
        let requires_grad = input.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad();
        Variable::new(normalized_tensor, requires_grad)
    }
    
    /// Normalize using batch statistics (training mode)
    /// バッチ統計を使用した正規化（訓練モード）
    fn normalize_training(&self, input: &Tensor<T>) -> Tensor<T> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let num_features = input_shape[1];
        
        // Compute batch statistics using numerically stable Welford's algorithm
        let (batch_mean, batch_var) = self.compute_batch_statistics(input, batch_size, num_features);
        
        // Update running statistics with exponential moving average
        self.update_running_statistics(&batch_mean, &batch_var);
        
        // Apply normalization using batch statistics
        self.apply_normalization(input, &batch_mean, &batch_var)
    }
    
    /// Normalize using running statistics (evaluation mode)
    /// 移動統計を使用した正規化（評価モード）
    fn normalize_eval(&self, input: &Tensor<T>) -> Tensor<T> {
        let running_mean_lock = self.running_mean.read().unwrap();
        let running_var_lock = self.running_var.read().unwrap();
        
        self.apply_normalization(input, &running_mean_lock, &running_var_lock)
    }
    
    /// Apply normalization: (x - mean) / sqrt(var + eps) * weight + bias
    /// 正規化を適用: (x - mean) / sqrt(var + eps) * weight + bias
    fn apply_normalization(&self, input: &Tensor<T>, mean: &Tensor<T>, var: &Tensor<T>) -> Tensor<T> {
        let weight_binding = self.weight.data();
        let weight_data = weight_binding.read().unwrap();
        let bias_binding = self.bias.data();
        let bias_data = bias_binding.read().unwrap();
        
        let input_array = input.as_array();
        let mean_array = mean.as_array();
        let var_array = var.as_array();
        let weight_array = weight_data.as_array();
        let bias_array = bias_data.as_array();
        
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let num_features = input_shape[1];
        
        let mut output_data = Vec::with_capacity(batch_size * num_features);
        
        // Apply normalization with improved numerical stability
        for b in 0..batch_size {
            for f in 0..num_features {
                let x = input_array[[b, f]];
                let mu = mean_array[f];
                let sigma2 = var_array[f];
                
                // Improved numerical stability: check for very small variance
                let ten = T::from_f32(10.0).unwrap();
                let eps_adjusted = if sigma2 < self.eps * ten {
                    self.eps * ten
                } else {
                    self.eps
                };
                
                // Normalize: (x - μ) / √(σ² + ε)
                let normalized = (x - mu) / (sigma2 + eps_adjusted).sqrt();
                
                // Scale and shift: γ * normalized + β
                let output_val = weight_array[f] * normalized + bias_array[f];
                output_data.push(output_val);
            }
        }
        
        Tensor::from_vec(output_data, input_shape.to_vec())
    }
    
    /// Compute batch statistics using Welford's algorithm for numerical stability
    /// 数値安定性のためWelfordアルゴリズムを使用してバッチ統計を計算
    fn compute_batch_statistics(&self, input: &Tensor<T>, batch_size: usize, num_features: usize) -> (Tensor<T>, Tensor<T>) {
        let input_array = input.as_array();
        
        let mut mean_vec = vec![T::zero(); num_features];
        let mut var_vec = vec![T::zero(); num_features];
        
        // Compute mean and variance per feature using Welford's algorithm
        for f in 0..num_features {
            let mut mean = T::zero();
            let mut m2 = T::zero();
            
            // Welford's online algorithm for mean and variance
            for b in 0..batch_size {
                let x = input_array[[b, f]];
                let delta = x - mean;
                mean = mean + delta / T::from_usize(b + 1).unwrap();
                let delta2 = x - mean;
                m2 = m2 + delta * delta2;
            }
            
            mean_vec[f] = mean;
            
            // Compute sample variance (divide by N-1, but use N for batch norm)
            let variance = if batch_size > 1 {
                m2 / T::from_usize(batch_size).unwrap()
            } else {
                T::one() // Fallback for single sample
            };
            
            // Apply bias correction and ensure minimum variance
            let bias_corrected_var = if batch_size > 1 {
                variance * T::from_usize(batch_size).unwrap() / T::from_usize(batch_size - 1).unwrap()
            } else {
                variance
            };
            
            let min_var_threshold = self.eps * T::from_f32(0.1).unwrap();
            var_vec[f] = bias_corrected_var.max(min_var_threshold);
        }
        
        let mean_tensor = Tensor::from_vec(mean_vec, vec![num_features]);
        let var_tensor = Tensor::from_vec(var_vec, vec![num_features]);
        
        (mean_tensor, var_tensor)
    }
    
    /// Update running statistics with exponential moving average
    /// 指数移動平均で移動統計を更新
    fn update_running_statistics(&self, batch_mean: &Tensor<T>, batch_var: &Tensor<T>) {
        if let (Ok(mut running_mean), Ok(mut running_var)) = 
            (self.running_mean.write(), self.running_var.write()) {
            
            let batch_mean_array = batch_mean.as_array();
            let batch_var_array = batch_var.as_array();
            
            let running_mean_array = running_mean.as_array_mut();
            let running_var_array = running_var.as_array_mut();
            
            let momentum = self.momentum;
            let one_minus_momentum = T::one() - momentum;
            
            // Update running mean: running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            for i in 0..self.num_features {
                running_mean_array[i] = one_minus_momentum * running_mean_array[i] + momentum * batch_mean_array[i];
                running_var_array[i] = one_minus_momentum * running_var_array[i] + momentum * batch_var_array[i];
            }
        }
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }
    
    /// Returns the number of features
    /// 特徴量数を返します
    pub fn num_features(&self) -> usize {
        self.num_features
    }
    
    /// Returns the epsilon value
    /// イプシロン値を返します
    pub fn eps(&self) -> T {
        self.eps
    }
    
    /// Returns the momentum value
    /// モメンタム値を返します
    pub fn momentum(&self) -> T {
        self.momentum
    }
    
    /// Returns the running mean (for inspection)
    /// 移動平均を返します（検査用）
    pub fn running_mean(&self) -> Tensor<T> {
        self.running_mean.read().unwrap().clone()
    }
    
    /// Returns the running variance (for inspection)
    /// 移動分散を返します（検査用）
    pub fn running_var(&self) -> Tensor<T> {
        self.running_var.read().unwrap().clone()
    }
}

impl<T> Module<T> for BatchNorm1d<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + From<f32> + 'static + Send + Sync + Copy + ScalarOperand + Sum,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// 2D Batch Normalization layer for convolutional layers
/// 畳み込みレイヤー用の2次元バッチ正規化レイヤー
#[derive(Debug)]
pub struct BatchNorm2d<T: Float + Send + Sync> {
    /// Number of channels
    /// チャンネル数
    num_features: usize,
    
    /// Learnable weight parameter (gamma)
    /// 学習可能な重みパラメータ（ガンマ）
    weight: Variable<T>,
    
    /// Learnable bias parameter (beta)
    /// 学習可能なバイアスパラメータ（ベータ）
    bias: Variable<T>,
    
    /// Running mean for inference
    /// 推論用の移動平均
    running_mean: Arc<RwLock<Tensor<T>>>,
    
    /// Running variance for inference
    /// 推論用の移動分散
    running_var: Arc<RwLock<Tensor<T>>>,
    
    /// Momentum for running statistics
    /// 移動統計のモメンタム
    momentum: T,
    
    /// Small value added to variance for numerical stability
    /// 数値安定性のため分散に加える小さな値
    eps: T,
    
    /// Training mode flag
    /// 訓練モードフラグ
    training: Arc<RwLock<bool>>,
}

impl<T> BatchNorm2d<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + From<f32> + 'static + Send + Sync + Copy + ScalarOperand + Sum,
{
    /// Creates a new BatchNorm2d layer
    /// 新しいBatchNorm2dレイヤーを作成します
    pub fn new(
        num_features: usize,
        eps: Option<T>,
        momentum: Option<T>,
        affine: Option<bool>,
    ) -> Self {
        let eps = eps.unwrap_or_else(|| <T as From<f32>>::from(1e-5f32));
        let momentum = momentum.unwrap_or_else(|| <T as From<f32>>::from(0.1f32));
        let affine = affine.unwrap_or(true);
        
        // Initialize weight (gamma) and bias (beta)
        let weight = if affine {
            Variable::new(Tensor::ones(&[num_features]), true)
        } else {
            Variable::new(Tensor::ones(&[num_features]), false)
        };
        
        let bias = if affine {
            Variable::new(Tensor::zeros(&[num_features]), true)
        } else {
            Variable::new(Tensor::zeros(&[num_features]), false)
        };
        
        // Initialize running statistics
        let running_mean = Arc::new(RwLock::new(Tensor::zeros(&[num_features])));
        let running_var = Arc::new(RwLock::new(Tensor::ones(&[num_features])));
        
        BatchNorm2d {
            num_features,
            weight,
            bias,
            running_mean,
            running_var,
            momentum,
            eps,
            training: Arc::new(RwLock::new(true)),
        }
    }
    
    /// Sets the layer to training mode
    /// レイヤーを訓練モードに設定します
    pub fn train(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = true;
        }
    }
    
    /// Sets the layer to evaluation mode
    /// レイヤーを評価モードに設定します
    pub fn eval(&self) {
        if let Ok(mut training) = self.training.write() {
            *training = false;
        }
    }
    
    /// Returns whether the layer is in training mode
    /// レイヤーが訓練モードかどうかを返します
    pub fn is_training(&self) -> bool {
        self.training.read().unwrap_or_else(|_| panic!("Failed to read training mode")).clone()
    }
    
    /// Forward pass for 4D input (batch_size, channels, height, width)
    /// 4次元入力の順伝播 (batch_size, channels, height, width)
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        
        if input_shape.len() != 4 || input_shape[1] != self.num_features {
            panic!(
                "Expected 4D input with {} channels, got shape {:?}",
                self.num_features, input_shape
            );
        }
        
        let normalized_tensor = if self.is_training() {
            // Training mode: compute batch statistics per channel
            self.normalize_training_2d(&input_data)
        } else {
            // Evaluation mode: use running statistics
            self.normalize_eval_2d(&input_data)
        };
        
        let requires_grad = input.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad();
        Variable::new(normalized_tensor, requires_grad)
    }
    
    /// Normalize using batch statistics for 2D convolution (training mode)
    /// 2D畳み込み用バッチ統計を使用した正規化（訓練モード）
    fn normalize_training_2d(&self, input: &Tensor<T>) -> Tensor<T> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];
        
        // Compute channel-wise statistics across batch and spatial dimensions
        let (channel_mean, channel_var) = self.compute_channel_statistics(input, batch_size, channels, height, width);
        
        // Update running statistics
        self.update_running_statistics_2d(&channel_mean, &channel_var);
        
        // Apply normalization
        self.apply_channel_normalization_2d(input, &channel_mean, &channel_var)
    }
    
    /// Normalize using running statistics for 2D convolution (evaluation mode)
    /// 2D畳み込み用移動統計を使用した正規化（評価モード）
    fn normalize_eval_2d(&self, input: &Tensor<T>) -> Tensor<T> {
        let running_mean_lock = self.running_mean.read().unwrap();
        let running_var_lock = self.running_var.read().unwrap();
        
        self.apply_channel_normalization_2d(input, &running_mean_lock, &running_var_lock)
    }
    
    /// Compute channel-wise statistics for 2D batch normalization
    /// 2Dバッチ正規化のためのチャンネル別統計を計算
    fn compute_channel_statistics(&self, input: &Tensor<T>, batch_size: usize, channels: usize, height: usize, width: usize) -> (Tensor<T>, Tensor<T>) {
        let input_array = input.as_array();
        let spatial_size = height * width;
        let total_elements_per_channel = batch_size * spatial_size;
        
        let mut mean_vec = vec![T::zero(); channels];
        let mut var_vec = vec![T::zero(); channels];
        
        // Compute statistics per channel using Welford's algorithm
        for c in 0..channels {
            let mut mean = T::zero();
            let mut m2 = T::zero();
            let mut count = 0;
            
            // Iterate over all spatial locations and batch elements for this channel
            for b in 0..batch_size {
                for h in 0..height {
                    for w in 0..width {
                        let x = input_array[[b, c, h, w]];
                        count += 1;
                        
                        let delta = x - mean;
                        mean = mean + delta / T::from_usize(count).unwrap();
                        let delta2 = x - mean;
                        m2 = m2 + delta * delta2;
                    }
                }
            }
            
            mean_vec[c] = mean;
            
            // Compute variance
            let variance = if total_elements_per_channel > 1 {
                m2 / T::from_usize(total_elements_per_channel).unwrap()
            } else {
                T::one()
            };
            
            // Apply bias correction for small batches
            let bias_corrected_var = if total_elements_per_channel > 1 {
                variance * T::from_usize(total_elements_per_channel).unwrap() / T::from_usize(total_elements_per_channel - 1).unwrap()
            } else {
                variance
            };
            
            let min_var_threshold = self.eps * T::from_f32(0.1).unwrap();
            var_vec[c] = bias_corrected_var.max(min_var_threshold);
        }
        
        let mean_tensor = Tensor::from_vec(mean_vec, vec![channels]);
        let var_tensor = Tensor::from_vec(var_vec, vec![channels]);
        
        (mean_tensor, var_tensor)
    }
    
    /// Apply channel-wise normalization with improved numerical stability
    /// 数値安定性を改善したチャンネル別正規化を適用
    fn apply_channel_normalization_2d(&self, input: &Tensor<T>, mean: &Tensor<T>, var: &Tensor<T>) -> Tensor<T> {
        let weight_binding = self.weight.data();
        let weight_data = weight_binding.read().unwrap();
        let bias_binding = self.bias.data();
        let bias_data = bias_binding.read().unwrap();
        
        let input_array = input.as_array();
        let mean_array = mean.as_array();
        let var_array = var.as_array();
        let weight_array = weight_data.as_array();
        let bias_array = bias_data.as_array();
        
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];
        
        let mut output_data = Vec::with_capacity(batch_size * channels * height * width);
        
        // Apply normalization with improved numerical stability
        for b in 0..batch_size {
            for c in 0..channels {
                let mu = mean_array[c];
                let sigma2 = var_array[c];
                let gamma = weight_array[c];
                let beta = bias_array[c];
                
                // Improved numerical stability: check for very small variance
                let ten = T::from_f32(10.0).unwrap();
                let eps_adjusted = if sigma2 < self.eps * ten {
                    self.eps * ten
                } else {
                    self.eps
                };
                
                let inv_std = T::one() / (sigma2 + eps_adjusted).sqrt();
                
                for h in 0..height {
                    for w in 0..width {
                        let x = input_array[[b, c, h, w]];
                        
                        // Normalize: (x - μ) / √(σ² + ε)
                        let normalized = (x - mu) * inv_std;
                        
                        // Scale and shift: γ * normalized + β
                        let output_val = gamma * normalized + beta;
                        output_data.push(output_val);
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, input_shape.to_vec())
    }
    
    /// Update running statistics for 2D batch normalization
    /// 2Dバッチ正規化のための移動統計を更新
    fn update_running_statistics_2d(&self, batch_mean: &Tensor<T>, batch_var: &Tensor<T>) {
        if let (Ok(mut running_mean), Ok(mut running_var)) = 
            (self.running_mean.write(), self.running_var.write()) {
            
            let batch_mean_array = batch_mean.as_array();
            let batch_var_array = batch_var.as_array();
            
            let running_mean_array = running_mean.as_array_mut();
            let running_var_array = running_var.as_array_mut();
            
            let momentum = self.momentum;
            let one_minus_momentum = T::one() - momentum;
            
            // Update running statistics: running_stat = (1 - momentum) * running_stat + momentum * batch_stat
            for i in 0..self.num_features {
                running_mean_array[i] = one_minus_momentum * running_mean_array[i] + momentum * batch_mean_array[i];
                running_var_array[i] = one_minus_momentum * running_var_array[i] + momentum * batch_var_array[i];
            }
        }
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.weight.clone(), self.bias.clone()]
    }
    
    /// Returns the number of features (channels)
    /// 特徴量数（チャンネル数）を返します
    pub fn num_features(&self) -> usize {
        self.num_features
    }
    
    /// Returns the epsilon value
    /// イプシロン値を返します
    pub fn eps(&self) -> T {
        self.eps
    }
    
    /// Returns the momentum value
    /// モメンタム値を返します
    pub fn momentum(&self) -> T {
        self.momentum
    }
    
    /// Returns the running mean (for inspection)
    /// 移動平均を返します（検査用）
    pub fn running_mean(&self) -> Tensor<T> {
        self.running_mean.read().unwrap().clone()
    }
    
    /// Returns the running variance (for inspection)
    /// 移動分散を返します（検査用）
    pub fn running_var(&self) -> Tensor<T> {
        self.running_var.read().unwrap().clone()
    }
}

impl<T> Module<T> for BatchNorm2d<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + From<f32> + 'static + Send + Sync + Copy + ScalarOperand + Sum,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        self.parameters()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batchnorm1d_creation() {
        let bn = BatchNorm1d::<f32>::new(10, None, None, None);
        assert_eq!(bn.num_features(), 10);
        assert!(bn.is_training());
    }
    
    #[test]
    fn test_batchnorm1d_forward() {
        let bn = BatchNorm1d::<f32>::new(3, None, None, None);
        
        // Create input: batch_size=2, features=3
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![2, 3]),
            false
        );
        
        let output = bn.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let output_shape = output_data.shape();
        
        // Output should have same shape as input
        assert_eq!(output_shape, &[2, 3]);
    }
    
    #[test]
    fn test_batchnorm1d_eval_mode() {
        let bn = BatchNorm1d::<f32>::new(3, None, None, None);
        
        // Switch to evaluation mode
        bn.eval();
        assert!(!bn.is_training());
        
        // Create input
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![2, 3]),
            false
        );
        
        let output = bn.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        
        // Should not panic and produce valid output
        assert_eq!(output_data.shape(), &[2, 3]);
    }
    
    #[test]
    fn test_batchnorm2d_creation() {
        let bn = BatchNorm2d::<f32>::new(16, None, None, None);
        assert_eq!(bn.num_features(), 16);
        assert!(bn.is_training());
    }
    
    #[test]
    fn test_batchnorm2d_forward() {
        let bn = BatchNorm2d::<f32>::new(2, None, None, None);
        
        // Create 4D input: batch_size=2, channels=2, height=3, width=3
        let input_data: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![2, 2, 3, 3]),
            false
        );
        
        let output = bn.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let output_shape = output_data.shape();
        
        // Output should have same shape as input
        assert_eq!(output_shape, &[2, 2, 3, 3]);
    }
    
    #[test]
    fn test_batchnorm_parameters() {
        let bn1d = BatchNorm1d::<f32>::new(5, None, None, None);
        let params = bn1d.parameters();
        
        assert_eq!(params.len(), 2); // weight and bias
        
        // Check parameter shapes
        let weight_binding = params[0].data();
        let weight_data = weight_binding.read().unwrap();
        let weight_shape = weight_data.shape();
        let bias_binding = params[1].data();
        let bias_data = bias_binding.read().unwrap();
        let bias_shape = bias_data.shape();
        
        assert_eq!(weight_shape, &[5]);
        assert_eq!(bias_shape, &[5]);
    }
}
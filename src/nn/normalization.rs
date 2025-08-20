//! Normalization layers implementation
//! 正規化層の実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use std::fmt::Debug;
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero, One};
use ndarray::ScalarOperand;
use std::iter::Sum;

/// Layer Normalization
/// レイヤー正規化
/// 
/// Normalizes across the feature dimension for each sample independently.
/// 各サンプルについて特徴量次元で独立して正規化します。
#[derive(Debug)]
pub struct LayerNorm<T: Float + Send + Sync> {
    /// Learnable scale parameter (gamma)
    /// 学習可能なスケールパラメータ（ガンマ）
    weight: Variable<T>,
    
    /// Learnable shift parameter (beta)
    /// 学習可能なシフトパラメータ（ベータ）
    bias: Variable<T>,
    
    /// Normalized shape (dimensions to normalize over)
    /// 正規化形状（正規化する次元）
    normalized_shape: Vec<usize>,
    
    /// Small value added to variance for numerical stability
    /// 数値安定性のため分散に加える小さな値
    eps: T,
    
    /// Whether to use learnable affine transformation
    /// 学習可能なアフィン変換を使用するかどうか
    elementwise_affine: bool,
}

impl<T> LayerNorm<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy,
{
    /// Creates a new LayerNorm layer
    /// 新しいLayerNorm層を作成します
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: Option<T>,
        elementwise_affine: Option<bool>,
    ) -> Self {
        assert!(!normalized_shape.is_empty(), "normalized_shape cannot be empty");
        
        let eps = eps.unwrap_or_else(|| T::from_f32(1e-5).unwrap());
        let elementwise_affine = elementwise_affine.unwrap_or(true);
        
        let num_features: usize = normalized_shape.iter().product();
        
        // Initialize weight (gamma) to ones
        let weight_data = vec![T::one(); num_features];
        let weight = Variable::new(
            Tensor::from_vec(weight_data, normalized_shape.clone()),
            elementwise_affine,
        );
        
        // Initialize bias (beta) to zeros
        let bias_data = vec![T::zero(); num_features];
        let bias = Variable::new(
            Tensor::from_vec(bias_data, normalized_shape.clone()),
            elementwise_affine,
        );
        
        LayerNorm {
            weight,
            bias,
            normalized_shape,
            eps,
            elementwise_affine,
        }
    }
    
    /// Forward pass of LayerNorm
    /// LayerNormの順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        
        // Verify input shape compatibility
        self.verify_input_shape(input_shape);
        
        // Calculate normalization
        let normalized_data = self.layer_normalize(&input_data);
        
        let requires_grad = input.requires_grad() || 
            (self.elementwise_affine && (self.weight.requires_grad() || self.bias.requires_grad()));
        
        Variable::new(normalized_data, requires_grad)
    }
    
    /// Verify input shape compatibility
    /// 入力形状の互換性を確認
    fn verify_input_shape(&self, input_shape: &[usize]) {
        let norm_dims = self.normalized_shape.len();
        let input_dims = input_shape.len();
        
        if input_dims < norm_dims {
            panic!("Input has {} dimensions but normalized_shape has {} dimensions", 
                   input_dims, norm_dims);
        }
        
        // Check that the last norm_dims dimensions match
        let input_suffix = &input_shape[input_dims - norm_dims..];
        if input_suffix != self.normalized_shape.as_slice() {
            panic!("Input shape suffix {:?} doesn't match normalized_shape {:?}", 
                   input_suffix, self.normalized_shape);
        }
    }
    
    /// Perform layer normalization
    /// レイヤー正規化を実行
    fn layer_normalize(&self, input: &Tensor<T>) -> Tensor<T> {
        let input_array = input.as_array();
        let input_shape = input.shape();
        let norm_dims = self.normalized_shape.len();
        let input_dims = input_shape.len();
        
        // Calculate batch dimensions (dimensions before normalized dimensions)
        let batch_dims = input_dims - norm_dims;
        let batch_size: usize = input_shape[..batch_dims].iter().product();
        let feature_size: usize = self.normalized_shape.iter().product();
        
        let mut output_data = Vec::with_capacity(input_array.len());
        
        // Process each batch element
        for batch_idx in 0..batch_size {
            // Extract features for this batch element
            let mut features = Vec::with_capacity(feature_size);
            
            for feat_idx in 0..feature_size {
                let linear_idx = batch_idx * feature_size + feat_idx;
                if let Some(slice) = input_array.as_slice() {
                    features.push(slice[linear_idx]);
                } else {
                    // Fallback for non-contiguous arrays
                    let indices = self.unravel_index(linear_idx, input_shape);
                    features.push(input_array[indices.as_slice()]);
                }
            }
            
            // Calculate mean and variance for this batch element
            let mean = self.calculate_mean(&features);
            let variance = self.calculate_variance(&features, mean);
            let std = (variance + self.eps).sqrt();
            
            // Normalize and apply affine transformation
            for (feat_idx, &feature_val) in features.iter().enumerate() {
                let normalized = (feature_val - mean) / std;
                
                let final_val = if self.elementwise_affine {
                    let weight_binding = self.weight.data();
                    let weight_data = weight_binding.read().unwrap();
                    let bias_binding = self.bias.data();
                    let bias_data = bias_binding.read().unwrap();
                    
                    let weight_indices = self.unravel_index(feat_idx, &self.normalized_shape);
                    let bias_indices = weight_indices.clone();
                    
                    let gamma = weight_data.as_array()[weight_indices.as_slice()];
                    let beta = bias_data.as_array()[bias_indices.as_slice()];
                    
                    gamma * normalized + beta
                } else {
                    normalized
                };
                
                output_data.push(final_val);
            }
        }
        
        Tensor::from_vec(output_data, input_shape.to_vec())
    }
    
    /// Calculate mean of features
    /// 特徴量の平均を計算
    fn calculate_mean(&self, features: &[T]) -> T {
        let sum: T = features.iter().fold(T::zero(), |acc, &x| acc + x);
        sum / T::from_usize(features.len()).unwrap()
    }
    
    /// Calculate variance of features
    /// 特徴量の分散を計算
    fn calculate_variance(&self, features: &[T], mean: T) -> T {
        let sum_sq_diff: T = features.iter()
            .fold(T::zero(), |acc, &x| acc + (x - mean).powi(2));
        sum_sq_diff / T::from_usize(features.len()).unwrap()
    }
    
    /// Convert linear index to multi-dimensional indices
    /// 線形インデックスを多次元インデックスに変換
    fn unravel_index(&self, mut index: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; shape.len()];
        
        for i in (0..shape.len()).rev() {
            indices[i] = index % shape[i];
            index /= shape[i];
        }
        
        indices
    }
    
    /// Returns the normalized shape
    /// 正規化形状を返します
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }
    
    /// Returns the epsilon value
    /// イプシロン値を返します
    pub fn eps(&self) -> T {
        self.eps
    }
    
    /// Returns whether elementwise affine is enabled
    /// 要素ごとのアフィン変換が有効かどうかを返します
    pub fn elementwise_affine(&self) -> bool {
        self.elementwise_affine
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        if self.elementwise_affine {
            vec![self.weight.clone(), self.bias.clone()]
        } else {
            vec![]
        }
    }
}

impl<T> Module<T> for LayerNorm<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy,
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

/// Group Normalization
/// グループ正規化
/// 
/// Normalizes features by dividing channels into groups and normalizing within each group.
/// チャンネルをグループに分割し、各グループ内で正規化します。
#[derive(Debug)]
pub struct GroupNorm<T: Float + Send + Sync> {
    /// Learnable scale parameter (gamma)
    /// 学習可能なスケールパラメータ（ガンマ）
    weight: Variable<T>,
    
    /// Learnable shift parameter (beta)
    /// 学習可能なシフトパラメータ（ベータ）
    bias: Variable<T>,
    
    /// Number of groups
    /// グループ数
    num_groups: usize,
    
    /// Number of channels
    /// チャンネル数
    num_channels: usize,
    
    /// Small value added to variance for numerical stability
    /// 数値安定性のため分散に加える小さな値
    eps: T,
    
    /// Whether to use learnable affine transformation
    /// 学習可能なアフィン変換を使用するかどうか
    affine: bool,
}

impl<T> GroupNorm<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy,
{
    /// Creates a new GroupNorm layer
    /// 新しいGroupNorm層を作成します
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: Option<T>,
        affine: Option<bool>,
    ) -> Self {
        assert!(num_groups > 0, "num_groups must be greater than 0");
        assert!(num_channels > 0, "num_channels must be greater than 0");
        assert!(num_channels % num_groups == 0, 
                "num_channels ({}) must be divisible by num_groups ({})", 
                num_channels, num_groups);
        
        let eps = eps.unwrap_or_else(|| T::from_f32(1e-5).unwrap());
        let affine = affine.unwrap_or(true);
        
        // Initialize weight (gamma) to ones
        let weight_data = vec![T::one(); num_channels];
        let weight = Variable::new(
            Tensor::from_vec(weight_data, vec![num_channels]),
            affine,
        );
        
        // Initialize bias (beta) to zeros
        let bias_data = vec![T::zero(); num_channels];
        let bias = Variable::new(
            Tensor::from_vec(bias_data, vec![num_channels]),
            affine,
        );
        
        GroupNorm {
            weight,
            bias,
            num_groups,
            num_channels,
            eps,
            affine,
        }
    }
    
    /// Forward pass of GroupNorm
    /// GroupNormの順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        
        // Input should be (N, C, H, W) for 4D or (N, C, L) for 3D
        if input_shape.len() < 3 {
            panic!("GroupNorm expects at least 3D input (N, C, ...), got {:?}", input_shape);
        }
        
        let _batch_size = input_shape[0];
        let channels = input_shape[1];
        
        if channels != self.num_channels {
            panic!("Input channels {} doesn't match layer channels {}", channels, self.num_channels);
        }
        
        // Calculate normalization
        let normalized_data = self.group_normalize(&input_data);
        
        let requires_grad = input.requires_grad() || 
            (self.affine && (self.weight.requires_grad() || self.bias.requires_grad()));
        
        Variable::new(normalized_data, requires_grad)
    }
    
    /// Perform group normalization
    /// グループ正規化を実行
    fn group_normalize(&self, input: &Tensor<T>) -> Tensor<T> {
        let input_array = input.as_array();
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let spatial_size: usize = input_shape[2..].iter().product();
        
        let channels_per_group = channels / self.num_groups;
        let group_size = channels_per_group * spatial_size;
        
        let mut output_data = Vec::with_capacity(input_array.len());
        
        // Process each batch element
        for b in 0..batch_size {
            // Process each group
            for g in 0..self.num_groups {
                let group_start_channel = g * channels_per_group;
                let group_end_channel = (g + 1) * channels_per_group;
                
                // Collect all values in this group
                let mut group_values = Vec::with_capacity(group_size);
                
                for c in group_start_channel..group_end_channel {
                    for _s in 0..spatial_size {
                        let mut indices = vec![b, c];
                        let spatial_indices = self.unravel_spatial_index(_s, &input_shape[2..]);
                        indices.extend(spatial_indices);
                        
                        group_values.push(input_array[indices.as_slice()]);
                    }
                }
                
                // Calculate group statistics
                let mean = self.calculate_mean(&group_values);
                let variance = self.calculate_variance(&group_values, mean);
                let std = (variance + self.eps).sqrt();
                
                // Normalize and apply affine transformation for this group
                let mut value_idx = 0;
                for c in group_start_channel..group_end_channel {
                    for _s in 0..spatial_size {
                        let normalized = (group_values[value_idx] - mean) / std;
                        
                        let final_val = if self.affine {
                            let weight_binding = self.weight.data();
                            let weight_data = weight_binding.read().unwrap();
                            let bias_binding = self.bias.data();
                            let bias_data = bias_binding.read().unwrap();
                            
                            let gamma = weight_data.as_array()[[c]];
                            let beta = bias_data.as_array()[[c]];
                            
                            gamma * normalized + beta
                        } else {
                            normalized
                        };
                        
                        output_data.push(final_val);
                        value_idx += 1;
                    }
                }
            }
        }
        
        Tensor::from_vec(output_data, input_shape.to_vec())
    }
    
    /// Convert spatial linear index to multi-dimensional indices
    /// 空間線形インデックスを多次元インデックスに変換
    fn unravel_spatial_index(&self, mut index: usize, spatial_shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; spatial_shape.len()];
        
        for i in (0..spatial_shape.len()).rev() {
            indices[i] = index % spatial_shape[i];
            index /= spatial_shape[i];
        }
        
        indices
    }
    
    /// Calculate mean of values
    /// 値の平均を計算
    fn calculate_mean(&self, values: &[T]) -> T {
        let sum: T = values.iter().fold(T::zero(), |acc, &x| acc + x);
        sum / T::from_usize(values.len()).unwrap()
    }
    
    /// Calculate variance of values
    /// 値の分散を計算
    fn calculate_variance(&self, values: &[T], mean: T) -> T {
        let sum_sq_diff: T = values.iter()
            .fold(T::zero(), |acc, &x| acc + (x - mean).powi(2));
        sum_sq_diff / T::from_usize(values.len()).unwrap()
    }
    
    /// Returns the number of groups
    /// グループ数を返します
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }
    
    /// Returns the number of channels
    /// チャンネル数を返します
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
    
    /// Returns the epsilon value
    /// イプシロン値を返します
    pub fn eps(&self) -> T {
        self.eps
    }
    
    /// Returns whether affine transformation is enabled
    /// アフィン変換が有効かどうかを返します
    pub fn affine(&self) -> bool {
        self.affine
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        if self.affine {
            vec![self.weight.clone(), self.bias.clone()]
        } else {
            vec![]
        }
    }
}


/// RMS Normalization (Root Mean Square Normalization)
/// RMS正規化（二乗平均平方根正規化）
/// 
/// A simplified normalization that only uses RMS, without centering.
/// 中心化せずにRMSのみを使用する簡略化された正規化。
#[derive(Debug)]
pub struct RMSNorm<T: Float + Send + Sync> {
    /// Learnable scale parameter
    /// 学習可能なスケールパラメータ
    weight: Variable<T>,
    
    /// Normalized shape (dimensions to normalize over)
    /// 正規化形状（正規化する次元）
    normalized_shape: Vec<usize>,
    
    /// Small value added to variance for numerical stability
    /// 数値安定性のため分散に加える小さな値
    eps: T,
}

impl<T> RMSNorm<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    /// Creates a new RMSNorm layer
    /// 新しいRMSNorm層を作成します
    pub fn new(normalized_shape: Vec<usize>, eps: Option<T>) -> Self {
        assert!(!normalized_shape.is_empty(), "normalized_shape cannot be empty");
        
        let eps = eps.unwrap_or_else(|| T::from(1e-8).unwrap());
        let num_features: usize = normalized_shape.iter().product();
        
        // Initialize weight to ones
        let weight_data = vec![T::one(); num_features];
        let weight = Variable::new(
            Tensor::from_vec(weight_data, normalized_shape.clone()),
            true,
        );
        
        RMSNorm {
            weight,
            normalized_shape,
            eps,
        }
    }
    
    /// Forward pass of RMSNorm
    /// RMSNormの順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let input_shape = input_data.shape();
        
        // Verify input shape compatibility
        self.verify_input_shape(input_shape);
        
        // Calculate RMS normalization
        let normalized_data = self.rms_normalize(&input_data);
        
        let requires_grad = input.requires_grad() || self.weight.requires_grad();
        Variable::new(normalized_data, requires_grad)
    }
    
    /// Verify input shape compatibility
    /// 入力形状の互換性を確認
    fn verify_input_shape(&self, input_shape: &[usize]) {
        let norm_dims = self.normalized_shape.len();
        let input_dims = input_shape.len();
        
        if input_dims < norm_dims {
            panic!("Input has {} dimensions but normalized_shape has {} dimensions", 
                   input_dims, norm_dims);
        }
        
        // Check that the last norm_dims dimensions match
        let input_suffix = &input_shape[input_dims - norm_dims..];
        if input_suffix != self.normalized_shape.as_slice() {
            panic!("Input shape suffix {:?} doesn't match normalized_shape {:?}", 
                   input_suffix, self.normalized_shape);
        }
    }
    
    /// Perform RMS normalization
    /// RMS正規化を実行
    fn rms_normalize(&self, input: &Tensor<T>) -> Tensor<T> {
        let input_array = input.as_array();
        let input_shape = input.shape();
        let norm_dims = self.normalized_shape.len();
        let input_dims = input_shape.len();
        
        // Calculate batch dimensions
        let batch_dims = input_dims - norm_dims;
        let batch_size: usize = input_shape[..batch_dims].iter().product();
        let feature_size: usize = self.normalized_shape.iter().product();
        
        let mut output_data = Vec::with_capacity(input_array.len());
        
        // Process each batch element
        for batch_idx in 0..batch_size {
            // Extract features for this batch element
            let mut features = Vec::with_capacity(feature_size);
            
            for feat_idx in 0..feature_size {
                let linear_idx = batch_idx * feature_size + feat_idx;
                if let Some(slice) = input_array.as_slice() {
                    features.push(slice[linear_idx]);
                } else {
                    // Fallback for non-contiguous arrays
                    let indices = self.unravel_index(linear_idx, input_shape);
                    features.push(input_array[indices.as_slice()]);
                }
            }
            
            // Calculate RMS for this batch element
            let mean_square: T = features.iter()
                .fold(T::zero(), |acc, &x| acc + x.powi(2)) / T::from_usize(features.len()).unwrap();
            let rms = (mean_square + self.eps).sqrt();
            
            // Normalize and apply scale
            for (feat_idx, &feature_val) in features.iter().enumerate() {
                let normalized = feature_val / rms;
                
                let weight_binding = self.weight.data();
                let weight_data = weight_binding.read().unwrap();
                let weight_indices = self.unravel_index(feat_idx, &self.normalized_shape);
                let gamma = weight_data.as_array()[weight_indices.as_slice()];
                
                let final_val = gamma * normalized;
                output_data.push(final_val);
            }
        }
        
        Tensor::from_vec(output_data, input_shape.to_vec())
    }
    
    /// Convert linear index to multi-dimensional indices
    /// 線形インデックスを多次元インデックスに変換
    fn unravel_index(&self, mut index: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; shape.len()];
        
        for i in (0..shape.len()).rev() {
            indices[i] = index % shape[i];
            index /= shape[i];
        }
        
        indices
    }
    
    /// Returns the epsilon value
    /// イプシロン値を返します
    pub fn eps(&self) -> T {
        self.eps
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.weight.clone()]
    }
}

impl<T> Module<T> for GroupNorm<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
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

impl<T> Module<T> for RMSNorm<T>
where
    T: Float + Debug + Default + FromPrimitive + ToPrimitive + Zero + One + 'static + Send + Sync + Copy + ScalarOperand + Sum + std::fmt::Display,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        vec![self.weight.clone()]
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layer_norm_creation() {
        let layer_norm = LayerNorm::<f32>::new(vec![128], None, None);
        
        assert_eq!(layer_norm.normalized_shape(), &[128]);
        assert!(layer_norm.elementwise_affine());
        
        let params = layer_norm.parameters();
        assert_eq!(params.len(), 2); // weight and bias
    }
    
    #[test]
    fn test_layer_norm_forward() {
        let layer_norm = LayerNorm::<f32>::new(vec![4], None, None);
        
        // Create input: batch_size=2, features=4
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]),
            false
        );
        
        let output = layer_norm.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        
        assert_eq!(output_data.shape(), &[2, 4]);
    }
    
    #[test]
    fn test_group_norm_creation() {
        let group_norm = GroupNorm::<f32>::new(2, 8, None, None);
        
        assert_eq!(group_norm.num_groups, 2);
        assert_eq!(group_norm.num_channels, 8);
        assert!(group_norm.affine);
        
        let params = group_norm.parameters();
        assert_eq!(params.len(), 2); // weight and bias
    }
    
    #[test]
    fn test_rms_norm_creation() {
        let rms_norm = RMSNorm::<f32>::new(vec![64], None);
        
        assert_eq!(rms_norm.normalized_shape, vec![64]);
        
        let params = rms_norm.parameters();
        assert_eq!(params.len(), 1); // only weight, no bias
    }
}

//! Common utilities for transposed convolution layers
//! 転置畳み込み層の共通ユーティリティ

use crate::autograd::Variable;
use crate::tensor::Tensor;
use num_traits::Float;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::fmt::Debug;

/// Initialize weight tensor with Kaiming uniform distribution
pub fn initialize_weights<T>(
    weight_shape: Vec<usize>,
    fan_in: usize,
) -> Variable<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    let weight_size = weight_shape.iter().product::<usize>();
    let bound = (6.0 / fan_in as f32).sqrt();

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, bound).unwrap();
    let weight_data: Vec<T> = (0..weight_size)
        .map(|_| <T as From<f32>>::from(normal.sample(&mut rng)))
        .collect();

    let weight_tensor = Tensor::from_vec(weight_data, weight_shape);
    Variable::new(weight_tensor, true)
}

/// Initialize bias tensor
pub fn initialize_bias<T>(out_channels: usize, use_bias: bool) -> Option<Variable<T>>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    if use_bias {
        let bias_data = vec![T::default(); out_channels];
        let bias_tensor = Tensor::from_vec(bias_data, vec![out_channels]);
        Some(Variable::new(bias_tensor, true))
    } else {
        None
    }
}

/// Validate common transposed convolution parameters
pub fn validate_parameters(
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    output_padding: impl IntoIterator<Item = usize>,
    stride: impl IntoIterator<Item = usize>,
) {
    assert!(in_channels % groups == 0, "in_channels must be divisible by groups");
    assert!(out_channels % groups == 0, "out_channels must be divisible by groups");
    
    // Validate output_padding < stride for each dimension
    for (out_pad, str) in output_padding.into_iter().zip(stride.into_iter()) {
        assert!(out_pad < str, "output_padding must be less than stride in all dimensions");
    }
}

/// Add bias to output tensor for any dimensionality
pub fn add_bias_nd<T>(
    output: &mut [T],
    output_shape: &[usize],
    bias: &[T],
    out_channels: usize,
    spatial_dims: usize,
) where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    let batch_size = output_shape[0];
    let spatial_size: usize = output_shape[2..2+spatial_dims].iter().product();
    
    for b in 0..batch_size {
        for ch in 0..out_channels {
            let ch_offset = b * out_channels * spatial_size + ch * spatial_size;
            let bias_val = bias[ch];
            
            for i in 0..spatial_size {
                output[ch_offset + i] = output[ch_offset + i] + bias_val;
            }
        }
    }
}

/// Calculate fan_in for Kaiming initialization based on dimensionality
pub fn calculate_fan_in(out_channels: usize, groups: usize, kernel_size: &[usize]) -> usize {
    let kernel_volume: usize = kernel_size.iter().product();
    (out_channels / groups) * kernel_volume
}

/// Validate input tensor shape for transposed convolution
pub fn validate_input_shape(
    input_shape: &[usize],
    expected_channels: usize,
    expected_dims: usize,
) {
    assert_eq!(
        input_shape.len(),
        expected_dims + 2, // batch + channels + spatial dims
        "Input must be {}D tensor (batch, channels, {})",
        expected_dims + 2,
        if expected_dims == 1 { "length" } 
        else if expected_dims == 2 { "height, width" }
        else { "depth, height, width" }
    );
    assert_eq!(
        input_shape[1],
        expected_channels,
        "Input channels mismatch"
    );
}
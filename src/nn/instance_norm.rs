//! Instance normalization layers implementation  
//! インスタンス正規化レイヤーの実装

use crate::autograd::Variable;
use crate::nn::Module;
#[cfg(not(target_arch = "wasm32"))]
use crate::simd::vectorized;
use crate::tensor::Tensor;
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::fmt::Debug;

/// SIMD-optimized mean calculation for instance normalization
fn calculate_mean_simd<T>(data: &[T]) -> T
where
    T: Float
        + Debug
        + Default
        + From<f32>
        + 'static
        + Send
        + Sync
        + Copy
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let f32_data: &[f32] = unsafe { std::mem::transmute(data) };
            return T::from_f32(vectorized::mean_f32_simd(f32_data)).unwrap();
        }
    }
    
    // Fallback implementation (used for WASM and non-f32 types)
    data.iter().fold(T::default(), |acc, &x| acc + x) / T::from_f32(data.len() as f32).unwrap()
}

/// SIMD-optimized variance calculation for instance normalization
fn calculate_variance_simd<T>(data: &[T], mean: T) -> T
where
    T: Float
        + Debug
        + Default
        + From<f32>
        + 'static
        + Send
        + Sync
        + Copy
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let f32_data: &[f32] = unsafe { std::mem::transmute(data) };
            let f32_mean = mean.to_f32().unwrap();
            return T::from_f32(vectorized::variance_f32_simd(f32_data) - f32_mean * f32_mean).unwrap();
        }
    }
    
    // Fallback implementation (used for WASM and non-f32 types)
    data.iter()
        .map(|&x| (x - mean) * (x - mean))
        .fold(T::default(), |acc, x| acc + x)
        / T::from_f32(data.len() as f32).unwrap()
}

/// Parallel normalization with affine transformation
fn normalize_channel_parallel<T>(
    input_slice: &[T],
    output_slice: &mut [T],
    mean: T,
    std_dev: T,
    weight: Option<T>,
    bias: Option<T>,
) where
    T: Float
        + Debug
        + Default
        + From<f32>
        + 'static
        + Send
        + Sync
        + Copy
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
{
    if input_slice.len() >= 64 {
        // Use parallel processing for large slices
        input_slice
            .par_iter()
            .zip(output_slice.par_iter_mut())
            .for_each(|(&input_val, output_val)| {
                let normalized = (input_val - mean) / std_dev;
                *output_val = match (weight, bias) {
                    (Some(w), Some(b)) => normalized * w + b,
                    (Some(w), None) => normalized * w,
                    (None, Some(b)) => normalized + b,
                    (None, None) => normalized,
                };
            });
    } else {
        // Use sequential processing for small slices
        for (i, &input_val) in input_slice.iter().enumerate() {
            let normalized = (input_val - mean) / std_dev;
            output_slice[i] = match (weight, bias) {
                (Some(w), Some(b)) => normalized * w + b,
                (Some(w), None) => normalized * w,
                (None, Some(b)) => normalized + b,
                (None, None) => normalized,
            };
        }
    }
}

/// 1D Instance Normalization layer
/// 1次元インスタンス正規化レイヤー
#[derive(Debug)]
pub struct InstanceNorm1d<
    T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    num_features: usize,
    eps: T,
    momentum: T,
    affine: bool,
    track_running_stats: bool,
    weight: Option<Variable<T>>,
    bias: Option<Variable<T>>,
    running_mean: Option<Variable<T>>,
    running_var: Option<Variable<T>>,
}

impl<T> InstanceNorm1d<T>
where
    T: Float
        + Debug
        + Default
        + From<f32>
        + 'static
        + Send
        + Sync
        + Copy
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
{
    pub fn new(
        num_features: usize,
        eps: Option<T>,
        momentum: Option<T>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> Self {
        let eps = eps.unwrap_or_else(|| T::from_f32(1e-5).unwrap());
        let momentum = momentum.unwrap_or_else(|| T::from_f32(0.1).unwrap());
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(false);

        let weight = if affine {
            let weight_data = vec![T::from_f32(1.0).unwrap(); num_features];
            let weight_tensor = Tensor::from_vec(weight_data, vec![num_features]);
            Some(Variable::new(weight_tensor, true))
        } else {
            None
        };

        let bias = if affine {
            let bias_data = vec![T::default(); num_features];
            let bias_tensor = Tensor::from_vec(bias_data, vec![num_features]);
            Some(Variable::new(bias_tensor, true))
        } else {
            None
        };

        let (running_mean, running_var) = if track_running_stats {
            let mean_data = vec![T::default(); num_features];
            let var_data = vec![T::from_f32(1.0).unwrap(); num_features];
            let mean_tensor = Tensor::from_vec(mean_data, vec![num_features]);
            let var_tensor = Tensor::from_vec(var_data, vec![num_features]);
            (
                Some(Variable::new(mean_tensor, false)),
                Some(Variable::new(var_tensor, false)),
            )
        } else {
            (None, None)
        };

        Self {
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
        }
    }

    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_tensor = input.data();
        let input_guard = input_tensor.read().unwrap();
        let input_shape = input_guard.shape();

        // Input validation for 1D: (N, C, L)
        assert!(
            input_shape.len() == 3,
            "Input must be 3D tensor (batch, channels, length)"
        );
        assert_eq!(
            input_shape[1], self.num_features,
            "Channel dimension mismatch"
        );

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let length = input_shape[2];

        let input_data = input_guard.as_slice().unwrap();
        let mut output_data = vec![T::default(); input_data.len()];

        // Compute instance normalization for each sample and channel independently
        for n in 0..batch_size {
            for c in 0..channels {
                let channel_offset = n * channels * length + c * length;
                let channel_slice = &input_data[channel_offset..channel_offset + length];

                // SIMD-optimized mean calculation
                let mean = calculate_mean_simd(channel_slice);

                // SIMD-optimized variance calculation
                let variance = calculate_variance_simd(channel_slice, mean);
                let std_dev = (variance + self.eps).sqrt();

                // Get weight and bias values for this channel if affine
                let (weight_val, bias_val) = if self.affine {
                    let weight_data_arc = self.weight.as_ref().unwrap().data();
                    let weight_guard = weight_data_arc.read().unwrap();
                    let bias_data_arc = self.bias.as_ref().unwrap().data();
                    let bias_guard = bias_data_arc.read().unwrap();

                    (
                        Some(weight_guard.as_slice().unwrap()[c]),
                        Some(bias_guard.as_slice().unwrap()[c]),
                    )
                } else {
                    (None, None)
                };

                // Parallel normalization and affine transformation
                let output_slice = &mut output_data[channel_offset..channel_offset + length];
                normalize_channel_parallel(
                    channel_slice,
                    output_slice,
                    mean,
                    std_dev,
                    weight_val,
                    bias_val,
                );
            }
        }

        let output_tensor = Tensor::from_vec(output_data, input_shape.to_vec());
        Variable::new(output_tensor, input.requires_grad())
    }

    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight.clone());
        }
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

impl<T> Module<T> for InstanceNorm1d<T>
where
    T: Float
        + Debug
        + Default
        + From<f32>
        + 'static
        + Send
        + Sync
        + Copy
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
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

/// 2D Instance Normalization layer
/// 2次元インスタンス正規化レイヤー
#[derive(Debug)]
pub struct InstanceNorm2d<
    T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    num_features: usize,
    eps: T,
    momentum: T,
    affine: bool,
    track_running_stats: bool,
    weight: Option<Variable<T>>,
    bias: Option<Variable<T>>,
    running_mean: Option<Variable<T>>,
    running_var: Option<Variable<T>>,
}

impl<T> InstanceNorm2d<T>
where
    T: Float
        + Debug
        + Default
        + From<f32>
        + 'static
        + Send
        + Sync
        + Copy
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
{
    pub fn new(
        num_features: usize,
        eps: Option<T>,
        momentum: Option<T>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> Self {
        let eps = eps.unwrap_or_else(|| T::from_f32(1e-5).unwrap());
        let momentum = momentum.unwrap_or_else(|| T::from_f32(0.1).unwrap());
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(false);

        let weight = if affine {
            let weight_data = vec![T::from_f32(1.0).unwrap(); num_features];
            let weight_tensor = Tensor::from_vec(weight_data, vec![num_features]);
            Some(Variable::new(weight_tensor, true))
        } else {
            None
        };

        let bias = if affine {
            let bias_data = vec![T::default(); num_features];
            let bias_tensor = Tensor::from_vec(bias_data, vec![num_features]);
            Some(Variable::new(bias_tensor, true))
        } else {
            None
        };

        let (running_mean, running_var) = if track_running_stats {
            let mean_data = vec![T::default(); num_features];
            let var_data = vec![T::from_f32(1.0).unwrap(); num_features];
            let mean_tensor = Tensor::from_vec(mean_data, vec![num_features]);
            let var_tensor = Tensor::from_vec(var_data, vec![num_features]);
            (
                Some(Variable::new(mean_tensor, false)),
                Some(Variable::new(var_tensor, false)),
            )
        } else {
            (None, None)
        };

        Self {
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
        }
    }

    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_tensor = input.data();
        let input_guard = input_tensor.read().unwrap();
        let input_shape = input_guard.shape();

        // Input validation for 2D: (N, C, H, W)
        assert!(
            input_shape.len() == 4,
            "Input must be 4D tensor (batch, channels, height, width)"
        );
        assert_eq!(
            input_shape[1], self.num_features,
            "Channel dimension mismatch"
        );

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];
        let spatial_size = height * width;

        let input_data = input_guard.as_slice().unwrap();
        let mut output_data = vec![T::default(); input_data.len()];

        // Compute instance normalization for each sample and channel independently
        for n in 0..batch_size {
            for c in 0..channels {
                let channel_offset = n * channels * spatial_size + c * spatial_size;

                // Calculate mean and variance for this instance and channel
                let mut sum = T::default();
                for i in 0..spatial_size {
                    sum = sum + input_data[channel_offset + i];
                }
                let mean = sum / T::from_f32(spatial_size as f32).unwrap();

                let mut var_sum = T::default();
                for i in 0..spatial_size {
                    let diff = input_data[channel_offset + i] - mean;
                    var_sum = var_sum + diff * diff;
                }
                let variance = var_sum / T::from_f32(spatial_size as f32).unwrap();
                let std_dev = (variance + self.eps).sqrt();

                // Normalize and apply affine transformation
                for i in 0..spatial_size {
                    let idx = channel_offset + i;
                    let normalized = (input_data[idx] - mean) / std_dev;

                    output_data[idx] = if self.affine {
                        let weight_data_arc = self.weight.as_ref().unwrap().data();
                        let weight_guard = weight_data_arc.read().unwrap();
                        let bias_data_arc = self.bias.as_ref().unwrap().data();
                        let bias_guard = bias_data_arc.read().unwrap();
                        let weight_val = weight_guard.as_slice().unwrap()[c];
                        let bias_val = bias_guard.as_slice().unwrap()[c];
                        normalized * weight_val + bias_val
                    } else {
                        normalized
                    };
                }
            }
        }

        let output_tensor = Tensor::from_vec(output_data, input_shape.to_vec());
        Variable::new(output_tensor, input.requires_grad())
    }

    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight.clone());
        }
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

impl<T> Module<T> for InstanceNorm2d<T>
where
    T: Float
        + Debug
        + Default
        + From<f32>
        + 'static
        + Send
        + Sync
        + Copy
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
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

/// 3D Instance Normalization layer
/// 3次元インスタンス正規化レイヤー
#[derive(Debug)]
pub struct InstanceNorm3d<
    T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    num_features: usize,
    eps: T,
    momentum: T,
    affine: bool,
    track_running_stats: bool,
    weight: Option<Variable<T>>,
    bias: Option<Variable<T>>,
    running_mean: Option<Variable<T>>,
    running_var: Option<Variable<T>>,
}

impl<T> InstanceNorm3d<T>
where
    T: Float
        + Debug
        + Default
        + From<f32>
        + 'static
        + Send
        + Sync
        + Copy
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
{
    pub fn new(
        num_features: usize,
        eps: Option<T>,
        momentum: Option<T>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> Self {
        let eps = eps.unwrap_or_else(|| T::from_f32(1e-5).unwrap());
        let momentum = momentum.unwrap_or_else(|| T::from_f32(0.1).unwrap());
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(false);

        let weight = if affine {
            let weight_data = vec![T::from_f32(1.0).unwrap(); num_features];
            let weight_tensor = Tensor::from_vec(weight_data, vec![num_features]);
            Some(Variable::new(weight_tensor, true))
        } else {
            None
        };

        let bias = if affine {
            let bias_data = vec![T::default(); num_features];
            let bias_tensor = Tensor::from_vec(bias_data, vec![num_features]);
            Some(Variable::new(bias_tensor, true))
        } else {
            None
        };

        let (running_mean, running_var) = if track_running_stats {
            let mean_data = vec![T::default(); num_features];
            let var_data = vec![T::from_f32(1.0).unwrap(); num_features];
            let mean_tensor = Tensor::from_vec(mean_data, vec![num_features]);
            let var_tensor = Tensor::from_vec(var_data, vec![num_features]);
            (
                Some(Variable::new(mean_tensor, false)),
                Some(Variable::new(var_tensor, false)),
            )
        } else {
            (None, None)
        };

        Self {
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            weight,
            bias,
            running_mean,
            running_var,
        }
    }

    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_tensor = input.data();
        let input_guard = input_tensor.read().unwrap();
        let input_shape = input_guard.shape();

        // Input validation for 3D: (N, C, D, H, W)
        assert!(
            input_shape.len() == 5,
            "Input must be 5D tensor (batch, channels, depth, height, width)"
        );
        assert_eq!(
            input_shape[1], self.num_features,
            "Channel dimension mismatch"
        );

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let depth = input_shape[2];
        let height = input_shape[3];
        let width = input_shape[4];
        let spatial_size = depth * height * width;

        let input_data = input_guard.as_slice().unwrap();
        let mut output_data = vec![T::default(); input_data.len()];

        // Compute instance normalization for each sample and channel independently
        for n in 0..batch_size {
            for c in 0..channels {
                let channel_offset = n * channels * spatial_size + c * spatial_size;

                // Calculate mean and variance for this instance and channel
                let mut sum = T::default();
                for i in 0..spatial_size {
                    sum = sum + input_data[channel_offset + i];
                }
                let mean = sum / T::from_f32(spatial_size as f32).unwrap();

                let mut var_sum = T::default();
                for i in 0..spatial_size {
                    let diff = input_data[channel_offset + i] - mean;
                    var_sum = var_sum + diff * diff;
                }
                let variance = var_sum / T::from_f32(spatial_size as f32).unwrap();
                let std_dev = (variance + self.eps).sqrt();

                // Normalize and apply affine transformation
                for i in 0..spatial_size {
                    let idx = channel_offset + i;
                    let normalized = (input_data[idx] - mean) / std_dev;

                    output_data[idx] = if self.affine {
                        let weight_data_arc = self.weight.as_ref().unwrap().data();
                        let weight_guard = weight_data_arc.read().unwrap();
                        let bias_data_arc = self.bias.as_ref().unwrap().data();
                        let bias_guard = bias_data_arc.read().unwrap();
                        let weight_val = weight_guard.as_slice().unwrap()[c];
                        let bias_val = bias_guard.as_slice().unwrap()[c];
                        normalized * weight_val + bias_val
                    } else {
                        normalized
                    };
                }
            }
        }

        let output_tensor = Tensor::from_vec(output_data, input_shape.to_vec());
        Variable::new(output_tensor, input.requires_grad())
    }

    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight.clone());
        }
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

impl<T> Module<T> for InstanceNorm3d<T>
where
    T: Float
        + Debug
        + Default
        + From<f32>
        + 'static
        + Send
        + Sync
        + Copy
        + ndarray::ScalarOperand
        + num_traits::FromPrimitive,
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
    fn test_instance_norm_1d() {
        let layer: InstanceNorm1d<f32> = InstanceNorm1d::new(64, None, None, Some(true), None);
        assert_eq!(layer.num_features, 64);
        assert!(layer.affine);
        assert_eq!(layer.parameters().len(), 2); // weight + bias
    }

    #[test]
    fn test_instance_norm_2d() {
        let layer: InstanceNorm2d<f32> = InstanceNorm2d::new(32, None, None, Some(true), None);
        assert_eq!(layer.num_features, 32);
        assert!(layer.affine);
        assert_eq!(layer.parameters().len(), 2); // weight + bias
    }

    #[test]
    fn test_instance_norm_3d() {
        let layer: InstanceNorm3d<f32> = InstanceNorm3d::new(16, None, None, Some(false), None);
        assert_eq!(layer.num_features, 16);
        assert!(!layer.affine);
        assert_eq!(layer.parameters().len(), 0); // no weight/bias
    }
}

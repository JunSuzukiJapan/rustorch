//! 1D transposed convolution layer implementation
//! 1次元転置畳み込みレイヤーの実装

use crate::autograd::Variable;
use crate::nn::Module;
use crate::nn::conv_transpose_common;
use crate::tensor::Tensor;
use num_traits::Float;
use std::fmt::Debug;

/// 1D Transposed Convolution layer
/// 1次元転置畳み込み層
#[derive(Debug)]
pub struct ConvTranspose1d<
    T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    weight: Variable<T>,
    bias: Option<Variable<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
}

impl<T> ConvTranspose1d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        output_padding: Option<usize>,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> Self {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let output_padding = output_padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        // Validate parameters
        conv_transpose_common::validate_parameters(
            in_channels,
            out_channels,
            groups,
            [output_padding],
            [stride],
        );

        // Initialize weight tensor [in_channels, out_channels/groups, kernel_size]
        let weight_shape = vec![in_channels, out_channels / groups, kernel_size];
        let fan_in = conv_transpose_common::calculate_fan_in(out_channels, groups, &[kernel_size]);
        let weight = conv_transpose_common::initialize_weights(weight_shape, fan_in);

        let bias = conv_transpose_common::initialize_bias(out_channels, use_bias);

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        }
    }

    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_tensor = input.data();
        let input_guard = input_tensor.read().unwrap();
        let input_shape = input_guard.shape();
        
        // Input validation for 1D: (N, C, L)
        conv_transpose_common::validate_input_shape(input_shape, self.in_channels, 1);
        
        let batch_size = input_shape[0];
        let input_length = input_shape[2];
        let output_length = self.calculate_output_size(input_length);
        let output_shape = vec![batch_size, self.out_channels, output_length];
        
        let input_data = input_guard.as_slice().unwrap();
        let mut output_data = vec![T::default(); output_shape.iter().product()];
        
        self.transposed_conv_1d(input_data, &mut output_data, input_shape, &output_shape);
        
        if let Some(ref bias) = self.bias {
            let bias_data_arc = bias.data();
            let bias_guard = bias_data_arc.read().unwrap();
            conv_transpose_common::add_bias_nd(
                &mut output_data,
                &output_shape,
                bias_guard.as_slice().unwrap(),
                self.out_channels,
                1, // 1D spatial dimensions
            );
        }
        
        let output_tensor = Tensor::from_vec(output_data, output_shape);
        Variable::new(output_tensor, input.requires_grad())
    }

    pub fn calculate_output_size(&self, input_size: usize) -> usize {
        (input_size - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + self.output_padding + 1
    }

    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn transposed_conv_1d(
        &self,
        input: &[T],
        output: &mut [T],
        input_shape: &[usize],
        output_shape: &[usize],
    ) {
        let batch_size = input_shape[0];
        let input_length = input_shape[2];
        let output_length = output_shape[2];
        
        let weight_data_arc = self.weight.data();
        let weight_guard = weight_data_arc.read().unwrap();
        let weight_data = weight_guard.as_slice().unwrap();
        
        for b in 0..batch_size {
            for in_ch in 0..self.in_channels {
                let input_ch_offset = b * self.in_channels * input_length + in_ch * input_length;
                
                let group_id = in_ch / (self.in_channels / self.groups);
                let out_ch_start = group_id * (self.out_channels / self.groups);
                let out_ch_end = out_ch_start + (self.out_channels / self.groups);
                
                for out_ch in out_ch_start..out_ch_end {
                    let out_ch_in_group = out_ch - out_ch_start;
                    let output_ch_offset = b * self.out_channels * output_length + out_ch * output_length;
                    let weight_offset = in_ch * (self.out_channels / self.groups) * self.kernel_size + out_ch_in_group * self.kernel_size;
                    
                    for i_l in 0..input_length {
                        let input_idx = input_ch_offset + i_l;
                        let input_val = input[input_idx];
                        
                        let out_l_start = i_l * self.stride;
                        
                        for k in 0..self.kernel_size {
                            let out_l = out_l_start + k;
                            
                            if out_l >= self.padding && out_l < output_length + self.padding - self.output_padding {
                                let adj_out_l = out_l - self.padding;
                                
                                if adj_out_l < output_length {
                                    let output_idx = output_ch_offset + adj_out_l;
                                    let weight_idx = weight_offset + k;
                                    output[output_idx] = output[output_idx] + input_val * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

impl<T> Module<T> for ConvTranspose1d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy + ndarray::ScalarOperand + num_traits::FromPrimitive,
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
    fn test_conv_transpose_1d_creation() {
        let layer: ConvTranspose1d<f32> = ConvTranspose1d::new(
            64,        // in_channels
            32,        // out_channels
            3,         // kernel_size
            Some(2),   // stride
            Some(1),   // padding
            None,      // output_padding
            None,      // dilation
            None,      // groups
            Some(true), // bias
        );

        assert_eq!(layer.in_channels, 64);
        assert_eq!(layer.out_channels, 32);
        assert_eq!(layer.kernel_size, 3);
        assert_eq!(layer.stride, 2);
        assert_eq!(layer.padding, 1);
    }

    #[test]
    fn test_output_size_calculation_1d() {
        let layer: ConvTranspose1d<f32> = ConvTranspose1d::new(16, 8, 3, Some(2), Some(1), None, None, None, Some(true));
        
        let input_size = 10;
        let output_size = layer.calculate_output_size(input_size);
        
        // Expected: (10-1)*2 - 2*1 + 1*(3-1) + 0 + 1 = 18 - 2 + 2 + 1 = 19
        assert_eq!(output_size, 19);
    }
}
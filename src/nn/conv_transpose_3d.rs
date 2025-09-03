//! 3D transposed convolution layer implementation
//! 3次元転置畳み込みレイヤーの実装

use crate::autograd::Variable;
use crate::nn::conv_transpose_common;
use crate::nn::Module;
use crate::tensor::Tensor;
use num_traits::Float;
use std::fmt::Debug;

/// 3D Transposed Convolution layer
/// 3次元転置畳み込み層
#[derive(Debug)]
pub struct ConvTranspose3d<
    T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
    weight: Variable<T>,
    bias: Option<Variable<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
}

impl<T> ConvTranspose3d<T>
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: Option<(usize, usize, usize)>,
        output_padding: Option<(usize, usize, usize)>,
        dilation: Option<(usize, usize, usize)>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> Self {
        let stride = stride.unwrap_or((1, 1, 1));
        let padding = padding.unwrap_or((0, 0, 0));
        let output_padding = output_padding.unwrap_or((0, 0, 0));
        let dilation = dilation.unwrap_or((1, 1, 1));
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        // Validate parameters
        conv_transpose_common::validate_parameters(
            in_channels,
            out_channels,
            groups,
            [output_padding.0, output_padding.1, output_padding.2],
            [stride.0, stride.1, stride.2],
        );

        // Initialize weight tensor [in_channels, out_channels/groups, kernel_d, kernel_h, kernel_w]
        let weight_shape = vec![
            in_channels,
            out_channels / groups,
            kernel_size.0,
            kernel_size.1,
            kernel_size.2,
        ];
        let fan_in = conv_transpose_common::calculate_fan_in(
            out_channels,
            groups,
            &[kernel_size.0, kernel_size.1, kernel_size.2],
        );
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

        // Input validation for 3D: (N, C, D, H, W)
        conv_transpose_common::validate_input_shape(input_shape, self.in_channels, 3);

        let batch_size = input_shape[0];
        let input_d = input_shape[2];
        let input_h = input_shape[3];
        let input_w = input_shape[4];

        let (output_d, output_h, output_w) =
            self.calculate_output_size((input_d, input_h, input_w));
        let output_shape = vec![batch_size, self.out_channels, output_d, output_h, output_w];

        let input_data = input_guard.as_slice().unwrap();
        let mut output_data = vec![T::default(); output_shape.iter().product()];

        self.transposed_conv_3d(input_data, &mut output_data, input_shape, &output_shape);

        if let Some(ref bias) = self.bias {
            let bias_data_arc = bias.data();
            let bias_guard = bias_data_arc.read().unwrap();
            conv_transpose_common::add_bias_nd(
                &mut output_data,
                &output_shape,
                bias_guard.as_slice().unwrap(),
                self.out_channels,
                3, // 3D spatial dimensions
            );
        }

        let output_tensor = Tensor::from_vec(output_data, output_shape);
        Variable::new(output_tensor, input.requires_grad())
    }

    pub fn calculate_output_size(
        &self,
        input_size: (usize, usize, usize),
    ) -> (usize, usize, usize) {
        let (input_d, input_h, input_w) = input_size;
        let (kernel_d, kernel_h, kernel_w) = self.kernel_size;
        let (stride_d, stride_h, stride_w) = self.stride;
        let (pad_d, pad_h, pad_w) = self.padding;
        let (out_pad_d, out_pad_h, out_pad_w) = self.output_padding;
        let (dil_d, dil_h, dil_w) = self.dilation;

        let out_d = (input_d - 1) * stride_d - 2 * pad_d + dil_d * (kernel_d - 1) + out_pad_d + 1;
        let out_h = (input_h - 1) * stride_h - 2 * pad_h + dil_h * (kernel_h - 1) + out_pad_h + 1;
        let out_w = (input_w - 1) * stride_w - 2 * pad_w + dil_w * (kernel_w - 1) + out_pad_w + 1;

        (out_d, out_h, out_w)
    }

    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn transposed_conv_3d(
        &self,
        input: &[T],
        output: &mut [T],
        input_shape: &[usize],
        output_shape: &[usize],
    ) {
        let batch_size = input_shape[0];
        let input_d = input_shape[2];
        let input_h = input_shape[3];
        let input_w = input_shape[4];
        let output_d = output_shape[2];
        let output_h = output_shape[3];
        let output_w = output_shape[4];

        let weight_data_arc = self.weight.data();
        let weight_guard = weight_data_arc.read().unwrap();
        let weight_data = weight_guard.as_slice().unwrap();

        let (kernel_d, kernel_h, kernel_w) = self.kernel_size;
        let (stride_d, stride_h, stride_w) = self.stride;
        let (pad_d, pad_h, pad_w) = self.padding;
        let (out_pad_d, out_pad_h, out_pad_w) = self.output_padding;

        for b in 0..batch_size {
            let input_batch_offset = b * self.in_channels * input_d * input_h * input_w;
            let output_batch_offset = b * self.out_channels * output_d * output_h * output_w;

            for in_ch in 0..self.in_channels {
                let input_ch_offset = input_batch_offset + in_ch * input_d * input_h * input_w;

                let group_id = in_ch / (self.in_channels / self.groups);
                let out_ch_start = group_id * (self.out_channels / self.groups);
                let out_ch_end = out_ch_start + (self.out_channels / self.groups);

                for out_ch in out_ch_start..out_ch_end {
                    let out_ch_in_group = out_ch - out_ch_start;
                    let output_ch_offset =
                        output_batch_offset + out_ch * output_d * output_h * output_w;
                    let weight_offset =
                        in_ch * (self.out_channels / self.groups) * kernel_d * kernel_h * kernel_w
                            + out_ch_in_group * kernel_d * kernel_h * kernel_w;

                    for i_d in 0..input_d {
                        for i_h in 0..input_h {
                            for i_w in 0..input_w {
                                let input_idx =
                                    input_ch_offset + i_d * input_h * input_w + i_h * input_w + i_w;
                                let input_val = input[input_idx];

                                let out_d_start = i_d * stride_d;
                                let out_h_start = i_h * stride_h;
                                let out_w_start = i_w * stride_w;

                                for k_d in 0..kernel_d {
                                    for k_h in 0..kernel_h {
                                        for k_w in 0..kernel_w {
                                            let out_d = out_d_start + k_d;
                                            let out_h = out_h_start + k_h;
                                            let out_w = out_w_start + k_w;

                                            if out_d >= pad_d
                                                && out_h >= pad_h
                                                && out_w >= pad_w
                                                && out_d < output_d + pad_d - out_pad_d
                                                && out_h < output_h + pad_h - out_pad_h
                                                && out_w < output_w + pad_w - out_pad_w
                                            {
                                                let adj_out_d = out_d - pad_d;
                                                let adj_out_h = out_h - pad_h;
                                                let adj_out_w = out_w - pad_w;

                                                if adj_out_d < output_d
                                                    && adj_out_h < output_h
                                                    && adj_out_w < output_w
                                                {
                                                    let output_idx = output_ch_offset
                                                        + adj_out_d * output_h * output_w
                                                        + adj_out_h * output_w
                                                        + adj_out_w;
                                                    let weight_idx = weight_offset
                                                        + k_d * kernel_h * kernel_w
                                                        + k_h * kernel_w
                                                        + k_w;
                                                    output[output_idx] = output[output_idx]
                                                        + input_val * weight_data[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

impl<T> Module<T> for ConvTranspose3d<T>
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
    fn test_conv_transpose_3d_creation() {
        let layer: ConvTranspose3d<f32> = ConvTranspose3d::new(
            8,               // in_channels
            16,              // out_channels
            (3, 3, 3),       // kernel_size
            Some((2, 2, 2)), // stride
            Some((1, 1, 1)), // padding
            None,            // output_padding
            None,            // dilation
            None,            // groups
            Some(true),      // bias
        );

        assert_eq!(layer.in_channels, 8);
        assert_eq!(layer.out_channels, 16);
        assert_eq!(layer.kernel_size, (3, 3, 3));
        assert_eq!(layer.stride, (2, 2, 2));
        assert_eq!(layer.padding, (1, 1, 1));
    }

    #[test]
    fn test_output_size_calculation_3d() {
        let layer: ConvTranspose3d<f32> = ConvTranspose3d::new(
            4,
            8,
            (3, 3, 3),
            Some((2, 2, 2)),
            Some((1, 1, 1)),
            None,
            None,
            None,
            Some(true),
        );

        let input_size = (8, 8, 8);
        let output_size = layer.calculate_output_size(input_size);

        // Expected: (8-1)*2 - 2*1 + 1*(3-1) + 0 + 1 = 14 - 2 + 2 + 1 = 15 for each dimension
        assert_eq!(output_size, (15, 15, 15));
    }
}

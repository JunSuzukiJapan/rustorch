//! Implementation of 2D Convolution layer
//! 2次元畳み込みレイヤーの実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use ndarray::Array4;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::fmt::Debug;
use num_traits::Float;

/// 2D Convolution layer
/// 2次元畳み込みレイヤー
///
/// This layer applies 2D convolution over an input signal composed of several input planes.
/// 複数の入力プレーンからなる入力信号に対して2次元畳み込みを適用します。
///
/// Input shape: (batch_size, in_channels, height, width)
/// Output shape: (batch_size, out_channels, out_height, out_width)
pub struct Conv2d<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    /// Weight tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
    /// 重みテンソル (出力チャンネル, 入力チャンネル, カーネル高さ, カーネル幅)
    weight: Variable<T>,
    
    /// Bias tensor of shape (out_channels,)
    /// バイアステンソル (出力チャンネル,)
    bias: Option<Variable<T>>,
    
    /// Input channels
    /// 入力チャンネル数
    in_channels: usize,
    
    /// Output channels
    /// 出力チャンネル数
    out_channels: usize,
    
    /// Kernel size (height, width)
    /// カーネルサイズ (高さ, 幅)
    kernel_size: (usize, usize),
    
    /// Stride (height, width)
    /// ストライド (高さ, 幅)
    stride: (usize, usize),
    
    /// Padding (height, width)
    /// パディング (高さ, 幅)
    padding: (usize, usize),
}

impl<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> std::fmt::Debug for Conv2d<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv2d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("has_bias", &self.bias.is_some())
            .finish()
    }
}

impl<T> Conv2d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + num_traits::FromPrimitive,
    T: ndarray::ScalarOperand,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Sub<Output = T> + std::ops::Neg<Output = T>,
    T: std::iter::Sum,
    T: std::fmt::Display + Copy,
{
    /// Creates a new Conv2d layer
    /// 新しいConv2dレイヤーを作成します
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> Self {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let use_bias = bias.unwrap_or(true);
        
        // Initialize weights using Kaiming initialization
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let k = (2.0 / fan_in as f32).sqrt();
        let normal = Normal::new(0.0, k as f64).unwrap();
        
        // Initialize weights: (out_channels, in_channels, kernel_height, kernel_width)
        let weight_data: Vec<T> = (0..out_channels * in_channels * kernel_size.0 * kernel_size.1)
            .map(|_| <T as From<f32>>::from(normal.sample(&mut rand::thread_rng()) as f32))
            .collect();
        
        let weight = Variable::new(
            Tensor::from_vec(
                weight_data,
                vec![out_channels, in_channels, kernel_size.0, kernel_size.1]
            ),
            true
        );
        
        // Initialize bias if needed
        let bias_var = if use_bias {
            let bias_data: Vec<T> = (0..out_channels)
                .map(|_| <T as From<f32>>::from(normal.sample(&mut rand::thread_rng()) as f32))
                .collect();
            
            Some(Variable::new(
                Tensor::from_vec(bias_data, vec![out_channels]),
                true
            ))
        } else {
            None
        };
        
        Conv2d {
            weight,
            bias: bias_var,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }
    
    /// Computes the output size given input size
    /// 入力サイズから出力サイズを計算します
    pub fn compute_output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let out_height = (input_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_width = (input_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        (out_height, out_width)
    }
    
    /// Applies padding to input tensor
    /// 入力テンソルにパディングを適用します
    fn apply_padding(&self, input: &Array4<T>) -> Array4<T> {
        if self.padding == (0, 0) {
            return input.clone();
        }
        
        let (batch_size, channels, height, width) = input.dim();
        let new_height = height + 2 * self.padding.0;
        let new_width = width + 2 * self.padding.1;
        
        let mut padded = Array4::zeros((batch_size, channels, new_height, new_width));
        
        // Copy original data to center of padded tensor
        let mut padded_slice = padded.slice_mut(ndarray::s![
            ..,
            ..,
            self.padding.0..self.padding.0 + height,
            self.padding.1..self.padding.1 + width
        ]);
        padded_slice.assign(input);
        
        padded
    }
    
    /// Performs 2D convolution operation
    /// 2次元畳み込み演算を実行します
    fn conv2d_forward(&self, input: &Array4<T>, weight: &Array4<T>) -> Array4<T> {
        let (batch_size, _in_channels, input_height, input_width) = input.dim();
        let (out_channels, _in_channels_w, kernel_height, kernel_width) = weight.dim();
        
        // Apply padding
        let padded_input = self.apply_padding(input);
        let (_, _, padded_height, padded_width) = padded_input.dim();
        
        // Compute output dimensions
        let (out_height, out_width) = self.compute_output_size(input_height, input_width);
        
        let mut output = Array4::zeros((batch_size, out_channels, out_height, out_width));
        
        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = T::zero();
                        
                        for ic in 0..self.in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * self.stride.0 + kh;
                                    let iw = ow * self.stride.1 + kw;
                                    
                                    if ih < padded_height && iw < padded_width {
                                        sum = sum + padded_input[[b, ic, ih, iw]] * weight[[oc, ic, kh, kw]];
                                    }
                                }
                            }
                        }
                        
                        output[[b, oc, oh, ow]] = sum;
                    }
                }
            }
        }
        
        output
    }
    
    /// Forward pass of the Conv2d layer
    /// Conv2dレイヤーの順伝播
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        let weight_binding = self.weight.data();
        let weight_data = weight_binding.read().unwrap();
        
        // Ensure input is 4D: (batch_size, channels, height, width)
        let input_shape = input_data.shape();
        if input_shape.len() != 4 {
            panic!(
                "Conv2d expects 4D input (batch_size, channels, height, width), got shape {:?}",
                input_shape
            );
        }
        
        // Reshape tensors to Array4 for convolution
        let input_array = input_data.as_array()
            .view()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();
        
        let weight_array = weight_data.as_array()
            .view()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();
        
        // Perform convolution
        let output_array = self.conv2d_forward(&input_array.to_owned(), &weight_array.to_owned());
        let mut output_tensor = Tensor::new(output_array.into_dyn());
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_binding = bias.data();
            let bias_data = bias_binding.read().unwrap();
            let _bias_shape = output_tensor.shape();
            
            // Reshape bias for broadcasting: (1, out_channels, 1, 1)
            let bias_reshaped = bias_data.as_array().clone()
                .into_shape_with_order((1, self.out_channels, 1, 1)).unwrap();
            let bias_tensor = Tensor::new(bias_reshaped.into_dyn());
            
            output_tensor = &output_tensor + &bias_tensor;
        }
        
        let requires_grad = input.requires_grad() || self.weight.requires_grad()
            || self.bias.as_ref().map_or(false, |b| b.requires_grad());
        
        Variable::new(output_tensor, requires_grad)
    }
    
    /// Returns the parameters of the layer
    /// レイヤーのパラメータを返します
    pub fn parameters(&self) -> Vec<Variable<T>> {
        match &self.bias {
            Some(bias) => vec![self.weight.clone(), bias.clone()],
            None => vec![self.weight.clone()],
        }
    }
    
    /// Returns input channels
    /// 入力チャンネル数を返します
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    
    /// Returns output channels
    /// 出力チャンネル数を返します
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    
    /// Returns kernel size
    /// カーネルサイズを返します
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }
    
    /// Returns stride
    /// ストライドを返します
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }
    
    /// Returns padding
    /// パディングを返します
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }
}

impl<T> Module<T> for Conv2d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + num_traits::FromPrimitive,
    T: ndarray::ScalarOperand,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Sub<Output = T> + std::ops::Neg<Output = T>,
    T: std::iter::Sum,
    T: std::fmt::Display + Copy,
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
    fn test_conv2d_output_size() {
        let conv = Conv2d::<f32>::new(3, 64, (3, 3), Some((1, 1)), Some((1, 1)), Some(true));
        
        // Test output size calculation
        let (out_h, out_w) = conv.compute_output_size(32, 32);
        assert_eq!((out_h, out_w), (32, 32)); // With padding=1, stride=1, size preserved
        
        let conv_no_pad = Conv2d::<f32>::new(3, 64, (3, 3), Some((1, 1)), Some((0, 0)), Some(true));
        let (out_h, out_w) = conv_no_pad.compute_output_size(32, 32);
        assert_eq!((out_h, out_w), (30, 30)); // Without padding, size reduces by kernel_size - 1
    }
    
    #[test]
    fn test_conv2d_forward() {
        let conv = Conv2d::<f32>::new(1, 1, (3, 3), Some((1, 1)), Some((0, 0)), Some(false));
        
        // Create a simple input: batch=1, channels=1, height=4, width=4
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![1, 1, 4, 4]),
            false
        );
        
        let output = conv.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let output_shape = output_data.shape();
        
        // Expected output shape: (1, 1, 2, 2) for 4x4 input with 3x3 kernel, no padding
        assert_eq!(output_shape, &[1, 1, 2, 2]);
    }
    
    #[test]
    fn test_conv2d_with_padding() {
        let conv = Conv2d::<f32>::new(1, 1, (3, 3), Some((1, 1)), Some((1, 1)), Some(false));
        
        let input_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![1, 1, 3, 3]),
            false
        );
        
        let output = conv.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let output_shape = output_data.shape();
        
        // With padding=1, input 3x3 should remain 3x3
        assert_eq!(output_shape, &[1, 1, 3, 3]);
    }
    
    #[test]
    fn test_conv2d_parameters() {
        let conv_with_bias = Conv2d::<f32>::new(3, 64, (3, 3), None, None, Some(true));
        let params = conv_with_bias.parameters();
        assert_eq!(params.len(), 2); // weight and bias
        
        let conv_no_bias = Conv2d::<f32>::new(3, 64, (3, 3), None, None, Some(false));
        let params = conv_no_bias.parameters();
        assert_eq!(params.len(), 1); // only weight
    }
}
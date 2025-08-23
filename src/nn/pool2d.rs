//! Implementation of 2D Pooling layers
//! 2次元プーリングレイヤーの実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use ndarray::Array4;
use std::fmt::Debug;
use num_traits::Float;

/// 2D Max Pooling layer
/// 2次元最大プーリングレイヤー
///
/// Applies a 2D max pooling over an input signal composed of several input planes.
/// 複数の入力プレーンからなる入力信号に対して2次元最大プーリングを適用します。
#[derive(Debug)]
pub struct MaxPool2d {
    /// Kernel size (height, width)
    /// カーネルサイズ (高さ, 幅)
    kernel_size: (usize, usize),
    
    /// Stride (height, width)
    /// ストライド (高さ, 幅)
    stride: Option<(usize, usize)>,
    
    /// Padding (height, width)
    /// パディング (高さ, 幅)
    padding: (usize, usize),
}

impl MaxPool2d {
    /// Creates a new MaxPool2d layer
    /// 新しいMaxPool2dレイヤーを作成します
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> Self {
        let stride = stride.unwrap_or(kernel_size); // Default stride = kernel_size
        let padding = padding.unwrap_or((0, 0));
        
        MaxPool2d {
            kernel_size,
            stride: Some(stride),
            padding,
        }
    }
    
    /// Computes the output size given input size
    /// 入力サイズから出力サイズを計算します
    pub fn compute_output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let stride = self.stride.unwrap_or(self.kernel_size);
        let out_height = (input_height + 2 * self.padding.0 - self.kernel_size.0) / stride.0 + 1;
        let out_width = (input_width + 2 * self.padding.1 - self.kernel_size.1) / stride.1 + 1;
        (out_height, out_width)
    }
    
    /// Applies padding to input tensor
    /// 入力テンソルにパディングを適用します
    fn apply_padding<T: Float + Copy>(&self, input: &Array4<T>) -> Array4<T> {
        if self.padding == (0, 0) {
            return input.clone();
        }
        
        let (batch_size, channels, height, width) = input.dim();
        let new_height = height + 2 * self.padding.0;
        let new_width = width + 2 * self.padding.1;
        
        let mut padded = Array4::from_elem((batch_size, channels, new_height, new_width), T::neg_infinity());
        
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
    
    /// Performs max pooling operation
    /// 最大プーリング演算を実行します
    fn maxpool2d_forward<T: Float + Copy>(&self, input: &Array4<T>) -> Array4<T> {
        let (batch_size, channels, input_height, input_width) = input.dim();
        let stride = self.stride.unwrap_or(self.kernel_size);
        
        // Apply padding
        let padded_input = self.apply_padding(input);
        
        // Compute output dimensions
        let (out_height, out_width) = self.compute_output_size(input_height, input_width);
        
        let mut output = Array4::from_elem((batch_size, channels, out_height, out_width), T::neg_infinity());
        
        // Perform max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut max_val = T::neg_infinity();
                        
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;
                                
                                max_val = max_val.max(padded_input[[b, c, ih, iw]]);
                            }
                        }
                        
                        output[[b, c, oh, ow]] = max_val;
                    }
                }
            }
        }
        
        output
    }
    
    /// Forward pass of the MaxPool2d layer
    /// MaxPool2dレイヤーの順伝播
    pub fn forward<T>(&self, input: &Variable<T>) -> Variable<T>
    where
        T: Float + Copy + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        
        // Ensure input is 4D: (batch_size, channels, height, width)
        let input_shape = input_data.shape();
        if input_shape.len() != 4 {
            panic!(
                "MaxPool2d expects 4D input (batch_size, channels, height, width), got shape {:?}",
                input_shape
            );
        }
        
        // Reshape tensor to Array4 for pooling
        let input_array = input_data.as_array()
            .view()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();
        
        // Perform max pooling
        let output_array = self.maxpool2d_forward(&input_array.to_owned());
        let output_tensor = Tensor::new(output_array.into_dyn());
        
        // MaxPool doesn't have learnable parameters, so gradient requirement is inherited
        Variable::new(output_tensor, input.requires_grad())
    }
    
    /// Returns kernel size
    /// カーネルサイズを返します
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }
    
    /// Returns stride
    /// ストライドを返します
    pub fn stride(&self) -> (usize, usize) {
        self.stride.unwrap_or(self.kernel_size)
    }
    
    /// Returns padding
    /// パディングを返します
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }
}

impl<T> Module<T> for MaxPool2d
where
    T: Float + Copy + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        vec![] // MaxPool has no parameters
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// 2D Average Pooling layer
/// 2次元平均プーリングレイヤー
#[derive(Debug)]
pub struct AvgPool2d {
    /// Kernel size (height, width)
    /// カーネルサイズ (高さ, 幅)
    kernel_size: (usize, usize),
    
    /// Stride (height, width)
    /// ストライド (高さ, 幅)
    stride: Option<(usize, usize)>,
    
    /// Padding (height, width)
    /// パディング (高さ, 幅)
    padding: (usize, usize),
}

impl AvgPool2d {
    /// Creates a new AvgPool2d layer
    /// 新しいAvgPool2dレイヤーを作成します
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        let padding = padding.unwrap_or((0, 0));
        
        AvgPool2d {
            kernel_size,
            stride: Some(stride),
            padding,
        }
    }
    
    /// Computes the output size given input size
    /// 入力サイズから出力サイズを計算します
    pub fn compute_output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let stride = self.stride.unwrap_or(self.kernel_size);
        let out_height = (input_height + 2 * self.padding.0 - self.kernel_size.0) / stride.0 + 1;
        let out_width = (input_width + 2 * self.padding.1 - self.kernel_size.1) / stride.1 + 1;
        (out_height, out_width)
    }
    
    /// Applies padding to input tensor
    /// 入力テンソルにパディングを適用します
    fn apply_padding<T: Float + Copy>(&self, input: &Array4<T>) -> Array4<T> {
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
    
    /// Performs average pooling operation
    /// 平均プーリング演算を実行します
    fn avgpool2d_forward<T: Float + Copy>(&self, input: &Array4<T>) -> Array4<T> {
        let (batch_size, channels, input_height, input_width) = input.dim();
        let stride = self.stride.unwrap_or(self.kernel_size);
        
        // Apply padding
        let padded_input = self.apply_padding(input);
        
        // Compute output dimensions
        let (out_height, out_width) = self.compute_output_size(input_height, input_width);
        
        let mut output = Array4::zeros((batch_size, channels, out_height, out_width));
        let pool_size = T::from(self.kernel_size.0 * self.kernel_size.1).unwrap();
        
        // Perform average pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = T::zero();
                        
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;
                                
                                sum = sum + padded_input[[b, c, ih, iw]];
                            }
                        }
                        
                        output[[b, c, oh, ow]] = sum / pool_size;
                    }
                }
            }
        }
        
        output
    }
    
    /// Forward pass of the AvgPool2d layer
    /// AvgPool2dレイヤーの順伝播
    pub fn forward<T>(&self, input: &Variable<T>) -> Variable<T>
    where
        T: Float + Copy + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
    {
        let input_binding = input.data();
        let input_data = input_binding.read().unwrap();
        
        // Ensure input is 4D: (batch_size, channels, height, width)
        let input_shape = input_data.shape();
        if input_shape.len() != 4 {
            panic!(
                "AvgPool2d expects 4D input (batch_size, channels, height, width), got shape {:?}",
                input_shape
            );
        }
        
        // Reshape tensor to Array4 for pooling
        let input_array = input_data.as_array()
            .view()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap();
        
        // Perform average pooling
        let output_array = self.avgpool2d_forward(&input_array.to_owned());
        let output_tensor = Tensor::new(output_array.into_dyn());
        
        Variable::new(output_tensor, input.requires_grad())
    }
    
    /// Returns kernel size
    /// カーネルサイズを返します
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }
    
    /// Returns stride
    /// ストライドを返します
    pub fn stride(&self) -> (usize, usize) {
        self.stride.unwrap_or(self.kernel_size)
    }
    
    /// Returns padding
    /// パディングを返します
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }
}

impl<T> Module<T> for AvgPool2d
where
    T: Float + Copy + Send + Sync + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive,
{
    fn forward(&self, input: &Variable<T>) -> Variable<T> {
        self.forward(input)
    }
    
    fn parameters(&self) -> Vec<Variable<T>> {
        vec![] // AvgPool has no parameters
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_maxpool2d_output_size() {
        let pool = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
        
        let (out_h, out_w) = pool.compute_output_size(4, 4);
        assert_eq!((out_h, out_w), (2, 2));
        
        let (out_h, out_w) = pool.compute_output_size(8, 8);
        assert_eq!((out_h, out_w), (4, 4));
    }
    
    #[test]
    fn test_maxpool2d_forward() {
        let pool = MaxPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
        
        // Create input: batch=1, channels=1, height=4, width=4
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
        
        let output = pool.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let output_shape = output_data.shape();
        
        // Expected output shape: (1, 1, 2, 2)
        assert_eq!(output_shape, &[1, 1, 2, 2]);
        
        // Check values - max pooling should take max of each 2x2 region
        let output_values: Vec<f32> = output_data.as_array().iter().cloned().collect();
        assert_eq!(output_values, vec![6.0, 8.0, 14.0, 16.0]);
    }
    
    #[test]
    fn test_avgpool2d_forward() {
        let pool = AvgPool2d::new((2, 2), Some((2, 2)), Some((0, 0)));
        
        // Create input: batch=1, channels=1, height=2, width=2
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Variable::new(
            Tensor::from_vec(input_data, vec![1, 1, 2, 2]),
            false
        );
        
        let output = pool.forward(&input);
        let output_binding = output.data();
        let output_data = output_binding.read().unwrap();
        let output_shape = output_data.shape();
        
        // Expected output shape: (1, 1, 1, 1)
        assert_eq!(output_shape, &[1, 1, 1, 1]);
        
        // Check value - average of [1, 2, 3, 4] should be 2.5
        let output_values: Vec<f32> = output_data.as_array().iter().cloned().collect();
        assert_eq!(output_values, vec![2.5]);
    }
}
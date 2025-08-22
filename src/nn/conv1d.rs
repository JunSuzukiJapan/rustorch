//! 1D convolution layer implementation
//! 1次元畳み込みレイヤーの実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use std::fmt::Debug;
use num_traits::Float;
use rand::distributions::Distribution;
use rand_distr::Normal;

/// 1D Convolution layer for sequence processing
/// シーケンス処理用の1D畳み込み層
///
/// This layer applies 1D convolution over an input signal composed of several input channels.
/// 複数の入力チャンネルからなる入力信号に対して1次元畳み込みを適用します。
///
/// Input shape: (batch_size, in_channels, length)
/// Output shape: (batch_size, out_channels, out_length)
#[derive(Debug)]
pub struct Conv1d<T: Float + Send + Sync> {
    /// Weight tensor of shape (out_channels, in_channels/groups, kernel_size)
    /// 重みテンソル (出力チャンネル, 入力チャンネル/グループ, カーネルサイズ)
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
    
    /// Kernel size
    /// カーネルサイズ
    kernel_size: usize,
    
    /// Stride
    /// ストライド
    stride: usize,
    
    /// Padding
    /// パディング
    padding: usize,
    
    /// Dilation
    /// 膨張
    dilation: usize,
    
    /// Groups for grouped convolution
    /// グループ畳み込み用のグループ数
    groups: usize,
}

impl<T> Conv1d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    /// Create a new Conv1d layer
    /// 新しいConv1d層を作成
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> Self {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        // Validate parameters
        assert!(in_channels % groups == 0, "in_channels must be divisible by groups");
        assert!(out_channels % groups == 0, "out_channels must be divisible by groups");
        assert!(kernel_size > 0, "kernel_size must be positive");
        assert!(stride > 0, "stride must be positive");
        assert!(dilation > 0, "dilation must be positive");
        assert!(groups > 0, "groups must be positive");

        // Initialize weight tensor with shape [out_channels, in_channels/groups, kernel_size]
        let weight_shape = vec![out_channels, in_channels / groups, kernel_size];
        let weight_size = weight_shape.iter().product::<usize>();
        
        // Initialize with Kaiming uniform (suitable for ReLU)
        let fan_in = (in_channels / groups) * kernel_size;
        let bound = (6.0 / fan_in as f32).sqrt();
        
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, bound).unwrap();
        let weight_data: Vec<T> = (0..weight_size)
            .map(|_| <T as From<f32>>::from(normal.sample(&mut rng)))
            .collect();
        
        let weight_tensor = Tensor::from_vec(weight_data, weight_shape);
        let weight = Variable::new(weight_tensor, true);

        // Initialize bias tensor if needed
        let bias = if use_bias {
            let bias_data = vec![T::default(); out_channels];
            let bias_tensor = Tensor::from_vec(bias_data, vec![out_channels]);
            Some(Variable::new(bias_tensor, true))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Perform forward pass
    /// 順伝搬を実行
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For now, return a simple placeholder implementation
        // In a real implementation, this would perform 1D convolution
        input.clone()
    }
    
    /// Get layer parameters
    /// レイヤーのパラメータを取得
    pub fn parameters(&self) -> Vec<Variable<T>> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    /// Calculate output length for 1D convolution
    /// 1D畳み込みの出力長を計算
    pub fn calculate_output_length(&self, input_length: usize) -> usize {
        (input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride + 1
    }
    
    /// Create a Conv1d for text/sequence processing (common parameters)
    /// テキスト/シーケンス処理用のConv1d作成（一般的なパラメータ）
    pub fn for_text_processing(
        in_channels: usize, 
        out_channels: usize, 
        kernel_size: usize
    ) -> Self {
        Self::new(
            in_channels,
            out_channels,
            kernel_size,
            Some(1),        // stride
            Some(0),        // padding
            None,           // dilation
            None,           // groups
            Some(true),     // bias
        )
    }
    
    /// Create a Conv1d with same padding (output length = input length)
    /// 同一パディングのConv1d作成（出力長 = 入力長）
    pub fn with_same_padding(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Self {
        let padding = (kernel_size - 1) / 2;
        Self::new(
            in_channels,
            out_channels,
            kernel_size,
            Some(1),        // stride
            Some(padding),  // padding for same output size
            None,           // dilation
            None,           // groups
            Some(true),     // bias
        )
    }
    
    /// Get number of parameters
    /// パラメータ数を取得
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.out_channels * (self.in_channels / self.groups) * self.kernel_size;
        let bias_params = if self.bias.is_some() { self.out_channels } else { 0 };
        weight_params + bias_params
    }
    
    /// Get receptive field size
    /// 受容野サイズを取得
    pub fn receptive_field(&self) -> usize {
        self.dilation * (self.kernel_size - 1) + 1
    }
}

impl<T> Module<T> for Conv1d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
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
    fn test_conv1d_creation() {
        let layer: Conv1d<f32> = Conv1d::new(
            128,            // in_channels
            64,             // out_channels
            3,              // kernel_size
            Some(1),        // stride
            Some(1),        // padding
            None,           // dilation
            None,           // groups
            Some(true),     // bias
        );
        
        assert_eq!(layer.in_channels, 128);
        assert_eq!(layer.out_channels, 64);
        assert_eq!(layer.kernel_size, 3);
        assert_eq!(layer.stride, 1);
        assert_eq!(layer.padding, 1);
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_output_length_calculation() {
        let layer: Conv1d<f32> = Conv1d::new(
            10, 20, 3, Some(1), Some(1), None, None, Some(true)
        );
        
        let input_length = 100;
        let output_length = layer.calculate_output_length(input_length);
        
        // Expected: (100 + 2*1 - 1*(3-1) - 1) / 1 + 1 = (100 + 2 - 2 - 1) / 1 + 1 = 100
        assert_eq!(output_length, 100);
    }

    #[test]
    fn test_for_text_processing() {
        let layer: Conv1d<f32> = Conv1d::for_text_processing(300, 100, 5);
        
        assert_eq!(layer.in_channels, 300);
        assert_eq!(layer.out_channels, 100);
        assert_eq!(layer.kernel_size, 5);
        assert_eq!(layer.stride, 1);
        assert_eq!(layer.padding, 0);
    }

    #[test]
    fn test_with_same_padding() {
        let layer: Conv1d<f32> = Conv1d::with_same_padding(64, 128, 3);
        
        assert_eq!(layer.kernel_size, 3);
        assert_eq!(layer.padding, 1); // (3-1)/2 = 1
        assert_eq!(layer.stride, 1);
        
        // With same padding, output length should equal input length
        let input_length = 50;
        let output_length = layer.calculate_output_length(input_length);
        assert_eq!(output_length, input_length);
    }

    #[test]
    fn test_num_parameters() {
        let layer: Conv1d<f32> = Conv1d::new(
            64, 32, 5, None, None, None, None, Some(true)
        );
        
        // Weight: 32 * 64 * 5 = 10240
        // Bias: 32
        // Total: 10272
        assert_eq!(layer.num_parameters(), 10272);
    }

    #[test]
    fn test_receptive_field() {
        let layer: Conv1d<f32> = Conv1d::new(
            16, 32, 3, None, None, Some(2), None, None
        );
        
        // Receptive field: 2 * (3 - 1) + 1 = 5
        assert_eq!(layer.receptive_field(), 5);
    }

    #[test]
    fn test_parameters() {
        let layer: Conv1d<f32> = Conv1d::new(
            8, 4, 3, None, None, None, None, Some(true)
        );
        
        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_no_bias() {
        let layer: Conv1d<f32> = Conv1d::new(
            8, 4, 3, None, None, None, None, Some(false)
        );
        
        let params = layer.parameters();
        assert_eq!(params.len(), 1); // weight only
        assert!(layer.bias.is_none());
    }

    #[test]
    fn test_grouped_convolution() {
        let layer: Conv1d<f32> = Conv1d::new(
            32, 64, 3, None, None, None, Some(4), Some(true)
        );
        
        assert_eq!(layer.groups, 4);
        // Weight shape should be [64, 32/4, 3] = [64, 8, 3]
        assert_eq!(layer.num_parameters(), 64 * 8 * 3 + 64); // weight + bias
    }
}
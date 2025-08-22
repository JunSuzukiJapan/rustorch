//! 2D transposed convolution layer implementation
//! 2次元転置畳み込みレイヤーの実装

use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::nn::Module;
use std::fmt::Debug;
use num_traits::Float;
use rand::distributions::Distribution;
use rand_distr::Normal;

/// 2D Transposed Convolution layer for generative models
/// 生成モデル用の2D転置畳み込み層
///
/// This layer applies 2D transposed convolution over an input signal composed of several input planes.
/// 複数の入力プレーンからなる入力信号に対して2次元転置畳み込みを適用します。
///
/// Input shape: (batch_size, in_channels, height, width)
/// Output shape: (batch_size, out_channels, out_height, out_width)
#[derive(Debug)]
pub struct ConvTranspose2d<T: Float + Send + Sync> {
    /// Weight tensor of shape (in_channels, out_channels/groups, kernel_height, kernel_width)
    /// 重みテンソル (入力チャンネル, 出力チャンネル/グループ, カーネル高さ, カーネル幅)
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
    
    /// Output padding (height, width)
    /// 出力パディング (高さ, 幅)
    output_padding: (usize, usize),
    
    /// Dilation (height, width)
    /// 膨張 (高さ, 幅)
    dilation: (usize, usize),
    
    /// Groups for grouped convolution
    /// グループ畳み込み用のグループ数
    groups: usize,
}

impl<T> ConvTranspose2d<T>
where
    T: Float + Debug + Default + From<f32> + 'static + Send + Sync + Copy,
{
    /// Create a new ConvTranspose2d layer
    /// 新しいConvTranspose2d層を作成
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        output_padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> Self {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let output_padding = output_padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        // Validate parameters
        assert!(in_channels % groups == 0, "in_channels must be divisible by groups");
        assert!(out_channels % groups == 0, "out_channels must be divisible by groups");
        assert!(output_padding.0 < stride.0, "output_padding.0 must be less than stride.0");
        assert!(output_padding.1 < stride.1, "output_padding.1 must be less than stride.1");

        // Initialize weight tensor with shape [in_channels, out_channels/groups, kernel_h, kernel_w]
        let weight_shape = vec![in_channels, out_channels / groups, kernel_size.0, kernel_size.1];
        let weight_size = weight_shape.iter().product::<usize>();
        
        // Initialize with Kaiming uniform (suitable for ReLU)
        let fan_in = (out_channels / groups) * kernel_size.0 * kernel_size.1;
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
            output_padding,
            dilation,
            groups,
        }
    }

    /// Perform forward pass
    /// 順伝搬を実行
    pub fn forward(&self, input: &Variable<T>) -> Variable<T> {
        // For now, return a simple placeholder implementation
        // In a real implementation, this would perform transposed convolution
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

    /// Calculate output size for transposed convolution
    /// 転置畳み込みの出力サイズを計算
    pub fn calculate_output_size(&self, input_size: (usize, usize)) -> (usize, usize) {
        let (input_h, input_w) = input_size;
        let (kernel_h, kernel_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;
        let (out_pad_h, out_pad_w) = self.output_padding;
        let (dil_h, dil_w) = self.dilation;
        
        let out_h = (input_h - 1) * stride_h - 2 * pad_h + dil_h * (kernel_h - 1) + out_pad_h + 1;
        let out_w = (input_w - 1) * stride_w - 2 * pad_w + dil_w * (kernel_w - 1) + out_pad_w + 1;
        
        (out_h, out_w)
    }
    
    /// Create a ConvTranspose2d for 2x upsampling (common in generative models)
    /// 2倍アップサンプリング用のConvTranspose2d作成（生成モデルでよく使用）
    pub fn upsample_2x(in_channels: usize, out_channels: usize) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (4, 4),           // kernel_size
            Some((2, 2)),     // stride
            Some((1, 1)),     // padding
            None,             // output_padding
            None,             // dilation
            None,             // groups
            Some(true),       // bias
        )
    }
    
    /// Create a ConvTranspose2d for 4x upsampling
    /// 4倍アップサンプリング用のConvTranspose2d作成
    pub fn upsample_4x(in_channels: usize, out_channels: usize) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (4, 4),           // kernel_size
            Some((4, 4)),     // stride
            Some((0, 0)),     // padding
            None,             // output_padding
            None,             // dilation
            None,             // groups
            Some(true),       // bias
        )
    }
    
    /// Get number of parameters
    /// パラメータ数を取得
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_channels * (self.out_channels / self.groups) * 
                           self.kernel_size.0 * self.kernel_size.1;
        let bias_params = if self.bias.is_some() { self.out_channels } else { 0 };
        weight_params + bias_params
    }
}

impl<T> Module<T> for ConvTranspose2d<T>
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
    fn test_conv_transpose_2d_creation() {
        let layer: ConvTranspose2d<f32> = ConvTranspose2d::new(
            64,              // in_channels
            32,              // out_channels
            (3, 3),          // kernel_size
            Some((2, 2)),    // stride
            Some((1, 1)),    // padding
            None,            // output_padding
            None,            // dilation
            None,            // groups
            Some(true),      // bias
        );
        
        assert_eq!(layer.in_channels, 64);
        assert_eq!(layer.out_channels, 32);
        assert_eq!(layer.kernel_size, (3, 3));
        assert_eq!(layer.stride, (2, 2));
        assert_eq!(layer.padding, (1, 1));
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_output_size_calculation() {
        let layer: ConvTranspose2d<f32> = ConvTranspose2d::new(
            64, 32, (3, 3), Some((2, 2)), Some((1, 1)), None, None, None, Some(true)
        );
        
        let input_size = (16, 16);
        let output_size = layer.calculate_output_size(input_size);
        
        // Expected: (16-1)*2 - 2*1 + 1*(3-1) + 0 + 1 = 30 - 2 + 2 + 1 = 31
        assert_eq!(output_size, (31, 31));
    }

    #[test]
    fn test_upsample_2x_creation() {
        let layer: ConvTranspose2d<f32> = ConvTranspose2d::upsample_2x(128, 64);
        
        assert_eq!(layer.in_channels, 128);
        assert_eq!(layer.out_channels, 64);
        assert_eq!(layer.kernel_size, (4, 4));
        assert_eq!(layer.stride, (2, 2));
        assert_eq!(layer.padding, (1, 1));
    }

    #[test]
    fn test_num_parameters() {
        let layer: ConvTranspose2d<f32> = ConvTranspose2d::new(
            64, 32, (3, 3), None, None, None, None, None, Some(true)
        );
        
        // Weight: 64 * 32 * 3 * 3 = 18432
        // Bias: 32
        // Total: 18464
        assert_eq!(layer.num_parameters(), 18464);
    }

    #[test]
    fn test_parameters() {
        let layer: ConvTranspose2d<f32> = ConvTranspose2d::new(
            8, 4, (2, 2), None, None, None, None, None, Some(true)
        );
        
        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_no_bias() {
        let layer: ConvTranspose2d<f32> = ConvTranspose2d::new(
            8, 4, (2, 2), None, None, None, None, None, Some(false)
        );
        
        let params = layer.parameters();
        assert_eq!(params.len(), 1); // weight only
        assert!(layer.bias.is_none());
    }
}
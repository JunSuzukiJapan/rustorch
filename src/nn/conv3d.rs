//! 3D convolution layer implementation
//! 3次元畳み込みレイヤーの実装

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Tensor;
use num_traits::Float;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::fmt::Debug;

/// 3D Convolution layer for volumetric data
/// ボリュメトリックデータ用の3D畳み込み層
///
/// This layer applies 3D convolution over an input signal composed of several input planes.
/// 複数の入力プレーンからなる入力信号に対して3次元畳み込みを適用します。
///
/// Input shape: (batch_size, in_channels, depth, height, width)
/// Output shape: (batch_size, out_channels, out_depth, out_height, out_width)
#[derive(Debug)]
pub struct Conv3d<T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive> {
    /// Weight tensor of shape (out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w)
    /// 重みテンソル (出力チャンネル, 入力チャンネル/グループ, カーネル深度, カーネル高さ, カーネル幅)
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

    /// Kernel size (depth, height, width)
    /// カーネルサイズ (深度, 高さ, 幅)
    kernel_size: (usize, usize, usize),

    /// Stride (depth, height, width)
    /// ストライド (深度, 高さ, 幅)
    stride: (usize, usize, usize),

    /// Padding (depth, height, width)
    /// パディング (深度, 高さ, 幅)
    padding: (usize, usize, usize),

    /// Dilation (depth, height, width)
    /// 膨張 (深度, 高さ, 幅)
    dilation: (usize, usize, usize),

    /// Groups for grouped convolution
    /// グループ畳み込み用のグループ数
    groups: usize,
}

impl<T> Conv3d<T>
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
    /// Create a new Conv3d layer
    /// 新しいConv3d層を作成
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: Option<(usize, usize, usize)>,
        padding: Option<(usize, usize, usize)>,
        dilation: Option<(usize, usize, usize)>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> Self {
        let stride = stride.unwrap_or((1, 1, 1));
        let padding = padding.unwrap_or((0, 0, 0));
        let dilation = dilation.unwrap_or((1, 1, 1));
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        // Validate parameters
        assert!(
            in_channels % groups == 0,
            "in_channels must be divisible by groups"
        );
        assert!(
            out_channels % groups == 0,
            "out_channels must be divisible by groups"
        );
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0 && kernel_size.2 > 0,
            "kernel_size must be positive"
        );
        assert!(
            stride.0 > 0 && stride.1 > 0 && stride.2 > 0,
            "stride must be positive"
        );
        assert!(
            dilation.0 > 0 && dilation.1 > 0 && dilation.2 > 0,
            "dilation must be positive"
        );
        assert!(groups > 0, "groups must be positive");

        // Initialize weight tensor with shape [out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w]
        let weight_shape = vec![
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
            kernel_size.2,
        ];
        let weight_size = weight_shape.iter().product::<usize>();

        // Initialize with Kaiming uniform (suitable for ReLU)
        let fan_in = (in_channels / groups) * kernel_size.0 * kernel_size.1 * kernel_size.2;
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
        // In a real implementation, this would perform 3D convolution
        // TODO: Implement actual 3D convolution computation
        Variable::new(input.data().read().unwrap().clone(), input.requires_grad())
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

    /// Calculate output size for 3D convolution
    /// 3D畳み込みの出力サイズを計算
    pub fn calculate_output_size(
        &self,
        input_size: (usize, usize, usize),
    ) -> (usize, usize, usize) {
        let (input_d, input_h, input_w) = input_size;
        let (kernel_d, kernel_h, kernel_w) = self.kernel_size;
        let (stride_d, stride_h, stride_w) = self.stride;
        let (pad_d, pad_h, pad_w) = self.padding;
        let (dil_d, dil_h, dil_w) = self.dilation;

        let out_d = (input_d + 2 * pad_d - dil_d * (kernel_d - 1) - 1) / stride_d + 1;
        let out_h = (input_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
        let out_w = (input_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

        (out_d, out_h, out_w)
    }

    /// Create a Conv3d for video processing (common parameters)
    /// ビデオ処理用のConv3d作成（一般的なパラメータ）
    pub fn for_video_processing(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Self {
        Self::new(
            in_channels,
            out_channels,
            kernel_size,
            Some((1, 1, 1)), // stride
            Some((0, 0, 0)), // padding
            None,            // dilation
            None,            // groups
            Some(true),      // bias
        )
    }

    /// Create a Conv3d with temporal kernel (3x1x1) for temporal modeling
    /// 時系列モデリング用の時間的カーネル（3x1x1）のConv3d作成
    pub fn temporal_conv(
        in_channels: usize,
        out_channels: usize,
        temporal_kernel_size: usize,
    ) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (temporal_kernel_size, 1, 1),
            Some((1, 1, 1)),                        // stride
            Some((temporal_kernel_size / 2, 0, 0)), // padding to maintain spatial size
            None,                                   // dilation
            None,                                   // groups
            Some(true),                             // bias
        )
    }

    /// Create a Conv3d with spatial kernel (1x3x3) for spatial modeling
    /// 空間モデリング用の空間的カーネル（1x3x3）のConv3d作成
    pub fn spatial_conv(
        in_channels: usize,
        out_channels: usize,
        spatial_kernel_size: usize,
    ) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (1, spatial_kernel_size, spatial_kernel_size),
            Some((1, 1, 1)),                                             // stride
            Some((0, spatial_kernel_size / 2, spatial_kernel_size / 2)), // padding
            None,                                                        // dilation
            None,                                                        // groups
            Some(true),                                                  // bias
        )
    }

    /// Get number of parameters
    /// パラメータ数を取得
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.out_channels
            * (self.in_channels / self.groups)
            * self.kernel_size.0
            * self.kernel_size.1
            * self.kernel_size.2;
        let bias_params = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        weight_params + bias_params
    }

    /// Get receptive field size
    /// 受容野サイズを取得
    pub fn receptive_field(&self) -> (usize, usize, usize) {
        let field_d = self.dilation.0 * (self.kernel_size.0 - 1) + 1;
        let field_h = self.dilation.1 * (self.kernel_size.1 - 1) + 1;
        let field_w = self.dilation.2 * (self.kernel_size.2 - 1) + 1;
        (field_d, field_h, field_w)
    }
}

impl<T> Module<T> for Conv3d<T>
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
    fn test_conv3d_creation() {
        let layer: Conv3d<f32> = Conv3d::new(
            64,              // in_channels
            32,              // out_channels
            (3, 3, 3),       // kernel_size
            Some((1, 1, 1)), // stride
            Some((1, 1, 1)), // padding
            None,            // dilation
            None,            // groups
            Some(true),      // bias
        );

        assert_eq!(layer.in_channels, 64);
        assert_eq!(layer.out_channels, 32);
        assert_eq!(layer.kernel_size, (3, 3, 3));
        assert_eq!(layer.stride, (1, 1, 1));
        assert_eq!(layer.padding, (1, 1, 1));
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_output_size_calculation() {
        let layer: Conv3d<f32> = Conv3d::new(
            16,
            32,
            (3, 3, 3),
            Some((1, 1, 1)),
            Some((1, 1, 1)),
            None,
            None,
            Some(true),
        );

        let input_size = (16, 32, 32);
        let output_size = layer.calculate_output_size(input_size);

        // With padding=1, stride=1, kernel=3, output should equal input
        assert_eq!(output_size, (16, 32, 32));
    }

    #[test]
    fn test_for_video_processing() {
        let layer: Conv3d<f32> = Conv3d::for_video_processing(3, 64, (3, 7, 7));

        assert_eq!(layer.in_channels, 3);
        assert_eq!(layer.out_channels, 64);
        assert_eq!(layer.kernel_size, (3, 7, 7));
        assert_eq!(layer.stride, (1, 1, 1));
        assert_eq!(layer.padding, (0, 0, 0));
    }

    #[test]
    fn test_temporal_conv() {
        let layer: Conv3d<f32> = Conv3d::temporal_conv(64, 128, 3);

        assert_eq!(layer.kernel_size, (3, 1, 1));
        assert_eq!(layer.padding, (1, 0, 0)); // temporal padding only
    }

    #[test]
    fn test_spatial_conv() {
        let layer: Conv3d<f32> = Conv3d::spatial_conv(64, 128, 3);

        assert_eq!(layer.kernel_size, (1, 3, 3));
        assert_eq!(layer.padding, (0, 1, 1)); // spatial padding only
    }

    #[test]
    fn test_num_parameters() {
        let layer: Conv3d<f32> = Conv3d::new(32, 64, (3, 3, 3), None, None, None, None, Some(true));

        // Weight: 64 * 32 * 3 * 3 * 3 = 55296
        // Bias: 64
        // Total: 55360
        assert_eq!(layer.num_parameters(), 55360);
    }

    #[test]
    fn test_receptive_field() {
        let layer: Conv3d<f32> =
            Conv3d::new(16, 32, (3, 5, 5), None, None, Some((2, 1, 1)), None, None);

        // Receptive field: (2*(3-1)+1, 1*(5-1)+1, 1*(5-1)+1) = (5, 5, 5)
        assert_eq!(layer.receptive_field(), (5, 5, 5));
    }

    #[test]
    fn test_parameters() {
        let layer: Conv3d<f32> = Conv3d::new(8, 16, (2, 2, 2), None, None, None, None, Some(true));

        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_no_bias() {
        let layer: Conv3d<f32> = Conv3d::new(8, 16, (2, 2, 2), None, None, None, None, Some(false));

        let params = layer.parameters();
        assert_eq!(params.len(), 1); // weight only
        assert!(layer.bias.is_none());
    }

    #[test]
    fn test_grouped_convolution() {
        let layer: Conv3d<f32> =
            Conv3d::new(32, 64, (2, 2, 2), None, None, None, Some(4), Some(true));

        assert_eq!(layer.groups, 4);
        // Weight shape should be [64, 32/4, 2, 2, 2] = [64, 8, 2, 2, 2]
        assert_eq!(layer.num_parameters(), 64 * 8 * 2 * 2 * 2 + 64); // weight + bias
    }

    #[test]
    fn test_output_size_with_stride() {
        let layer: Conv3d<f32> = Conv3d::new(
            16,
            32,
            (3, 3, 3),
            Some((2, 2, 2)),
            Some((0, 0, 0)),
            None,
            None,
            None,
        );

        let input_size = (16, 16, 16);
        let output_size = layer.calculate_output_size(input_size);

        // Expected: (16 + 0 - 1*(3-1) - 1) / 2 + 1 = (16 - 2 - 1) / 2 + 1 = 13/2 + 1 = 6 + 1 = 7
        assert_eq!(output_size, (7, 7, 7));
    }
}

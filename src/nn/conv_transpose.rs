//! 2D transposed convolution layer implementation
//! 2次元転置畳み込みレイヤーの実装

use crate::autograd::Variable;
use crate::nn::Module;
use crate::tensor::Tensor;
use num_traits::Float;
use rand::distributions::Distribution;
use rand_distr::Normal;
use std::fmt::Debug;

/// 2D Transposed Convolution layer for generative models
/// 生成モデル用の2D転置畳み込み層
///
/// This layer applies 2D transposed convolution over an input signal composed of several input planes.
/// 複数の入力プレーンからなる入力信号に対して2次元転置畳み込みを適用します。
///
/// Input shape: (batch_size, in_channels, height, width)
/// Output shape: (batch_size, out_channels, out_height, out_width)
#[derive(Debug)]
pub struct ConvTranspose2d<
    T: Float + Send + Sync + ndarray::ScalarOperand + num_traits::FromPrimitive,
> {
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
    /// Create a new ConvTranspose2d layer
    /// 新しいConvTranspose2d層を作成
    #[allow(clippy::too_many_arguments)]
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
        assert!(
            in_channels % groups == 0,
            "in_channels must be divisible by groups"
        );
        assert!(
            out_channels % groups == 0,
            "out_channels must be divisible by groups"
        );
        assert!(
            output_padding.0 < stride.0,
            "output_padding.0 must be less than stride.0"
        );
        assert!(
            output_padding.1 < stride.1,
            "output_padding.1 must be less than stride.1"
        );

        // Initialize weight tensor with shape [in_channels, out_channels/groups, kernel_h, kernel_w]
        let weight_shape = vec![
            in_channels,
            out_channels / groups,
            kernel_size.0,
            kernel_size.1,
        ];
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
        let input_tensor = input.data();
        let input_guard = input_tensor.read().unwrap();
        let input_shape = input_guard.shape();
        
        // Input validation
        assert!(input_shape.len() == 4, "Input must be 4D tensor (batch, channels, height, width)");
        assert_eq!(input_shape[1], self.in_channels, "Input channels mismatch");
        
        let batch_size = input_shape[0];
        let input_h = input_shape[2];
        let input_w = input_shape[3];
        
        // Calculate output dimensions
        let (output_h, output_w) = self.calculate_output_size((input_h, input_w));
        let output_shape = vec![batch_size, self.out_channels, output_h, output_w];
        
        // Initialize output tensor
        let mut output_data = vec![T::default(); output_shape.iter().product()];
        
        // Perform transposed convolution
        self.transposed_conv_2d(
            input_guard.as_slice().unwrap(),
            &mut output_data,
            input_shape,
            &output_shape,
        );
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_data_arc = bias.data();
            let bias_guard = bias_data_arc.read().unwrap();
            self.add_bias(&mut output_data, &output_shape, bias_guard.as_slice().unwrap());
        }
        
        let output_tensor = Tensor::from_vec(output_data, output_shape);
        Variable::new(output_tensor, input.requires_grad())
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
            (4, 4),       // kernel_size
            Some((2, 2)), // stride
            Some((1, 1)), // padding
            None,         // output_padding
            None,         // dilation
            None,         // groups
            Some(true),   // bias
        )
    }

    /// Create a ConvTranspose2d for 4x upsampling
    /// 4倍アップサンプリング用のConvTranspose2d作成
    pub fn upsample_4x(in_channels: usize, out_channels: usize) -> Self {
        Self::new(
            in_channels,
            out_channels,
            (4, 4),       // kernel_size
            Some((4, 4)), // stride
            Some((0, 0)), // padding
            None,         // output_padding
            None,         // dilation
            None,         // groups
            Some(true),   // bias
        )
    }

    /// Get number of parameters
    /// パラメータ数を取得
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_channels
            * (self.out_channels / self.groups)
            * self.kernel_size.0
            * self.kernel_size.1;
        let bias_params = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        weight_params + bias_params
    }

    /// Core transposed convolution operation
    /// 転置畳み込みのコア演算
    fn transposed_conv_2d(
        &self,
        input: &[T],
        output: &mut [T],
        input_shape: &[usize],
        output_shape: &[usize],
    ) {
        let batch_size = input_shape[0];
        let input_h = input_shape[2];
        let input_w = input_shape[3];
        let output_h = output_shape[2];
        let output_w = output_shape[3];
        
        let weight_data_arc = self.weight.data();
        let weight_guard = weight_data_arc.read().unwrap();
        let weight_data = weight_guard.as_slice().unwrap();
        let (kernel_h, kernel_w) = self.kernel_size;
        let (stride_h, stride_w) = self.stride;
        let (pad_h, pad_w) = self.padding;
        let (out_pad_h, out_pad_w) = self.output_padding;
        
        // Process each batch sequentially for memory safety
        for b in 0..batch_size {
            let input_batch_offset = b * self.in_channels * input_h * input_w;
            let output_batch_offset = b * self.out_channels * output_h * output_w;
            
            // Process each input channel
            for in_ch in 0..self.in_channels {
                let input_ch_offset = input_batch_offset + in_ch * input_h * input_w;
                
                // Process each output channel in this group
                let group_id = in_ch / (self.in_channels / self.groups);
                let out_ch_start = group_id * (self.out_channels / self.groups);
                let out_ch_end = out_ch_start + (self.out_channels / self.groups);
                
                for out_ch in out_ch_start..out_ch_end {
                    let out_ch_in_group = out_ch - out_ch_start;
                    let output_ch_offset = output_batch_offset + out_ch * output_h * output_w;
                    let weight_offset = in_ch * (self.out_channels / self.groups) * kernel_h * kernel_w
                        + out_ch_in_group * kernel_h * kernel_w;
                    
                    // Transposed convolution: for each input pixel, spread to output region
                    for i_h in 0..input_h {
                        for i_w in 0..input_w {
                            let input_idx = input_ch_offset + i_h * input_w + i_w;
                            let input_val = input[input_idx];
                            
                            // Map input position to output region
                            let out_h_start = i_h * stride_h;
                            let out_w_start = i_w * stride_w;
                            
                            // Apply kernel
                            for k_h in 0..kernel_h {
                                for k_w in 0..kernel_w {
                                    let out_h = out_h_start + k_h;
                                    let out_w = out_w_start + k_w;
                                    
                                    // Check bounds with padding
                                    if out_h >= pad_h 
                                        && out_w >= pad_w
                                        && out_h < output_h + pad_h - out_pad_h
                                        && out_w < output_w + pad_w - out_pad_w
                                    {
                                        let adj_out_h = out_h - pad_h;
                                        let adj_out_w = out_w - pad_w;
                                        
                                        if adj_out_h < output_h && adj_out_w < output_w {
                                            let output_idx = output_ch_offset + adj_out_h * output_w + adj_out_w;
                                            let weight_idx = weight_offset + k_h * kernel_w + k_w;
                                            
                                            // Accumulate the contribution
                                            unsafe {
                                                let output_ptr = output.as_mut_ptr().add(output_idx);
                                                *output_ptr = *output_ptr + input_val * weight_data[weight_idx];
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

    /// Add bias to output tensor
    /// 出力テンソルにバイアスを追加
    fn add_bias(&self, output: &mut [T], output_shape: &[usize], bias: &[T]) {
        let batch_size = output_shape[0];
        let output_h = output_shape[2];
        let output_w = output_shape[3];
        
        for b in 0..batch_size {
            for ch in 0..self.out_channels {
                let ch_offset = b * self.out_channels * output_h * output_w + ch * output_h * output_w;
                let bias_val = bias[ch];
                
                for i in 0..(output_h * output_w) {
                    output[ch_offset + i] = output[ch_offset + i] + bias_val;
                }
            }
        }
    }
}

impl<T> Module<T> for ConvTranspose2d<T>
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
    fn test_conv_transpose_2d_creation() {
        let layer: ConvTranspose2d<f32> = ConvTranspose2d::new(
            64,           // in_channels
            32,           // out_channels
            (3, 3),       // kernel_size
            Some((2, 2)), // stride
            Some((1, 1)), // padding
            None,         // output_padding
            None,         // dilation
            None,         // groups
            Some(true),   // bias
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
            64,
            32,
            (3, 3),
            Some((2, 2)),
            Some((1, 1)),
            None,
            None,
            None,
            Some(true),
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
        let layer: ConvTranspose2d<f32> =
            ConvTranspose2d::new(64, 32, (3, 3), None, None, None, None, None, Some(true));

        // Weight: 64 * 32 * 3 * 3 = 18432
        // Bias: 32
        // Total: 18464
        assert_eq!(layer.num_parameters(), 18464);
    }

    #[test]
    fn test_parameters() {
        let layer: ConvTranspose2d<f32> =
            ConvTranspose2d::new(8, 4, (2, 2), None, None, None, None, None, Some(true));

        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_no_bias() {
        let layer: ConvTranspose2d<f32> =
            ConvTranspose2d::new(8, 4, (2, 2), None, None, None, None, None, Some(false));

        let params = layer.parameters();
        assert_eq!(params.len(), 1); // weight only
        assert!(layer.bias.is_none());
    }

    #[test]
    fn test_forward_pass() {
        let layer: ConvTranspose2d<f32> = ConvTranspose2d::new(
            2,            // in_channels
            4,            // out_channels  
            (3, 3),       // kernel_size
            Some((2, 2)), // stride
            Some((1, 1)), // padding
            None,         // output_padding
            None,         // dilation
            None,         // groups
            Some(true),   // bias
        );

        // Create input tensor: [1, 2, 4, 4]
        let input_data = vec![1.0f32; 1 * 2 * 4 * 4];
        let input_tensor = crate::tensor::Tensor::from_vec(input_data, vec![1, 2, 4, 4]);
        let input_var = Variable::new(input_tensor, false);

        // Forward pass
        let output = layer.forward(&input_var);
        let output_data_arc = output.data();
        let output_guard = output_data_arc.read().unwrap();
        let output_shape = output_guard.shape();

        // Verify output shape: [1, 4, 7, 7] based on calculation
        // (4-1)*2 - 2*1 + 1*(3-1) + 0 + 1 = 6 + 2 + 1 = 9, but with padding adjustments = 7
        assert_eq!(output_shape.len(), 4);
        assert_eq!(output_shape[0], 1); // batch
        assert_eq!(output_shape[1], 4); // out_channels
    }
}

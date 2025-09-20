//! CoreML convolution operations
//! CoreML畳み込み演算

use super::*;
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};

/// CoreML Convolution operations trait
/// CoreML畳み込み演算トレイト
pub trait CoreMLConvolution<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    /// CoreML 2D convolution
    /// CoreML 2D畳み込み
    fn coreml_conv2d(
        &self,
        kernel: &Self,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>>;

    /// CoreML batch convolution
    /// CoreMLバッチ畳み込み
    fn coreml_batch_conv2d(
        &self,
        kernel: &Self,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>>;

    /// CoreML transposed convolution
    /// CoreML転置畳み込み
    fn coreml_conv2d_transpose(
        &self,
        kernel: &Self,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>>;

    /// CoreML depthwise convolution
    /// CoreML深度別畳み込み
    fn coreml_depthwise_conv2d(
        &self,
        kernel: &Self,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>>;
}

/// Convolution operation for CoreML
/// CoreML用畳み込み演算
pub struct ConvolutionOperation<T> {
    input: Tensor<T>,
    kernel: Tensor<T>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    conv_type: ConvolutionType,
}

/// Types of convolution operations
/// 畳み込み演算のタイプ
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolutionType {
    /// Standard 2D convolution
    /// 標準2D畳み込み
    Conv2D,

    /// Transposed convolution (deconvolution)
    /// 転置畳み込み（逆畳み込み）
    TransposedConv2D,

    /// Depthwise separable convolution
    /// 深度別分離畳み込み
    DepthwiseConv2D,

    /// Grouped convolution
    /// グループ畳み込み
    GroupedConv2D,
}

impl<T> ConvolutionOperation<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    /// Create new convolution operation
    /// 新しい畳み込み演算を作成
    pub fn new(
        input: Tensor<T>,
        kernel: Tensor<T>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        conv_type: ConvolutionType,
    ) -> Self {
        Self {
            input,
            kernel,
            stride,
            padding,
            conv_type,
        }
    }

    /// Validate convolution parameters
    /// 畳み込みパラメータを検証
    fn validate_parameters(&self) -> CoreMLResult<()> {
        let input_shape = self.input.shape();
        let kernel_shape = self.kernel.shape();

        // Check input dimensions (should be 4D: [N, C, H, W])
        if input_shape.len() != 4 {
            return Err(error_helpers::unsupported_operation(
                "Convolution input must be 4D tensor [N, C, H, W]"
            ));
        }

        // Check kernel dimensions (should be 4D: [out_channels, in_channels, kH, kW])
        if kernel_shape.len() != 4 {
            return Err(error_helpers::unsupported_operation(
                "Convolution kernel must be 4D tensor [out_channels, in_channels, kH, kW]"
            ));
        }

        // Validate channel consistency
        let input_channels = input_shape[1];
        let kernel_input_channels = kernel_shape[1];

        match self.conv_type {
            ConvolutionType::Conv2D => {
                if input_channels != kernel_input_channels {
                    return Err(error_helpers::tensor_op_error(&format!(
                        "Channel mismatch: input has {} channels, kernel expects {}",
                        input_channels, kernel_input_channels
                    )));
                }
            }
            ConvolutionType::DepthwiseConv2D => {
                // For depthwise, kernel input channels should be 1
                if kernel_input_channels != 1 {
                    return Err(error_helpers::unsupported_operation(
                        "Depthwise convolution kernel should have 1 input channel"
                    ));
                }
            }
            _ => {
                // Other types have different validation rules
                // TODO: Implement specific validation for each type
            }
        }

        // Validate stride and padding
        if self.stride.len() != 2 {
            return Err(error_helpers::unsupported_operation(
                "Stride must have 2 elements [height, width]"
            ));
        }

        if self.padding.len() != 2 && self.padding.len() != 4 {
            return Err(error_helpers::unsupported_operation(
                "Padding must have 2 elements [height, width] or 4 elements [top, bottom, left, right]"
            ));
        }

        Ok(())
    }

    /// Check if convolution is efficient on CoreML
    /// 畳み込みがCoreMLで効率的かチェック
    fn is_efficient_on_coreml(&self) -> bool {
        let input_shape = self.input.shape();
        let kernel_shape = self.kernel.shape();

        // Extract dimensions
        let batch_size = input_shape[0];
        let input_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        let output_channels = kernel_shape[0];

        // Calculate spatial size and total parameters
        let spatial_size = input_height * input_width;
        let total_params = input_channels * output_channels * kernel_shape[2] * kernel_shape[3];

        // CoreML is efficient for:
        // 1. Reasonable spatial sizes (not too small)
        // 2. Sufficient channels for parallelization
        // 3. Not too large to cause memory issues
        let spatial_efficient = spatial_size >= 256 && spatial_size <= 1_048_576; // 16x16 to 1024x1024
        let channel_efficient = input_channels >= 4 && output_channels >= 4;
        let batch_efficient = batch_size >= 1 && batch_size <= 32; // Reasonable batch sizes
        let param_efficient = total_params >= 1024 && total_params <= 16_777_216; // 1K to 16M parameters

        spatial_efficient && channel_efficient && batch_efficient && param_efficient
    }

    /// Calculate output shape for convolution
    /// 畳み込みの出力形状を計算
    fn calculate_output_shape(&self) -> Vec<usize> {
        let input_shape = self.input.shape();
        let kernel_shape = self.kernel.shape();

        let batch_size = input_shape[0];
        let output_channels = kernel_shape[0];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        let kernel_height = kernel_shape[2];
        let kernel_width = kernel_shape[3];

        // Calculate padding (handle both 2 and 4 element padding)
        let (pad_top, pad_bottom, pad_left, pad_right) = if self.padding.len() == 2 {
            (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
        } else {
            (self.padding[0], self.padding[1], self.padding[2], self.padding[3])
        };

        // Calculate output dimensions
        let output_height = (input_height + pad_top + pad_bottom - kernel_height) / self.stride[0] + 1;
        let output_width = (input_width + pad_left + pad_right - kernel_width) / self.stride[1] + 1;

        vec![batch_size, output_channels, output_height, output_width]
    }
}

impl<T> CoreMLOperation<T> for ConvolutionOperation<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    fn execute_coreml(&self, device_id: usize) -> CoreMLResult<Tensor<T>> {
        self.validate_parameters()?;

        coreml_feature! {
            use crate::gpu::coreml_backend::CoreMLGraph;

            let graph = CoreMLGraph::new(device_id)?;

            match self.conv_type {
                ConvolutionType::Conv2D => {
                    graph.conv2d(&self.input, &self.kernel, &self.stride, &self.padding)
                }
                ConvolutionType::TransposedConv2D => {
                    // TODO: Implement transposed convolution in backend
                    Err(error_helpers::unsupported_operation(
                        "Transposed convolution not yet implemented in CoreML backend"
                    ))
                }
                ConvolutionType::DepthwiseConv2D => {
                    // TODO: Implement depthwise convolution in backend
                    Err(error_helpers::unsupported_operation(
                        "Depthwise convolution not yet implemented in CoreML backend"
                    ))
                }
                ConvolutionType::GroupedConv2D => {
                    // TODO: Implement grouped convolution in backend
                    Err(error_helpers::unsupported_operation(
                        "Grouped convolution not yet implemented in CoreML backend"
                    ))
                }
            }
        }

        #[cfg(not(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback")))]
        {
            Err(error_helpers::feature_disabled())
        }
    }

    fn is_supported_by_coreml(&self) -> bool {
        // Check if convolution type is supported and parameters are valid
        let type_supported = matches!(self.conv_type, ConvolutionType::Conv2D);
        let params_valid = self.validate_parameters().is_ok();
        let efficient = self.is_efficient_on_coreml();

        type_supported && params_valid && efficient
    }

    fn estimated_execution_time(&self) -> Option<std::time::Duration> {
        if !self.is_supported_by_coreml() {
            return None;
        }

        let input_shape = self.input.shape();
        let kernel_shape = self.kernel.shape();
        let output_shape = self.calculate_output_shape();

        // Calculate computational complexity
        let output_elements: usize = output_shape.iter().product();
        let kernel_flops = kernel_shape[2] * kernel_shape[3] * kernel_shape[1]; // kH * kW * input_channels
        let total_flops = output_elements * kernel_flops * 2; // multiply + add

        // Rough estimation: ~2ns per FLOP on CoreML for convolution
        Some(std::time::Duration::from_nanos((total_flops * 2) as u64))
    }
}

/// Implement CoreML convolution for Tensor
/// TensorにCoreML畳み込みを実装
impl<T> CoreMLConvolution<T> for Tensor<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    fn coreml_conv2d(
        &self,
        kernel: &Self,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>> {
        let operation = ConvolutionOperation::new(
            self.clone(),
            kernel.clone(),
            stride.to_vec(),
            padding.to_vec(),
            ConvolutionType::Conv2D,
        );
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_batch_conv2d(
        &self,
        kernel: &Self,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>> {
        // Batch convolution is the same as regular convolution for batched input
        self.coreml_conv2d(kernel, stride, padding)
    }

    fn coreml_conv2d_transpose(
        &self,
        kernel: &Self,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>> {
        let operation = ConvolutionOperation::new(
            self.clone(),
            kernel.clone(),
            stride.to_vec(),
            padding.to_vec(),
            ConvolutionType::TransposedConv2D,
        );
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_depthwise_conv2d(
        &self,
        kernel: &Self,
        stride: &[usize],
        padding: &[usize],
    ) -> CoreMLResult<Tensor<T>> {
        let operation = ConvolutionOperation::new(
            self.clone(),
            kernel.clone(),
            stride.to_vec(),
            padding.to_vec(),
            ConvolutionType::DepthwiseConv2D,
        );
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_parameter_validation() {
        let input = Tensor::<f32>::zeros(&[1, 3, 32, 32]);   // [N, C, H, W]
        let kernel = Tensor::<f32>::zeros(&[16, 3, 3, 3]);   // [out_ch, in_ch, kH, kW]
        let stride = vec![1, 1];
        let padding = vec![1, 1];

        let operation = ConvolutionOperation::new(
            input,
            kernel,
            stride,
            padding,
            ConvolutionType::Conv2D,
        );

        assert!(operation.validate_parameters().is_ok());
        assert!(operation.is_supported_by_coreml());
    }

    #[test]
    fn test_conv2d_channel_mismatch() {
        let input = Tensor::<f32>::zeros(&[1, 3, 32, 32]);   // 3 input channels
        let kernel = Tensor::<f32>::zeros(&[16, 4, 3, 3]);   // 4 input channels - mismatch!
        let stride = vec![1, 1];
        let padding = vec![1, 1];

        let operation = ConvolutionOperation::new(
            input,
            kernel,
            stride,
            padding,
            ConvolutionType::Conv2D,
        );

        assert!(operation.validate_parameters().is_err());
        assert!(!operation.is_supported_by_coreml());
    }

    #[test]
    fn test_small_convolution_not_efficient() {
        let input = Tensor::<f32>::zeros(&[1, 1, 8, 8]);     // Too few channels and small spatial
        let kernel = Tensor::<f32>::zeros(&[1, 1, 3, 3]);
        let stride = vec![1, 1];
        let padding = vec![1, 1];

        let operation = ConvolutionOperation::new(
            input,
            kernel,
            stride,
            padding,
            ConvolutionType::Conv2D,
        );

        assert!(operation.validate_parameters().is_ok());
        assert!(!operation.is_efficient_on_coreml()); // Too small/few channels
        assert!(!operation.is_supported_by_coreml());
    }

    #[test]
    fn test_output_shape_calculation() {
        let input = Tensor::<f32>::zeros(&[2, 3, 32, 32]);
        let kernel = Tensor::<f32>::zeros(&[16, 3, 3, 3]);
        let stride = vec![2, 2];
        let padding = vec![1, 1];

        let operation = ConvolutionOperation::new(
            input,
            kernel,
            stride,
            padding,
            ConvolutionType::Conv2D,
        );

        let output_shape = operation.calculate_output_shape();

        // Expected: [(32 + 2*1 - 3) / 2 + 1] = [16, 16]
        assert_eq!(output_shape, vec![2, 16, 16, 16]);
    }

    #[test]
    fn test_execution_time_estimation() {
        let input = Tensor::<f32>::zeros(&[1, 16, 64, 64]);  // Large enough for efficiency
        let kernel = Tensor::<f32>::zeros(&[32, 16, 3, 3]);
        let stride = vec![1, 1];
        let padding = vec![1, 1];

        let operation = ConvolutionOperation::new(
            input,
            kernel,
            stride,
            padding,
            ConvolutionType::Conv2D,
        );

        let estimated_time = operation.estimated_execution_time();
        assert!(estimated_time.is_some());

        let time = estimated_time.unwrap();
        assert!(time.as_nanos() > 0);
    }

    #[test]
    fn test_depthwise_validation() {
        let input = Tensor::<f32>::zeros(&[1, 8, 32, 32]);
        let kernel = Tensor::<f32>::zeros(&[8, 1, 3, 3]);    // Depthwise: input_channels = 1
        let stride = vec![1, 1];
        let padding = vec![1, 1];

        let operation = ConvolutionOperation::new(
            input,
            kernel,
            stride,
            padding,
            ConvolutionType::DepthwiseConv2D,
        );

        assert!(operation.validate_parameters().is_ok());
    }
}
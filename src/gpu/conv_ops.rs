//! GPU Convolution Operations
//! GPU畳み込み演算
//!
//! This module provides GPU-accelerated convolution operations including
//! cuDNN integration for CUDA and Metal Performance Shaders for Apple Silicon.

use crate::backends::ConvolutionParams;
use crate::error::{RusTorchError, RusTorchResult};
#[cfg(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
))]
use crate::gpu::hybrid_executor::HybridExecution;
#[cfg(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
))]
use crate::gpu::{DeviceType, OpType};
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};

/// GPU Convolution trait for convolution operations with hybrid execution
/// ハイブリッド実行による GPU 畳み込み trait
pub trait GpuConvolution<T: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static> {
    /// GPU-accelerated 2D convolution with automatic device selection
    /// 自動デバイス選択によるGPU加速2D畳み込み
    fn gpu_conv2d(&self, kernel: &Self, params: &ConvolutionParams) -> RusTorchResult<Tensor<T>>;

    /// GPU-accelerated transposed convolution (deconvolution)
    /// GPU加速転置畳み込み（逆畳み込み）
    fn gpu_conv_transpose2d(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>>;

    /// GPU-accelerated depthwise separable convolution
    /// GPU加速深度分離可能畳み込み
    fn gpu_depthwise_conv2d(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>>;

    /// GPU-accelerated grouped convolution
    /// GPU加速グループ畳み込み
    fn gpu_grouped_conv2d(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
        groups: usize,
    ) -> RusTorchResult<Tensor<T>>;

    /// GPU-accelerated 3D convolution
    /// GPU加速3D畳み込み
    fn gpu_conv3d(&self, kernel: &Self, params: &ConvolutionParams) -> RusTorchResult<Tensor<T>>;
}

#[cfg(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
))]
impl<T: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static> GpuConvolution<T>
    for Tensor<T>
{
    fn gpu_conv2d(&self, kernel: &Self, params: &ConvolutionParams) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml_ops::CoreMLConvolution, OpType};

            return self.hybrid_operation(OpType::Convolution, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML convolution implementation
                        let stride = &[params.stride[0], params.stride[1]];
                        let padding = &[params.padding[0], params.padding[1]];
                        self.coreml_conv2d(kernel, stride, padding)
                    }
                    super::DeviceType::Cuda(_) => {
                        // Use CUDA convolution implementation - fallback to CPU
                        self.conv2d_fallback(kernel, params)
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal convolution implementation - fallback to CPU
                        self.conv2d_fallback(kernel, params)
                    }
                    super::DeviceType::OpenCL(_) => {
                        // Use OpenCL convolution implementation - fallback to CPU
                        self.conv2d_fallback(kernel, params)
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback implementation
                        self.conv2d_fallback(kernel, params)
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for convolution".to_string(),
                    )),
                }
            });
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        {
            // For non-CoreML builds, use CPU fallback directly
            self.conv2d_fallback(kernel, params)
        }
    }

    fn gpu_conv_transpose2d(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml_ops::CoreMLConvolution, OpType};

            return self.hybrid_operation(OpType::Convolution, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // CoreML transpose convolution not yet implemented
                        self.conv_transpose2d_fallback(kernel, params)
                    }
                    super::DeviceType::Cuda(_) => {
                        // Use CUDA transpose convolution implementation - fallback to CPU
                        self.conv_transpose2d_fallback(kernel, params)
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal transpose convolution implementation - fallback to CPU
                        self.conv_transpose2d_fallback(kernel, params)
                    }
                    super::DeviceType::OpenCL(_) => {
                        // Use OpenCL transpose convolution implementation - fallback to CPU
                        self.conv_transpose2d_fallback(kernel, params)
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback implementation
                        self.conv_transpose2d_fallback(kernel, params)
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for transpose convolution".to_string(),
                    )),
                }
            });
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        {
            // For non-CoreML builds, use CPU fallback directly
            self.conv_transpose2d_fallback(kernel, params)
        }
    }

    fn gpu_depthwise_conv2d(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml_ops::CoreMLConvolution, OpType};

            return self.hybrid_operation(OpType::Convolution, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // CoreML depthwise convolution not yet implemented
                        self.depthwise_conv2d_fallback(kernel, params)
                    }
                    super::DeviceType::Cuda(_) => {
                        // Use CUDA depthwise convolution implementation - fallback to CPU
                        self.depthwise_conv2d_fallback(kernel, params)
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal depthwise convolution implementation - fallback to CPU
                        self.depthwise_conv2d_fallback(kernel, params)
                    }
                    super::DeviceType::OpenCL(_) => {
                        // Use OpenCL depthwise convolution implementation - fallback to CPU
                        self.depthwise_conv2d_fallback(kernel, params)
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback implementation
                        self.depthwise_conv2d_fallback(kernel, params)
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for depthwise convolution".to_string(),
                    )),
                }
            });
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        {
            // For non-CoreML builds, use CPU fallback directly
            self.depthwise_conv2d_fallback(kernel, params)
        }
    }

    fn gpu_grouped_conv2d(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
        groups: usize,
    ) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml_ops::CoreMLConvolution, OpType};

            return self.hybrid_operation(OpType::Convolution, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // CoreML grouped convolution not yet implemented
                        self.grouped_conv2d_fallback(kernel, params, groups)
                    }
                    super::DeviceType::Cuda(_) => {
                        // Use CUDA grouped convolution implementation - fallback to CPU
                        self.grouped_conv2d_fallback(kernel, params, groups)
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal grouped convolution implementation - fallback to CPU
                        self.grouped_conv2d_fallback(kernel, params, groups)
                    }
                    super::DeviceType::OpenCL(_) => {
                        // Use OpenCL grouped convolution implementation - fallback to CPU
                        self.grouped_conv2d_fallback(kernel, params, groups)
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback implementation
                        self.grouped_conv2d_fallback(kernel, params, groups)
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for grouped convolution".to_string(),
                    )),
                }
            });
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        {
            // For non-CoreML builds, use CPU fallback directly
            self.grouped_conv2d_fallback(kernel, params, groups)
        }
    }

    fn gpu_conv3d(&self, kernel: &Self, params: &ConvolutionParams) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml_ops::CoreMLConvolution, OpType};

            return self.hybrid_operation(OpType::Convolution, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // CoreML 3D convolution not yet implemented
                        self.conv3d_fallback(kernel, params)
                    }
                    super::DeviceType::Cuda(_) => {
                        // Use CUDA 3D convolution implementation - fallback to CPU
                        self.conv3d_fallback(kernel, params)
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal 3D convolution implementation - fallback to CPU
                        self.conv3d_fallback(kernel, params)
                    }
                    super::DeviceType::OpenCL(_) => {
                        // Use OpenCL 3D convolution implementation - fallback to CPU
                        self.conv3d_fallback(kernel, params)
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback implementation
                        self.conv3d_fallback(kernel, params)
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for 3D convolution".to_string(),
                    )),
                }
            });
        }

        #[cfg(not(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        )))]
        {
            // For non-CoreML builds, use CPU fallback directly
            self.conv3d_fallback(kernel, params)
        }
    }
}

// Non-CoreML implementation of GpuConvolution trait
// 非CoreML用のGpuConvolution trait実装
#[cfg(not(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
)))]
impl<T: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static> GpuConvolution<T>
    for Tensor<T>
{
    fn gpu_conv2d(&self, kernel: &Self, params: &ConvolutionParams) -> RusTorchResult<Tensor<T>> {
        self.conv2d_fallback(kernel, params)
    }

    fn gpu_conv_transpose2d(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>> {
        self.conv_transpose2d_fallback(kernel, params)
    }

    fn gpu_depthwise_conv2d(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>> {
        self.depthwise_conv2d_fallback(kernel, params)
    }

    fn gpu_grouped_conv2d(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
        groups: usize,
    ) -> RusTorchResult<Tensor<T>> {
        self.grouped_conv2d_fallback(kernel, params, groups)
    }

    fn gpu_conv3d(&self, kernel: &Self, params: &ConvolutionParams) -> RusTorchResult<Tensor<T>> {
        self.conv3d_fallback(kernel, params)
    }
}

// Fallback implementations for convolution operations
// 畳み込み演算のフォールバック実装
impl<T: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static> Tensor<T> {
    /// CPU fallback 2D convolution implementation
    /// CPU フォールバック 2D 畳み込み実装
    pub fn conv2d_fallback(
        &self,
        kernel: &Self,
        params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>> {
        // Simplified CPU convolution implementation
        // This is a basic implementation - would need optimization for production

        let input_shape = self.shape();
        let kernel_shape = kernel.shape();

        if input_shape.len() != 4 || kernel_shape.len() != 4 {
            return Err(RusTorchError::TensorOp {
                message: "Conv2D requires 4D tensors (NCHW format)".to_string(),
                source: None,
            });
        }

        let [batch_size, in_channels, input_height, input_width] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];
        let [out_channels, kernel_in_channels, kernel_height, kernel_width] = [
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        ];

        if in_channels != kernel_in_channels {
            return Err(RusTorchError::TensorOp {
                message: format!(
                    "Input channels ({}) don't match kernel channels ({})",
                    in_channels, kernel_in_channels
                ),
                source: None,
            });
        }

        let stride_h = params.stride[0];
        let stride_w = params.stride[1];
        let pad_h = params.padding[0];
        let pad_w = params.padding[1];

        let output_height = (input_height + 2 * pad_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * pad_w - kernel_width) / stride_w + 1;

        let output_shape = vec![batch_size, out_channels, output_height, output_width];
        let mut output_data = vec![T::zero(); output_shape.iter().product()];

        // Simple nested loop convolution (very basic implementation)
        for n in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = T::zero();
                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * stride_h + kh;
                                    let iw = ow * stride_w + kw;

                                    if ih >= pad_h && iw >= pad_w {
                                        let ih = ih - pad_h;
                                        let iw = iw - pad_w;

                                        if ih < input_height && iw < input_width {
                                            let input_idx =
                                                n * in_channels * input_height * input_width
                                                    + ic * input_height * input_width
                                                    + ih * input_width
                                                    + iw;
                                            let kernel_idx = oc
                                                * kernel_in_channels
                                                * kernel_height
                                                * kernel_width
                                                + ic * kernel_height * kernel_width
                                                + kh * kernel_width
                                                + kw;

                                            sum = sum
                                                + self.data.as_slice().unwrap()[input_idx]
                                                    * kernel.data.as_slice().unwrap()[kernel_idx];
                                        }
                                    }
                                }
                            }
                        }

                        let output_idx = n * out_channels * output_height * output_width
                            + oc * output_height * output_width
                            + oh * output_width
                            + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }

        Ok(Tensor::from_vec(output_data, output_shape))
    }

    /// CPU fallback transpose convolution implementation
    /// CPU フォールバック転置畳み込み実装
    pub fn conv_transpose2d_fallback(
        &self,
        _kernel: &Self,
        _params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::TensorOp {
            message: "Transpose convolution fallback not yet implemented".to_string(),
            source: None,
        })
    }

    /// CPU fallback depthwise convolution implementation
    /// CPU フォールバック深度分離畳み込み実装
    pub fn depthwise_conv2d_fallback(
        &self,
        _kernel: &Self,
        _params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::TensorOp {
            message: "Depthwise convolution fallback not yet implemented".to_string(),
            source: None,
        })
    }

    /// CPU fallback grouped convolution implementation
    /// CPU フォールバック グループ畳み込み実装
    pub fn grouped_conv2d_fallback(
        &self,
        _kernel: &Self,
        _params: &ConvolutionParams,
        _groups: usize,
    ) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::TensorOp {
            message: "Grouped convolution fallback not yet implemented".to_string(),
            source: None,
        })
    }

    /// CPU fallback 3D convolution implementation
    /// CPU フォールバック 3D 畳み込み実装
    pub fn conv3d_fallback(
        &self,
        _kernel: &Self,
        _params: &ConvolutionParams,
    ) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::TensorOp {
            message: "3D convolution fallback not yet implemented".to_string(),
            source: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::ConvolutionParams;

    #[test]
    fn test_conv2d_fallback_basic() {
        // Create a simple 2x2 input with 1 channel
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::<f32>::from_vec(input_data, vec![1, 1, 2, 2]);

        // Create a 2x2 kernel
        let kernel_data = vec![1.0, 0.0, 0.0, 1.0];
        let kernel = Tensor::<f32>::from_vec(kernel_data, vec![1, 1, 2, 2]);

        let params = ConvolutionParams {
            kernel_size: vec![2, 2],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        };

        let result = input.conv2d_fallback(&kernel, &params).unwrap();
        let result_shape = result.shape();

        // Output should be 1x1x1x1 with value 5.0 (1*1 + 4*1)
        assert_eq!(result_shape, &[1, 1, 1, 1]);
        assert_eq!(result.as_slice().unwrap()[0], 5.0);
    }

    #[test]
    fn test_gpu_conv2d_fallback() {
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::<f32>::from_vec(input_data, vec![1, 1, 2, 2]);

        let kernel_data = vec![1.0, 0.0, 0.0, 1.0];
        let kernel = Tensor::<f32>::from_vec(kernel_data, vec![1, 1, 2, 2]);

        let params = ConvolutionParams {
            kernel_size: vec![2, 2],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        };

        // Should fallback to CPU implementation
        let result = input.gpu_conv2d(&kernel, &params).unwrap();
        let result_shape = result.shape();

        assert_eq!(result_shape, &[1, 1, 1, 1]);
        assert_eq!(result.as_slice().unwrap()[0], 5.0);
    }
}

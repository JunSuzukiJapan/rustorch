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
            use crate::gpu::{coreml::CoreMLConvolution, OpType};

            return self.hybrid_operation(OpType::Convolution, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML convolution implementation
                        let stride = &[params.stride[0], params.stride[1]];
                        let padding = &[params.padding[0], params.padding[1]];
                        self.coreml_conv2d(kernel, stride, padding)
                    }
                    super::DeviceType::Cuda(_) => {
                        // Use CUDA convolution implementation
                        self.conv2d_cuda(kernel, params)
                    }
                    super::DeviceType::Metal(_) => {
                        // Use actual Metal GPU convolution implementation
                        self.conv2d_metal(kernel, params)
                    }
                    super::DeviceType::OpenCL(_) => {
                        // OpenCL convolution not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "OpenCL convolution not yet implemented".to_string()
                        ))
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
            use crate::gpu::{coreml::CoreMLConvolution, OpType};

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
                        // Metal transpose convolution not yet implemented
                        Err(RusTorchError::UnsupportedOperation(
                            "Metal transpose convolution not yet implemented".to_string(),
                        ))
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
            use crate::gpu::{coreml::CoreMLConvolution, OpType};

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
                        // Metal depthwise convolution not yet implemented
                        Err(RusTorchError::UnsupportedOperation(
                            "Metal depthwise convolution not yet implemented".to_string(),
                        ))
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
            use crate::gpu::{coreml::CoreMLConvolution, OpType};

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
                        // Metal grouped convolution not yet implemented
                        Err(RusTorchError::UnsupportedOperation(
                            "Metal grouped convolution not yet implemented".to_string(),
                        ))
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
            use crate::gpu::{coreml::CoreMLConvolution, OpType};

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
                        // Metal 3D convolution not yet implemented
                        Err(RusTorchError::UnsupportedOperation(
                            "Metal 3D convolution not yet implemented".to_string(),
                        ))
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
    fn conv2d_fallback(&self, kernel: &Self, params: &ConvolutionParams) -> RusTorchResult<Self> {
        // Try Metal implementation first if available
        #[cfg(feature = "metal")]
        {
            return self.conv2d_metal(kernel, params);
        }

        #[cfg(not(feature = "metal"))]
        {
            // CPU fallback implementation
            let output_height = (self.shape()[2] + 2 * params.padding[0] - kernel.shape()[2]) / params.stride[0] + 1;
            let output_width = (self.shape()[3] + 2 * params.padding[1] - kernel.shape()[3]) / params.stride[1] + 1;

            // Simple placeholder implementation
            let output_size = self.shape()[0] * kernel.shape()[0] * output_height * output_width;
            let output_data = vec![T::zero(); output_size];

            // Create output tensor shape: [batch, output_channels, output_height, output_width]
            let output_shape = vec![self.shape()[0], kernel.shape()[0], output_height, output_width];

            Ok(Tensor::from_vec(output_data, output_shape))
        }
    }

    /// Metal GPU convolution implementation
    /// Metal GPU 畳み込み実装
    #[cfg(feature = "metal")]
    fn conv2d_metal(&self, kernel: &Self, params: &ConvolutionParams) -> RusTorchResult<Self> {
        use crate::gpu::metal_kernels::metal_conv2d_f32;
        
        // Convert tensors to f32 for Metal kernel
        let input_data = self.data.iter().map(|&x| x.to_f32().unwrap()).collect::<Vec<f32>>();
        let kernel_data = kernel.data.iter().map(|&x| x.to_f32().unwrap()).collect::<Vec<f32>>();
        
        // Get tensor dimensions
        let input_shape = self.data.shape();
        let kernel_shape = kernel.data.shape();
        
        if input_shape.len() != 4 || kernel_shape.len() != 4 {
            return Err(RusTorchError::InvalidOperation {
                operation: "conv2d_metal".to_string(),
                message: "Input and kernel must be 4D tensors [N, C, H, W]".to_string(),
            });
        }
        
        let batch_size = input_shape[0];
        let input_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        
        let output_channels = kernel_shape[0];
        let kernel_height = kernel_shape[2];
        let kernel_width = kernel_shape[3];
        
        // Calculate output dimensions
        let output_height = (input_height + 2 * params.padding[0] - kernel_height) / params.stride[0] + 1;
        let output_width = (input_width + 2 * params.padding[1] - kernel_width) / params.stride[1] + 1;
        
        let output_size = batch_size * output_channels * output_height * output_width;
        let mut output_data = vec![0.0f32; output_size];
        
        // Process each batch
        for batch in 0..batch_size {
            let input_batch_start = batch * input_channels * input_height * input_width;
            let input_batch_end = input_batch_start + input_channels * input_height * input_width;
            let input_batch = &input_data[input_batch_start..input_batch_end];
            
            let output_batch_start = batch * output_channels * output_height * output_width;
            let output_batch_end = output_batch_start + output_channels * output_height * output_width;
            let output_batch = &mut output_data[output_batch_start..output_batch_end];
            
            // Call Metal convolution
            metal_conv2d_f32(
                input_batch,
                &kernel_data,
                output_batch,
                input_height,
                input_width,
                input_channels,
                output_channels,
                kernel_height,
                kernel_width,
                params.stride[0],
                params.stride[1],
                params.padding[0],
                params.padding[1],
            ).map_err(|e| RusTorchError::InvalidOperation {
                operation: "conv2d_metal".to_string(),
                message: format!("Metal convolution failed: {}", e),
            })?;
        }
        
        // Convert result back to tensor
        let result_data: Vec<T> = output_data.into_iter()
            .map(|x| T::from_f32(x).unwrap())
            .collect();
        
        let output_shape = vec![batch_size, output_channels, output_height, output_width];
        
        Ok(Tensor::from_vec(result_data, output_shape))
    }
    
    #[cfg(not(feature = "metal"))]
    fn conv2d_metal(&self, _kernel: &Self, _params: &ConvolutionParams) -> RusTorchResult<Self> {
        Err(RusTorchError::UnsupportedDevice(
            "Metal not available".to_string(),
        ))
    }

    /// CUDA GPU convolution implementation
    /// CUDA GPU畳み込み実装
    #[cfg(feature = "cuda")]
    fn conv2d_cuda(&self, kernel: &Self, params: &ConvolutionParams) -> RusTorchResult<Self> {
        use crate::gpu::cuda_kernels::cuda_conv2d_f32;

        // Convert tensors to f32 for CUDA kernel
        let input_data = self.data.iter().map(|&x| x.to_f32().unwrap()).collect::<Vec<f32>>();
        let kernel_data = kernel.data.iter().map(|&x| x.to_f32().unwrap()).collect::<Vec<f32>>();

        // Get tensor dimensions
        let input_shape = self.data.shape();
        let kernel_shape = kernel.data.shape();

        if input_shape.len() != 4 || kernel_shape.len() != 4 {
            return Err(RusTorchError::InvalidOperation {
                operation: "conv2d_cuda".to_string(),
                message: "Input and kernel must be 4D tensors [N, C, H, W]".to_string(),
            });
        }

        let batch_size = input_shape[0];
        let input_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        let output_channels = kernel_shape[0];
        let kernel_height = kernel_shape[2];
        let kernel_width = kernel_shape[3];

        // Calculate output dimensions
        let output_height = (input_height + 2 * params.padding[0] - kernel_height) / params.stride[0] + 1;
        let output_width = (input_width + 2 * params.padding[1] - kernel_width) / params.stride[1] + 1;

        let output_size = batch_size * output_channels * output_height * output_width;
        let mut output_data = vec![0.0f32; output_size];

        // Process each batch
        for b in 0..batch_size {
            let batch_input_start = b * input_channels * input_height * input_width;
            let batch_input_end = batch_input_start + input_channels * input_height * input_width;
            let batch_input = &input_data[batch_input_start..batch_input_end];

            let batch_output_start = b * output_channels * output_height * output_width;
            let batch_output_end = batch_output_start + output_channels * output_height * output_width;
            let batch_output = &mut output_data[batch_output_start..batch_output_end];

            // Call CUDA convolution for this batch
            cuda_conv2d_f32(
                batch_input,
                &kernel_data,
                batch_output,
                input_height,
                input_width,
                input_channels,
                output_channels,
                kernel_height,
                kernel_width,
                params.stride[0],
                params.stride[1],
                params.padding[0],
                params.padding[1],
            ).map_err(|e| RusTorchError::InvalidOperation {
                operation: "conv2d_cuda".to_string(),
                message: format!("CUDA convolution failed: {}", e),
            })?;
        }

        // Convert result back to tensor
        let result_data: Vec<T> = output_data.into_iter()
            .map(|x| T::from_f32(x).unwrap())
            .collect();

        let output_shape = vec![batch_size, output_channels, output_height, output_width];

        Ok(Tensor::from_vec(result_data, output_shape))
    }

    #[cfg(not(feature = "cuda"))]
    fn conv2d_cuda(&self, _kernel: &Self, _params: &ConvolutionParams) -> RusTorchResult<Self> {
        Err(RusTorchError::UnsupportedDevice(
            "CUDA not available".to_string(),
        ))
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

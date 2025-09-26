/// GPU Activation Operations
/// GPU活性化演算
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

/// GPU activation operations trait
/// GPU活性化演算トレイト
pub trait GpuActivation<T: Float + FromPrimitive + ScalarOperand + 'static> {
    /// GPU ReLU activation
    /// GPU ReLU活性化
    fn gpu_relu(&self) -> RusTorchResult<Tensor<T>>;

    /// GPU Sigmoid activation
    /// GPU Sigmoid活性化
    fn gpu_sigmoid(&self) -> RusTorchResult<Tensor<T>>;

    /// GPU Tanh activation
    /// GPU Tanh活性化
    fn gpu_tanh(&self) -> RusTorchResult<Tensor<T>>;

    /// GPU Softmax activation
    /// GPU Softmax活性化
    fn gpu_softmax(&self, dim: isize) -> RusTorchResult<Tensor<T>>;

    /// GPU GELU activation
    /// GPU GELU活性化
    fn gpu_gelu(&self) -> RusTorchResult<Tensor<T>>;

    /// GPU Leaky ReLU activation
    /// GPU Leaky ReLU活性化
    fn gpu_leaky_relu(&self, negative_slope: f64) -> RusTorchResult<Tensor<T>>;

    /// GPU ELU activation
    /// GPU ELU活性化
    fn gpu_elu(&self, alpha: f64) -> RusTorchResult<Tensor<T>>;

    /// GPU Swish activation
    /// GPU Swish活性化
    fn gpu_swish(&self) -> RusTorchResult<Tensor<T>>;
}

#[cfg(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
))]
impl<T: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static> GpuActivation<T>
    for Tensor<T>
{
    fn gpu_relu(&self) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml::CoreMLActivation, OpType};

            return self.hybrid_operation(OpType::Activation, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML ReLU implementation
                        self.coreml_relu()
                    }
                    super::DeviceType::Cuda(_) => {
                        // CUDA ReLU not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "CUDA ReLU not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal GPU ReLU implementation
                        self.relu_metal()
                    }
                    super::DeviceType::OpenCL(_) => {
                        // OpenCL ReLU not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "OpenCL ReLU not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback implementation
                        self.relu_fallback()
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for activation".to_string(),
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
            self.relu_fallback()
        }
    }

    fn gpu_sigmoid(&self) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml::CoreMLActivation, OpType};

            return self.hybrid_operation(OpType::Activation, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML Sigmoid implementation
                        self.coreml_sigmoid()
                    }
                    super::DeviceType::Cuda(_) => {
                        // CUDA Sigmoid not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "CUDA Sigmoid not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal GPU Sigmoid implementation
                        self.sigmoid_metal()
                    }
                    super::DeviceType::OpenCL(_) => {
                        // OpenCL Sigmoid not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "OpenCL Sigmoid not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback
                        self.sigmoid_fallback()
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for activation".to_string(),
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
            self.sigmoid_fallback()
        }
    }

    fn gpu_tanh(&self) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml::CoreMLActivation, OpType};

            return self.hybrid_operation(OpType::Activation, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML Tanh implementation
                        self.coreml_tanh()
                    }
                    super::DeviceType::Cuda(_) => {
                        // CUDA Tanh not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "CUDA Tanh not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal GPU Tanh implementation
                        self.tanh_metal()
                    }
                    super::DeviceType::OpenCL(_) => {
                        // OpenCL Tanh not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "OpenCL Tanh not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback
                        Ok(self.tanh())
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for activation".to_string(),
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
            Ok(self.tanh())
        }
    }

    fn gpu_softmax(&self, dim: isize) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml::CoreMLActivation, OpType};

            return self.hybrid_operation(OpType::Activation, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML Softmax implementation
                        self.coreml_softmax(dim)
                    }
                    super::DeviceType::Cuda(_) => {
                        // CUDA Softmax not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "CUDA Softmax not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Metal(_) => {
                        // Metal Softmax not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "Metal Softmax not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::OpenCL(_) => {
                        // OpenCL Softmax not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "OpenCL Softmax not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback
                        self.softmax_fallback(dim)
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for activation".to_string(),
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
            self.softmax_fallback(dim)
        }
    }

    fn gpu_gelu(&self) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml::CoreMLActivation, OpType};

            return self.hybrid_operation(OpType::Activation, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML GELU implementation
                        self.coreml_gelu()
                    }
                    super::DeviceType::Cuda(_) => {
                        // CUDA GELU not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "CUDA GELU not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal GPU GELU implementation
                        self.gelu_metal()
                    }
                    super::DeviceType::OpenCL(_) => {
                        // OpenCL GELU not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "OpenCL GELU not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback
                        self.gelu_fallback()
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for activation".to_string(),
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
            self.gelu_fallback()
        }
    }

    fn gpu_leaky_relu(&self, negative_slope: f64) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml::CoreMLActivation, OpType};

            return self.hybrid_operation(OpType::Activation, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML Leaky ReLU implementation
                        self.coreml_leaky_relu(negative_slope)
                    }
                    super::DeviceType::Cuda(_) => {
                        // CUDA Leaky ReLU not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "CUDA Leaky ReLU not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal GPU Leaky ReLU implementation
                        self.leaky_relu_metal(negative_slope)
                    }
                    super::DeviceType::OpenCL(_) => {
                        // OpenCL Leaky ReLU not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "OpenCL Leaky ReLU not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback
                        self.leaky_relu_fallback(negative_slope)
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for activation".to_string(),
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
            self.leaky_relu_fallback(negative_slope)
        }
    }

    fn gpu_elu(&self, alpha: f64) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml::CoreMLActivation, OpType};

            return self.hybrid_operation(OpType::Activation, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML ELU implementation
                        self.coreml_elu(alpha)
                    }
                    super::DeviceType::Cuda(_) => {
                        // CUDA ELU not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "CUDA ELU not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal GPU ELU implementation
                        self.elu_metal(alpha)
                    }
                    super::DeviceType::OpenCL(_) => {
                        // OpenCL ELU not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "OpenCL ELU not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback
                        self.elu_fallback(alpha)
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for activation".to_string(),
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
            self.elu_fallback(alpha)
        }
    }

    fn gpu_swish(&self) -> RusTorchResult<Tensor<T>> {
        // Use hybrid execution for CoreML + GPU fallback
        #[cfg(any(
            feature = "coreml",
            feature = "coreml-hybrid",
            feature = "coreml-fallback"
        ))]
        {
            use crate::gpu::hybrid_executor::HybridExecution;
            use crate::gpu::{coreml::CoreMLActivation, OpType};

            return self.hybrid_operation(OpType::Activation, |device| {
                match device {
                    super::DeviceType::CoreML(_) => {
                        // Use CoreML Swish implementation
                        self.coreml_swish()
                    }
                    super::DeviceType::Cuda(_) => {
                        // CUDA Swish not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "CUDA Swish not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Metal(_) => {
                        // Use Metal GPU Swish implementation
                        self.swish_metal()
                    }
                    super::DeviceType::OpenCL(_) => {
                        // OpenCL Swish not implemented - return error instead of CPU fallback
                        Err(RusTorchError::UnsupportedOperation(
                            "OpenCL Swish not yet implemented".to_string()
                        ))
                    }
                    super::DeviceType::Cpu => {
                        // CPU fallback
                        self.swish_fallback()
                    }
                    _ => Err(RusTorchError::UnsupportedDevice(
                        "Unsupported device for activation".to_string(),
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
            self.swish_fallback()
        }
    }
}

// Non-CoreML implementation of GpuActivation trait
// 非CoreML用のGpuActivation trait実装
#[cfg(not(any(
    feature = "coreml",
    feature = "coreml-hybrid",
    feature = "coreml-fallback"
)))]
impl<T: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static> GpuActivation<T>
    for Tensor<T>
{
    fn gpu_relu(&self) -> RusTorchResult<Tensor<T>> {
        self.relu_fallback()
    }

    fn gpu_sigmoid(&self) -> RusTorchResult<Tensor<T>> {
        self.sigmoid_fallback()
    }

    fn gpu_tanh(&self) -> RusTorchResult<Tensor<T>> {
        Ok(self.tanh())
    }

    fn gpu_softmax(&self, dim: isize) -> RusTorchResult<Tensor<T>> {
        self.softmax_fallback(dim)
    }

    fn gpu_gelu(&self) -> RusTorchResult<Tensor<T>> {
        self.gelu_fallback()
    }

    fn gpu_leaky_relu(&self, negative_slope: f64) -> RusTorchResult<Tensor<T>> {
        self.leaky_relu_fallback(negative_slope)
    }

    fn gpu_elu(&self, alpha: f64) -> RusTorchResult<Tensor<T>> {
        self.elu_fallback(alpha)
    }

    fn gpu_swish(&self) -> RusTorchResult<Tensor<T>> {
        self.swish_fallback()
    }
}

// Fallback implementations for tensor activation operations
// テンソル活性化演算のフォールバック実装
impl<T: Float + FromPrimitive + ScalarOperand + Send + Sync + 'static> Tensor<T> {
    /// CPU fallback ReLU implementation
    /// CPU フォールバック ReLU 実装
    pub fn relu_fallback(&self) -> RusTorchResult<Tensor<T>> {
        let result_data = self.data.mapv(|x| x.max(T::zero()));
        Ok(Tensor::from_ndarray(result_data))
    }

    /// CPU fallback Sigmoid implementation
    /// CPU フォールバック Sigmoid 実装
    pub fn sigmoid_fallback(&self) -> RusTorchResult<Tensor<T>> {
        let result_data = self.data.mapv(|x| T::one() / (T::one() + (-x).exp()));
        Ok(Tensor::from_ndarray(result_data))
    }

    /// CPU fallback Softmax implementation
    /// CPU フォールバック Softmax 実装
    pub fn softmax_fallback(&self, dim: isize) -> RusTorchResult<Tensor<T>> {
        // Simple softmax implementation - subtract max for numerical stability
        let max_val = self.data.fold(T::neg_infinity(), |acc, &x| acc.max(x));
        let exp_data = self.data.mapv(|x| (x - max_val).exp());
        let sum = exp_data.sum();
        let result_data = exp_data.mapv(|x| x / sum);
        Ok(Tensor::from_ndarray(result_data))
    }

    /// CPU fallback GELU implementation
    /// CPU フォールバック GELU 実装
    pub fn gelu_fallback(&self) -> RusTorchResult<Tensor<T>> {
        let sqrt_2_pi = T::from(0.7978845608).unwrap(); // sqrt(2/π)
        let result_data = self.data.mapv(|x| {
            let tanh_arg = sqrt_2_pi * (x + T::from(0.044715).unwrap() * x.powi(3));
            T::from(0.5).unwrap() * x * (T::one() + tanh_arg.tanh())
        });
        Ok(Tensor::from_ndarray(result_data))
    }

    /// CPU fallback Leaky ReLU implementation
    /// CPU フォールバック Leaky ReLU 実装
    pub fn leaky_relu_fallback(&self, negative_slope: f64) -> RusTorchResult<Tensor<T>> {
        let slope = T::from(negative_slope).unwrap();
        let result_data = self
            .data
            .mapv(|x| if x > T::zero() { x } else { slope * x });
        Ok(Tensor::from_ndarray(result_data))
    }

    /// CPU fallback ELU implementation
    /// CPU フォールバック ELU 実装
    pub fn elu_fallback(&self, alpha: f64) -> RusTorchResult<Tensor<T>> {
        let alpha_val = T::from(alpha).unwrap();
        let result_data = self.data.mapv(|x| {
            if x > T::zero() {
                x
            } else {
                alpha_val * (x.exp() - T::one())
            }
        });
        Ok(Tensor::from_ndarray(result_data))
    }

    /// CPU fallback Swish implementation
    /// CPU フォールバック Swish 実装
    pub fn swish_fallback(&self) -> RusTorchResult<Tensor<T>> {
        let result_data = self.data.mapv(|x| x / (T::one() + (-x).exp()));
        Ok(Tensor::from_ndarray(result_data))
    }

    /// Metal GPU ReLU implementation
    /// Metal GPU ReLU実装
    #[cfg(feature = "metal")]
    pub fn relu_metal(&self) -> RusTorchResult<Tensor<T>> {
        use crate::gpu::metal_kernels::metal_relu_f32;

        let input_data: Vec<f32> = self.data.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
        let mut output_data = vec![0.0f32; input_data.len()];

        metal_relu_f32(&input_data, &mut output_data)?;

        let output_tensor_data: Vec<T> = output_data.iter().map(|&x| T::from(x).unwrap()).collect();
        Ok(Tensor::from_vec(output_tensor_data, self.shape().to_vec()))
    }

    /// Metal GPU Sigmoid implementation
    /// Metal GPU Sigmoid実装
    #[cfg(feature = "metal")]
    pub fn sigmoid_metal(&self) -> RusTorchResult<Tensor<T>> {
        use crate::gpu::metal_kernels::metal_sigmoid_f32;

        let input_data: Vec<f32> = self.data.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
        let mut output_data = vec![0.0f32; input_data.len()];

        metal_sigmoid_f32(&input_data, &mut output_data)?;

        let output_tensor_data: Vec<T> = output_data.iter().map(|&x| T::from(x).unwrap()).collect();
        Ok(Tensor::from_vec(output_tensor_data, self.shape().to_vec()))
    }

    /// Metal GPU Tanh implementation
    /// Metal GPU Tanh実装
    #[cfg(feature = "metal")]
    pub fn tanh_metal(&self) -> RusTorchResult<Tensor<T>> {
        use crate::gpu::metal_kernels::metal_tanh_f32;

        let input_data: Vec<f32> = self.data.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
        let mut output_data = vec![0.0f32; input_data.len()];

        metal_tanh_f32(&input_data, &mut output_data)?;

        let output_tensor_data: Vec<T> = output_data.iter().map(|&x| T::from(x).unwrap()).collect();
        Ok(Tensor::from_vec(output_tensor_data, self.shape().to_vec()))
    }

    /// Metal GPU GELU implementation
    /// Metal GPU GELU実装
    #[cfg(feature = "metal")]
    pub fn gelu_metal(&self) -> RusTorchResult<Tensor<T>> {
        use crate::gpu::metal_kernels::metal_gelu_f32;

        let input_data: Vec<f32> = self.data.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
        let mut output_data = vec![0.0f32; input_data.len()];

        metal_gelu_f32(&input_data, &mut output_data)?;

        let output_tensor_data: Vec<T> = output_data.iter().map(|&x| T::from(x).unwrap()).collect();
        Ok(Tensor::from_vec(output_tensor_data, self.shape().to_vec()))
    }

    /// Metal GPU Leaky ReLU implementation
    /// Metal GPU Leaky ReLU実装
    #[cfg(feature = "metal")]
    pub fn leaky_relu_metal(&self, negative_slope: f64) -> RusTorchResult<Tensor<T>> {
        use crate::gpu::metal_kernels::metal_leaky_relu_f32;

        let input_data: Vec<f32> = self.data.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
        let mut output_data = vec![0.0f32; input_data.len()];

        metal_leaky_relu_f32(&input_data, &mut output_data, negative_slope as f32)?;

        let output_tensor_data: Vec<T> = output_data.iter().map(|&x| T::from(x).unwrap()).collect();
        Ok(Tensor::from_vec(output_tensor_data, self.shape().to_vec()))
    }

    /// Metal GPU ELU implementation
    /// Metal GPU ELU実装
    #[cfg(feature = "metal")]
    pub fn elu_metal(&self, alpha: f64) -> RusTorchResult<Tensor<T>> {
        use crate::gpu::metal_kernels::metal_elu_f32;

        let input_data: Vec<f32> = self.data.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
        let mut output_data = vec![0.0f32; input_data.len()];

        metal_elu_f32(&input_data, &mut output_data, alpha as f32)?;

        let output_tensor_data: Vec<T> = output_data.iter().map(|&x| T::from(x).unwrap()).collect();
        Ok(Tensor::from_vec(output_tensor_data, self.shape().to_vec()))
    }

    /// Metal GPU Swish implementation
    /// Metal GPU Swish実装
    #[cfg(feature = "metal")]
    pub fn swish_metal(&self) -> RusTorchResult<Tensor<T>> {
        use crate::gpu::metal_kernels::metal_swish_f32;

        let input_data: Vec<f32> = self.data.iter().map(|&x| x.to_f32().unwrap_or(0.0)).collect();
        let mut output_data = vec![0.0f32; input_data.len()];

        metal_swish_f32(&input_data, &mut output_data)?;

        let output_tensor_data: Vec<T> = output_data.iter().map(|&x| T::from(x).unwrap()).collect();
        Ok(Tensor::from_vec(output_tensor_data, self.shape().to_vec()))
    }

    // Non-Metal fallback implementations for missing Metal methods
    #[cfg(not(feature = "metal"))]
    pub fn relu_metal(&self) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::UnsupportedDevice("Metal not available".to_string()))
    }

    #[cfg(not(feature = "metal"))]
    pub fn sigmoid_metal(&self) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::UnsupportedDevice("Metal not available".to_string()))
    }

    #[cfg(not(feature = "metal"))]
    pub fn tanh_metal(&self) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::UnsupportedDevice("Metal not available".to_string()))
    }

    #[cfg(not(feature = "metal"))]
    pub fn gelu_metal(&self) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::UnsupportedDevice("Metal not available".to_string()))
    }

    #[cfg(not(feature = "metal"))]
    pub fn leaky_relu_metal(&self, _negative_slope: f64) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::UnsupportedDevice("Metal not available".to_string()))
    }

    #[cfg(not(feature = "metal"))]
    pub fn elu_metal(&self, _alpha: f64) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::UnsupportedDevice("Metal not available".to_string()))
    }

    #[cfg(not(feature = "metal"))]
    pub fn swish_metal(&self) -> RusTorchResult<Tensor<T>> {
        Err(RusTorchError::UnsupportedDevice("Metal not available".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_gpu_relu_fallback() {
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let tensor = Tensor::<f32>::from_vec(data, vec![2, 2]);

        let result = tensor.gpu_relu().unwrap();
        let result_data = result.as_slice().unwrap().to_vec();

        assert_eq!(result_data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gpu_sigmoid_fallback() {
        let data = vec![0.0, 1.0];
        let tensor = Tensor::<f32>::from_vec(data, vec![2]);

        let result = tensor.gpu_sigmoid().unwrap();
        let result_data = result.as_slice().unwrap().to_vec();

        // Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.731
        assert!((result_data[0] - 0.5).abs() < 0.001);
        assert!((result_data[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_gpu_tanh_fallback() {
        let data = vec![0.0, 1.0];
        let tensor = Tensor::<f32>::from_vec(data, vec![2]);

        let result = tensor.gpu_tanh().unwrap();
        let result_data = result.as_slice().unwrap().to_vec();

        // Tanh(0) = 0, Tanh(1) ≈ 0.762
        assert!((result_data[0] - 0.0).abs() < 0.001);
        assert!((result_data[1] - 0.762).abs() < 0.01);
    }
}

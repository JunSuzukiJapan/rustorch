//! CoreML activation function operations
//! CoreML活性化関数演算

use super::*;
use crate::tensor::Tensor;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive};

/// CoreML Activation operations trait
/// CoreML活性化演算トレイト
pub trait CoreMLActivation<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    /// CoreML ReLU activation
    /// CoreML ReLU活性化
    fn coreml_relu(&self) -> CoreMLResult<Tensor<T>>;

    /// CoreML Sigmoid activation
    /// CoreML Sigmoid活性化
    fn coreml_sigmoid(&self) -> CoreMLResult<Tensor<T>>;

    /// CoreML Tanh activation
    /// CoreML Tanh活性化
    fn coreml_tanh(&self) -> CoreMLResult<Tensor<T>>;

    /// CoreML Softmax activation
    /// CoreML Softmax活性化
    fn coreml_softmax(&self, dim: isize) -> CoreMLResult<Tensor<T>>;

    /// CoreML GELU activation
    /// CoreML GELU活性化
    fn coreml_gelu(&self) -> CoreMLResult<Tensor<T>>;

    /// CoreML Leaky ReLU activation
    /// CoreML Leaky ReLU活性化
    fn coreml_leaky_relu(&self, negative_slope: f64) -> CoreMLResult<Tensor<T>>;

    /// CoreML ELU activation
    /// CoreML ELU活性化
    fn coreml_elu(&self, alpha: f64) -> CoreMLResult<Tensor<T>>;

    /// CoreML Swish activation
    /// CoreML Swish活性化
    fn coreml_swish(&self) -> CoreMLResult<Tensor<T>>;
}

/// Generic activation operation for CoreML
/// CoreML用汎用活性化演算
pub struct ActivationOperation<T> {
    input: Tensor<T>,
    activation_type: CoreMLActivationType,
    parameters: ActivationParameters,
}

/// Parameters for activation functions
/// 活性化関数用パラメータ
#[derive(Debug, Clone)]
pub struct ActivationParameters {
    /// For Leaky ReLU: negative slope
    /// Leaky ReLU用: 負の傾き
    pub negative_slope: Option<f64>,

    /// For ELU: alpha parameter
    /// ELU用: alphaパラメータ
    pub alpha: Option<f64>,

    /// For Softmax: dimension
    /// Softmax用: 次元
    pub dim: Option<isize>,
}

impl Default for ActivationParameters {
    fn default() -> Self {
        Self {
            negative_slope: None,
            alpha: None,
            dim: None,
        }
    }
}

impl<T> ActivationOperation<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    /// Create new activation operation
    /// 新しい活性化演算を作成
    pub fn new(input: Tensor<T>, activation_type: CoreMLActivationType) -> Self {
        Self {
            input,
            activation_type,
            parameters: ActivationParameters::default(),
        }
    }

    /// Create activation with parameters
    /// パラメータ付き活性化を作成
    pub fn with_parameters(
        input: Tensor<T>,
        activation_type: CoreMLActivationType,
        parameters: ActivationParameters,
    ) -> Self {
        Self {
            input,
            activation_type,
            parameters,
        }
    }

    /// Check if activation is efficient on CoreML
    /// 活性化がCoreMLで効率的かチェック
    fn is_efficient_on_coreml(&self) -> bool {
        let elements: usize = self.input.shape().iter().product();

        match self.activation_type {
            CoreMLActivationType::ReLU |
            CoreMLActivationType::Sigmoid |
            CoreMLActivationType::Tanh => elements > 256, // Efficient for medium+ sizes
            CoreMLActivationType::Softmax |
            CoreMLActivationType::GELU => elements > 1024, // More complex operations need larger sizes
            CoreMLActivationType::LeakyReLU |
            CoreMLActivationType::ELU |
            CoreMLActivationType::Swish => elements > 512, // Moderately complex
        }
    }
}

impl<T> CoreMLOperation<T> for ActivationOperation<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    fn execute_coreml(&self, device_id: usize) -> CoreMLResult<Tensor<T>> {
        coreml_feature! {
            use crate::gpu::coreml_backend::{CoreMLGraph, ActivationType};

            let graph = CoreMLGraph::new(device_id)?;

            // Convert our activation type to backend type
            let backend_activation = match self.activation_type {
                CoreMLActivationType::ReLU => ActivationType::ReLU,
                CoreMLActivationType::Sigmoid => ActivationType::Sigmoid,
                CoreMLActivationType::Tanh => ActivationType::Tanh,
                CoreMLActivationType::Softmax => ActivationType::Softmax,
                CoreMLActivationType::GELU => ActivationType::GELU,
                _ => {
                    // Unsupported activations fall back to CPU
                    return Err(error_helpers::unsupported_operation(&format!(
                        "Activation {:?} not yet implemented in CoreML backend",
                        self.activation_type
                    )));
                }
            };

            graph.activation(&self.input, backend_activation)
        }

        #[cfg(not(any(feature = "coreml", feature = "coreml-hybrid", feature = "coreml-fallback")))]
        {
            Err(error_helpers::feature_disabled())
        }
    }

    fn is_supported_by_coreml(&self) -> bool {
        // Check if activation type is supported and size is efficient
        let is_supported = matches!(
            self.activation_type,
            CoreMLActivationType::ReLU |
            CoreMLActivationType::Sigmoid |
            CoreMLActivationType::Tanh |
            CoreMLActivationType::Softmax |
            CoreMLActivationType::GELU
        );

        is_supported && self.is_efficient_on_coreml()
    }

    fn estimated_execution_time(&self) -> Option<std::time::Duration> {
        if !self.is_supported_by_coreml() {
            return None;
        }

        let elements: usize = self.input.shape().iter().product();

        // Rough estimation based on activation complexity
        let nanos_per_element = match self.activation_type {
            CoreMLActivationType::ReLU => 1,        // Very fast
            CoreMLActivationType::Sigmoid => 10,    // Moderate
            CoreMLActivationType::Tanh => 8,        // Moderate
            CoreMLActivationType::Softmax => 20,    // Complex (needs reduction)
            CoreMLActivationType::GELU => 15,       // Complex
            _ => 5,                                 // Default
        };

        Some(std::time::Duration::from_nanos((elements * nanos_per_element) as u64))
    }
}

/// Implement CoreML activation for Tensor
/// TensorにCoreML活性化を実装
impl<T> CoreMLActivation<T> for Tensor<T>
where
    T: Float + FromPrimitive + ScalarOperand + 'static,
{
    fn coreml_relu(&self) -> CoreMLResult<Tensor<T>> {
        let operation = ActivationOperation::new(self.clone(), CoreMLActivationType::ReLU);
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_sigmoid(&self) -> CoreMLResult<Tensor<T>> {
        let operation = ActivationOperation::new(self.clone(), CoreMLActivationType::Sigmoid);
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_tanh(&self) -> CoreMLResult<Tensor<T>> {
        let operation = ActivationOperation::new(self.clone(), CoreMLActivationType::Tanh);
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_softmax(&self, dim: isize) -> CoreMLResult<Tensor<T>> {
        let parameters = ActivationParameters {
            dim: Some(dim),
            ..Default::default()
        };
        let operation = ActivationOperation::with_parameters(
            self.clone(),
            CoreMLActivationType::Softmax,
            parameters,
        );
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_gelu(&self) -> CoreMLResult<Tensor<T>> {
        let operation = ActivationOperation::new(self.clone(), CoreMLActivationType::GELU);
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_leaky_relu(&self, negative_slope: f64) -> CoreMLResult<Tensor<T>> {
        let parameters = ActivationParameters {
            negative_slope: Some(negative_slope),
            ..Default::default()
        };
        let operation = ActivationOperation::with_parameters(
            self.clone(),
            CoreMLActivationType::LeakyReLU,
            parameters,
        );
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_elu(&self, alpha: f64) -> CoreMLResult<Tensor<T>> {
        let parameters = ActivationParameters {
            alpha: Some(alpha),
            ..Default::default()
        };
        let operation = ActivationOperation::with_parameters(
            self.clone(),
            CoreMLActivationType::ELU,
            parameters,
        );
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }

    fn coreml_swish(&self) -> CoreMLResult<Tensor<T>> {
        let operation = ActivationOperation::new(self.clone(), CoreMLActivationType::Swish);
        let executor = CoreMLExecutor::new(0)?;
        executor.execute(&operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_operation_creation() {
        let input = Tensor::<f32>::zeros(&[10, 10]);
        let operation = ActivationOperation::new(input, CoreMLActivationType::ReLU);

        assert!(operation.is_efficient_on_coreml()); // 100 elements > 256 threshold for ReLU
    }

    #[test]
    fn test_small_tensor_not_efficient() {
        let input = Tensor::<f32>::zeros(&[5, 5]); // 25 elements
        let operation = ActivationOperation::new(input, CoreMLActivationType::ReLU);

        assert!(!operation.is_efficient_on_coreml()); // Too small
        assert!(!operation.is_supported_by_coreml());
    }

    #[test]
    fn test_supported_activations() {
        let input = Tensor::<f32>::zeros(&[32, 32]); // 1024 elements

        let supported_types = vec![
            CoreMLActivationType::ReLU,
            CoreMLActivationType::Sigmoid,
            CoreMLActivationType::Tanh,
            CoreMLActivationType::Softmax,
            CoreMLActivationType::GELU,
        ];

        for activation_type in supported_types {
            let operation = ActivationOperation::new(input.clone(), activation_type);
            assert!(operation.is_supported_by_coreml());
        }
    }

    #[test]
    fn test_execution_time_estimation() {
        let input = Tensor::<f32>::zeros(&[64, 64]); // 4096 elements
        let operation = ActivationOperation::new(input, CoreMLActivationType::ReLU);

        let estimated_time = operation.estimated_execution_time();
        assert!(estimated_time.is_some());

        let time = estimated_time.unwrap();
        assert!(time.as_nanos() > 0);
    }

    #[test]
    fn test_parameters() {
        let input = Tensor::<f32>::zeros(&[32, 32]);
        let parameters = ActivationParameters {
            negative_slope: Some(0.01),
            alpha: Some(1.0),
            dim: Some(-1),
        };

        let operation = ActivationOperation::with_parameters(
            input,
            CoreMLActivationType::LeakyReLU,
            parameters,
        );

        assert_eq!(operation.parameters.negative_slope, Some(0.01));
        assert_eq!(operation.parameters.alpha, Some(1.0));
        assert_eq!(operation.parameters.dim, Some(-1));
    }
}
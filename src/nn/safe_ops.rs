//! Safe operations for neural network layers
//! ニューラルネットワーク層の安全な操作

use crate::autograd::Variable;
use crate::error::{RusTorchError, RusTorchResult};
use crate::tensor::Tensor;
type NNResult<T> = RusTorchResult<T>;
use num_traits::Float;
use std::fmt::Debug;

/// Safe tensor operations with proper error handling
/// 適切なエラーハンドリングを持つ安全なテンソル操作
pub struct SafeOps;

impl SafeOps {
    /// Safely create a variable with validation
    /// 検証付きで安全にVariableを作成
    pub fn create_variable<T>(
        data: Vec<T>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> NNResult<Variable<T>>
    where
        T: Float
            + Send
            + Sync
            + 'static
            + Debug
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    {
        // Validate shape consistency
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(RusTorchError::InvalidDimensions(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            )));
        }

        // Check for empty dimensions
        if shape.iter().any(|&dim| dim == 0) {
            return Err(RusTorchError::InvalidDimensions(
                "Shape dimensions must be positive".to_string(),
            ));
        }

        let tensor = Tensor::from_vec(data, shape);
        Ok(Variable::new(tensor, requires_grad))
    }

    /// Safely reshape a variable
    /// 安全にVariableをリシェイプ
    pub fn reshape<T>(variable: &Variable<T>, new_shape: Vec<usize>) -> NNResult<Variable<T>>
    where
        T: Float
            + Send
            + Sync
            + 'static
            + Debug
            + Clone
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    {
        let binding = variable.data();
        let data_guard = binding
            .read()
            .map_err(|_| RusTorchError::MemoryError("Failed to acquire data lock".to_string()))?;

        let current_size = data_guard.shape().iter().product::<usize>();
        let new_size: usize = new_shape.iter().product();

        if current_size != new_size {
            return Err(RusTorchError::InvalidDimensions(format!(
                "Cannot reshape tensor of size {} to size {}",
                current_size, new_size
            )));
        }

        // Create new tensor with same data but new shape
        let data_vec = data_guard.as_array().iter().cloned().collect();
        let new_tensor = Tensor::from_vec(data_vec, new_shape);

        Ok(Variable::new(new_tensor, variable.requires_grad()))
    }

    /// Safely apply a function to tensor data
    /// テンソルデータに安全に関数を適用
    pub fn apply_function<T, F>(variable: &Variable<T>, f: F) -> NNResult<Variable<T>>
    where
        T: Float
            + Send
            + Sync
            + 'static
            + Debug
            + Clone
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
        F: Fn(T) -> T,
    {
        let binding = variable.data();
        let data_guard = binding
            .read()
            .map_err(|_| RusTorchError::MemoryError("Failed to acquire data lock".to_string()))?;

        let new_data: Vec<T> = data_guard.as_array().iter().map(|&x| f(x)).collect();

        let new_tensor = Tensor::from_vec(new_data, data_guard.shape().to_vec());
        Ok(Variable::new(new_tensor, variable.requires_grad()))
    }

    /// Safely get tensor statistics
    /// 安全にテンソル統計を取得
    pub fn get_stats<T>(variable: &Variable<T>) -> NNResult<TensorStats<T>>
    where
        T: Float
            + Send
            + Sync
            + 'static
            + Debug
            + Clone
            + PartialOrd
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    {
        let binding = variable.data();
        let data_guard = binding
            .read()
            .map_err(|_| RusTorchError::MemoryError("Failed to acquire data lock".to_string()))?;

        let data = data_guard.as_array();

        if data.is_empty() {
            return Err(RusTorchError::InvalidDimensions("Empty tensor"));
        }

        let first = *data.iter().next().unwrap();
        let min = data
            .iter()
            .fold(first, |acc, &x| if x < acc { x } else { acc });
        let max = data
            .iter()
            .fold(first, |acc, &x| if x > acc { x } else { acc });

        let sum = data.iter().fold(T::zero(), |acc, &x| acc + x);
        let mean = sum / T::from(data.len()).unwrap();

        // Compute variance
        let variance = data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
            / T::from(data.len()).unwrap();

        Ok(TensorStats {
            min,
            max,
            mean,
            variance,
            count: data.len(),
        })
    }

    /// Safely validate tensor for NaN or infinity
    /// NaNや無限大のテンソルを安全に検証
    pub fn validate_finite<T>(variable: &Variable<T>) -> NNResult<()>
    where
        T: Float
            + Send
            + Sync
            + 'static
            + Debug
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    {
        let binding = variable.data();
        let data_guard = binding
            .read()
            .map_err(|_| RusTorchError::MemoryError("Failed to acquire data lock".to_string()))?;

        for (i, &value) in data_guard.as_array().iter().enumerate() {
            if value.is_nan() {
                return Err(RusTorchError::ComputationError(format!(
                    "NaN detected at index {}",
                    i
                )));
            }
            if value.is_infinite() {
                return Err(RusTorchError::ComputationError(format!(
                    "Infinity detected at index {}",
                    i
                )));
            }
        }

        Ok(())
    }

    /// Apply ReLU activation function: max(0, x)
    /// ReLU活性化関数を適用: max(0, x)
    pub fn relu<T>(variable: &Variable<T>) -> NNResult<Variable<T>>
    where
        T: Float
            + Send
            + Sync
            + 'static
            + Debug
            + Clone
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive,
    {
        let binding = variable.data();
        let data_guard = binding
            .read()
            .map_err(|_| RusTorchError::MemoryError("Failed to acquire data lock".to_string()))?;

        let new_data: Vec<T> = data_guard
            .as_array()
            .iter()
            .map(|&x| if x < T::zero() { T::zero() } else { x })
            .collect();

        let new_tensor = Tensor::from_vec(new_data, data_guard.shape().to_vec());
        Ok(Variable::new(new_tensor, variable.requires_grad()))
    }
}

/// Statistics about a tensor
/// テンソルの統計情報
#[derive(Debug, Clone)]
pub struct TensorStats<T> {
    /// Minimum value
    /// 最小値
    pub min: T,
    /// Maximum value
    /// 最大値
    pub max: T,
    /// Mean value
    /// 平均値
    pub mean: T,
    /// Variance
    /// 分散
    pub variance: T,
    /// Number of elements
    /// 要素数
    pub count: usize,
}

impl<T: Float + ndarray::ScalarOperand + num_traits::FromPrimitive> TensorStats<T> {
    /// Get standard deviation
    /// 標準偏差を取得
    pub fn std_dev(&self) -> T {
        self.variance.sqrt()
    }

    /// Check if values are within reasonable range
    /// 値が妥当な範囲内かチェック
    pub fn is_reasonable(&self) -> bool {
        !self.min.is_infinite()
            && !self.max.is_infinite()
            && !self.mean.is_nan()
            && !self.variance.is_nan()
            && self.variance >= T::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_variable_creation() {
        let result = SafeOps::create_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        assert!(result.is_ok());

        // Test mismatched dimensions
        let result = SafeOps::create_variable(vec![1.0, 2.0, 3.0], vec![2, 2], false);
        assert!(result.is_err());

        // Test zero dimension
        let result = SafeOps::create_variable(vec![1.0], vec![1, 0], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_safe_reshape() {
        let var = SafeOps::create_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();

        // Valid reshape
        let result = SafeOps::reshape(&var, vec![4, 1]);
        assert!(result.is_ok());

        // Invalid reshape
        let result = SafeOps::reshape(&var, vec![3, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_function() {
        let var = SafeOps::create_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();

        let result = SafeOps::apply_function(&var, |x| x * 2.0);
        assert!(result.is_ok());

        let doubled = result.unwrap();
        let binding = doubled.data();
        let data_guard = binding.read().unwrap();
        let expected = vec![2.0, 4.0, 6.0, 8.0];
        assert_eq!(
            data_guard.as_array().as_slice().unwrap(),
            expected.as_slice()
        );
    }

    #[test]
    fn test_tensor_stats() {
        let var = SafeOps::create_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();

        let stats = SafeOps::get_stats(&var).unwrap();
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 4.0);
        assert_eq!(stats.mean, 2.5);
        assert_eq!(stats.count, 4);
        assert!(stats.is_reasonable());
    }

    #[test]
    fn test_finite_validation() {
        let var = SafeOps::create_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false).unwrap();
        assert!(SafeOps::validate_finite(&var).is_ok());

        let nan_var =
            SafeOps::create_variable(vec![1.0, f32::NAN, 3.0, 4.0], vec![2, 2], false).unwrap();
        assert!(SafeOps::validate_finite(&nan_var).is_err());

        let inf_var =
            SafeOps::create_variable(vec![1.0, f32::INFINITY, 3.0, 4.0], vec![2, 2], false)
                .unwrap();
        assert!(SafeOps::validate_finite(&inf_var).is_err());
    }

    #[test]
    fn test_stats_calculation() {
        let var = SafeOps::create_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false).unwrap();
        let stats = SafeOps::get_stats(&var).unwrap();

        // Mean should be 2.5
        assert!((stats.mean - 2.5).abs() < 1e-6);

        // Variance should be 1.25 (for population variance)
        assert!((stats.variance - 1.25).abs() < 1e-6);

        // Standard deviation should be sqrt(1.25) ≈ 1.118
        assert!((stats.std_dev() - 1.118033988749895).abs() < 1e-6);
    }

    #[test]
    fn test_relu_function() {
        let var =
            SafeOps::create_variable(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false).unwrap();

        let relu_result = SafeOps::relu(&var).unwrap();
        let binding = relu_result.data();
        let data_guard = binding.read().unwrap();
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        assert_eq!(
            data_guard.as_array().as_slice().unwrap(),
            expected.as_slice()
        );
    }
}

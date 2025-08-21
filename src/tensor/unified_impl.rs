//! Unified tensor implementations to reduce code duplication
//! 重複コードを削減するための統一テンソル実装

use crate::tensor::{Tensor, parallel_errors::ParallelResult};
use num_traits::Float;

/// Common trait bounds for tensor operations
/// テンソル操作の共通トレイト境界
pub trait TensorFloat: Float + Send + Sync + Clone + 'static + std::fmt::Debug {}

impl<T> TensorFloat for T where T: Float + Send + Sync + Clone + 'static + std::fmt::Debug {}

/// Unified tensor operations implementation
/// 統一テンソル操作実装
pub struct TensorOps;

impl TensorOps {
    /// Common element-wise operation implementation
    /// 共通要素ごと演算実装
    pub fn elementwise_op<T, F>(
        lhs: &Tensor<T>,
        rhs: &Tensor<T>,
        op: F,
    ) -> ParallelResult<Tensor<T>>
    where
        T: TensorFloat,
        F: Fn(T, T) -> T + Send + Sync,
    {
        if lhs.shape() != rhs.shape() {
            return Err(crate::tensor::parallel_errors::ParallelError::shape_mismatch(
                lhs.shape(),
                rhs.shape(),
                "elementwise_operation",
            ));
        }

        let lhs_data = lhs.as_slice().ok_or_else(|| {
            crate::tensor::parallel_errors::ParallelError::ParallelExecutionError {
                message: "Cannot get slice from tensor".to_string(),
            }
        })?;
        let rhs_data = rhs.as_slice().ok_or_else(|| {
            crate::tensor::parallel_errors::ParallelError::ParallelExecutionError {
                message: "Cannot get slice from tensor".to_string(),
            }
        })?;

        let result_data: Vec<T> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Ok(Tensor::from_vec(result_data, lhs.shape().to_vec()))
    }

    /// Common reduction operation implementation
    /// 共通リダクション演算実装
    pub fn reduce_op<T, F, R>(
        tensor: &Tensor<T>,
        init: R,
        op: F,
    ) -> ParallelResult<R>
    where
        T: TensorFloat,
        F: Fn(R, T) -> R + Send + Sync,
        R: Clone + Send + Sync,
    {
        let data = tensor.as_slice().ok_or_else(|| {
            crate::tensor::parallel_errors::ParallelError::ParallelExecutionError {
                message: "Cannot get slice from tensor".to_string(),
            }
        })?;

        Ok(data.iter().fold(init, |acc, &x| op(acc, x)))
    }

    /// Common matrix multiplication validation
    /// 共通行列乗算検証
    pub fn validate_matmul_shapes(
        lhs_shape: &[usize],
        rhs_shape: &[usize],
    ) -> ParallelResult<Vec<usize>> {
        match (lhs_shape.len(), rhs_shape.len()) {
            (2, 2) => {
                if lhs_shape[1] != rhs_shape[0] {
                    return Err(crate::tensor::parallel_errors::ParallelError::MatmulDimensionMismatch {
                        left_shape: lhs_shape.to_vec(),
                        right_shape: rhs_shape.to_vec(),
                    });
                }
                Ok(vec![lhs_shape[0], rhs_shape[1]])
            }
            (3, 3) => {
                // Batch matrix multiplication
                if lhs_shape[0] != rhs_shape[0] || lhs_shape[2] != rhs_shape[1] {
                    return Err(crate::tensor::parallel_errors::ParallelError::MatmulDimensionMismatch {
                        left_shape: lhs_shape.to_vec(),
                        right_shape: rhs_shape.to_vec(),
                    });
                }
                Ok(vec![lhs_shape[0], lhs_shape[1], rhs_shape[2]])
            }
            _ => Err(crate::tensor::parallel_errors::ParallelError::InsufficientDimensions {
                required: 2,
                actual: lhs_shape.len(),
                operation: "matrix_multiplication".to_string(),
            }),
        }
    }

    /// Common memory transfer validation
    /// 共通メモリ転送検証
    pub fn validate_device_transfer<T>(
        tensor: &Tensor<T>,
        target_device: &str,
    ) -> ParallelResult<()>
    where
        T: TensorFloat,
    {
        // Validate tensor is not empty
        if tensor.shape().iter().any(|&dim| dim == 0) {
            return Err(crate::tensor::parallel_errors::ParallelError::EmptyTensorList {
                operation: "device_transfer".to_string(),
            });
        }

        // Validate device type
        match target_device {
            "cuda" | "metal" | "opencl" | "cpu" => Ok(()),
            _ => Err(crate::tensor::parallel_errors::ParallelError::ParallelExecutionError {
                message: format!("Unsupported device type: {}", target_device),
            }),
        }
    }
}

/// Unified arithmetic operations for tensors
/// テンソルの統一算術演算
pub struct ArithmeticOps;

impl ArithmeticOps {
    /// Generic addition implementation
    /// 汎用加算実装
    pub fn add<T>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T>
    where
        T: TensorFloat,
    {
        TensorOps::elementwise_op(lhs, rhs, |a, b| a + b)
            .unwrap_or_else(|_| lhs.clone())
    }

    /// Generic subtraction implementation
    /// 汎用減算実装
    pub fn sub<T>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T>
    where
        T: TensorFloat,
    {
        TensorOps::elementwise_op(lhs, rhs, |a, b| a - b)
            .unwrap_or_else(|_| lhs.clone())
    }

    /// Generic multiplication implementation
    /// 汎用乗算実装
    pub fn mul<T>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T>
    where
        T: TensorFloat,
    {
        TensorOps::elementwise_op(lhs, rhs, |a, b| a * b)
            .unwrap_or_else(|_| lhs.clone())
    }

    /// Generic division implementation
    /// 汎用除算実装
    pub fn div<T>(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Tensor<T>
    where
        T: TensorFloat,
    {
        TensorOps::elementwise_op(lhs, rhs, |a, b| a / b)
            .unwrap_or_else(|_| lhs.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_elementwise_ops() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], vec![3]);

        let result = TensorOps::elementwise_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_unified_arithmetic() {
        let a = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::<f32>::from_vec(vec![4.0, 5.0, 6.0], vec![3]);

        let add_result = ArithmeticOps::add(&a, &b);
        assert_eq!(add_result.as_slice().unwrap(), &[5.0, 7.0, 9.0]);

        let sub_result = ArithmeticOps::sub(&a, &b);
        assert_eq!(sub_result.as_slice().unwrap(), &[-3.0, -3.0, -3.0]);
    }

    #[test]
    fn test_matmul_validation() {
        let result = TensorOps::validate_matmul_shapes(&[2, 3], &[3, 4]).unwrap();
        assert_eq!(result, vec![2, 4]);

        let batch_result = TensorOps::validate_matmul_shapes(&[2, 3, 4], &[2, 4, 5]).unwrap();
        assert_eq!(batch_result, vec![2, 3, 5]);
    }

    #[test]
    fn test_device_transfer_validation() {
        let tensor = Tensor::<f32>::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        
        assert!(TensorOps::validate_device_transfer(&tensor, "cuda").is_ok());
        assert!(TensorOps::validate_device_transfer(&tensor, "metal").is_ok());
        assert!(TensorOps::validate_device_transfer(&tensor, "invalid").is_err());
    }
}

//! Matrix operations for tensors
//! テンソルの行列演算

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Matrix multiplication (new implementation)
    /// 行列乗算（新実装）
    pub fn matmul_v2(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        let self_shape = self.shape();
        let other_shape = other.shape();

        // Handle different dimensionalities
        match (self_shape.len(), other_shape.len()) {
            (2, 2) => {
                // Standard 2D matrix multiplication
                let (m, k) = (self_shape[0], self_shape[1]);
                let (k2, n) = (other_shape[0], other_shape[1]);

                if k != k2 {
                    return Err(RusTorchError::InvalidOperation {
                        operation: "matmul".to_string(),
                        message: format!("Inner dimensions must match: {} vs {}", k, k2),
                    });
                }

                let mut result = vec![T::zero(); m * n];

                // Optimized matrix multiplication
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = T::zero();
                        for k_idx in 0..k {
                            let a_val = self
                                .data
                                .get(ndarray::IxDyn(&[i, k_idx]))
                                .copied()
                                .unwrap_or(T::zero());
                            let b_val = other
                                .data
                                .get(ndarray::IxDyn(&[k_idx, j]))
                                .copied()
                                .unwrap_or(T::zero());
                            sum = sum + a_val * b_val;
                        }
                        result[i * n + j] = sum;
                    }
                }

                Ok(Tensor::from_vec(result, vec![m, n]))
            }
            (1, 2) => {
                // Vector-matrix multiplication
                let k = self_shape[0];
                let (k2, n) = (other_shape[0], other_shape[1]);

                if k != k2 {
                    return Err(RusTorchError::InvalidOperation {
                        operation: "matmul".to_string(),
                        message: format!("Dimensions must match: {} vs {}", k, k2),
                    });
                }

                let mut result = vec![T::zero(); n];

                for j in 0..n {
                    let mut sum = T::zero();
                    for k_idx in 0..k {
                        let a_val = self
                            .data
                            .get(ndarray::IxDyn(&[k_idx]))
                            .copied()
                            .unwrap_or(T::zero());
                        let b_val = other
                            .data
                            .get(ndarray::IxDyn(&[k_idx, j]))
                            .copied()
                            .unwrap_or(T::zero());
                        sum = sum + a_val * b_val;
                    }
                    result[j] = sum;
                }

                Ok(Tensor::from_vec(result, vec![n]))
            }
            (2, 1) => {
                // Matrix-vector multiplication
                let (m, k) = (self_shape[0], self_shape[1]);
                let k2 = other_shape[0];

                if k != k2 {
                    return Err(RusTorchError::InvalidOperation {
                        operation: "matmul".to_string(),
                        message: format!("Dimensions must match: {} vs {}", k, k2),
                    });
                }

                let mut result = vec![T::zero(); m];

                for i in 0..m {
                    let mut sum = T::zero();
                    for k_idx in 0..k {
                        let a_val = self
                            .data
                            .get(ndarray::IxDyn(&[i, k_idx]))
                            .copied()
                            .unwrap_or(T::zero());
                        let b_val = other
                            .data
                            .get(ndarray::IxDyn(&[k_idx]))
                            .copied()
                            .unwrap_or(T::zero());
                        sum = sum + a_val * b_val;
                    }
                    result[i] = sum;
                }

                Ok(Tensor::from_vec(result, vec![m]))
            }
            _ => Err(RusTorchError::UnsupportedOperation(
                "Matrix multiplication not supported for these dimensions".to_string(),
            )),
        }
    }

    /// Transpose the last two dimensions
    /// 最後の2次元を転置
    pub fn transpose_last_two(&self) -> RusTorchResult<Self> {
        let shape = self.shape();
        if shape.len() < 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "transpose_last_two".to_string(),
                message: "Tensor must have at least 2 dimensions".to_string(),
            });
        }

        let mut new_shape = shape.to_vec();
        let last_idx = shape.len() - 1;
        new_shape.swap(last_idx - 1, last_idx);

        // For 2D case, use optimized transpose
        if shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            let mut result = vec![T::zero(); rows * cols];

            for i in 0..rows {
                for j in 0..cols {
                    let val = self
                        .data
                        .get(ndarray::IxDyn(&[i, j]))
                        .copied()
                        .unwrap_or(T::zero());
                    result[j * rows + i] = val;
                }
            }

            return Ok(Tensor::from_vec(result, new_shape));
        }

        // For higher dimensions, use general transpose
        self.transpose_general(&new_shape)
    }

    /// Simple 2D transpose (new implementation)
    /// 単純な2D転置（新実装）
    pub fn transpose_v2(&self) -> RusTorchResult<Self> {
        let shape = self.shape();

        match shape.len() {
            1 => {
                // 1D vector becomes column vector
                Ok(Tensor::from_vec(
                    self.data.iter().copied().collect(),
                    vec![shape[0], 1],
                ))
            }
            2 => self.transpose_last_two(),
            _ => Err(RusTorchError::InvalidOperation {
                operation: "transpose".to_string(),
                message: "Simple transpose only supports 1D and 2D tensors".to_string(),
            }),
        }
    }

    /// General transpose implementation for higher dimensions
    /// 高次元用の一般的な転置実装
    fn transpose_general(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        // This is a simplified implementation
        // In practice, we'd use ndarray's transpose capabilities
        let total_size: usize = new_shape.iter().product();
        let mut result = vec![T::zero(); total_size];

        // Copy data (placeholder implementation)
        for (i, &val) in self.data.iter().enumerate() {
            if i < total_size {
                result[i] = val;
            }
        }

        Ok(Tensor::from_vec(result, new_shape.to_vec()))
    }

    /// Compute determinant (for 2D square matrices only)
    /// 行列式の計算（2D正方行列のみ）
    pub fn det(&self) -> RusTorchResult<T> {
        let shape = self.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(RusTorchError::InvalidOperation {
                operation: "det".to_string(),
                message: "Determinant only supported for square 2D matrices".to_string(),
            });
        }

        let n = shape[0];

        match n {
            1 => {
                if let Some(value) = self.data.get(ndarray::IxDyn(&[0, 0])) {
                    Ok(*value)
                } else {
                    Err(RusTorchError::InvalidOperation {
                        operation: "det".to_string(),
                        message: "Cannot access matrix element (0, 0)".to_string(),
                    })
                }
            }
            2 => {
                let a = self
                    .data
                    .get(ndarray::IxDyn(&[0, 0]))
                    .copied()
                    .unwrap_or(T::zero());
                let b = self
                    .data
                    .get(ndarray::IxDyn(&[0, 1]))
                    .copied()
                    .unwrap_or(T::zero());
                let c = self
                    .data
                    .get(ndarray::IxDyn(&[1, 0]))
                    .copied()
                    .unwrap_or(T::zero());
                let d = self
                    .data
                    .get(ndarray::IxDyn(&[1, 1]))
                    .copied()
                    .unwrap_or(T::zero());
                Ok(a * d - b * c)
            }
            _ => {
                // For larger matrices, would need LU decomposition
                Err(RusTorchError::UnsupportedOperation(
                    "Determinant for matrices larger than 2x2 not implemented".to_string(),
                ))
            }
        }
    }

    /// Trace (sum of diagonal elements)
    /// トレース（対角要素の和）
    pub fn trace(&self) -> RusTorchResult<T> {
        let shape = self.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(RusTorchError::InvalidOperation {
                operation: "trace".to_string(),
                message: "Trace only supported for square 2D matrices".to_string(),
            });
        }

        let n = shape[0];
        let mut trace = T::zero();

        for i in 0..n {
            if let Some(value) = self.data.get(ndarray::IxDyn(&[i, i])) {
                trace = trace + *value;
            } else {
                return Err(RusTorchError::InvalidOperation {
                    operation: "trace".to_string(),
                    message: format!("Cannot access diagonal element at ({}, {})", i, i),
                });
            }
        }

        Ok(trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = a.matmul_v2(&b).unwrap();

        // Expected: [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19, 22; 43, 50]
        assert_eq!(result.as_slice().unwrap(), &[19.0, 22.0, 43.0, 50.0]);
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_transpose_2d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = a.transpose_v2().unwrap();

        // Expected: [[1, 2, 3], [4, 5, 6]]^T = [[1, 4], [2, 5], [3, 6]]
        assert_eq!(result.as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(result.shape(), &[3, 2]);
    }

    #[test]
    fn test_det_2x2() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let det = a.det().unwrap();

        // Expected: 1*4 - 2*3 = -2
        assert_eq!(det, -2.0);
    }

    #[test]
    fn test_trace() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let trace = a.trace().unwrap();

        // Expected: 1 + 4 = 5
        assert_eq!(trace, 5.0);
    }
}

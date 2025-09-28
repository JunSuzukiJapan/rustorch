//! Matrix operations for tensors
//! テンソルの行列演算
//!
//! Note: Core matrix methods (matmul, transpose) are now defined in core.rs
//! 注意: コア行列メソッド (matmul, transpose) は core.rs で定義されています

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    // Core methods (matmul, transpose) are defined in core.rs to avoid duplication
    // コアメソッド (matmul, transpose) は重複を避けるため core.rs で定義

    /// Matrix multiplication with intelligent device selection (mac-hybrid feature)
    /// インテリジェント・デバイス選択による行列乗算（mac-hybridフィーチャー）
    #[cfg(feature = "mac-hybrid")]
    pub fn matmul_hybrid(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        use crate::gpu::{DeviceType, OpType};

        // Calculate tensor size for device selection
        let tensor_size = self.data.len() + other.data.len();

        // Select optimal device based on operation type and size
        let device = DeviceType::select_best_for_operation(&OpType::LinearAlgebra, Some(tensor_size));

        // Route to appropriate backend - prefer hardware acceleration
        match device {
            #[cfg(feature = "coreml")]
            DeviceType::CoreML(id) => self.matmul_coreml(other, id),
            DeviceType::Metal(id) => self.matmul_metal(other, id),
            _ => {
                // mac-hybrid should never reach this case due to panic in select_best_for_operation
                unreachable!("mac-hybrid device selection should always return Metal or CoreML")
            }
        }
    }

    /// Matrix multiplication
    /// 行列乗算
    pub fn matmul(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
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
    pub fn transpose(&self) -> RusTorchResult<Self> {
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
        let result = a.matmul(&b).unwrap();

        // Expected: [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19, 22; 43, 50]
        assert_eq!(result.as_slice().unwrap(), &[19.0, 22.0, 43.0, 50.0]);
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_transpose_2d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = a.transpose().unwrap();

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

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    #[cfg(feature = "coreml")]
    pub fn matmul_coreml(&self, other: &Tensor<T>, _device_id: usize) -> RusTorchResult<Self> {
        // Use actual CoreML Neural Engine hardware acceleration
        use crate::gpu::coreml::operations::linear_algebra::CoreMLLinearAlgebra;

        self.coreml_matmul(other)
            .map_err(|e| RusTorchError::InvalidOperation {
                operation: "matmul_coreml".to_string(),
                message: format!("CoreML matmul failed: {}", e),
            })
    }

    #[cfg(not(feature = "coreml"))]
    fn matmul_coreml(&self, _other: &Tensor<T>, _device_id: usize) -> RusTorchResult<Self> {
        Err(RusTorchError::UnsupportedOperation(
            "CoreML feature not enabled".to_string(),
        ))
    }

    pub fn matmul_metal(&self, other: &Tensor<T>, _device_id: usize) -> RusTorchResult<Self> {
        // Use actual Metal GPU hardware acceleration
        use crate::gpu::metal_kernels::metal_matmul_f32;
        
        // Convert to f32 for Metal kernel
        let a_data = self.data.iter().map(|&x| x.to_f32().unwrap()).collect::<Vec<f32>>();
        let b_data = other.data.iter().map(|&x| x.to_f32().unwrap()).collect::<Vec<f32>>();
        let a_shape = self.data.shape();
        let b_shape = other.data.shape();
        
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "matmul_metal".to_string(),
                message: "Only 2D matrix multiplication supported".to_string(),
            });
        }
        
        let (m, k) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);
        
        if k != k2 {
            return Err(RusTorchError::InvalidOperation {
                operation: "matmul_metal".to_string(),
                message: "Matrix dimensions don't match for multiplication".to_string(),
            });
        }
        
        let mut c_data = vec![0.0f32; m * n];
        
        // Call actual Metal GPU implementation
        metal_matmul_f32(&a_data, &b_data, &mut c_data, m, n, k)
            .map_err(|e| RusTorchError::InvalidOperation {
                operation: "matmul_metal".to_string(),
                message: format!("Metal matmul failed: {}", e),
            })?;
        
        // Convert result back to tensor
        let result_data: Vec<T> = c_data.into_iter()
            .map(|x| T::from_f32(x).unwrap())
            .collect();
        
        let result_array = ndarray::Array::from_shape_vec((m, n), result_data)
            .map_err(|e| RusTorchError::InvalidOperation {
                operation: "matmul_metal".to_string(),
                message: format!("Failed to create result array: {}", e),
            })?;
        
        Ok(Tensor {
            data: result_array.into_dyn(),
            device: self.device.clone(),
            requires_grad: self.requires_grad || other.requires_grad,
        })
    }
}

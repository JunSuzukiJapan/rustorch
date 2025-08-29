//! Utility operations for tensors
//! テンソルのユーティリティ操作

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;
use rand::prelude::*;
use rand_distr::StandardNormal;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Get the batch size (first dimension)
    /// バッチサイズを取得（最初の次元）
    pub fn batch_size(&self) -> usize {
        self.shape().get(0).copied().unwrap_or(1)
    }

    /// Apply function to each element
    /// 各要素に関数を適用
    pub fn map<F>(&self, f: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
    {
        let mapped_data: Vec<T> = self.data.iter().map(|&x| f(x)).collect();
        Tensor::from_vec(mapped_data, self.shape().to_vec())
    }

    /// Apply function to each element in-place
    /// 各要素に関数をin-placeで適用
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        for element in &mut self.data {
            *element = f(*element);
        }
    }

    /// Stack tensors along a new axis (new implementation)
    /// 新しい軸に沿ってテンソルを積み重ね（新実装）
    pub fn stack(tensors: &[&Tensor<T>]) -> RusTorchResult<Tensor<T>> {
        if tensors.is_empty() {
            return Err(RusTorchError::InvalidOperation {
                operation: "stack".to_string(),
                message: "Cannot stack empty list of tensors".to_string(),
            });
        }

        let first_shape = tensors[0].shape();

        // Verify all tensors have the same shape
        for tensor in tensors.iter().skip(1) {
            if tensor.shape() != first_shape {
                return Err(RusTorchError::shape_mismatch(first_shape, tensor.shape()));
            }
        }

        let num_tensors = tensors.len();
        let tensor_size: usize = first_shape.iter().product();
        let mut stacked_data = Vec::with_capacity(num_tensors * tensor_size);

        for tensor in tensors {
            stacked_data.extend(tensor.data.iter().copied());
        }

        let mut new_shape = vec![num_tensors];
        new_shape.extend_from_slice(first_shape);

        Ok(Tensor::from_vec(stacked_data, new_shape))
    }

    /// Concatenate tensors along an existing axis (new implementation)
    /// 既存の軸に沿ってテンソルを連結（新実装）
    pub fn concatenate(tensors: &[&Tensor<T>], axis: usize) -> RusTorchResult<Tensor<T>> {
        if tensors.is_empty() {
            return Err(RusTorchError::InvalidOperation {
                operation: "concatenate".to_string(),
                message: "Cannot concatenate empty list of tensors".to_string(),
            });
        }

        let first_shape = tensors[0].shape();

        if axis >= first_shape.len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "concatenate".to_string(),
                message: format!(
                    "Axis {} is out of bounds for tensor with {} dimensions",
                    axis,
                    first_shape.len()
                ),
            });
        }

        // Verify all tensors have compatible shapes (same except along concat axis)
        let mut total_axis_size = first_shape[axis];
        for tensor in tensors.iter().skip(1) {
            let shape = tensor.shape();
            if shape.len() != first_shape.len() {
                return Err(RusTorchError::InvalidOperation {
                    operation: "concatenate".to_string(),
                    message: "All tensors must have the same number of dimensions".to_string(),
                });
            }

            for (i, (&dim1, &dim2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i != axis && dim1 != dim2 {
                    return Err(RusTorchError::InvalidOperation {
                        operation: "concatenate".to_string(),
                        message: format!("Dimension mismatch at axis {}: {} vs {}", i, dim1, dim2),
                    });
                }
            }

            total_axis_size += shape[axis];
        }

        // Create result shape
        let mut result_shape = first_shape.to_vec();
        result_shape[axis] = total_axis_size;

        // For simplicity, handle only the 1D and 2D cases efficiently
        match first_shape.len() {
            1 => {
                let mut result_data = Vec::new();
                for tensor in tensors {
                    result_data.extend(tensor.data.iter().copied());
                }
                Ok(Tensor::from_vec(result_data, result_shape))
            }
            2 => {
                if axis == 0 {
                    // Concatenate along rows
                    let mut result_data = Vec::new();
                    for tensor in tensors {
                        result_data.extend(tensor.data.iter().copied());
                    }
                    Ok(Tensor::from_vec(result_data, result_shape))
                } else {
                    // Concatenate along columns - more complex
                    let _cols = first_shape[1];
                    let mut result_data = Vec::new();

                    for row in 0..first_shape[0] {
                        for tensor in tensors {
                            let start_idx = row * tensor.shape()[1];
                            let end_idx = start_idx + tensor.shape()[1];
                            let tensor_vec: Vec<T> = tensor.data.iter().copied().collect();
                            result_data.extend_from_slice(&tensor_vec[start_idx..end_idx]);
                        }
                    }

                    Ok(Tensor::from_vec(result_data, result_shape))
                }
            }
            _ => Err(RusTorchError::UnsupportedOperation(
                "Concatenation for >2D tensors not yet implemented".to_string(),
            )),
        }
    }

    /// Clone tensor data (explicit for clarity)
    /// テンソルデータのクローン（明示的）
    pub fn clone_tensor(&self) -> Self {
        Tensor::from_vec(self.data.iter().copied().collect(), self.shape().to_vec())
    }

    /// Create a tensor with random values from standard normal distribution
    /// 標準正規分布からランダムな値を持つテンソルを作成
    pub fn randn(shape: &[usize]) -> Tensor<T>
    where
        StandardNormal: Distribution<T>,
    {
        let size: usize = shape.iter().product();
        let mut rng = thread_rng();
        let data: Vec<T> = (0..size).map(|_| rng.sample(StandardNormal)).collect();
        Tensor::from_vec(data, shape.to_vec())
    }

    /// Create a tensor with random values from uniform distribution [0, 1)
    /// 一様分布[0, 1)からランダムな値を持つテンソルを作成
    pub fn rand(shape: &[usize]) -> Tensor<T> {
        let size: usize = shape.iter().product();
        let mut rng = thread_rng();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.gen::<f64>()).unwrap_or(T::zero()))
            .collect();
        Tensor::from_vec(data, shape.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let result = Tensor::stack(&[&a, &b]).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concatenate_1d() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let result = Tensor::concatenate(&[&a, &b], 0).unwrap();

        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let reshaped = tensor.reshape(&[3, 2]).unwrap();

        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(
            reshaped.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[3]);

        let unsqueezed = squeezed.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 3]);
    }

    #[test]
    fn test_map() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let result = tensor.map(|x| x * 2.0);

        assert_eq!(result.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }
}

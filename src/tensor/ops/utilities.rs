//! Utility operations for tensors
//! テンソルのユーティリティ操作

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;
use rand::prelude::*;
use rand_distr::StandardNormal;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Create tensor with random values from normal distribution (new implementation)
    /// 正規分布からランダム値でテンソルを作成（新実装）
    pub fn randn_v2(shape: &[usize]) -> Tensor<T>
    where
        T: From<f32>,
    {
        let mut rng = thread_rng();
        let total_size: usize = shape.iter().product();
        let data: Vec<T> = (0..total_size)
            .map(|_| <T as From<f32>>::from(rng.sample::<f32, _>(StandardNormal)))
            .collect();

        Tensor::from_vec(data, shape.to_vec())
    }

    /// Create tensor with random uniform values [0, 1) (new implementation)
    /// 一様分布[0, 1)からランダム値でテンソルを作成（新実装）
    pub fn rand_v2(shape: &[usize]) -> Tensor<T>
    where
        T: From<f32>,
    {
        let mut rng = thread_rng();
        let total_size: usize = shape.iter().product();
        let data: Vec<T> = (0..total_size)
            .map(|_| <T as From<f32>>::from(rng.gen::<f32>()))
            .collect();

        Tensor::from_vec(data, shape.to_vec())
    }

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
    pub fn stack_v2(tensors: &[&Tensor<T>]) -> RusTorchResult<Tensor<T>> {
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
    pub fn concatenate_v2(tensors: &[&Tensor<T>], axis: usize) -> RusTorchResult<Tensor<T>> {
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

    /// Reshape tensor (must preserve total number of elements)
    /// テンソルの形状変更（総要素数は保持）
    pub fn reshape(&self, new_shape: &[usize]) -> RusTorchResult<Self> {
        let old_size: usize = self.shape().iter().product();
        let new_size: usize = new_shape.iter().product();

        if old_size != new_size {
            return Err(RusTorchError::InvalidOperation {
                operation: "reshape".to_string(),
                message: format!(
                    "Cannot reshape tensor of size {} to size {}",
                    old_size, new_size
                ),
            });
        }

        Ok(Tensor::from_vec(
            self.data.iter().copied().collect(),
            new_shape.to_vec(),
        ))
    }

    /// Flatten tensor to 1D (new implementation)
    /// テンソルを1Dに平坦化（新実装）
    pub fn flatten_v2(&self) -> Self {
        let total_size: usize = self.shape().iter().product();
        Tensor::from_vec(self.data.iter().copied().collect(), vec![total_size])
    }

    /// Squeeze dimensions of size 1 (new implementation)
    /// サイズ1の次元を削除（新実装）
    pub fn squeeze_v2(&self) -> Self {
        let new_shape: Vec<usize> = self
            .shape()
            .iter()
            .filter(|&&dim| dim != 1)
            .copied()
            .collect();

        if new_shape.is_empty() {
            // If all dimensions were 1, keep at least one dimension
            Tensor::from_vec(self.data.iter().copied().collect(), vec![1])
        } else {
            Tensor::from_vec(self.data.iter().copied().collect(), new_shape)
        }
    }

    /// Add dimension of size 1 at specified position (new implementation)
    /// 指定位置にサイズ1の次元を追加（新実装）
    pub fn unsqueeze_v2(&self, dim: usize) -> RusTorchResult<Self> {
        let mut new_shape = self.shape().to_vec();

        if dim > new_shape.len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "unsqueeze".to_string(),
                message: format!("Dimension {} is out of bounds for unsqueeze", dim),
            });
        }

        new_shape.insert(dim, 1);
        Ok(Tensor::from_vec(
            self.data.iter().copied().collect(),
            new_shape,
        ))
    }

    /// Clone tensor data (explicit for clarity)
    /// テンソルデータのクローン（明示的）
    pub fn clone_tensor(&self) -> Self {
        Tensor::from_vec(self.data.iter().copied().collect(), self.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randn() {
        let tensor = Tensor::<f32>::randn_v2(&[2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data.len(), 6);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let result = Tensor::stack_v2(&[&a, &b]).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_concatenate_1d() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let result = Tensor::concatenate_v2(&[&a, &b], 0).unwrap();

        assert_eq!(result.shape(), &[4]);
        assert_eq!(result.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let reshaped = tensor.reshape_v2(&[3, 2]).unwrap();

        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(
            reshaped.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let squeezed = tensor.squeeze_v2();
        assert_eq!(squeezed.shape(), &[3]);

        let unsqueezed = squeezed.unsqueeze_v2(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 3]);
    }

    #[test]
    fn test_map() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let result = tensor.map(|x| x * 2.0);

        assert_eq!(result.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }
}

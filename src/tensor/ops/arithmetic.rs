//! Basic arithmetic operations for tensors
//! テンソルの基本算術演算

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Element-wise addition with broadcasting support
    /// ブロードキャスト対応の要素ごとの加算
    pub fn add_v2(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        let self_shape = self.shape();
        let other_shape = other.shape();

        // Check if shapes are identical (fastest path)
        if self_shape == other_shape {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            return Ok(Tensor::from_vec(result_data, self_shape.to_vec()));
        }

        // Handle broadcasting cases
        let result_shape = self.broadcast_shapes(self_shape, other_shape)?;
        let total_elements: usize = result_shape.iter().product();
        let mut result_data = vec![T::zero(); total_elements];

        for i in 0..total_elements {
            let self_idx = self.broadcast_index(i, &result_shape, self_shape);
            let other_idx = self.broadcast_index(i, &result_shape, other_shape);

            let self_val = self
                .data
                .as_slice()
                .map(|s| s.get(self_idx).copied().unwrap_or(T::zero()))
                .unwrap_or(T::zero());
            let other_val = other
                .data
                .as_slice()
                .map(|s| s.get(other_idx).copied().unwrap_or(T::zero()))
                .unwrap_or(T::zero());

            result_data[i] = self_val + other_val;
        }

        Ok(Tensor::from_vec(result_data, result_shape))
    }

    /// Helper function to compute broadcast compatible shape
    /// ブロードキャスト互換形状を計算するヘルパー関数
    fn broadcast_shapes(&self, shape1: &[usize], shape2: &[usize]) -> RusTorchResult<Vec<usize>> {
        let max_dims = shape1.len().max(shape2.len());
        let mut result_shape = vec![1; max_dims];

        for i in 0..max_dims {
            let dim1 = shape1
                .get(shape1.len().wrapping_sub(max_dims - i))
                .copied()
                .unwrap_or(1);
            let dim2 = shape2
                .get(shape2.len().wrapping_sub(max_dims - i))
                .copied()
                .unwrap_or(1);

            if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
                result_shape[i] = dim1.max(dim2);
            } else {
                return Err(RusTorchError::InvalidOperation {
                    operation: "broadcasting".to_string(),
                    message: format!("Cannot broadcast shapes {:?} and {:?}", shape1, shape2),
                });
            }
        }

        Ok(result_shape)
    }

    /// Helper function to map linear index to broadcasted index
    /// 線形インデックスをブロードキャストインデックスにマップするヘルパー関数
    fn broadcast_index(
        &self,
        linear_idx: usize,
        result_shape: &[usize],
        original_shape: &[usize],
    ) -> usize {
        let mut idx = 0;
        let mut remaining = linear_idx;

        for i in (0..result_shape.len()).rev() {
            let stride = result_shape.iter().skip(i + 1).product::<usize>();
            let coord = remaining / stride;
            remaining %= stride;

            let dim_size = original_shape
                .get(original_shape.len().wrapping_sub(result_shape.len() - i))
                .copied()
                .unwrap_or(1);
            if dim_size > 1 {
                let original_stride = if original_shape.len() > result_shape.len() - i {
                    original_shape
                        .iter()
                        .skip(original_shape.len().wrapping_sub(result_shape.len() - i) + 1)
                        .product::<usize>()
                } else {
                    1
                };
                idx += (coord % dim_size) * original_stride;
            }
        }

        idx
    }

    /// Element-wise subtraction with another tensor (new implementation)
    /// 他のテンソルとの要素ごとの減算（新実装）
    pub fn sub_v2(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }

    /// Element-wise multiplication with another tensor (new implementation)
    /// 他のテンソルとの要素ごとの乗算（新実装）
    pub fn mul_v2(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }

    /// Element-wise division with another tensor (new implementation)
    /// 他のテンソルとの要素ごとの除算（新実装）
    pub fn div_v2(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| {
                if b == T::zero() {
                    T::infinity() // Handle division by zero
                } else {
                    a / b
                }
            })
            .collect();

        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }

    /// Add scalar to all elements (new implementation)
    /// 全要素にスカラーを加算（新実装）
    pub fn add_scalar_v2(&self, scalar: T) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x + scalar).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Subtract scalar from all elements (new implementation)
    /// 全要素からスカラーを減算（新実装）
    pub fn sub_scalar_v2(&self, scalar: T) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x - scalar).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Multiply all elements by scalar (new implementation)
    /// 全要素をスカラーで乗算（新実装）
    pub fn mul_scalar_v2(&self, scalar: T) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Divide all elements by scalar (new implementation)
    /// 全要素をスカラーで除算（新実装）
    pub fn div_scalar_v2(&self, scalar: T) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x / scalar).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Negate all elements (new implementation)
    /// 全要素の符号を反転（新実装）
    pub fn neg_v2(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| -x).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Element-wise maximum with another tensor (new implementation)
    /// 他のテンソルとの要素ごとの最大値（新実装）
    pub fn maximum_v2(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a > b { a } else { b })
            .collect();

        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }

    /// Element-wise minimum with another tensor (new implementation)
    /// 他のテンソルとの要素ごとの最小値（新実装）
    pub fn minimum_v2(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a < b { a } else { b })
            .collect();

        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);
        let result = a.add_v2(&b).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let result = a.add_scalar_v2(10.0);
        assert_eq!(result.as_slice().unwrap(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let result = a.add_v2(&b);
        assert!(result.is_err());
    }
}

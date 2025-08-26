//! Basic arithmetic operations for tensors
//! テンソルの基本算術演算

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Element-wise addition with another tensor (new implementation)
    /// 他のテンソルとの要素ごとの加算（新実装）
    pub fn add_v2(&self, other: &Tensor<T>) -> RusTorchResult<Self> {
        if self.shape() != other.shape() {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        // Direct element-wise addition
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Ok(Tensor::from_vec(result_data, self.shape().to_vec()))
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

//! Basic arithmetic operations for tensors
//! テンソルの基本算術演算

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Element-wise addition with another tensor
    /// 他のテンソルとの要素ごとの加算
    pub fn add(&self, other: &Tensor<T>) -> RusTorchResult<Self>
    where
        T: ndarray::ScalarOperand + Copy,
    {
        // Same shape - direct element-wise addition
        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            return Ok(Tensor::from_vec(result_data, self.shape().to_vec()));
        }

        // Broadcasting addition using existing broadcast functionality
        if !self.can_broadcast_with(other) {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let (broadcasted_self, broadcasted_other) = self.broadcast_with(other)?;
        let result_data: Vec<T> = broadcasted_self
            .data
            .iter()
            .zip(broadcasted_other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            broadcasted_self.shape().to_vec(),
        ))
    }

    /// Element-wise subtraction with another tensor
    /// 他のテンソルとの要素ごとの減算
    pub fn sub(&self, other: &Tensor<T>) -> RusTorchResult<Self>
    where
        T: ndarray::ScalarOperand + Copy,
    {
        // Same shape - direct element-wise subtraction
        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect();
            return Ok(Tensor::from_vec(result_data, self.shape().to_vec()));
        }

        // Broadcasting subtraction using existing broadcast functionality
        if !self.can_broadcast_with(other) {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let (broadcasted_self, broadcasted_other) = self.broadcast_with(other)?;
        let result_data: Vec<T> = broadcasted_self
            .data
            .iter()
            .zip(broadcasted_other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            broadcasted_self.shape().to_vec(),
        ))
    }

    /// Element-wise multiplication with another tensor
    /// 他のテンソルとの要素ごとの乗算
    pub fn mul(&self, other: &Tensor<T>) -> RusTorchResult<Self>
    where
        T: ndarray::ScalarOperand + Copy,
    {
        // Same shape - direct element-wise multiplication
        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .collect();
            return Ok(Tensor::from_vec(result_data, self.shape().to_vec()));
        }

        // Broadcasting multiplication using existing broadcast functionality
        if !self.can_broadcast_with(other) {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let (broadcasted_self, broadcasted_other) = self.broadcast_with(other)?;
        let result_data: Vec<T> = broadcasted_self
            .data
            .iter()
            .zip(broadcasted_other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            broadcasted_self.shape().to_vec(),
        ))
    }

    /// Element-wise division with another tensor
    /// 他のテンソルとの要素ごとの除算
    pub fn div(&self, other: &Tensor<T>) -> RusTorchResult<Self>
    where
        T: ndarray::ScalarOperand + Copy,
    {
        // Same shape - direct element-wise division
        if self.shape() == other.shape() {
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
            return Ok(Tensor::from_vec(result_data, self.shape().to_vec()));
        }

        // Broadcasting division using existing broadcast functionality
        if !self.can_broadcast_with(other) {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let (broadcasted_self, broadcasted_other) = self.broadcast_with(other)?;
        let result_data: Vec<T> = broadcasted_self
            .data
            .iter()
            .zip(broadcasted_other.data.iter())
            .map(
                |(&a, &b)| {
                    if b == T::zero() {
                        T::infinity()
                    } else {
                        a / b
                    }
                },
            )
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            broadcasted_self.shape().to_vec(),
        ))
    }

    /// Add scalar to all elements (new implementation)
    /// 全要素にスカラーを加算（新実装）
    pub fn add_scalar(&self, scalar: T) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x + scalar).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Subtract scalar from all elements (new implementation)
    /// 全要素からスカラーを減算（新実装）
    pub fn sub_scalar(&self, scalar: T) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x - scalar).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Multiply all elements by scalar (new implementation)
    /// 全要素をスカラーで乗算（新実装）
    pub fn mul_scalar(&self, scalar: T) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x * scalar).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Divide all elements by scalar (new implementation)
    /// 全要素をスカラーで除算（新実装）
    pub fn div_scalar(&self, scalar: T) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| x / scalar).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Negate all elements (new implementation)
    /// 全要素の符号を反転（新実装）
    pub fn neg(&self) -> Self {
        let result_data: Vec<T> = self.data.iter().map(|&x| -x).collect();
        Tensor::from_vec(result_data, self.shape().to_vec())
    }

    /// Element-wise maximum with another tensor (new implementation)
    /// 他のテンソルとの要素ごとの最大値（新実装）
    pub fn maximum(&self, other: &Tensor<T>) -> RusTorchResult<Self>
    where
        T: ndarray::ScalarOperand + Copy,
    {
        // Same shape - direct element-wise maximum
        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| if a > b { a } else { b })
                .collect();
            return Ok(Tensor::from_vec(result_data, self.shape().to_vec()));
        }

        // Broadcasting maximum using existing broadcast functionality
        if !self.can_broadcast_with(other) {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let (broadcasted_self, broadcasted_other) = self.broadcast_with(other)?;
        let result_data: Vec<T> = broadcasted_self
            .data
            .iter()
            .zip(broadcasted_other.data.iter())
            .map(|(&a, &b)| if a > b { a } else { b })
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            broadcasted_self.shape().to_vec(),
        ))
    }

    /// Element-wise minimum with another tensor (new implementation)
    /// 他のテンソルとの要素ごとの最小値（新実装）
    pub fn minimum(&self, other: &Tensor<T>) -> RusTorchResult<Self>
    where
        T: ndarray::ScalarOperand + Copy,
    {
        // Same shape - direct element-wise minimum
        if self.shape() == other.shape() {
            let result_data: Vec<T> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| if a < b { a } else { b })
                .collect();
            return Ok(Tensor::from_vec(result_data, self.shape().to_vec()));
        }

        // Broadcasting minimum using existing broadcast functionality
        if !self.can_broadcast_with(other) {
            return Err(RusTorchError::shape_mismatch(self.shape(), other.shape()));
        }

        let (broadcasted_self, broadcasted_other) = self.broadcast_with(other)?;
        let result_data: Vec<T> = broadcasted_self
            .data
            .iter()
            .zip(broadcasted_other.data.iter())
            .map(|(&a, &b)| if a < b { a } else { b })
            .collect();

        Ok(Tensor::from_vec(
            result_data,
            broadcasted_self.shape().to_vec(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]);
        let result = a.add(&b).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_scalar() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let result = a.add_scalar(10.0);
        assert_eq!(result.as_slice().unwrap(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let result = a.add(&b);
        assert!(result.is_err());
    }
}

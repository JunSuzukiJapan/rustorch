//! Basic arithmetic operations for tensors
//! テンソルの基本算術演算
//!
//! This module provides basic arithmetic operations including element-wise operations,
//! scalar operations, and broadcasting support.
//! このモジュールは要素ごとの操作、スカラー操作、ブロードキャスト対応を含む基本算術演算を提供します。

use crate::tensor::Tensor;
use num_traits::Float;
use std::ops;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Element-wise addition with another tensor (supports broadcasting).
    /// 別のテンソルとの要素ごとの加算（ブロードキャスト対応）
    pub fn add(&self, other: &Tensor<T>) -> Result<Self, String> {
        // Check if shapes are compatible for broadcasting
        if self.shape() == other.shape() {
            // Direct addition when shapes match
            let result_data = self.as_array() + other.as_array();
            Ok(Tensor::new(result_data))
        } else if self.can_broadcast_with(other) {
            // Simple broadcasting for common case: (N, M) + (1, M)
            let self_shape = self.shape();
            let other_shape = other.shape();
            
            // Case 1: (N, M) + (1, M) - bias addition pattern
            if self_shape.len() == 2 && other_shape.len() == 2 && 
               self_shape[1] == other_shape[1] && other_shape[0] == 1 {
                let mut result_data = Vec::with_capacity(self.data.len());
                let other_row = other.data.as_slice().unwrap();
                let self_data = self.data.as_slice().unwrap();
                
                for i in 0..self_shape[0] {
                    for j in 0..self_shape[1] {
                        let self_idx = i * self_shape[1] + j;
                        result_data.push(self_data[self_idx] + other_row[j]);
                    }
                }
                
                Ok(Tensor::from_vec(result_data, self_shape.to_vec()))
            }
            // Case 2: (N, M) + (M,) - vector addition
            else if self_shape.len() == 2 && other_shape.len() == 1 && 
                    self_shape[1] == other_shape[0] {
                let mut result_data = Vec::with_capacity(self.data.len());
                let other_vec = other.data.as_slice().unwrap();
                let self_data = self.data.as_slice().unwrap();
                
                for i in 0..self_shape[0] {
                    for j in 0..self_shape[1] {
                        let self_idx = i * self_shape[1] + j;
                        result_data.push(self_data[self_idx] + other_vec[j]);
                    }
                }
                
                Ok(Tensor::from_vec(result_data, self_shape.to_vec()))
            } else {
                Err(format!("Broadcasting not implemented for shapes: {:?} vs {:?}", self_shape, other_shape))
            }
        } else {
            Err(format!("Incompatible shapes for addition: {:?} vs {:?}", self.shape(), other.shape()))
        }
    }

    /// Element-wise subtraction with another tensor.
    /// 別のテンソルとの要素ごとの減算
    pub fn sub(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err("Shape mismatch for subtraction".to_string());
        }
        
        let result_data = self.as_array() - other.as_array();
        Ok(Tensor::new(result_data))
    }

    /// Element-wise multiplication with another tensor (supports broadcasting).
    /// 別のテンソルとの要素ごとの乗算（ブロードキャスト対応）
    pub fn mul(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() == other.shape() {
            // Direct multiplication when shapes match
            let result_data = self.as_array() * other.as_array();
            Ok(Tensor::new(result_data))
        } else if self.can_broadcast_with(other) {
            // Simple broadcasting for common cases
            let self_shape = self.shape();
            let other_shape = other.shape();
            
            // Case 1: (N, M) * (1, M)
            if self_shape.len() == 2 && other_shape.len() == 2 && 
               self_shape[1] == other_shape[1] && other_shape[0] == 1 {
                let mut result_data = Vec::with_capacity(self.data.len());
                let other_row = other.data.as_slice().unwrap();
                let self_data = self.data.as_slice().unwrap();
                
                for i in 0..self_shape[0] {
                    for j in 0..self_shape[1] {
                        let self_idx = i * self_shape[1] + j;
                        result_data.push(self_data[self_idx] * other_row[j]);
                    }
                }
                
                Ok(Tensor::from_vec(result_data, self_shape.to_vec()))
            }
            // Case 2: (N, M) * (M,)
            else if self_shape.len() == 2 && other_shape.len() == 1 && 
                    self_shape[1] == other_shape[0] {
                let mut result_data = Vec::with_capacity(self.data.len());
                let other_vec = other.data.as_slice().unwrap();
                let self_data = self.data.as_slice().unwrap();
                
                for i in 0..self_shape[0] {
                    for j in 0..self_shape[1] {
                        let self_idx = i * self_shape[1] + j;
                        result_data.push(self_data[self_idx] * other_vec[j]);
                    }
                }
                
                Ok(Tensor::from_vec(result_data, self_shape.to_vec()))
            } else {
                Err(format!("Broadcasting not implemented for shapes: {:?} vs {:?}", self_shape, other_shape))
            }
        } else {
            Err(format!("Incompatible shapes for multiplication: {:?} vs {:?}", self.shape(), other.shape()))
        }
    }

    /// Element-wise division with another tensor.
    /// 別のテンソルとの要素ごとの除算
    pub fn div(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err("Shape mismatch for division".to_string());
        }
        
        let result_data = self.as_array() / other.as_array();
        Ok(Tensor::new(result_data))
    }

    /// Scalar addition.
    /// スカラー加算
    pub fn add_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() + scalar;
        Tensor::new(result_data)
    }

    /// Scalar subtraction.
    /// スカラー減算
    pub fn sub_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() - scalar;
        Tensor::new(result_data)
    }

    /// Scalar multiplication.
    /// スカラー乗算
    pub fn mul_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() * scalar;
        Tensor::new(result_data)
    }

    /// Scalar division.
    /// スカラー除算
    pub fn div_scalar(&self, scalar: T) -> Self {
        let result_data = self.as_array() / scalar;
        Tensor::new(result_data)
    }

    /// Negation.
    /// 符号反転
    pub fn neg(&self) -> Self {
        let result_data = self.as_array().mapv(|x| -x);
        Tensor::new(result_data)
    }

    /// Element-wise maximum with another tensor
    /// 別のテンソルとの要素ごとの最大値
    pub fn maximum(&self, other: &Tensor<T>) -> Result<Self, String> {
        if self.shape() != other.shape() {
            return Err("Shape mismatch for maximum operation".to_string());
        }
        
        let max_data: Vec<T> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| if a > b { a } else { b })
            .collect();
            
        Ok(Tensor::from_vec(max_data, self.shape().to_vec()))
    }

    // Note: stack function moved to utils.rs to avoid conflicts with operations.rs
    // 注意: stack関数は operations.rs との競合を避けるため utils.rs に移動

    /// Helper function to check if two tensors can be broadcast together
    /// 二つのテンソルがブロードキャスト可能かチェックするヘルパー関数
    pub(crate) fn can_broadcast_with(&self, other: &Tensor<T>) -> bool {
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        // Simple broadcasting rules implementation
        if self_shape.len() != other_shape.len() {
            // Allow broadcasting between (N, M) and (M,)
            if self_shape.len() == 2 && other_shape.len() == 1 {
                return self_shape[1] == other_shape[0];
            }
            return false;
        }
        
        // Same number of dimensions - check dimension compatibility
        for (&self_dim, &other_dim) in self_shape.iter().zip(other_shape.iter()) {
            if self_dim != other_dim && self_dim != 1 && other_dim != 1 {
                return false;
            }
        }
        
        true
    }
}

// Note: Operator overloading implementations are kept in the legacy operations.rs
// to avoid conflicts. This module focuses on providing method-based operations.
// 注意: 演算子オーバーロード実装は競合を避けるためレガシーoperations.rsに保持されています。
// このモジュールはメソッドベースの操作の提供に焦点を当てます。

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_addition() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], vec![3]);
        
        let result = a.add(&b).unwrap();
        let expected = vec![5.0f32, 7.0, 9.0];
        
        assert_eq!(result.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_scalar_operations() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]);
        
        let add_result = tensor.add_scalar(10.0);
        assert_eq!(add_result.as_slice().unwrap(), &[11.0f32, 12.0, 13.0]);
        
        let mul_result = tensor.mul_scalar(2.0);
        assert_eq!(mul_result.as_slice().unwrap(), &[2.0f32, 4.0, 6.0]);
    }

    #[test]
    fn test_broadcasting() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![10.0f32, 20.0], vec![1, 2]);
        
        let result = a.add(&b).unwrap();
        let expected = vec![11.0f32, 22.0, 13.0, 24.0];
        
        assert_eq!(result.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_maximum() {
        let a = Tensor::from_vec(vec![1.0f32, 5.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![2.0f32, 4.0, 6.0], vec![3]);
        
        let result = a.maximum(&b).unwrap();
        let expected = vec![2.0f32, 5.0, 6.0];
        
        assert_eq!(result.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0f32, 4.0], vec![2]);
        let c = Tensor::from_vec(vec![5.0f32, 6.0], vec![2]);
        
        let result = Tensor::stack(&[&a, &b, &c]).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        
        let expected = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(result.as_slice().unwrap(), &expected);
    }
}
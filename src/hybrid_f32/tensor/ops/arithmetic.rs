//! F32Tensor 算術演算
//! F32Tensor arithmetic operations

use crate::error::{RusTorchResult, RusTorchError};
use crate::hybrid_f32::tensor::core::F32Tensor;
use crate::hybrid_f32_experimental;
use std::ops::{Add, Sub, Mul, Div, Neg};

// ========================================
// 演算子オーバーロード実装
// Operator overload implementations  
// ========================================

impl Add for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn add(self, other: F32Tensor) -> Self::Output {
        hybrid_f32_experimental!();
        self.add_tensor(&other)
    }
}

impl Add for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn add(self, other: &F32Tensor) -> Self::Output {
        hybrid_f32_experimental!();
        self.add_tensor(other)
    }
}

impl Add<f32> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn add(self, scalar: f32) -> Self::Output {
        hybrid_f32_experimental!();
        self.add_scalar(scalar)
    }
}

impl Add<f32> for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn add(self, scalar: f32) -> Self::Output {
        hybrid_f32_experimental!();
        self.add_scalar(scalar)
    }
}

impl Sub for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn sub(self, other: F32Tensor) -> Self::Output {
        hybrid_f32_experimental!();
        self.sub_tensor(&other)
    }
}

impl Sub for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn sub(self, other: &F32Tensor) -> Self::Output {
        hybrid_f32_experimental!();
        self.sub_tensor(other)
    }
}

impl Sub<f32> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn sub(self, scalar: f32) -> Self::Output {
        hybrid_f32_experimental!();
        self.sub_scalar(scalar)
    }
}

impl Sub<f32> for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn sub(self, scalar: f32) -> Self::Output {
        hybrid_f32_experimental!();
        self.sub_scalar(scalar)
    }
}

impl Mul for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn mul(self, other: F32Tensor) -> Self::Output {
        hybrid_f32_experimental!();
        self.mul_tensor(&other)
    }
}

impl Mul for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn mul(self, other: &F32Tensor) -> Self::Output {
        hybrid_f32_experimental!();
        self.mul_tensor(other)
    }
}

impl Mul<f32> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn mul(self, scalar: f32) -> Self::Output {
        hybrid_f32_experimental!();
        self.mul_scalar(scalar)
    }
}

impl Mul<f32> for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn mul(self, scalar: f32) -> Self::Output {
        hybrid_f32_experimental!();
        self.mul_scalar(scalar)
    }
}

impl Div for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn div(self, other: F32Tensor) -> Self::Output {
        hybrid_f32_experimental!();
        self.div_tensor(&other)
    }
}

impl Div for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn div(self, other: &F32Tensor) -> Self::Output {
        hybrid_f32_experimental!();
        self.div_tensor(other)
    }
}

impl Div<f32> for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn div(self, scalar: f32) -> Self::Output {
        hybrid_f32_experimental!();
        self.div_scalar(scalar)
    }
}

impl Div<f32> for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn div(self, scalar: f32) -> Self::Output {
        hybrid_f32_experimental!();
        self.div_scalar(scalar)
    }
}

impl Neg for F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn neg(self) -> Self::Output {
        hybrid_f32_experimental!();
        self.neg_tensor()
    }
}

impl Neg for &F32Tensor {
    type Output = RusTorchResult<F32Tensor>;

    fn neg(self) -> Self::Output {
        hybrid_f32_experimental!();
        self.neg_tensor()
    }
}

// ========================================
// 内部実装メソッド
// Internal implementation methods
// ========================================

impl F32Tensor {
    /// テンソル加算
    /// Tensor addition
    pub fn add_tensor(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch(
                format!("Cannot add tensors with shapes {:?} and {:?}", 
                        self.shape(), other.shape())
            ));
        }

        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        F32Tensor::new(result_data, self.shape().to_vec())
    }

    /// スカラー加算
    /// Scalar addition
    pub fn add_scalar(&self, scalar: f32) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        let result_data: Vec<f32> = self.data.iter()
            .map(|x| x + scalar)
            .collect();

        F32Tensor::new(result_data, self.shape().to_vec())
    }

    /// テンソル減算
    /// Tensor subtraction
    pub fn sub_tensor(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch(
                format!("Cannot subtract tensors with shapes {:?} and {:?}", 
                        self.shape(), other.shape())
            ));
        }

        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        F32Tensor::new(result_data, self.shape().to_vec())
    }

    /// スカラー減算
    /// Scalar subtraction
    pub fn sub_scalar(&self, scalar: f32) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        let result_data: Vec<f32> = self.data.iter()
            .map(|x| x - scalar)
            .collect();

        F32Tensor::new(result_data, self.shape().to_vec())
    }

    /// テンソル乗算
    /// Tensor multiplication
    pub fn mul_tensor(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch(
                format!("Cannot multiply tensors with shapes {:?} and {:?}", 
                        self.shape(), other.shape())
            ));
        }

        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        F32Tensor::new(result_data, self.shape().to_vec())
    }

    /// スカラー乗算
    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        let result_data: Vec<f32> = self.data.iter()
            .map(|x| x * scalar)
            .collect();

        F32Tensor::new(result_data, self.shape().to_vec())
    }

    /// テンソル除算
    /// Tensor division
    pub fn div_tensor(&self, other: &F32Tensor) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        if self.shape() != other.shape() {
            return Err(RusTorchError::ShapeMismatch(
                format!("Cannot divide tensors with shapes {:?} and {:?}", 
                        self.shape(), other.shape())
            ));
        }

        let result_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| {
                if *b == 0.0 {
                    f32::INFINITY
                } else {
                    a / b
                }
            })
            .collect();

        F32Tensor::new(result_data, self.shape().to_vec())
    }

    /// スカラー除算
    /// Scalar division
    pub fn div_scalar(&self, scalar: f32) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        if scalar == 0.0 {
            return Err(RusTorchError::InvalidOperation(
                "Division by zero".to_string()
            ));
        }

        let result_data: Vec<f32> = self.data.iter()
            .map(|x| x / scalar)
            .collect();

        F32Tensor::new(result_data, self.shape().to_vec())
    }

    /// 負号
    /// Negation
    pub fn neg_tensor(&self) -> RusTorchResult<F32Tensor> {
        hybrid_f32_experimental!();
        
        let result_data: Vec<f32> = self.data.iter()
            .map(|x| -x)
            .collect();

        F32Tensor::new(result_data, self.shape().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_addition() {
        let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0]).unwrap();
        let b = F32Tensor::from_vec(vec![4.0, 5.0, 6.0]).unwrap();
        
        let result = (&a + &b).unwrap();
        let expected = vec![5.0, 7.0, 9.0];
        
        assert_eq!(result.data.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_scalar_addition() {
        let a = F32Tensor::from_vec(vec![1.0, 2.0, 3.0]).unwrap();
        let result = (&a + 10.0).unwrap();
        let expected = vec![11.0, 12.0, 13.0];
        
        assert_eq!(result.data.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_tensor_multiplication() {
        let a = F32Tensor::from_vec(vec![2.0, 3.0, 4.0]).unwrap();
        let b = F32Tensor::from_vec(vec![5.0, 6.0, 7.0]).unwrap();
        
        let result = (&a * &b).unwrap();
        let expected = vec![10.0, 18.0, 28.0];
        
        assert_eq!(result.data.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_tensor_negation() {
        let a = F32Tensor::from_vec(vec![1.0, -2.0, 3.0]).unwrap();
        let result = (-&a).unwrap();
        let expected = vec![-1.0, 2.0, -3.0];
        
        assert_eq!(result.data.as_slice().unwrap(), &expected);
    }
}
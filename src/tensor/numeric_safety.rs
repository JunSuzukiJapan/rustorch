//! Numeric safety and overflow protection
//! 数値安全性とオーバーフロー保護

use num_traits::Float;
use std::fmt::Debug;
use crate::error::{RusTorchError, RusTorchResult};

/// Safe numeric operations with overflow detection
/// オーバーフロー検出付きの安全な数値演算
pub trait SafeNumeric: Float + Debug + Clone {
    /// Safe addition that detects overflow
    /// オーバーフローを検出する安全な加算
    fn safe_add(&self, other: Self) -> RusTorchResult<Self>;
    
    /// Safe multiplication that detects overflow
    /// オーバーフローを検出する安全な乗算
    fn safe_mul(&self, other: Self) -> RusTorchResult<Self>;
    
    /// Safe division with zero check
    /// ゼロチェック付きの安全な除算
    fn safe_div(&self, other: Self) -> RusTorchResult<Self>;
    
    /// Safe exponential with overflow check
    /// オーバーフローチェック付きの安全な指数関数
    fn safe_exp(&self) -> RusTorchResult<Self>;
    
    /// Check if value is finite (not NaN or infinite)
    /// 値が有限か確認（NaNや無限大でない）
    fn is_safe(&self) -> bool;
    
    /// Clamp value to safe range
    /// 値を安全な範囲にクランプ
    fn clamp_safe(&self) -> Self;
}

impl SafeNumeric for f32 {
    fn safe_add(&self, other: Self) -> RusTorchResult<Self> {
        let result = self + other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(RusTorchError::numeric("Addition overflow"))
        }
    }
    
    fn safe_mul(&self, other: Self) -> RusTorchResult<Self> {
        let result = self * other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(RusTorchError::numeric("Multiplication overflow"))
        }
    }
    
    fn safe_div(&self, other: Self) -> RusTorchResult<Self> {
        if other.abs() < Self::epsilon() {
            return Err(RusTorchError::numeric("Division by zero"));
        }
        
        let result = self / other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(RusTorchError::numeric("Division overflow"))
        }
    }
    
    fn safe_exp(&self) -> RusTorchResult<Self> {
        // Check for potential overflow before computing
        if *self > 88.0 {  // exp(88) is close to f32::MAX
            return Err(RusTorchError::numeric("Numeric overflow"));
        }
        
        let result = self.exp();
        if result.is_finite() {
            Ok(result)
        } else {
            Err(RusTorchError::numeric("Exponential overflow"))
        }
    }
    
    fn is_safe(&self) -> bool {
        self.is_finite()
    }
    
    fn clamp_safe(&self) -> Self {
        if self.is_nan() {
            0.0
        } else if *self == Self::infinity() {
            <Self as num_traits::Bounded>::max_value()
        } else if *self == Self::neg_infinity() {
            <Self as num_traits::Bounded>::min_value()
        } else {
            *self
        }
    }
}

impl SafeNumeric for f64 {
    fn safe_add(&self, other: Self) -> RusTorchResult<Self> {
        let result = self + other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(RusTorchError::numeric("Addition overflow"))
        }
    }
    
    fn safe_mul(&self, other: Self) -> RusTorchResult<Self> {
        let result = self * other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(RusTorchError::numeric("Multiplication overflow"))
        }
    }
    
    fn safe_div(&self, other: Self) -> RusTorchResult<Self> {
        if other.abs() < Self::epsilon() {
            return Err(RusTorchError::numeric("Division by zero"));
        }
        
        let result = self / other;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(RusTorchError::numeric("Division overflow"))
        }
    }
    
    fn safe_exp(&self) -> RusTorchResult<Self> {
        // Check for potential overflow before computing
        if *self > 709.0 {  // exp(709) is close to f64::MAX
            return Err(RusTorchError::numeric("Exponential overflow"));
        }
        
        let result = self.exp();
        if result.is_finite() {
            Ok(result)
        } else {
            Err(RusTorchError::numeric("Exponential overflow"))
        }
    }
    
    fn is_safe(&self) -> bool {
        self.is_finite()
    }
    
    fn clamp_safe(&self) -> Self {
        if self.is_nan() {
            0.0
        } else if *self == Self::infinity() {
            <Self as num_traits::Bounded>::max_value()
        } else if *self == Self::neg_infinity() {
            <Self as num_traits::Bounded>::min_value()
        } else {
            *self
        }
    }
}

/// Safe tensor with numeric overflow protection
/// 数値オーバーフロー保護付きの安全テンソル
#[derive(Debug, Clone, PartialEq)]
pub struct SafeTensor<T: SafeNumeric> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T: SafeNumeric> SafeTensor<T> {
    /// Create new safe tensor with validation
    /// 検証付きで新しい安全テンソルを作成
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> RusTorchResult<Self> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(RusTorchError::shape_mismatch(&[data.len()], &[shape.iter().product::<usize>()]));
        }
        
        // Validate all values are safe
        if !data.iter().all(|x| x.is_safe()) {
            return Err(RusTorchError::numeric("Tensor contains unsafe values (NaN or infinity)"));
        }
        
        Ok(Self { data, shape })
    }
    
    /// Get tensor shape
    /// テンソル形状を取得
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get tensor data
    /// テンソルデータを取得
    pub fn data(&self) -> &[T] {
        &self.data
    }
    
    /// Safe element-wise addition
    /// 安全な要素ごとの加算
    pub fn safe_add(&self, other: &Self) -> RusTorchResult<Self> {
        if self.shape != other.shape {
            return Err(RusTorchError::shape_mismatch(&self.shape, &other.shape));
        }
        
        let mut result_data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            result_data.push(a.safe_add(*b)?);
        }
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    /// Safe element-wise multiplication
    /// 安全な要素ごとの乗算
    pub fn safe_mul(&self, other: &Self) -> RusTorchResult<Self> {
        if self.shape != other.shape {
            return Err(RusTorchError::shape_mismatch(&self.shape, &other.shape));
        }
        
        let mut result_data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            result_data.push(a.safe_mul(*b)?);
        }
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    /// Safe scalar multiplication
    /// 安全なスカラー乗算
    pub fn safe_mul_scalar(&self, scalar: T) -> RusTorchResult<Self> {
        let mut result_data = Vec::with_capacity(self.data.len());
        for value in self.data.iter() {
            result_data.push(value.safe_mul(scalar)?);
        }
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    /// Safe exponential function
    /// 安全な指数関数
    pub fn safe_exp(&self) -> RusTorchResult<Self> {
        let mut result_data = Vec::with_capacity(self.data.len());
        for value in self.data.iter() {
            result_data.push(value.safe_exp()?);
        }
        
        Ok(Self {
            data: result_data,
            shape: self.shape.clone(),
        })
    }
    
    /// Clamp all values to safe range
    /// すべての値を安全な範囲にクランプ
    pub fn clamp_safe(&self) -> Self {
        let clamped_data: Vec<T> = self.data.iter()
            .map(|x| x.clamp_safe())
            .collect();
        
        Self {
            data: clamped_data,
            shape: self.shape.clone(),
        }
    }
    
    /// Check if all values are safe
    /// すべての値が安全か確認
    pub fn is_all_safe(&self) -> bool {
        self.data.iter().all(|x| x.is_safe())
    }
    
    /// Get statistics about unsafe values
    /// 安全でない値の統計を取得
    pub fn unsafe_stats(&self) -> UnsafeStats {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut neg_inf_count = 0;
        
        for value in &self.data {
            if value.is_nan() {
                nan_count += 1;
            } else if value.is_infinite() {
                if value.is_sign_positive() {
                    inf_count += 1;
                } else {
                    neg_inf_count += 1;
                }
            }
        }
        
        UnsafeStats {
            total_elements: self.data.len(),
            nan_count,
            inf_count,
            neg_inf_count,
        }
    }
}

/// Statistics about unsafe numeric values
/// 安全でない数値の統計
#[derive(Debug, Clone, PartialEq)]
pub struct UnsafeStats {
    /// Total number of elements in the tensor
    /// テンソル内の要素の総数
    pub total_elements: usize,
    /// Number of NaN (Not-a-Number) values
    /// NaN（非数）値の数
    pub nan_count: usize,
    /// Number of positive infinity values
    /// 正の無限大値の数
    pub inf_count: usize,
    /// Number of negative infinity values
    /// 負の無限大値の数
    pub neg_inf_count: usize,
}

impl UnsafeStats {
    /// Check if tensor has any unsafe values
    /// テンソルに安全でない値があるか確認
    pub fn has_unsafe(&self) -> bool {
        self.nan_count > 0 || self.inf_count > 0 || self.neg_inf_count > 0
    }
    
    /// Get total unsafe value count
    /// 安全でない値の総数を取得
    pub fn unsafe_total(&self) -> usize {
        self.nan_count + self.inf_count + self.neg_inf_count
    }
    
    /// Get percentage of unsafe values
    /// 安全でない値の割合を取得
    pub fn unsafe_percentage(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            (self.unsafe_total() as f64 / self.total_elements as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_operations_f32() {
        let a = 5.0f32;
        let b = 3.0f32;
        
        assert_eq!(a.safe_add(b).unwrap(), 8.0);
        assert_eq!(a.safe_mul(b).unwrap(), 15.0);
        assert_eq!(a.safe_div(b).unwrap(), 5.0/3.0);
    }
    
    #[test]
    fn test_overflow_detection() {
        let large = f32::MAX;
        let result = large.safe_add(large);
        assert!(result.is_err());
        
        if let Err(_) = result {
            // Expected error
        } else {
            panic!("Expected overflow error");
        }
    }
    
    #[test]
    fn test_division_by_zero() {
        let a = 5.0f32;
        let zero = 0.0f32;
        
        let result = a.safe_div(zero);
        assert!(result.is_err());
        
        if let Err(_) = result {
            // Expected error
        } else {
            panic!("Expected division by zero error");
        }
    }
    
    #[test]
    fn test_safe_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        
        let tensor = SafeTensor::new(data, shape).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &[1.0f32, 2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_safe_tensor_operations() {
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![2.0f32, 3.0, 4.0, 5.0];
        let shape = vec![2, 2];
        
        let tensor1 = SafeTensor::new(data1, shape.clone()).unwrap();
        let tensor2 = SafeTensor::new(data2, shape).unwrap();
        
        let sum = tensor1.safe_add(&tensor2).unwrap();
        assert_eq!(sum.data(), &[3.0f32, 5.0, 7.0, 9.0]);
        
        let product = tensor1.safe_mul(&tensor2).unwrap();
        assert_eq!(product.data(), &[2.0f32, 6.0, 12.0, 20.0]);
    }
    
    #[test]
    fn test_unsafe_value_detection() {
        let data = vec![1.0f32, f32::NAN, f32::INFINITY, -f32::INFINITY, 2.0];
        let shape = vec![5];
        
        let tensor = SafeTensor::new(data.clone(), shape);
        assert!(tensor.is_err()); // Should fail due to NaN and infinity
        
        // Create tensor without validation for testing
        let unsafe_tensor = SafeTensor { data, shape: vec![5] };
        let stats = unsafe_tensor.unsafe_stats();
        
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.inf_count, 1);
        assert_eq!(stats.neg_inf_count, 1);
        assert!(stats.has_unsafe());
        assert_eq!(stats.unsafe_percentage(), 60.0); // 3 out of 5
    }
    
    #[test]
    fn test_clamp_safe() {
        let nan = f32::NAN;
        let inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let normal = 5.0f32;
        
        assert_eq!(nan.clamp_safe(), 0.0);
        assert_eq!(inf.clamp_safe(), f32::MAX);
        assert_eq!(neg_inf.clamp_safe(), f32::MIN);
        assert_eq!(normal.clamp_safe(), 5.0);
    }
    
    #[test]
    fn test_exp_overflow_protection() {
        let large_value = 100.0f32; // exp(100) would overflow
        let result = large_value.safe_exp();
        assert!(result.is_err());
        
        let safe_value = 1.0f32;
        let result = safe_value.safe_exp();
        assert!(result.is_ok());
        assert!((result.unwrap() - 1.0f32.exp()).abs() < 1e-6);
    }
}
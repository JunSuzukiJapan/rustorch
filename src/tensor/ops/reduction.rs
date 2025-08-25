//! Reduction and statistical operations for tensors
//! テンソルのリダクション・統計演算
//!
//! This module provides reduction operations that aggregate tensor values along
//! specified dimensions or across the entire tensor.
//! このモジュールは指定次元または全テンソルにわたってテンソル値を集約するリダクション操作を提供します。

use crate::tensor::Tensor;
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Sum all elements in the tensor
    /// テンソル内の全要素の合計
    pub fn sum(&self) -> T {
        self.as_array().sum()
    }

    /// Calculate mean of all elements
    /// 全要素の平均を計算
    pub fn mean(&self) -> T {
        let sum = self.sum();
        let count = T::from(self.numel()).unwrap_or(T::one());
        sum / count
    }

    /// Get tensor as a scalar value (tensor must have exactly one element)
    /// テンソルをスカラー値として取得（テンソルは正確に1つの要素を持つ必要がある）
    pub fn item(&self) -> T {
        if self.numel() != 1 {
            panic!("Tensor must have exactly one element to use item(), but has {}", self.numel());
        }
        
        if let Some(slice) = self.as_slice() {
            slice[0]
        } else {
            // Handle non-contiguous tensor
            *self.data.iter().next().unwrap()
        }
    }

    /// Sum along specified axis
    /// 指定軸に沿って合計
    pub fn sum_axis(&self, axis: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} is out of bounds for tensor with {} dimensions", axis, self.ndim()));
        }
        
        let result = self.data.sum_axis(ndarray::Axis(axis));
        Ok(Tensor::new(result))
    }

    /// Mean along specified axis
    /// 指定軸に沿って平均
    pub fn mean_axis(&self, axis: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} is out of bounds for tensor with {} dimensions", axis, self.ndim()));
        }
        
        let result = self.data.mean_axis(ndarray::Axis(axis));
        match result {
            Some(mean_array) => Ok(Tensor::new(mean_array)),
            None => Err("Mean calculation failed (empty tensor along axis)".to_string())
        }
    }

    /// Find maximum value in tensor
    /// テンソル内の最大値を検索
    pub fn max(&self) -> Option<T> {
        self.data.iter()
            .max_by(|&&a, &&b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }

    /// Find minimum value in tensor
    /// テンソル内の最小値を検索
    pub fn min(&self) -> Option<T> {
        self.data.iter()
            .min_by(|&&a, &&b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }

    /// Find maximum value along specified axis
    /// 指定軸に沿って最大値を検索
    pub fn max_axis(&self, axis: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} is out of bounds for tensor with {} dimensions", axis, self.ndim()));
        }
        
        // Manual implementation since ndarray doesn't have max_axis
        let shape = self.shape();
        let mut new_shape = shape.to_vec();
        new_shape.remove(axis);
        
        if new_shape.is_empty() {
            // Reducing to scalar
            let max_val = self.max().unwrap_or(T::zero());
            return Ok(Tensor::from_vec(vec![max_val], vec![]));
        }
        
        let new_size: usize = new_shape.iter().product();
        let mut result_data = Vec::with_capacity(new_size);
        
        // This is a simplified implementation
        // In practice, you'd want more efficient axis-specific reduction
        let axis_size = shape[axis];
        let stride_before: usize = shape[..axis].iter().product();
        let stride_after: usize = shape[axis+1..].iter().product();
        
        for i in 0..stride_before {
            for k in 0..stride_after {
                let mut max_val: Option<T> = None;
                for j in 0..axis_size {
                    let idx = i * (axis_size * stride_after) + j * stride_after + k;
                    if let Some(slice) = self.as_slice() {
                        if idx < slice.len() {
                            match max_val {
                                None => max_val = Some(slice[idx]),
                                Some(current) => {
                                    if slice[idx] > current {
                                        max_val = Some(slice[idx]);
                                    }
                                }
                            }
                        }
                    }
                }
                    result_data.push(val);
                } else {
                    result_data.push(T::zero()); // Default value if no data found
                }
        }
        
        Ok(Tensor::from_vec(result_data, new_shape))
    }

    /// Find minimum value along specified axis
    /// 指定軸に沿って最小値を検索
    pub fn min_axis(&self, axis: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} is out of bounds for tensor with {} dimensions", axis, self.ndim()));
        }
        
        let shape = self.shape();
        let mut new_shape = shape.to_vec();
        new_shape.remove(axis);
        
        if new_shape.is_empty() {
            // Reducing to scalar
            let min_val = self.min().unwrap_or(T::zero());
            return Ok(Tensor::from_vec(vec![min_val], vec![]));
        }
        
        let new_size: usize = new_shape.iter().product();
        let mut result_data = Vec::with_capacity(new_size);
        
        let axis_size = shape[axis];
        let stride_before: usize = shape[..axis].iter().product();
        let stride_after: usize = shape[axis+1..].iter().product();
        
        for i in 0..stride_before {
            for k in 0..stride_after {
                let mut min_val = T::infinity();
                for j in 0..axis_size {
                    let idx = i * (axis_size * stride_after) + j * stride_after + k;
                    if let Some(slice) = self.as_slice() {
                        if idx < slice.len() {
                            min_val = min_val.min(slice[idx]);
                        }
                    }
                }
                result_data.push(min_val);
            }
        }
        
        Ok(Tensor::from_vec(result_data, new_shape))
    }

    /// Calculate variance of all elements
    /// 全要素の分散を計算
    pub fn var(&self) -> T {
        let mean = self.mean();
        let sum_squared_diff = self.data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x);
        
        let count = T::from(self.numel()).unwrap_or(T::one());
        sum_squared_diff / count
    }

    /// Calculate standard deviation of all elements
    /// 全要素の標準偏差を計算
    pub fn std(&self) -> T {
        self.var().sqrt()
    }

    /// Calculate variance along specified axis
    /// 指定軸に沿って分散を計算
    pub fn var_axis(&self, axis: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} is out of bounds for tensor with {} dimensions", axis, self.ndim()));
        }
        
        let mean_tensor = self.mean_axis(axis)?;
        
        // Calculate squared differences
        let shape = self.shape();
        let axis_size = shape[axis];
        let mut new_shape = shape.to_vec();
        new_shape.remove(axis);
        
        let new_size: usize = new_shape.iter().product();
        let mut var_data = vec![T::zero(); new_size];
        
        let stride_before: usize = shape[..axis].iter().product();
        let stride_after: usize = shape[axis+1..].iter().product();
        let mean_data = mean_tensor.as_slice().unwrap();
        let tensor_data = self.as_slice().unwrap();
        
        for i in 0..stride_before {
            for k in 0..stride_after {
                let mean_idx = i * stride_after + k;
                let mean_val = mean_data[mean_idx];
                let mut sum_sq_diff = T::zero();
                
                for j in 0..axis_size {
                    let tensor_idx = i * (axis_size * stride_after) + j * stride_after + k;
                    let diff = tensor_data[tensor_idx] - mean_val;
                    sum_sq_diff = sum_sq_diff + diff * diff;
                }
                
                var_data[mean_idx] = sum_sq_diff / T::from(axis_size).unwrap();
            }
        }
        
        Ok(Tensor::from_vec(var_data, new_shape))
    }

    /// Calculate standard deviation along specified axis
    /// 指定軸に沿って標準偏差を計算
    pub fn std_axis(&self, axis: usize) -> Result<Self, String> {
        let var_tensor = self.var_axis(axis)?;
        let std_data: Vec<T> = var_tensor.data.iter().map(|&x| x.sqrt()).collect();
        Ok(Tensor::from_vec(std_data, var_tensor.shape().to_vec()))
    }

    /// Count non-zero elements
    /// 非ゼロ要素をカウント
    pub fn count_nonzero(&self) -> usize {
        self.data.iter()
            .filter(|&&x| x != T::zero())
            .count()
    }

    /// Count non-zero elements along specified axis
    /// 指定軸に沿って非ゼロ要素をカウント
    pub fn count_nonzero_axis(&self, axis: usize) -> Result<Tensor<T>, String> {
        if axis >= self.ndim() {
            return Err(format!("Axis {} is out of bounds for tensor with {} dimensions", axis, self.ndim()));
        }
        
        let shape = self.shape();
        let mut new_shape = shape.to_vec();
        new_shape.remove(axis);
        
        let new_size: usize = new_shape.iter().product();
        let mut count_data = vec![T::zero(); new_size];
        
        let axis_size = shape[axis];
        let stride_before: usize = shape[..axis].iter().product();
        let stride_after: usize = shape[axis+1..].iter().product();
        let tensor_data = self.as_slice().unwrap();
        
        for i in 0..stride_before {
            for k in 0..stride_after {
                let count_idx = i * stride_after + k;
                let mut count = 0;
                
                for j in 0..axis_size {
                    let tensor_idx = i * (axis_size * stride_after) + j * stride_after + k;
                    if tensor_data[tensor_idx] != T::zero() {
                        count += 1;
                    }
                }
                
                count_data[count_idx] = T::from(count).unwrap_or(T::zero());
            }
        }
        
        Ok(Tensor::from_vec(count_data, new_shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        assert_eq!(tensor.sum(), 10.0);
    }

    #[test]
    fn test_mean() {
        let tensor = Tensor::from_vec(vec![2.0f32, 4.0, 6.0, 8.0], vec![4]);
        assert_eq!(tensor.mean(), 5.0);
    }

    #[test]
    fn test_item() {
        let scalar = Tensor::from_vec(vec![42.0f32], vec![]);
        assert_eq!(scalar.item(), 42.0);
    }

    #[test]
    #[should_panic(expected = "Tensor must have exactly one element")]
    fn test_item_multiple_elements() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        tensor.item(); // Should panic
    }

    #[test]
    fn test_sum_axis() {
        let tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 
            vec![2, 3]
        );
        
        let sum_axis_0 = tensor.sum_axis(0).unwrap();
        assert_eq!(sum_axis_0.shape(), &[3]);
        assert_eq!(sum_axis_0.as_slice().unwrap(), &[5.0f32, 7.0, 9.0]);
        
        let sum_axis_1 = tensor.sum_axis(1).unwrap();
        assert_eq!(sum_axis_1.shape(), &[2]);
        assert_eq!(sum_axis_1.as_slice().unwrap(), &[6.0f32, 15.0]);
    }

    #[test]
    fn test_mean_axis() {
        let tensor = Tensor::from_vec(
            vec![2.0f32, 4.0, 6.0, 8.0], 
            vec![2, 2]
        );
        
        let mean_axis_0 = tensor.mean_axis(0).unwrap();
        assert_eq!(mean_axis_0.shape(), &[2]);
        assert_eq!(mean_axis_0.as_slice().unwrap(), &[5.0f32, 5.0]);
        
        let mean_axis_1 = tensor.mean_axis(1).unwrap();
        assert_eq!(mean_axis_1.shape(), &[2]);
        assert_eq!(mean_axis_1.as_slice().unwrap(), &[3.0f32, 7.0]);
    }

    #[test]
    fn test_max_min() {
        let tensor = Tensor::from_vec(vec![3.0f32, 1.0, 4.0, 2.0], vec![4]);
        assert_eq!(tensor.max(), Some(4.0));
        assert_eq!(tensor.min(), Some(1.0));
    }

    #[test]
    fn test_var_std() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        
        let var = tensor.var();
        let std = tensor.std();
        
        // Variance of [1,2,3,4,5] with mean=3 is ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²)/5 = 2
        assert!((var - 2.0).abs() < 1e-6);
        assert!((std - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_count_nonzero() {
        let tensor = Tensor::from_vec(vec![1.0f32, 0.0, 3.0, 0.0, 5.0], vec![5]);
        assert_eq!(tensor.count_nonzero(), 3);
    }

    #[test]
    fn test_max_axis() {
        let tensor = Tensor::from_vec(
            vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], 
            vec![2, 3]
        );
        
        let max_axis_0 = tensor.max_axis(0).unwrap();
        assert_eq!(max_axis_0.shape(), &[3]);
        assert_eq!(max_axis_0.as_slice().unwrap(), &[2.0f32, 5.0, 6.0]);
        
        let max_axis_1 = tensor.max_axis(1).unwrap();
        assert_eq!(max_axis_1.shape(), &[2]);
        assert_eq!(max_axis_1.as_slice().unwrap(), &[4.0f32, 6.0]);
    }

    #[test]
    fn test_axis_out_of_bounds() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        
        assert!(tensor.sum_axis(2).is_err());
        assert!(tensor.mean_axis(5).is_err());
        assert!(tensor.max_axis(10).is_err());
    }
}
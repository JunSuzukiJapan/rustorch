//! Statistical operations for tensors
//! テンソルの統計演算

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;

impl<T: Float + 'static + ndarray::ScalarOperand + num_traits::FromPrimitive> Tensor<T> {
    /// Sum all elements (new implementation)
    /// 全要素の合計（新実装）
    pub fn sum(&self) -> T {
        self.data.iter().copied().fold(T::zero(), |acc, x| acc + x)
    }

    /// Mean of all elements (new implementation)
    /// 全要素の平均（新実装）
    pub fn mean(&self) -> T {
        let sum = self.sum();
        let count = T::from(self.data.len()).unwrap_or(T::one());
        sum / count
    }

    /// Get a single scalar value (for 0-dim or 1-element tensors)
    /// スカラー値を取得（0次元または1要素テンソル用）
    pub fn item(&self) -> T {
        if self.data.len() == 1 {
            self.data[0]
        } else {
            panic!("item() can only be called on tensors with exactly one element")
        }
    }

    /// Sum along a specific axis (new implementation)
    /// 特定の軸に沿った合計（新実装）
    pub fn sum_axis(&self, axis: usize) -> RusTorchResult<Self> {
        let shape = self.shape();

        if axis >= shape.len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "sum_axis".to_string(),
                message: format!(
                    "Axis {} is out of bounds for tensor with {} dimensions",
                    axis,
                    shape.len()
                ),
            });
        }

        let mut new_shape = shape.to_vec();
        let _axis_size = new_shape.remove(axis);

        if new_shape.is_empty() {
            // Result is a scalar
            return Ok(Tensor::from_vec(vec![self.sum()], vec![1]));
        }

        let new_size: usize = new_shape.iter().product();
        let mut result = vec![T::zero(); new_size];

        match shape.len() {
            1 => {
                // 1D case - sum all elements
                result[0] = self.sum();
            }
            2 => {
                // 2D case - optimized implementation
                if axis == 0 {
                    // Sum along rows (result is column vector)
                    let cols = shape[1];
                    for j in 0..cols {
                        let mut sum = T::zero();
                        for i in 0..shape[0] {
                            if let Some(value) = self.data.get(ndarray::IxDyn(&[i, j])) {
                                sum = sum + *value;
                            }
                        }
                        result[j] = sum;
                    }
                } else {
                    // Sum along columns (result is row vector)
                    let cols = shape[1];
                    for i in 0..shape[0] {
                        let mut sum = T::zero();
                        for j in 0..cols {
                            if let Some(value) = self.data.get(ndarray::IxDyn(&[i, j])) {
                                sum = sum + *value;
                            }
                        }
                        result[i] = sum;
                    }
                }
            }
            _ => {
                // General N-dimensional case (simplified)
                // This is a placeholder - real implementation would handle strided access
                return Err(RusTorchError::UnsupportedOperation(
                    "sum_axis for >2D tensors not yet implemented".to_string(),
                ));
            }
        }

        Ok(Tensor::from_vec(result, new_shape))
    }

    /// Mean along a specific axis (new implementation)
    /// 特定の軸に沿った平均（新実装）
    pub fn mean_axis(&self, axis: usize) -> RusTorchResult<Self> {
        let sum_result = self.sum_axis(axis)?;
        let axis_size = T::from(self.shape()[axis]).unwrap_or(T::one());
        Ok(sum_result.div_scalar(axis_size))
    }

    /// Variance of all elements
    /// 全要素の分散
    pub fn var(&self) -> T {
        let mean = self.mean();
        let squared_diffs: T = self
            .data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x);

        let count = T::from(self.data.len()).unwrap_or(T::one());
        squared_diffs / count
    }

    /// Standard deviation of all elements
    /// 全要素の標準偏差
    pub fn std(&self) -> T {
        self.var().sqrt()
    }

    /// Minimum value
    /// 最小値
    pub fn min(&self) -> T {
        self.data
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(T::zero())
    }

    /// Maximum value
    /// 最大値
    pub fn max(&self) -> T {
        self.data
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(T::zero())
    }

    /// Argmin - index of minimum value (flattened)
    /// 最小値のインデックス（平坦化）
    pub fn argmin(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Argmax - index of maximum value (flattened)
    /// 最大値のインデックス（平坦化）
    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Median value (approximate for efficiency)
    /// 中央値（効率性のため近似）
    pub fn median(&self) -> T {
        let mut sorted_data: Vec<T> = self.data.iter().copied().collect();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted_data.len();
        if len % 2 == 1 {
            sorted_data[len / 2]
        } else {
            let mid1 = sorted_data[len / 2 - 1];
            let mid2 = sorted_data[len / 2];
            (mid1 + mid2) / T::from(2.0).unwrap()
        }
    }

    /// Quantile (0.0 to 1.0)
    /// 分位数（0.0から1.0）
    pub fn quantile(&self, q: f64) -> T {
        if q < 0.0 || q > 1.0 {
            panic!("Quantile must be between 0.0 and 1.0");
        }

        let mut sorted_data: Vec<T> = self.data.iter().copied().collect();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted_data.len();
        let idx = (q * (len - 1) as f64) as usize;

        if idx >= len {
            sorted_data[len - 1]
        } else {
            sorted_data[idx]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(tensor.sum(), 10.0);
    }

    #[test]
    fn test_mean() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(tensor.mean(), 2.5);
    }

    #[test]
    fn test_sum_axis() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        // Sum along axis 0 (rows)
        let result = tensor.sum_axis(0).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]

        // Sum along axis 1 (columns)
        let result = tensor.sum_axis(1).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_var_std() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let var = tensor.var();
        let std = tensor.std();

        // Variance should be 1.25, std should be sqrt(1.25) ≈ 1.118
        assert!((var - 1.25).abs() < 0.001);
        assert!((std - 1.118).abs() < 0.01);
    }

    #[test]
    fn test_min_max() {
        let tensor = Tensor::from_vec(vec![3.0, 1.0, 4.0, 2.0], vec![4]);
        assert_eq!(tensor.min(), 1.0);
        assert_eq!(tensor.max(), 4.0);
        assert_eq!(tensor.argmin(), 1);
        assert_eq!(tensor.argmax(), 2);
    }

    #[test]
    fn test_median() {
        let tensor = Tensor::from_vec(vec![3.0, 1.0, 4.0, 2.0], vec![4]);
        assert_eq!(tensor.median(), 2.5); // (2.0 + 3.0) / 2

        let tensor_odd = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(tensor_odd.median(), 2.0);
    }
}

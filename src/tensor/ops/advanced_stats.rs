//! Advanced statistical functions for tensors
//! テンソル用高度統計関数

use super::super::core::Tensor;
use crate::error::{RusTorchError, RusTorchResult};
use ndarray::{Axis, IxDyn};
use num_traits::Float;

impl<
        T: Float
            + 'static
            + ndarray::ScalarOperand
            + num_traits::FromPrimitive
            + std::cmp::PartialOrd
            + Clone,
    > Tensor<T>
{
    // Variance and standard deviation
    // 分散と標準偏差

    // Note: var method is defined in statistical.rs to avoid duplication
    // 注意: varメソッドは重複を避けるためstatistical.rsで定義されています

    /// Calculate unbiased variance along all dimensions
    /// 全次元での不偏分散を計算
    pub fn var_unbiased(&self) -> T {
        let mean_val = self.mean();
        let n = T::from(self.numel()).unwrap();

        if n <= T::one() {
            return T::zero();
        }

        let variance: T = self
            .data
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
            / (n - T::one());

        variance
    }

    /// Calculate variance along a specific axis
    /// 特定軸での分散を計算
    pub fn var_axis(&self, axis: usize) -> RusTorchResult<Tensor<T>> {
        if axis >= self.shape().len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "var_axis".to_string(),
                message: format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    self.shape().len()
                ),
            });
        }

        let mean_tensor = self.mean_axis(axis)?;
        let expanded_mean = mean_tensor.expand_dims(axis)?;

        // Calculate squared differences
        let diff = self.sub(&expanded_mean)?;
        let squared_diff = diff.square();

        // Calculate mean of squared differences
        squared_diff.mean_axis(axis)
    }

    // Note: std method is defined in statistical.rs to avoid duplication
    // 注意: stdメソッドは重複を避けるためstatistical.rsで定義されています

    /// Calculate unbiased standard deviation along all dimensions
    /// 全次元での不偏標準偏差を計算
    pub fn std_unbiased(&self) -> T {
        self.var_unbiased().sqrt()
    }

    /// Calculate standard deviation along a specific axis
    /// 特定軸での標準偏差を計算
    pub fn std_axis(&self, axis: usize) -> RusTorchResult<Tensor<T>> {
        let var_result = self.var_axis(axis)?;
        Ok(var_result.sqrt())
    }

    // Median and quantile functions
    // 中央値と分位数関数

    // Note: median method is defined in statistical.rs to avoid duplication
    // 注意: medianメソッドは重複を避けるためstatistical.rsで定義されています

    /// Calculate quantile value (q between 0 and 1) - advanced version
    /// 分位数値を計算 (qは0から1の間) - 高度版
    pub fn quantile_advanced(&self, q: f64) -> RusTorchResult<T> {
        if q < 0.0 || q > 1.0 {
            return Err(RusTorchError::InvalidOperation {
                operation: "quantile".to_string(),
                message: "Quantile must be between 0 and 1".to_string(),
            });
        }

        let mut sorted_data = self.data.iter().cloned().collect::<Vec<_>>();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_data.len();
        let index = q * (n - 1) as f64;
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;

        if lower_idx == upper_idx {
            Ok(sorted_data[lower_idx])
        } else {
            let weight = T::from(index - lower_idx as f64).unwrap();
            let lower_val = sorted_data[lower_idx];
            let upper_val = sorted_data[upper_idx];
            Ok(lower_val + weight * (upper_val - lower_val))
        }
    }

    /// Calculate percentile value (p between 0 and 100)
    /// パーセンタイル値を計算 (pは0から100の間)
    pub fn percentile(&self, p: f64) -> RusTorchResult<T> {
        if p < 0.0 || p > 100.0 {
            return Err(RusTorchError::InvalidOperation {
                operation: "percentile".to_string(),
                message: "Percentile must be between 0 and 100".to_string(),
            });
        }

        self.quantile_advanced(p / 100.0)
    }

    // Cumulative functions
    // 累積関数

    /// Calculate cumulative sum along a specific axis (simplified implementation)
    /// 特定軸での累積和を計算（簡単化実装）
    pub fn cumsum(&self, axis: usize) -> RusTorchResult<Tensor<T>> {
        if axis >= self.shape().len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "cumsum".to_string(),
                message: format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    self.shape().len()
                ),
            });
        }

        let shape = self.shape().to_vec();
        let axis_size = shape[axis];

        // Simplified implementation for 1D case only for now
        if shape.len() != 1 || axis != 0 {
            return Err(RusTorchError::InvalidOperation {
                operation: "cumsum".to_string(),
                message: "Cumsum currently only supports 1D tensors".to_string(),
            });
        }

        let mut result_data = Vec::new();
        let mut cumulative = T::zero();

        for &value in self.data.iter() {
            cumulative = cumulative + value;
            result_data.push(cumulative);
        }

        let result = Tensor::from_vec(result_data, shape.to_vec());

        Ok(result)
    }

    /// Calculate cumulative product along a specific axis (simplified implementation)
    /// 特定軸での累積積を計算（簡单化実装）
    pub fn cumprod(&self, axis: usize) -> RusTorchResult<Tensor<T>> {
        if axis >= self.shape().len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "cumprod".to_string(),
                message: format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    self.shape().len()
                ),
            });
        }

        let shape = self.shape().to_vec();
        let axis_size = shape[axis];

        // Simplified implementation for 1D case only for now
        if shape.len() != 1 || axis != 0 {
            return Err(RusTorchError::InvalidOperation {
                operation: "cumprod".to_string(),
                message: "Cumprod currently only supports 1D tensors".to_string(),
            });
        }

        let mut result_data = Vec::new();
        let mut cumulative = T::one();

        for &value in self.data.iter() {
            cumulative = cumulative * value;
            result_data.push(cumulative);
        }

        let result = Tensor::from_vec(result_data, shape.to_vec());

        Ok(result)
    }

    // Argmin and argmax functions
    // 最小値・最大値のインデックス関数

    // Note: argmin method is defined in statistical.rs to avoid duplication
    // 注意: argminメソッドは重複を避けるためstatistical.rsで定義されています

    // Note: argmax method is defined in statistical.rs to avoid duplication
    // 注意: argmaxメソッドは重複を避けるためstatistical.rsで定義されています

    /// Find indices of minimum values along an axis
    /// 軸に沿った最小値のインデックスを検索
    pub fn argmin_axis(&self, axis: usize) -> RusTorchResult<Vec<usize>> {
        if axis >= self.shape().len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "argmin_axis".to_string(),
                message: format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    self.shape().len()
                ),
            });
        }

        let shape = self.shape();
        let axis_size = shape[axis];
        let mut result = Vec::new();

        // Calculate reduced shape
        let mut reduced_shape = shape.to_vec();
        reduced_shape.remove(axis);
        let reduced_size: usize = reduced_shape.iter().product();

        for i in 0..reduced_size {
            let mut min_idx = 0;
            let mut min_val = None;

            for j in 0..axis_size {
                // Convert flat index to multidimensional coordinates
                let mut coords = Vec::new();
                let mut remaining = i;

                for (dim_idx, &dim_size) in reduced_shape.iter().enumerate().rev() {
                    coords.insert(0, remaining % dim_size);
                    remaining /= dim_size;
                }

                // Insert axis coordinate
                coords.insert(axis, j);

                // Convert back to flat index
                let mut flat_idx = 0;
                let mut multiplier = 1;
                for (dim_idx, &coord) in coords.iter().enumerate().rev() {
                    flat_idx += coord * multiplier;
                    multiplier *= shape[dim_idx];
                }

                if let Some(val) = self.data.get(IxDyn(&[flat_idx])) {
                    if min_val.is_none() || val < min_val.as_ref().unwrap() {
                        min_val = Some(*val);
                        min_idx = j;
                    }
                }
            }
            result.push(min_idx);
        }

        Ok(result)
    }

    /// Find indices of maximum values along an axis
    /// 軸に沿った最大値のインデックスを検索
    pub fn argmax_axis(&self, axis: usize) -> RusTorchResult<Vec<usize>> {
        if axis >= self.shape().len() {
            return Err(RusTorchError::InvalidOperation {
                operation: "argmax_axis".to_string(),
                message: format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    self.shape().len()
                ),
            });
        }

        let shape = self.shape();
        let axis_size = shape[axis];
        let mut result = Vec::new();

        // Calculate reduced shape
        let mut reduced_shape = shape.to_vec();
        reduced_shape.remove(axis);
        let reduced_size: usize = reduced_shape.iter().product();

        for i in 0..reduced_size {
            let mut max_idx = 0;
            let mut max_val = None;

            for j in 0..axis_size {
                // Convert flat index to multidimensional coordinates
                let mut coords = Vec::new();
                let mut remaining = i;

                for (dim_idx, &dim_size) in reduced_shape.iter().enumerate().rev() {
                    coords.insert(0, remaining % dim_size);
                    remaining /= dim_size;
                }

                // Insert axis coordinate
                coords.insert(axis, j);

                // Convert back to flat index
                let mut flat_idx = 0;
                let mut multiplier = 1;
                for (dim_idx, &coord) in coords.iter().enumerate().rev() {
                    flat_idx += coord * multiplier;
                    multiplier *= shape[dim_idx];
                }

                if let Some(val) = self.data.get(IxDyn(&[flat_idx])) {
                    if max_val.is_none() || val > max_val.as_ref().unwrap() {
                        max_val = Some(*val);
                        max_idx = j;
                    }
                }
            }
            result.push(max_idx);
        }

        Ok(result)
    }

    /// Find top-k largest values and their indices
    /// 上位k個の最大値とそのインデックスを検索
    pub fn topk(&self, k: usize) -> RusTorchResult<(Tensor<T>, Vec<usize>)> {
        if k > self.numel() {
            return Err(RusTorchError::InvalidOperation {
                operation: "topk".to_string(),
                message: format!(
                    "k ({}) cannot be larger than tensor size ({})",
                    k,
                    self.numel()
                ),
            });
        }

        let mut indexed_data: Vec<(usize, T)> = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, val))
            .collect();

        // Sort by value in descending order
        indexed_data
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let top_values: Vec<T> = indexed_data.iter().take(k).map(|(_, val)| *val).collect();
        let top_indices: Vec<usize> = indexed_data.iter().take(k).map(|(idx, _)| *idx).collect();

        let values_tensor = Tensor::from_vec(top_values, vec![k]);

        Ok((values_tensor, top_indices))
    }

    /// Find bottom-k smallest values and their indices
    /// 下位k個の最小値とそのインデックスを検索
    pub fn bottomk(&self, k: usize) -> RusTorchResult<(Tensor<T>, Vec<usize>)> {
        if k > self.numel() {
            return Err(RusTorchError::InvalidOperation {
                operation: "bottomk".to_string(),
                message: format!(
                    "k ({}) cannot be larger than tensor size ({})",
                    k,
                    self.numel()
                ),
            });
        }

        let mut indexed_data: Vec<(usize, T)> = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, val))
            .collect();

        // Sort by value in ascending order
        indexed_data
            .sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let bottom_values: Vec<T> = indexed_data.iter().take(k).map(|(_, val)| *val).collect();
        let bottom_indices: Vec<usize> = indexed_data.iter().take(k).map(|(idx, _)| *idx).collect();

        let values_tensor = Tensor::from_vec(bottom_values, vec![k]);

        Ok((values_tensor, bottom_indices))
    }

    // Correlation and covariance
    // 相関と共分散

    /// Calculate correlation coefficient between two 1D tensors
    /// 2つの1次元テンソル間の相関係数を計算
    pub fn corrcoef(&self, other: &Tensor<T>) -> RusTorchResult<T> {
        if self.shape().len() != 1 || other.shape().len() != 1 || self.numel() != other.numel() {
            return Err(RusTorchError::InvalidOperation {
                operation: "corrcoef".to_string(),
                message: "Both tensors must be 1D and have the same length".to_string(),
            });
        }

        let mean_x = self.mean();
        let mean_y = other.mean();

        let n = T::from(self.numel()).unwrap();

        let mut sum_xy = T::zero();
        let mut sum_x2 = T::zero();
        let mut sum_y2 = T::zero();

        for (x_val, y_val) in self.data.iter().zip(other.data.iter()) {
            let x_dev = *x_val - mean_x;
            let y_dev = *y_val - mean_y;

            sum_xy = sum_xy + x_dev * y_dev;
            sum_x2 = sum_x2 + x_dev * x_dev;
            sum_y2 = sum_y2 + y_dev * y_dev;
        }

        let denominator = (sum_x2 * sum_y2).sqrt();
        if denominator == T::zero() {
            Ok(T::zero())
        } else {
            Ok(sum_xy / denominator)
        }
    }

    /// Calculate covariance between two 1D tensors
    /// 2つの1次元テンソル間の共分散を計算
    pub fn cov(&self, other: &Tensor<T>) -> RusTorchResult<T> {
        if self.shape().len() != 1 || other.shape().len() != 1 || self.numel() != other.numel() {
            return Err(RusTorchError::InvalidOperation {
                operation: "cov".to_string(),
                message: "Both tensors must be 1D and have the same length".to_string(),
            });
        }

        let mean_x = self.mean();
        let mean_y = other.mean();
        let n = T::from(self.numel()).unwrap();

        let covariance: T = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .fold(T::zero(), |acc, x| acc + x)
            / n;

        Ok(covariance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_and_std() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);

        let var = tensor.var();
        let std = tensor.std();
        let mean = tensor.mean();

        // Variance of [1,2,3,4,5] should be 2.0
        assert!((var - 2.0).abs() < 1e-10);
        assert!((std - var.sqrt()).abs() < 1e-10);
        assert_eq!(mean, 3.0);
    }

    #[test]
    fn test_median_and_quantiles() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);

        let median = tensor.median();
        let q25 = tensor.quantile_advanced(0.25).unwrap();
        let q75 = tensor.quantile_advanced(0.75).unwrap();
        let p50 = tensor.percentile(50.0).unwrap();

        assert_eq!(median, 3.0);
        assert_eq!(p50, median);
        assert_eq!(q25, 2.0);
        assert_eq!(q75, 4.0);
    }

    #[test]
    fn test_cumulative_functions() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let cumsum = tensor.cumsum(0).unwrap();
        let cumprod = tensor.cumprod(0).unwrap();

        assert_eq!(cumsum.as_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(cumprod.as_slice().unwrap(), &[1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_argmin_argmax() {
        let tensor = Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]);

        let argmin = tensor.argmin();
        let argmax = tensor.argmax();

        assert_eq!(argmin, 1); // First occurrence of minimum value (1.0)
        assert_eq!(argmax, 4); // Index of maximum value (5.0)
    }

    #[test]
    fn test_topk() {
        let tensor = Tensor::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5]);

        let (top_values, top_indices) = tensor.topk(3).unwrap();

        assert_eq!(top_values.as_slice().unwrap(), &[5.0, 4.0, 3.0]);
        assert_eq!(top_indices, vec![4, 2, 0]);

        let (bottom_values, bottom_indices) = tensor.bottomk(2).unwrap();

        assert_eq!(bottom_values.as_slice().unwrap(), &[1.0, 1.0]);
        assert_eq!(bottom_indices, vec![1, 3]);
    }

    #[test]
    fn test_correlation_and_covariance() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let y = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0], vec![5]);

        let corr = x.corrcoef(&y).unwrap();
        let cov = x.cov(&y).unwrap();

        // Perfect positive correlation
        assert!((corr - 1.0).abs() < 1e-10);

        // Covariance should be positive
        assert!(cov > 0.0);
    }

    #[test]
    fn test_var_axis() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

        let var_axis0 = tensor.var_axis(0).unwrap();
        let var_axis1 = tensor.var_axis(1).unwrap();

        assert_eq!(var_axis0.shape(), &[3]);
        assert_eq!(var_axis1.shape(), &[2]);

        // Check that variances are calculated correctly
        let slice0 = var_axis0.as_slice().unwrap();
        let slice1 = var_axis1.as_slice().unwrap();

        // Variance along axis 0: variance of [1,4], [2,5], [3,6]
        assert!((slice0[0] - 2.25).abs() < 1e-10); // var([1,4]) = 2.25
        assert!((slice0[1] - 2.25).abs() < 1e-10); // var([2,5]) = 2.25
        assert!((slice0[2] - 2.25).abs() < 1e-10); // var([3,6]) = 2.25
    }

    #[test]
    fn test_unbiased_variance() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        let var_biased = tensor.var();
        let var_unbiased = tensor.var_unbiased();
        let std_unbiased = tensor.std_unbiased();

        // Unbiased variance should be larger than biased variance
        assert!(var_unbiased > var_biased);
        assert!((std_unbiased - var_unbiased.sqrt()).abs() < 1e-10);

        // For 3 samples: unbiased = biased * 3/2
        assert!((var_unbiased - var_biased * 1.5).abs() < 1e-10);
    }
}

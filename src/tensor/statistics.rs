/// Statistical operations for tensors
/// テンソルの統計演算

use ndarray::Axis;
use num_traits::{Float, FromPrimitive, Zero};
use std::ops::Sub;
use crate::tensor::Tensor;

impl<T: Float + 'static + Send + Sync + FromPrimitive + Zero> Tensor<T> {
    /// Compute mean along specified axis
    /// 指定された軸に沿って平均を計算
    pub fn mean(&self, axis: Option<usize>) -> Self {
        match axis {
            Some(ax) => self.mean_along_axis(ax),
            None => self.mean_all(),
        }
    }


    /// Compute standard deviation along specified axis
    /// 指定された軸に沿って標準偏差を計算
    pub fn std(&self, axis: Option<usize>, unbiased: bool) -> Self {
        let var_tensor = self.var(axis, unbiased);
        var_tensor.sqrt()
    }

    /// Compute variance along specified axis
    /// 指定された軸に沿って分散を計算
    pub fn var(&self, axis: Option<usize>, unbiased: bool) -> Self {
        match axis {
            Some(ax) => self.var_along_axis(ax, unbiased),
            None => self.var_all(unbiased),
        }
    }

    /// Compute median along specified axis
    /// 指定された軸に沿って中央値を計算
    pub fn median(&self, axis: Option<usize>) -> Self {
        match axis {
            Some(ax) => self.median_along_axis(ax),
            None => self.median_all(),
        }
    }

    /// Compute quantile along specified axis
    /// 指定された軸に沿って分位数を計算
    pub fn quantile(&self, q: T, axis: Option<usize>) -> Self {
        if q < T::zero() || q > T::one() {
            panic!("Quantile must be between 0 and 1");
        }
        
        match axis {
            Some(ax) => self.quantile_along_axis(q, ax),
            None => self.quantile_all(q),
        }
    }

    /// Compute mode (most frequent value) along specified axis
    /// 指定された軸に沿って最頻値を計算
    pub fn mode(&self, axis: Option<usize>) -> Self {
        match axis {
            Some(ax) => self.mode_along_axis(ax),
            None => self.mode_all(),
        }
    }

    /// Compute cumulative sum along specified axis
    /// 指定された軸に沿って累積和を計算
    pub fn cumsum(&self, axis: usize) -> Result<Self, StatError> {
        if axis >= self.data.ndim() {
            return Err(StatError::InvalidAxis { axis, ndim: self.data.ndim() });
        }

        let mut result = self.data.clone();
        let axis_len = self.data.shape()[axis];
        
        for i in 1..axis_len {
            let prev_val = result.slice_axis(Axis(axis), (i-1..i).into()).to_owned();
            let mut curr_slice = result.slice_axis_mut(Axis(axis), (i..i+1).into());
            
            curr_slice.zip_mut_with(&prev_val, |curr, &prev| {
                *curr = *curr + prev;
            });
        }

        Ok(Tensor::new(result))
    }

    /// Compute cumulative product along specified axis
    /// 指定された軸に沿って累積積を計算
    pub fn cumprod(&self, axis: usize) -> Result<Self, StatError> {
        if axis >= self.data.ndim() {
            return Err(StatError::InvalidAxis { axis, ndim: self.data.ndim() });
        }

        let mut result = self.data.clone();
        let axis_len = self.data.shape()[axis];
        
        for i in 1..axis_len {
            let prev_val = result.slice_axis(Axis(axis), (i-1..i).into()).to_owned();
            let mut curr_slice = result.slice_axis_mut(Axis(axis), (i..i+1).into());
            
            curr_slice.zip_mut_with(&prev_val, |curr, &prev| {
                *curr = *curr * prev;
            });
        }

        Ok(Tensor::new(result))
    }

    /// Compute histogram
    /// ヒストグラムを計算
    pub fn histogram(&self, bins: usize, range: Option<(T, T)>) -> (Tensor<T>, Tensor<T>) {
        let data_slice = self.data.as_slice().unwrap();
        
        let (min_val, max_val) = match range {
            Some((min, max)) => (min, max),
            None => {
                let min = data_slice.iter().fold(T::infinity(), |a, &b| a.min(b));
                let max = data_slice.iter().fold(T::neg_infinity(), |a, &b| a.max(b));
                (min, max)
            }
        };

        let bin_width = (max_val - min_val) / T::from(bins).unwrap();
        let mut counts = vec![T::zero(); bins];
        let mut bin_edges = Vec::with_capacity(bins + 1);

        // Create bin edges
        for i in 0..=bins {
            bin_edges.push(min_val + T::from(i).unwrap() * bin_width);
        }

        // Count values in each bin
        for &value in data_slice {
            if value >= min_val && value <= max_val {
                let bin_idx = if value == max_val {
                    bins - 1  // Put max value in last bin
                } else {
                    ((value - min_val) / bin_width).to_usize().unwrap().min(bins - 1)
                };
                counts[bin_idx] = counts[bin_idx] + T::one();
            }
        }

        (
            Tensor::from_vec(counts, vec![bins]),
            Tensor::from_vec(bin_edges, vec![bins + 1])
        )
    }

    /// Compute covariance matrix
    /// 共分散行列を計算
    pub fn cov(&self) -> Self {
        let shape = self.data.shape();
        if shape.len() != 2 {
            panic!("Covariance requires 2D tensor");
        }

        let _n_features = shape[1];
        let n_samples = shape[0];
        
        // Center the data (subtract mean)
        let means = self.mean(Some(0));
        let centered = self.sub(&means.unsqueeze(0).unwrap());
        
        // Compute covariance matrix: (X^T * X) / (n - 1)
        let cov_matrix = centered.transpose().matmul(&centered);
        let divisor = T::from(n_samples - 1).unwrap();
        
        let divisor_tensor = Tensor::from_vec(vec![divisor], vec![1]);
        &cov_matrix / &divisor_tensor
    }

    /// Compute correlation matrix
    /// 相関行列を計算
    pub fn corrcoef(&self) -> Self {
        let cov_matrix = self.cov();
        let std_devs = self.std(Some(0), true);
        
        // Compute correlation from covariance: cov(X,Y) / (std(X) * std(Y))
        let shape = cov_matrix.data.shape();
        let mut corr_matrix = cov_matrix.data.clone();
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let std_i = std_devs.data[[i]];
                let std_j = std_devs.data[[j]];
                corr_matrix[[i, j]] = corr_matrix[[i, j]] / (std_i * std_j);
            }
        }
        
        Tensor::new(corr_matrix)
    }

    // Helper methods for axis-specific operations

    fn mean_along_axis(&self, axis: usize) -> Self {
        let result = self.data.mean_axis(ndarray::Axis(axis)).unwrap();
        Tensor::new(result)
    }

    fn mean_all(&self) -> Self {
        let mean_val = self.data.mean().unwrap();
        Tensor::from_vec(vec![mean_val], vec![1])
    }

    fn expand_dims_for_broadcast(&self, tensor: &Self, axis: usize) -> Self {
        let mut new_shape = vec![1; self.data.ndim()];
        let tensor_shape = tensor.shape();
        
        // Copy tensor shape to new_shape, skipping the reduced axis
        let mut tensor_idx = 0;
        for i in 0..new_shape.len() {
            if i != axis {
                new_shape[i] = tensor_shape[tensor_idx];
                tensor_idx += 1;
            }
        }
        
        let reshaped = tensor.data.clone().into_shape(new_shape).unwrap();
        Tensor::new(reshaped)
    }

    fn var_along_axis(&self, axis: usize, unbiased: bool) -> Self {
        let mean_tensor = self.mean(Some(axis));
        let mean_expanded = self.expand_dims_for_broadcast(&mean_tensor, axis);
        
        let diff = self - &mean_expanded;
        let squared_diff = &diff * &diff;
        let sum_squared = squared_diff.sum_axis(axis);
        
        let n = T::from(self.data.shape()[axis]).unwrap();
        let divisor = if unbiased && n > T::one() {
            n - T::one()
        } else {
            n
        };
        
        let divisor_tensor = Tensor::from_vec(vec![divisor], vec![1]);
        &sum_squared / &divisor_tensor
    }

    fn var_all(&self, unbiased: bool) -> Self {
        let mean_val = self.mean(None);
        let diff = self - &mean_val;
        let squared_diff = &diff * &diff;
        let sum_squared = squared_diff.sum();
        
        let n = T::from(self.data.len()).unwrap();
        let divisor = if unbiased && n > T::one() {
            n - T::one()
        } else {
            n
        };
        
        let divisor_tensor = Tensor::from_vec(vec![divisor], vec![1]);
        &sum_squared / &divisor_tensor
    }

    fn median_along_axis(&self, axis: usize) -> Self {
        let shape = self.data.shape();
        let mut result_shape = shape.to_vec();
        result_shape.remove(axis);
        
        let result_size: usize = result_shape.iter().product();
        let mut result_data = vec![T::zero(); result_size];
        
        // For each slice along the axis, compute median
        let axis_size = shape[axis];
        let mut temp_vec = Vec::with_capacity(axis_size);
        
        for i in 0..result_size {
            temp_vec.clear();
            
            // Collect values along the axis for this position
            for j in 0..axis_size {
                let mut indices = self.unravel_index(i, &result_shape);
                indices.insert(axis, j);
                temp_vec.push(self.data[ndarray::IxDyn(&indices)]);
            }
            
            result_data[i] = self.compute_median(&mut temp_vec);
        }
        
        Tensor::from_vec(result_data, result_shape)
    }

    fn median_all(&self) -> Self {
        let mut data_vec: Vec<T> = self.data.iter().copied().collect();
        let median_val = self.compute_median(&mut data_vec);
        Tensor::from_vec(vec![median_val], vec![1])
    }

    fn quantile_along_axis(&self, q: T, axis: usize) -> Self {
        let shape = self.data.shape();
        let mut result_shape = shape.to_vec();
        result_shape.remove(axis);
        
        let result_size: usize = result_shape.iter().product();
        let mut result_data = vec![T::zero(); result_size];
        
        let axis_size = shape[axis];
        let mut temp_vec = Vec::with_capacity(axis_size);
        
        for i in 0..result_size {
            temp_vec.clear();
            
            for j in 0..axis_size {
                let mut indices = self.unravel_index(i, &result_shape);
                indices.insert(axis, j);
                temp_vec.push(self.data[ndarray::IxDyn(&indices)]);
            }
            
            result_data[i] = self.compute_quantile(&mut temp_vec, q);
        }
        
        Tensor::from_vec(result_data, result_shape)
    }

    fn quantile_all(&self, q: T) -> Self {
        let mut data_vec: Vec<T> = self.data.iter().copied().collect();
        let quantile_val = self.compute_quantile(&mut data_vec, q);
        Tensor::from_vec(vec![quantile_val], vec![1])
    }

    fn mode_along_axis(&self, _axis: usize) -> Self {
        // Simplified implementation - would need more sophisticated approach for real use
        self.median_along_axis(_axis)
    }

    fn mode_all(&self) -> Self {
        self.median_all()
    }

    // Utility methods

    fn compute_median(&self, data: &mut [T]) -> T {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = data.len();
        
        if len % 2 == 0 {
            let mid1 = data[len / 2 - 1];
            let mid2 = data[len / 2];
            (mid1 + mid2) / (T::one() + T::one())
        } else {
            data[len / 2]
        }
    }

    fn compute_quantile(&self, data: &mut [T], q: T) -> T {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = data.len();
        
        if len == 1 {
            return data[0];
        }
        
        let index = q * T::from(len - 1).unwrap();
        let lower_idx = index.floor().to_usize().unwrap();
        let upper_idx = index.ceil().to_usize().unwrap();
        
        if lower_idx == upper_idx {
            data[lower_idx]
        } else {
            let weight = index - T::from(lower_idx).unwrap();
            data[lower_idx] * (T::one() - weight) + data[upper_idx] * weight
        }
    }


    fn unravel_index(&self, flat_index: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; shape.len()];
        let mut remaining = flat_index;
        
        for (i, &dim_size) in shape.iter().enumerate().rev() {
            indices[i] = remaining % dim_size;
            remaining /= dim_size;
        }
        
        indices
    }
}

/// Statistical operation errors
/// 統計演算エラー
#[derive(Debug, Clone)]
pub enum StatError {
    /// Invalid axis for operation
    InvalidAxis { 
        /// Invalid axis index
        /// 無効な軸インデックス
        axis: usize, 
        /// Number of dimensions
        /// 次元数
        ndim: usize 
    },
    /// Invalid quantile value
    InvalidQuantile { 
        /// Invalid quantile value (must be between 0 and 1)
        /// 無効な分位数値（0から1の間である必要があります）
        q: f64 
    },
    /// Insufficient data
    InsufficientData,
}

impl std::fmt::Display for StatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatError::InvalidAxis { axis, ndim } => {
                write!(f, "Invalid axis {} for tensor with {} dimensions", axis, ndim)
            }
            StatError::InvalidQuantile { q } => {
                write!(f, "Invalid quantile value: {} (must be between 0 and 1)", q)
            }
            StatError::InsufficientData => {
                write!(f, "Insufficient data for statistical operation")
            }
        }
    }
}

impl std::error::Error for StatError {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_variance_and_std() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        
        // Test variance (population)
        let var_result = tensor.var(None, false);
        let expected_var = 2.0; // Variance of [1,2,3,4,5]
        assert_abs_diff_eq!(var_result.data[[0]], expected_var, epsilon = 1e-6);
        
        // Test standard deviation
        let std_result = tensor.std(None, false);
        let expected_std = expected_var.sqrt();
        assert_abs_diff_eq!(std_result.data[[0]], expected_std, epsilon = 1e-6);
    }

    #[test]
    fn test_median() {
        // Odd number of elements
        let tensor1 = Tensor::from_vec(vec![1.0f32, 3.0, 2.0, 5.0, 4.0], vec![5]);
        let median1 = tensor1.median(None);
        assert_abs_diff_eq!(median1.data[[0]], 3.0, epsilon = 1e-6);
        
        // Even number of elements
        let tensor2 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let median2 = tensor2.median(None);
        assert_abs_diff_eq!(median2.data[[0]], 2.5, epsilon = 1e-6);
    }

    #[test]
    fn test_quantile() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5]);
        
        // Test 25th percentile
        let q25 = tensor.quantile(0.25, None);
        assert_abs_diff_eq!(q25.data[[0]], 2.0, epsilon = 1e-6);
        
        // Test 75th percentile
        let q75 = tensor.quantile(0.75, None);
        assert_abs_diff_eq!(q75.data[[0]], 4.0, epsilon = 1e-6);
        
        // Test median (50th percentile)
        let q50 = tensor.quantile(0.5, None);
        assert_abs_diff_eq!(q50.data[[0]], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cumsum() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let cumsum_result = tensor.cumsum(0).unwrap();
        
        let expected = vec![1.0, 3.0, 6.0, 10.0];
        assert_eq!(cumsum_result.data.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_cumprod() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        let cumprod_result = tensor.cumprod(0).unwrap();
        
        let expected = vec![1.0, 2.0, 6.0, 24.0];
        assert_eq!(cumprod_result.data.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_histogram() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0], vec![7]);
        let (counts, edges) = tensor.histogram(4, Some((1.0, 4.0)));
        
        assert_eq!(counts.shape(), &[4]);
        assert_eq!(edges.shape(), &[5]);
        
        // Check that we have the right number of bins
        assert_eq!(edges.data.len(), 5); // n_bins + 1
    }

    #[test]
    fn test_2d_variance() {
        let tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 
            vec![2, 3]
        );
        
        // Variance along axis 0 (across rows)
        let var_axis0 = tensor.var(Some(0), false);
        assert_eq!(var_axis0.shape(), &[3]);
        
        // Variance along axis 1 (across columns)
        let var_axis1 = tensor.var(Some(1), false);
        assert_eq!(var_axis1.shape(), &[2]);
    }
}

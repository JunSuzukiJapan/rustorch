use super::Tensor;
use num_traits::Float;
use rayon::prelude::*;
use std::sync::Arc;

/// Parallel batch operations for tensors
/// テンソルの並列バッチ演算
impl<T: Float + Send + Sync + Clone + 'static> Tensor<T> {
    /// Parallel batch matrix multiplication
    /// 並列バッチ行列乗算
    pub fn batch_matmul_parallel(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        
        if self_shape.len() < 3 || other_shape.len() < 3 {
            return Err("Batch matmul requires at least 3D tensors".to_string());
        }
        
        let batch_size = self_shape[0];
        if batch_size != other_shape[0] {
            return Err("Batch sizes must match".to_string());
        }
        
        let m = self_shape[1];
        let k = self_shape[2];
        let n = other_shape[2];
        
        if k != other_shape[1] {
            return Err("Inner dimensions must match for matrix multiplication".to_string());
        }
        
        let result_shape = vec![batch_size, m, n];
        let mut result = Self::zeros(&result_shape);
        
        // Parallel processing over batch dimension
        let self_data = Arc::new(self.data.clone());
        let other_data = Arc::new(other.data.clone());
        
        let results: Vec<_> = (0..batch_size).into_par_iter().map(|b| {
            let mut batch_result = vec![T::zero(); m * n];
            
            // Extract batch matrices
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();
                    for l in 0..k {
                        let a_idx = b * m * k + i * k + l;
                        let b_idx = b * k * n + l * n + j;
                        
                        if let (Some(a_val), Some(b_val)) = (
                            self_data.as_slice().and_then(|s| s.get(a_idx)),
                            other_data.as_slice().and_then(|s| s.get(b_idx))
                        ) {
                            sum = sum + *a_val * *b_val;
                        }
                    }
                    batch_result[i * n + j] = sum;
                }
            }
            batch_result
        }).collect();
        
        // Combine results
        if let Some(result_slice) = result.data.as_slice_mut() {
            for (b, batch_result) in results.iter().enumerate() {
                let start_idx = b * m * n;
                for (i, &val) in batch_result.iter().enumerate() {
                    if let Some(dest) = result_slice.get_mut(start_idx + i) {
                        *dest = val;
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Parallel batch element-wise operations
    /// 並列バッチ要素ごと演算
    pub fn batch_add_parallel(&self, other: &Tensor<T>) -> Result<Tensor<T>, String> {
        if self.data.shape() != other.data.shape() {
            return Err("Tensor shapes must match for addition".to_string());
        }
        
        let mut result = Self::zeros(self.data.shape());
        
        if let (Some(self_slice), Some(other_slice), Some(result_slice)) = (
            self.data.as_slice(),
            other.data.as_slice(),
            result.data.as_slice_mut()
        ) {
            result_slice.par_iter_mut()
                .zip(self_slice.par_iter())
                .zip(other_slice.par_iter())
                .for_each(|((r, &a), &b)| {
                    *r = a + b;
                });
        }
        
        Ok(result)
    }
    
    /// Parallel batch scalar multiplication
    /// 並列バッチスカラー乗算
    pub fn batch_mul_scalar_parallel(&self, scalar: T) -> Tensor<T> {
        let mut result = Self::zeros(self.data.shape());
        
        if let (Some(self_slice), Some(result_slice)) = (
            self.data.as_slice(),
            result.data.as_slice_mut()
        ) {
            result_slice.par_iter_mut()
                .zip(self_slice.par_iter())
                .for_each(|(r, &a)| {
                    *r = a * scalar;
                });
        }
        
        result
    }
    
    /// Parallel batch normalization
    /// 並列バッチ正規化
    pub fn batch_normalize_parallel(&self, epsilon: T) -> Tensor<T> {
        let shape = self.data.shape();
        if shape.len() < 2 {
            return self.clone();
        }
        
        let batch_size = shape[0];
        let feature_size: usize = shape[1..].iter().product();
        
        let mut result = Self::zeros(shape);
        
        if let (Some(self_slice), Some(result_slice)) = (
            self.data.as_slice(),
            result.data.as_slice_mut()
        ) {
            // Parallel computation of batch statistics and normalization
            let batch_results: Vec<_> = (0..batch_size).into_par_iter().map(|b| {
                let start_idx = b * feature_size;
                let end_idx = start_idx + feature_size;
                let batch_data = &self_slice[start_idx..end_idx];
                
                // Compute mean
                let mean = batch_data.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(feature_size).unwrap();
                
                // Compute variance
                let variance = batch_data.iter()
                    .fold(T::zero(), |acc, &x| {
                        let diff = x - mean;
                        acc + diff * diff
                    }) / T::from(feature_size).unwrap();
                
                let std_dev = (variance + epsilon).sqrt();
                
                // Normalize
                let normalized: Vec<T> = batch_data.iter()
                    .map(|&x| (x - mean) / std_dev)
                    .collect();
                
                normalized
            }).collect();
            
            // Copy results back
            for (b, normalized) in batch_results.iter().enumerate() {
                let start_idx = b * feature_size;
                for (i, &val) in normalized.iter().enumerate() {
                    if let Some(dest) = result_slice.get_mut(start_idx + i) {
                        *dest = val;
                    }
                }
            }
        }
        
        result
    }
    
    /// Parallel batch convolution (simplified 2D)
    /// 並列バッチ畳み込み（簡略化2D）
    pub fn batch_conv2d_parallel(&self, kernel: &Tensor<T>, stride: usize, padding: usize) -> Result<Tensor<T>, String> {
        let input_shape = self.data.shape();
        let kernel_shape = kernel.data.shape();
        
        if input_shape.len() != 4 || kernel_shape.len() != 4 {
            return Err("Input and kernel must be 4D tensors (batch, channels, height, width)".to_string());
        }
        
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        
        let out_channels = kernel_shape[0];
        let kernel_height = kernel_shape[2];
        let kernel_width = kernel_shape[3];
        
        if in_channels != kernel_shape[1] {
            return Err("Input channels must match kernel input channels".to_string());
        }
        
        let out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        let out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
        
        let result_shape = vec![batch_size, out_channels, out_height, out_width];
        let mut result = Self::zeros(&result_shape);
        
        // Parallel processing over batch and output channels
        let self_data = Arc::new(self.data.clone());
        let kernel_data = Arc::new(kernel.data.clone());
        
        let batch_channel_pairs: Vec<(usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..out_channels).map(move |oc| (b, oc)))
            .collect();
        
        let results: Vec<_> = batch_channel_pairs.into_par_iter().map(|(b, oc)| {
            let mut channel_result = vec![T::zero(); out_height * out_width];
            
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = T::zero();
                    
                    for ic in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                
                                if ih >= padding && iw >= padding {
                                    let ih = ih - padding;
                                    let iw = iw - padding;
                                    
                                    if ih < in_height && iw < in_width {
                                        let input_idx = b * in_channels * in_height * in_width +
                                                       ic * in_height * in_width +
                                                       ih * in_width + iw;
                                        let kernel_idx = oc * in_channels * kernel_height * kernel_width +
                                                        ic * kernel_height * kernel_width +
                                                        kh * kernel_width + kw;
                                        
                                        if let (Some(input_val), Some(kernel_val)) = (
                                            self_data.as_slice().and_then(|s| s.get(input_idx)),
                                            kernel_data.as_slice().and_then(|s| s.get(kernel_idx))
                                        ) {
                                            sum = sum + *input_val * *kernel_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    channel_result[oh * out_width + ow] = sum;
                }
            }
            
            (b, oc, channel_result)
        }).collect();
        
        // Combine results
        if let Some(result_slice) = result.data.as_slice_mut() {
            for (b, oc, channel_result) in results {
                let start_idx = b * out_channels * out_height * out_width +
                               oc * out_height * out_width;
                
                for (i, &val) in channel_result.iter().enumerate() {
                    if let Some(dest) = result_slice.get_mut(start_idx + i) {
                        *dest = val;
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Parallel batch reduction operations
    /// 並列バッチリダクション演算
    pub fn batch_sum_parallel(&self, dim: usize) -> Result<Tensor<T>, String> {
        let shape = self.data.shape();
        if dim >= shape.len() {
            return Err("Dimension out of bounds".to_string());
        }
        
        let mut result_shape = shape.to_vec();
        result_shape.remove(dim);
        
        if result_shape.is_empty() {
            // Scalar result
            if let Some(slice) = self.data.as_slice() {
                let sum = slice.par_iter().fold(|| T::zero(), |acc, &x| acc + x).reduce(|| T::zero(), |a, b| a + b);
                return Ok(Tensor::from_vec(vec![sum], vec![]));
            }
        }
        
        let mut result = Self::zeros(&result_shape);
        
        // Parallel reduction along specified dimension
        let total_elements = shape.iter().product::<usize>();
        let dim_size = shape[dim];
        let stride_before: usize = shape[..dim].iter().product();
        let stride_after: usize = shape[dim+1..].iter().product();
        
        if let Some(self_slice) = self.data.as_slice() {
            let result_elements = result_shape.iter().product::<usize>();
            
            let computed_results: Vec<_> = (0..result_elements).into_par_iter().map(|result_idx| {
                let before_idx = result_idx / stride_after;
                let after_idx = result_idx % stride_after;
                
                let mut sum = T::zero();
                for d in 0..dim_size {
                    let source_idx = before_idx * dim_size * stride_after + 
                                   d * stride_after + after_idx;
                    if let Some(&val) = self_slice.get(source_idx) {
                        sum = sum + val;
                    }
                }
                (result_idx, sum)
            }).collect();
            
            // Copy results back
            if let Some(result_slice) = result.data.as_slice_mut() {
                for (idx, val) in computed_results {
                    if let Some(dest) = result_slice.get_mut(idx) {
                        *dest = val;
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Parallel batch mean computation
    /// 並列バッチ平均計算
    pub fn batch_mean_parallel(&self, dim: usize) -> Result<Tensor<T>, String> {
        let shape = self.data.shape();
        if dim >= shape.len() {
            return Err("Dimension out of bounds".to_string());
        }
        
        let sum_result = self.batch_sum_parallel(dim)?;
        let dim_size = T::from(shape[dim]).unwrap();
        
        Ok(sum_result.batch_mul_scalar_parallel(T::one() / dim_size))
    }
}

/// Specialized f32 implementations with SIMD integration
/// SIMD統合を含むf32特殊化実装
impl Tensor<f32> {
    /// High-performance parallel batch operations for f32
    /// f32用高性能並列バッチ演算
    pub fn batch_simd_add_parallel(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, String> {
        if self.data.shape() != other.data.shape() {
            return Err("Tensor shapes must match".to_string());
        }
        
        let mut result = Self::zeros(self.data.shape());
        
        if let (Some(self_slice), Some(other_slice), Some(result_slice)) = (
            self.data.as_slice(),
            other.data.as_slice(),
            result.data.as_slice_mut()
        ) {
            // Use chunked parallel processing for better SIMD utilization
            const CHUNK_SIZE: usize = 1024;
            
            self_slice.par_chunks(CHUNK_SIZE)
                .zip(other_slice.par_chunks(CHUNK_SIZE))
                .zip(result_slice.par_chunks_mut(CHUNK_SIZE))
                .for_each(|((a_chunk, b_chunk), r_chunk)| {
                    // Use SIMD operations for each chunk
                    crate::simd::ops::add_optimized(a_chunk, b_chunk, r_chunk);
                });
        }
        
        Ok(result)
    }
    
    /// Parallel batch matrix multiplication with SIMD optimization
    /// SIMD最適化を含む並列バッチ行列乗算
    pub fn batch_simd_matmul_parallel(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, String> {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        
        if self_shape.len() < 3 || other_shape.len() < 3 {
            return Err("Batch matmul requires at least 3D tensors".to_string());
        }
        
        let batch_size = self_shape[0];
        let m = self_shape[1];
        let k = self_shape[2];
        let n = other_shape[2];
        
        let result_shape = vec![batch_size, m, n];
        let mut result = Self::zeros(&result_shape);
        
        // Parallel processing with SIMD matrix multiplication
        if let (Some(self_slice), Some(other_slice)) = (
            self.data.as_slice(),
            other.data.as_slice()
        ) {
            let batch_results: Vec<_> = (0..batch_size).into_par_iter().map(|b| {
                let self_batch = &self_slice[b * m * k..(b + 1) * m * k];
                let other_batch = &other_slice[b * k * n..(b + 1) * k * n];
                
                // Create a temporary result for this batch
                let mut batch_result = vec![0.0f32; m * n];
                
                // Use SIMD-optimized matrix multiplication
                crate::simd::ops::matmul_optimized(
                    self_batch, m, k,
                    other_batch, k, n,
                    &mut batch_result
                );
                
                batch_result
            }).collect();
            
            // Copy results back
            if let Some(result_slice) = result.data.as_slice_mut() {
                for (b, batch_result) in batch_results.iter().enumerate() {
                    let start_idx = b * m * n;
                    for (i, &val) in batch_result.iter().enumerate() {
                        if let Some(dest) = result_slice.get_mut(start_idx + i) {
                            *dest = val;
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_add_parallel() {
        let a = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2]
        );
        let b = Tensor::<f32>::from_vec(
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2, 2, 2]
        );
        
        let result = a.batch_add_parallel(&b).unwrap();
        let expected = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        assert_eq!(result.data.as_slice().unwrap(), &expected);
    }
    
    #[test]
    fn test_batch_matmul_parallel() {
        let a = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2]
        );
        let b = Tensor::<f32>::from_vec(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![2, 2, 2]
        );
        
        let result = a.batch_matmul_parallel(&b).unwrap();
        
        // Should be identity multiplication for each batch
        assert_eq!(result.size(), vec![2, 2, 2]);
        assert_eq!(result.data.as_slice().unwrap(), a.data.as_slice().unwrap());
    }
    
    #[test]
    fn test_batch_normalize_parallel() {
        let a = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4]
        );
        
        let result = a.batch_normalize_parallel(1e-5);
        
        // Check that each batch is normalized (mean ≈ 0, std ≈ 1)
        assert_eq!(result.size(), vec![2, 4]);
        
        if let Some(slice) = result.data.as_slice() {
            // First batch
            let batch1_mean: f32 = slice[0..4].iter().sum::<f32>() / 4.0;
            assert!((batch1_mean).abs() < 1e-5);
            
            // Second batch
            let batch2_mean: f32 = slice[4..8].iter().sum::<f32>() / 4.0;
            assert!((batch2_mean).abs() < 1e-5);
        }
    }
    
    #[test]
    fn test_batch_sum_parallel() {
        let a = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        );
        
        // Sum along dimension 1 (columns)
        let result = a.batch_sum_parallel(1).unwrap();
        assert_eq!(result.size(), vec![2]);
        
        let expected = vec![6.0, 15.0]; // [1+2+3, 4+5+6]
        assert_eq!(result.data.as_slice().unwrap(), &expected);
    }
    
    #[test]
    fn test_batch_simd_add_parallel() {
        let size = 1000;
        let a = Tensor::<f32>::from_vec(
            (0..size).map(|i| i as f32).collect(),
            vec![10, 100]
        );
        let b = Tensor::<f32>::from_vec(
            vec![1.0; size],
            vec![10, 100]
        );
        
        let result = a.batch_simd_add_parallel(&b).unwrap();
        
        if let Some(slice) = result.data.as_slice() {
            for (i, &val) in slice.iter().enumerate() {
                assert_eq!(val, i as f32 + 1.0);
            }
        }
    }
    
    #[test]
    fn test_large_batch_performance() {
        let batch_size = 100;
        let feature_size = 1000;
        
        let a = Tensor::<f32>::from_vec(
            (0..batch_size * feature_size).map(|i| (i % 100) as f32).collect(),
            vec![batch_size, feature_size]
        );
        let b = Tensor::<f32>::from_vec(
            vec![0.1; batch_size * feature_size],
            vec![batch_size, feature_size]
        );
        
        let result = a.batch_add_parallel(&b).unwrap();
        assert_eq!(result.size(), vec![batch_size, feature_size]);
        
        // Verify correctness
        if let (Some(a_slice), Some(b_slice), Some(result_slice)) = (
            a.data.as_slice(),
            b.data.as_slice(),
            result.data.as_slice()
        ) {
            for i in 0..batch_size * feature_size {
                assert!((result_slice[i] - (a_slice[i] + b_slice[i])).abs() < 1e-6);
            }
        }
    }
}

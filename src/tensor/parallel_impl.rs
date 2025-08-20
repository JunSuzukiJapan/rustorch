//! 並列テンソル操作の実装
//! Implementation of parallel tensor operations

use super::Tensor;
use super::parallel_errors::{ParallelError, ParallelResult};
use super::parallel_traits::{ParallelOp, BatchParallelOp, MatrixParallelOp, ReductionParallelOp, SimdParallelOp, parallel_utils};
use num_traits::Float;
use std::sync::Arc;
use rayon::prelude::*;

/// Tensorの並列操作実装
/// Parallel operations implementation for Tensor
impl<T: Float + Send + Sync + Clone + 'static> ParallelOp<T> for Tensor<T> {}

impl<T: Float + Send + Sync + Clone + 'static> BatchParallelOp<T> for Tensor<T> {
    fn batch_elementwise_op<F>(&self, other: &Tensor<T>, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(T, T) -> T + Send + Sync,
    {
        if self.data.shape() != other.data.shape() {
            return Err(ParallelError::shape_mismatch(
                self.data.shape(),
                other.data.shape(),
                "element-wise operation"
            ));
        }
        
        let mut result = Self::with_strategy(self.data.shape(), crate::tensor::memory_optimized::AllocationStrategy::Pool);
        
        if let (Some(self_slice), Some(other_slice), Some(result_slice)) = (
            self.data.as_slice(),
            other.data.as_slice(),
            result.data.as_slice_mut()
        ) {
            if self.should_parallelize(self_slice.len()) {
                result_slice.par_iter_mut()
                    .zip(self_slice.par_iter())
                    .zip(other_slice.par_iter())
                    .for_each(|((r, &a), &b)| {
                        *r = op(a, b);
                    });
            } else {
                result_slice.iter_mut()
                    .zip(self_slice.iter())
                    .zip(other_slice.iter())
                    .for_each(|((r, &a), &b)| {
                        *r = op(a, b);
                    });
            }
        }
        
        Ok(result)
    }
    
    fn batch_scalar_op<F>(&self, scalar: T, op: F) -> Tensor<T>
    where
        F: Fn(T, T) -> T + Send + Sync,
    {
        let mut result = Self::with_strategy(self.data.shape(), crate::tensor::memory_optimized::AllocationStrategy::Pool);
        
        if let (Some(self_slice), Some(result_slice)) = (
            self.data.as_slice(),
            result.data.as_slice_mut()
        ) {
            if self.should_parallelize(self_slice.len()) {
                result_slice.par_iter_mut()
                    .zip(self_slice.par_iter())
                    .for_each(|(r, &a)| {
                        *r = op(a, scalar);
                    });
            } else {
                result_slice.iter_mut()
                    .zip(self_slice.iter())
                    .for_each(|(r, &a)| {
                        *r = op(a, scalar);
                    });
            }
        }
        
        result
    }
    
    fn batch_normalize(&self, epsilon: T) -> Tensor<T> {
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
            let batch_results: Vec<_> = (0..batch_size).into_par_iter().map(|b| {
                let start_idx = b * feature_size;
                let end_idx = start_idx + feature_size;
                let batch_data = &self_slice[start_idx..end_idx];
                
                // 平均を計算
                let mean = batch_data.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(feature_size).unwrap();
                
                // 分散を計算
                let variance = batch_data.iter()
                    .fold(T::zero(), |acc, &x| {
                        let diff = x - mean;
                        acc + diff * diff
                    }) / T::from(feature_size).unwrap();
                
                let std_dev = (variance + epsilon).sqrt();
                
                // 正規化
                let normalized: Vec<T> = batch_data.iter()
                    .map(|&x| (x - mean) / std_dev)
                    .collect();
                
                normalized
            }).collect();
            
            // 結果をコピー
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
}

impl<T: Float + Send + Sync + Clone + 'static> MatrixParallelOp<T> for Tensor<T> {
    fn batch_matmul(&self, other: &Tensor<T>) -> ParallelResult<Tensor<T>> {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        
        if self_shape.len() < 3 || other_shape.len() < 3 {
            return Err(ParallelError::insufficient_dimensions(
                3,
                self_shape.len().min(other_shape.len()),
                "batch matmul"
            ));
        }
        
        let batch_size = self_shape[0];
        if batch_size != other_shape[0] {
            return Err(ParallelError::batch_size_mismatch(batch_size, other_shape[0]));
        }
        
        let m = self_shape[1];
        let k = self_shape[2];
        let n = other_shape[2];
        
        if k != other_shape[1] {
            return Err(ParallelError::matmul_dimension_mismatch(self_shape, other_shape));
        }
        
        let result_shape = vec![batch_size, m, n];
        let mut result = Self::zeros(&result_shape);
        
        // バッチ次元での並列処理
        let self_data = Arc::new(self.data.clone());
        let other_data = Arc::new(other.data.clone());
        
        let results: Vec<Vec<T>> = parallel_utils::parallel_batch_process::<T, _, Vec<T>>(batch_size, |b| {
            let mut batch_result = vec![T::zero(); m * n];
            
            // バッチ行列を抽出して乗算
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
        });
        
        // 結果を結合
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
    
    fn batch_conv2d(&self, kernel: &Tensor<T>, stride: usize, padding: usize) -> ParallelResult<Tensor<T>> {
        let input_shape = self.data.shape();
        let kernel_shape = kernel.data.shape();
        
        if input_shape.len() != 4 || kernel_shape.len() != 4 {
            return Err(ParallelError::insufficient_dimensions(
                4,
                input_shape.len().min(kernel_shape.len()),
                "convolution"
            ));
        }
        
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        
        let out_channels = kernel_shape[0];
        let kernel_height = kernel_shape[2];
        let kernel_width = kernel_shape[3];
        
        if in_channels != kernel_shape[1] {
            return Err(ParallelError::convolution_error(
                in_channels,
                kernel_shape[1],
                "input channels must match kernel input channels"
            ));
        }
        
        let out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
        let out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
        
        let result_shape = vec![batch_size, out_channels, out_height, out_width];
        let mut result = Self::zeros(&result_shape);
        
        // バッチと出力チャンネルでの並列処理
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
        
        // 結果を結合
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
}

impl<T: Float + Send + Sync + Clone + 'static> ReductionParallelOp<T> for Tensor<T> {
    fn parallel_reduce<F, R>(&self, dim: usize, init: R, op: F) -> ParallelResult<Tensor<T>>
    where
        F: Fn(R, T) -> R + Send + Sync + Clone,
        R: Send + Sync + Clone + Into<T>,
    {
        let shape = self.data.shape();
        if dim >= shape.len() {
            return Err(ParallelError::dimension_error(
                dim,
                shape.len() - 1,
                "parallel reduce"
            ));
        }
        
        let mut result_shape = shape.to_vec();
        result_shape.remove(dim);
        
        if result_shape.is_empty() {
            // スカラー結果
            if let Some(slice) = self.data.as_slice() {
                let result = slice.par_iter()
                    .fold(|| init.clone(), |acc, &x| op(acc, x))
                    .reduce(|| init.clone(), |a, b| op(a, b.into()));
                return Ok(Tensor::from_vec(vec![result.into()], vec![]));
            }
        }
        
        let mut result = Self::zeros(&result_shape);
        
        // 指定された次元での並列リダクション
        let dim_size = shape[dim];
        let _stride_before: usize = shape[..dim].iter().product();
        let stride_after: usize = shape[dim+1..].iter().product();
        
        if let Some(self_slice) = self.data.as_slice() {
            let result_elements = result_shape.iter().product::<usize>();
            
            let computed_results: Vec<_> = (0..result_elements).into_par_iter().map(|result_idx| {
                let before_idx = result_idx / stride_after;
                let after_idx = result_idx % stride_after;
                
                let mut acc = init.clone();
                for d in 0..dim_size {
                    let source_idx = before_idx * dim_size * stride_after + 
                                   d * stride_after + after_idx;
                    if let Some(&val) = self_slice.get(source_idx) {
                        acc = op(acc, val);
                    }
                }
                (result_idx, acc.into())
            }).collect();
            
            // 結果をコピー
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
    
    fn parallel_mean(&self, dim: usize) -> ParallelResult<Tensor<T>> {
        let shape = self.data.shape();
        if dim >= shape.len() {
            return Err(ParallelError::dimension_error(
                dim,
                shape.len() - 1,
                "parallel mean"
            ));
        }
        
        let sum_result = self.parallel_sum(dim)?;
        let dim_size = T::from(shape[dim]).unwrap();
        
        Ok(sum_result.batch_scalar_op(dim_size, |a, b| a / b))
    }
}

/// f32特化のSIMD並列操作実装
/// SIMD parallel operations implementation specialized for f32
impl SimdParallelOp for Tensor<f32> {
    fn simd_parallel_add(&self, other: &Tensor<f32>) -> ParallelResult<Tensor<f32>> {
        if self.data.shape() != other.data.shape() {
            return Err(ParallelError::shape_mismatch(
                self.data.shape(),
                other.data.shape(),
                "SIMD parallel addition"
            ));
        }
        
        let mut result = Self::with_strategy(self.data.shape(), crate::tensor::memory_optimized::AllocationStrategy::Pool);
        
        if let (Some(self_slice), Some(other_slice), Some(result_slice)) = (
            self.data.as_slice(),
            other.data.as_slice(),
            result.data.as_slice_mut()
        ) {
            // SIMD最適化のためのチャンク並列処理
            const CHUNK_SIZE: usize = 1024;
            
            self_slice.par_chunks(CHUNK_SIZE)
                .zip(other_slice.par_chunks(CHUNK_SIZE))
                .zip(result_slice.par_chunks_mut(CHUNK_SIZE))
                .for_each(|((a_chunk, b_chunk), r_chunk)| {
                    // 各チャンクでSIMD演算を使用
                    crate::simd::ops::add_optimized(a_chunk, b_chunk, r_chunk);
                });
        }
        
        Ok(result)
    }
    
    fn simd_parallel_matmul(&self, other: &Tensor<f32>) -> ParallelResult<Tensor<f32>> {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();
        
        if self_shape.len() < 3 || other_shape.len() < 3 {
            return Err(ParallelError::insufficient_dimensions(
                3,
                self_shape.len().min(other_shape.len()),
                "SIMD batch matmul"
            ));
        }
        
        let batch_size = self_shape[0];
        let m = self_shape[1];
        let k = self_shape[2];
        let n = other_shape[2];
        
        let result_shape = vec![batch_size, m, n];
        let mut result = Self::zeros(&result_shape);
        
        // SIMD最適化行列乗算での並列処理
        if let (Some(self_slice), Some(other_slice)) = (
            self.data.as_slice(),
            other.data.as_slice()
        ) {
            let batch_results: Vec<Vec<f32>> = parallel_utils::parallel_batch_process::<f32, _, Vec<f32>>(batch_size, |b| {
                let self_batch = &self_slice[b * m * k..(b + 1) * m * k];
                let other_batch = &other_slice[b * k * n..(b + 1) * k * n];
                
                // このバッチの一時的な結果を作成
                let mut batch_result = vec![0.0f32; m * n];
                
                // SIMD最適化行列乗算を使用
                crate::simd::ops::matmul_optimized(
                    self_batch, m, k,
                    other_batch, k, n,
                    &mut batch_result
                );
                
                batch_result
            });
            
            // 結果をコピー
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
    
    fn simd_parallel_scalar_mul(&self, scalar: f32) -> Tensor<f32> {
        let mut result = Self::with_strategy(self.data.shape(), crate::tensor::memory_optimized::AllocationStrategy::Pool);
        
        if let (Some(self_slice), Some(result_slice)) = (
            self.data.as_slice(),
            result.data.as_slice_mut()
        ) {
            const CHUNK_SIZE: usize = 1024;
            
            self_slice.par_chunks(CHUNK_SIZE)
                .zip(result_slice.par_chunks_mut(CHUNK_SIZE))
                .for_each(|(a_chunk, r_chunk)| {
                    // SIMD最適化スカラー乗算を使用
                    crate::simd::ops::mul_scalar_optimized(a_chunk, scalar, r_chunk);
                });
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_elementwise_op() {
        let a = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2]
        );
        let b = Tensor::<f32>::from_vec(
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2, 2]
        );
        
        let result = a.batch_elementwise_op(&b, |x, y| x + y).unwrap();
        let expected = vec![2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(result.data.as_slice().unwrap(), &expected);
    }
    
    #[test]
    fn test_batch_scalar_op() {
        let a = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2]
        );
        
        let result = a.batch_scalar_op(2.0, |x, y| x * y);
        let expected = vec![2.0, 4.0, 6.0, 8.0];
        
        assert_eq!(result.data.as_slice().unwrap(), &expected);
    }
    
    #[test]
    fn test_parallel_reduce() {
        let a = Tensor::<f32>::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3]
        );
        
        // 次元1での合計（列方向）
        let result = a.parallel_sum(1).unwrap();
        assert_eq!(result.size(), vec![2]);
        
        let expected = vec![6.0, 15.0]; // [1+2+3, 4+5+6]
        assert_eq!(result.data.as_slice().unwrap(), &expected);
    }
    
    #[test]
    fn test_simd_parallel_add() {
        let a = Tensor::<f32>::from_vec(
            (0..1000).map(|i| i as f32).collect(),
            vec![10, 100]
        );
        let b = Tensor::<f32>::from_vec(
            vec![1.0; 1000],
            vec![10, 100]
        );
        
        let result = a.simd_parallel_add(&b).unwrap();
        
        if let Some(slice) = result.data.as_slice() {
            for (i, &val) in slice.iter().enumerate() {
                assert_eq!(val, i as f32 + 1.0);
            }
        }
    }
}

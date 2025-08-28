//! CPU Fallback Operations for GPU Memory
//! GPUメモリのCPUフォールバック操作

use crate::error::{RusTorchError, RusTorchResult};
use num_traits::Float;
use std::sync::Arc;

use super::buffer::GpuBuffer;

/// CPU fallback operations for GPU memory operations
/// GPUメモリ操作のCPUフォールバック操作
pub struct CpuFallback;

impl CpuFallback {
    /// Execute element-wise operation using CPU fallback
    /// CPUフォールバックを使用した要素ごとの演算実行
    pub fn execute_elementwise<T, F>(
        lhs: &GpuBuffer<T>,
        rhs: &GpuBuffer<T>,
        op: &F,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: Float,
        F: Fn(T, T) -> T + Send + Sync,
    {
        let lhs_data = match lhs {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
        };

        let rhs_data = match rhs {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU fallback requires CPU buffers"))
            }
        };

        let result: Vec<T> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Ok(GpuBuffer::Cpu(Arc::new(result)))
    }

    /// Execute batch normalization using CPU fallback
    /// CPUフォールバックを使用したバッチ正規化実行
    pub fn execute_batch_normalize<T>(
        data: &Arc<Vec<T>>,
        epsilon: T,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: Float,
    {
        let input_data = data.as_slice();
        let n = input_data.len();

        if n == 0 {
            return Ok(GpuBuffer::Cpu(data.clone()));
        }

        // 平均計算
        let mean = input_data.iter().fold(T::zero(), |acc, &x| acc + x)
            / T::from(n).ok_or_else(|| RusTorchError::gpu("Failed to convert size to float"))?;

        // 分散計算
        let variance = input_data
            .iter()
            .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean))
            / T::from(n).ok_or_else(|| RusTorchError::gpu("Failed to convert size to float"))?;

        // 正規化
        let std_dev = (variance + epsilon).sqrt();
        let normalized: Vec<T> = input_data.iter().map(|&x| (x - mean) / std_dev).collect();

        Ok(GpuBuffer::Cpu(Arc::new(normalized)))
    }

    /// Execute attention mechanism using CPU fallback
    /// CPUフォールバックを使用したアテンション機構実行
    pub fn execute_attention<T>(
        query: &GpuBuffer<T>,
        key: &GpuBuffer<T>,
        value: &GpuBuffer<T>,
    ) -> RusTorchResult<GpuBuffer<T>>
    where
        T: Float,
    {
        // Extract data from all buffers (assuming all are CPU buffers for fallback)
        let query_data = match query {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
        };

        let key_data = match key {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
        };

        let value_data = match value {
            GpuBuffer::Cpu(data) => data.as_slice(),
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "metal")]
            GpuBuffer::Metal { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
            #[cfg(feature = "opencl")]
            GpuBuffer::OpenCL { .. } => {
                return Err(RusTorchError::gpu("CPU attention requires CPU buffers"))
            }
        };

        // 簡単な行列乗算ベースのアテンション（単純化版）
        // scores = query @ key^T
        let scores: Vec<T> = query_data
            .iter()
            .zip(key_data.iter())
            .map(|(&q, &k)| q * k) // 簡単なドット積近似
            .collect();

        // softmax適用（簡単版）
        let max_score = scores
            .iter()
            .fold(T::neg_infinity(), |max, &x| if x > max { x } else { max });

        let exp_scores: Vec<T> = scores.iter().map(|&x| (x - max_score).exp()).collect();

        let sum_exp = exp_scores.iter().fold(T::zero(), |acc, &x| acc + x);

        let attention_weights: Vec<T> = exp_scores.iter().map(|&x| x / sum_exp).collect();

        // 重み付きvalue計算
        let result: Vec<T> = attention_weights
            .iter()
            .zip(value_data.iter())
            .map(|(&w, &v)| w * v)
            .collect();

        Ok(GpuBuffer::Cpu(Arc::new(result)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_elementwise_addition() {
        let lhs = GpuBuffer::Cpu(Arc::new(vec![1.0f32, 2.0, 3.0]));
        let rhs = GpuBuffer::Cpu(Arc::new(vec![4.0f32, 5.0, 6.0]));

        let result = CpuFallback::execute_elementwise(&lhs, &rhs, &|a, b| a + b).unwrap();

        if let GpuBuffer::Cpu(data) = result {
            assert_eq!(data.as_ref(), &vec![5.0, 7.0, 9.0]);
        } else {
            panic!("Expected CPU buffer");
        }
    }

    #[test]
    fn test_cpu_batch_normalize() {
        let data = Arc::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let epsilon = 1e-5f32;

        let result = CpuFallback::execute_batch_normalize(&data, epsilon).unwrap();

        if let GpuBuffer::Cpu(normalized_data) = result {
            // Check that the normalized data has zero mean (approximately)
            let mean: f32 = normalized_data.iter().sum::<f32>() / normalized_data.len() as f32;
            assert!((mean.abs()) < 1e-6, "Mean should be approximately zero, got {}", mean);
        } else {
            panic!("Expected CPU buffer");
        }
    }

    #[test]
    fn test_cpu_attention() {
        let query = GpuBuffer::Cpu(Arc::new(vec![1.0f32, 0.5]));
        let key = GpuBuffer::Cpu(Arc::new(vec![0.8f32, 1.2]));
        let value = GpuBuffer::Cpu(Arc::new(vec![2.0f32, 3.0]));

        let result = CpuFallback::execute_attention(&query, &key, &value).unwrap();

        if let GpuBuffer::Cpu(attention_result) = result {
            assert_eq!(attention_result.len(), 2);
            // Check that we got some reasonable values
            assert!(attention_result.iter().all(|&x| x.is_finite()));
        } else {
            panic!("Expected CPU buffer");
        }
    }
}
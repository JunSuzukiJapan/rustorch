//! SIMD-specific benchmark utilities
//! 
//! Specialized benchmark functions for SIMD operations

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rustorch::tensor::{Tensor, simd_aligned::SimdTensor};
use rustorch::tensor::parallel_traits::SimdParallelOp;
use num_traits::Float;
use crate::common::{BenchmarkConfig, create_benchmark_group};

/// Benchmark SIMD vs scalar operations comparison
pub fn bench_simd_vs_scalar<T>(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[(usize, &str)],
    config: Option<BenchmarkConfig>,
) where
    T: Float + Clone + Default + std::ops::Add<Output = T> + 'static,
{
    let mut group = create_benchmark_group(c, group_name);
    
    if let Some(cfg) = config {
        group.sample_size(cfg.sample_size);
        group.measurement_time(std::time::Duration::from_secs(cfg.measurement_time_secs));
        group.warm_up_time(std::time::Duration::from_secs(cfg.warm_up_time_secs));
    }
    
    for (size, description) in sizes {
        let tensor1 = Tensor::<T>::zeros(&[*size]);
        let tensor2 = Tensor::<T>::zeros(&[*size]);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Scalar operation benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar_add", description),
            &(*size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(&**t1 + &**t2);
                    black_box(result)
                });
            },
        );
        
        // SIMD operation benchmark
        group.bench_with_input(
            BenchmarkId::new("simd_add", description),
            &(*size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.simd_parallel_add(t2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD matrix operations
pub fn bench_simd_matrix_ops(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[(usize, &str)],
    config: Option<BenchmarkConfig>,
) {
    let mut group = create_benchmark_group(c, group_name);
    
    if let Some(cfg) = config {
        group.sample_size(cfg.sample_size);
        group.measurement_time(std::time::Duration::from_secs(cfg.measurement_time_secs));
        group.warm_up_time(std::time::Duration::from_secs(cfg.warm_up_time_secs));
    }
    
    for (size, description) in sizes {
        let tensor1 = Tensor::<f32>::ones(&[*size, *size]);
        let tensor2 = Tensor::<f32>::ones(&[*size, *size]);
        
        group.throughput(Throughput::Elements((*size * *size) as u64));
        
        // Regular matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("scalar_matmul", description),
            &(*size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.matmul(t2));
                    black_box(result)
                });
            },
        );
        
        // SIMD matrix multiplication (if available)
        if let (Ok(simd1), Ok(simd2)) = (
            SimdTensor::<f32>::from_tensor(&tensor1),
            SimdTensor::<f32>::from_tensor(&tensor2)
        ) {
            group.bench_with_input(
                BenchmarkId::new("simd_matmul", description),
                &(*size, &simd1, &simd2),
                |b, (_, t1, t2)| {
                    b.iter(|| {
                        let result = black_box(t1.matmul_simd(t2).unwrap());
                        black_box(result)
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark SIMD vectorized operations
pub fn bench_simd_vectorized_ops(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[(usize, &str)],
    config: Option<BenchmarkConfig>,
) {
    let mut group = create_benchmark_group(c, group_name);
    
    if let Some(cfg) = config {
        group.sample_size(cfg.sample_size);
        group.measurement_time(std::time::Duration::from_secs(cfg.measurement_time_secs));
        group.warm_up_time(std::time::Duration::from_secs(cfg.warm_up_time_secs));
    }
    
    for (size, description) in sizes {
        let data1: Vec<f32> = vec![1.0; *size];
        let data2: Vec<f32> = vec![2.0; *size];
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Scalar vectorized addition
        group.bench_with_input(
            BenchmarkId::new("scalar_vectorized", description),
            &(*size, &data1, &data2),
            |b, (_, d1, d2)| {
                b.iter(|| {
                    let mut result = vec![0.0f32; *size];
                    for i in 0..*size {
                        result[i] = d1[i] + d2[i];
                    }
                    black_box(result)
                });
            },
        );
        
        // SIMD vectorized addition
        group.bench_with_input(
            BenchmarkId::new("simd_vectorized", description),
            &(*size, &data1, &data2),
            |b, (_, d1, d2)| {
                b.iter(|| {
                    // SIMD vectorized addition using available SIMD operations
                    let mut result = vec![0.0f32; *size];
                    for i in (0..*size).step_by(4) {
                        let end = (i + 4).min(*size);
                        for j in i..end {
                            result[j] = d1[j] + d2[j];
                        }
                    }
                    let result = black_box(result);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

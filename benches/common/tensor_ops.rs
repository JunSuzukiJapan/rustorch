//! Common tensor operation benchmarks
//! 
//! Reusable benchmark functions for standard tensor operations

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rustorch::tensor::Tensor;
use num_traits::Float;
use crate::common::{BenchmarkConfig, create_benchmark_group, elements_count};

/// Benchmark element-wise addition operations
pub fn bench_elementwise_add<T>(
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
        
        group.bench_with_input(
            BenchmarkId::new("elementwise_add", description),
            &(*size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(&**t1 + &**t2);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark matrix multiplication operations
pub fn bench_matrix_multiply<T>(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[(usize, &str)],
    config: Option<BenchmarkConfig>,
) where
    T: Float + Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + 'static,
{
    let mut group = create_benchmark_group(c, group_name);
    
    if let Some(cfg) = config {
        group.sample_size(cfg.sample_size);
        group.measurement_time(std::time::Duration::from_secs(cfg.measurement_time_secs));
        group.warm_up_time(std::time::Duration::from_secs(cfg.warm_up_time_secs));
    }
    
    for (size, description) in sizes {
        let tensor1 = Tensor::<T>::zeros(&[*size, *size]);
        let tensor2 = Tensor::<T>::zeros(&[*size, *size]);
        
        group.throughput(Throughput::Elements(elements_count(&[*size, *size])));
        
        group.bench_with_input(
            BenchmarkId::new("matrix_multiply", description),
            &(*size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.matmul(t2));
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark tensor allocation and deallocation
pub fn bench_tensor_allocation<T>(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[(usize, &str)],
    config: Option<BenchmarkConfig>,
) where
    T: Float + Clone + Default + 'static,
{
    let mut group = create_benchmark_group(c, group_name);
    
    if let Some(cfg) = config {
        group.sample_size(cfg.sample_size);
        group.measurement_time(std::time::Duration::from_secs(cfg.measurement_time_secs));
        group.warm_up_time(std::time::Duration::from_secs(cfg.warm_up_time_secs));
    }
    
    for (size, description) in sizes {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("tensor_allocation", description),
            size,
            |b, sz| {
                b.iter(|| {
                    let tensor = black_box(Tensor::<T>::zeros(&[*sz]));
                    black_box(tensor)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark tensor copying operations
pub fn bench_tensor_copy<T>(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[(usize, &str)],
    config: Option<BenchmarkConfig>,
) where
    T: Float + Clone + Default + 'static,
{
    let mut group = create_benchmark_group(c, group_name);
    
    if let Some(cfg) = config {
        group.sample_size(cfg.sample_size);
        group.measurement_time(std::time::Duration::from_secs(cfg.measurement_time_secs));
        group.warm_up_time(std::time::Duration::from_secs(cfg.warm_up_time_secs));
    }
    
    for (size, description) in sizes {
        let tensor = Tensor::<T>::zeros(&[*size]);
        
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("tensor_copy", description),
            &(*size, &tensor),
            |b, (_, t)| {
                b.iter(|| {
                    let copied = black_box(t.clone());
                    black_box(copied)
                });
            },
        );
    }
    
    group.finish();
}

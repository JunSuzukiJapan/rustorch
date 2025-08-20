//! Memory operation benchmarks
//! 
//! Specialized benchmark functions for memory allocation, copying, and optimization strategies

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use rustorch::tensor::{Tensor, zero_copy::SharedTensor};
use num_traits::Float;
use crate::common::{BenchmarkConfig, create_benchmark_group};

/// Benchmark different memory allocation strategies
pub fn bench_allocation_strategies<T>(
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
        let tensor_size = *size;
        let elements = tensor_size * tensor_size;
        
        group.throughput(Throughput::Elements(elements as u64));
        
        // Direct allocation benchmark
        group.bench_with_input(
            BenchmarkId::new("direct_allocation", description),
            &tensor_size,
            |b, sz| {
                b.iter(|| {
                    let tensor1 = Tensor::<T>::zeros(&[*sz, *sz]);
                    let tensor2 = Tensor::<T>::zeros(&[*sz, *sz]);
                    let result = black_box(&tensor1 + &tensor2);
                    black_box(result)
                });
            },
        );
        
        // Pool allocation simulation (simplified)
        group.bench_with_input(
            BenchmarkId::new("pool_allocation", description),
            &tensor_size,
            |b, sz| {
                b.iter(|| {
                    let tensor1 = Tensor::<T>::zeros(&[*sz, *sz]);
                    let tensor2 = Tensor::<T>::zeros(&[*sz, *sz]);
                    let result = black_box(&tensor1 + &tensor2);
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark zero-copy operations vs regular copying
pub fn bench_zero_copy_operations<T>(
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
        
        // Regular copy benchmark
        group.bench_with_input(
            BenchmarkId::new("regular_copy", description),
            &(*size, &tensor),
            |b, (_, t)| {
                b.iter(|| {
                    let copied = black_box(t.clone());
                    black_box(copied)
                });
            },
        );
        
        // Zero-copy view benchmark
        group.bench_with_input(
            BenchmarkId::new("zero_copy_view", description),
            &(*size, &tensor),
            |b, (_, t)| {
                b.iter(|| {
                    let view = black_box(t.slice_view(&[0..*size]).unwrap());
                    black_box(view)
                });
            },
        );
        
        // Shared tensor benchmark
        group.bench_with_input(
            BenchmarkId::new("shared_tensor", description),
            &(*size, &tensor),
            |b, (_, t)| {
                b.iter(|| {
                    let shared = black_box(SharedTensor::new(t.clone()));
                    black_box(shared)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory pool operations with different allocation counts
pub fn bench_memory_pool_operations<T>(
    c: &mut Criterion,
    group_name: &str,
    allocation_counts: &[(usize, &str)],
    tensor_size: usize,
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
    
    for (count, description) in allocation_counts {
        group.throughput(Throughput::Elements((*count * tensor_size) as u64));
        
        // Direct allocations
        group.bench_with_input(
            BenchmarkId::new("direct_allocations", description),
            &(*count, tensor_size),
            |b, (cnt, size)| {
                b.iter(|| {
                    let mut tensors = Vec::new();
                    for _ in 0..*cnt {
                        let tensor = Tensor::<T>::zeros(&[*size]);
                        tensors.push(tensor);
                    }
                    black_box(tensors)
                });
            },
        );
        
        // Pool allocations (simulated)
        group.bench_with_input(
            BenchmarkId::new("pool_allocations", description),
            &(*count, tensor_size),
            |b, (cnt, size)| {
                b.iter(|| {
                    let mut tensors = Vec::new();
                    for _ in 0..*cnt {
                        let tensor = Tensor::<T>::zeros(&[*size]);
                        tensors.push(tensor);
                    }
                    black_box(tensors)
                });
            },
        );
    }
    
    group.finish();
}

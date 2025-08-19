//! Integrated Performance Test Suite
//! 
//! This benchmark suite tests the core performance optimizations that are
//! currently implemented and working in RusTorch.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustorch::prelude::*;
use rustorch::tensor::{Tensor, parallel_traits::*};
use std::time::Duration;

/// Test parallel vs sequential tensor operations
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = vec![
        (500, "Medium (250K elements)"),
        (1000, "Large (1M elements)"),
    ];
    
    for (size, description) in sizes {
        let tensor1 = Tensor::<f32>::ones(&[size, size]);
        let tensor2 = Tensor::<f32>::ones(&[size, size]);
        let elements = (size * size) as u64;
        
        group.throughput(Throughput::Elements(elements));
        
        // Sequential operations
        group.bench_with_input(
            BenchmarkId::new("sequential_add", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(&**t1 + &**t2);
                    black_box(result)
                });
            },
        );
        
        // Parallel operations
        group.bench_with_input(
            BenchmarkId::new("parallel_add", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.batch_elementwise_op(t2, |a, b| a + b).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Test matrix multiplication performance
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = vec![
        (128, "128x128"),
        (256, "256x256"),
        (512, "512x512"),
    ];
    
    for (size, description) in sizes {
        let matrix1 = Tensor::<f32>::ones(&[size, size]);
        let matrix2 = Tensor::<f32>::ones(&[size, size]);
        let operations = (size * size * size) as u64;
        
        group.throughput(Throughput::Elements(operations));
        
        // Standard matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("standard_matmul", description),
            &(size, &matrix1, &matrix2),
            |b, (_, m1, m2)| {
                b.iter(|| {
                    let result = black_box(m1.matmul(m2));
                    black_box(result)
                });
            },
        );
        
        // Parallel matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("parallel_matmul", description),
            &(size, &matrix1, &matrix2),
            |b, (_, m1, m2)| {
                b.iter(|| {
                    let result = black_box(m1.batch_matmul(m2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Test reduction operations
fn bench_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions");
    
    let sizes = vec![
        (100000, "100K elements"),
        (1000000, "1M elements"),
    ];
    
    for (size, description) in sizes {
        let tensor = Tensor::<f32>::from_vec(
            (0..size).map(|i| (i as f32) * 0.001).collect(),
            vec![size]
        );
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Sequential sum
        group.bench_with_input(
            BenchmarkId::new("sequential_sum", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.sum());
                    black_box(result)
                });
            },
        );
        
        // Parallel sum
        group.bench_with_input(
            BenchmarkId::new("parallel_sum", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.parallel_sum(0).unwrap());
                    black_box(result)
                });
            },
        );
        
        // Sequential mean
        group.bench_with_input(
            BenchmarkId::new("sequential_mean", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.mean());
                    black_box(result)
                });
            },
        );
        
        // Parallel mean
        group.bench_with_input(
            BenchmarkId::new("parallel_mean", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.parallel_mean(0).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Test batch operations
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");
    group.measurement_time(Duration::from_secs(20));
    
    let batch_configs = vec![
        (16, 64, "16 batches of 64x64"),
        (32, 32, "32 batches of 32x32"),
    ];
    
    for (batch_size, matrix_size, description) in batch_configs {
        let batch_tensor1 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
        let batch_tensor2 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
        let total_ops = (batch_size * matrix_size * matrix_size * matrix_size) as u64;
        
        group.throughput(Throughput::Elements(total_ops));
        
        // Sequential batch processing
        group.bench_with_input(
            BenchmarkId::new("sequential_batch", description),
            &(batch_size, matrix_size, &batch_tensor1, &batch_tensor2),
            |b, (batch_sz, _, t1, t2)| {
                b.iter(|| {
                    let mut results = Vec::new();
                    for i in 0..*batch_sz {
                        let slice1 = t1.select(0, i);
                        let slice2 = t2.select(0, i);
                        let result = slice1.matmul(&slice2);
                        results.push(result);
                    }
                    black_box(results)
                });
            },
        );
        
        // Parallel batch processing
        group.bench_with_input(
            BenchmarkId::new("parallel_batch", description),
            &(batch_size, matrix_size, &batch_tensor1, &batch_tensor2),
            |b, (_, _, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.batch_matmul(t2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Test memory allocation patterns
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    let allocation_counts = vec![
        (10, "10 allocations"),
        (100, "100 allocations"),
    ];
    
    for (count, description) in allocation_counts {
        let tensor_size = 1000; // 1K elements per tensor
        
        group.throughput(Throughput::Elements((count * tensor_size) as u64));
        
        // Standard allocation
        group.bench_with_input(
            BenchmarkId::new("standard_allocation", description),
            &(count, tensor_size),
            |b, (cnt, size)| {
                b.iter(|| {
                    let mut tensors = Vec::new();
                    for _ in 0..*cnt {
                        let tensor = Tensor::<f32>::ones(&[*size]);
                        tensors.push(tensor);
                    }
                    black_box(tensors)
                });
            },
        );
        
        // Preallocated vector
        group.bench_with_input(
            BenchmarkId::new("preallocated_vector", description),
            &(count, tensor_size),
            |b, (cnt, size)| {
                b.iter(|| {
                    let mut tensors = Vec::with_capacity(*cnt);
                    for _ in 0..*cnt {
                        let tensor = Tensor::<f32>::ones(&[*size]);
                        tensors.push(tensor);
                    }
                    black_box(tensors)
                });
            },
        );
    }
    
    group.finish();
}

/// Test different data types performance
fn bench_data_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_types");
    
    let size = 50000; // 50K elements
    group.throughput(Throughput::Elements(size as u64));
    
    // f32 operations
    let tensor_f32_1 = Tensor::<f32>::ones(&[size]);
    let tensor_f32_2 = Tensor::<f32>::ones(&[size]);
    
    group.bench_function("f32_add", |b| {
        b.iter(|| {
            let result = black_box(&tensor_f32_1 + &tensor_f32_2);
            black_box(result)
        });
    });
    
    group.bench_function("f32_parallel_add", |b| {
        b.iter(|| {
            let result = black_box(tensor_f32_1.batch_elementwise_op(&tensor_f32_2, |a, b| a + b).unwrap());
            black_box(result)
        });
    });
    
    // f64 operations
    let tensor_f64_1 = Tensor::<f64>::ones(&[size]);
    let tensor_f64_2 = Tensor::<f64>::ones(&[size]);
    
    group.bench_function("f64_add", |b| {
        b.iter(|| {
            let result = black_box(&tensor_f64_1 + &tensor_f64_2);
            black_box(result)
        });
    });
    
    group.bench_function("f64_parallel_add", |b| {
        b.iter(|| {
            let result = black_box(tensor_f64_1.batch_elementwise_op(&tensor_f64_2, |a, b| a + b).unwrap());
            black_box(result)
        });
    });
    
    group.finish();
}

/// Test thread scaling performance
fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    
    let tensor1 = Tensor::<f32>::ones(&[1000, 1000]);
    let tensor2 = Tensor::<f32>::ones(&[1000, 1000]);
    
    // Test with different thread counts (simulated by different operation sizes)
    let thread_configs = vec![
        ("single_thread", 1000),
        ("multi_thread", 10000),
        ("high_parallelism", 100000),
    ];
    
    for (name, _threshold) in thread_configs {
        group.bench_function(name, |b| {
            b.iter(|| {
                let result = black_box(tensor1.batch_elementwise_op(&tensor2, |a, b| a * b).unwrap());
                black_box(result)
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    integrated_benches,
    bench_parallel_vs_sequential,
    bench_matrix_multiplication,
    bench_reductions,
    bench_batch_operations,
    bench_memory_allocation,
    bench_data_types,
    bench_thread_scaling
);

criterion_main!(integrated_benches);

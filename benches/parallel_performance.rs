//! Parallel Processing Performance Benchmarks
//! 
//! This benchmark suite measures the performance improvements achieved through
//! parallel tensor operations compared to sequential processing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustorch::tensor::{Tensor, parallel_traits::*};
use std::time::Duration;

/// Benchmark parallel vs sequential element-wise operations
fn bench_parallel_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_elementwise");
    
    // Test different tensor sizes
    let sizes = vec![
        (100, "Small (10K elements)"),
        (500, "Medium (250K elements)"),
        (1000, "Large (1M elements)"),
        (2000, "XLarge (4M elements)"),
    ];
    
    for (size, description) in sizes {
        let tensor1 = Tensor::<f32>::ones(&[size, size]);
        let tensor2 = Tensor::<f32>::ones(&[size, size]);
        let elements = size * size;
        
        group.throughput(Throughput::Elements(elements as u64));
        
        // Sequential benchmark (using regular tensor operations)
        group.bench_with_input(
            BenchmarkId::new("sequential", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(&**t1 + &**t2);
                    black_box(result)
                });
            },
        );
        
        // Parallel benchmark
        group.bench_with_input(
            BenchmarkId::new("parallel", description),
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

/// Benchmark parallel matrix multiplication performance
fn bench_parallel_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_matmul");
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = vec![
        (64, "Small (64x64)"),
        (128, "Medium (128x128)"),
        (256, "Large (256x256)"),
        (512, "XLarge (512x512)"),
    ];
    
    for (size, description) in sizes {
        let tensor1 = Tensor::<f32>::ones(&[size, size]);
        let tensor2 = Tensor::<f32>::ones(&[size, size]);
        let operations = (size * size * size) as u64; // O(n³) operations
        
        group.throughput(Throughput::Elements(operations));
        
        // Sequential matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("sequential", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.matmul(t2));
                    black_box(result)
                });
            },
        );
        
        // Parallel matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("parallel", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.batch_matmul(t2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel reduction operations
fn bench_parallel_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_reduction");
    
    let sizes = vec![
        (1000, "1K elements"),
        (10000, "10K elements"),
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
                    let result = black_box(t.mean(None));
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

/// Benchmark batch processing performance
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");
    group.measurement_time(Duration::from_secs(15));
    
    let batch_configs = vec![
        (8, 64, "Small batch (8x64x64)"),
        (16, 128, "Medium batch (16x128x128)"),
        (32, 64, "Large batch (32x64x64)"),
        (64, 32, "XLarge batch (64x32x32)"),
    ];
    
    for (batch_size, matrix_size, description) in batch_configs {
        let batch_tensor1 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
        let batch_tensor2 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
        let total_ops = (batch_size * matrix_size * matrix_size * matrix_size) as u64;
        
        group.throughput(Throughput::Elements(total_ops));
        
        // Sequential batch processing (process each matrix individually)
        group.bench_with_input(
            BenchmarkId::new("sequential_batch", description),
            &(batch_size, matrix_size, &batch_tensor1, &batch_tensor2),
            |b, (batch_sz, mat_sz, t1, t2)| {
                b.iter(|| {
                    let mut results = Vec::new();
                    for i in 0..*batch_sz {
                        let slice1 = t1.select(0, &[i]).unwrap();
                        let slice2 = t2.select(0, &[i]).unwrap();
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

/// Benchmark thread scaling performance
fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    group.measurement_time(Duration::from_secs(20));
    
    let tensor1 = Tensor::<f32>::ones(&[1000, 1000]);
    let tensor2 = Tensor::<f32>::ones(&[1000, 1000]);
    
    // Test different thread counts
    let thread_counts = vec![1, 2, 4, 8];
    
    for thread_count in thread_counts {
        // Configure Rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("elementwise", format!("{}_threads", thread_count)),
            &thread_count,
            |b, _| {
                b.iter(|| {
                    pool.install(|| {
                        let result = black_box(tensor1.batch_elementwise_op(&tensor2, |a, b| a + b).unwrap());
                        black_box(result)
                    });
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("matmul", format!("{}_threads", thread_count)),
            &thread_count,
            |b, _| {
                b.iter(|| {
                    pool.install(|| {
                        let result = black_box(tensor1.batch_matmul(&tensor2).unwrap());
                        black_box(result)
                    });
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel execution strategies
fn bench_execution_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution_strategies");
    
    let tensor1 = Tensor::<f32>::ones(&[500, 500]);
    let tensor2 = Tensor::<f32>::ones(&[500, 500]);
    
    // Test different execution strategies
    let strategies = vec![
        ("auto", ParallelStrategy::Auto),
        ("force_parallel", ParallelStrategy::ForceParallel),
        ("cpu_parallel", ParallelStrategy::CpuParallel),
    ];
    
    for (name, strategy) in strategies {
        let config = ParallelConfig {
            strategy,
            chunk_size: 1024,
            num_threads: Some(4),
        };
        
        group.bench_with_input(
            BenchmarkId::new("elementwise", name),
            &config,
            |b, cfg| {
                b.iter(|| {
                    // Note: In a real implementation, we would pass the config to the operation
                    let result = black_box(tensor1.batch_elementwise_op(&tensor2, |a, b| a + b).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    parallel_benches,
    bench_parallel_elementwise,
    bench_parallel_matmul,
    bench_parallel_reduction,
    bench_batch_operations,
    bench_thread_scaling,
    bench_execution_strategies
);

criterion_main!(parallel_benches);

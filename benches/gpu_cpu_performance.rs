//! GPU vs CPU Performance Benchmarks
//! 
//! This benchmark suite compares performance between GPU-accelerated operations
//! and CPU-based operations, including fallback scenarios.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustorch::prelude::*;
use rustorch::tensor::{Tensor, gpu_parallel::*, parallel_traits::*};
use rustorch::gpu::DeviceType;
use std::time::Duration;

/// Benchmark GPU vs CPU element-wise operations
fn bench_gpu_cpu_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_cpu_elementwise");
    group.measurement_time(Duration::from_secs(10));
    
    let sizes = vec![
        (500, "Medium (250K elements)"),
        (1000, "Large (1M elements)"),
        (2000, "XLarge (4M elements)"),
    ];
    
    for (size, description) in sizes {
        let tensor1 = Tensor::<f32>::ones(&[size, size]);
        let tensor2 = Tensor::<f32>::ones(&[size, size]);
        let elements = (size * size) as u64;
        
        group.throughput(Throughput::Elements(elements));
        
        // CPU parallel operations
        group.bench_with_input(
            BenchmarkId::new("cpu_parallel", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.batch_elementwise_op(t2, |a, b| a + b).unwrap());
                    black_box(result)
                });
            },
        );
        
        // GPU operations (with automatic fallback to CPU if GPU unavailable)
        group.bench_with_input(
            BenchmarkId::new("gpu_with_fallback", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.gpu_elementwise_op(t2, |a, b| a + b).unwrap());
                    black_box(result)
                });
            },
        );
        
        // CPU sequential operations (baseline)
        group.bench_with_input(
            BenchmarkId::new("cpu_sequential", description),
            &(size, &tensor1, &tensor2),
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

/// Benchmark GPU vs CPU matrix multiplication
fn bench_gpu_cpu_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_cpu_matmul");
    group.measurement_time(Duration::from_secs(15));
    
    let sizes = vec![
        (128, "128x128"),
        (256, "256x256"),
        (512, "512x512"),
        (1024, "1024x1024"),
    ];
    
    for (size, description) in sizes {
        let matrix1 = Tensor::<f32>::ones(&[size, size]);
        let matrix2 = Tensor::<f32>::ones(&[size, size]);
        let operations = (size * size * size) as u64;
        
        group.throughput(Throughput::Elements(operations));
        
        // CPU matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("cpu_matmul", description),
            &(size, &matrix1, &matrix2),
            |b, (_, m1, m2)| {
                b.iter(|| {
                    let result = black_box(m1.matmul(m2));
                    black_box(result)
                });
            },
        );
        
        // CPU parallel matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("cpu_parallel_matmul", description),
            &(size, &matrix1, &matrix2),
            |b, (_, m1, m2)| {
                b.iter(|| {
                    let result = black_box(m1.batch_matmul(m2).unwrap());
                    black_box(result)
                });
            },
        );
        
        // GPU matrix multiplication (with fallback)
        group.bench_with_input(
            BenchmarkId::new("gpu_matmul", description),
            &(size, &matrix1, &matrix2),
            |b, (_, m1, m2)| {
                b.iter(|| {
                    let result = black_box(m1.gpu_matmul(m2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark GPU vs CPU batch operations
fn bench_gpu_cpu_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_cpu_batch_operations");
    group.measurement_time(Duration::from_secs(20));
    
    let batch_configs = vec![
        (16, 64, "16 batches of 64x64"),
        (32, 128, "32 batches of 128x128"),
        (64, 64, "64 batches of 64x64"),
    ];
    
    for (batch_size, matrix_size, description) in batch_configs {
        let batch_tensor1 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
        let batch_tensor2 = Tensor::<f32>::ones(&[batch_size, matrix_size, matrix_size]);
        let total_ops = (batch_size * matrix_size * matrix_size * matrix_size) as u64;
        
        group.throughput(Throughput::Elements(total_ops));
        
        // CPU batch processing
        group.bench_with_input(
            BenchmarkId::new("cpu_batch", description),
            &(batch_size, matrix_size, &batch_tensor1, &batch_tensor2),
            |b, (_, _, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.batch_matmul(t2).unwrap());
                    black_box(result)
                });
            },
        );
        
        // GPU batch processing (with fallback)
        group.bench_with_input(
            BenchmarkId::new("gpu_batch", description),
            &(batch_size, matrix_size, &batch_tensor1, &batch_tensor2),
            |b, (_, _, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.gpu_batch_matmul(t2).unwrap());
                    black_box(result)
                });
            },
        );
        
        // Sequential batch processing (baseline)
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
    }
    
    group.finish();
}

/// Benchmark GPU vs CPU reduction operations
fn bench_gpu_cpu_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_cpu_reductions");
    
    let sizes = vec![
        (100000, "100K elements"),
        (1000000, "1M elements"),
        (10000000, "10M elements"),
    ];
    
    for (size, description) in sizes {
        let tensor = Tensor::<f32>::from_vec(
            (0..size).map(|i| (i as f32) * 0.001).collect(),
            vec![size]
        );
        
        group.throughput(Throughput::Elements(size as u64));
        
        // CPU sequential sum
        group.bench_with_input(
            BenchmarkId::new("cpu_sequential_sum", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.sum());
                    black_box(result)
                });
            },
        );
        
        // CPU parallel sum
        group.bench_with_input(
            BenchmarkId::new("cpu_parallel_sum", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.parallel_sum(0).unwrap());
                    black_box(result)
                });
            },
        );
        
        // GPU sum (with fallback)
        group.bench_with_input(
            BenchmarkId::new("gpu_sum", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.gpu_sum(0).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark GPU execution strategies
fn bench_gpu_execution_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_execution_strategies");
    
    let tensor1 = Tensor::<f32>::ones(&[1000, 1000]);
    let tensor2 = Tensor::<f32>::ones(&[1000, 1000]);
    
    let strategies = vec![
        ("cpu_parallel", GpuExecutionStrategy::CpuParallel),
        ("gpu_preferred", GpuExecutionStrategy::GpuPreferred { fallback_threshold: 10000 }),
        ("hybrid", GpuExecutionStrategy::Hybrid { gpu_threshold: 50000, cpu_threads: 4 }),
        ("auto", GpuExecutionStrategy::Auto),
    ];
    
    for (name, strategy) in strategies {
        group.bench_with_input(
            BenchmarkId::new("elementwise", name),
            &strategy,
            |b, _| {
                b.iter(|| {
                    // Note: In a real implementation, we would use the strategy to determine execution path
                    let result = black_box(tensor1.gpu_elementwise_op(&tensor2, |a, b| a + b).unwrap());
                    black_box(result)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("matmul", name),
            &strategy,
            |b, _| {
                b.iter(|| {
                    let result = black_box(tensor1.gpu_matmul(&tensor2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark GPU memory transfer overhead
fn bench_gpu_memory_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_memory_transfer");
    
    let sizes = vec![
        (1000, "1K elements"),
        (10000, "10K elements"),
        (100000, "100K elements"),
        (1000000, "1M elements"),
    ];
    
    for (size, description) in sizes {
        let tensor = Tensor::<f32>::ones(&[size]);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // CPU-only operation (no transfer)
        group.bench_with_input(
            BenchmarkId::new("cpu_only", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.sum());
                    black_box(result)
                });
            },
        );
        
        // GPU operation with transfer overhead (simulated)
        group.bench_with_input(
            BenchmarkId::new("gpu_with_transfer", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    // Simulate GPU transfer and computation
                    match t.to_device(DeviceType::Cuda(0)) {
                        Ok(gpu_tensor) => {
                            let result = gpu_tensor.gpu_sum(0).unwrap_or_else(|_| t.sum());
                            let cpu_result = result.to_cpu().unwrap_or(result);
                            black_box(cpu_result)
                        }
                        Err(_) => {
                            // Fallback to CPU
                            let result = black_box(t.sum());
                            black_box(result)
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark device selection performance
fn bench_device_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_selection");
    
    let tensor_sizes = vec![
        ([100, 100], "Small tensor"),
        ([1000, 1000], "Large tensor"),
        ([2000, 2000], "XLarge tensor"),
    ];
    
    for (shape, description) in tensor_sizes {
        group.bench_with_input(
            BenchmarkId::new("optimal_device_selection", description),
            &shape,
            |b, sh| {
                b.iter(|| {
                    let result = black_box(select_optimal_device(sh));
                    black_box(result)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("device_availability_check", description),
            &shape,
            |b, _| {
                b.iter(|| {
                    let cuda_available = black_box(DeviceType::Cuda(0).is_available());
                    let metal_available = black_box(DeviceType::Metal(0).is_available());
                    let opencl_available = black_box(DeviceType::OpenCl(0).is_available());
                    black_box((cuda_available, metal_available, opencl_available))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark GPU vs CPU with different data types
fn bench_gpu_cpu_data_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_cpu_data_types");
    
    let size = 50000; // 50K elements
    
    // f32 benchmarks
    let tensor_f32_1 = Tensor::<f32>::ones(&[size]);
    let tensor_f32_2 = Tensor::<f32>::ones(&[size]);
    
    group.throughput(Throughput::Elements(size as u64));
    
    group.bench_function("f32_cpu_add", |b| {
        b.iter(|| {
            let result = black_box(&tensor_f32_1 + &tensor_f32_2);
            black_box(result)
        });
    });
    
    group.bench_function("f32_gpu_add", |b| {
        b.iter(|| {
            let result = black_box(tensor_f32_1.gpu_elementwise_op(&tensor_f32_2, |a, b| a + b).unwrap());
            black_box(result)
        });
    });
    
    // f64 benchmarks
    let tensor_f64_1 = Tensor::<f64>::ones(&[size]);
    let tensor_f64_2 = Tensor::<f64>::ones(&[size]);
    
    group.bench_function("f64_cpu_add", |b| {
        b.iter(|| {
            let result = black_box(&tensor_f64_1 + &tensor_f64_2);
            black_box(result)
        });
    });
    
    group.bench_function("f64_gpu_add", |b| {
        b.iter(|| {
            let result = black_box(tensor_f64_1.gpu_elementwise_op(&tensor_f64_2, |a, b| a + b).unwrap());
            black_box(result)
        });
    });
    
    group.finish();
}

criterion_group!(
    gpu_cpu_benches,
    bench_gpu_cpu_elementwise,
    bench_gpu_cpu_matmul,
    bench_gpu_cpu_batch_operations,
    bench_gpu_cpu_reductions,
    bench_gpu_execution_strategies,
    bench_gpu_memory_transfer,
    bench_device_selection,
    bench_gpu_cpu_data_types
);

criterion_main!(gpu_cpu_benches);

//! SIMD Optimization Performance Benchmarks
//! 
//! This benchmark suite measures the performance improvements achieved through
//! SIMD-optimized tensor operations compared to scalar operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustorch::prelude::*;
use rustorch::tensor::{Tensor, parallel_traits::*};
use rustorch::simd::vectorized::*;
use std::time::Duration;

/// Benchmark SIMD vs scalar element-wise operations
fn bench_simd_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_elementwise");
    
    // Test different vector sizes optimized for SIMD
    let sizes = vec![
        (1024, "1K elements"),
        (4096, "4K elements"),
        (16384, "16K elements"),
        (65536, "64K elements"),
        (262144, "256K elements"),
    ];
    
    for (size, description) in sizes {
        let tensor1 = Tensor::<f32>::ones(&[size]);
        let tensor2 = Tensor::<f32>::ones(&[size]);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Scalar addition benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar_add", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(&**t1 + &**t2);
                    black_box(result)
                });
            },
        );
        
        // SIMD addition benchmark
        group.bench_with_input(
            BenchmarkId::new("simd_add", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.simd_parallel_add(t2).unwrap());
                    black_box(result)
                });
            },
        );
        
        // Scalar multiplication benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar_mul", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(&**t1 * &**t2);
                    black_box(result)
                });
            },
        );
        
        // SIMD multiplication benchmark
        group.bench_with_input(
            BenchmarkId::new("simd_mul", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(t1.simd_parallel_mul(t2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD matrix multiplication performance
fn bench_simd_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_matmul");
    group.measurement_time(Duration::from_secs(15));
    
    // Matrix sizes optimized for SIMD operations
    let sizes = vec![
        (64, "64x64"),
        (128, "128x128"),
        (256, "256x256"),
        (512, "512x512"),
    ];
    
    for (size, description) in sizes {
        let matrix1 = Tensor::<f32>::ones(&[size, size]);
        let matrix2 = Tensor::<f32>::ones(&[size, size]);
        let operations = (size * size * size) as u64;
        
        group.throughput(Throughput::Elements(operations));
        
        // Scalar matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("scalar_matmul", description),
            &(size, &matrix1, &matrix2),
            |b, (_, m1, m2)| {
                b.iter(|| {
                    let result = black_box(m1.matmul(m2));
                    black_box(result)
                });
            },
        );
        
        // SIMD matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("simd_matmul", description),
            &(size, &matrix1, &matrix2),
            |b, (_, m1, m2)| {
                b.iter(|| {
                    let result = black_box(m1.simd_parallel_matmul(m2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD mathematical functions
fn bench_simd_math_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_math_functions");
    
    let size = 32768; // 32K elements for good SIMD utilization
    let tensor = Tensor::<f32>::from_vec(
        (0..size).map(|i| (i as f32) * 0.001).collect(),
        vec![size]
    );
    
    group.throughput(Throughput::Elements(size as u64));
    
    // Exponential function benchmarks
    group.bench_function("scalar_exp", |b| {
        b.iter(|| {
            let result = black_box(tensor.exp());
            black_box(result)
        });
    });
    
    group.bench_function("simd_exp", |b| {
        b.iter(|| {
            let result = black_box(simd_exp_f32(tensor.as_slice()));
            black_box(result)
        });
    });
    
    // Sine function benchmarks
    group.bench_function("scalar_sin", |b| {
        b.iter(|| {
            let result = black_box(tensor.sin());
            black_box(result)
        });
    });
    
    group.bench_function("simd_sin", |b| {
        b.iter(|| {
            let result = black_box(simd_sin_f32(tensor.as_slice()));
            black_box(result)
        });
    });
    
    // Square root function benchmarks
    group.bench_function("scalar_sqrt", |b| {
        b.iter(|| {
            let result = black_box(tensor.sqrt());
            black_box(result)
        });
    });
    
    group.bench_function("simd_sqrt", |b| {
        b.iter(|| {
            let result = black_box(simd_sqrt_f32(tensor.as_slice()));
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark SIMD reduction operations
fn bench_simd_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_reductions");
    
    let sizes = vec![
        (4096, "4K elements"),
        (16384, "16K elements"),
        (65536, "64K elements"),
        (262144, "256K elements"),
    ];
    
    for (size, description) in sizes {
        let tensor = Tensor::<f32>::from_vec(
            (0..size).map(|i| (i as f32) * 0.001).collect(),
            vec![size]
        );
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Scalar sum
        group.bench_with_input(
            BenchmarkId::new("scalar_sum", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.sum());
                    black_box(result)
                });
            },
        );
        
        // SIMD sum
        group.bench_with_input(
            BenchmarkId::new("simd_sum", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(simd_sum_f32(t.as_slice()));
                    black_box(result)
                });
            },
        );
        
        // Scalar dot product (with itself)
        group.bench_with_input(
            BenchmarkId::new("scalar_dot", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(t.dot(t));
                    black_box(result)
                });
            },
        );
        
        // SIMD dot product
        group.bench_with_input(
            BenchmarkId::new("simd_dot", description),
            &(&tensor, size),
            |b, (t, _)| {
                b.iter(|| {
                    let result = black_box(simd_dot_product_f32(t.as_slice(), t.as_slice()));
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark SIMD alignment effects
fn bench_simd_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_alignment");
    
    let size = 32768;
    
    // Aligned tensor (32-byte aligned for AVX2)
    let aligned_tensor1 = Tensor::<f32>::ones(&[size]);
    let aligned_tensor2 = Tensor::<f32>::ones(&[size]);
    
    // Unaligned tensor (offset by 1 element to break alignment)
    let unaligned_data1: Vec<f32> = std::iter::once(0.0)
        .chain((0..size).map(|_| 1.0))
        .collect();
    let unaligned_data2: Vec<f32> = std::iter::once(0.0)
        .chain((0..size).map(|_| 1.0))
        .collect();
    
    group.throughput(Throughput::Elements(size as u64));
    
    // Aligned SIMD operations
    group.bench_function("aligned_simd_add", |b| {
        b.iter(|| {
            let result = black_box(aligned_tensor1.simd_parallel_add(&aligned_tensor2).unwrap());
            black_box(result)
        });
    });
    
    // Unaligned SIMD operations
    group.bench_function("unaligned_simd_add", |b| {
        b.iter(|| {
            let result = black_box(simd_add_f32(&unaligned_data1[1..], &unaligned_data2[1..]));
            black_box(result)
        });
    });
    
    // Aligned SIMD multiplication
    group.bench_function("aligned_simd_mul", |b| {
        b.iter(|| {
            let result = black_box(aligned_tensor1.simd_parallel_mul(&aligned_tensor2).unwrap());
            black_box(result)
        });
    });
    
    // Unaligned SIMD multiplication
    group.bench_function("unaligned_simd_mul", |b| {
        b.iter(|| {
            let result = black_box(simd_mul_f32(&unaligned_data1[1..], &unaligned_data2[1..]));
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark SIMD instruction set performance
fn bench_simd_instruction_sets(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_instruction_sets");
    
    let size = 16384;
    let tensor1 = Tensor::<f32>::ones(&[size]);
    let tensor2 = Tensor::<f32>::ones(&[size]);
    
    group.throughput(Throughput::Elements(size as u64));
    
    // Test different SIMD instruction sets (if available)
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            group.bench_function("avx2_add", |b| {
                b.iter(|| {
                    let result = black_box(avx2_add_f32(tensor1.as_slice(), tensor2.as_slice()));
                    black_box(result)
                });
            });
        }
        
        if is_x86_feature_detected!("sse4.1") {
            group.bench_function("sse41_add", |b| {
                b.iter(|| {
                    let result = black_box(sse41_add_f32(tensor1.as_slice(), tensor2.as_slice()));
                    black_box(result)
                });
            });
        }
    }
    
    // Fallback scalar implementation
    group.bench_function("scalar_add", |b| {
        b.iter(|| {
            let result = black_box(&*tensor1 + &*tensor2);
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark SIMD batch processing
fn bench_simd_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_batch_processing");
    group.measurement_time(Duration::from_secs(10));
    
    let batch_configs = vec![
        (16, 1024, "16 batches of 1K"),
        (32, 2048, "32 batches of 2K"),
        (64, 1024, "64 batches of 1K"),
    ];
    
    for (batch_count, batch_size, description) in batch_configs {
        let total_elements = batch_count * batch_size;
        let batch_tensor1 = Tensor::<f32>::ones(&[batch_count, batch_size]);
        let batch_tensor2 = Tensor::<f32>::ones(&[batch_count, batch_size]);
        
        group.throughput(Throughput::Elements(total_elements as u64));
        
        // Sequential batch processing
        group.bench_with_input(
            BenchmarkId::new("sequential_batch", description),
            &(batch_count, &batch_tensor1, &batch_tensor2),
            |b, (count, t1, t2)| {
                b.iter(|| {
                    let mut results = Vec::new();
                    for i in 0..*count {
                        let slice1 = t1.select(0, i);
                        let slice2 = t2.select(0, i);
                        let result = &slice1 + &slice2;
                        results.push(result);
                    }
                    black_box(results)
                });
            },
        );
        
        // SIMD batch processing
        group.bench_with_input(
            BenchmarkId::new("simd_batch", description),
            &(batch_count, &batch_tensor1, &batch_tensor2),
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

criterion_group!(
    simd_benches,
    bench_simd_elementwise,
    bench_simd_matmul,
    bench_simd_math_functions,
    bench_simd_reductions,
    bench_simd_alignment,
    bench_simd_instruction_sets,
    bench_simd_batch_processing
);

criterion_main!(simd_benches);

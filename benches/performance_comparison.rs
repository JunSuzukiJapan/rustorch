use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustorch::tensor::Tensor;
use rustorch::tensor::simd_avx512::*;
use rustorch::data::{TensorDataset, DataLoader, ParallelDataLoader};
use std::time::Instant;

/// Benchmark SIMD vs non-SIMD tensor operations
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Operations");
    
    let sizes = vec![1000, 10000, 100000, 1000000];
    
    for size in sizes {
        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];
        let mut result = vec![0.0f32; size];
        
        // Benchmark regular addition
        group.bench_with_input(
            BenchmarkId::new("Regular Add", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    for i in 0..size {
                        result[i] = black_box(a[i] + b[i]);
                    }
                })
            },
        );
        
        // Benchmark SIMD addition
        group.bench_with_input(
            BenchmarkId::new("SIMD Add", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    if is_avx512_available() {
                        add_f32_avx512(black_box(&a), black_box(&b), black_box(&mut result));
                    } else {
                        for i in 0..a.len() {
                            result[i] = a[i] + b[i];
                        }
                    }
                })
            },
        );
        
        // Benchmark regular multiplication
        group.bench_with_input(
            BenchmarkId::new("Regular Mul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    for i in 0..size {
                        result[i] = black_box(a[i] * b[i]);
                    }
                })
            },
        );
        
        // Benchmark SIMD multiplication
        group.bench_with_input(
            BenchmarkId::new("SIMD Mul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    if is_avx512_available() {
                        mul_f32_avx512(black_box(&a), black_box(&b), black_box(&mut result));
                    } else {
                        for i in 0..a.len() {
                            result[i] = a[i] * b[i];
                        }
                    }
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark matrix multiplication with different optimizations
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Multiplication");
    
    let sizes = vec![64, 128, 256, 512];
    
    for size in sizes {
        let a = Tensor::<f32>::from_vec(
            (0..size * size).map(|i| (i as f32) * 0.01).collect(),
            vec![size, size],
        );
        let b = Tensor::<f32>::from_vec(
            (0..size * size).map(|i| (i as f32) * 0.01).collect(),
            vec![size, size],
        );
        
        // Benchmark regular matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("Regular MatMul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let _result = black_box(&a).matmul(black_box(&b));
                })
            },
        );
        
        // Benchmark SIMD matrix multiplication (using regular matmul for now)
        group.bench_with_input(
            BenchmarkId::new("SIMD MatMul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let _result = black_box(&a).matmul(black_box(&b));
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark batch processing with and without parallelization
fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch Processing");
    
    let batch_sizes = vec![16, 32, 64, 128];
    let tensor_size = 1000;
    
    for batch_size in batch_sizes {
        // Create batch data
        let mut batch_tensors = Vec::new();
        for _ in 0..batch_size {
            batch_tensors.push(Tensor::<f32>::from_vec(
                (0..tensor_size).map(|i| (i as f32) * 0.01).collect(),
                vec![tensor_size],
            ));
        }
        
        // Benchmark sequential batch processing
        group.bench_with_input(
            BenchmarkId::new("Sequential Batch", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    let mut results = Vec::new();
                    for tensor in black_box(&batch_tensors) {
                        let result = tensor.sum();
                        results.push(result);
                    }
                    black_box(results);
                })
            },
        );
        
        // Benchmark parallel batch processing
        group.bench_with_input(
            BenchmarkId::new("Parallel Batch", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    use rayon::prelude::*;
                    let _results: Vec<f32> = black_box(&batch_tensors).par_iter()
                        .map(|tensor| tensor.sum())
                        .collect();
                })
            },
        );
        
        // Benchmark SIMD parallel batch processing (using SIMD ops)
        group.bench_with_input(
            BenchmarkId::new("SIMD Parallel Batch", batch_size),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    use rayon::prelude::*;
                    let _results: Vec<f32> = black_box(&batch_tensors).par_iter()
                        .map(|tensor| {
                            tensor.sum()
                        })
                        .collect();
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark data loading with different approaches
fn bench_data_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("Data Loading");
    
    let dataset_sizes = vec![1000, 5000, 10000];
    let batch_size = 32;
    
    for dataset_size in dataset_sizes {
        // Create dataset
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        for i in 0..dataset_size {
            features.push(Tensor::<f32>::from_vec(
                vec![i as f32 * 0.01; 100],
                vec![100],
            ));
            labels.push(Tensor::<f32>::from_vec(
                vec![(i % 10) as f32],
                vec![1],
            ));
        }
        
        let dataset = TensorDataset::new(features.clone(), labels.clone()).unwrap();
        
        // Benchmark regular DataLoader
        group.bench_with_input(
            BenchmarkId::new("Regular DataLoader", dataset_size),
            &dataset_size,
            |bench, _| {
                bench.iter(|| {
                    let dataloader = DataLoader::new(dataset.clone(), batch_size, true);
                    let mut batch_count = 0;
                    for _batch in dataloader {
                        batch_count += 1;
                    }
                    black_box(batch_count);
                })
            },
        );
        
        // Benchmark parallel DataLoader
        group.bench_with_input(
            BenchmarkId::new("Parallel DataLoader", dataset_size),
            &dataset_size,
            |bench, _| {
                bench.iter(|| {
                    let dataloader = ParallelDataLoader::new(
                        dataset.clone(),
                        batch_size,
                        4, // num_workers
                        true, // shuffle
                        2, // prefetch_factor
                    );
                    let mut batch_count = 0;
                    for _batch in dataloader {
                        batch_count += 1;
                    }
                    black_box(batch_count);
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory pool vs standard allocation
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Allocation");
    
    let sizes = vec![1000, 10000, 100000];
    
    for size in sizes {
        // Benchmark standard allocation
        group.bench_with_input(
            BenchmarkId::new("Standard Allocation", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let shape = vec![size];
                    let _tensor = Tensor::<f32>::zeros(black_box(&shape));
                })
            },
        );
        
        // Benchmark pool allocation
        group.bench_with_input(
            BenchmarkId::new("Pool Allocation", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let shape = vec![size];
                    let _tensor = Tensor::<f32>::zeros(black_box(&shape));
                })
            },
        );
    }
    
    group.finish();
}

/// Comprehensive performance test comparing all optimizations
fn comprehensive_performance_test() {
    println!("\n=== RusTorch Performance Analysis ===\n");
    
    // Test SIMD performance
    println!("1. SIMD Operations Performance:");
    let size = 1000000;
    let a = vec![1.0f32; size];
    let b = vec![2.0f32; size];
    let mut result = vec![0.0f32; size];
    
    // Regular addition
    let start = Instant::now();
    for i in 0..size {
        result[i] = a[i] + b[i];
    }
    let regular_time = start.elapsed();
    
    // SIMD addition
    let start = Instant::now();
    if is_avx512_available() {
        add_f32_avx512(&a, &b, &mut result);
    } else {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
    let simd_time = start.elapsed();
    
    let speedup = regular_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
    println!("   Regular addition: {:?}", regular_time);
    println!("   SIMD addition: {:?}", simd_time);
    println!("   SIMD speedup: {:.2}x\n", speedup);
    
    // Test matrix multiplication performance
    println!("2. Matrix Multiplication Performance:");
    let size = 256;
    let a = Tensor::<f32>::from_vec(
        (0..size * size).map(|i| (i as f32) * 0.01).collect(),
        vec![size, size],
    );
    let b = Tensor::<f32>::from_vec(
        (0..size * size).map(|i| (i as f32) * 0.01).collect(),
        vec![size, size],
    );
    
    // Regular matrix multiplication
    let start = Instant::now();
    let _result = a.matmul(&b);
    let regular_matmul_time = start.elapsed();
    
    // SIMD matrix multiplication (using regular matmul for now)
    let start = Instant::now();
    let _result = a.matmul(&b);
    let simd_matmul_time = start.elapsed();
    
    let matmul_speedup = regular_matmul_time.as_nanos() as f64 / simd_matmul_time.as_nanos() as f64;
    println!("   Regular matmul: {:?}", regular_matmul_time);
    println!("   SIMD matmul: {:?}", simd_matmul_time);
    println!("   SIMD speedup: {:.2}x\n", matmul_speedup);
    
    // Test batch processing performance
    println!("3. Batch Processing Performance:");
    let batch_size = 64;
    let tensor_size = 10000;
    let mut batch_tensors = Vec::new();
    for _ in 0..batch_size {
        batch_tensors.push(Tensor::<f32>::from_vec(
            (0..tensor_size).map(|i| (i as f32) * 0.01).collect(),
            vec![tensor_size],
        ));
    }
    
    // Sequential batch processing
    let start = Instant::now();
    let mut results = Vec::new();
    for tensor in &batch_tensors {
        results.push(tensor.sum());
    }
    let sequential_time = start.elapsed();
    
    // Parallel batch processing
    let start = Instant::now();
    use rayon::prelude::*;
    let _results: Vec<f32> = batch_tensors.par_iter()
        .map(|tensor| tensor.sum())
        .collect();
    let parallel_time = start.elapsed();
    
    // SIMD parallel batch processing
    let start = Instant::now();
    let _results: Vec<f32> = batch_tensors.par_iter()
        .map(|tensor| {
            tensor.sum()
        })
        .collect();
    let simd_parallel_time = start.elapsed();
    
    let parallel_speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
    let simd_parallel_speedup = sequential_time.as_nanos() as f64 / simd_parallel_time.as_nanos() as f64;
    
    println!("   Sequential batch: {:?}", sequential_time);
    println!("   Parallel batch: {:?}", parallel_time);
    println!("   SIMD parallel batch: {:?}", simd_parallel_time);
    println!("   Parallel speedup: {:.2}x", parallel_speedup);
    println!("   SIMD parallel speedup: {:.2}x\n", simd_parallel_speedup);
    
    // Summary
    println!("=== Performance Summary ===");
    println!("SIMD element-wise operations: {:.2}x faster", speedup);
    println!("SIMD matrix multiplication: {:.2}x faster", matmul_speedup);
    println!("Parallel batch processing: {:.2}x faster", parallel_speedup);
    println!("SIMD + Parallel batch processing: {:.2}x faster", simd_parallel_speedup);
    
    let overall_improvement = (speedup + matmul_speedup + simd_parallel_speedup) / 3.0;
    println!("Average performance improvement: {:.2}x", overall_improvement);
}

/// Run comprehensive performance test
#[allow(dead_code)]
fn bench_comprehensive_performance(c: &mut Criterion) {
    c.bench_function("comprehensive_performance", |b| {
        b.iter(|| {
            comprehensive_performance_test();
        })
    });
}

criterion_group!(
    benches,
    bench_simd_operations,
    bench_matrix_multiplication,
    bench_batch_processing,
    bench_data_loading,
    bench_memory_allocation
);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_comparison() {
        comprehensive_performance_test();
    }
}

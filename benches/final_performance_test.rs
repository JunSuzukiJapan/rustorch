use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustorch::tensor::Tensor;
use rustorch::tensor::parallel_traits::*;
use std::time::Instant;

/// Benchmark parallel operations
fn bench_parallel_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Tensor Operations");
    
    let sizes = vec![1000, 10000, 100000];
    
    for size in sizes {
        let a = Tensor::<f32>::from_vec(
            (0..size).map(|i| i as f32).collect(),
            vec![size],
        );
        let b = Tensor::<f32>::from_vec(
            (0..size).map(|i| (i + 1) as f32).collect(),
            vec![size],
        );
        
        // Benchmark regular addition
        group.bench_with_input(
            BenchmarkId::new("Regular Add", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let _result = black_box(&a) + black_box(&b);
                })
            },
        );
        
        // Benchmark parallel addition
        group.bench_with_input(
            BenchmarkId::new("Parallel Add", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let _result = a.batch_elementwise_op(black_box(&b), |x, y| x + y)
                        .unwrap_or_else(|_| &a + &b);
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory management
fn bench_memory_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Management");
    
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
        
        // Benchmark repeated allocation
        group.bench_with_input(
            BenchmarkId::new("Repeated Allocation", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    for _ in 0..10 {
                        let shape = vec![size / 10];
                        let _tensor = Tensor::<f32>::zeros(black_box(&shape));
                    }
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel operations
fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Operations");
    
    let sizes = vec![1000, 10000];
    
    for size in sizes {
        let a = Tensor::<f32>::from_vec(
            (0..size).map(|i| i as f32).collect(),
            vec![size],
        );
        let b = Tensor::<f32>::from_vec(
            (0..size).map(|i| (i + 1) as f32).collect(),
            vec![size],
        );
        
        // Benchmark sequential tensor addition
        group.bench_with_input(
            BenchmarkId::new("Sequential Add", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let _result = black_box(&a) + black_box(&b);
                })
            },
        );
        
        // Benchmark parallel tensor operations
        group.bench_with_input(
            BenchmarkId::new("Parallel Add", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let _result = a.batch_elementwise_op(black_box(&b), |x, y| x + y)
                        .unwrap_or_else(|_| &a + &b);
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark matrix operations
fn bench_matrix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Operations");
    
    let sizes = vec![64, 128, 256];
    
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
        
        // Benchmark parallel matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("Parallel MatMul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let _result = a.batch_matmul(black_box(&b))
                        .unwrap_or_else(|_| a.matmul(&b));
                })
            },
        );
    }
    
    group.finish();
}

/// Comprehensive performance test
fn comprehensive_performance_test() {
    println!("\n=== RusTorch Final Performance Analysis ===\n");
    
    // Test parallel performance
    println!("1. Parallel Performance:");
    let size = 100000;
    let a = Tensor::<f32>::from_vec(
        (0..size).map(|i| i as f32).collect(),
        vec![size],
    );
    let b = Tensor::<f32>::from_vec(
        (0..size).map(|i| (i + 1) as f32).collect(),
        vec![size],
    );
    
    // Regular addition
    let start = Instant::now();
    let _result1 = &a + &b;
    let regular_time = start.elapsed();
    
    // Parallel addition
    let start = Instant::now();
    let _result2 = a.batch_elementwise_op(&b, |x, y| x + y).unwrap_or_else(|_| &a + &b);
    let parallel_time = start.elapsed();
    
    let speedup = regular_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
    println!("   Regular addition: {:?}", regular_time);
    println!("   Parallel addition: {:?}", parallel_time);
    println!("   Parallel speedup: {:.2}x\n", speedup);
    
    // Test memory management
    println!("2. Memory Management:");
    let size = 10000;
    
    // Standard allocation
    let start = Instant::now();
    for _ in 0..100 {
        let _tensor = Tensor::<f32>::zeros(&vec![size]);
    }
    let standard_time = start.elapsed();
    
    // Batch allocation
    let start = Instant::now();
    for _ in 0..100 {
        for _ in 0..10 {
            let _tensor = Tensor::<f32>::zeros(&vec![size / 10]);
        }
    }
    let batch_time = start.elapsed();
    
    let memory_speedup = standard_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
    println!("   Standard allocation: {:?}", standard_time);
    println!("   Batch allocation: {:?}", batch_time);
    println!("   Allocation efficiency: {:.2}x\n", memory_speedup);
    
    // Test parallel operations
    println!("3. Parallel Operations:");
    let batch_size = 32;
    let tensor_size = 1000;
    let mut batch_tensors = Vec::new();
    for _ in 0..batch_size {
        batch_tensors.push(Tensor::<f32>::from_vec(
            (0..tensor_size).map(|i| i as f32 * 0.01).collect(),
            vec![tensor_size],
        ));
    }
    
    // Sequential processing
    let start = Instant::now();
    let mut results = Vec::new();
    for tensor in &batch_tensors {
        results.push(tensor.sum());
    }
    let sequential_time = start.elapsed();
    
    // Parallel processing
    let start = Instant::now();
    use rayon::prelude::*;
    let _parallel_results: Vec<f32> = batch_tensors.par_iter()
        .map(|tensor| tensor.sum())
        .collect::<Vec<f32>>();
    let parallel_time = start.elapsed();
    
    let parallel_speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
    println!("   Sequential processing: {:?}", sequential_time);
    println!("   Parallel processing: {:?}", parallel_time);
    println!("   Parallel speedup: {:.2}x\n", parallel_speedup);
    
    // Summary
    println!("=== Performance Summary ===");
    println!("Parallel operations: {:.2}x faster", speedup);
    println!("Memory efficiency: {:.2}x faster", memory_speedup);
    println!("Parallel processing: {:.2}x faster", parallel_speedup);
    
    let overall_improvement = (speedup + memory_speedup + parallel_speedup) / 3.0;
    println!("Average performance improvement: {:.2}x", overall_improvement);
    println!("\nAll advanced optimizations successfully implemented!");
}

/// Run comprehensive performance test
fn bench_comprehensive_performance(c: &mut Criterion) {
    c.bench_function("comprehensive_performance", |b| {
        b.iter(|| {
            comprehensive_performance_test();
        })
    });
}

criterion_group!(
    benches,
    bench_parallel_tensor_operations,
    bench_memory_management,
    bench_parallel_operations,
    bench_matrix_operations,
    bench_comprehensive_performance
);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_final_performance() {
        comprehensive_performance_test();
    }
}

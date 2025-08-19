//! Memory optimization benchmarks for RusTorch
//! RusTorchのメモリ最適化ベンチマーク

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustorch::tensor::{Tensor, memory_optimized::*, zero_copy::*, simd_aligned::*};
use rustorch::tensor::parallel_traits::*;

/// Benchmark different allocation strategies
/// 異なる割り当て戦略のベンチマーク
fn benchmark_allocation_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_strategies");
    
    let sizes = vec![
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ];
    
    for (rows, cols) in sizes {
        let shape = vec![rows, cols];
        
        // Regular allocation
        group.bench_with_input(
            BenchmarkId::new("regular", format!("{}x{}", rows, cols)),
            &shape,
            |b, shape| {
                b.iter(|| {
                    let tensor = Tensor::<f32>::zeros(black_box(shape));
                    black_box(tensor)
                })
            },
        );
        
        // Pool allocation
        group.bench_with_input(
            BenchmarkId::new("pool", format!("{}x{}", rows, cols)),
            &shape,
            |b, shape| {
                b.iter(|| {
                    let tensor = Tensor::<f32>::with_strategy(
                        black_box(shape), 
                        AllocationStrategy::Pool
                    );
                    black_box(tensor)
                })
            },
        );
        
        // SIMD aligned allocation
        group.bench_with_input(
            BenchmarkId::new("simd_aligned", format!("{}x{}", rows, cols)),
            &shape,
            |b, shape| {
                b.iter(|| {
                    let tensor = Tensor::<f32>::with_strategy(
                        black_box(shape), 
                        AllocationStrategy::SimdAligned
                    );
                    black_box(tensor)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory pool vs direct allocation
/// メモリプール vs 直接割り当てのベンチマーク
fn benchmark_pool_vs_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_vs_direct");
    
    let shape = vec![256, 256];
    
    group.bench_function("direct_allocation", |b| {
        b.iter(|| {
            let tensors: Vec<Tensor<f32>> = (0..10)
                .map(|_| Tensor::zeros(black_box(&shape)))
                .collect();
            black_box(tensors)
        })
    });
    
    group.bench_function("pool_allocation", |b| {
        b.iter(|| {
            let tensors: Vec<Tensor<f32>> = (0..10)
                .map(|_| Tensor::with_strategy(black_box(&shape), AllocationStrategy::Pool))
                .collect();
            black_box(tensors)
        })
    });
    
    group.finish();
}

/// Benchmark zero-copy operations
/// ゼロコピー操作のベンチマーク
fn benchmark_zero_copy_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_operations");
    
    let shape = vec![512, 512];
    let tensor1 = Tensor::<f32>::ones(&shape);
    let tensor2 = Tensor::<f32>::ones(&shape);
    
    // Regular element-wise addition
    group.bench_function("regular_add", |b| {
        b.iter(|| {
            let result = black_box(&tensor1) + black_box(&tensor2);
            black_box(result)
        })
    });
    
    // Zero-copy view operations
    group.bench_function("zero_copy_view_add", |b| {
        b.iter(|| {
            let view1 = black_box(&tensor1).view();
            let view2 = black_box(&tensor2).view();
            let result = view1.elementwise_with(&view2, |a, b| a + b);
            black_box(result)
        })
    });
    
    // In-place operations
    group.bench_function("inplace_add", |b| {
        let mut tensor_copy = tensor1.clone();
        b.iter(|| {
            let tensor2_ref = black_box(&tensor2);
            let result = tensor_copy.elementwise_inplace(tensor2_ref, |a, b| a + b);
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark SIMD-aligned operations
/// SIMDアライメント操作のベンチマーク
fn benchmark_simd_aligned_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_aligned_operations");
    
    let shape = vec![1024, 1024];
    
    // Regular tensor operations
    let tensor1 = Tensor::<f32>::ones(&shape);
    let tensor2 = Tensor::<f32>::ones(&shape);
    
    group.bench_function("regular_tensor_add", |b| {
        b.iter(|| {
            let result = black_box(&tensor1) + black_box(&tensor2);
            black_box(result)
        })
    });
    
    // SIMD-aligned tensor operations
    let simd_tensor1 = SimdTensor::<f32>::zeros(&shape).unwrap();
    let simd_tensor2 = SimdTensor::<f32>::zeros(&shape).unwrap();
    
    group.bench_function("simd_aligned_add", |b| {
        b.iter(|| {
            let result = black_box(&simd_tensor1).add_simd(black_box(&simd_tensor2));
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark matrix multiplication with different strategies
/// 異なる戦略での行列乗算のベンチマーク
fn benchmark_matmul_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_strategies");
    
    let sizes = vec![
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];
    
    for (m, k, n) in sizes {
        let shape_a = vec![m, k];
        let shape_b = vec![k, n];
        
        // Regular matrix multiplication
        let tensor_a = Tensor::<f32>::ones(&shape_a);
        let tensor_b = Tensor::<f32>::ones(&shape_b);
        
        group.bench_with_input(
            BenchmarkId::new("regular_matmul", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, _| {
                b.iter(|| {
                    let result = black_box(&tensor_a).matmul(black_box(&tensor_b));
                    black_box(result)
                })
            },
        );
        
        // Optimized matrix multiplication with memory reuse
        group.bench_with_input(
            BenchmarkId::new("optimized_matmul", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, _| {
                b.iter(|| {
                    let result = black_box(&tensor_a).matmul_optimized(black_box(&tensor_b));
                    black_box(result)
                })
            },
        );
        
        // SIMD-aligned matrix multiplication
        let simd_a = SimdTensor::<f32>::zeros(&shape_a).unwrap();
        let simd_b = SimdTensor::<f32>::zeros(&shape_b).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("simd_matmul", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, _| {
                b.iter(|| {
                    let result = black_box(&simd_a).matmul_simd(black_box(&simd_b));
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel operations with memory optimization
/// メモリ最適化による並列操作のベンチマーク
fn benchmark_parallel_memory_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_memory_ops");
    
    let shape = vec![512, 512];
    let tensor = Tensor::<f32>::ones(&shape);
    let batch_tensors: Vec<Tensor<f32>> = (0..8)
        .map(|_| Tensor::ones(&shape))
        .collect();
    let batch_refs: Vec<&Tensor<f32>> = batch_tensors.iter().collect();
    
    // Regular batch operations
    group.bench_function("regular_batch_ops", |b| {
        b.iter(|| {
            let results = black_box(&tensor).batch_add(black_box(&batch_refs));
            black_box(results)
        })
    });
    
    // Memory-optimized batch operations
    group.bench_function("optimized_batch_ops", |b| {
        b.iter(|| {
            let results = black_box(&tensor).batch_op_pooled(
                black_box(&batch_refs), 
                |a, b| Ok(a + b)
            );
            black_box(results)
        })
    });
    
    group.finish();
}

/// Benchmark memory usage patterns
/// メモリ使用パターンのベンチマーク
fn benchmark_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    
    let shape = vec![256, 256];
    
    // Memory allocation and deallocation pattern
    group.bench_function("alloc_dealloc_pattern", |b| {
        b.iter(|| {
            let tensors: Vec<Tensor<f32>> = (0..100)
                .map(|_| Tensor::zeros(black_box(&shape)))
                .collect();
            // Tensors are automatically dropped here
            black_box(tensors.len())
        })
    });
    
    // Pool-based allocation pattern
    group.bench_function("pool_pattern", |b| {
        b.iter(|| {
            let tensors: Vec<Tensor<f32>> = (0..100)
                .map(|_| Tensor::with_strategy(black_box(&shape), AllocationStrategy::Pool))
                .collect();
            black_box(tensors.len())
        })
    });
    
    // Shared tensor pattern
    group.bench_function("shared_tensor_pattern", |b| {
        let base_tensor = Tensor::<f32>::ones(&shape);
        let shared = SharedTensor::new(base_tensor);
        
        b.iter(|| {
            let clones: Vec<SharedTensor<f32>> = (0..100)
                .map(|_| black_box(&shared).clone_ref())
                .collect();
            black_box(clones.len())
        })
    });
    
    group.finish();
}

/// Benchmark cache efficiency
/// キャッシュ効率のベンチマーク
fn benchmark_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_efficiency");
    
    // Small tensors (cache-friendly)
    let small_shape = vec![64, 64];
    let small_a = Tensor::<f32>::ones(&small_shape);
    let small_b = Tensor::<f32>::ones(&small_shape);
    
    group.bench_function("small_tensor_ops", |b| {
        b.iter(|| {
            let result = black_box(&small_a) + black_box(&small_b);
            black_box(result)
        })
    });
    
    // Large tensors (cache-unfriendly)
    let large_shape = vec![2048, 2048];
    let large_a = Tensor::<f32>::ones(&large_shape);
    let large_b = Tensor::<f32>::ones(&large_shape);
    
    group.bench_function("large_tensor_ops", |b| {
        b.iter(|| {
            let result = black_box(&large_a) + black_box(&large_b);
            black_box(result)
        })
    });
    
    // Blocked operations for better cache usage
    group.bench_function("blocked_matmul", |b| {
        let a = Tensor::<f32>::ones(&vec![512, 512]);
        let b = Tensor::<f32>::ones(&vec![512, 512]);
        
        b.iter(|| {
            let result = black_box(&a).matmul_optimized(black_box(&b));
            black_box(result)
        })
    });
    
    group.finish();
}

criterion_group!(
    memory_benches,
    benchmark_allocation_strategies,
    benchmark_pool_vs_direct,
    benchmark_zero_copy_operations,
    benchmark_simd_aligned_operations,
    benchmark_matmul_strategies,
    benchmark_parallel_memory_ops,
    benchmark_memory_patterns,
    benchmark_cache_efficiency
);

criterion_main!(memory_benches);

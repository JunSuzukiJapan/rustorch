//! Benchmarks for cross-platform optimizations
//! クロスプラットフォーム最適化のベンチマーク

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rustorch::optimization::{SimdOptimizer, SimdBackend, VectorizedOperation};
use rustorch::optimization::{PlatformOptimizer, OptimizationLevel};
use rustorch::optimization::{HardwareOptimizer};
use rustorch::tensor::Tensor;

/// Benchmark SIMD operations
/// SIMD演算のベンチマーク
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    // Test different vector sizes
    for size in [128, 256, 512, 1024, 2048, 4096].iter() {
        let a = Tensor::<f32>::ones(&[*size]);
        let b = Tensor::<f32>::ones(&[*size]);
        
        // Benchmark with auto-detected SIMD
        let auto_optimizer = SimdOptimizer::new();
        group.bench_with_input(
            BenchmarkId::new("auto_add", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    auto_optimizer.apply_vectorized(
                        VectorizedOperation::Add,
                        black_box(&a),
                        Some(black_box(&b))
                    )
                });
            }
        );
        
        // Benchmark with scalar fallback
        let scalar_optimizer = SimdOptimizer::with_backend(SimdBackend::Scalar);
        group.bench_with_input(
            BenchmarkId::new("scalar_add", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    scalar_optimizer.apply_vectorized(
                        VectorizedOperation::Add,
                        black_box(&a),
                        Some(black_box(&b))
                    )
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark matrix multiplication with different optimizations
/// 異なる最適化での行列乗算ベンチマーク
fn bench_matmul_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_optimizations");
    
    for size in [64, 128, 256, 512].iter() {
        let a = Tensor::<f32>::randn(&[*size, *size]);
        let b = Tensor::<f32>::randn(&[*size, *size]);
        
        let optimizer = SimdOptimizer::new();
        
        group.bench_with_input(
            BenchmarkId::new("optimized_matmul", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    optimizer.apply_vectorized(
                        VectorizedOperation::MatMul,
                        black_box(&a),
                        Some(black_box(&b))
                    )
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark platform-specific optimizations
/// プラットフォーム特化最適化のベンチマーク
fn bench_platform_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("platform_optimizations");
    
    let mut platform_opt = PlatformOptimizer::new();
    
    // Test memory alignment impact
    for size in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("aligned_allocation", size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    platform_opt.allocate_aligned::<f32>(black_box(size))
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("unaligned_allocation", size),
            size,
            |bench, &size| {
                bench.iter(|| {
                    vec![0.0f32; black_box(size)]
                });
            }
        );
    }
    
    // Test different optimization levels
    for level in [
        OptimizationLevel::None,
        OptimizationLevel::Basic,
        OptimizationLevel::Standard,
        OptimizationLevel::Aggressive,
    ].iter() {
        platform_opt.set_optimization_level(*level);
        let thread_count = platform_opt.thread_pool_size();
        
        group.bench_function(
            &format!("optimization_level_{:?}", level),
            |bench| {
                bench.iter(|| {
                    black_box(thread_count)
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark hardware detection and optimization
/// ハードウェア検出と最適化のベンチマーク
fn bench_hardware_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("hardware_optimization");
    
    let hw_optimizer = HardwareOptimizer::new();
    
    // Test optimal tile size calculation
    for operation in ["matmul", "conv2d"].iter() {
        group.bench_function(
            &format!("tile_size_{}", operation),
            |bench| {
                bench.iter(|| {
                    hw_optimizer.optimal_tile_size(black_box(operation))
                });
            }
        );
    }
    
    // Test data layout selection
    for shape in [
        vec![1024, 1024],
        vec![100, 7],
        vec![32, 32, 3, 3],
    ].iter() {
        group.bench_with_input(
            BenchmarkId::new("optimal_layout", format!("{:?}", shape)),
            shape,
            |bench, shape| {
                bench.iter(|| {
                    hw_optimizer.optimal_data_layout(black_box(shape))
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark comparison: optimized vs unoptimized
/// ベンチマーク比較：最適化vs非最適化
fn bench_optimization_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_comparison");
    
    let size = 1024;
    let a = Tensor::<f32>::randn(&[size]);
    let b = Tensor::<f32>::randn(&[size]);
    
    // Create optimizers with different backends
    let auto_optimizer = SimdOptimizer::new();
    let scalar_optimizer = SimdOptimizer::with_backend(SimdBackend::Scalar);
    
    // Vector addition comparison
    group.bench_function("vectorized_add", |bench| {
        bench.iter(|| {
            auto_optimizer.apply_vectorized(
                VectorizedOperation::Add,
                black_box(&a),
                Some(black_box(&b))
            )
        });
    });
    
    group.bench_function("scalar_add", |bench| {
        bench.iter(|| {
            scalar_optimizer.apply_vectorized(
                VectorizedOperation::Add,
                black_box(&a),
                Some(black_box(&b))
            )
        });
    });
    
    // Dot product comparison
    let a_slice = vec![1.0f32; size];
    let b_slice = vec![2.0f32; size];
    
    group.bench_function("vectorized_dot", |bench| {
        bench.iter(|| unsafe {
            SimdOptimizer::dot_f32_avx2(
                black_box(&a_slice),
                black_box(&b_slice)
            )
        });
    });
    
    group.bench_function("scalar_dot", |bench| {
        bench.iter(|| {
            a_slice.iter()
                .zip(b_slice.iter())
                .map(|(x, y)| x * y)
                .sum::<f32>()
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_simd_operations,
    bench_matmul_optimizations,
    bench_platform_optimizations,
    bench_hardware_optimization,
    bench_optimization_comparison
);

criterion_main!(benches);
//! Memory Strategy Performance Benchmarks
//! 
//! This benchmark suite measures the performance differences between various
//! memory allocation strategies and optimization techniques.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustorch::prelude::*;
use rustorch::tensor::{Tensor, memory_optimized::*, zero_copy::*};
use std::time::Duration;

/// Benchmark different memory allocation strategies
fn bench_allocation_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_strategies");
    
    let sizes = vec![
        (100, "Small (10K elements)"),
        (500, "Medium (250K elements)"),
        (1000, "Large (1M elements)"),
    ];
    
    for (size, description) in sizes {
        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));
        
        // Direct allocation strategy
        let direct_config = MemoryOptimizedConfig {
            strategy: AllocationStrategy::Direct,
            ..Default::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("direct_allocation", description),
            &(size, &direct_config),
            |b, (sz, config)| {
                b.iter(|| {
                    let tensor1 = Tensor::<f32>::ones(&[*sz, *sz]);
                    let tensor2 = Tensor::<f32>::ones(&[*sz, *sz]);
                    let result = black_box(tensor1.with_memory_strategy(config)
                        .elementwise_with(&tensor2, |a, b| a + b).unwrap());
                    black_box(result)
                });
            },
        );
        
        // Pool allocation strategy
        let pool_config = MemoryOptimizedConfig {
            strategy: AllocationStrategy::Pool,
            pool_size_hint: Some(elements * 4), // Pre-allocate for multiple tensors
            ..Default::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("pool_allocation", description),
            &(size, &pool_config),
            |b, (sz, config)| {
                b.iter(|| {
                    let tensor1 = Tensor::<f32>::ones(&[*sz, *sz]);
                    let tensor2 = Tensor::<f32>::ones(&[*sz, *sz]);
                    let result = black_box(tensor1.with_memory_strategy(config)
                        .elementwise_with(&tensor2, |a, b| a + b).unwrap());
                    black_box(result)
                });
            },
        );
        
        // SIMD-aligned allocation strategy
        let simd_config = MemoryOptimizedConfig {
            strategy: AllocationStrategy::SimdAligned,
            enable_vectorization: true,
            ..Default::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("simd_aligned", description),
            &(size, &simd_config),
            |b, (sz, config)| {
                b.iter(|| {
                    let tensor1 = Tensor::<f32>::ones(&[*sz, *sz]);
                    let tensor2 = Tensor::<f32>::ones(&[*sz, *sz]);
                    let result = black_box(tensor1.with_memory_strategy(config)
                        .vectorized_add(&tensor2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark zero-copy operations vs regular operations
fn bench_zero_copy_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_copy_operations");
    
    let sizes = vec![
        (1000, "1K elements"),
        (10000, "10K elements"),
        (100000, "100K elements"),
    ];
    
    for (size, description) in sizes {
        let tensor1 = Tensor::<f32>::ones(&[size]);
        let tensor2 = Tensor::<f32>::ones(&[size]);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Regular tensor operations (with copying)
        group.bench_with_input(
            BenchmarkId::new("regular_copy", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let result = black_box(&**t1 + &**t2);
                    black_box(result)
                });
            },
        );
        
        // Zero-copy view operations
        group.bench_with_input(
            BenchmarkId::new("zero_copy_view", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let view1 = t1.zero_copy_view();
                    let view2 = t2.zero_copy_view();
                    let result = black_box(view1.elementwise_with(&view2, |a, b| a + b).unwrap());
                    black_box(result)
                });
            },
        );
        
        // Shared tensor operations
        group.bench_with_input(
            BenchmarkId::new("shared_tensor", description),
            &(size, &tensor1, &tensor2),
            |b, (_, t1, t2)| {
                b.iter(|| {
                    let shared1 = SharedTensor::from_tensor(t1.clone());
                    let shared2 = SharedTensor::from_tensor(t2.clone());
                    let view1 = shared1.view();
                    let view2 = shared2.view();
                    let result = black_box(view1.elementwise_with(&view2, |a, b| a + b).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark in-place operations vs out-of-place operations
fn bench_inplace_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("inplace_operations");
    
    let size = 50000; // 50K elements
    group.throughput(Throughput::Elements(size as u64));
    
    let config = MemoryOptimizedConfig {
        strategy: AllocationStrategy::Pool,
        enable_inplace: true,
        ..Default::default()
    };
    
    // Out-of-place operations (creates new tensor)
    group.bench_function("out_of_place_add", |b| {
        b.iter(|| {
            let mut tensor1 = Tensor::<f32>::ones(&[size]);
            let tensor2 = Tensor::<f32>::ones(&[size]);
            let result = black_box(&tensor1 + &tensor2);
            black_box(result)
        });
    });
    
    // In-place operations (modifies existing tensor)
    group.bench_function("in_place_add", |b| {
        b.iter(|| {
            let mut tensor1 = Tensor::<f32>::ones(&[size]);
            let tensor2 = Tensor::<f32>::ones(&[size]);
            black_box(tensor1.inplace_add_with(&tensor2, &config).unwrap());
            black_box(tensor1)
        });
    });
    
    // Out-of-place scalar multiplication
    group.bench_function("out_of_place_scalar_mul", |b| {
        b.iter(|| {
            let tensor = Tensor::<f32>::ones(&[size]);
            let result = black_box(&tensor * 2.0);
            black_box(result)
        });
    });
    
    // In-place scalar multiplication
    group.bench_function("in_place_scalar_mul", |b| {
        b.iter(|| {
            let mut tensor = Tensor::<f32>::ones(&[size]);
            black_box(tensor.inplace_mul_scalar_with(2.0, &config).unwrap());
            black_box(tensor)
        });
    });
    
    group.finish();
}

/// Benchmark cache-friendly block operations
fn bench_cache_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_optimization");
    group.measurement_time(Duration::from_secs(15));
    
    let matrix_sizes = vec![
        (256, "256x256"),
        (512, "512x512"),
        (1024, "1024x1024"),
    ];
    
    for (size, description) in matrix_sizes {
        let matrix1 = Tensor::<f32>::ones(&[size, size]);
        let matrix2 = Tensor::<f32>::ones(&[size, size]);
        let operations = (size * size * size) as u64;
        
        group.throughput(Throughput::Elements(operations));
        
        // Regular matrix multiplication (no cache optimization)
        group.bench_with_input(
            BenchmarkId::new("regular_matmul", description),
            &(size, &matrix1, &matrix2),
            |b, (_, m1, m2)| {
                b.iter(|| {
                    let result = black_box(m1.matmul(m2));
                    black_box(result)
                });
            },
        );
        
        // Cache-optimized blocked matrix multiplication
        let block_config = MemoryOptimizedConfig {
            strategy: AllocationStrategy::Pool,
            cache_block_size: 64, // 64x64 blocks for cache efficiency
            enable_blocking: true,
            ..Default::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("blocked_matmul", description),
            &(size, &matrix1, &matrix2, &block_config),
            |b, (_, m1, m2, config)| {
                b.iter(|| {
                    let result = black_box(m1.with_memory_strategy(config)
                        .blocked_matmul(m2).unwrap());
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory pool efficiency
fn bench_memory_pool_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_efficiency");
    
    let allocation_counts = vec![
        (10, "10 allocations"),
        (100, "100 allocations"),
        (1000, "1000 allocations"),
    ];
    
    for (count, description) in allocation_counts {
        let tensor_size = 1000; // 1K elements per tensor
        
        group.throughput(Throughput::Elements((count * tensor_size) as u64));
        
        // Direct allocation (no pooling)
        let direct_config = MemoryOptimizedConfig {
            strategy: AllocationStrategy::Direct,
            ..Default::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("direct_allocations", description),
            &(count, tensor_size, &direct_config),
            |b, (cnt, size, config)| {
                b.iter(|| {
                    let mut tensors = Vec::new();
                    for _ in 0..*cnt {
                        let tensor = Tensor::<f32>::ones(&[*size]);
                        let optimized = tensor.with_memory_strategy(config);
                        tensors.push(optimized);
                    }
                    black_box(tensors)
                });
            },
        );
        
        // Pool allocation
        let pool_config = MemoryOptimizedConfig {
            strategy: AllocationStrategy::Pool,
            pool_size_hint: Some(count * tensor_size * 2), // Pre-allocate pool
            ..Default::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("pool_allocations", description),
            &(count, tensor_size, &pool_config),
            |b, (cnt, size, config)| {
                b.iter(|| {
                    let mut tensors = Vec::new();
                    for _ in 0..*cnt {
                        let tensor = Tensor::<f32>::ones(&[*size]);
                        let optimized = tensor.with_memory_strategy(config);
                        tensors.push(optimized);
                    }
                    black_box(tensors)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory alignment effects on performance
fn bench_memory_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_alignment");
    
    let size = 32768; // 32K elements
    group.throughput(Throughput::Elements(size as u64));
    
    // Regular allocation (system default alignment)
    let regular_config = MemoryOptimizedConfig {
        strategy: AllocationStrategy::Direct,
        ..Default::default()
    };
    
    group.bench_function("regular_alignment", |b| {
        b.iter(|| {
            let tensor1 = Tensor::<f32>::ones(&[size]);
            let tensor2 = Tensor::<f32>::ones(&[size]);
            let result = black_box(tensor1.with_memory_strategy(&regular_config)
                .elementwise_with(&tensor2, |a, b| a + b).unwrap());
            black_box(result)
        });
    });
    
    // SIMD-aligned allocation (32-byte alignment for AVX2)
    let aligned_config = MemoryOptimizedConfig {
        strategy: AllocationStrategy::SimdAligned,
        enable_vectorization: true,
        ..Default::default()
    };
    
    group.bench_function("simd_alignment", |b| {
        b.iter(|| {
            let tensor1 = Tensor::<f32>::ones(&[size]);
            let tensor2 = Tensor::<f32>::ones(&[size]);
            let result = black_box(tensor1.with_memory_strategy(&aligned_config)
                .vectorized_add(&tensor2).unwrap());
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark memory strategy switching overhead
fn bench_strategy_switching(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_switching");
    
    let tensor = Tensor::<f32>::ones(&[10000]);
    
    let strategies = vec![
        ("direct", AllocationStrategy::Direct),
        ("pool", AllocationStrategy::Pool),
        ("zero_copy", AllocationStrategy::ZeroCopy),
        ("simd_aligned", AllocationStrategy::SimdAligned),
    ];
    
    for (name, strategy) in strategies {
        let config = MemoryOptimizedConfig {
            strategy,
            ..Default::default()
        };
        
        group.bench_with_input(
            BenchmarkId::new("strategy_application", name),
            &config,
            |b, cfg| {
                b.iter(|| {
                    let optimized = black_box(tensor.with_memory_strategy(cfg));
                    black_box(optimized)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    memory_benches,
    bench_allocation_strategies,
    bench_zero_copy_operations,
    bench_inplace_operations,
    bench_cache_optimization,
    bench_memory_pool_efficiency,
    bench_memory_alignment,
    bench_strategy_switching
);

criterion_main!(memory_benches);

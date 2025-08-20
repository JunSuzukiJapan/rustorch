//! Refactored Memory Performance Benchmarks
//! 
//! This benchmark suite uses common utilities to measure memory allocation
//! and optimization strategies with cleaner code structure.

mod common;

use criterion::{criterion_group, criterion_main, Criterion};
use common::{
    MATRIX_SIZES, BENCHMARK_SIZES,
    memory_ops::{bench_allocation_strategies, bench_zero_copy_operations, bench_memory_pool_operations},
    BenchmarkConfig
};

/// Benchmark different memory allocation strategies
fn bench_allocation_strategies_f32(c: &mut Criterion) {
    bench_allocation_strategies::<f32>(
        c,
        "allocation_strategies",
        MATRIX_SIZES,
        Some(BenchmarkConfig::default()),
    );
}

/// Benchmark zero-copy operations vs regular copying
fn bench_zero_copy_ops(c: &mut Criterion) {
    bench_zero_copy_operations::<f32>(
        c,
        "zero_copy_operations",
        &[(1000, "1K elements"), (10000, "10K elements"), (100000, "100K elements")],
        Some(BenchmarkConfig::default()),
    );
}

/// Benchmark memory pool operations
fn bench_memory_pool_ops(c: &mut Criterion) {
    bench_memory_pool_operations::<f32>(
        c,
        "memory_pool_operations",
        &[(10, "10 allocations"), (100, "100 allocations"), (1000, "1000 allocations")],
        1000, // tensor size
        Some(BenchmarkConfig::default()),
    );
}

criterion_group!(
    benches,
    bench_allocation_strategies_f32,
    bench_zero_copy_ops,
    bench_memory_pool_ops
);
criterion_main!(benches);

//! Refactored SIMD Performance Benchmarks
//! 
//! This benchmark suite uses the common utilities to measure SIMD performance
//! improvements with cleaner, more maintainable code.

mod common;

use criterion::{criterion_group, criterion_main, Criterion};
use common::{
    BENCHMARK_SIZES, MATRIX_SIZES,
    simd_ops::{bench_simd_vs_scalar, bench_simd_matrix_ops, bench_simd_vectorized_ops},
    BenchmarkConfig
};

/// Benchmark SIMD vs scalar element-wise operations
fn bench_simd_elementwise(c: &mut Criterion) {
    bench_simd_vs_scalar::<f32>(
        c,
        "simd_elementwise",
        BENCHMARK_SIZES,
        Some(BenchmarkConfig::default()),
    );
}

/// Benchmark SIMD matrix operations
fn bench_simd_matmul(c: &mut Criterion) {
    bench_simd_matrix_ops(
        c,
        "simd_matmul",
        &[(32, "32x32"), (64, "64x64"), (128, "128x128")],
        Some(BenchmarkConfig::default()),
    );
}

/// Benchmark SIMD vectorized operations
fn bench_simd_vectorized(c: &mut Criterion) {
    bench_simd_vectorized_ops(
        c,
        "simd_vectorized",
        BENCHMARK_SIZES,
        Some(BenchmarkConfig::default()),
    );
}

criterion_group!(
    benches,
    bench_simd_elementwise,
    bench_simd_matmul,
    bench_simd_vectorized
);
criterion_main!(benches);

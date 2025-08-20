//! Refactored Tensor Performance Benchmarks
//! 
//! This benchmark suite uses common utilities to measure basic tensor operations
//! with consistent structure and configuration.

mod common;

use criterion::{criterion_group, criterion_main, Criterion};
use common::{
    BENCHMARK_SIZES,
    tensor_ops::{bench_elementwise_add, bench_matrix_multiply, bench_tensor_allocation, bench_tensor_copy},
    BenchmarkConfig
};

/// Benchmark element-wise tensor addition
fn bench_tensor_add(c: &mut Criterion) {
    bench_elementwise_add::<f32>(
        c,
        "tensor_elementwise_add",
        BENCHMARK_SIZES,
        Some(BenchmarkConfig::default()),
    );
}

/// Benchmark matrix multiplication
fn bench_tensor_matmul(c: &mut Criterion) {
    bench_matrix_multiply::<f32>(
        c,
        "tensor_matrix_multiply",
        &[(64, "64x64"), (128, "128x128"), (256, "256x256")],
        Some(BenchmarkConfig::default()),
    );
}

/// Benchmark tensor allocation
fn bench_tensor_alloc(c: &mut Criterion) {
    bench_tensor_allocation::<f32>(
        c,
        "tensor_allocation",
        BENCHMARK_SIZES,
        Some(BenchmarkConfig::quick()),
    );
}

/// Benchmark tensor copying
fn bench_tensor_copying(c: &mut Criterion) {
    bench_tensor_copy::<f32>(
        c,
        "tensor_copy",
        BENCHMARK_SIZES,
        Some(BenchmarkConfig::default()),
    );
}

criterion_group!(
    benches,
    bench_tensor_add,
    bench_tensor_matmul,
    bench_tensor_alloc,
    bench_tensor_copying
);
criterion_main!(benches);

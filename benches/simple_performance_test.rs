use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::tensor::parallel_traits::*;
use rustorch::tensor::Tensor;

/// Benchmark basic tensor operations
fn bench_tensor_operations(c: &mut Criterion) {
    let size = 10000;
    let tensor_a = Tensor::<f32>::from_vec((0..size).map(|i| i as f32).collect(), vec![size]);
    let tensor_b = Tensor::<f32>::from_vec((0..size).map(|i| (i + 1) as f32).collect(), vec![size]);

    c.bench_function("tensor_addition", |b| {
        b.iter(|| {
            let _result = black_box(&tensor_a) + black_box(&tensor_b);
        })
    });

    c.bench_function("tensor_sum", |b| {
        b.iter(|| {
            let _result = black_box(&tensor_a).sum();
        })
    });

    c.bench_function("parallel_elementwise", |b| {
        b.iter(|| {
            let _result = black_box(&tensor_a)
                .batch_elementwise_op(black_box(&tensor_b), |x, y| x + y)
                .unwrap_or_else(|_| black_box(&tensor_a) + black_box(&tensor_b));
        })
    });
}

/// Benchmark matrix operations
fn bench_matrix_operations(c: &mut Criterion) {
    let size = 128;
    let mat_a = Tensor::<f32>::from_vec(
        (0..size * size).map(|i| (i as f32) * 0.01).collect(),
        vec![size, size],
    );
    let mat_b = Tensor::<f32>::from_vec(
        (0..size * size).map(|i| (i as f32) * 0.01).collect(),
        vec![size, size],
    );

    c.bench_function("matrix_multiplication", |b| {
        b.iter(|| {
            let _result = black_box(&mat_a).matmul(black_box(&mat_b)).unwrap();
        })
    });
}

/// Benchmark memory operations
fn bench_memory_operations(c: &mut Criterion) {
    c.bench_function("tensor_creation", |b| {
        b.iter(|| {
            let _tensor = Tensor::<f32>::zeros(&[black_box(1000)]);
        })
    });

    let tensor = Tensor::<f32>::from_vec((0..1000).map(|i| i as f32).collect(), vec![1000]);

    c.bench_function("tensor_clone", |b| {
        b.iter(|| {
            let _clone = black_box(&tensor).clone();
        })
    });
}

criterion_group!(
    benches,
    bench_tensor_operations,
    bench_matrix_operations,
    bench_memory_operations
);
criterion_main!(benches);

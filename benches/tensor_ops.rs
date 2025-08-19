//! Tensor operations benchmarks
//! テンソル演算のベンチマーク

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::prelude::*;

fn bench_tensor_creation(c: &mut Criterion) {
    c.bench_function("tensor_creation_1000", |b| {
        b.iter(|| {
            let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
            black_box(Tensor::from_vec(data, vec![1000]))
        })
    });
    
    c.bench_function("tensor_creation_1000x1000", |b| {
        b.iter(|| {
            let data: Vec<f32> = (0..1_000_000).map(|i| i as f32).collect();
            black_box(Tensor::from_vec(data, vec![1000, 1000]))
        })
    });
}

fn bench_tensor_arithmetic(c: &mut Criterion) {
    let a = Tensor::from_vec((0..10000).map(|i| i as f32).collect(), vec![100, 100]);
    let b_data = Tensor::from_vec((0..10000).map(|i| (i * 2) as f32).collect(), vec![100, 100]);
    
    c.bench_function("tensor_add_100x100", |bencher| {
        bencher.iter(|| black_box(&a + &b_data))
    });
    
    c.bench_function("tensor_mul_100x100", |bencher| {
        bencher.iter(|| black_box(&a * &b_data))
    });
    
    c.bench_function("tensor_sub_100x100", |bencher| {
        bencher.iter(|| black_box(&a - &b_data))
    });
}

fn bench_matrix_operations(c: &mut Criterion) {
    let a = Tensor::from_vec((0..10000).map(|i| i as f32).collect(), vec![100, 100]);
    let b_data = Tensor::from_vec((0..10000).map(|i| (i + 1) as f32).collect(), vec![100, 100]);
    
    c.bench_function("matmul_100x100", |bencher| {
        bencher.iter(|| black_box(a.matmul(&b_data)))
    });
    
    c.bench_function("transpose_100x100", |bencher| {
        bencher.iter(|| black_box(a.transpose()))
    });
    
    let large_a = Tensor::from_vec((0..1_000_000).map(|i| i as f32).collect(), vec![1000, 1000]);
    let large_b = Tensor::from_vec((0..1_000_000).map(|i| (i + 1) as f32).collect(), vec![1000, 1000]);
    
    c.bench_function("matmul_1000x1000", |bencher| {
        bencher.iter(|| black_box(large_a.matmul(&large_b)))
    });
}

fn bench_tensor_reductions(c: &mut Criterion) {
    let tensor = Tensor::from_vec((0..100000).map(|i| i as f32).collect(), vec![100, 1000]);
    
    c.bench_function("sum_100x1000", |bencher| {
        bencher.iter(|| black_box(tensor.sum()))
    });
    
    c.bench_function("mean_axis_100x1000", |bencher| {
        bencher.iter(|| black_box(tensor.mean_axis(0)))
    });
    
    c.bench_function("sum_axis_100x1000", |bencher| {
        bencher.iter(|| black_box(tensor.sum_axis(0)))
    });
}

criterion_group!(
    tensor_benches,
    bench_tensor_creation,
    bench_tensor_arithmetic,
    bench_matrix_operations,
    bench_tensor_reductions
);
criterion_main!(tensor_benches);

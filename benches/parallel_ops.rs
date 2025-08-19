//! Parallel operations benchmarks
//! 並列演算のベンチマーク

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::prelude::*;

fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let large_tensor = Tensor::from_vec(
        (0..1_000_000).map(|i| i as f32).collect(),
        vec![1000, 1000]
    );
    let other_tensor = Tensor::from_vec(
        (0..1_000_000).map(|i| (i + 1) as f32).collect(),
        vec![1000, 1000]
    );
    
    c.bench_function("parallel_sum_1M_elements", |b| {
        b.iter(|| black_box(large_tensor.sum_parallel()))
    });
    
    c.bench_function("sequential_sum_1M_elements", |b| {
        b.iter(|| black_box(large_tensor.sum()))
    });
    
    c.bench_function("parallel_inplace_add_1M", |b| {
        b.iter(|| {
            let mut tensor = large_tensor.clone();
            tensor.add_inplace(&other_tensor);
            black_box(tensor)
        })
    });
}

fn bench_parallel_matmul(c: &mut Criterion) {
    let a = Tensor::from_vec(
        (0..250000).map(|i| i as f32 * 0.001).collect(),
        vec![500, 500]
    );
    let b = Tensor::from_vec(
        (0..250000).map(|i| (i + 1) as f32 * 0.001).collect(),
        vec![500, 500]
    );
    
    c.bench_function("parallel_matmul_500x500", |bencher| {
        bencher.iter(|| black_box(a.matmul_parallel(&b)))
    });
    
    c.bench_function("regular_matmul_500x500", |bencher| {
        bencher.iter(|| black_box(a.matmul(&b)))
    });
    
    let large_a = Tensor::from_vec(
        (0..1_000_000).map(|i| i as f32 * 0.0001).collect(),
        vec![1000, 1000]
    );
    let large_b = Tensor::from_vec(
        (0..1_000_000).map(|i| (i + 1) as f32 * 0.0001).collect(),
        vec![1000, 1000]
    );
    
    c.bench_function("parallel_matmul_1000x1000", |bencher| {
        bencher.iter(|| black_box(large_a.matmul_parallel(&large_b)))
    });
}

fn bench_parallel_element_wise(c: &mut Criterion) {
    let tensor = Tensor::from_vec(
        (0..1_000_000).map(|i| i as f32 * 0.001).collect(),
        vec![1000, 1000]
    );
    
    c.bench_function("parallel_apply_relu", |b| {
        b.iter(|| {
            black_box(tensor.apply_parallel(|x| if x > 0.0 { x } else { 0.0 }))
        })
    });
    
    c.bench_function("parallel_apply_sigmoid", |b| {
        b.iter(|| {
            black_box(tensor.apply_parallel(|x| 1.0 / (1.0 + (-x).exp())))
        })
    });
    
    c.bench_function("parallel_scalar_mul", |b| {
        b.iter(|| {
            let mut t = tensor.clone();
            t.mul_scalar_inplace(2.0);
            black_box(t)
        })
    });
}

fn bench_batch_parallel_processing(c: &mut Criterion) {
    // Simulate batch processing for neural networks
    c.bench_function("parallel_batch_linear_forward", |b| {
        b.iter(|| {
            let batch_input = Tensor::from_vec(
                (0..65536).map(|i| i as f32 * 0.01).collect(),
                vec![128, 512]  // batch_size=128, input_size=512
            );
            let weight = Tensor::from_vec(
                (0..131072).map(|i| i as f32 * 0.01).collect(),
                vec![512, 256]  // input_size=512, output_size=256
            );
            
            black_box(batch_input.matmul_parallel(&weight))
        })
    });
    
    c.bench_function("parallel_batch_activation", |b| {
        b.iter(|| {
            let batch_data = Tensor::from_vec(
                (0..32768).map(|i| i as f32 * 0.01).collect(),
                vec![128, 256]  // batch_size=128, features=256
            );
            
            // Apply ReLU activation
            black_box(batch_data.apply_parallel(|x| if x > 0.0 { x } else { 0.0 }))
        })
    });
}

criterion_group!(
    parallel_benches,
    bench_parallel_vs_sequential,
    bench_parallel_matmul,
    bench_parallel_element_wise,
    bench_batch_parallel_processing
);
criterion_main!(parallel_benches);

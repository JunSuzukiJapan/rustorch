//! Autograd operations benchmarks
//! 自動微分演算のベンチマーク

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch::prelude::*;

fn bench_variable_creation(c: &mut Criterion) {
    c.bench_function("variable_creation_1000", |b| {
        b.iter(|| {
            let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
            let tensor = Tensor::from_vec(data, vec![1000]);
            black_box(Variable::new(tensor, true))
        })
    });
}

fn bench_backward_propagation(c: &mut Criterion) {
    c.bench_function("simple_backward_10x10", |b| {
        b.iter(|| {
            let a = Variable::new(
                Tensor::from_vec((0..100).map(|i| i as f32).collect(), vec![10, 10]),
                true,
            );
            let b = Variable::new(
                Tensor::from_vec((0..100).map(|i| (i + 1) as f32).collect(), vec![10, 10]),
                true,
            );

            let c = &a + &b;
            let d = &c * &a;
            let loss = d.sum();

            black_box(loss.backward());
        })
    });

    c.bench_function("complex_backward_50x50", |b| {
        b.iter(|| {
            let a = Variable::new(
                Tensor::from_vec((0..2500).map(|i| i as f32).collect(), vec![50, 50]),
                true,
            );
            let b = Variable::new(
                Tensor::from_vec((0..2500).map(|i| (i + 1) as f32).collect(), vec![50, 50]),
                true,
            );

            let c = a.matmul(&b);
            let d = &c + &a;
            let e = &d * &b;
            let loss = e.mean();

            black_box(loss.backward());
        })
    });
}

fn bench_gradient_computation(c: &mut Criterion) {
    c.bench_function("matmul_gradient_100x100", |b| {
        b.iter(|| {
            let a = Variable::new(
                Tensor::from_vec((0..10000).map(|i| i as f32).collect(), vec![100, 100]),
                true,
            );
            let b = Variable::new(
                Tensor::from_vec((0..10000).map(|i| (i + 1) as f32).collect(), vec![100, 100]),
                true,
            );

            let c = a.matmul(&b);
            let loss = c.sum();

            black_box(loss.backward());
        })
    });
}

fn bench_chain_operations(c: &mut Criterion) {
    c.bench_function("long_chain_operations", |b| {
        b.iter(|| {
            let x = Variable::new(
                Tensor::from_vec((0..1000).map(|i| i as f32 * 0.01).collect(), vec![10, 100]),
                true,
            );

            // Simulate a deep computation chain
            let mut result = x.clone();
            for _ in 0..10 {
                let temp = &result + &x;
                result = temp.mean_autograd();
            }

            black_box(result.backward());
        })
    });
}

criterion_group!(
    autograd_benches,
    bench_variable_creation,
    bench_backward_propagation,
    bench_gradient_computation,
    bench_chain_operations
);
criterion_main!(autograd_benches);
